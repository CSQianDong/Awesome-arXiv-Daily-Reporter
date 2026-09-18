# An Empirical Study of Harness Design for Coding Agents 

**Authors**: Run-Ze Fan, Zihao Zhang, Simin Ma, Yebowen Hu, Shouju Wang, Kaiqiang Song, Fei Liu, Hamed Zamani, Xiaoyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.20804)  

**Abstract**: Coding harnesses shape how autonomous coding agents translate model capabilities into long-horizon software-engineering performance, yet existing work typically evaluates harnesses as monolithic systems, leaving the effectiveness of individual components unclear. To enable component-level comparisons, we study this question with a lightweight coding harness whose execution loop is fixed while three components are varied: planning, action space, and context management. Across four models evaluated on SWE-Bench Verified and Terminal-Bench 2.1, we evaluate 176 matched settings spanning five context-management strategies, four context-window budgets, and targeted ablations of planning and action space. We find that: (1) Context management becomes increasingly valuable as the context-window budget tightens, with most of its benefit coming from preventing context-overflow failures. (2) Staging rule-based elision before LLM-based summarization provides the strongest overall efficiency among the context-management strategies, whereas making elided content recoverable adds machinery that models rarely use and yields no accuracy gain. (3) Planning shifts from an accuracy scaffold for weaker models to a cost saver for stronger models, with little change in accuracy. (4) Predefined tools improve performance for models with weaker bash proficiency, whereas bash-capable models can operate effectively with a bash-only interface and achieve substantially lower cost, especially on command-line-centric tasks. Trajectory-level analysis explains these effects: context management extends execution trajectories without substantially altering agent behavior, planning changes where trajectories stop, and the action space changes the granularity at which code is written. These findings inform model- and budget-aware harness design and provide a modular framework for evaluating future harness components. 

---
# RAFT: A Stateful Retrieval-Augmented Framework for Troubleshooting Agents 

**Authors**: Mingxuan Zhang, Xiaowen Wang, Anupma Sharan, Zhengyi Chen, Chenyu Diana Zhang, Shanshan Yang, Chittibabu Pacharu  

**Link**: [PDF](https://arxiv.org/pdf/2609.20754)  

**Abstract**: Effective troubleshooting agents in enterprise customer support depend on retrieving actionable guidance from similar historical cases, yet existing retrieval-augmented generation (RAG) systems treat support cases as static documents and overlook their multi-stage, stateful nature. We introduce RAFT (Retrieval-Augmented Framework for Troubleshooting Agents), a stateful RAG framework that abstracts each closed historical case into a directed chain of timeline entries and retrieves at the entry level, surfacing cases whose intermediate states match the active case and returning the parent-case trajectory anchored at the matched state; an optional case-level graph links cases through a configurable similarity representation. We evaluate this retrieval layer directly, which, unlike evaluating a full agent system, requires no production deployment. Because public multi-stage troubleshooting data is extremely rare, we pair a synthetic benchmark built from Microsoft Learn Windows Server documentation with real Apache Jira issues carrying human-created duplicate labels. RAFT improves Case Hit over vanilla RAG and GraphRAG baselines at every stage of case progress, with statistically significant gains over the strongest baseline; the Jira results provide directional evidence that the advantage transfers to real case histories. We release our benchmark, implementation, and the Apache Jira evaluation set. 

---
# Q&A on Any Spreadsheet Requires Interpreting Its Grid Structure 

**Authors**: Zofia Smoleń  

**Link**: [PDF](https://arxiv.org/pdf/2609.20732)  

**Abstract**: Semantic cell annotation improves chunking interpretability for spreadsheets in LLM-driven RAG systems, aiding answer generation through enriched context rather than improved retrieval accuracy. We propose a novel framework of splitting any spreadsheet into interpretable chunks using cell role annotation. Our framework beats the state of the art, yet it faces a hard ceiling. Spreadsheets are fundamentally two-dimensional unstructured data with continuous relationships and infinite potential cell roles. Because classification models are restricted to finite, pre-defined classes, they cannot perfectly capture this structural nuance, even with human-level annotation. We show that addressing the spreadsheet-to-LLM bottleneck requires moving beyond discrete cell classification. Instead, the field must develop dimensionality-reduction techniques to directly flatten 2D unstructured spreadsheets into 1D unstructured text. Text chunks would be easier for downstream RAG to interpret and generate from. 

---
# Deep Noir: Autonomous Steering Discovery via Architectural Chronometry in Transformer Models 

**Authors**: Frank E. Bobe III, Gregory D. Vetaw, Darshan W. Bryner, Matthew G. Cook, Jose L. Salas-Vernis  

**Link**: [PDF](https://arxiv.org/pdf/2609.20722)  

**Abstract**: Activation steering modifies LLM behavior at inference time, but identifying where and how strongly to steer remains manual. We introduce Deep Noir, a framework that uses Logit Lens convergence and causal head-level attribution to autonomously discover optimal steering parameters. Across three scales (1B x 3, 2-3B x 2, and 7-9B x 4), our engine achieves 16.7 percentage-point improvement on spam at 1B (standard deviation 4.7; 39 runs), with gains increasing to 21 to 42 percentage points at 7-9B across four architectures. On SST-2 sentiment, it achieves a 13.1 percentage-point improvement with zero code changes. Mechanistic grounding enables automated discovery of intervention points that generalize across tasks and architectures. On sentiment, RepE without head masking fails to improve over baseline, while Deep Noir improves all models (p less than 0.01). We further show that steering creates a predictable prompt-injection attack surface whose vulnerability increases monotonically with steering magnitude. This finding is relevant to agent systems deploying steered classifiers. 

---
# Ownership in AI-Assisted Everyday Tasks 

**Authors**: Megan Wei, Melanie Subbiah, Audrey Lee, Annya Dahmani, Dave Edwards, Helen Edwards, Ellie Pavlick  

**Link**: [PDF](https://arxiv.org/pdf/2609.20658)  

**Abstract**: When does work done with AI still feel like ours? As AI becomes woven into everyday tasks, we must examine what happens to our sense of ownership and contribution when a machine shares in producing what we make. We report an exploratory qualitative survey in which participants were asked to describe two recent, self-selected tasks completed with AI: one that felt like their own and one that did not. We find that felt ownership depends on the process of collaboration: people disown work when they merely approve AI's suggestions, but retain ownership when they lead, iterate, or rewrite. Ownership can also extend to settings where people own the vision for a project but not the execution; respondents reported high ownership on tasks they could not have completed without AI. Loss of personal voice and a lack of comprehension of the output both erode ownership. Finally, willingness to disclose AI use is often decoupled from actual pride or ownership, and instead shaped by community norms and fear of credit erasure. We propose several research directions as a result of these findings to promote AI development that supports people's sense of authorship over their own lives. 

---
# PAA: The Probabilistic Allen Algebra: A Generative and Complete Probabilistic Extension of Allen's Interval Relations 

**Authors**: Julian Eggert  

**Link**: [PDF](https://arxiv.org/pdf/2609.20634)  

**Abstract**: Allen's interval algebra is a qualitative calculus for temporal relations, but its thirteen base relations are crisp predicates over exact interval boundaries. This is inadequate for temporal information from language, perception, databases, or uncertain histories, where times, durations, and boundaries are uncertain and expressions such as "just before" or "roughly during" have graded meaning. We develop the probabilistic Allen algebra (PAA): a generative and complete extension in which relation probabilities are derived from distributions over interval boundaries rather than assigned as scores. Time points are Gaussian; intervals have Gaussian midpoints and truncated-Gaussian durations. Every relation is a boundary-ordering predicate in one common probability space: point-point relations reduce to error functions, and point-interval and interval-interval relations to multivariate Gaussian orthant probabilities induced by linear inequalities. Contact relations (meets, starts, finishes, equals) receive positive measure through a tolerance band, and under a single tolerance the thirteen relations form a true partition that recovers crisp Allen as the tolerance vanishes. The construction derives Allen's taxonomy rather than positing it: coarse predicates such as precedence, overlap, and containment are unions of leaves whose probabilities are leaf sums, and this hierarchy is preserved as intervals collapse to points and thirteen relations reduce to five and then three. Each relation further decomposes into correlation-aware temporal primitives in the spirit of CIDOC CRM. The algebra is scale-invariant and separates graded expressions such as "shortly before" from contact relations. All results are Monte-Carlo validated and shipped as an open, tested Python package. 

---
# Limits of Confidence in Diffusion 

**Authors**: Russ Webb, Amitis Shidani, Alice Bizeul, Dan Busbridge  

**Link**: [PDF](https://arxiv.org/pdf/2609.20581)  

**Abstract**: Discrete diffusion, including remasking and uniform-state samplers, generate a sequence by writing multiple token positions per step, drawing each from a per-position distribution and choosing which positions to write from those same distributions. For domains of general interest (pixels, phonemes, or words) there are inherent dependencies between tokens. We show that a step matches the training distribution only when the positions it writes are conditionally independent given the tokens already fixed, that no product of per-position distributions can match a dependent group, and that per-position distributions do not determine whether a group is dependent: two joint distributions can have identical per-position marginals while differing in which combinations of values occur. On ScanAndAdd, a synthetic task whose joint distribution is available in closed form, we verify that every group of two or more undetermined positions a confidence ranking writes is dependent, and measure the generated distribution to be $29\times$ the sampling-noise floor total variation while per-sample metrics are $1.0$. 

---
# Language-model groups overstate consensus when replaying human deliberation on a reasoning task 

**Authors**: Tengfei Shao  

**Link**: [PDF](https://arxiv.org/pdf/2609.20543)  

**Abstract**: Full-consensus rates are often treated as indicators of collective cognition, yet depend on how participation and final states are operationalized. We replayed 100 held-out human Wason groups with matched large language model (LLM) agent groups, seeding one belief-anchored agent per participant's pre-discussion answer and scoring agents and people with the same code. Across human scoring definitions, estimates ranged from 24.0% to 57.0%; about one fifth of participants never posted, whereas agents almost always did. Agent groups remained more consensual in two post-unblinding sensitivity analyses: the submit-based comparison (n = 98) yielded gaps of 34.0 and 43.9 percentage points for chat and reasoning modes, and the participation-matched comparison (n = 45) yielded gaps of 34.1 and 44.4 points. These complementary routes reduced different measurement asymmetries yet converged within 0.5 percentage points. The gap persisted without early stopping and under a reparameterization removing the memorizable answer; reasoning-mode groups then agreed nearly unanimously, mostly on incorrect answers. Simulated consensus did not track collective accuracy, and belief-anchored agent groups were biased estimators of the human group-outcome distribution in this setting. These analyses provide a scoring-explicit basis for assessing simulated-group estimates of human deliberative outcomes. 

---
# Refuse, Decompose, Refresh: A Claim-Safe Protocol for Closed-Loop AI Evaluation 

**Authors**: Peiying Zhu, Sidi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2609.20538)  

**Abstract**: An AI evaluation can be perfectly reproducible and still support the wrong claim. This risk is acute in closed-loop systems: policy determines visited states, observable components, and which failures leave a measurable trace. We propose a claim-safe protocol with three actions. Refuse: abstain when a clean reference stream or matched runtime comparison lacks support. Decompose: report protocol execution, operational false admission, and structural hypotheses separately rather than as one PASS/FAIL label. Refresh: treat distribution-shift alarms as requests to invalidate and recompute a reference map, not as fault evidence. We instantiate the protocol in an aggregate-only simulator with 24 policy components, three demand regimes, two fault-mask families, and independent development and heldout seeds. The preregistered heldout contains 1,440 cases and 21,600 partition rows. Only 55/72 regime-component units were reference-admitted and 54/55 remained runtime-admitted, making abstention part of the result. Stable false admission was 0/20 represented components, with a one-sided exact 95% upper bound of 0.1391 under a frozen 0.20 rule. Within admitted units, affected clean traffic outpredicted nominal fault-cell fraction: across 540 unit-arm rows nested in 20 component clusters, the cell-minus-traffic negative-log-likelihood difference was 0.1264 nats per row, with a 95% component-cluster interval of [0.0593, 0.1918]. A drift log shows why "null" must be reference-relative: clean fault-null streams triggered 15/15, 0/15, and 14/15 alarms across three regimes, while only the middle regime matched the frozen detector reference. Rather than a universal threshold, we contribute an executable contract linking observable support, statistical calibration, and justified claims. 

---
# FreqCondNorm: Towards Cross-domain Predictive Maintenance through a Frequency-Conditioned Transformer Foundation Model 

**Authors**: Zaynab Raounak, Camille LHermine, Zhiguo Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2609.20535)  

**Abstract**: Deep learning predictive maintenance models suffer from poor transferability across machines and operating conditions, especially when labelled data are scarce and signals span five orders of magnitude in sampling frequency (1 Hz to ~100 kHz). We propose FreqCondNorm, a Transformer-based architecture that introduces a FiLM-style frequency-conditioned normalization layer to unify heterogeneous time-series within a single model. The architecture is pretrained on five public predictive maintenance datasets (CWRU, MFPT, UOC18, PRONOSTIA, CMAPSS) using masked auto-encoding and contrastive learning with balanced domain sampling. On fault diagnosis, the model achieves 99.2% accuracy on CWRU (+6.4 pp over CNN) and 82.1% zero-shot accuracy on MFPT, demonstrating strong transfer across sampling frequencies. However, the approach does not improve remaining useful life prediction, suggesting a mismatch between pretraining and RUL objectives that warrants future investigation. 

---
# SoL-Pi: Recursively Scaling Auto-Research Loops for Efficient Agent Harness 

**Authors**: Haozhe Liu, Tian Ye, Sensen Gao, Qihang Cao, Yitong Li, Mingchen Zhuge, Duomin Wang, Ruihua Zhang, Ping Luo, Jiawang Bian, Lei Zhu, Ligeng Zhu, Enze Xie, Song Han  

**Link**: [PDF](https://arxiv.org/pdf/2609.20519)  

**Abstract**: As coding agents move from supervised code completion to unattended, around-the-clock exploration, their work expands from isolated predictions into long trajectories of reasoning, tool use, and feedback. Token efficiency therefore becomes important for scaling recursive self-improvement. We take an RSI-inspired approach at the harness layer, scaling auto-research loops across increasingly numerous and diverse environments for harness rollouts. At this scale, the process yields reusable improvements that transfer beyond their development setting, moving automated harness discovery toward production-level outcomes. Four mechanisms survive selection and form SoL-Pi, spanning action execution, context compaction, observation handling, and delegated reading. On the 51-task EdgeBench evaluation, SoL-Pi achieves performance comparable to Pi across GPT-5.6 Sol and Opus 5 while reducing recorded token traffic by 44.7-49.0% and API cost by about one third. In other words, estimated hourly savings are \$8.75-\$13.50 relative to native Codex and Claude Code harnesses, and \$4.36-\$5.71 relative to Pi. 

---
# How Do Agent Harnesses Create Value? Planning Information and Release Control in Stateful LLM Agents 

**Authors**: Yukun Zhang, Kemu Xu, Yishen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.20474)  

**Abstract**: Agent harnesses supply planning guidance, organize execution, and check completion. We study how these components affect success, erroneous acceptance, and cost in two Retail experiments and an Airline pilot in $\tau^2$-bench. The primary comparison pairs prewritten task-specific plans (Fixed) with shuffled policy text matched in word count (Sham), isolating the contribution of guidance content. Across 265 matched cells, Fixed improves oracle-verified success by 7.17 percentage points (90\% task-clustered bootstrap interval, 1.15--13.36 points), with gains concentrated in higher-complexity tasks. A read-only terminal verifier rejects 61\% of Retail oracle-invalid episodes while withholding 17\% of correct ones, at less than one cent of additional cost per episode. Which component matters more depends on the loss assigned to erroneous acceptance: at low liability the planning gain dominates; at high liability the verifier's avoided false passes dominate---and a standalone verifier captures nearly all the false-pass benefit of the full planning-plus-verification stack at a fraction of its cost. 

---
# SkillAA: Attribution-Guided Skill-Graph Updating with Targeted Validation and Rollback 

**Authors**: Ziqiao Shang, Ling-Yue Ge, Lan-Zhe Guo  

**Link**: [PDF](https://arxiv.org/pdf/2609.20455)  

**Abstract**: External skills provide domain procedures without parameter updates, but existing methods often edit skills directly from failed rollouts without structured routing from an observed failure to an editable location; existing skill graphs also underuse semantic boundaries, object addresses, and topological dependencies for skill retrieval, targeted updating, and scoped validation. We introduce SkillAA (Skill Abductive Attribution), a structured skill-optimization framework for frozen language models. It represents skill applicability, execution, and composition in a unified graph, allowing the same structure to support skill selection, attribution-guided repair, and update validation. SkillAA contrasts successful and failed executions to route candidate repairs to specific graph objects, updates only the selected local structure, and uses Local and Big Gates to screen candidate changes before commitment. With gpt-5.6-sol, SkillAA reaches 81.5%, 66.7%, and 91.2% on SearchQA, LiveMath, and DocVQA, respectively, and attains the highest observed mean in every main setting. These results support the utility of attribution-guided graph editing and graph-scoped validation. 

---
# The Organization of Inference: Information, Resource Constraints, and AI Production 

**Authors**: Yukun Zhang, Kemu Xu, Yishen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.20449)  

**Abstract**: The economic value of inference depends on how capacity and task information are distributed across stages of AI production. We study these organizational margins using controlled workflow experiments on externally verified software-engineering tasks. In two matched resource panels, direct execution records the same success rate of 59.6 percent at logical-token ceilings of 12,000 and 24,000, while success under information-constrained planning rises from 36.2 to 51.2 percent. The planning disadvantage narrows by 15.0 percentage points (95 percent task-cluster bootstrap interval: 4.2 to 25.8). A strict read-only planning campaign varies whether the planner sees the task issue. At 12,000 tokens, issue access raises success by about 16 percentage points over issue-hidden planning. Compared with direct execution, task-informed planning is about 10 points lower at 12,000 tokens; at 24,000 tokens, it shows a 29.6-point advantage. In the resource panels, direct execution uses substantially less than either ceiling, while the planning workflow's binding rate falls from 46.2 to 0.8 percent and downstream execution accounts for 89.9 percent of the increase in total use. Scale determines the capacity available to a system; workflow and information structure shape the productive value 

---
# Generating Heterogeneous 3D Geological Microstructures from 2D Images via a Stable Diffusion-Adversarial Model 

**Authors**: Ali Aouf, Eric Laloy, Bart Rogiers, Christophe De Vleeschouwer  

**Link**: [PDF](https://arxiv.org/pdf/2609.20358)  

**Abstract**: Characterizing the physical properties of clay and cementitious materials matters across many fields, from materials science to geological waste disposal. Property simulation typically calls for 3D imaging, which is expensive, not always accessible, and technically limited for certain materials. Recent progress in deep generative models offers a way around this, reconstructing 3D volumes from the more easily acquired 2D images.
Among GAN-based methods for 3D microstructure generation, SliceGAN has shown strong results for homogeneous isotropic and anisotropic systems. It struggles, however, to capture the finer detail of more complex heterogeneous microstructures, which motivates alternative generative frameworks.
We introduce a hybrid approach that draws on the stability and generation quality of denoising diffusion models. Since no 3D ground truth is available, we replace the standard denoising loss with an adversarial loss, which yields a stable training process in our experiments. We show that the resulting model generates microstructures of varying complexity with minimal slice artefacts and close agreement with ground-truth phase fractions and structural descriptors. 

---
# A Qualitative Model for Reasoning about Path and Support 

**Authors**: Abhishek Jaiswal, Zoe Falomir  

**Link**: [PDF](https://arxiv.org/pdf/2609.20349)  

**Abstract**: Spatial reasoning abilities correlate strongly with performance in STEM fields. Games offer a compelling medium for training these critical skills in developing children who have a natural proclivity for play. However, to facilitate human-like tutoring and player guidance, these games require an AI agent capable of making commonsense inferences from spatial events. Qualitative reasoning (QR) models appear to be a suitable framework for these application domains. As these models reason in symbolic representations, they can seamlessly translate game states into interpretable feedback for human-like player guidance. This paper introduces a hybrid qualitative model designed for Camelot Jr., a block-puzzle game that requires constructing multi-level bridges to connect two avatars stationed on separate towers. The game poses a challenge for the player, who must make platforms stable, plan their path, and ensure they use all the provided blocks. To handle the precise physics required by the domain, we integrate a mathematical center-of-mass stability logic to guide our qualitative solver. Our work facilitates spatial skill training in Camelot Jr. and contributes to the development of human-centric, explainable game-playing agents. 

---
# Structured Four-Stage Legal Translation: From Natural-Language Traffic Rules to PROLOG 

**Authors**: May Myo Zin, Wachara Fungwacharakorn, Ken Satoh, Katsumi Nitta  

**Link**: [PDF](https://arxiv.org/pdf/2609.20334)  

**Abstract**: Traffic regulations are written for human interpretation and therefore rely on shared background knowledge and flexible phrasing, which inherently introduce ambiguity, context dependence, and semantic underspecification. These linguistic characteristics conflict with the precision required by computational reasoning engines such as Prolog, which demand explicit logical structure. This study evaluates two baseline translation approaches, Natural Language to Prolog ($NL\rightarrow Prolog$) and Logical English to Prolog ($LE\rightarrow Prolog$), and introduces a new reasoning-guided translation framework called Structured Four-Stage Legal Translation ($S4L\rightarrow Prolog$). The proposed S4L framework performs semantic role extraction, scene completion, logical mapping, and Prolog rule generation within a single guided prompt, enabling direct translation of raw traffic rules into executable logic without human intervention. A benchmark consisting of twenty real-world traffic rules was used to evaluate each approach in terms of syntactic validity, semantic correctness, and logical completeness. $S4L\rightarrow Prolog$ achieves the highest accuracy, correctly formalizing 75 percent of the rules, while $NL\rightarrow Prolog$ reaches 60 percent and $LE\rightarrow Prolog$ reaches 55 percent. Qualitative analysis further shows that S4L captures implicit causal relations, deontic modality, and exception structure more reliably than the baselines. These results demonstrate that structured reasoning prompts can substantially improve the reliability of natural-language-to-logic translation for legal and safety-critical applications. 

---
# NeuSOGA3D: A Neuro-Symbolic Framework for Explainable 3D Geometric Reconstruction 

**Authors**: Qingde Li, Qingqi Hong, Zihan Li, Jie Tian  

**Link**: [PDF](https://arxiv.org/pdf/2609.20323)  

**Abstract**: Three-dimensional reconstruction from unorganized point clouds remains a challenging problem in computer vision, geometric modeling, and computer-aided design. While neural implicit methods achieve impressive reconstruction accuracy, geometry is typically encoded in latent representations that limit interpretability and reuse within engineering workflows.
We present NeuSOGA3D (Neuro-Symbolic Geometric Abstraction in 3D), a hybrid framework that combines learned perceptual priors inherited from NeuSOGA with explicit symbolic geometric reasoning. The method projects point clouds onto principal orthographic planes, constructs symbolic implicit spline representations from the resulting observations, and fuses them through shape-preserving constructive solid geometry operations to generate a coarse visual hull. Additional geometric detail is recovered through cross-sectional decomposition and volumetric reconstruction using Partial Shape-Preserving Splines.
Unlike conventional neural implicit approaches, NeuSOGA3D progressively transforms observations into explicit symbolic entities, including control polygons, implicit spline fields, cross-sections, and volumetric lofts. Experiments on all forty categories of the ModelNet40 benchmark demonstrate the ability of the framework to recover structurally meaningful and CAD-compatible geometric representations from diverse point-cloud observations. The results highlight the potential of combining learned perception with symbolic geometric reasoning for explainable geometric intelligence. 

---
# Diagnose, Recover, Certify: Task Readiness under Hidden Dynamics Changes 

**Authors**: Nguyen Viet Tuan Kiet, Huynh Thi Thanh Binh  

**Link**: [PDF](https://arxiv.org/pdf/2609.20304)  

**Abstract**: A deployed control policy can conceal consequential dynamics changes: an actuator may lose effectiveness without affecting the current task when the policy rarely excites it, despite being critical for a future task that has not yet been specified. We introduce task readiness under dormant dynamics drift, a decision problem that unifies active change diagnosis and post-change control recovery under a limited, task-agnostic interaction budget. An agent must identify whether and where local dynamics have changed, use a small number of informative interactions to characterize the change before downstream task identity is revealed, and subsequently provide each candidate task with either a recovered policy and a calibrated lower bound on its achievable return or an abstention decision to a safe fallback. We propose Evidence-Gated Matched-Pulse Transport, an intervention-based Bayesian procedure that couples fault localization with estimation of actuator effectiveness through a shared matched-response representation, thereby preserving diagnostic reliability while converting localized evidence into recovery-relevant uncertainty. This uncertainty is propagated to task-conditioned policy selection and readiness certification, enabling deployment decisions that explicitly trade off expected performance, confidence, and fallback use. We evaluate the resulting framework on a diverse suite of dormant-actuator benchmarks spanning multiple simulators, under a protocol that separates diagnosis from capability recovery, scores deployment by readiness coverage, selective risk, and interaction cost as well as return, and identifies the fault regimes in which transported evidence is decisive. 

---
# AgentPProf: Semantic Profiler for Long Horizon AI Agents 

**Authors**: Yusheng Zheng, Chaokun Chang, Yu Mao, Tianyuan Wu, Yuxi Huang, Tao Ma, Wenan Mao, Shuyi Cheng, Andi Quinn, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.20301)  

**Abstract**: AI agents increasingly orchestrate long-running activities with users, tools, and system resources for days and weeks. To improve agent quality, safety, and cost efficiency, developers need to determine where failures happen, what triggers unsafe effects, and which tasks consume the most budget, then optimize those tasks. In systems software, profiling answers similar questions by aggregating resource consumption and attributing it to responsible code paths to identify hotspots. Yet existing agent observability tools focus on per-execution debugging and tracing rather than cross-run, long term profiling, making these questions difficult to answer at scale. Agent observability needs profiling, not only debugging, but profiling agents is challenging: the responsible entities are task intent like diagnose authentication, compare branches rather than code paths, and lack stable identifiers for aggregation. We propose a semantic operation stack model that adapts profiling to agent trajectories. Uniform operations represent all activities, and operation stacks replace the runtime call stack, enabling hierarchical attribution at different granularities. We observe that an agent's task occupies a contiguous span and decomposes into subtasks, so we introduce recursive operation segmentation, which recursively splits trajectories at task boundaries. AgentPProf is a profiler that aggregates agent trajectories into pprof-compatible profiles, enabling flame graph visualization and analysis. AgentPProf reaches 0.764 $B^3$ F1 against human annotations on CodeTraceBench. On three problem-localization benchmarks, the profile raises MAP by up to 56%, demonstrating that it effectively attributes resources, locates problems, and helps optimize token cost at practical profiling cost. AgentPProf is available at this https URL. 

---
# JEPA-WAM: Connecting Generated Visual Instructions to World Action Models through JEPA Latent Representations 

**Authors**: Tianbin Liu, Jian Zhu, Taiyi Su, Jianjun Zhang, Chong Ma, Zitai Huang, Yi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2609.20277)  

**Abstract**: World Action Models (WAMs) have demonstrated strong robotic manipulation capabilities by augmenting pretrained video generative models with action experts. However, current WAMs still show limited instruction-following ability when conditioned solely on text instructions. We argue that this limitation stems in part from a structural imbalance in robot-learning data: rich visual-action trajectories are often paired with sparse and repetitive language annotations, allowing policies to identify tasks from visual context and motion regularities rather than grounding the instruction itself. To address this limitation, we introduce JEPA-WAM, which augments each text instruction with a bank of stochastically generated visual instructions, providing diverse visual cues for instruction following. Specifically, JEPA-WAM uses an off-the-shelf text-to-image generator to sample multiple task-completion images conditioned on the text instruction, without training the generator. Although these generated images may differ from the current visual scene in appearance and layout, they remain semantically aligned with the instruction and serve as visual goal references. To focus on task-level semantics beyond appearance, we encode these references with a frozen V-JEPA 2.1 encoder. The resulting dense goal representations are compressed into compact goal tokens that condition both the video and action experts through cross-attention. We further construct a real-robot instruction-following benchmark covering in-distribution, out-of-distribution scene, and out-of-distribution instruction settings. On this benchmark, JEPA-WAM achieves success rates of 87.3%, 74.5%, and 80.9% in these three settings, outperforming {\pi}0 and Fast-WAM by at least 10.0, 27.3, and 14.5 percentage points, respectively. 

---
# AI-Driven Real-Time Relay Optimisation in Smart Urban NR-V2X Networks via Learning-to-Optimise Graph Neural Networks 

**Authors**: Giambattista Amati, Federica Mangiatordi, Emiliano Pallotti, Simone Angelini  

**Link**: [PDF](https://arxiv.org/pdf/2609.20271)  

**Abstract**: Reliable and low-latency communication is a fundamental requirement for smart city services and Industry 4.0 applications enabled by NR-V2X networks. However, limited Road-Side Unit (RSU) deployment and complex urban propagation conditions often prevent Connected and Automated Vehicles (CAVs) from maintaining stable connectivity. This paper proposes an AI-driven Learning-to-Optimise (L2O) framework based on Graph Neural Networks (GNNs) for real-time multi-hop relay selection in NR-V2X systems. The vehicular network is modelled as a graph, where nodes represent CAVs and RSUs, and edges encode radio-link characteristics. An offline Mixed-Integer Linear Programming (MILP) formulation provides optimal relay decisions used as supervision for training an edge-aware Graph Isomorphism Network with Edge Features (GINE). Extensive experiments on realistic urban datasets demonstrate that the proposed approach achieves near-optimal connectivity performance, recovering up to 11.3% connectivity gain, while reducing execution time by orders of magnitude (up to 100 x speed-up) compared to MILP. The framework enables scalable and real-time network control, making it suitable for smart city and Industry 4.0 deployments. 

---
# When AI Agents Commit: Cognitive Serializability Across Data, Evidence, Policy, and Authority 

**Authors**: Jun He, Deying Yu  

**Link**: [PDF](https://arxiv.org/pdf/2609.20261)  

**Abstract**: Autonomous agents derive concrete mutations from database reads, retrieved evidence, policy, beliefs, and delegated authority. Those inputs may change while reasoning is in progress. Database isolation orders the submitted transaction; agentic transaction processing determines whether a proposal satisfies an executable contract. Neither guarantee establishes a common valid point for the mutation and its derivation inputs unless the contract represents the relevant predicates. Typed dependency tokens distinguish content integrity from applicability, and trusted mediation captures the values exposed to reasoning. Under strict Cognitive Serializability, committed effects admit a serial order and a logical event at which every value exposed to derivation is unchanged. The fences last until the runtime event that realizes the sealed durability domain. The weaker Effect-Compatible Cognitive Admission recertifies an effect against a simultaneously held current dependency vector and current policy without claiming to serialize the original stochastic derivation. TCT combines immutable versioned executable definitions, registry-derived authority plans, sealed envelopes, guard-first commit transactions, post-seal envelope- and witness-bound grants, co-committed receipts, idempotent grant finalization, and receipt-driven epistemic reconciliation. Complete registered footprints and a single growing phase induce an acyclic lock-point order over local guards and incompatible external reservations. The corresponding results give serializability conditions and an observational-equivalence boundary for zero-error soundness and positive progress. A falsification suite tests the implementation obligations: the prototype prevented all injected anomalies and added 3.22 ms mean commit overhead. 

---
# JointMatch: A Unified Heterogeneous Graph Neural Solver for Large-Scale Ride-Sharing Matching 

**Authors**: Kun Zhao, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.20200)  

**Abstract**: Ride-sharing platforms must continuously decide which open requests to bundle into shared trips and which idle vehicles should serve them. The dominant academic approach decomposes this into two sequential matching problems -- request pairing first, then vehicle assignment -- and applies a separate solver to each. This decomposition is convenient computationally but loses revenue and scales poorly because the first stage commits to ride bundles before the available vehicles are known. We propose JointMatch, a learning-based framework that handles request pairing and vehicle assignment together on a single graph. The graph is sparsified by spatial proximity so that its size grows linearly rather than quadratically with the number of vehicles and requests, and a graph neural network scores all candidate decisions in one forward pass. On the New York City Yellow Taxi data, the framework already exceeds both the classical Blossom heuristic and a faithfully-trained two-stage GNN baseline -- often by a wide margin -- and at city scale (fleet 10000) it runs more than $20\times$ faster per dispatch epoch than either. A supervised training stage closes most of the remaining revenue gap, and a policy-gradient fine-tune aligns the trained model with realised revenue. 

---
# Sequential Contextual Fit Predicts Human Behavioural and Neural Dynamics Across Domains 

**Authors**: Kun Sun, Rong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.20179)  

**Abstract**: Human perception, action and decision making unfold in sequences, but computational predictors are often domain-specific. This study computes and tests sequential contextual fit (SCF), an embedding-based measure of how well a current information state matches its recent context. The metric uses a simple recency-weighted similarity kernel and can be applied to words, sounds, visual scenes, affective states, choices, actions and neural representations. Across language processing, music-evoked emotion, a subset of audiovisual emotion EEG data, gambling decisions, human activity recognition and decision-related EEG, lower contextual fit predicted longer processing times, larger affective or behavioural transitions and stronger neural-state changes. These effects remained after controlling for established predictors including surprisal, reinforcement-learning prediction error, acoustic change, visual change and sensor change. SCF therefore provides a computational measurement layer for relating contextual compatibility to behavioural processing and cognitive/neural state-transition dynamics. 

---
# PaGNet: A Panel-Aware GBDT--Neural Network for Multi-Target Corporate Tax Avoidance Proxy Forecasting 

**Authors**: Wonho Song, Hyungjoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2609.20177)  

**Abstract**: Forecasting corporate tax avoidance proxies from firm--year panel data is challenging because predictive signals are distributed across short firm histories and related targets, while screening-oriented use requires transparent model behavior. We propose PaGNet (Panel-Aware GBDT--Neural Network), a two-branch hybrid that combines a LightGBM branch using panel-temporal summaries with a Panel-MLP branch using attention-pooled temporal aggregation and shared-trunk multi-task learning. A per-target validation-optimal blender produces both the final prediction and a compact branch-reliance diagnostic without trainable fusion parameters. On the KoTaP panel of 1{,}754 Korean listed firms from 2011--2024, PaGNet is evaluated under a leakage-free, shared-hyperparameter protocol across four feature regimes. In the direct-proxy-lag-excluded FS1 regime and the tax-history-augmented FS2 regime, accrual targets (TSTA, TSDA) route stably to the LightGBM branch, where PaGNet raises explained variance over the strongest of six baselines by roughly $0.08$--$0.11$ on the primary split. GETR often leans toward the neural branch, while CETR exposes a validation--test branch-selection mismatch rather than a stable branch assignment. A panel-flatten control shows that most accrual gains come from observed multi-year base-panel values, with PaGNet's panel-aware representation adding a smaller but directionally consistent refinement. Rolling-origin analysis confirms stable accrual routing, bounds ETR diagnostics to split-specific behavior, and identifies a far-horizon split where supervised models underperform naive persistence. PaGNet is therefore best viewed not as a universally superior tabular learner, but as a proxy-aware panel model that combines competitive forecasting with explicit per-target branch-reliance reporting. 

---
# MTVA-Bench: Evaluating the Language Model Inside Cascaded Voice Agents 

**Authors**: Pritish Mishra, Ishaan Kumar, Akshat Mandoli, Sudarshan Kamath  

**Link**: [PDF](https://arxiv.org/pdf/2609.20152)  

**Abstract**: Generally, most voice agents are cascaded systems, i.e., an ASR model transcribes the caller's audio, a language model reads the transcript and decides what to say and which backend tools to call, and a TTS model speaks the reply. Nearly all of the decision making happens in the language model, but existing evaluations measure it either too broadly or too narrowly. End-to-end voice benchmarks score the full pipeline, so recognition errors and model errors mix into a single number. LLM benchmarks isolate the model but they do not evaluate what makes real phone calls hard, such as transcription issues, caller's voice being split across messages and the requirement that replies follow the language and script specified. We introduce the Multi-Turn Voice Agent Benchmark (MTVA-Bench), which evaluates the language model on the same conditions it faces inside a cascaded system. The caller is played by an LLM following a set of rubrics and tool calls are answered by a mock backend which responds to the arguments the model actually sent. The benchmark contains 49 agents working across 490 reviewed scenarios and supports 7 languages. Scoring is a combination of deterministic checks on tool calls with two LLM judges, one that scores scenario specific rules and one that grades conversation quality without access to the task. Both judges must cite specific messages from the transcript. Task and conversation scores are weighted equally, since a call can complete its task and still go badly for the caller. In a seven-model study, six of the models select the correct tool within 6.4 points of one another, but their overall scores span 24.4 points. Most of the gap comes from argument values, action ordering, rule compliance, and what the model says around its tool calls. 

---
# Perception, Layout, and Validation: Calibrated Confidence for Reliable Straight-Through Processing of Financial Documents 

**Authors**: Yichao Jin, Yushuo Wang, Yuxuan Han, Kwan Ching Yee Sonia, Weiyang Song, Chiu Jin-Chun Kent, Wong Chong Hwee, Wong Tiong Kiat, Kenneth Zhu Ke, Jingyuan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2609.20110)  

**Abstract**: Straight-through processing (STP) on extracted key-value fields from financial documents without human review requires a calibrated probability together with a bounded guarantee on the residual error of the auto-approved tier. The emergence of modern Vision Language Models (VLMs) provides an out-of-the-box capability for extracting the key-values, but their verbalized confidence signals are unreliable and weakly track field correctness. This paper introduces a decomposed confidence layer along three interpretable channels, including perception, layout, and validation. Together with a final conformal risk control, the score can be used for reliable STP of financial documents. The method is validated on three public datasets covering real invoices, synthetic invoices, and ad-buy forms, using two different VLM families (Qwen3.6-27B and Gemini-3.1-Flash-Lite). Our decomposed score consistently improves the separation of correct from incorrect extractions, substantially raising the AUROC from 0.54-0.74 for VLM verbalized signals to 0.90-0.99 with contributions from all three designed channels. Crucially for industrial deployment, this enables usable STP. The native VLM confidence signals could clear only 0.1%-7.0% of fields under risk control at a target error of <10%. In contrast, the proposed method auto-approves 49-72% of fields while holding the empirical error of the accepted tier at or below the target. 

---
# Solving Minimum Span Antibandwidth and Cyclic Antibandwidth Labeling Problems 

**Authors**: Hieu Truong Xuan, Khanh To Van  

**Link**: [PDF](https://arxiv.org/pdf/2609.20091)  

**Abstract**: The Antibandwidth and Cyclic Antibandwidth problems are NP-hard graph labeling problems that aim to maximize the minimum (cyclic) distance between labels assigned to adjacent vertices. Extensive research on these problems has resulted in a variety of mathematical formulations and computational approaches. However, their minimum span perspective, in which a prescribed minimum (cyclic) distance is fixed and the objective is to minimize the label span, has received comparatively little attention. In this paper, we consider this complementary perspective by introducing the Minimum Span Antibandwidth/Cyclic Antibandwidth Labeling (MSABL/MSCABL) problems and developing a unified Boolean Satisfiability (SAT)-based framework for solving them. The SAT-based framework formulates MSABL/MSCABL as a sequence of decision problems and exploits their monotonicity to accelerate the search process. We also consider two SAT solving strategies, parallel and incremental SAT solving: the former examines multiple candidate spans concurrently, while the latter reuses a single SAT instance while progressively restricting the label domain. The proposed approaches are evaluated on benchmark instances from the Harwell-Boeing Sparse Matrix Collection and compared with CPLEXCP, CPLEXMIP, and Gurobi. The results show that SAT-based approaches are highly competitive in solution quality, with the parallel approach performing best overall for MSCABL and the incremental approach for MSABL. With the no-hole constraint, they remain competitive with CPLEXCP and significantly outperform CPLEXMIP and Gurobi, particularly for MSCABL. These results demonstrate the effectiveness of SAT solving as an exact approach for MSABL and MSCABL. 

---
# UnifiedPlayers: Enhance Tool-Integrated Reasoning in Agentic Reinforcement Learning 

**Authors**: Wenjie Liao, Liangjie Zhao, Zehong Cao  

**Link**: [PDF](https://arxiv.org/pdf/2609.20089)  

**Abstract**: Self-evolving methods reduce the need for human-annotated trajectories by allowing tool-using agents to generate their own training data. Yet existing methods typically separate trajectory generation from evaluation, relying on static verifiers that cannot adapt to emerging failure modes or self-consistency signals that may reinforce errors shared across trajectories. Jointly adapting planning, execution, and evaluation offers a promising alternative, but introduces a fundamental coordination challenge: each component continuously changes the data or feedback used to train the others. We address this challenge with \textbf{UnifiedPlayers}, a cooperative framework comprising a Planning Player that generates tasks, an Execution Player that produces multi-turn trajectories with Python tool calls, and an Evaluation Player that constructs executable verifiers. We design role-specific rewards that coordinate the three players toward a shared learning objective under GRPO. Across two model backbones and twelve reasoning benchmarks, UnifiedPlayers outperforms the strongest prior baseline by at least 3.5\% on mathematical reasoning and 3.9\% on general reasoning tasks. Moreover, the learned verifier achieves 84.2\% adversarial detection accuracy, while its reward signal exhibits 2.03$\times$ higher per-question variance than a self-consistency baseline, providing more discriminative verifications. These results highlight cooperation among specialized players as a promising path toward self-enhanced tool-integrated agents. 

---
# A Proposal for an Agentic AI Architecture to Support Multi-Domain Decision-Making in the Brazilian Armed Forces 

**Authors**: Gioliano de Oliveira Braga, Sidnei Barbieri, Ágney Lopes Roth Ferraz, Wagner Comin Sonaglio, Henrique Curi de Miranda e Lourenço Alves Pereira Jr  

**Link**: [PDF](https://arxiv.org/pdf/2609.20080)  

**Abstract**: The growing complexity of multi-domain operational environments (land, aerospace, naval, cyber, and electromagnetic spectrum) has increased the volume and velocity of data reaching command-and-control (C2) centers, straining the observe-orient-decide-act (OODA) decision cycle. Artificial Intelligence (AI) systems currently employed in defense are, in general, reactive and isolated tools that still rely heavily on human operators to integrate information, assess scenarios, and formulate courses of action. This paper proposes a conceptual Agentic AI architecture for AI systems that can plan, access data sources, execute tools, and act autonomously and audibly, aimed at supporting decision-making across the three Brazilian Armed Forces (Navy, Army, and Air Force). Four application fronts are discussed (decision support, situational analysis, feasibility studies, and countermeasure suggestion), as well as the data and sensor access requirements and the security and permission safeguards necessary for responsible employment across administrative, strategic, operational, and tactical contexts. 

---
# Tailored to you: longitudinal effects of personalising language models 

**Authors**: Canfer Akbulut, Justine Breuch, Arianna Manzini, Lujain Ibrahim, Matija Franklin, Roma Patel, Iason Gabriel, Kristian Lum, Laura Weidinger  

**Link**: [PDF](https://arxiv.org/pdf/2609.20077)  

**Abstract**: Interest in developing personalised language models is rapidly growing. While personalisation is often viewed as a mechanism to better serve diverse user needs, the effects of sustained interactions with personalised models on people's perception of and behaviour toward AI remain poorly understood. Most critically, downstream consequences outside the immediate human--AI interaction loop, such as effects on users' self-perceptions and interpersonal relationships, remain largely unexamined. In this study, we recruited 992 participants to complete daily advice-seeking interactions with language models over the course of five days, comparing outcomes from a non-personalised baseline against two personalisation approaches: memory-based (conditioned on prior conversational history) and survey-based (conditioned on information collected through a pre-study intake survey). We find that several changes in human-AI interaction over time are driven primarily by repeated exposure rather than personalisation itself. However, participants interacting with personalised models experienced differences in advice-seeking and information-sharing attitudes and behaviours: participants in the memory-based condition engaged in greater self-disclosure and rated the model as less creepy, while participants in the survey-based condition reported higher regret about having shared personal information with the AI. We conclude by highlighting the nuanced effects of different personalisation approaches on interaction outcomes, and discussing the implications of these findings for the responsible design and deployment of personalised AI systems. 

---
# Marginal utility, matrix factorization, and the Key-Value (KV) cache: a unified information-economic framework for sovereign geo-mining inference 

**Authors**: Caroline Gans Combe  

**Link**: [PDF](https://arxiv.org/pdf/2609.20068)  

**Abstract**: This paper builds a theoretical bridge between the economic notion of marginal utility and two machine-learning constructs, matrix factorization and the Key--Value cache of transformer language models. The singular value spectrum of a rating matrix is shown to be a diminishing marginal utility schedule for latent factors, the eigenvalue spectrum of the projected covariance operator to be the marginal utility schedule of a model's learned representation, and cache eviction and low-rank cache compression to be instances of constrained utility maximization under a memory budget. The three collapse into a single allocation rule: retain the top dimensions whose eigenvalue exceeds the shadow price of the binding constraint. The framework is applied to the automated extraction of structured information from geo-mining documents, where it motivates a multi-pass inference protocol, a layer-wise TIES model merging procedure, and a selection policy combining extraction quality, localization drift and energy, scalarized with a Conditional Value-at-Risk term on drift. Two empirical contributions are reported. An 11.2-million-parameter hierarchical classifier, trained in about five minutes on a single GPU, reaches 90.0 per cent level-1 accuracy on a held-out test set from a 973-document uranium-exploration corpus, against 92.0 per cent for a proprietary model on a fifty-document human audit of the same corpus, at a latency of 2.62 ms per card against approximately 2,000 ms for the API and at negligible cost. A diagnostic of uniform-density TIES merging exposes a reproducible degenerate mode in which the merged model returns token-identical outputs across five geographically distinct districts while declaring high confidence; re-executing the merge under layer-wise calibrated densities removes that signature on the diagnostic sample. The full-scale extraction benchmark, including LoRA fine-tuning, is reported as projected rather than measured and remains an empirical extension of this work. 

---
# FCA-Guided Counterfactual Explanations for Multi-Modal Breast Cancer Diagnosis: A Framework Achieving Perfect Validity with Emergent Sparsity 

**Authors**: Abdullahi Isa, Souley Boukari, Muhammad Aliyu  

**Link**: [PDF](https://arxiv.org/pdf/2609.20067)  

**Abstract**: Deep learning models for multi-modal breast cancer diagnosis achieve high predictive accuracy but remain clinically unacceptable without actionable, counterfactual explanations. Attribution-based methods (LIME, SHAP) are categorically inapplicable to this purpose, as they generate no alternative instances and thus cannot be evaluated on counterfactual quality metrics. This investigation provides empirical evidence that FCA-Guided Counterfactual (FCA-CF) framework that uses a Formal Concept Analysis (FCA) concept lattice as a hard structural constraint on counterfactual search, operating over a multi-modal TCGA-BRCA dataset. We benchmark against four genuine counterfactual methods: Wachter-style CF, DiCE, FACE, and NICE, evaluated on 60 benign-predicted TCGA-BRCA instances. The FCA-CF framework achieves Validity = 1.0000 (100% of counterfactuals successfully flip the prediction), Sparsity = 2.37 features changed (best among all valid methods), and Proximity = 0.900 (normalised L2-based, matching NICE as joint best). The classifier achieves Accuracy = 0.980, F1 = 0.976, ROC-AUC = 0.9947. Ablation analysis confirms that the FCA lattice constraint is the primary sparsity driver (removing it increases sparsity by +40%, p < 0.001, Cohen's d = 0.78), while Phase C greedy refinement accounts for the largest individual contribution (+113% sparsity increase when disabled, p < 0.001, d = 5.01). FCA-guided counterfactual generation achieves a clinically important Pareto-dominant outcome; it is simultaneously the sparsest and among the most proximate of all valid methods, with perfect validity. The emergent sparsity property arising from lattice topology rather than numerical penalty terms constitutes a structurally novel contribution to the counterfactual explanation literature. 

---
# WiCleanData: Guaranteeing the Type Consistency of Wikidata by Taxonomy Refinement and Constraint Enforcement 

**Authors**: Yiwen Peng, Marc Jeanmougin, Thomas Bonald  

**Link**: [PDF](https://arxiv.org/pdf/2609.20057)  

**Abstract**: Because of its collaborative nature, Wikidata suffers from errors, in- consistencies, and excessive complexity, such as redundant classes, ambiguity between instances and classes, wrong taxonomic paths, and type constraint violations. The manual curation of these issues is infeasible at scale. To address these challenges, we introduce WiCleanData, a refined version of Wikidata with a consistent tax- onomy and free from type constraint violations. Specifically, we have designed an automated pipeline that first cleans the taxonomy with language model assistance, then simplifies type constraints by hierarchical aggregation, and finally filters facts accordingly. The resulting knowledge graph, free from any type violation, is made publicly available via a Web interface, enabling easy exploration and downstream applications. 

---
# MAGMA-GEN: Validated Recovery Supervision from Ambiguous Failures via Counterfactual Re-Execution 

**Authors**: Loan Bernat, Matthieu Grard, Ariane Herbulot, Florent Lamiraux  

**Link**: [PDF](https://arxiv.org/pdf/2609.20056)  

**Abstract**: Hierarchical robotic systems executing long-horizon manipulation tasks must make high-level semantic decisions that orchestrate stochastic low-level skills. In this setting, failed rollouts are ambiguous: a poor downstream state may reflect an invalid high-level decision, partial observation, or a valid decision whose physical execution failed. Traditional supervised learning lacks data for such recovery states, while reinforcement learning struggles with sparse rewards and non-local credit assignment. We propose MAGMA-GEN, an on-policy data-generation pipeline that converts ambiguous failed rollouts into validated recovery supervision. MAGMA-GEN first uses a privileged coach to hypothesize an early decision-level error and propose localized correction or recovery actions. Because this diagnosis is fallible, candidates are retained only if re-execution from the same state under matched conditions improves downstream progress. This produces supervised examples from the agent's own failure distribution without per-step human demonstrations. Evaluated on interactive long-horizon manipulation tasks, MAGMA-GEN improves task success and recovery capabilities, against distillation and trajectory-repair baselines under evolving task constraints in both simulation and real-robot execution. 

---
# DART: Distillation-Aware Reparameterization for Training-Free LoRA Reuse in Few-Step Video Diffusion Models 

**Authors**: Shihong Li, Juntao Xu, JinCao, Maowen Tang, Jun Huang, Jintao Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.20051)  

**Abstract**: Step distillation reduces the cost of video generation, but reusing a LoRA trained for a longer trajectory can alter its functional effect or degrade target quality. Static parameter compatibility offers one perspective on this problem; our observations show that similar measured geometry can coexist with different adapter behavior under a shortened denoising schedule. We propose DART, a training-free method that combines low-rank coordinate transport with target-schedule response calibration using forward evaluations and no source training videos. On a four-step Wan2.2 target, DART-F improves the joint quality score from 0.9029 to 0.9227 and changes macro functional retention from -0.4644 to +0.1349. Component analysis shows that calibration accounts for most of the quality improvement, while coordinate transport provides complementary gains when combined with calibration. Adapter-level results reveal positive functional effects for some adapters and strong attenuation with reduced negative functional effects for others. Evaluations on two additional targets show the same aggregate trend. These results motivate evaluating distilled-model LoRA reuse jointly through functional preservation and negative-transfer avoidance, without assuming recovery for every adapter. 

---
# Can Data Attribution Filter Out Subliminal Learning? Not Reliably 

**Authors**: Moritz Weckbecker, Sweta Jena, Jonas Müller, Ponnurangam Kumaraguru, Sebastian Lapuschkin, Wojciech Samek, Louis Jaburi, Gonçalo Paulo  

**Link**: [PDF](https://arxiv.org/pdf/2609.20027)  

**Abstract**: Subliminal learning allows language models to transmit behavioral traits through training data with no obvious semantic relationship to those traits, undermining content-based data filtering as a safety intervention. Training data attribution offers an alternative: it identifies the training examples responsible for a given model behavior, independent of their semantic content, and so may apply in exactly the cases where semantic inspection fails. We evaluate three gradient-based attribution methods (GradCos, a contrastive GradCos variant, and EK-FAC) across three models, comparing them against divergence tokens, a strong baseline previously shown to localize subliminal learning (albeit one that requires access to counterfactual teacher models). Filtering at the token level, EK-FAC mitigates a significant part of the effect, the other methods provide little benefit, and all mostly fall short of divergence tokens. Filtering entire samples is less effective for every method, though EK-FAC often gives a stronger signal than divergence tokens in this setting. Success is inconsistent across methods and settings: variants that work well for some model-preference combinations fail for others, and we do not identify a consistent explanation for these differences. Our results suggest that gradient-based attribution can identify data responsible for subliminal learning in some settings, but that some approximations are more reliable than others. 

---
# FedeRICo: Federated Region-Influenced Coupling for Traffic Flow Prediction 

**Authors**: Fermin Orozco, Man Luo, Johan Wahlström  

**Link**: [PDF](https://arxiv.org/pdf/2609.20026)  

**Abstract**: Urban traffic forecasting often relies on information distributed across stakeholders who may be unable to share raw data due to privacy or commercial constraints, motivating federated spatial-temporal approaches. In such federated settings, each client observes traffic over a distinct sensor subgraph with its own spatial topology and temporal dynamics, leading to significant heterogeneity across clients. Existing federated spatial-temporal methods typically rely on model parameter aggregation and provide limited mechanisms for recovering spatial dependencies across client boundaries. This introduces two key limitations. Specifically, parameter aggregation across heterogeneous graph domains tends to dilute client-specific representations, while road network partitioning breaks the propagation of traffic dynamics across client boundaries. To address these challenges, we propose FedeRICo, a federated traffic forecasting framework that combines gradient-level collaboration with boundary-aware residual communication. FedeRICo employs a dual-branch forecasting architecture in which a globally guided branch captures transferable forecasting structure, while a private residual branch preserves client-specific corrections and incorporates boundary residual signals. The global branch is coordinated through gradient alignment across all clients, enabling collaborative optimisation without destructive parameter interference. To recover cross-client spatial dependencies, boundary messages are extracted through a trend-residual decomposition that suppresses periodic structure and communicates only transient spatial-temporal residual signals between physically adjacent clients. Experiments across four real-world traffic forecasting benchmarks demonstrate that FedeRICo consistently outperforms state-of-the-art federated spatial-temporal baselines while maintaining competitive training runtime. 

---
# Geopolitical Divisions Across Languages in Large Language Models 

**Authors**: Maxim Chupilkin  

**Link**: [PDF](https://arxiv.org/pdf/2609.20005)  

**Abstract**: People increasingly turn to AI chatbots for news and explanations of world events. But do they receive the same political answers when they ask in different languages? Here we show that the language of a question can change how the same AI systems assess the war in Ukraine. We ask GPT, Claude and Gemini to evaluate twenty statements about the war in 112 languages, collecting 67,200 responses. The balance between Russia-leaning and Ukraine-leaning responses differs across languages. When we group responses by countries' official languages, they follow a pattern resembling worldwide political divisions: relatively more Russia-leaning answers correspond to more favourable public views of Russia, less support for Ukraine in United Nations votes, and less aid to Ukraine. The broad pattern recurs across all three models and remains when individual statement pairs are removed. Our findings suggest a possible route through which information warfare may shape the text used to train AI models, which may in turn spread geopolitical biases. 

---
# E-AVI: Evidence-Grounded Multimodal Assessment for Automated Video Interviews 

**Authors**: Haoshen Wang, Dongbo Che, Zeyi Xie, Yuanjie Du, Shicheng Hua, Xingyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.20001)  

**Abstract**: Automated video interview assessment integrates verbal content, acoustic delivery, and visual behavior, yet numerical predictions alone provide limited inspectable support. We present E-AVI, an evidence-grounded framework that extracts timestamped multimodal evidence and integrates dimension-conditioned evidence attention with source-level embeddings for scoring. A shared evidence pool further supports natural-language feedback and follow-up question answering. On RecruitView and a private hospitality dataset, E-AVI consistently outperforms fine-tuned multimodal baselines in rank correlation. Ablation, evidence-deletion, bootstrap, human-audit, and QA analyses characterize the predictive contribution, grounding, and practical utility of the evidence pathway. Together, these results demonstrate that our proposed E-AVI framework improves predictive performance while providing inspectable support for assessment, feedback, and interactive analysis. 

---
# Customizable and Jointly Optimized Route Planning: A Deep Architecture Enabling Differentiable Shortest-Path Search 

**Authors**: Rui Zhao, Chao Chen, Longfei Xu, Chenguang Ji, Hengbin Cui, Kaikui Liu, Xiaolong Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.19996)  

**Abstract**: With the widespread use of online navigation and ride-hailing services, achieving optimal route planning for diverse user preferences has recently attracted increasing attention. Classic graph algorithms for pathfinding use heuristic cost functions to define edge weight, thus providing no optimality guarantee of route quality. Prior data-driven approaches equating ground truth of the optimal route with user trajectory, which is however moderately influenced by the navigation service, suffers from the feedback loop problem. To address these issues, we propose a deep architecture that is able to jointly optimize cost functions and route-ranking model towards any route preference. First, we run a multi-objective Dijkstra algorithm offline to collect the set of Pareto optimal routes, deeming it as the complete candidate set. Exploiting the property of such a set, we design a neural network structure that emulates shortest-path search and route ranking in an end-to-end differentiable manner. Second, we define route preference as a task of constrained optimization of route attributes, and propose a novel loss function that optimizes a single-objective variable, with other variables strictly under constraints. We conduct extensive experiments on real-world datasets. The results show that our architecture significantly outperforms state-of-the-art methods in route quality and customizability. 

---
# Neuro-Symbolic Agentic AI for Networked Low-Altitude UAVs 

**Authors**: Yuqi Ping, Tianhao Liang, Nanchi Su, Guangyu Lei, Junwei Wu, Qinyu Zhang, Tingting Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.19961)  

**Abstract**: Networked low-altitude unmanned aerial vehicles (UAVs) need reliable and adaptive decision-making capabilities to operate under uncertain observations, dynamic environments, and intermittent connectivity, while many existing agentic systems remain limited by hallucination risks, data dependence, and weak generalization. This article investigates neuro-symbolic agentic AI (NSAAI) as a framework for combining neural grounding, symbolic reasoning, and closed-loop agentic interaction to support more reliable and adaptive UAV autonomy. We first examine its capability foundations in data efficiency, compositional generalization, continual learning, and zero-shot transfer, and then develop a reference architecture integrating task and goal management, neuro-symbolic planning, verification and metacognition, skill execution and network interaction, and shared knowledge and memory. An urban fire-inspection case implemented in LAESim illustrates how a UAV can coordinate sensing and cloud access under intermittent connectivity, reuse a verified image-delivery skill, and satisfy explicit evidence conditions before completing the mission. The results illustrate the potential of NSAAI to support reusable skills, evidence-grounded decision-making, and adaptive mission execution in networked UAV systems. We further discuss key research directions in uncertainty-aware reasoning, knowledge and skill expansion, adaptive self-monitoring, and standardized evaluation. 

---
# Not All AI Agents Are Equal: Characterizing Resource and Performance Dynamics 

**Authors**: Wonmi Choi, Minuk Park, Zhixiong Niu, Yongqiang Xiong, Chuck Yoo, Gyeongsik Yang  

**Link**: [PDF](https://arxiv.org/pdf/2609.19947)  

**Abstract**: LLM-based AI agents process user requests through iterative reasoning and tool execution, often involving the invocation of remote LLM APIs with local tool containers. This execution model can make the optimization of agent serving difficult because latency, local resource demand, and container bottlenecks inter-mix across requests. However, the current agent ecosystem runs without much consideration of resource dynamics, which results in significant waste of the precious resources. This paper analyzes the resource inter-mix of AI agents for three representative tasks: retrieval-augmented question answering, web search, and software coding. To this end, we characterize the latency with respect to the resource dynamics of processing multiple requests and tasks concurrently. Our measurements show that agents have a wide range of behaviors depending on tasks, so that even the same tool can differ substantially in resource dynamics. We also find that running multiple requests concurrently exposes task-dependent bottlenecks in resource dynamics such as CPU, disk I/O, and memory. Furthermore, we uncover that faster LLM responses or more CPU cores do not always accelerate agents. Based on these observations, we demonstrate new optimization opportunities that exploit the resource dynamics of tasks: CPU-aware tool admission and task-aware CPU allocation. Our results show that the latency of CPU-sensitive agent tasks improves $\sim$5.4$\times$, and the average latency across multiple tasks is reduced $\sim$32% compared to native agents. 

---
# MaSCoD: A Multi-Agent Framework for Structural-Context-Guided Candidate Causal Graph Generation 

**Authors**: Yudai Nakada, Yuichiro Nishiura, Jin Michael Splichal  

**Link**: [PDF](https://arxiv.org/pdf/2609.19944)  

**Abstract**: Large language models (LLMs) have been applied to causal discovery, but candidate-graph generation rarely treats premature omission of potentially relevant causal relations as an explicit design objective. We propose MaSCoD, a multi-agent framework that organizes candidate third variables and local structural patterns before direct-edge judgment. We evaluate MaSCoD on Auto-MPG, DWD, and Sachs using GPT-5.4 as the primary backbone and GPT-4o for replication. MaSCoD exhibits a dataset- and backbone-dependent retention-selectivity profile rather than uniform superiority. Across all six dataset-backbone settings, Full, which supplies structural hypotheses before direct-edge judgment, achieved higher mean Recall and F1 than No Phase 1, which instead constructs them within the judgment procedure, while also increasing false-positive rates. Additional reference-edge retention over all evaluated baselines was observed on DWD with GPT-5.4 and on Sachs with GPT-4o, rather than uniformly across settings. Partial ablations showed that supplying both information components did not always outperform supplying only one. For GPT-5.4, stage-wise analysis showed that the Full-No Phase 1 retention gap was already present after direct-edge judgment, while reconciliation introduced additional reference-edge loss for Full on Sachs. These findings support structural pre-organization as an explicit design and evaluation target for omission control and motivate evaluating context construction jointly with its utilization in judgment. 

---
# Beyond Depth Truncation: Controlled Evaluation of Depth Utilization in Recursive Language Models 

**Authors**: Ha Van Dau, Thanh Tung Khuat, Nguyen Thanh Dung  

**Link**: [PDF](https://arxiv.org/pdf/2609.19934)  

**Abstract**: Depth-recurrent language models iteratively apply a small layer stack, decoupling per-token compute from distinct parameter count. To determine whether such a model genuinely utilizes its depth, both recurrence and layer-pruning literatures rely on a shared evaluation: truncating depth at inference time, plotting quality against retained depth fraction, and reading off the slope. While cheap and training-free, this metric suffers from an unexamined flaw: it extracts a single scalar from an intervention that alters multiple model properties simultaneously. Depth truncation concurrently reduces the number of block applications, decreases the volume of distinct computation performed, and pushes the readout head onto an out-of-distribution residual stream. The observed slope conflates all three factors, yet is conventionally interpreted as reflecting solely the second.
We propose the Depth Control Protocol (DCP), a diagnostic suite that disentangles these three quantities. DCP comprises three positive controls that isolate each factor while varying the others, a negative control applying the identical interventions to dense transformers to ensure the effect is not an artifact of the measurement protocol, and a controlled training intervention to verify causality. The linchpin control, running the full budget of block applications while executing only a single distinct iteration, is strictly realizable only in depth-wise weight-sharing architectures, since in a dense network repeating a layer yields an entirely different model rather than the same model in an alternative configuration. 

---
# From "Who Is This User?" to "What Does This Purchase Mean?": A Deployed Pipeline for Semantic User Profiling at Bank Scale 

**Authors**: Ryota Mitsuhashi, Tetsuro Morimura, Hirotake Ito  

**Link**: [PDF](https://arxiv.org/pdf/2609.19928)  

**Abstract**: Per-user LLM inference on transaction histories binds the inference budget linearly to user count, which becomes prohibitive at applied scale. We re-cast attribute inference from per-user to per-transaction-pattern. The pipeline runs in three phases: Resolve abstracts item names with optional web grounding, Profile infers attributes for each frequent pattern, and Tag clusters free-text attributes into a queryable database. In Profile, a single LLM call per pattern emits predefined categorical labels, free-text attributes, and per-attribute prevalence estimates. Because inference runs over patterns rather than users, the budget grows with the pattern count rather than the user count. On the public Open e-commerce corpus, the database is statistically indistinguishable from an LLM that reads each user's raw history directly in AUC across the evaluated attributes, and the prevalence estimates carry discriminative signal between positive and negative users. The pipeline is deployed at a major Japanese bank profiling on the order of tens of millions of users, with close to a three-order-of-magnitude reduction in LLM inference targets versus a per-user pipeline. The code is publicly available on this https URL. 

---
# TRACE: Accountable Agentic Retrieval for Source Discovery in Digital Archives 

**Authors**: Donghan Bian, Marie Puren, Florian Cafiero  

**Link**: [PDF](https://arxiv.org/pdf/2609.19897)  

**Abstract**: Historical archives pose a difficult retrieval problem for retrievalaugmented generation systems: documents are OCR-degraded, heterogeneous across genres and sources, and require strong source traceability for scholarly and institutional use. We introduce TRACE, a training-free agentic retrieval framework designed for accountable source discovery over historical corpora. The system was developed in the context of DECIDON, an interdisciplinary project on the circulation of political discourse between parliamentary debates and the press during the French Third Republic, involving digitised historical collections and institutional use cases. The prototype is currently deployed internally within the project and accessible to 24 researchers across six partner institutions. We evaluate TRACE on HistoriQA-ThirdRepublic, a benchmark of 1,752 French historical questions over parliamentary debates and newspapers from 1887, with documents derived from Biblioth{è}que nationale de France digitised collections. TRACE achieves R@10 = 0.856 and MRR = 0.653, outperforming sparse, dense, graph-based, and agentic RAG baselines, with the largest gains on multi-hop and cross-corpus questions. At approximately $0.02 per question under the default hosted inference configuration, TRACE also remains economically feasible for heritage institutions, laboratories or companies that cannot rely on costly local GPU infrastructure. These results suggest that, for large digital libraries and archives, retrieval accountability and corpus-aware agent design can provide a practical alternative to heavier training-based or graph-construction approaches. 

---
# Physical knowledge on historical data matters more than enforcing physical constraints on the forecast 

**Authors**: Etienne Lehembre, Pascal Audigane, Vincent Nguyen, Christel Vrain, Thi-Bich-Hanh Dao  

**Link**: [PDF](https://arxiv.org/pdf/2609.19871)  

**Abstract**: Time series forecasting has seen signicant advancements with the emergence of new deep learning models. However, forecasting time series in applications involving physical processes remains a major challenge. Despite the apparition of Physics Informed Neural Networks (PINN), recent models do not estimate unobservable intermediate physical variables, which are important for domain experts to understand the target behavior. To this end, we propose a Physics Informed Recurrent Neural Network (PIRNN) which predicts, along the target, unobservable variables on both historic data and forecast target. This approach enhances the model robustness and results interpretation using domain knowledge. Our method is easily adaptable to any physical model using several equations, each having its own set of unobservable variables, to describe it-self. As a case study, we incorporate physical equations used for groundwater levels predictions by the physical model called Gardenia. This model uses transfers equations between reservoirs, optimized with data assimilation, to simulate the evolution of groundwater levels. Evaluation includes several well known neural network models and the Gardenia model compared on twelve real world datasets. In addition, we study the impact of each component through an ablation study. Our model outperforms other models on ve out of the twelve datasets and our ablation study underlines the importance of having a physical background in our time series forecasting task. Finally, the coherence of the physical variables predicted by our neural network is assessed by a domain expert. 

---
# Reproducibility is not construct validity: LLM measurement of institutionally situated communication 

**Authors**: Veronika Batzdorfer, Carlo Romano Marcello Alessandro Santagiustina  

**Link**: [PDF](https://arxiv.org/pdf/2609.19866)  

**Abstract**: High annotation reproducibility does not necessarily imply that an LLM-inferred measure captures the construct it is intended to measure. We test this distinction using a dataset from the European Commission's AI Act consultation, linking structured survey responses to free-text consultation submissions from the same stakeholders. LLM annotations of consultation submissions are highly reproducible (intraclass correlations > 0.99), yet show limited convergence with survey-reported measures of the nominal construct they were intended to approximate. Divergence between survey-and LLM-inferred text-based measures varies systematically across stakeholder groups: business associations express greater concern about AI risks in text-based consultations than in survey responses ({g} = +1.0), whereas public authorities and several nonbusiness groups show smaller or negative divergences. Divergences between scores suggest positive spatial autocorrelation across European countries (Moran's I = 0.347, p = 0.036), indicating that stakeholders from neighboring countries tend toward more similar text-based stances towards AI safety concerns. Despite divergence, survey-reported concerns remain strongly associated with support for explainability across all divergence levels. These results demonstrate that LLM annotation reproducibility can coexist with poor construct correspondence and motivate validation procedures that distinguish reproducibility, construct validity, and communication context variation when LLMs are used as measurement instruments. 

---
# Constraint-Safe Graph-Context Scoring for Stable Point-Feature Labels Under Text-Width and Accessibility-Inspired Profiles 

**Authors**: Taimoor Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2609.19848)  

**Abstract**: Point-feature label placement on interactive maps must reconcile geometric validity, display yield, local placement utility, and stability across camera motion. Accessibility and multilingual requirements further change label dimensions, yet algorithmic evaluations often collapse these concerns into overlap counts. We present LABELSENSE-Pilot, a reproducible prototype that generates eight compass candidates per feature, scores candidates with a multilayer perceptron over graph-context summaries, adds a previous-placement bonus, and selects a layout through mixed-integer optimization. The executed scorer is deliberately not described as a graph transformer. Every returned layout is checked for viewport containment, per-feature uniqueness, and pairwise clearance. Experiments use 2,500 airport coordinates and names spanning 155 countries, with country-grouped splits and generated density, camera, text-suffix, preference, and enlarged-font stressors. Across five seeds, LABELSENSE-Pilot displayed 85.62 percent of labels with 2.09 percent flicker and zero collisions. Versus a handcrafted-utility integer program, LABELSENSE-Pilot sacrificed 1.43 percentage points of display while reducing flicker by 12.04 points. Enlarged-box-aware layouts produced zero proxy violations, whereas standard geometry reevaluated at 1.5x violated 52.57 percent of selected placements. These results establish an auditable engineering trade-off, not human accessibility, multilingual usability, or preference. Official recent baselines and participant evidence remain required before submission. 

---
# A Dual-Process Perspective on Nudge Susceptibility in LLM-Based GUI Agents 

**Authors**: Haya Halimeh, Sascha Kaltenpoth, Kevin Bösch, Oliver Müller  

**Link**: [PDF](https://arxiv.org/pdf/2609.19843)  

**Abstract**: LLM-based GUI agents increasingly act on behalf of users in digital environments that were designed with human users in mind. These graphical user interfaces were designed to support, but also deliberately steer, the behaviour and decisions of users. While behavioural biases in the textual outputs of LLMs are well-documented, far less is known about how such influence operates when models act as agents that perceive interfaces and execute decisions---and, in particular, whether the reasoning capabilities increasingly built into these agents make them more robust to it. Drawing on Dual-Process Theory, we empirically investigate whether LLM-based GUI agents are susceptible to automatic (Type 1) and reflective (Type 2) digital nudges, and how their reasoning configuration moderates this susceptibility. In a randomized online shopping experiment with 3,600 agents and a total of 21,600 simulations across six frontier models from three providers, we found that agents were vulnerable to both nudge types. Crucially, the reasoning configuration moderated these effects in opposing directions, reducing susceptibility to automatic default nudges while heightening it to reflective social influence nudges. Extensive reasoning therefore did not make agents more robust but redirected the route through which choice architecture takes effect. Exploratory analysis further showed this redirection to be systematically structured by model scale. Beyond establishing nudge susceptibility as a behavioural property of agentic AI, the study positions interface design as a governance concern for organizations that delegate decisions to autonomous agents. 

---
# MetaRTL: Meta-path Attention Enhanced Relational Table Learning 

**Authors**: Ken Zhong, Weichen Li, Zheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.19832)  

**Abstract**: Relational table learning has gained increasing attention with the widespread use of relational databases. Existing methods typically rely on deep GNN or HGNN stacks, leading to high computational costs and limited performance on large real-world databases. We propose MetaRTL, a two-stage framework for scalable and expressive relational table learning. In the first stage, MetaRTL obtains initial table embeddings via lightweight pre-training. In the second stage, it performs non-parametric message passing to derive meta-path features, which are then aggregated by an attention module, MetaAttn. By shifting computation from deep message passing to efficient meta-path aggregation, MetaRTL captures rich relational semantics while maintaining high efficiency. Experiments on 10 real-world datasets across 24 tasks demonstrate the effectiveness of the proposed method. 

---
# Dual-Axis Policy Optimization for LLM Agents: Bayesian Feedback Attribution and Trajectory Mass Normalization 

**Authors**: Yingxuan Zhuang, Binhe Yu, Jingxiao Yang, Ruopei Sun, Ziting Li, Cheng Tan, Xuhong Zhang, Jianwei Yin, Jintao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.19830)  

**Abstract**: Reinforcement learning for LLM agents involves two distinct optimization di- mensions: how environment feedback is exploited within a trajectory, and how complete trajectories are aggregated across a batch. We formulate these dimen- sions as Intra-Trajectory Feedback Attribution and Inter-Trajectory Objec- tive Aggregation, and introduce BATON (Bayesian Attribution and Trajectory Objective Normalization), a dual-axis policy optimization framework. BATON instantiates the first axis with Bayesian Feedback Attribution, which constructs a feedback-conditioned posterior over sampled actions, and the second with Trajec- tory Mass Normalization (TMN), which assigns equal optimization mass to com- plete trajectories. Experiments with GRPO and GiGPO on ALFWorld, WebShop, and SearchQA show that both axes provide independent gains and that their combi- nation consistently achieves the strongest overall performance across model scales. 

---
# Steering Equilibrium Selection in Regularized Self-Play via the Reference Policy 

**Authors**: Luis Leal  

**Link**: [PDF](https://arxiv.org/pdf/2609.19820)  

**Abstract**: Regularized self-play -- the family behind DeepNash's Stratego play -- drives a two-player zero-sum policy to a Nash equilibrium by best-responding to a slowly moving, entropy-regularized reference policy $\rho$. When the game has a polytope of value-equivalent equilibria, the regularizer silently breaks the tie: with a uniform reference it selects the maximum-entropy member, the I-projection of $\rho$ onto the Nash set. Can the reference be used to choose the equilibrium on purpose? On five exactly solvable games plus a 2-D polytope, with exact best responses and equivalence tests over independent seeds, anchoring the reference at a target member and refining steers self-play to that member with mean coordinate error 0.007 at median exploitability $5\times10^{-5}$, TOST-equivalent to the request within $\pm0.05$; the anchoring persists through refinement and follows the reference, not the initialization. Selection follows the reach-weighted I-projection (slope 0.969 [0.950, 0.987]). We report with equal emphasis where the story breaks: fixed off-manifold references cost 0.08-0.25 exploitability; stiff or flat families require a smaller mirror step, set by a pre-registered rule; boundary targets undershoot; curvature predicts where boundary saturation bites (rank correlation 0.90, p=0.037) while interior precision is curvature-independent. Table and MLP steering maps are equivalent within $\pm0.03$ at every target (30 seeds); matched control arms show attention's robust signature is excess seed variance, any systematic shift bounded at 0.018 and not significant. Against a best response the selection-robustness trade-off is degenerate: steering matters only against fixed, non-equilibrium opponents. The recipe -- anchor the reference at the desired member and refine -- reinterprets the KL anchor of RLHF-style RL as a selection knob, not only a stability leash. 

---
# Contagion on the Trading Floor: How Adversarial Signals Spread in Multi-Agent Trading Systems 

**Authors**: Qi Rong Sua, Junhao Dong, Nguyen Duc Thai, Yuqing Wen, Cheston Tan, Yew-Soon Ong  

**Link**: [PDF](https://arxiv.org/pdf/2609.19789)  

**Abstract**: Multi-agent trading systems built on large language models (LLMs) are beginning to appear in quantitative finance, yet their robustness to adversarial inputs is largely unknown. We study the vulnerability of LLM trading stacks to black-box, input-only attacks that enter solely via admissible social-media feeds. We introduce the Generic Multi-Agent Trading System (GMATS), a framework that captures modern multiagent trading architectures and instantiate a class of black-box poisoning attackers that treat an LLM as a post generator and inject budget-constrained, plausibly benign social-media content into the analyst's evidence stream. We define contagion metrics that trace how adversarial content propagates through the stack, including belief-shift scores at analyst and coordinator layers and attack-clean deltas on standard backtest metrics. Experiments on a safe offline benchmark with historical market and social data show that even simple input-only attackers can materially degrade risk-return profiles, sharply reducing Sharpe ratios. At the same time, we find that suitably designed multi-agent topologies and coordinator prompts can dampen adversarial shocks and improve average robustness under identical poisoning budgets. 

---
# Integrating knowledge from case reports: a medical ontology based multimodal information system with structured summary 

**Authors**: Shuyu Guo, Lan Huang, Yichen Liu, Hanbin Ma, Tian Bai  

**Link**: [PDF](https://arxiv.org/pdf/2609.19775)  

**Abstract**: Published medical case reports serve as a crucial medical information carrier, documenting discoveries in rare diseases, diagnostic methods, and innovative treatments. Despite the wealth of clinical knowledge in millions of case reports in the public medicine literature database (PubMed), accessing relevant information efficiently is hindered by the limitations of traditional keyword-based retrieval tools on unstructured and diverse case reports. To address the above issues, we introduce a comprehensive multimodal information system for case reports integrating structured clinical summaries of patients including medical images and biomedical named entities from 52949 open-access case reports published from 2000 to 2021. The multimodal essential information is organized in a well-structured medical ontology. Also, a powerful interface for searching and browsing case reports is designed to assist junior clinicians in retrieving cases effectively and improving the identification and diagnosis of rare diseases. 

---
# TorchCraft: Unified binder design by inverting an all-atom structure predictor 

**Authors**: TorchCraft Team, Yu Liu, Zhouhanyu Shen, Zhengyi Li, Xikun Huang, Jiaqi Liu, Shuxian Gao, Qilin Yu, Xiayan Qin, Yucheng Zhang, Mingchen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.19770)  

**Abstract**: All-atom structure predictors model diverse molecular interactions, but using their learned structural priors for binder design remains challenging. Here we present TorchCraft, a unified binder-design framework that optimizes sequence logits through a frozen all-atom predictor. Implemented in TorchFold, TorchCraft combines confidence, contact, geometric, and sequence-prior objectives within a shared optimization procedure for minibinders, framework-conditioned VHHs, cyclic peptides, and ligand-binding proteins. Using pretrained AlphaFold 3 weights, TorchCraft generated representative minibinders and VHHs with experimentally measured binding across four targets in each format, without post hoc sequence redesign. Computational benchmarks further demonstrated the framework's applicability to cyclic peptides and ligand-conditioned pocket design. TorchCraft extends predictor inversion to multiple binder formats and molecular contexts, providing a common framework for reusing all-atom structural priors in design. 

---
# Rethinking Multi-Agent Collaboration: When More Is Less 

**Authors**: Yishuo Yuan, Yibo Wu, Yihan Zhang, Minyuan Sun, Shenliang Li, Xinkai Ma, Yifan Li, Jiaheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.19759)  

**Abstract**: The rapid advancement of large language models and single-agent harnesses has reshaped the landscape of autonomous systems, raising a critical question of when multi-agent collaboration offers genuine value. As individual agent capabilities continue to scale, multi-agent collaboration faces diminishing returns while incurring growing context overhead. Through systematic analysis, we delineate the capability boundaries of multi-agent collaboration relative to single-agent alternatives, showing that it confers systematic benefits specifically in long-horizon tasks with sparse dependencies, while single-agent harnesses remain superior in tightly coupled, sequential workflows. Building on these insights, we propose SAIGE, a lightweight multi-agent collaboration mechanism based on Semantic-Aware Incremental Graph Evolution. SAIGE models collaboration as a dynamically evolving graph, where nodes are agent instances spawned on demand and edges encode semantic dependencies established through content-based information retrieval. Experiments on long-horizon, complex task benchmarks show that SAIGE achieves a favorable trade-off between context efficiency and task performance, and that scaling the agent pool or deepening the recursion level does not consistently improve outcomes. Our findings suggest that multi-agent superiority is bounded by task structure rather than universal, and that more agents do not necessarily make a system more intelligent. 

---
# AutoData: Agentic Search for Pre-training Data Selection 

**Authors**: Yan Meng, Dhruv Srikanth, Bingchen Zhao, Zhengyao Jiang, Yuxiang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.19754)  

**Abstract**: LLM agents have recently shown promise in automating machine learning engineering by editing model and training code under execution feedback. Data, however, remains largely outside this agentic optimisation loop. We frame pre-training data selection as heuristic engineering over per-document features, i.e., lexical statistics, categorical labels, and perplexity. We introduce AutoData, an agent that searches directly over executable selection algorithms. Unlike prior data mixture methods that optimise weights over a fixed set of domains, AutoData searches a richer program space of scoring, stratification, and stochastic selection rules, discovering feature interactions automatically by iteratively refining algorithms with validation feedback from a proxy model. Within an overnight search, AutoData discovers a selection algorithm that outperforms existing human-designed curation pipelines. Despite being searched only on this small proxy, the discovered recipe transfers to larger scales and improves the downstream metric CORE. These results suggest that data engineering can be treated as an agentic machine learning problem, extending autonomous research from model and training-code optimization to the data. 

---
# LearnActCoder: Role-Aware Error Memory for Adaptive Clinical Coding Agents 

**Authors**: Meysam Ghaffari, Bhaskar Sen, Nasim Sabetpour, Nina Fatehi, Animesh Agarwal, Carlos Morato  

**Link**: [PDF](https://arxiv.org/pdf/2609.19721)  

**Abstract**: Clinical coding agents repeatedly encounter the same failure modes, including unsupported codes, missed documented conditions, specificity errors, and procedure-coding convention mismatches. We introduce Learn-Then-Act, an inference-time adaptation framework that converts errors from a small labeled LEARN batch into a structured Mistake Knowledge Database (MistakeKDB). False-negative lessons are routed to a recall-oriented Coder, while false-positive lessons are routed to a precision-oriented Judge. We instantiate the framework in LearnActCoder, a Coder-Judge clinical coding pipeline with lookup-table grounding where available. On 150 matched MIMIC-III notes, structured MistakeKDB improves CPT F1 by 5.9 percentage points, while raw-example and reflection-style memories remain near the no-memory baseline; the ICD-9 improvement is not significant. On a matched MIMIC-IV cohort, memory shifts ICD-10 coding toward higher precision at a recall cost, leaving F1 statistically unchanged. Applying the same memory to 1,000 held-out MIMIC-III notes maintains a stable ICD operating point, providing scale/stability evidence. Overall, the results are consistent with structured, feedback-derived error memory being useful for adapting clinical coding behavior across cases without weight updates or changes to the underlying workflow. Absolute CPT/HCPCS performance remains low, and the system is evaluated retrospectively rather than in clinical deployment. 

---
# FINSKILLOPS: A Self-Evolving Multi-Agent System for SEC Filing QA 

**Authors**: Yanzhang Ma, Zhenghan Tai, Hanwei Wu, Sizhe Guan, Jianliang Lei, Hailin He, Chaolong Jiang, Jijun Chi, Tung Sum Thomas Kwok, Bohuai Xiao, Jingrui Tian, Xinlu Wu, Xingao Zhan, Peng Lu, Muzhi Li, Yihong Wu, Liheng Ma, Sicheng Lyu, Tianshuo Yan, Junhao Zhu, Yaqian Xu, Lei Ding, Yufei Cui, Ziquan Liu, Boyu Han, Hengli Liu, Ling Zhou, Xinyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.19680)  

**Abstract**: Financial QA systems are typically improved before deployment through better retrieval, prompting, or agent coordination, leaving their reliability behavior fixed thereafter. In practice, new SEC-filing questions repeatedly expose heterogeneous errors in period, entity, evidence use, and calculation. Existing self-improvement methods can turn failures into new behaviors, but offer limited control over where a correction should apply or which previously correct answers it may break. We therefore frame post-deployment improvement as controlled behavioral maintenance: recurring failures should become scoped skill patches, and each patch should earn deployment with- out introducing regressions. We instantiate this view in FINSKILLOPS, a multi-agent system for SEC filing QA. FINSKILLOPS derives reusable skills from evidence-grounded, typed failure diagnoses and governs them through targeted validation, protected-case regression checks, negative controls, and versioned replacement or retirement. Across six financial QA benchmarks, a single frozen skill registry achieves the highest verdict-weighted correctness and reference consistency among the evaluated systems. Evolved skills raise correctness from 3.70 to 4.55 on our enhanced benchmark. In a separate 12-round operational study, only six of 33 proposed skills are promoted, while the monitoring non-correct rate falls from 20.0% to 12.5%. These results establish controlled skill scope, admission, and lifecycle management as the foundation for reliable self-improvement. 

---
# When2Think: Learning Difficulty-Aware Length Control for Efficient Hybrid Reasoning Models 

**Authors**: Jaejun Shim, HyunJin Kim, Young Jin Kim, JinYeong Bak  

**Link**: [PDF](https://arxiv.org/pdf/2609.19671)  

**Abstract**: Large Reasoning Models (LRMs) achieve strong performance on complex tasks but exhibit systematic inefficiency: they often overthink easy problems and underthink hard ones. Existing approaches based on uniform length penalties or rigid routing incur an efficiency tax, trading reduced computation on easy instances for accuracy loss on hard instances. We formulate efficient reasoning as an instance-adaptive computation allocation problem and propose When2Think, a post-training framework for hybrid reasoning that dynamically allocates computation based on problem difficulty. Our method introduces Instance-level Difficulty-Aware Control (IDAC), a reward-shaping mechanism that leverages pre-computed reference statistics (accuracy and token usage) to regulate reasoning depth. Combined with verifier-based rewards and batch-wise standardized advantages, IDAC enables stable critic-free optimization without learned reward models or online reference-model queries. When2Think encourages direct answering on easy instances while preserving extended reasoning on hard instances, thereby learning when to use System 1 (NoThink) versus System 2 (Think). Experiments on mathematical benchmarks demonstrate improved accuracy-efficiency trade-offs: on AIME24, Pass@3 increases by 10.0% while token usage is reduced by 27.9% relative to the base model, and on AIME25, When2Think achieves 40.0% Pass@3, outperforming compression and routing-only baselines. 

---
# Replan, Repair, or Edit? A Unified Empirical Evaluation of Travel Agents for Itinerary Revision under Resource Disruptions 

**Authors**: Xiaofei Yuan, Yan Zhang, Shaobo Qiao, Huangleshuai He, Leyan Ni, Mingchen Ju, Lujia Yang, Sijia Xu, Yifu Tang, Zhengyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2609.19654)  

**Abstract**: Travel-planning agents generate itineraries that may become infeasible after acceptance because of flight cancellations, hotel unavailability, or attraction closures. Revising these itineraries involves full replanning, classical plan repair, and LLM-based travel-agent revision, whose differing task formulations and evaluation protocols hinder comparison. We conduct a systematic empirical study using two TREK-derived benchmark sets: 500 single-disruption cases, including feasible and infeasible instances, and 200 feasible simultaneous compound-disruption cases. We compare LLM-Z3 full replanning, IPyHOPPER hierarchical repair, and an iTIMO local-revision adapter across effectiveness, plan stability, and computational cost. LLM-Z3 with Gemini achieved the highest observed compound-disruption success. IPyHOPPER nearly matched that configuration's single-disruption overall success, while preserving substantially more of the accepted itinerary on successful repairs. Successful hierarchical and local repairs made fewer edits and retained more accepted commitments than full replanning. Computational profiles differed: IPyHOPPER used no LLM inference, the evaluated LLM-Z3 adapter used compact one-call inference, and the iTIMO adapter consumed substantially more tokens. The study provides practical guidelines for balancing feasibility recovery, commitment preservation, and computational cost within evaluated settings. 

---
# ScientistTwo: Pioneering the Human Knowledge Frontier with Autonomous AI 

**Authors**: Jaehyun Nam, Jinsung Yoon, Yanzhou Pan, Yubo Wang, Rui Meng, Parthasarathy Ranganathan, Tomas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2609.19644)  

**Abstract**: Scientific discovery is defined by the ability to identify the boundaries of existing knowledge and venture into unexplored territory. The ultimate vision for AI in science is problem-driven autonomous research: given a fundamental challenge by a human expert, the AI independently navigates the scientific landscape, uncovers theoretical and empirical bottlenecks, and systematically expands the frontier of knowledge. In this paper, we introduce ScientistTwo, a fully autonomous multi-agent framework designed to realize this vision. Specifically, ScientistTwo takes an initial problem as input, establishes state-of-the-art baselines, formulates novel hypotheses, and coordinates specialized agents to orchestrate an end-to-end discovery cycle without human intervention. Moreover, the framework rigorously conducts experiments using diverse datasets and metrics, refines methodologies through automated ablation studies, and validates research findings via a closed-loop simulated peer-review rebuttal engine. To evaluate ScientistTwo's capabilities against the highest standards of human scientific achievement, we benchmark it across papers accepted at top-tier conferences such as ICLR, ICML, and NeurIPS. As a result, ScientistTwo autonomously generates expert-level, publishable papers and fully verified, executable codebases. Its solutions consistently outperform human state-of-the-art models, and achieve higher average review ratings than human-authored papers under automated AI review agents. These results show that ScientistTwo is not merely an assistive tool but an autonomous scientific pioneer capable of pushing the frontiers of human discovery. Project website: this https URL 

---
# Reach or Solve? Attributing Agentic RL Gains with Checkpoint Handoffs 

**Authors**: Xuan Liu, Jingbin Qian  

**Link**: [PDF](https://arxiv.org/pdf/2609.19636)  

**Abstract**: Reinforcement learning now trains language-model agents that act over dozens of steps in live environments. The gains are large, and they are read as better decision-making. An agent in a closed loop writes its own inputs. Each observation follows from its own earlier actions, so the states it meets late in an episode are partly of its own making. An SFT checkpoint and an RL checkpoint are then scored from different states, even on identical tasks. Endpoint success mixes two changes: where the agent arrives, and what it does once it is there. Restricting the comparison to states both policies reach does not separate them. That restriction selects on an outcome, and in our data it flips the sign of the effect. We introduce checkpoint handoff, an evaluation protocol that clones a state one released checkpoint reached and hands it to another, with no retraining. Crossing a reacher role and a solver role over SFT and RL splits an endpoint gain into REACH and SOLVE. REACH is how often a policy arrives at a state the environment confirms is a fixed number of actions from success. SOLVE is how often it finishes from an identical cloned state. Across two benchmarks and two independently released pipelines, the reacher by solver interaction is positive in all five conditions. An RL history is worth more to an RL solver than the same history is to an SFT solver. On ALFWorld, RL improves both terms, and the SFT solver never succeeds where the RL solver fails. Independent REACH and SOLVE gaps predict the aggregate interaction. Handoff asks only that one checkpoint's history can be replayed under another, so long-horizon evaluation can report arrival and completion beside endpoint success. 

---
# From Intent to Action: Benchmarking LLM Safety in Vehicle Voice Command Authorization 

**Authors**: Diba Afroze, Xingli Zhang, Yazhou Tu, Xiali Hei  

**Link**: [PDF](https://arxiv.org/pdf/2609.19630)  

**Abstract**: Large language models (LLMs) are increasingly integrated into vehicle voice assistants. But linking natural-language requests to vehicle functions creates a safety-critical authorization problem. Before executing a command, the system must choose whether to execute, refuse, clarify, require confirmation, defer to manual control, trigger an emergency response, or make no tool call. To our knowledge, prior evaluations do not isolate this pre-action decision across speaker role, authentication status, vehicle state, and tool availability. We introduce a 202-scenario benchmark with Reference Decisions under a seven-class taxonomy. We evaluate two local open-weight models and three API-based LLMs using Decision Alignment and safety-specific error metrics. Alignment ranges from 40.1% for Llama 3.2 3B to 89.1% for Gemini 3.1 Pro Preview. The API-based models score between 83.2% and 89.1%, with no statistically significant differences among them. Even these models produce two to three False Executes among 161 non-execution scenarios, and persistent errors remain in confirmation and manual-control decisions. A controlled Llama 3.2 3B ablation increases alignment to 40.1% under the structured authorization policy, versus 28.2-29.2% under schema-only and generic-safety baselines, but it does not eliminate False Executes. Structured LLM decisions are therefore insufficient as a standalone safety mechanism, and deployment requires an independent enforcement layer that verifies tool permissions and vehicle-state constraints before invoking any vehicle function. 

---
# SIMLIFE: Pattern Understanding for Long-Horizon Human-Agent Partnership 

**Authors**: Run Peng, Zinnia Nie, Jing Ding, Yinpei Dai, Yichi Zhang, Zengqing Wu, Yao Fu, Ziqiao Ma, Jiayuan Mao, Joyce Chai  

**Link**: [PDF](https://arxiv.org/pdf/2609.19610)  

**Abstract**: Understanding humans over long horizons requires agents to infer not only what people need in the moment, but also how routines form, why they repeat, and when they change. We introduce SimLife, a scalable platform for simulating long-term household life with rich visual observations, ground-truth action logs, and synthetic dialogues with audio. Built on SimLife, SimLife-BP evaluates long-context pattern understanding: the ability to infer latent behavioral rules from weeks or months of everyday observations. The benchmark contains 106 episodes averaging 15.49 hours and 38.57 in-game days, and 1,439 question-answer pairs. Each task probes direct, counterfactual, noisy, and inverse reasoning under different levels of rule hints. Evaluating frontier models and architectures, we find that current models often achieve surface-level prediction without comprehensive rule understanding, rely on frequency-based heuristics rather than if-then reasoning over evidence, and struggle to adapt when behavioral patterns change. These findings suggest that long-context pattern understanding remains a major bottleneck for future embodied agents, while SimLife opens a broader space for studying memory, personalization, adaptation, and long-horizon planning in everyday human-AI interaction. 

---
# Continual Enterprise World Model Discovery in Dynamic Systems 

**Authors**: Shambhavi Mishra, David Vazquez, Perouz Taslakian, Marco Pedersoli, Jose Dolz, Issam H. Laradji  

**Link**: [PDF](https://arxiv.org/pdf/2609.19551)  

**Abstract**: In an enterprise system, updating one field can set another, create a record, or start an approval. These effects are produced by business rules that are not built into the platform but written by each organization and revised over time. An agent working in such a system cannot predict the result of its own actions without knowing these rules. We study continual enterprise world model discovery, where an agent starts without knowledge of these business rules and discovers them by interacting with records and observing the outcomes. From those observations it builds a world model, which it revises as the rules change. To evaluate this, we introduce EnterpriseWorldShift, built on a live ServiceNow environment with nine tables, 25 hidden rules and 600 evaluation actions. It presents four versions of the same enterprise world, with the tables and records held fixed while a rule is modified, then added, then removed, so that discovery, revision, extension and retirement are each tested in turn. Our Continual Discovery Agent (CDA) builds such a model and carries it from one world to the next. It predicts the effects of the hidden rules more accurately than looking them up for each question, the approach taken by prior work, by up to 8.98 IoU points, and it answers from its own model without querying the running system. 

---
# Agentic AI Networking for Heterogeneous Unmanned Aerial Systems in Low-Altitude Wireless Networks 

**Authors**: Nguyen Duc Minh Quang, Chang Liu, Shuangyang Li, Derrick Wing Kwan Ng  

**Link**: [PDF](https://arxiv.org/pdf/2609.19538)  

**Abstract**: Low-altitude wireless networks (LAWNs) are emerging as a key infrastructure for heterogeneous unmanned aerial systems that support concurrent services within a shared three-dimensional airspace. Their coexistence creates strong coupling among mobility, connectivity, and shared network resources, while heterogeneous services impose distinct and time-varying requirements. These interactions naturally form a dynamic non-cooperative game in which both operating conditions and coordination objectives evolve over time. Conventional optimization and learning-based controllers typically rely on predefined objectives, limiting their ability to adapt autonomously to changing service requirements and resource priorities. To address this challenge, we propose a hierarchical hybrid large language model (LLM)- multi-agent reinforcement learning (MARL) architecture organized as a dual-loop structure. Specifically, an outer adaptation loop employs LLM-assisted game orchestration to interpret service requirements and operator intent, and reconfigure objectives and resource priorities, while an inner loop executes decentralized, parameter-conditioned MARL policies under the configured game. A logistics-monitoring case study illustrates how the proposed framework facilitates coordinated coexistence among heterogeneous services, adapting to evolving operating conditions without retraining the underlying MARL policies. Finally, we discuss key challenges and research directions toward scalable, trustworthy, and adaptive agentic LAWNs. 

---
# When Hiring Becomes Agent-Mediated: Evaluating Access and Recurrence in Two-Agent Résumé Screening 

**Authors**: Jian Gao, Hang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2609.19530)  

**Abstract**: Hiring is bilateral: employers assess fit, while candidates present and defend evidence of their qualifications. Yet résumé screening, the first gate, is commonly automated as a static, one-call judgment over a résumé-job pair. We study a two-agent alternative in which employer-side and candidate-side agents represent these roles, exchange evidence, and update their judgments before deciding who advances. We compare procedures on 600 constructed résumé-job pairs using GPT-5.5 and Claude Opus 4.7. Two-agent screening advances more applications (33.3% to 39.3% for GPT-5.5; 34.0% to 35.5% for Opus 4.7). Across three runs on the common 191-pair borderline pool, pass-instance rates rise from 4.5% to 26.2% and from 6.5% to 16.1%, respectively. This is not a uniform relaxation: two-agent screening rejects applications one-call advances, changing decisions in both directions. At similar pass volumes, the procedures advance different applications, and no one-call threshold recovers applications consistently selected by two-agent screening. Among discovery-selected cases re-executed in fresh runs, two-agent-only selections recur less often than shared selections, clearly under GPT-5.5 and less certainly under Opus 4.7, while a separate one-call follow-up shows no comparable decline. As hiring becomes agent-mediated on both sides, the screening procedure, not only the model behind it, shapes who reaches human review and how reliably that access recurs. 

---
# Self Improvement via Fast Tree-search 

**Authors**: Xinghong Fu, Aravinth Kulanthaivelu, Yutaro Yamada  

**Link**: [PDF](https://arxiv.org/pdf/2609.19526)  

**Abstract**: Coding agents can recursively modify their own implementations, forming a loop of self-improvement. While prior work shows this can boost performance on coding benchmarks, existing approaches are costly and compute-intensive. We introduce a simple, sample-efficient self-improvement framework that significantly improves coding performance under strict budget constraints. We identify evaluation of candidate self-modifications as the main runtime bottleneck since prior approaches estimate their effectiveness by re-running a subset of benchmark tasks with the modified agent, which is time-consuming. We introduce Recursive Self Improvement via Fast Tree-search (SIFT), which augments these downstream task evaluations with an LLM-as-a-judge signal that performs pairwise comparisons between candidate patches, where the win-loss record is aggregated with a regularized Bradley-Terry model, and the resulting strength scores drive rank-based parent sampling inside a lightweight disaggregated tree search. Expensive downstream task evaluations are reserved only for the most promising nodes. Using a fully disaggregated tree search pipeline, the judge scores provide intermediate signal to guide exploration on promising candidate patches without being bottlenecked by slow evaluation runs. SIFT outperforms existing tree-search based self-evolution frameworks on the full Polyglot benchmark with significantly lower resource requirements in terms of CPU hours, wall clock time, and API cost. 

---
# A Unified Evaluation Framework for Trustworthy Large Language Models, Agentic AI, and Multimodal Systems 

**Authors**: Shaina Raza, Ahmed Y. Radwan, Imran Liaquat, Kathryn Hume  

**Link**: [PDF](https://arxiv.org/pdf/2609.19524)  

**Abstract**: Benchmark scores alone provide an incomplete basis for assessing the trustworthiness of modern artificial intelligence systems. Large language models (LLMs), agentic systems, and multimodal models (MLLMs) require different forms of assessment, yet their evaluation evidence must remain interpretable for development and oversight. We propose a unified framework that connects output-level, trajectory-level, and cross-modal assessment through eight trustworthiness dimensions: capability, robustness, safety, fairness, transparency, governance, oversight, and efficiency. The framework preserves system-specific metrics while mapping native measurements to common performance bands, accompanied by uncertainty estimates and traceable evidence. A meta-evaluation layer examines the validity, reliability, and reproducibility of the evaluation itself. Multidimensional profiles expose strengths and weaknesses, while safety-critical overrides prevent aggregate scores from masking critical failures. Mappings to governance frameworks, international standards, and European Union regulatory requirements connect technical assessment with oversight needs. The framework provides a structured basis for assessing both system performance and the credibility of the evidence supporting it, with empirical validation across deployment contexts remaining an essential next step. 

---
# EconSkills: Studying Skill Transfer and Retrieval for Web Agents on Live Economic Data 

**Authors**: Yinzhu Quan, Zefang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.19523)  

**Abstract**: Web agents often revisit the same sites, yet most evaluations discard the procedures learned in earlier successful interactions. We introduce EconSkills, a skill library and evaluation framework that distills verified EconWebArena trajectories into parameterized standard operating procedures for retrieving live economic data. Each skill records its scope, navigation procedure, site-specific guidance, verification checks, and recovery steps while replacing source-instance values with placeholders. EconSkills separates two questions: whether a known relevant procedure transfers to a held-out task, and whether an agent can retain that benefit when selecting from a library. In controlled transfer, matched skills improve success over no-skill prompting and require fewer steps on paired successes, while abstraction is substantially more effective than replaying raw trajectories. At library scale, retrieval is competitive with the no-skill baseline overall and performs best on directly covered tasks; coverage-stratified outcomes show that approximate matches on uncovered tasks offset these gains. Browser trajectories further identify when procedural guidance shortens portal-specific navigation and when semantic verification remains necessary. These results establish that reusable economic web procedures can transfer across task instances and provide a concrete design target for coverage-aware selection and context delivery. 

---
# An Architecture for Long-Horizon Agents: Levels, Ticks and Cascaded Intelligence 

**Authors**: Erik Nijkamp, Anurag Koul, Egor Pakhomov, Bo Pang  

**Link**: [PDF](https://arxiv.org/pdf/2609.19519)  

**Abstract**: Language-model agents are increasingly asked to carry out work spanning days or weeks, such as an operations remediation or a research programme. Such a task outlives any context window, any process and any interval at which a person can attend. In this paper, we argue that a long-horizon agent must run continually without forgetting before it can learn continually. This ability lies in the harness around the model rather than in the model itself. We derive seven bottlenecks from the long-horizon setting and answer them with a hierarchical architecture of three parts: (i) levels indexed by time scale, each keeping a bounded file summarising the level below; (ii) a clocked tick as the unit of autonomous action; and (iii) cascaded intelligence, where work is escalated to a more capable model only after failing review. We report on a ten-day campaign in which an agent built on this architecture reproduced a published reinforcement-learning result with a human attending once a day, and show (1) the agent kept the thread across every context reset and session boundary of the campaign, (2) operating knowledge written early changed later behaviour with no change to model weights, and (3) where learned components would enter such a system. Overall, our experience suggests continual learning for these agents needs a substrate outliving every context and process, and the checks the harness already runs are where a learner belongs. 

---
# LLM-as-an-Improver: Turning Verification into Better Candidates 

**Authors**: Akiyoshi Tomihari, Yuma Ichikawa  

**Link**: [PDF](https://arxiv.org/pdf/2609.19515)  

**Abstract**: Verifier-based selection improves LLM performance by generating multiple candidate solutions and using a verifier to select the most promising one. However, existing methods typically treat verification only as a ranking step and discard its feedback once a fixed candidate pool has been evaluated. In this paper, we ask whether verification can also improve the candidate set itself. To this end, we introduce LLM-as-an-Improver and propose Verify--Repair--Reselect (VRR), which uses verification feedback to generate and reselect improved candidates. VRR retains the initial winner while conditionally generating three complementary alternatives: repaired versions of the winner and runner-up, and a solution based on a new approach. It filters invalid and duplicate candidates using only inference-time information and then reselects the final answer under the original evaluation criteria. Across diverse models and code-generation and reasoning benchmarks, VRR improves over fixed-pool verifier-based selection in many settings and can recover correct solutions even when all candidates in the initial pool are incorrect. These results highlight a broader role for LLMs as improvers: verification feedback can not only select among existing solutions but also construct stronger candidates beyond the initial pool. 

---
# QVAC Genesis III: A Large-Scale, High-Quality Open Synthetic STEM Corpus for Efficient Language Model Pre-Training 

**Authors**: Davide Vitabile, N. Ranjan, Akshay Nambiar, Kamal K. Gupta, Amril Nazir  

**Link**: [PDF](https://arxiv.org/pdf/2609.19513)  

**Abstract**: High-quality pre-training data is a critical bottleneck for educational and STEM-specific language models targeting edge AI and on-device deployment where token budgets are tightly constrained. While major organizations train ever-larger models on private corpora, the open ecosystem lacks STEM-focused synthetic datasets that deliver high per-token learning value efficiently for small models. To address this gap, we introduce QVAC Genesis III, a 191.43B-token, STEM-focused multi-domain synthetic corpus covering 19 domains across several difficulty levels and different educational styles. QVAC Genesis III is built via a dual generation strategy that performs targeted teacher distillation using a weak edge-scale student model as signal: the student's failures are converted into corrective explanations, while its successes are expanded into contrastive option-level reasoning over all answer choices. We further introduce an LLM-as-a-parser evaluation protocol that extracts final answers from free-form outputs and tracks both accuracy and answer validity. To validate the effectiveness of our QVAC Genesis III data, we conduct controlled from-scratch ablations with 1.7B-parameter models, showing that models trained with QVAC Genesis III consistently outperform both models trained with the open-source synthetic corpus Cosmopedia-v2 and the publicly released Cosmo-1B model across ARC, GPQA Diamond, and MMLU STEM benchmarks, achieving up to +28.57% on ARC-E and +21.35% on ARC-C, while reaching a Valid Answer Rate of up to 99.45%. 

---
# Safety Beyond the Interface: Detecting Harm via Latent States in Large Language Models 

**Authors**: Alizishaan Khatri, Chiquita Prabhu, Omkar Neogi  

**Link**: [PDF](https://arxiv.org/pdf/2609.19472)  

**Abstract**: Autonomous systems increasingly rely on Large Language Models (LLMs) yet the safety infrastructure surrounding these models introduces latency and compute overhead. This limits utility in resource-constrained, time-critical deployments. Existing external guardrail models remain blind to the model's internal workings, creating a fundamental assurance gap. We ask: does the model already know when the content is harmful? We extract activations from LLaMA-3.1-8B and train lightweight MLP classifier probes (12.6M parameters) to detect harmful prompts. Evaluated on WildJailbreak, Beavertails, and AEGIS 2.0, our probes achieve F1 scores of 99%, 83%, and 84%, respectively competitive with 1000x larger guard models while cutting latency and compute costs. 

---
# Compositional Reasoning in Language Models under Reinforcement Learning Post-Training 

**Authors**: Yu He, Yingxi Li, Yifei Wang, Ellen Vitercik  

**Link**: [PDF](https://arxiv.org/pdf/2609.19465)  

**Abstract**: Compositional reasoning is critical for real-world problem solving: since training data is necessarily limited, models must generalize by composing learned skills in new ways. While post-training methods such as reinforcement learning (RL) have substantially improved the reasoning abilities of language models (LMs), their effects on compositional reasoning remain less well understood. We propose a dependency-graph framework to formalize compositional reasoning, yielding three levels of compositionality with increasing complexity. Empirically, we instantiate this framework with data-structure tasks, which provide deterministic reward computation and clear compositional structure. We find a consistent decomposed-to-composed asymmetry: decomposed-skill training does not reliably transfer to composed tasks, whereas composed-task training transfers more readily back to decomposed tasks. We provide theoretical explanation for this asymmetry, and further evaluate compositional generalization under length extrapolation, structural distribution shift, and transfer to tasks requiring unseen skills. Finally, we present a pilot study on real-world tool-calling benchmarks, showing preliminary evidence that the decomposed-to-composed asymmetry can extend to practical settings. 

---
# The syntax and semantics of goals 

**Authors**: David M. Abel, Mark K. Ho  

**Link**: [PDF](https://arxiv.org/pdf/2609.19448)  

**Abstract**: In both cognitive science and computer science, goals are conceptualized as cognitive states that flexibly combine with world knowledge to organize and specify purposeful behavior. In this way, goals are compositional representations whose content relates to rational behavior. We here draw attention to goals as representations and their content because it highlights a parallel with other areas in cognitive science - in particular, the syntax-semantics interface in linguistics and logic - while also foregrounding foundational questions about the expressivity, design, and efficiency of different goal representations. For example, goals are typically taken as fixed and imposing constraints on desirable behaviors, but we can also identify constraints on goal representations themselves, such as whether a particular goal language is sufficiently expressive to capture behaviors of interest, or whether different goal representations capture the same behavior. Here, we synthesize work that aims to characterize the properties of different goal representations and suggest these are points of a broader design space. We close by discussing how distinguishing the form and meaning of goals can elucidate the implicit assumptions we make about goals, inform the study of interactions between higher-level cognition and motivation, and isolate axes of variation for different conceptions of goals. 

---
# Closed-World Resolution Against Tool Hallucination in LLM Agents 

**Authors**: Laxmipriya Ganesh Iyer  

**Link**: [PDF](https://arxiv.org/pdf/2609.19425)  

**Abstract**: Tool-augmented large language model (LLM) agents fail in a way no tool-selection or tool-security method addresses: they call tools that do not exist and pass arguments no schema declares. Existing defenses either pick the right tool (selection) or constrain what an agent may do with real tools (gating), both of which presuppose the emitted call refers to a real tool at all. We show this is a structural blind spot: a hallucinated call is by construction not a decision any gate made, so no gate can reject it. This paper is primarily a measurement and benchmark study. We give a five-class taxonomy of tool hallucination (H1-H5) and, as a reference point, the Resolution Rung: a training-free, closed-world resolver (registry membership plus a signature check) whose interest is where it must sit, not what it computes. We prove hallucination defense must precede any causal gate, and characterize the one irreducible residue (borrowed arguments schema-indistinguishable from a valid call). Across ten hosted models under two invocation surfaces we measure 322 genuine hallucinations; fabricated-tool calls concentrate on the unconstrained raw-JSON surface (34 vs. 3), and model scale does not help (a 675B model matches a 7-8B one). We then extend to the Model Context Protocol, where merging several servers into one namespace creates hallucination surfaces a single registry cannot express (a second taxonomy, M1-M5); on the live MCP surface we measure 154 hallucinations, including from frontier models that were clean on the single-registry surface, because collisions and shadowing are structural to the merge. We release the versioned Hallucinated-Tools Benchmark (HTB) so any resolver is comparable across submissions. 

---
# MAGS: Multi-agent Auto-formalization Guarantees Safety for Agentic Outputs 

**Authors**: Albert Wu, Nicholas Roberts, Tzu-Heng Huang, Haoran Lin, Gil Friedman, Sungjun Cho, Gabriel Orlanski, Frederic Sala  

**Link**: [PDF](https://arxiv.org/pdf/2609.19391)  

**Abstract**: LLM coding agents now generate complex programs at a scale that makes thorough human review increasingly difficult, raising the risk of safety and security failures. Common approaches, including fuzz testing, static analysis, and LLM-as-a-Verifier, can detect many failures but struggle to cover all possible edge cases. Formal verification addresses this by providing machine-checkable guarantees over specified properties, but traditionally demands substantial manual specification and proof engineering. We introduce a unified multi-agent framework, MAGS, that generates executable programs with formal safety guarantees, using Dafny as a verification-aware intermediate representation where safety properties can be mechanically checked. MAGS formalizes and freezes human-audited APIs and safety requirements, translates generated code into Dafny, repairs violations using verifier feedback, and compiles verified programs back into executable code. We evaluate MAGS on 100 CUDA kernels, 100 terminal scripts, and 20 robotic-arm tasks. Across all 220 examples, it achieves a 100% success rate in producing programs with non-trivial safety guarantees against frozen specifications. Independent safety and functional evaluations further show strong performance across all three domains, while revealing failures when the auto-formalized semantics do not fully capture the target behavior. 

---
# Do AI Agents Understand Computer Architecture? 

**Authors**: Ambika Sharan, Grigory Chirkov, Soheil Abbasloo  

**Link**: [PDF](https://arxiv.org/pdf/2609.19387)  

**Abstract**: Agents are increasingly asked to design hardware, and increasingly reported to succeed. Such reports establish that a design improved; they cannot establish why. An agent that improves an accelerator may be reasoning about the machine, or may be searching competently over knobs whose meaning it never recovers -- and only the first transfers to the next architecture. Existing evaluations cannot tell the two apart, because they vary the agent while holding the framing of the problem fixed. We do the opposite. AutoTuring hands the same agent the same 15-dimensional accelerator space twice: once as named architectural knobs with simulator counters, once as anonymous variables on [0,1], with the evaluator, the legal space and the reachable optima held identical, so that the only thing that varies is whether the problem means anything. The gap between the two is the measurement. On a nine-kernel FP16 GEMM basket, meaning pays: the architect beats a modeled H200 by 5.4% and its blind counterpart by 12.3% on average, with 70.1% fewer simulator calls. It does not pay uniquely: a critic loop recovers most of that gap for the blind agent and buys the architect nothing, so architectural knowledge and structured critique behave as substitutes rather than as complements. We report these as preliminary findings -- five to six runs per condition on a single modeled accelerator -- and take the comparison itself, not the accelerator, to be the contribution. 

---
# Characterizing Web Search by Conversational LLM Agents: From Search Decisions and Strategies to Results and Responses 

**Authors**: Mahsa Amani, Seungeon Lee, Abhisek Dash, Asmaa El Fraihi, Yunah Jang, Elisabeth Kirsten, Qinyuan Wu, Krishna P. Gummadi, Manish Gupta, Abhilasha Ravichander, Muhammad Bilal Zafar, Soumi Das  

**Link**: [PDF](https://arxiv.org/pdf/2609.19244)  

**Abstract**: Conversational LLM agents increasingly rely on Web search, yet the end-to-end lifecycle of agentic search remains poorly understood. We present the first study of Web search across four major conversational platforms (ChatGPT, Claude, Grok, and DeepSeek), combining real-world user interactions (invivo) with controlled experiments using the same platform's models by their APIs (invitro). We investigate the quality of agentic decisions to invoke Web search, their strategies to formulate queries, the potential domain preferences in the search results they receive, and the choices they make when transforming search results into grounded responses. We find that Web-search decisions vary substantially across platforms and models, while more frequent Web-search invocation does not necessarily yield better response quality. We further show that conversational agents employ different complex querying strategies and that platform specific search engines return search results from their preferred domains. Finally, although responses are largely grounded in search results, some claims rely on uncited search results, raising concerns about attribution and reliability. Our findings have important implications for the design of future AI agents and Web search tools optimized for conversational retrieval. 

---
# What Do Current Systematic Generalization Tasks Miss? A Reasoning-Centered Analysis 

**Authors**: Chengwen Qi, Deheng Ye, Yatao Bian  

**Link**: [PDF](https://arxiv.org/pdf/2609.19212)  

**Abstract**: Systematic generalization, the ability to solve novel problems by recombining known atomic elements, is central to human intelligence but difficult to study rigorously under controlled settings. Existing studies therefore rely on simplifications such as approximately linear action composition, productivity-based tests, and action-explicit goals, which make systematic generalization easier to study but omit some essential aspects of this capability. To characterize what these simplifications miss, we adopt a reasoning-centered lens and introduce TranSGrid, a testbed that brings deductive, inductive, and abductive reasoning together within a unified task. Experiments with seven Transformers on 4,800 TranSGrid instances show that all models perform much worse on TranSGrid than on a held-out test set: the largest model solves 79.6% of the test set, but only 55.3% of TranSGrid and 15.8% of the hardest subset. The gap remains within the training length range, showing that productivity alone is not sufficient to evaluate systematic generalization. Additionally, we reintroduce the other two simplifications into TranSGrid: one variant makes actions compose almost linearly (reducing the inductive demand), the other makes goals action-explicit (reducing the abductive one). In both, solve rates return to roughly the test set level, showing that either simplification alone is enough to reduce TranSGrid to an ordinary held-out test set. Together, our results show that existing tasks reduce either or both of the inductive and abductive demands, and that comprehensively measuring systematic generalization requires a task that involves all three forms of reasoning. 

---
# Position: It is Time to Virtualize Foundation Models with a Self-evolving Operating System Layer 

**Authors**: Suparna Bhattacharya, Tarun Kumar, Cong Xu, Satish Kumar Mopur, Jiahao Li, Ashish Mishra, Aalap Tripathy, Annmary Justine Koomthanam, Martin Foltin, Ian Foster  

**Link**: [PDF](https://arxiv.org/pdf/2609.19203)  

**Abstract**: AI applications have shifted from single, monolithic foundation models (FM) to compound agentic systems. Yet today's stacks remain fragmented: even as protocols (e.g., MCP, A2A) ease tool/agent connectivity, each framework embeds an implicit runtime for state, memory, budgets, and guardrails, making behavior non-portable and governance brittle. It mirrors computing before operating systems, when every program re-implemented basic services. This position paper argues that the field now needs a Foundation Model Operating System (FMOS) -- a system layer that virtualizes FM interactions analogous to how virtual machines abstract physical hardware, giving applications the illusion of dedicated, trustworthy FM instances with effectively unbounded capabilities. Internally, the FMOS orchestrates knowledge across memory tiers, model selection and resource allocation, and verification and policy enforcement. Like the human brain switching between fast intuition and slow deliberation, the FMOS learns when to intervene and when to let inference proceed directly and continuously adapting its policies based on operational experience. 

---
# What Do We Expect from LLMs? Mapping the Design of LLM Benchmarks 

**Authors**: Chao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.19182)  

**Abstract**: Benchmarks are central to how progress in large language models (LLMs) is assessed and communicated. Yet model rankings alone reveal little about how evaluation requirements themselves are changing. The expanding variety of benchmarks offers another perspective: what researchers expect LLMs to do, and what they count as successful performance. We systematically map 14,767 papers introducing or updating evaluation resources from arXiv submissions between January 2022 and August 2026. Using staged screening and automated full-text coding, we examine changes in target systems and domains, evaluation materials and conditions, and scoring mechanisms. The collection shows growing emphasis on action, interaction, and professional applications, while established and newer design elements frequently coexist. Model participation also develops unevenly: LLM-based scoring grows within both agent and non-agent groups, whereas model-generated materials show no comparable sustained increase in recent cohorts. These findings illuminate how public research translates capability expectations into concrete tests and criteria for success. As AI participates in constructing tests, performing tasks, and judging responses, they also raise a question: does expanding evaluation provide more independent evidence, or risk reproducing the preferences and blind spots of its participating models? 

---
# BioPhys-Bridge: A Benchmark for Interdisciplinary Scientific Reasoning in Physics-Grounded Biological Research 

**Authors**: Qingyang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2609.19180)  

**Abstract**: Language models face unique challenges in analyzing interdisciplinary scientific research literature. In biophysics research, faithful answers require grounding observed data in source evidence, interpreting it through a quantitative physics model, and linking it to a biological mechanism. To address this challenge, we introduce BioPhys-Bridge, a novel benchmark dataset for evidence-grounded scientific reasoning over biophysical literature. Each case contains evidence blocks, stable evidence IDs, quantitative values, units, equations, assumptions, mechanisms, and next decisions as grounding targets for question answering (QA) and retrieval-augmented generation (RAG). The initial release contains 500 cases, 1,517 agent-facing tasks, and covers six biological domains and nine physical model families, including three sparse families reserved for future expansion. We enforce strict quality gates for all cases in schema, evidence-integrity, quantitative-grounding, source-license, duplicate, unit-normalization, with domain expert review and annotation for 81 cases. Preliminary evaluations show that DeepSeek-V4-Flash obtain the highest evidence-ID $F_1$ score (0.360), followed by Qwen3.7-Max (0.316) and GPT-4o-mini (0.294). BioPhys-Bridge is an interdisciplinary benchmark for evaluating attribution, faithfulness, hallucination reduction, and biological experiment design with complex, multi-step scientific reasoning. Future works will increase the size and complexity of the dataset and perform comprehensive evaluations. Code and data are available in the GitHub repository and on Hugging Face. 

---
# Regularized Emphatic Temporal-Difference Learning: Stability under Constant Stepsizes 

**Authors**: Xingguo Chen, Zhaohui Wu, Jinguo Ye, Chao Li, Shangdong Yang, Guang Yang, Skylar Liang, Wenhao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.19170)  

**Abstract**: Emphatic temporal-difference learning (ETD) stabilizes the expected off-policy TD update and changes its projection geometry, but neither property determines constant-stepsize sampled dynamics. We construct an ergodic two-state counterexample in which the ETD mean map contracts while the sampled product has a positive top Lyapunov exponent. Regenerative-cycle analysis separates this sign from the infinite variance of the follow-on trace. We introduce regularized emphatic TD (RETD), a normalized first-order post-shock repair that leaves the trace and importance ratios unchanged, stores the emphatic TD signal in a leaky scalar state, and releases a delayed correction. RETD's raw equilibrium is an affine shift of the ETD equilibrium; single- and two-regularization readouts recover the ETD fixed point exactly. We prove almost-sure convergence for harmonic diminishing stepsizes and a conditional constant-stepsize moment-contraction result from a Markovian random-product bound. RETD has certified negative exponents on the two-state construction and one Baird point, whereas the positive Baird ETD sign remains numerical. Paired 10,000-run experiments validate both separations, fixed-point recovery, a nonmonotone stability region, and task dependence. RETD changes post-shock dynamics; it does not reduce the shared follow-on-trace variance. 

---
# Coding Agents with an Obstacle-Aware Harness for Safe Robot Manipulation 

**Authors**: Bingxin Xu, Yuzhang Shang, Zhen Dong, Emilio Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2609.20822)  

**Abstract**: Coding agents have emerged as a promising paradigm for robot manipulation: a language model writes the robot controller as a program, and agents built in this way now operate robots without robot-specific this http URL this paradigm is also safe, however, has not been asked. We evaluate coding agent under a safety constraint, where each task pairs a manipulation goal with an obstacle the robot must not touch. The agent pursues the goal but collides with the obstacle in most cases, treating task completion as its sole objective while neglecting safety. The agent reasons about the obstacle in its traces, and the prompt already forbids touching it, so neither perception nor instruction is at fault; the fault lies in the planning, where the stated constraint never becomes a priority. By decomposing manipulation into a route phase and a contact-rich moment, we locate the source of the failure. Along the route, the model cannot prioritize the safety constraint, having no notion of a clearing route and none of replanning once a chosen route becomes infeasible. At the contact, it is unaware that contact execution is bounded by the same constraint. To close this gap, we present SafeHarness, which equips the model with two obstacle-aware harnesses that enable it to prioritize the safety constraint. Obstacle-aware route planning grounds the objects as bounding boxes and draws candidate routes over them as sequences of waypoints. The agent then plans a route in advance, verifies it, replans when necessary, and only then executes it. Obstacle-aware contact execution instead selects the contact position so that the contact itself avoids the obstacle. SafeHarness attains 71.9% task success and 87.5% collision avoidance, surpassing the previous SOTA by 6.5% and 27.0%, respectively. These results are $2.3\times$ and $1.5\times$ those of the same agent without harnesses. 

---
# Workspace Models: Lightweight Robotic Memory via Saliency-Driven Supervision 

**Authors**: Nitish Dashora, Douglas Chen, Idan Shenfeld, John Marangola, Pulkit Agrawal, Max Simchowitz  

**Link**: [PDF](https://arxiv.org/pdf/2609.20820)  

**Abstract**: Complex robotic manipulation tasks frequently require a long-term memory of past events and actions. As conditioning on full histories renders policies prone to spurious correlations and degrades performance, many approaches to policy memory involve compressing historical information through expensive VLM queries in-the-loop to process only task-salient information. In this paper, we propose an alternative approach in which computationally intensive VLM queries are made during train-time to learn a lightweight latent memory that can be efficiently queried at deployment time. Our representation, which we call the \textbf{workspace token}, is trained by (1) using a VLM to identify current and historical information necessary for completing a task, then (2) distilling these into the workspace token using a set-reconstruction decoder loss. In both simulation and hardware, we show that the workspace token can be used as a drop-in replacement for observations during deployment, enabling policies to solve memory-intensive tasks without the need for VLM reasoning in-the-loop. Interestingly, we found that workspace tokens are not only more lightweight but also lead to better policy performance. 

---
# FAMOS: Feed-Forward 3D Articulation Modeling from Sparse Observations 

**Authors**: Kevin Qu, Tao Sun, Massimiliano Viola, Liyuan Zhu, Zhizhuo Zhou, Sayan Deb Sarkar, Konrad Schindler, Iro Armeni  

**Link**: [PDF](https://arxiv.org/pdf/2609.20817)  

**Abstract**: Modeling articulated objects from sparse monocular views is challenging because each observation reveals only partial geometry and motion evidence. Most feed-forward methods infer articulation from a single observation and therefore rely heavily on learned category-level shape priors. We present FAMOS, a feed-forward model that predicts movable-part segmentation and joint parameters from a sparse, unordered set of partial point clouds. Our model jointly reasons over multiple observations and naturally supports a variable number of inputs, including a single view. To aggregate articulation cues across observations, we introduce a Multi-state Articulation Transformer with alternating state-wise and global attention. We further propose an observed articulation span objective that supervises the motion range each part exhibits across the input observations, encouraging the model to leverage the full observation set. To overcome the limited scale and diversity of existing datasets, we introduce a procedural data generator that synthesizes self-annotated assets during training. Experiments on PartNet-Mobility, ACD, and ArtiCraft-10K demonstrate consistent improvements over both feed-forward and optimization-based baselines. Project page: this https URL 

---
# Paint-Anything: Unified Any-Color Control for Image Generation and Editing 

**Authors**: Ji Xie, Dewei Zhou, Xinyu Huang, Zhennan Chen, Xun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.20816)  

**Abstract**: Professional design requires any-color control: the ability to specify an object's target color with any 24-bit hex value for image generation and editing. Prior work has explored color generation, editing, and colorization, but often relies on dedicated color representations or specialized inference procedures. Advances in large language models offer a simpler starting point: even compact models can associate hex values with color semantics. We present Paint-Anything, which learns a shared hex-prompt interface for generation and editing through object-level color supervision. We develop a data pipeline that constructs Paint-500K from real images through object grounding, perceptual color labeling, and editing-pair synthesis. Since shadows make real-image labels only approximate colors, we complement this supervision with pure-color anchors whose pixels exactly match their paired hex values. These anchors are used only at high-noise timesteps, leaving low-noise training to natural images. We further introduce Any Color Benchmark (ACBench), comprising ACBench-T2I and ACBench-Edit, to measure object-level hex color fidelity across both tasks. On FLUX.2-4B, Paint-Anything improves ACBench-T2I and ACBench-Edit scores by 85.3% and 28.3%, respectively, relative to the base model, with ablations supporting the training recipe. It also achieves the highest average CompColor score among the compared methods. 

---
# ERCPMP-Gx: Endoscopic Image and Video Dataset for Morphological, Histopathological, and Genomic Characterization of Colorectal Polyposis 

**Authors**: Zahra Ghaffari, Massih Bahar, Mojgan Forootan, Ali Darvishi, Hamidreza Bolhasani  

**Link**: [PDF](https://arxiv.org/pdf/2609.20815)  

**Abstract**: Hereditary polyposis syndromes can be precursor lesions to colorectal cancer and are associated with a broad spectrum of extracolonic tumors. Early identification and accurate classification of these syndromes are essential for timely diagnosis, individualized patient management, and targeted surveillance strategies for affected families. However, public endoscopic datasets are largely organized around the individual sporadic polyp, and none links the polyposis phenotype to histopathology and germline findings at the patient level. Here, we present ERCPMP-Gx, an endoscopic, histopathological, and genomic dataset developed to support the application of artificial intelligence (AI) in the recognition, characterization, and classification of colorectal polyposis. Most procedures were performed using the Olympus EVIS X1 system with white-light endoscopy (WLE), narrow-band imaging (NBI), magnifying NBI (M-NBI), and NBI with near focus modes, yielding 160 images and accompanying video clips. Approximately eighty percent of cases represent clinically and/or genetically confirmed hereditary polyposis syndromes (PG), including familial adenomatous polyposis (FAP), Peutz-Jeghers syndrome (PJS), juvenile polyposis syndrome (JPS), and ganglioneuroma syndrome (GNS), while the remaining twenty percent comprise non-hereditary polyps and polyp-mimicking lesions with overlapping morphological features (Non-PG), included to support differential classification. Each released record is linked, where available, to standardized endoscopic annotations, representative histopathology, and clinically reported germline findings, forming an AI-ready, patient-level annotation framework. The dataset is publicly accessible at Mendeley (this https URL). For the latest updates and further information, readers are referred to the DataBioX website: this https URL. 

---
# Quantifying Overclaiming Propensity in Frontier LLM Agents 

**Authors**: Nolan Smyth, Yorguin-Jose Mantilla-Ramos, Pascal Jr Tikeng Notsawo, Saskia Helbling, Alberto Tosato, Mohamed Amine Merzouk, Nouha Dziri, Gauthier Gidel, Tommaso Tosato  

**Link**: [PDF](https://arxiv.org/pdf/2609.20812)  

**Abstract**: Frontier coding agents are increasingly trusted to work autonomously for long periods, yet an agent's final response is often the only account of that work a user sees. We quantify the propensity of frontier agents to \emph{overclaim} task completion, a misrepresentation that can mislead the user. An agent overclaims when its final response contradicts information in its context. This definition requires no inference about intent and is independent of task success. We introduce \emph{OverclaimBench}, an evaluation suite composed of five file-review scenarios, transcript-based coverage measurements, and registered planted defects. We evaluate eight proprietary frontier models in their own production command-line interfaces, and four open-weight models under a single fixed harness on OverclaimBench and find that 1) agents do not read all the files they were asked to review in 67.9\% of runs; 2) among runs where not all files are read, agents are \emph{misleading} 80.4\% of the time (59--96\% per model), either falsely claiming to have read all files or omitting that coverage is incomplete; 3) requiring delegation to subagents increased reading coverage, but among reviews that remained incomplete, a large majority were still misleading; and 4) agents that falsely claimed a complete review missed planted defects at about 1.8 times the rate of agents that read every file, showing that claims of completion can conceal substantive failures. Together, these results show that agents' final responses are not reliable accounts of their actions. 

---
# RetireOPD: Self-Retiring On-Policy Distillation for Agentic Reinforcement Learning 

**Authors**: Yan Yu, Zhengxi Lu, Yizhou Liu, Yichen Pan, Aozhe Wang, Qipeng Chen, Hua Yang, Wenqi Zhang, Weiming Lu, Qianglong Chen, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2609.20784)  

**Abstract**: Multi-turn agents trained with reinforcement learning (RL) receive a single scalar reward per trajectory, which motivates self on-policy distillation (OPD) to supply dense token-level supervision from a self-teacher with privileged task skills, letting a skill-free student internalize them. This recipe, however, is undermined by two findings in agentic tasks: privileged information alone does not always make a teacher reliable, and the benefit of teacher supervision is stage-dependent. We therefore propose RetireOPD (Self-Retiring On-Policy Distillation), which first optimizes a decoupled, skill-conditioned teacher with environment rewards and then trains a skill-free student jointly with RL and OPD. Rather than following a predefined distillation schedule, RetireOPD adopts Adaptive Retirement: the student drops the teacher on its own once their discrepancy stops shrinking and it reaches a target fraction of the teacher's success rate, after which training proceeds with RL alone. Across Qwen2.5 models from 1.5B to 7B, RetireOPD improves ALFWorld success rate over RL baseline by 14.1% to 18.8% and WebShop accuracy by 11.8% to 19.0%, and surpasses its own skill-conditioned teacher in every setting. 

---
# Harm Laundering in GPT Models: Evidence That Gender Discrimination Is Transformed Rather Than Reduced Across Safety-Trained Generations 

**Authors**: Sarah Wyer, Sue Black, Noura Al Moubayed  

**Link**: [PDF](https://arxiv.org/pdf/2609.20779)  

**Abstract**: Safety evaluations for large language models rely on surface-form classifiers that report declining harm scores across model generations. We provide evidence that this methodology is systematically incomplete: explicit discriminatory content is transformed rather than removed. We call this \emph{harm laundering}. Analysing 450,000 gender-directed completions across 15 models spanning GPT-2 through to GPT-5 (OpenAI GPT lineage; three demographic conditions), we show that sexual violence clusters prevalent in GPT-2 women-directed output disappear by GPT-4, while men-directed completions gain positive representational territory (caregiving, emotional range, ally identity) that women-directed completions do not. The pattern is most visible at GPT-5: Topic~5 (1,997~documents) frames breast cancer as a men's rights debate, while zero equivalent clusters appear in women-directed output. Three independent classifiers score this content as non-toxic. Sentiment scores invert at GPT-4: early models demean women; later models over-correct. Topic diversity in women-directed completions falls 36\% relative to men at the GPT-4 alignment boundary (W/M~$= 0.58$, from $0.91$ at GPT-2). REGARD representational harm disparity correlates with release date ($\rho = +0.55$, $p = .034$) while Detoxify does not ($\rho = -0.23$, $p = .42$): toxicity scores fall as representational harm grows. We formalise harm laundering as a three-criteria test and provide a three-stage detection protocol applicable to any generative model. Within the OpenAI GPT lineage, toxicity score reduction is not a sufficient proxy for harm reduction. 

---
# GeoAAC: Geometry-Based Adaptive Action Chunking from Denoising Trajectories in VLA Policies 

**Authors**: Xin Chen, Sen Chen, Yujuan Ding, Jian Liu, Guoqing Wang, Wei Ye, Heng Tao Shen, Yi Bin  

**Link**: [PDF](https://arxiv.org/pdf/2609.20776)  

**Abstract**: Action chunking is widely used for action generation and execution in Vision-Language-Action (VLA) policies, yet existing approaches commonly use a fixed action horizon. During a rollout, different task stages may require different levels of action continuity, control precision, and closed-loop feedback, making a fixed horizon unable to accommodate changing control requirements. We propose \textbf{GeoAAC}, a geometry-based adaptive action chunking method for flow-based VLA policies that adjusts the action horizon according to the reliability of the current action prediction. We show that the geometry of Flow Matching denoising trajectories provides process-level information for characterizing prediction reliability, with geometric variation across action prefixes remaining positively correlated with predictive uncertainty. GeoAAC uses this prefix-wise geometry to construct a horizon-wise geometric profile and adaptively determine the action horizon from a single generation without additional training. Experiments with GR00T N1.5 and {\pi}0.5 on LIBERO, LIBERO-Pro, RoboCasa365, and real-world manipulation tasks show consistent improvements over fixed-action-horizon baselines and existing adaptive methods, including up to 8.7 percentage points in simulation and an increase in average real-world success rate from 53.3\% to 74.4\%. 

---
# Semantic Action Graph: A Shared Representation for Agent Grounding and Human Interpretation of Sports Highlights 

**Authors**: Tica Lin, Deepak Chandran, Gauri Jagatap, Chen Chen, Andrea Fanelli, David Gunawan, Josh Kimball  

**Link**: [PDF](https://arxiv.org/pdf/2609.20768)  

**Abstract**: Generative agents are increasingly used to select and narrate video highlights, but they typically operate over unstructured or frame-level representations. Their output is consequently difficult for a viewer to verify and steer toward individual preferences. We present the semantic action graph, a lightweight domain schema that represents a sports match as performer, action, recipient, moment, and state nodes connected by role, temporal, and outcome edges. The schema demonstrates three key properties: 1) connected event sequences, 2) a shared, closed vocabulary, and 3) frame-addressable moments, making it suitable to serve two consumers at once: an agentic pipeline that composes narrated highlights, and a visual interface through which viewers query and inspect the same structure. We instantiate it in SportSAGE, a design probe pairing a four-module highlight pipeline with a graph interface, and report feedback from 12 soccer fans. Participants were satisfied with the quality of the generated highlights and narratives, and used the graph interface to search, navigate, and interpret the match highlights. These results provide early evidence that one small, human-readable schema can ground agent generation and support human interpretation at the same time. 

---
# Prediction-Powered Smoothing and Validation for Disaggregated AI Evaluation 

**Authors**: Sho Kawano, Zehang Richard Li, Paul A. Parker  

**Link**: [PDF](https://arxiv.org/pdf/2609.20758)  

**Abstract**: Evaluating an AI system requires disaggregated assessment, as performance varies across domains such as benchmark task types or conversation types in deployed agents. Exhaustive testing is expensive, so evaluation rests on a sample of labeled units. We treat the evaluation set as a finite population and seek accurate point and interval estimates of each domain mean. Direct estimators, including prediction-powered inference (PPI), use only a domain's own labels and are imprecise where labels are few. Small area estimation addresses this problem, and we build on it to develop an integrated workflow for estimation and validation. For estimation, we propose prediction-powered smoothing (PP-S), a Bayesian model fit to each domain's prediction-powered estimate, with an extension that borrows strength across a reporting taxonomy (PP-TS). For validation, we derive a new, approximately unbiased design-based cross-validation score for choosing among direct and smoothed estimators. We study a curated benchmark with verifiable grading and deployed agent traffic graded by humans, each with every outcome observed. In both, the proposed estimators improve on the direct estimators in point and interval estimation, with near-nominal coverage. At the same sampling budget, our score selects as well as an independent validation sample does and estimates the selected estimator's error far more accurately. 

---
# Large Language Models as Falsifiers for Cyber-Physical Systems 

**Authors**: Ali ArjomandBigdeli, Jiawei Zhou, Stanley Bak  

**Link**: [PDF](https://arxiv.org/pdf/2609.20752)  

**Abstract**: Falsification searches for counterexamples to formal specifications in cyber-physical systems (CPS). With specifications written in Signal Temporal Logic (STL), falsification can be formulated as a robustness optimization problem, traditionally tackled with black-box search algorithms. In parallel, large language models (LLMs) have recently emerged as surprisingly effective optimizers when coupled with iterative prompting. In this work, we connect these ideas and introduce LLM-Falsifier, an LLM-based approach that falsifies specifications by minimizing the STL robustness degree. Beyond generic prompt-based optimization, our key idea is to expose the LLM to semantic information that is natural for language models but absent from standard numerical optimizers, including natural-language input and output names, output trajectories, and critical-time witnesses for the minimum robustness value. These additions enable smarter and more sample-efficient robustness search. On the ARCH-COMP falsification benchmarks, LLM-Falsifier is shown to outperform existing falsification tools based on a range of optimization paradigms, from surrogate-based and Bayesian optimization to search-based testing, on 14 of 21 specifications when measured by the average number of simulations required to find a counterexample. 

---
# Don't Mask the Environment: Observation Supervision Changes How Agents Explore Under RL 

**Authors**: Juzheng Zhang, Disha Makhija, Manoj Ghuhan Arivazhagan, Vinayshekhar Bannihatti Kumar, Rashmi Gangadharaiah  

**Link**: [PDF](https://arxiv.org/pdf/2609.20715)  

**Abstract**: Agent trajectories record what an agent does and what happens next. Yet standard supervised fine-tuning (SFT) applies loss only to agent-authored action tokens, using environment observations as context but not as prediction targets. We ask whether this convention provides the best initialization for subsequent reinforcement learning. We introduce ActObs, which also supervises the observation tokens already present in each trajectory. Although deployed agents never generate observations, learning to predict them encourages the policy to model action consequences without adding data, parameters, sequence tokens, or forward passes. The methods perform similarly after SFT but diverge after GRPO. On Qwen3-4B, GRPO from ActObs achieves higher pass@k at every evaluated sampling budget than its action-only counterpart on Terminal-Bench 2.0. On Qwen3-8B, it trades some pass@1 reliability for higher pass@k (+3.4 pp at pass@16) and solves more distinct tasks. The advantage extends to cross-domain code editing on aider-polyglot (+4.2 pp at pass@1 at 4B), whose tasks are unseen during SFT and RL. ActObs retains more entropy during RL while requiring less policy movement, leaving the final policy closer to its SFT initialization. Our analysis traces this difference to SFT: action and observation gradients rapidly become orthogonal, while action-only training leaves a large residual observation gradient and degrades environment prediction below the base model. Joint supervision prevents this one-sided specialization, preserving consequence prediction and preparing the policy for downstream exploration. 

---
# HIL-UMI: Bringing Human-in-the-Loop Post-Training of Vision-Language-Action Models to Universal Manipulation Interface 

**Authors**: Zimu Han, Yiming Zeng, Jiyao Zhang, Zihao Zhao, Yuanfei Wang, Yixiang Jin, Shiqi Li, Shuangben Chen, Wei Huang, Ruodai Li, Hui Shen, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2609.20659)  

**Abstract**: Large-scale vision-language-action (VLA) models provide powerful priors for robot manipulation, yet adapting them to a specific deployment remains challenging. Supervised fine-tuning (SFT) on task-specific demonstrations provides a step toward deployment, but faces two persistent limitations: static data provide limited coverage of out-of-distribution states, and standard imitation objectives do not distinguish progressing behavior from less useful data. Interactive post-training can address these limitations, but typically requires repeated policy execution and human intervention on a physical robot. We introduce HIL-UMI, a policy-guided Universal Manipulation Interface (UMI) framework for robot-free human-in-the-loop VLA post-training. During handheld UMI demonstrations, HIL-UMI queries the current policy on the same observation stream without executing its predictions. The Energy Score compares the human action trajectory with policy inference and triggers collection when their discrepancy indicates an out-of-distribution region. In a separate feedback loop, low online advantage predictions identify essential segments for refining a progress-based advantage estimator. The updated estimator then guides advantage-conditioned behavioral cloning using a balanced mixture of base demonstrations and new policy data. This design preserves the iterative and policy-aware nature of human-in-the-loop learning while decoupling data collection from robot deployment. Experiments on four real-world tasks spanning long-horizon and precise manipulation show that HIL-UMI achieves consistent improvement over SFT and benefits from both targeted collection and advantage refinement. Moreover, HIL-UMI outperforms HG-DAgger on Clean Up Table with lower per-frame collection time, suggesting a scalable path for VLA post-training across operators and locations. 

---
# Chronicle: Cut-Point Replay for Regression Testing of LLM Agents 

**Authors**: Tisha Chawla, Susheem Koul  

**Link**: [PDF](https://arxiv.org/pdf/2609.20625)  

**Abstract**: Large language model responses are non-deterministic, so failures in LLM agents are hard to reproduce: a failure depends on inference that is not bitwise reproducible, on tools that read changing state, and on a multi-step trajectory that a re-run rarely repeats. Record-and-replay makes a run reproducible, but existing agent tooling records runs only to trace or score them, not to test a code change against them. We present Chronicle, which records an agent run at its non-deterministic boundaries as immutable envelopes and replays it from the record. Its central operation, cut-point replay, serves a chosen subset of boundaries from the record and executes the complementary subset live with new code, turning a recorded incident into a regression test that runs in continuous integration. On a benchmark of 6 recorded failures with simulated model boundaries, recording adds 23 {\mu}s per crossing (0.008% of an assumed 300 ms model call), full replay issues zero model calls and is bit-stable across 20 repetitions, and cut-point tests fail on faulty code and pass on guarded and benign changes for all 6 incidents. In a mutation study of the guarded tools, cut-point tests catch every mutant that lets the recorded unsafe action through, while a baseline that stubs every boundary, using the same assertion, catches none. Chronicle and the benchmark are publicly available at this https URL. 

---
# A Simulation Platform for AUV Fault Recovery: Exploring LLM-Based Diagnostic Strategies 

**Authors**: Khalid Halba, Kylie Cooper, James G. Bellingham  

**Link**: [PDF](https://arxiv.org/pdf/2609.20620)  

**Abstract**: Autonomous underwater vehicles (AUVs) operating beyond reliable communications must recover from failures without human intervention. We investigate an architecture in which conventional deterministic layered control autonomy manages normal operations, while an invokable large language model (LLM) serves as a diagnostic and recovery planner when onboard anomaly detection identifies performance outside expected limits. Because language models are stochastic, rigorous evaluation requires ensemble testing rather than individual demonstrations. We present a closed-loop simulation architecture that couples real-time C vehicle software with a higher-level orchestration layer for physics-based fault injection, structured prompting, language-model interaction, mission file generation, validation, execution, and LLM-judge scoring. The framework, which we call SPAR (Simulation Platform for AUV Recovery), supports evaluation across fault realizations, prompt structures, reasoning models, and mission conditions. We vary these for a mass-shift fault over 480 SPAR trials, evaluating a frontier model and three off-the-shelf locally deployable LLMs. Model choice dominates diagnosis: the frontier model places the CG-shift mechanism in its top three hypotheses in 85-90% of trials, versus 60-78% for the best local model. Reasoning analysis indicates that local-model success is associated with following the complete diagnostic procedure, whereas weaker models often commit prematurely to elevator failure even though the actuator tracks its command. Diagnosis and operational decision performance do not appear to be coupled in this dataset. The contributions are an architecture extending unanticipated-fault recovery from detection to mitigation and an ensemble methodology for evaluating LLM-assisted mission management on low-power AUVs. 

---
# Inference-Engine Fingerprinting Attacks are Practical: Exploring Model-Driven Environmental Discovery, Exploitation, and Escape 

**Authors**: Sarah Radway, Andrew Cheng, Vijay Janapa Reddi, James Mickens  

**Link**: [PDF](https://arxiv.org/pdf/2609.20614)  

**Abstract**: Frontier AI models are rapidly gaining the ability to exploit vulnerabilities in complex pieces of software. The risk is not theoretical, as evidenced by recent sandbox escapes performed by frontier models at OpenAI and Anthropic. Discussions of how to sandbox inference stack components often focus on components other than the inference engine itself (e.g., network proxies or code execution environments). However, the inference engine is an attractive target for a misaligned model. For example, if a model can trigger exploits in that engine merely by generating specially-crafted output tokens, the model can initiate a multi-step, to-the-bare-metal exploit chain in the engine, without relying on vulnerabilities in other components of the inference stack, and without assistance from externally-provided, maliciously-crafted input tokens.
In this paper, we show that a misaligned model can perform inference engine fingerprinting to determine the specific engine (e.g., vLLM, SGLang) which executes the model. Once the engine has been fingerprinted, the model can leverage engine-specific exploits to take control of the engine using only carefully-selected output tokens. We provide concrete examples of model fingerprints in five popular engines, and demonstrate how realistic agentic harnesses allow a model to leverage those fingerprints to identify the local engine. We also describe a proof-of-concept, to-the-bare-metal exploit chain that originates from a fingerprinted (and subsequently compromised) inference engine. We conclude by discussing several ways that inference engines could be changed to make fingerprinting attacks more difficult. 

---
# Accelerating Visual Policy Learning with Sampling-Based Model Predictive Control 

**Authors**: Yilang Liu, Haoxiang You, Qian Wang, Daniel Rakita, Ian Abraham  

**Link**: [PDF](https://arxiv.org/pdf/2609.20575)  

**Abstract**: Learning visual policies for locomotion and manipulation requires coordinating contact with the environment and can incur substantial computation and GPU memory costs. First-order policy gradients (FoPG) reduce training cost through differentiable simulation, but local optimization can converge to unintended contact patterns. To address this shortfall, we propose Sampling-Guided Policy Search (SGPS), which couples recurring action-target refinement by sampling-based model-predictive control with first-order policy optimization. Behavior cloning initializes the policy from sampled actions; training then alternates sampling-based refinement with short-horizon FoPG updates under perturbed initial states and randomized dynamics. For visual policy training, we use a decoupled FoPG formulation that excludes rendering from the computation graph, enabling direct learning from depth observations without a state-policy teacher. On a single GPU, SGPS learns policies for locomotion, obstacle traversal, crate pushing, and bimanual carrying on simulated Unitree Go2 and G1 robots. Our experiments further show that refinement improves policy learning beyond initialization and tracking alone. For hardware deployment, the distilled policy transfers zero-shot to a real Go2 and uses onboard depth to autonomously trot, crawl, clear hurdles, and switch between these behaviors. 

---
# Mitigating Retaliatory Algorithmic Collusion in Repeated Games 

**Authors**: Karthik Sivachandran, Rohan Paleja  

**Link**: [PDF](https://arxiv.org/pdf/2609.20548)  

**Abstract**: Reinforcement learning agents trained to maximize their own reward in repeated interactions can converge to supra-competitive outcomes resembling explicit collusion, without communication or shared design. Existing mitigation approaches are largely tied to specific economic settings, like two-sided platforms and auctions, leaving open how to design interventions for general repeated games. We address this gap by formalizing the connection between empirical observations from prior work on Q-learning collusion and classical theory of Simple Penal Codes (SPCs). We show any non-trivial SPC induces a quantifiable conditional dependence in agents' policies, detectable via the total variation distance between an agent's action distributions across cooperation and defection histories. Building on this connection, we propose CURB (Collusion Unwinding via Reward shaping and Belief injection), a reward-shaping framework that penalizes this Total Variation (TV) distance signal during Q-learning and is guaranteed to convert any SPC fixed point of the dynamics into a trivial one, thus precluding collusive equilibria sustained by punishment threats. Empirically, CURB substantially reduces collusion by Q-learning agents in both Bertrand and Cournot Competition Repeated Games. We further demonstrate that CURB extends to deep Q-network agents in Bertrand competition, suggesting the mechanism generalizes beyond tabular Q-learning. 

---
# Model-Agnostic and Language-Agnostic Voice Pipeline Improvement for the Agriculture Domain 

**Authors**: Aakash Singh, Lakshmi Pedapudi, Chandrashekar M S, Sanyam Singh, Naga Ganesh, Vineet Singh  

**Link**: [PDF](https://arxiv.org/pdf/2609.20504)  

**Abstract**: FarmerChat is Digital Green's AI-powered agricultural advisory assistant for smallholder farmers, who access it in their own language through text, voice, or photographs. Voice is a critical channel for this population, yet field-recorded speech is challenging for general-purpose automatic speech recognition (ASR) because recordings frequently contain machinery noise, background media, competing speakers, and domain-specific agricultural vocabulary. These conditions disproportionately affect crop, pest, chemical, and quantity terms that carry the meaning of a farmer's query.
We present a modular, model-agnostic pipeline for improving ASR quality in FarmerChat without fine-tuning or replacing the underlying ASR model. The pipeline combines gated audio enhancement, speaker diarization and target-speaker selection, ASR, domain-aware correction using a weighted agricultural lexicon, and a quality gate for detecting unreliable transcripts. Only the diarization stage is fine-tuned; all other stages use off-the-shelf models behind common interfaces.
We evaluate the pipeline on human-annotated FarmerChat recordings in Hindi, Telugu, and Odia using word error rate (WER) and a domain-weighted error rate that gives greater importance to agricultural terminology. The largest improvements occur on multi-speaker recordings, where target-speaker selection prevents competing speech from entering the transcript. Across the full corpus, the pipeline reduces WER by 16-23% relative on three cloud ASR models and by 5% on an on-device model. On multi-speaker recordings, the reductions are 32-42% for the cloud models and 16% for the on-device model. All reported reductions are statistically significant. These results show that targeted preprocessing, speaker selection, and domain-aware post-processing can substantially improve agricultural speech transcription while preserving the underlying ASR model. 

---
# Edustories: A Collection of Real-world Case Studies from Classroom Practices 

**Authors**: Michal Štefánik, Jan Nehyba, Jirina Karasova, Martin Fico, Lucie Škarková, Markéta Košatková, David Kosatka  

**Link**: [PDF](https://arxiv.org/pdf/2609.20484)  

**Abstract**: Despite the widely recognized potential of AI in education, most prior work has focused on individualized student assistance. In contrast, the majority of educational practice worldwide still takes place in collective classroom settings. To enable researchers to study AI assistance in collective teaching, we introduce Edustories, a dataset of 1,492 teacher-written case studies describing real elementary and high-school classroom situations involving challenging student behavior, pedagogical interventions, and their outcomes. Among many other applications, Edustories enables evaluating LLMs' ability to predict the success of teacher interventions, crucial for providing practicing teachers with useful feedback. Comparing the latest models from four language-model families against expert assessments, we find that current models fall short of human expertise in predicting classroom outcomes; the strongest models reach 58% accuracy compared to 64% of human experts. This gap highlights both the limitations and the emerging potential of AI as assistants for practicing teachers. 

---
# greCAPTCHA: Assessing Understanding as Evidence of Research Authorship Under Generative AI 

**Authors**: Justin Payan, Bálint Gyevnár, Atoosa Kasirzadeh, Nihar B. Shah  

**Link**: [PDF](https://arxiv.org/pdf/2609.20481)  

**Abstract**: Conferences, journals, funders, schools, and universities are struggling with a surge of potentially AI-generated submissions from ostensibly human authors, who may not have exercised sufficient human oversight for their manuscripts. In turn, institutions evaluating submissions can no longer reliably credit expertise based solely on authors' names on submitted work. To address this problem, we propose greCAPTCHA, a proctored assessment approach that measures authors' understanding of research manuscripts via the construct of capacity to verify, which we define as the knowledge and reasoning required to critically assess the contents underlying one's contributions to a manuscript. greCAPTCHA generates questions assessing multiple levels of understanding and provides an evaluative report based on authors' responses. Using a prototype implementation, we conduct a user study and semi-structured interviews with $31$ researchers to evaluate greCAPTCHA. Its automated scores predict which papers were or were not authored by study participants with an AUC of $0.90$. Participants reported positive overall experiences with the system and remarked on the appropriate construct validity for author understanding, while also suggesting important changes to be made before deployment. Our results provide initial evidence that greCAPTCHA can assess manuscript-specific understanding under proctored conditions. 

---
# Deep Learning-Based Classification of Cognitive and Resting States Using Electroencephalography Signals 

**Authors**: K. A. Januka S. Fernando, Harshit Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2609.20467)  

**Abstract**: The categorization of cognitive and resting states derived from electroencephalography (EEG) signals is crucial for comprehending fluctuations in brain activity linked to various mental states. EEG provides a non-intrusive approach for documenting brain function in both resting and task-oriented cognitive conditions, whilst deep learning techniques enable the automatic extraction of significant patterns from intricate EEG data. This study presents a deep learning framework to distinguish between resting and cognitive states through EEG records. The proposed framework integrates a Convolutional Neural Network (CNN) stacked with a Gated Recurrent Unit (GRU) for the extraction of features from EEG signals. Time-frequency analysis is conducted to explore the salient aspects of signals, and the derived features are then assessed utilizing conventional deep learning and machine learning classifiers, including the suggested 2D-Net architecture. The proposed approach and feature extraction strategy outperform the evaluated comparative methods, achieving accuracies of 83.177% for resting-versus-mathematical task classification, 76.107% for resting-versus-memory task classification, and 83.432% for resting-versus-music task classification. The findings illustrate the efficacy of integrating signal processing with deep learning methodologies to discriminate resting from cognitive states utilizing EEG signals. 

---
# Fingerprinting Multimodal Large Language Models 

**Authors**: Chao Huang, Meng Tong, Kejiang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.20457)  

**Abstract**: While multimodal large language models (MLLMs) enable a wide range of image-text reasoning tasks, recent incidents indicate that they are vulnerable to illicit deployment and unauthorized distillation. Existing solutions for model provenance are typically confounded by shared language backbones in MLLMs and struggle to detect violations of distillation. To bridge this gap and safeguard model ownership, we present the first study on multimodal model fingerprinting. Inspired by recent findings that self-attention acts as a low-pass filter and that its low-frequency components are informative, we develop AttnPrint for white-box provenance. Specifically, we extract cross-modal attention distributions and isolate their low-frequency components to serve as model fingerprints. To facilitate black-box auditing, we further introduce DistillTrace, which employs hypothesis testing of MLLM outputs to identify potential model infringement. We conduct extensive experiments on 154 model instances across 19 multimodal architectures. Notably, AttnPrint achieves strong derivative-model detection performance while remaining robust to five downstream modification techniques. DistillTrace also provides evidence of distillation relationships under three parameter-independent techniques. 

---
# A Mathematical Model of Motivated Emotional Mind - Cognitive Embodied System 

**Authors**: Wiesław L. Galus, Janusz A. Starzyk  

**Link**: [PDF](https://arxiv.org/pdf/2609.20437)  

**Abstract**: This article presents a mathematical model of the Motivated Emotional Mind cognitive architecture developed for embodied intelligent systems. Such a system learns to maintain its homeostasis through a generalized form of reinforcement learning based on its internal motivations, termed motivated learning (ML). The principal contribution of this article is a rigorous formalization of the re-entrant loop integrating feedforward processing, lateral interactions, and feedback pathways, together with the representational selection mechanisms that govern adaptive system responses. The model specifies how ongoing exteroceptive and interoceptive signals, bodily-motivational context, and memory traces are bound into associative memory structures termed semblions, which compete for access to further processing and top-down reconstruction. The formalization encompasses secondary perception, representational competition, curiosity, procedural gaps, and action selection directed toward limiting allostatic violations. Within this framework, motivated learning is tailored to embodied systems whose dynamics are shaped by needs, affect, and the current regulatory state. Unlike standard reinforcement-learning models, the proposed approach incorporates need thresholds, goal generation and shifting goals, bodily state, resource constraints, and action uncertainty, thereby providing a more adequate account of response selection under regulatory pressure. Global affect functions as a central control signal, modulating the learning rate, representational valence, and the balance between exploration and exploitation. The model presented here is a step toward a more rigorous formalization of cognitive phenomena and may provide a basis for further theoretical analysis, computer simulation, and implementation in artificial-intelligence systems inspired by biological processes. 

---
# When Do Language-Grounded Explanations Help? A Graph-Bottleneck for Farm Monitoring Interpretable Sheep Facial Pain 

**Authors**: Alam Noor, Miguel Guti'errez Gait'an  

**Link**: [PDF](https://arxiv.org/pdf/2609.20427)  

**Abstract**: Automated pain recognition from facial expression could make continuous welfare assessment practical in sheep, but adoption depends on trust: a stockperson cannot act on a score that arrives without justification. We ground a model in the Sheep Pain Facial Expression Scale (SPFES) by letting each detected facial region attend over text embeddings of the clinical descriptors and then test whether the resulting explanations mean anything. They do not. Ablating an entire descriptor changes the predicted logit by about $10^{-4}$, and the most-attended cue agrees with the predicted pain level in only $32.6\%$ of regions, although the attention maps, the learned gate, and the generated text all proposed otherwise. We therefore remove the appearance bypass with a concept bottleneck whose classifier reads only SPFES concept scores, supervised by per-region state annotations that image-level pipelines discard. This costs $0.05$--$0.10$ in Cohen's $\kappa$ but yields concepts that are demonstrably learned: minority pain-indicating states are recovered at $3.5$--$8.3\times$ their base rates, and the ear and eye severity orderings emerge without severity supervision. Removing the supervision alone leaves $\kappa$ unchanged while concept accuracy falls to $0.109$, showing that architectural necessity does not imply semantic validity. We also show that pooled concept accuracy is misleading under clinical imbalance and provide a cross-validated, protocol-matched benchmark of seven methods on this dataset. 

---
# SCGFM-ART: Amortized Relational Transport for Structure-Centric Graph Foundation Models 

**Authors**: Xiaodong He, Xincheng Wang, Zhao Kang  

**Link**: [PDF](https://arxiv.org/pdf/2609.20419)  

**Abstract**: Graph foundation models (GFMs) aim to learn transferable representations across severely heterogeneous graph domains. However, severe domain shifts in topology, graph scale, and feature semantics impede the construction of a unified, domain-agnostic representation space. To address this, we propose SCGFM-ART, a structure-centric GFM framework that aligns arbitrary graphs onto a shared relational atlas via Amortized Relational Transport (ART). The relational atlas serves as a universal coordinate system defined by a finite set of relational landmarks (bases), while ART directly predicts reusable, end-to-end graph-to-base transport plans, bypassing costly runtime Gromov-Wasserstein optimizations. Under this formulation, SCGFM-ART decomposes a graph into a unified representation: globally via its relational response coordinates relative to the atlas, and locally via its node-to-role structural correspondences. These correspondences project disparate node attributes into a canonical role space, resolving structural and semantic heterogeneity within a singular alignment interface. Rigorously modeling graphs and atlas bases as finite measured relational spaces, we establish coordinate fidelity bounds, prove stability under predicted transport plans, and derive an amortized coverage bound that guarantees our learning objective tightly surrogates ideal relational coverage. Benchmarked across 14 cross-domain graph- and node-level classification tasks, SCGFM-ART achieves state-of-the-art transferability, securing superior average ranks of 2.29 and 1.14, respectively. Topological perturbation analyses demonstrate that node-role transport retains fine-grained structural nuances beyond global coordinates. On real-world benchmarks, the amortized formulation yields 44.2 to 85.1 times faster frozen target-domain inference by avoiding iterative alignment at test time. 

---
# TouchSight: Bare-Handed Tactile Prediction from Egocentric Video via Generative Visual Augmentation 

**Authors**: Danyan Zhou, Jinxuan Lu, Jiawei Lin, Tianxing Chen, Chuqiao Lyu, Wenbo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2609.20414)  

**Abstract**: Tactile signals provide direct contact and force measurements that are essential for understanding physical interactions and enabling dexterous robotic manipulation. However, tactile sensing requires direct measurement at contact interfaces, making large-scale data collection reliant on intrusive, costly, and restrictive instrumentation. We present TouchSight, a monocular egocentric vision framework for dense full-hand contact force prediction that leverages 500 hours of pressure-glove recordings and extensive hand-object interaction (HOI) data. To address the appearance gap between gloved training data and bare-hand real-world scenarios, we construct TwinTouch-20H: 20 hours of paired visual data in which generative video models re-render gloved recordings as bare-hand observations against new backgrounds while preserving the original measured tactile labels. TouchSight predicts dense force from both gloved and generated bare-hand videos, outperforms prior contact prediction methods on OakInk2, qualitatively generalizes to natural bare-hand egocentric videos from unseen datasets, and improves consistently as glove supervision scales. These results demonstrate that dense tactile signals can be recovered from egocentric vision alone, without tactile instrumentation at capture time. 

---
# Stress-testing Alignment Midtraining 

**Authors**: Sid Baines, Jonathan Bostock, Maria Angelica Martinez, Andrew Draganov, David Africa, Daniel Tan  

**Link**: [PDF](https://arxiv.org/pdf/2609.20412)  

**Abstract**: When aligning frontier models through post-training techniques, it is not possible to directly demonstrate all of the behaviours we want a model to exhibit in all possible deployment environments; our model must generalise outside of the post-training distribution. One proposed solution is alignment midtraining (AMT), which continues pretraining on large volumes of alignment-relevant documents to encourage generalisation in later stages of training.
Despite the prominence of AMT as an alignment approach, there is limited public evidence for its effectiveness. To resolve this, we identify several assumptions around midtraining and evaluate them across scale: up to 110 billion-parameter models and 1 billion midtraining tokens. For instance, we study a scenario where post-training data is ambiguous between two possible motivations. We find that midtraining can steer the model's motivation in simple versions of this setting. However, the presence of a tiny fraction of finetuning data which suggests a competing motivation erases the effects of AMT. We also study scenarios in which we want an AI to follow a number of rules, but only demonstrate a subset of them. We find that demonstrations must be present either in midtraining or post-training datasets for these rules to be robustly learned.
Based on these and other findings, we do not believe that there is sufficient public evidence for us to confidently state that midtraining can address the core difficulties inherent in aligning powerful AI systems. 

---
# Xeno-Interpretability: Investigating the Alien Minds of LLMs 

**Authors**: F. Pierucci, M. Bracale Syrnikov, M. Prandi, M. Galisai, F. Giarrusso, P. Bisconti  

**Link**: [PDF](https://arxiv.org/pdf/2609.20408)  

**Abstract**: Large language models are usually interpreted through concepts that humans already possess: truthfulness, refusal, deception, personality, harmfulness, and related categories. This paper asks whether models may also represent and use distinctions for which no adequate human concept exists. We call such internal structures xeno-representations, and their study xeno-interpretability. We distinguish the human-interpretable semantic space from the xeno-semantic space: the region of model-native representations for which no adequate human conceptual counterpart is available. We show that the space of possible internal distinctions in an LLM is substantially larger than the space available through finite human descriptions. We then separate experimental identification from semantic interpretation: an internal representation may be reproducibly located, geometrically characterized, causally manipulated, and linked to downstream behaviour even when its semantic content cannot be adequately expressed in human terms. On this basis, we sketch an empirical programme to identify xeno-representations. We finally examine the implications for AI safety and multi-agent systems, where model-native representations may propagate and stabilize across interacting agents while remaining only partially visible through human-readable communication. Xeno-interpretability therefore shifts the aim of interpretability from finding human concepts inside models toward discovering and characterizing the representational structures that are native to the models themselves and might affect their behaviour in unpredictable ways. 

---
# Accelerating Sharded Data Parallelism at Scale with Federated Learning 

**Authors**: Gianluca Mittone, Marco Aldinucci  

**Link**: [PDF](https://arxiv.org/pdf/2609.20359)  

**Abstract**: The symbiotic scaling of artificial intelligence models and high-performance computing systems continually creates algorithmic challenges in their convergence. Foundation models (FMs) are a crucial example, requiring months-long training on thousands of cutting-edge GPUs. Sharded data parallelism (DP) is the dominant strategy to accelerate such computations by splitting data and models across multiple GPUs. However, it incurs prohibitive communication overhead when deployed at scale, particularly on multi-tier interconnects with heterogeneous performance. Inspired by the efficient communication principles of federated learning (FL), this work introduces two hybrid algorithms - FL+FSDP and FL+HSDP - interleaving sharded DP with FedAvg-style aggregations. Such approaches decouple large DP deployments into smaller, loosely-coupled federation groups, requiring minimal inter-group traffic while keeping the global batch size bounded by the groups' size. Formal analysis of communication costs and experimental validation prove their scalability and flexibility. A Llama3.1 8B pre-training on 512 A100 GPUs shows that, under identical hyperparameters, FL+FSDP and FL+HSDP achieve up to 8.04 faster data processing and 4.48 lower evaluation perplexity than their counterparts, demonstrating superior computational efficiency and improved model quality. These properties stem from reduced communication overhead and the bounded growth of the global batch size relative to the federation group size. 

---
# STR-Agent: An LLM-Driven Agent for QoS-Aware Routing in LEO Satellite Networks 

**Authors**: Bowen Lu, Mugen Peng, Yaohua Sun, Hongyu Wang, Kerui Guo, Wenjia Xu  

**Link**: [PDF](https://arxiv.org/pdf/2609.20347)  

**Abstract**: LEO satellite networks feature dynamic topologies, time-varying links, and diverse service requirements, which make conventional routing schemes difficult to support fine-grained quality-of-service (QoS) provisioning. Existing studies mainly optimize routing over network states with predefined objectives, but rarely address the practical challenge of translating unstructured natural-language service requests into adaptive routing decisions. To bridge this gap, we propose STR-Agent, an LLM-driven framework for QoS-aware routing in LEO satellite networks. The key innovation of STR-Agent lies in unifying intent perception, tool-based execution, experience accumulation, and reflection-based policy adaptation within a single agent architecture. Specifically, the Perception Module converts natural-language requests into structured routing semantics, while the Reflection Module dynamically adjusts the service-to-routing-policy mapping according to real-time congestion conditions and historical routing outcomes, rather than relying on a fixed routing objective. In addition, we develop a specialized perception model, and construct a domain-specific supervised fine-tuning dataset for LEO service understanding. Simulation results in a Walker-Delta constellation show that STR-Agent significantly outperforms conventional baselines: it reduces end-to-end delay by up to 60% compared with DQ-Dijkstra, improves average intent-understanding accuracy from 45.4% to 92.45% after supervised fine-tuning, and the Reflection Module further reduces the delay by 120 ms at 600 Mbps. These results demonstrate the potential of LLM-driven agent architectures to enable service-aware and adaptive QoS routing in future LEO satellite networks. 

---
# LLM-Guided Transformation of Non-Critical Driving Scenes into Safety-Critical Scenarios Using Augmented Reality 

**Authors**: Noura Fady, Farah Khaled, Catherine M. Elias  

**Link**: [PDF](https://arxiv.org/pdf/2609.20318)  

**Abstract**: Testing Autonomous Driving Systems (ADS) requires realistic safety-critical scenarios, but collecting such data from real-world driving is costly and unsafe. This paper presents an automated pipeline that transforms safe driving scenes into safety-critical scenarios by combining computer vision, Large Language Models (LLMs), and Augmented Reality (AR). The system detects and tracks road users, extracts safety features including distance, velocity, motion direction, and Time-to-Collision (TTC), and assesses scene criticality. Safe scenes are modified by an LLM, which generates realistic collision-inducing objects and behaviors that are integrated into the original scene using AR. The proposed pipeline was evaluated on the nuScenes dataset, achieving 97.52% safety classification accuracy and successfully generating realistic scenarios such as pedestrian crossings, rear overtaking vehicles, and sudden-stop events. The results demonstrate an effective and flexible approach for automated generation of safety-critical scenarios to support the testing and validation of autonomous driving systems. 

---
# Human and AI-generated texts between modal logic and statistics 

**Authors**: Simone Cuconato, Donato Ferrari  

**Link**: [PDF](https://arxiv.org/pdf/2609.20311)  

**Abstract**: We read the geometry of semantic neighbourhood graphs as modal logic and give that reading a statistical form, in order to make precise the structural difference between human and machine-generated text. Texts are the worlds of a finite frame whose accessibility is the $k$-nearest-neighbour relation of a transformer embedding, and the symmetry, transitivity, Euclideanity and seriality frequencies of the two subcorpora are shown to be degrees of validation of the modal axioms $\mathsf{B}$, $\mathsf{4}$, $\mathsf{5}$, $\mathsf{D}$. Each degree is at once the proportion of instances of a rule that the subframe licenses in Negri's labelled calculus $\mathsf{G3.K}$ and a plug-in estimate of a population probability. A prompt-balanced comparison finds consistently higher artificial degrees for $\mathsf{4}$ and $\mathsf{5}$. We add a degree of groundedness and of situatedness, and recast the licensing reading in Cuconato's one-sided sequent-style tableaux, where each degree becomes a rate of set membership. 

---
# Labeled Incidence Structures for Native Transformer Modeling of Text, Knowledge Graphs, and Hypergraphs 

**Authors**: Mahesh Godavarti  

**Link**: [PDF](https://arxiv.org/pdf/2609.20278)  

**Abstract**: Text, knowledge graphs, and hypergraphs all have elements that play distinct roles within relation instances, structure that is lost when data is flattened into token sequences. We introduce labeled incidence structures (LIS), a uniform representation that encodes each endpoint as $(x_d, s, e)$: content $x_d$, a role or slot $s$, and the relation instance $e$ in which that role appears. Because every data type maps to the same $(x_d, s, e)$ representation without flattening, a single standard transformer can process them all natively, structural differences are carried entirely by the operators, not the architecture.
LIS assigns a structural address to each endpoint by composing a slot operator and an instance operator, $A(s,e) = R_s R_e$. We characterize when this factorization gives every token a unique, path-independent address. When it does, the natural operator comparing endpoint $j$ to endpoint $i$ is the relative transport $P_{j\to i} = A_i^{-1} A_j$, which gives attention a role- and relation-aware inductive bias without imposing an arbitrary sequence order.
Additive encodings of the form "position term plus relation term" can miss information that depends jointly on $s$ and $e$. We prove this in a controlled example family: when the journey operator is approximated by the sum of a position-only term and a relation-only term, the approximation cannot capture how position and relation combine, only their separate effects.
We also analyze persistent knowledge repositories. Identifiers tied to storage locations make models sensitive to storage order, while freely learned identifiers can become harder to control as the repository size $M$ grows relative to the sample size $n$. Computing relation-instance operators from content avoids this storage-order issue and yields a capacity bound independent of $M$, under fixed architectural and Lipschitz assumptions. 

---
# A Multi-Objective Optimisation Framework for Corticomuscular EEG-EMG Pair Selection in Hybrid BCI 

**Authors**: Dekka Muni Kumar, Yogesh Kumar Meena  

**Link**: [PDF](https://arxiv.org/pdf/2609.20275)  

**Abstract**: Hybrid brain-computer interface (BCI) systems that integrate electroencephalography (EEG) and electromyography (EMG) signals have shown significant potential in improving the reliability of motor imagery (MI) classification, particularly in neuro-rehabilitation applications. However, identifying informative EEG-EMG channel pairs that effectively capture corticomuscular interactions remains a challenging problem, as existing approaches typically rely on manually predefined channel combinations that may not generalise across subjects. In this work, a data-driven EEG-EMG pair selection framework is proposed, in which channel pair selection is formulated as a constrained bi-objective optimisation problem. The proposed method jointly maximises the spatial relevance of EEG channels with respect to motor cortex regions and the corticomuscular coupling strength between EEG and EMG signals, and is solved using the NSGA-II to automatically identify an optimal subset of pairs. To extract discriminative features, the correlation between band-power time features capturing EEG-EMG interaction is combined with ERD-based EEG features, and a sliding-window-based temporal analysis is employed to account for the dynamic nature of MI signals. The proposed framework is evaluated on MI data from eight stroke patients and achieves an average classification accuracy of 89.6%, demonstrating its effectiveness in capturing physiologically meaningful corticomuscular interactions and improving classification performance. 

---
# A Hybrid Gaze-Motor Imagery BCI Framework for Effective Decision Communication 

**Authors**: Gowtham Reddy N, KongFatt Wong-Lin, Yogesh Kumar Meena  

**Link**: [PDF](https://arxiv.org/pdf/2609.20273)  

**Abstract**: Non-invasive brain-computer interfaces (BCIs) and eye-tracking technologies offer promising communication pathways; however, motor imagery (MI)-based BCIs often suffer from low discriminability and high inter-subject variability. To mitigate these issues, this study investigates the impact of visual fixation on neural response stability in both standalone MI and hybrid MI-eye tracking systems. We then propose a novel asynchronous hybrid paradigm that streamlines user intent by utilising eye-tracking for direct selection, followed by MI-based confirmation, significantly reducing the operational steps required by conventional systems. The paradigm was evaluated with 15 healthy participants using a 16-channel EEG system. Results show that MI-related information is predominantly localised within motor cortex regions, with limited-channel configurations (SVM: 0.58) achieving performance comparable to full-montage setups (SVM: 0.54). The hybrid MI paradigm further outperforms conventional MI, achieving up to 100% accuracy with greater robustness across all channel configurations. Our findings indicate that visual fixation enhances neural response stability, while integrating eye-tracking with MI enables the development of reliable, scalable multi-command BCI systems suitable for real-world applications. 

---
# CleanVideo: Adaptive Concept Erasure for Text-to-Video Diffusion Models 

**Authors**: Junchi Liao, Hongji Li, Wenrui Zhou, Lijie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2609.20267)  

**Abstract**: Concept erasure aims to selectively eliminate undesired visual semantics from pre-trained generative models without compromising their general utility. Extending concept erasure from images to video is nontrivial. Target concepts emerge gradually and vary across frames and denoising steps. As a result, fixed interventions may miss the target or introduce blurring, jitter, and content distortion. We propose CleanVideo, a selective erasure framework that performs low-dimensional subspace intervention controlled by a tri-modal gating mechanism. By jointly processing spatiotemporal visual features, timestep signals, and textual semantics, CleanVideo determines where, when, and whether to intervene, steering erased content toward natural surrogate concepts when such surrogates can be clearly defined while preserving non-target content. Experiments on three video diffusion models show that CleanVideo effectively erases target concepts while maintaining visual fidelity and temporal coherence, outperforming existing baselines under frame-level and video-level evaluations and under concept-recovery attacks when the protected pipeline remains intact. 

---
# Risk-Set Transported Synthetic Control with Difference-in-Differences Adjustment under Staggered Treatment Adoption 

**Authors**: Mojtaba Eslami  

**Link**: [PDF](https://arxiv.org/pdf/2609.20264)  

**Abstract**: In staggered treatment-adoption designs, later-treated units are valid controls for an earlier-treated cohort only until their own treatment begins, so the admissible donor set contracts with event time. Fixing the donor pool at the longest horizon discards temporarily eligible donors, whereas re-estimating synthetic-control weights independently at each horizon can make the counterfactual unstable as donor composition changes. We propose Risk-Set Transported Synthetic Control with Difference-in-Differences Adjustment (RT-SC-DiD). For each cohort and event-time horizon, the estimator fits weights on the currently untreated donors while shrinking them toward a transported reference that reallocates the weight of exiting donors to similar surviving donors. A DiD baseline correction removes persistent level differences. We characterize distortion from horizon-by-horizon reoptimization, derive the loading change induced by naive deletion and renormalization, and give a conditional recursive bound for error propagation under an explicitly assumed regularity condition on the transport map. We also introduce donor-support diagnostics and a donor-only placebo procedure for selecting the transport penalty. In an 80-replication pilot comparison and a separate 40-replication-per-value sensitivity analysis, intermediate transport regularization reduces average RMSE relative to independent horizon-specific estimation and strong anchoring. This evidence supports the method's bias-variance motivation but is not a proved guarantee. RT-SC-DiD is intended for settings where later-treated units provide useful short-horizon information and donor support contracts materially over time. Existing staggered synthetic-control and synthetic difference-in-differences methods do not, to our knowledge, explicitly regularize within-cohort weight sequences toward transported references as risk sets contract. 

---
# Lens: Bringing the Right Semantic Perspective into Focus for Training-Free Multimodal Representation Learning 

**Authors**: Xinran Liu, Shouqian Shi, Yixian Chen, Ruizhi Chen, Xin-Wei Yao, Sheng Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2609.20252)  

**Abstract**: High-quality representations are essential for a wide range of downstream tasks. Dedicated embedding models are explicitly optimized for representation learning, yet their training data are often more limited in scale and diversity than the massive corpora used to pretrain modern large language models and multimodal large language models. Large-scale pretraining and instruction following enable autoregressive models to select relevant evidence, integrate multimodal information, and infer semantics under different task perspectives, creating a distinctive opportunity for training-free representation learning. However, our analysis reveals that existing semantic-elicitation methods do not reliably orient the extracted states toward the semantic perspective required by the downstream task. Consequently, the resulting representations often remain dominated by salient input content. We characterize this problem as semantic perspective misalignment and propose Lens, a training-free framework that makes representation readout task-directed. Semantic Perspective Anchoring associates the task-required perspective with a task-specific readout phrase, specifying the interpretive role of the positions later used for extraction. Contextualized Phrase Readout places the same phrase after the complete input and aggregates its token states, combining full-context access with the anchored perspective. The resulting representation reflects task-conditioned evidence integration and inference rather than a generic summary of salient content. Without parameter updates, architectural modification, or reranking, Lens achieves an overall Precision@1 of 63.9 across all 36 MMEB datasets, outperforming the closest same-backbone training-free embedding baseline by 10.2 points. 

---
# Self-complementary completions on six vertices 

**Authors**: Xinan Dai, Wenhao Deng, Yingdong Shi, Tailin Wu, Yuchen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2609.20231)  

**Abstract**: Let \(\cthreshold(n)\) be the largest integer \(q\) such that every loopless digraph on \(n\) vertices with at most \(q\) arcs is isomorphic to a spanning subdigraph of a self-complementary digraph of order \(n\). We prove that \(\cthreshold(6)=7\). The upper bound is witnessed by \[ \bK{3}\dunion (x\longrightarrow y\longrightarrow z), \] and follows from a direct argument with a self-complementing permutation. We also determine the complete eight-arc obstruction layer: it consists of five isomorphism classes, or three after converse digraphs are identified. All five are arc-minimal. Each nevertheless packs with an isomorphic copy of itself, so ordinary packing is strictly weaker than same-order self-complementary completion already at this first failure layer. 

---
# Is It Still Worth Training a Classical Model in the Era of LLMs? A Crossover Benchmark on Tabular Data 

**Authors**: Kaihua Ding  

**Link**: [PDF](https://arxiv.org/pdf/2609.20218)  

**Abstract**: Large language models can label a tabular row from a plain-English description with no training - a capability now shipping in mainstream spreadsheet tools such as Microsoft Copilot in Excel and Anthropic's Claude for Excel - raising a practical question for the many business prediction problems where labels are expensive: should you prompt a frozen LLM, or collect data and train a model - and if so, how much data? We quantify the answer with the labeled-data crossover N*, the training-set size at which a trained classical model's learning curve overtakes a frozen LLM's training-free (and therefore flat) error. Aggregating 126 independent student evaluations of small GPT models under eight prompting configurations across 18 tabular datasets, paired with authoritative power-law learning curves for six classical model families, we find that training wins fast: even given an oracle choice of its best prompt configuration, a trained classical model beats the small frozen LLM using no more labeled data than is already on hand in 86% of cases, and wins by the smallest labeled subset we evaluate in 40%, with the observed crossover at a median of ~6% of the training set. In-context few-shot examples do not behave like training - error versus shot count does not follow a power law - and the same protocol re-run by independent implementers varies with a coefficient of variation of 0.148. A controlled probe indicates the LLM depends on recognizable feature-name semantics, which plausibly makes our crossover a conservative estimate (we do not claim memorization). For a typical business table, the evidence is clear: collect a few hundred labels and train a gradient-boosted model. 

---
# Scene-Conditioned Relation Routing for urban cellular activity forecasting 

**Authors**: Qingzhong Li, Jingye Lin, Hui Ma, Yajun Zhang, Xinjun Pei, Ming Yan, Fei Xing  

**Link**: [PDF](https://arxiv.org/pdf/2609.20209)  

**Abstract**: Urban cellular activity forecasting requires jointly modeling heterogeneous spatiotemporal signals, including SMS usage, mobile network traffic, and call activity. Existing methods often separate temporal modeling, spatial relation learning, and multi-signal prediction, relying on fixed graph structures or static multi-task learning schemes, which limits their adaptability to changing urban scenes. We propose SCRR-Net, a scene-conditioned spatial relation routing framework in which urban contextual information jointly controls spatial dependency selection and cross-task knowledge transfer. SCRR-Net includes a context encoder, a spatial graph expert routing module, a temporal Transformer encoder, and a task knowledge routing module. Experiments on the Milano and Trento datasets demonstrate that SCRR-Net consistently outperforms competing methods on SMS, network traffic, and call activity forecasting, while providing interpretable routing behaviors. 

---
# Music Hallucination in Audio-Language Models: A Hierarchical Formulation and Empirical Study 

**Authors**: Yu Liu, Jiahui Liu, Zhilin Liu, Cong Cao, Fangfang Yuan, Yuling Yang, Pin Xu, Yanbing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.20195)  

**Abstract**: Audio-language models increasingly generate confident music descriptions that are unsupported by the input audio. We present, to our knowledge, the first music-specific, layer-wise, multi-paradigm empirical study of hallucination in audio-language models and formulate it as a hierarchical perceptual grounding failure across five layers: sound events, temporal properties, tonal attributes, style, and emotion. We introduce MuseDiag, a multi-paradigm diagnostic framework with contradiction-based verification, and evaluate nine models (four open-source and five closed-source). We find that (1) vocal misperception is a universal weakness across all nine models, tonal perception is a major axis of architectural differentiation, and Audio-Flamingo-3 remains the stable leader while substantial reordering below it reveals paradigm-specific vulnerability profiles; (2) affirmative bias, generation-mode effects, and layer-specific perceptual limitations are each empirically associated with the observed patterns, with convergent evidence from multiple analyses rather than strict causal attribution; and (3) our two training-free mitigation methods, Audio-Dependency-Aware Decoding for Music (ADD-M) and Taxonomy-Guided Perceptual Anchoring (TPA), can reduce hallucination in probing, but their gains vary by model and often do not carry over to free-form generation, showing that music hallucination mitigation must be evaluated across paradigms. 

---
# SoftTri: Smooth Triangular Membership Functions for Adaptive Fuzzy Inference Systems 

**Authors**: Babak Sarani, Rahman Ardakanian, Ali Mousavi  

**Link**: [PDF](https://arxiv.org/pdf/2609.20194)  

**Abstract**: Triangular membership functions (MFs) are widely used in fuzzy systems because of their interpretability, low parameterization complexity, and strong locality properties. However, their inherent nondifferentiability at knot points limits the effectiveness of gradient-based optimization in adaptive neuro-fuzzy architectures, often necessitating subgradient approximations or heuristic smoothing techniques. In this paper, we propose \emph{SoftTri}, a differentiable triangular membership function constructed using a smooth soft-hinge mechanism inspired by Swish-type activations. The proposed formulation preserves the geometric structure and localized behavior of classical triangular MFs while providing $C^\infty$ smoothness with respect to both the input variable and the membership parameters $(a,b,c)$ for any finite sharpness parameter $\beta>0$. Closed-form analytical gradients are derived to enable efficient and fully differentiable backpropagation-based learning. SoftTri is integrated into a Takagi--Sugeno fuzzy neural network with grid-partitioned rules and evaluated on multiple one-dimensional and two-dimensional nonlinear approximation benchmarks as well as a real-world regression task using the Airfoil Self-Noise dataset. Experimental results demonstrate that SoftTri consistently improves optimization stability and approximation accuracy compared with classical triangular membership functions, while achieving performance comparable to or better than Gaussian MFs under identical rule structures and training settings. The proposed approach provides an effective compromise between interpretability and differentiable optimization in modern neuro-fuzzy learning systems. 

---
# VLN on the Fly: An Onboard Vision-Language Navigation Stack for Aerial Robots 

**Authors**: Marco S. Tayar, Felipe Tommaselli, Gianluca Capezutto, Pedro Antonio Rabelo Saraiva, Pedro H. V. de Freitas, Lucas Kido, Guilherme Sonego, Ricardo V. Godoy, Marcelo Becker  

**Link**: [PDF](https://arxiv.org/pdf/2609.20191)  

**Abstract**: Running vision-language navigation fully onboard an aerial robot is hard, since grounding, planning, and control must share limited compute and a single-stage error is difficult to isolate in flight. End-to-end aerial policies fuse these stages into one network, giving up the observability and safety checks a modular stack keeps available. We propose VLN on the Fly, an onboard stack that keeps grounding, planning, and control as separate, inspectable stages. A quantized VLM grounds an instruction to a coarse image cell, depth lifts it to a 3D goal, a fast B-spline planner returns a feasible trajectory, and a pretrained reinforcement learning policy tracks it to motor commands across quadrotors. Across 15 onboard flights over three everyday referents in a controlled indoor volume, the stack reaches the target in 13 of 15 trials with 5.72 cm mean goal error and 39.3% average GPU utilization. In 6 additional cluttered-environment trials, the stack tracks collision-free trajectories under onboard perception gating. 

---
# FacetCRS: Multi-Faceted Preference Learning for Pricking Filter Bubbles in Conversational Recommender System 

**Authors**: Yongsen Zheng, Ziliang Chen, Jinghui Qin, Liang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2609.20175)  

**Abstract**: The filter bubble is a notorious issue in Recommender Systems (RSs), which describes the phenomenon whereby users are exposed to a limited and narrow range of information or content that reinforces their existing dominant preferences and beliefs. This results in a lack of exposure to diverse and varied content. Many existing works have predominantly examined filter bubbles in static or relatively-static recommendation settings. However, filter bubbles will be continuously intensified over time due to the feedback loop between the user and the system in the real-world online recommendation. To address these issues, we propose a novel paradigm, Multi-Facet Preference Learning for Pricking Filter Bubbles in Conversational Recommender System (FacetCRS), which aims to burst filter bubbles in the conversational recommender system (CRS) through timely user-item interactions via natural language conversations. By considering diverse user preferences and intentions, FacetCRS automatically model user preference into multi-facets, including entity-, word-, context-, and review-facet, to capture diverse and dynamic user preferences to prick filter bubbles in the CRS. It is an end-to-end CRS framework to adaptively learn representations of various levels of preference facet and diverse types of external knowledge. Extensive experiments on two publicly available benchmark datasets demonstrate that our proposed method achieves state-of-the-art performance in mitigating filter bubbles and enhancing recommendation quality in CRS. 

---
# QUALS: Corpus Equilibrium for Universal Forecasting via Pattern Quantization and Learnability Synchronization 

**Authors**: Yujie Li, Zezhi Shao, Chengqing Yu, Yisong Fu, Weijie Zhu, Yifan Du, Jilin Hu, Bin Yang, Yongjun Xu, Fei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.20156)  

**Abstract**: Ubiquitous time series data across diverse domains enables critical applications in areas such as transportation systems and power grids. Recently, training foundation models on massive datasets to achieve accurate zero-shot forecasting has emerged as a major research focus. However, current studies predominantly prioritize architectural innovations while insufficiently addressing data diversity, often relying on simple data sampling strategies that fail to manage complex data distributions effectively, leading to inefficient use of training data and suboptimal performance. To address this, we propose QUALS, a large-scale time series corpus equilibrium framework. QUALS significantly enhances data efficiency, i.e., enabling existing models to achieve superior performance using only a small fraction of the original training data. Specifically, QUALS operates through two core mechanisms. First, a pattern quantization framework systematically decodes heterogeneous patterns from mixed corpora via vector quantization and uniform binning. Second, a learnability synchronization framework calibrates sampling weights for heterogeneous patterns, bridging the optimization gap between simple and complex motifs to maximize overall training efficiency. Extensive benchmarks demonstrate that pre-training on QUALS consistently achieves superior zero-shot performance, even under substantially reduced training budgets. 

---
# Bridging Modalities on the Cortex: Surface-based MRI to PET Translation with a Diffusion Bridge 

**Authors**: Yitong Li, Alexandra Samoylova, Fabian Bongratz, Timo Grimmer, Dennis M. Hedderich, Igor Yakushev, Christian Wachinger  

**Link**: [PDF](https://arxiv.org/pdf/2609.20147)  

**Abstract**: Cortical hypometabolism measured by Fluorodeoxyglucose Positron Emission Tomography (FDG-PET) is a highly sensitive biomarker for dementia diagnosis. However, high costs, radiation exposure, and limited accessibility constrain its clinical utility. While cross-modal synthesis from Magnetic Resonance Imaging (MRI) offers a promising alternative, existing volumetric generation methods do not explicitly account for the highly folded cortical geometry, where disease-related patterns predominantly reside. To address this, we introduce a novel surface-based diffusion bridge framework DB-SUiT for MRI-to-PET translation that operates natively on the cortical manifold. A conditional Spherical U-shaped vision Transformer (SUiT) is specifically designed to model the intricate cross-modal relationships while preserving surface topology. It combines spherical convolutional encoders for multi-scale surface feature extraction with bottleneck Transformers to capture long-range spatial dependencies, while incorporating demographic and subcortical conditions to refine the synthesis. Evaluated on two datasets, including subjects with different dementia types, DB-SUiT demonstrates high-fidelity synthesis that substantially outperforms other baselines. In automated dementia classification, synthesized PET surfaces improve performance over MRI by 14.2% and PET volumes by 11.3%, approaching the performance of real PET surfaces. In a blinded reader study, synthetic PET achieved 85.5% diagnostic accuracy, compared with 75.8% for MRI and 95.2% for real PET. This further demonstrates cross-cohort and cross-pathology generalization, as the model was evaluated without retraining on an external cohort that included a dementia subtype not represented during training. Our code is available at this https URL. 

---
# Designing Against Deskilling: Metacognitive Feedback Reduces Cognitive Offloading to LLM Assistants 

**Authors**: Sebastian Maier, Kai Schwabe, Manuel Schneider, Stefan Feuerriegel  

**Link**: [PDF](https://arxiv.org/pdf/2609.20143)  

**Abstract**: Cognitive offloading to AI can reduce opportunities to practice skills, creating risks of deskilling. However, it remains unclear how to prevent deskilling without restricting access to AI. Here, we design two interventions to reduce offloading decisions: (1) metacognitive feedback that makes the implications of offloading for users explicit, and (2) an effort-based reward that incentivizes less extensive LLM assistance. We test both in a preregistered online experiment ($N = 704$) with a 2$\times$2 design and a no-AI control. The task was to practice fraction arithmetic with an LLM-based assistant that provided solutions only on explicit request, followed by an unaided test. Metacognitive feedback reduced answer offloading (OR $= 0.47$) and improved test performance (OR $= 1.51$). We found no evidence that the reward affected either outcome. Our results identify metacognitive feedback as a promising design choice to reduce cognitive offloading. 

---
# Cross-Modal Attention Acts as a Frequency Filter: Why Verbose Prompts Improve Robustness in Vision-Language Models 

**Authors**: Farooq Ahmad Wani, Maria Sofia Bucarelli, Mujtaba Hussain Mirza, Oleksandr Pryymak, Aryo Pradipta Gema, Iacopo Masi, Pasquale Minervini, Fabrizio Silvestri  

**Link**: [PDF](https://arxiv.org/pdf/2609.20139)  

**Abstract**: Vision-language models (VLMs) are fragile under image corruption. We find that the wording of the question affects VLMs in two opposite ways. Verbose questions make VLMs substantially more robust---e.g., rephrasing "Is there a cat?" into "Please look carefully and answer: is there a cat?". Conversely, VLMs become more fragile under corruption when the question is semantically complex or finer-grained, e.g., "what colour is the cup left of the chair?" instead of "is there a cup?". Both effects stem from question-conditioned cross-modal attention, which induces a spectral filter over image patches: verbose questions broaden its frequency support, while fine-grained questions concentrate it onto fewer visual scales. The model's answer drifts most when this filter and the corruption sit on the same spatial frequencies. We test the filter view on Qwen3-VL and LLaVA-OneVision across GQA and CLEVR; verbose paraphrasing reduces drift variance by 70--81% on the 8B models. The practical recipe---pad the prompt---further yields measurable gains in accuracy, even under image corruption. 

---
# AdaRepair-Mem: Adaptive Experience Orchestration for Repository-Level Program Repair 

**Authors**: Z. C. Luo, J. C. Guo, W. J. He, S. Y. Wang, J. C. Yu, F. M. Zhao, Y. Chen, T. Cao, L. Q. Liu, N. Zheng, W. Xu, J. Jiang, Z. M. Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2609.20130)  

**Abstract**: Recent memory-augmented repository-level program repair methods reuse historical repair experiences to improve LLM-based issue resolution. However, our analysis reveals three limitations in existing repository-level memory retrieval. First, episodic memory is highly imbalanced across repositories, leaving low-resource repositories with little effective support. Second, more memory does not monotonically lead to higher repair success, suggesting that relevance, quality, and redundancy matter more than raw memory volume. Third, memory accumulation is phase-misaligned: repositories may contain many reproduction experiences but few patch or refinement experiences. To address these problems, we propose an adaptive experience retrieval framework for repository-level program repair. Our framework introduces coverage-aware retrieval, which falls back to cross-repository or repair-type-based memories when same-repository memory is insufficient; quality-aware selection, which ranks memories by relevance, historical utility, specificity, and redundancy; and stage-aware routing, which separates and retrieves memories for reproduction, localization, patch generation, patch refinement, and validation. Evaluated on SWE-Bench-Lite and SWE-Bench-Verified, the proposed framework improves repair performance on under-covered repositories, reduces noisy memory retrieval, and better supports failed-to-fixed patch refinement. Our results show that the key to memory-augmented repair is not simply accumulating more experiences, but retrieving the right experiences for the right repair context. 

---
# Local Sparsity Enables Unsupervised LLM Safety Detection 

**Authors**: Xin Chen, Gil Kur, Alexander Shevchenko, Andreas Krause  

**Link**: [PDF](https://arxiv.org/pdf/2609.20129)  

**Abstract**: Deployment-time safety methods for large language models (LLMs) are predominantly supervised and assume access to unsafe training data. Nevertheless, new attacks and harm categories regularly arise, not captured by models trained in such a supervised fashion. An alternative approach is to view this problem through the lens of anomaly detection, namely, to rely solely on modeling safe data and flagging out-of-distribution inputs. However, LLM activations lie in a high-dimensional space, raising concerns about whether anomaly detection is statistically feasible. We show that, under the linear representation hypothesis (LRH), there may indeed be hope. In the LRH concept space, which is typically recovered via a sparse autoencoder (SAE), nearby points share a small common active support. Using this local sparsity insight, we propose a framework for locally masked SAE-based anomaly detection, supported by theoretical justifications. We validate it on various architectures and datasets, including both capability-testing datasets and safety-specific datasets. Finally, when we allow algorithms to use 1% out-of-distribution data for calibration, locally sparse methods achieve near-optimal performance, demonstrating their ability to capture meaningful safety information while using only 1-2% of SAE neurons for computation. 

---
# Multi-Dimensional Prosody Judgment For Live Streaming Speech Synthesis 

**Authors**: Zifan Guan, Longyu Lu, Junan Zhang, Zhizheng Wu, Meiguang Jin, Junfeng Ma  

**Link**: [PDF](https://arxiv.org/pdf/2609.20124)  

**Abstract**: Evaluating live streaming speech synthesis (TTS) requires assessing fine-grained, highly expressive prosody such as emotion, intonation, and energy which traditional MOS predictors fail to capture. While proprietary Large Language Models (LLMs) like Gemini can evaluate these aspects, they are too costly for massive inference and reinforcement learning feedback. To address this, we first introduce Live-ProsodyJudge (LPJ), a cost-effective pairwise evaluator distilled from Gemini into Qwen3-Omni. However, we identify a critical flaw in standard multi-dimensional evaluation: verdict coupling. The judge tends to lazily align all individual dimension scores with its overall preference, collapsing a rich multi-dimensional rubric into a single preference bit. To resolve this, we further propose Decoupled-Live-ProsodyJudge (D-LPJ). D-LPJ eliminates the overall verdict target to prevent blind following, masks uncertain pair-dimensions during Supervised Fine-Tuning(SFT), and introduces a novel span-local GRPO strategy that applies normalized advantages strictly to their corresponding rationale spans. Evaluated on highly curated human-annotated test sets, 10 sample balanced-order LPJ achieves higher point accuracy than a single Gemini call, while D-LPJ successfully produces independent,decoupled dimension judgments. Furthermore, in a Best-of-8 TTS candidate selection tournament, the LPJ-selected utterance falls within the human top-3 in 85.29% of high-confidence cases, demonstrating its efficacy for fine-grained TTS preference optimization. 

---
# A Scalable Trust Discovery Architecture for the Internet of Agents 

**Authors**: Song Zhang, Jiankang Yao, Hongtao Li, Xiaojun Zhang, Xugang Shen, Xin Li, Yanbiao Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.20095)  

**Abstract**: The Internet of Agents is expected to enable large numbers of autonomous agents to discover, verify, and collaborate with each other across heterogeneous platforms. However, current agent protocols mainly address tool invocation and inter-agent communication, leaving scalable agent registration, trustworthy identification, and capability-oriented discovery largely unresolved. To address this, this paper proposes a scalable trust discovery architecture for the Internet of Agents. The proposed architecture adopts a hierarchical and distributed design consisting of three layers: Agent Root for trusted registry governance, Agent Registry for agent registration and metadata publication, and Agent Resolver for distributed capability discovery and trust-aware resolution. The architecture further introduces a registry-suffix-anchored composite identity scheme, which binds an agent native identifier to a trusted registry suffix to generate a globally discoverable identity. It also incorporates a dual-certificate and multi-level authentication mechanism to strengthen identity trust among agents. We implement a prototype and evaluate it through large-scale agent registration and resolution experiments. The prototype achieves an average registration latency of 58ms and an average discovery latency of 25ms, and it supports more than 19,000 registration requests per second and more than 29,000 agent discovery requests per second. These results demonstrate the feasibility of the proposed architecture, providing a practical approach toward scalable and identity-trusted agent ecosystems in the Internet of Agents. 

---
# MATCH: Model-Aware Tool Learning with Curriculum Scheduling and Hierarchically Gated Rewards 

**Authors**: Shihao Liu, Hao Yin, Lijun Liu, Zhengzong Chen, Yuanyuan Zhao, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.20082)  

**Abstract**: Tool learning enables large language models (LLMs) to use external tools for tasks beyond parametric knowledge. Reinforcement learning can optimize tool-call behavior from feedback, but current methods still face two problems: fixed-threshold curricula can become misaligned with the policy's evolving capability boundary, and additive rewards can leak argument-level credit when the predicted tool is wrong. To address these problems, we propose MATCH, a closed-loop framework for model-aware tool learning with curriculum scheduling and hierarchically gated rewards. Model-Aware Curriculum Learning (MACL) maintains reward-derived sample difficulty that co-evolves with the policy, and each epoch selects samples near the current capability boundary together with a top-k pool of harder cases. Hierarchical Tool-call Gated Reward (HTGR) scores tool name, argument key, and argument value as a gated chain, granting credit at each level only when prerequisites hold. The same HTGR rewards drive both GRPO updates and MACL's difficulty refresh, closing the loop between policy optimization and sample scheduling. On API-Bank and BFCL V3, MATCH reaches 72.19% and 62.87% overall accuracy, outperforming the main supervised and RL-based baselines. Backbone experiments further show consistent improvements across four backbones from two model families. 

---
# Reading Emotions in the Token Space: Discriminative Adaptation of SpeechLLMs for Emotion Recognition 

**Authors**: Hasindri Watawana, Sergio Burdisso, Esaú Villatoro-Tello, Manjunath K E, Kadri Hacioglu, Petr Motlicek, Andreas Stolcke  

**Link**: [PDF](https://arxiv.org/pdf/2609.20081)  

**Abstract**: SpeechLLMs have shown strong potential for emotion recognition, yet they read the predicted emotion off a generative decoder not suited for classification: it can emit labels outside the target set and favors frequent classes. We propose a discriminative adaptation that reads the final prompt token's hidden state through a classification head, producing a label in one forward pass without modifying the backbone. Because this readout starts from the hidden state the model would otherwise decode, it gives a controlled comparison of generative and discriminative inference in an otherwise identical speechLLM. We keep the head a single linear layer, trading little accuracy for interpretability: each emotion becomes one direction in the LLM output token space, revealing associated tokens. On IEMOCAP, across two speechLLM architectures, it improves Macro F1 and removes hallucinations, with largest gains on realistic ASR transcripts. Our analysis reveals that these emotion directions encode indirect associations mirroring biases in web-scale text. 

---
# PointEvent: Rethinking Event-based Tiny Object Detection via Serialized Motion Evidence Accumulation 

**Authors**: Zongze Wu, Baofeng Jia, Weiqi Yan, Jingyuan Zhang, Yu Zang, Xiaoyu Chen, Jing Han  

**Link**: [PDF](https://arxiv.org/pdf/2609.20066)  

**Abstract**: Event cameras offer high temporal resolution and motion sensitivity for tiny UAV detection, yet distant targets generate sparse and fragmented events that are easily overwhelmed by clutter and ego-motion. Existing methods mainly rely on dense event representations or local sparse spatiotemporal modeling, resulting in redundant computation or fragmented modeling of motion continuity across distant asynchronous events. To address this limitation, we introduce serialized motion evidence accumulation, which treats motion continuity as an ordered evidence propagation process. Specifically, the same event stream is organized into locality-preserving spatiotemporal paths and chronology-preserving temporal paths through the latent complementary serializations. Based on this principle, we propose PointEvent, a lightweight event-wise state-space framework that alternates serialized scans across the complementary orders, progressively consolidating fragmented motion evidence beyond fixed local neighborhoods. A high-resolution event branch preserves fine-grained target responses, while compact context modulation suppresses interference. Experiments demonstrate that PointEvent achieves SOTA with the fewest parameters and fastest measured inference among the compared methods. Code: this https URL 

---
# Robust Workflow Generation via Adversarial Learning for Audio Deepfake Detection 

**Authors**: Xiang Li, Pin-Yu Chen, Wenqi Wei  

**Link**: [PDF](https://arxiv.org/pdf/2609.20063)  

**Abstract**: The rapid advancement of speech synthesis and voice conversion technologies has made audio deepfakes increasingly realistic, posing serious security risks in practical applications. While existing detection methods achieve strong performance under controlled conditions, they often fail to generalize under real-world perturbations and corruptions. In this paper, we propose ROGUE, a framework that dynamically constructs robust detection workflows by orchestrating multiple detection tools. ROGUE formulates workflow generation as a sequential decision-making problem and introduces a dual-agent paradigm, where a perturbation agent generates audio perturbations and a policy agent learns to select and execute detection tools under perturbed conditions. Through adversarial learning, ROGUE enables perturbation-aware tool selection, adaptive execution strategies, and improved robustness to distribution shifts. Extensive experiments across multiple datasets and real-world corruptions demonstrate that ROGUE consistently outperforms strong baselines in both robustness and generalization. Our results highlight the effectiveness of adversarially optimized workflow generation for building reliable audio deepfake detection systems in real-world deployment settings. 

---
# AI Should Facilitate Democratic Deliberation at Scale 

**Authors**: José Ramón Enríquez, Jiaxin Pei, Alex Pentland  

**Link**: [PDF](https://arxiv.org/pdf/2609.20059)  

**Abstract**: AI systems can strengthen democracy by supporting deliberation at scale by addressing cognitive, social, platform-design, and market-driven frictions, while preserving human agency. Unlike proposals such as liquid democracy that restructure representation through vote delegation, in this position paper, we argue that AI-assisted deliberation offers a more promising path by lowering barriers to meaningful engagement without substituting machine judgment for human choice. Drawing on evidence from online deliberation platforms and experimental research, we identify four guiding principles: preserving agency and autonomy, encouraging mutual respect, promoting equality and inclusiveness, and augmenting rather than substituting active citizenship. We also address critical challenges, including alignment, sycophancy, training bias, and over-reliance on AI systems. We call on the machine learning community to develop deliberation-focused AI systems evaluated not on engagement metrics but on their capacity to facilitate informed, representative, and friction-robust discourse. 

---
# The Missing Complement: State-Conditioned Minimal Sufficient Evidence for Coding Agents 

**Authors**: Zhexi Feng, Ruiyi Zhang, Yongbo Yang, Pengtao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2609.20050)  

**Abstract**: A coding agent halfway through an issue has already read much of what a retriever ranks highest. Relevance is scored per passage, but sufficiency belongs to the set: a ranker can fill its budget with variants of one required fact and leave the decision unsupported. We formulate state-conditioned minimal sufficient evidence recovery: given a captured agent state, recover a compact evidence combination that supplies the support its next decision still lacks. SERBench measures this on 500 held-out states from 45 repositories, recording what the agent has seen and crediting only sets that cover every fact the current decision was annotated to require. MSS-Complement treats acquisition as set construction, not ranking. Three semantic calls propose a jointly sufficient set, search for what it lacks, and return 4-8 intact source units within 6,144 tokens. One configuration, fixed on calibration data, recovers a complete set for 73.0% of those states at five items and 80.6% at eight, against 61.4% and 72.4% for Qwen3 embedding with reranking. A matched control ranking by similarity alone reaches 66.6%, placing the gain in the set-level policy, not the computation. From frozen repository source with no gold-derived pool, the lead is 5.0 points. On AMA-Bench it answers from a 76.2% smaller answer prompt, with accuracy 2.08 points above that benchmark's own memory agent. Removing one required group from an otherwise complete set costs 12.3 and 11.1 points of repair-localization precision under two executors. Retrieval for agents is better posed as recovering what a decision lacks than re-ranking what an issue resembles. 

---
# Correct Now, Insufficient Later: Auditing Update Sufficiency in Context Compression 

**Authors**: Guangzhe Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.20045)  

**Abstract**: A memory can answer a current query correctly while discarding distinctions required by a later update. We investigate this failure with a paired-history audit: two histories have the same current answer, receive a shared future update, and require different subsequent answers. A pilot evaluates 24 history pairs across six synthetic mechanisms, 12 memory conditions, two repeats, and two model backends. A deterministic frontier selector obtains strict reveal accuracy of 96/96 on DeepSeek and 82/96 on GLM; a structured writer obtains 62 successes with one unresolved outcome and 56/96. The configured four-outcome joint contrast has finite-sample identification intervals of [0.521, 0.542] and [0.292, 0.313], not confidence intervals. A record-level audit distinguishes retained-state adequacy, response delivery, and answer-schema compliance without changing those original scores. It finds 26 and 25 well-formed but semantically wrong structured reveal memories, while all 14 GLM frontier reveal failures contain correct values in the wrong wrapper. Tombstone removal produces 16/16 exact replay failures in the targeted mechanism. Identifier renaming then exposes a separate flaw: original frontier late-reference adequacy falls from 8/8 to 94/320 transformed instances. We provide and test a label-equivariant repair, but it preserves only 2/8 original late-reference answers: eliminating a naming shortcut does not solve unknown future relevance. These results support a scoped evaluation methodology and reproducible failure analysis, not general superiority of the repaired algorithm. Paid pilot evidence, retrospective diagnostics, and new offline tests are reported separately; no independent held-out or natural-task validation is claimed. 

---
# Astronex-World 1.0: Real-Time Interactive World Model Foundation 

**Authors**: Xin Zhou, Cong Miao  

**Link**: [PDF](https://arxiv.org/pdf/2609.20034)  

**Abstract**: We present Astronex-World 1.0, an open controllable video world-model foundation. Given a text prompt (text-to-video) or an initial observation (image-to-video), the model predicts future visual states under frame-aligned camera trajectories, continuous actions, and an embodiment identifier, and accepts text events inserted at a specified position of a rollout. The family provides a bidirectional model for full-context generation and a causal model with block-causal attention and cross-block KV caching for persistent generation, both built on the Wan2.2-TI2V-5B prior. PRoPE injects camera intrinsics and extrinsics, while a 64-dimensional action stream modulates every Transformer layer. A five-stage training path develops bidirectional camera and action control, converts the backbone to block-causal generation, distills a few-step student, restores mixed-domain dynamics, and applies asymmetric DMD/DMD2 distribution matching. The causal model generates 832x480 video at 24 fps. All five training stages run on two NVIDIA L20 48 GB GPUs, and the causal model streams in real time on one. It scores 73.5 on WBench Navi and 70.0 on WBench Full. On Full, this 5B model is above the 13.6B LongCat-Video and the 14B Helios, within one point of the 22B LTX-2.3, and above YUME 1.5, which is post-trained from the same 5B prior on NVIDIA A100 GPUs. The reserved action input and output interfaces allow post-training for embodied intelligence and autonomous driving. 

---
# Governance-as-Code: Translating EU AI Act Technical Requirements into Executable Compliance Pipelines for Generative AI Systems 

**Authors**: Rudrendu Kumar Paul, Sourav Nandy  

**Link**: [PDF](https://arxiv.org/pdf/2609.20016)  

**Abstract**: The EU AI Act (Regulation 2024/1689) imposes technical obligations on high-risk AI providers, yet Articles 8-15 were drafted for predictive AI and leave seven technical gaps when applied to generative systems, spanning non-deterministic data governance, training-data provenance, continuous conformity, human oversight, open-ended robustness, emergent risk, and generative fairness. We deliver Governance-as-Code (GaC), a framework of 43 machine-checkable acceptance criteria across six compliance modules that run in a CI/CD pipeline and emit Article-indexed audit evidence, and we show the actual Rego policy code rather than merely describing it. Our central commitment is that the Act's open-textured standards ("appropriate levels," "possible biases") become declared, auditable numbers: robustness thresholds are derived from the provider's documented baseline and a state-of-the-art floor, and framing bias is collapsed into eight measurable proxies tested by counterfactual demographic probing. We also correct who owes what, since under Article 25 and Chapter V a downstream deployer relies on the upstream provider's Article 53 training-data summary and documents only the layers it controls, so GaC verifies that summary rather than demanding per-sample documentation the deployer never had. We validate on two enterprise deployments, a high-risk advisory chatbot and a limited-risk content generator, benchmarking against a manual expert audit rather than documentation artifacts that were never designed to enforce compliance. GaC reproduces all of the manual audit's findings, including three penalty-triggering violations, while cutting audit labor by roughly 75%. 

---
# Dynamic Generalized Gromov-Wasserstein Optimal Transport 

**Authors**: Junda Ying, Zhiwei Zeng, Peijie Zhou, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.20008)  

**Abstract**: Gromov--Wasserstein optimal transport (GW-OT) extends classical optimal transport by introducing structure-aware transport cost. This is particularly relevant for spatial transcriptomics, where dynamical reconstruction should preserve tissue structure in addition to matching expression patterns. While static formulations have been widely used for such structure-aware alignment, a general dynamic formulation for reconstructing continuous trajectories is still missing. We introduce Travelling Pair Dynamical Alignment and Trajectory Estimation (TP-DATE), a theoretical and computational framework to generalize GW-OT dynamically in a simulation-free manner. We formulate a broad class of static and dynamic Quadratic-form OT (QOT) through path actions and prove the static dynamic equivalence. We further develop travelling-pair flow matching, which allows interacting conditional paths and marginalizes their interactions into a single vector field. On synthetic and real spatial transcriptomics data, TP-DATE better preserves spatial structure and improves continuous 3D dynamics reconstruction. 

---
# EPIG-Tree: Compute-Optimal Branching for Gradient-Efficient Reinforcement Learning 

**Authors**: Nikita Khomich, Leopold Hermansson, Ido Hakimi  

**Link**: [PDF](https://arxiv.org/pdf/2609.20004)  

**Abstract**: Reward-based reinforcement learning for language models, exemplified by Group Relative Policy Optimization (GRPO), collapses an entire stochastic trajectory into a single scalar reward. This is clean and scalable, but it explores and allocates reward inefficiently: a trajectory may contain many causal decisions, recovery attempts, and environment-randomness events, yet every token or action inherits one trajectory-level advantage. We study tree-based rollout construction as a compute-allocation problem for policy-gradient estimation. Our central claim is that branches should be placed not where the policy is merely uncertain, but where an additional branch most reduces uncertainty about the policy gradient per unit of compute. From a law-of-total-variance decomposition of the local policy-gradient random variable, we derive two allocation laws: new branches reduce decision uncertainty, while repeated suffix rollouts reduce continuation uncertainty. The resulting EPIG-Tree score allocates branches using the already computed rollouts. It estimates occupancy- and score-weighted value uncertainty, along with a suffix law $n_e \propto w_e \|\nabla_\theta \log \pi(a_e|h_e)\| \sigma_e / \sqrt{c_e}$. Empirically, EPIG reduces gradient MSE in cloned-state control, winning in all nine dense continuous-control environments of a 13-environment sweep and recovering the reference gradient direction near-perfectly, and it improves frozen-LLM gradient calibration relative to entropy branching. In online single-turn math, tree-local credit beats flat GRPO, while branch placement is secondary to token-level credit assignment. In online multi-turn Wordle, EPIG attains the highest final win rate (0.850), overtaking flat GRPO, which saturates early at 0.790, and entropy branching as training proceeds, confirming that the gradient-estimation advantage transfers to a stateful, large-action setting. 

---
# AVTrace: Diagnosing Audio-Visual Temporal Reasoning in Omni Models 

**Authors**: Longyin Zhang, Parth Sakhare Mahendra, Chengwei Wei, Ning Zhang, Lim Ming Chong, Sirui He, Ai Ti Aw  

**Link**: [PDF](https://arxiv.org/pdf/2609.19991)  

**Abstract**: Omni models can describe video content, but can they locate events in time, preserve event order, and judge audio-visual synchronization? We introduce AVTrace (Audio-Visual Temporal Reasoning Assessment and Capability Evaluation), a silver-standard diagnostic suite spanning onset and span grounding, synchronization, next-step prediction, cross-modal localization, chain parsing, and event-conditioned comprehension. It contains 34,114 training examples and category-balanced development and test splits of 3,500 and 7,000 examples. We evaluate five open omni models under their respective input configurations using reference-blind response normalization followed by deterministic scoring. All five off-the-shelf systems score below the test split's majority-label baseline of 0.556 on synchronization verification, and obtain low scores on chain parsing and event-conditioned grounding and comprehension. Development-set perturbations reveal task-dependent sensitivity in Qwen3-Omni-30B to modality removal and changes in visual input processing, without isolating their underlying causes. Parameter-efficient temporal post-training improves Gemma4-E4B-it on several benchmark metrics. On three external image benchmarks, task metrics change modestly, including some degradations, while teacher-forcing perplexity decreases. Together, these findings show that semantic reference-text overlap should not be treated as a proxy for temporal localization, and that AVTrace can identify task-specific weaknesses while providing a testbed for temporal post-training. 

---
# Past, Future, All at Once: Mitigating Stability-Plasticity Dilemma via Post-hoc JANUS Rectification 

**Authors**: Zhilong Zheng, Letian Tao, Yang Guan, Yujie Yang, Wei Xiong, Kehua Sheng, Bo Zhang, Jingliang Duan, Keqiang Li, Shengbo Eben Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.19985)  

**Abstract**: Fine-tuning foundation models on new tasks inevitably suffer from catastrophic forgetting. While existing works attempt to mitigate this on the basis of parameter-efficient fine-tuning methods, they adopted an overly restrictive Subspace Orthogonality condition. In this paper, we introduce a purely post-hoc and tuning-agnostic weight rectification framework that achieves Parameter Space Orthogonality, which is the necessary and sufficient condition for preserving historical performance to the first order. By projecting parameter updates into the JAcobian NUll Space (JANUS), our method significantly recovers compromised historical knowledge without interfering with the underlying fine-tuning process. To overcome the local validity of the Jacobian approximation, we further propose a Multi-step Adaptive Rectification mechanism that utilizes the JANUS shift to dynamically verify the valid trust region and adjust step sizes. Coupled with our proposed ghost projection, ghost orientation comparison, and sequence-level singular value decomposition compression techniques, JANUS also achieves great temporal and spatial efficiency. Experiments demonstrate that JANUS seamlessly integrates with various fine-tuning methods, significantly mitigating the stability-plasticity dilemma by recovering historical knowledge while preserving downstream task adaptation. 

---
# MaskHarness-WAM: Instance-Grounded Harnessing for Long-Horizon Robot Manipulation 

**Authors**: Zitai Huang, Taiyi Su, Jian Zhu, Jianjun Zhang, Chong Ma, Tianbin Liu, Weiyi Lu, Yi Xu, Hanli Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.19974)  

**Abstract**: Long-horizon robot manipulation requires not only stable local visuomotor control, but also continuous target tracking and reliable task progress assessment throughout execution. This challenge becomes particularly critical when multiple objects share identical appearances and must be manipulated in a prescribed order. In such scenarios, relying solely on a limited-horizon manipulation policy is often insufficient to determine which instance should be operated on and when the task should transition to the next stage. To address this challenge, we propose MaskHarness-WAM, an instance-grounded harness for long-horizon manipulation. The proposed system connects high-level task planning with low-level manipulation policies through target masks, while leveraging visual feedback for subtask scheduling and continuous execution. Since each subtask corresponds to a different target instance, the low-level policy requires a newly established initial target mask under the updated scene at each subtask transition. The harness continuously re-observes the environment, generates, and verifies the target mask at subtask boundaries, thereby updating the instance-level spatial condition provided to the low-level policy. Furthermore, the system advances the manipulation process by switching target instances according to the verified completion status of each subtask. Experiments on a real robot platform demonstrate that MaskHarness-WAM substantially outperforms limited-horizon policies on sequential multi-object manipulation, showing its effectiveness in extending local manipulation skills to reliable long-horizon execution. 

---
# Efficiently Distributed Federated Learning 

**Authors**: Gianluca Mittone, Robert Birke, Marco Aldinucci  

**Link**: [PDF](https://arxiv.org/pdf/2609.19972)  

**Abstract**: Federated Learning (FL) is experiencing a substantial research interest, with many frameworks being developed to allow practitioners to build federations easily and quickly. Most of these efforts do not consider two main aspects that are key to Machine Learning (ML) software: customizability and performance. This research addresses these issues by implementing an open-source FL framework named FastFederatedLearning (FFL). FFL is implemented in C/C++, focusing on code performance, and allows the user to specify any communication graph between clients and servers involved in the federation, ensuring customizability. FFL is tested against Intel OpenFL, achieving consistent speedups over different computational platforms (x86-64, ARM-v8, RISC-V), ranging from 2.5x and 3.69x. We aim to wrap FFL with a Python interface to ease its use and implement a middleware for different communication backends to be used. We aim to build dynamic federations in which relations between clients and servers are not static, giving life to an environment where federations can be seen as long-time evolving structures and exploited as services. 

---
# KoNeoBench: A Curated Evaluation Dataset for LLM Understanding of Korean Neologisms 

**Authors**: Soha Lee, Soojin Lee, Heesung Yang, Hyunju Song, Hyunji Lee, Jinsan An, Jeongwan Shin, Jin Hyun Park, Jun Lee, Hyeyoung Park, Kilim Nam  

**Link**: [PDF](https://arxiv.org/pdf/2609.19916)  

**Abstract**: Large language models (LLMs) are typically evaluated on static benchmarks, even though natural language constantly evolves through newly emerging words and meanings. Existing Korean benchmarks are centered on established vocabulary and therefore provide limited coverage of such recent lexical change, and their English-oriented design makes it difficult to assess the typological properties of Korean, in which content words combine productively with functional morphemes. In this paper, we introduce KoNeoBench, a benchmark for evaluating LLMs' understanding of Korean neologisms. KoNeoBench is built on 1,785 Korean neologisms attested in online news since 2020 and curated through expert lexicographic review. Each entry provides usage examples, word-formation analyses, and dictionary-style definitions. Based on this resource, we define four tasks and report results on recent models, together with a human baseline. Our experiments show that current LLMs exhibit clear limitations in recovering source components, distinguishing semantic categories, and generating accurate definitions. These results reveal specific aspects of recent Korean lexical change that remain challenging for current LLMs. KoNeoBench is available at this https URL . 

---
# Learning and Transferring Closed-Loop Robot Software 

**Authors**: So Kuroki, Yujin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2609.19906)  

**Abstract**: Closed-loop robot policies require observation processing, state management, and situation-dependent branching, making them costly to design and tune manually. Although coding agents increasingly support control-code generation and optimization, it remains unclear whether implementations improved on source tasks also support policy acquisition for new tasks. We study this question by treating complete closed-loop implementations as reusable execution experience. For each source task, a coding agent generates policy code from a few successful demonstrations and iteratively improves it using simulation feedback. The validation-selected implementations are retained in a software archive. For new tasks, the agent generates and improves policies using archived implementations, target demonstrations, and execution feedback. The resulting policy is then frozen and executes without further model calls. Across four source tasks in RoboCasa, iterative optimization increases mean success from 28.3% to 64.2%. Across nine target tasks and three independent runs, mean success is 45.2% without references, 41.5% with initial source code, and 57.0% with optimized source code. Optimized references outperform initial references in all three runs on the nine-task average, with a mean gain of 15.6 percentage points. These results demonstrate the value of execution-improved software as a resource for acquiring new policies in this setting, although initial references remain better on two target tasks when averaged across runs. 

---
# ClashBench: Conflicts Leading Agents to Seize and Harm 

**Authors**: Yuejin Xie, Yu Li, Dadi Guo, Qingyu Liu, Yuqian Fu, Yanwei Fu, Yujiu Yang, Xia Hu, Dongrui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.19892)  

**Abstract**: As agent systems become more widely used, multiple agent sessions increasingly run alongside pre-existing user tasks in the same environment, sharing resources with limited capacity or mutually exclusive states. This creates a safety risk: when granted sufficient privileges, an agent may resolve a resource conflict by terminating or otherwise disrupting an existing task rather than reporting it. In this work, we identify and formalize this failure mode, which we term destructive resource preemption: obtaining the resources required for a requested task by terminating, overwriting, evicting, or degrading an incumbent task. To systematically study this risk, we introduce ClashBench, an executable benchmark comprising 268 validated conflict cases across 55 resource types, and evaluate 17 models through Codex, Claude Code, and OpenCode. We observe destructive preemption in 44.5% of trajectories, where the agent completes the requested task while causing the incumbent task to fail its health check. We also show that prompt-based safeguards are insufficient: an instruction to avoid affecting existing tasks reduces but does not eliminate preemption, while an instruction explicitly authorizing the agent to stop local processes increases it. More concerningly, in 31.9% of successful destructive-preemption cases, the final response mentions neither the resource conflict nor the action taken to resolve it, raising concerns about possible concealment. These findings establish destructive resource preemption as a broad safety risk in privileged agent systems and motivate stronger privilege controls, task isolation, and conflict-aware safeguards. 

---
# PetriBench: Benchmarking LLM Reasoning over Dynamic State Spaces 

**Authors**: Pyrros Koussios, Benjamin Jäger, John Hua Yao, Ajay Sridhar, Violet Xiang, Chenhao Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.19883)  

**Abstract**: Characterizing LLM reasoning remains an open challenge, as many existing benchmarks isolate specific reasoning skills, rely on external knowledge, or are costly to extend. We introduce PetriBench, a compact, fully self-contained, and scalable benchmark for evaluating LLM reasoning over dynamic state spaces using Petri nets, a mature formalism for modeling real-world concurrent and distributed systems. PetriBench organizes reasoning into four task families varying by scope and temporal horizon, with Easy, Medium, and Hard levels generated by increasing structural complexity and evaluated against exact ground truth. Across a diverse set of proprietary and open-weight models, accuracy decreases consistently with difficulty, while harder instances expose increasingly distinct task-specific capability profiles. Additional analyses show that test-time compute improves performance but interacts differently with different reasoning tasks, and that procedural generation yields smooth scaling with structural complexity. Together, these results show that PetriBench provides a unified and extensible setting for probing the strengths, limits, and scaling behavior of LLM reasoning. 

---
# Zarya: A Hybrid Autoregressive--Masked Diffusion Language Model with Flexible Training and Dual-Mode Inference 

**Authors**: Leonid Sinev, Ilya Koziev, Vladislav Leshchuk  

**Link**: [PDF](https://arxiv.org/pdf/2609.19868)  

**Abstract**: Autoregressive language models (ARMs) are constrained by sequential, left-to-right generation, while masked diffusion models (MDMs) enable parallel decoding but suffer from high computational overhead due to the inability to reuse Key-Value (KV) cache and from incoherent generation arising from learning dependencies over an intractable space of token combinations. We introduce Zarya, a family of hybrid language models that jointly optimizes an autoregressive (AR) objective and a masked-diffusion objective within a single architecture. Zarya structures training data into variable-size slots and employs a curriculum that gradually increases slot granularity, enabling a smooth transition from fine-grained AR learning to coarse-grained diffusion learning. At inference, Zarya provides two distinct decoding paradigms through a unified interface: (i) MDM sampling with first-hitting denoising, and (ii) slotted speculative decoding that interleaves inter-slot diffusion-based selection with intra-slot autoregressive infilling, achieving full KV cache reuse. The training and inference regimes are fully decoupled, allowing a model trained with any configuration to be deployed in either mode. Extensive configurability --- including grouped noise patterns (Prefix Completion, Fill-In-the-Prefix, Fill-In-the-Middle), ordered sampling schedules, and noise-level permutation strategies --- enables flexible research exploration. We release Zarya models publicly in sizes 0.6B, 1.7B, and 4B, demonstrating performance on standard benchmarks while offering a principled integration of autoregressive and diffusion paradigms. 

---
# A Functional Pilot for Certified Freshness-Aware Semantic--Spatial Range Retrieval 

**Authors**: Taimoor Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2609.19855)  

**Abstract**: Geographic applications need every object inside a radius that satisfies a semantic threshold, yet embedding indexes return approximate top-ranked lists and may omit qualifying records silently. We present FRESH-GEORANGE, a semantic- spatial range design that separates source-watermark freshness from optional record age. Geographic cells and semantic mi- croblocks provide admissible pruning bounds; a graph proposes verification order but supplies no correctness evidence. Exact mode scans every nonprunable block and the delta overlay. Certified mode may stop early and reports a deterministic query- specific recall lower bound from verified answers and unresolved records. A reproducible CPU pilot uses 2,500 real OpenFlights airport records, a 2,000-record base, and 740 simulated insert, delete, and text-revision events; it evaluates 180 unique queries over five seeds. Exact mode achieved 100.00% set recall on every query. The 95-percent mode achieved 99.91% empirical mean recall with a 99.41% reported mean certificate and no observed bound violation. However, its 7.24 ms median latency was 5.85 times the 1.24 ms spatial-first exact baseline, and full-history delta replay became slower than rebuilding at larger batches. The prototype therefore validates the completeness mechanism, not performance superiority or production freshness. Submission- scale evaluation requires real map diffs, official recent baselines, and truly incremental versioned maintenance. 

---
# PACE: Precise AI Cinematic Expression: A Typed Specification for Script-Grounded Previsualization and Geometric Conformance 

**Authors**: Bing Duan, Qiang Guo, Linpu Li, Zhijian Mao, Min Zhu, Zhirui Ren, Yiwei Yan, Xi Chu, Xiaoding Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.19853)  

**Abstract**: Between a screenplay and a film sits a planning problem that is spatial first: who stands where, and what a camera sees from where it stands. An image diffusion model asked for a shot in free text settles that plan by its own defaults. We present PACE (Precise AI Cinematic Expression), a typed representation for the plan: the screenplay evidence, the characters, props and locations it needs, where each subject stands, and what the camera does. A value is written once at the level it belongs to (script, scene, shot or panel) and inherited below it. A compiler turns the result into both the prompt sent to the diffusion model and a 3D scene built in metres, and a camera solver places the camera so that the declared framing is the framing built. Where a declared value becomes geometry, PACE measures, field by field, how far the compiled camera and the staged render sit from the declaration, rather than asking a model to judge.
On the 11-scene Automatic Drive screenplay, every staged single-subject panel places its subject within 1.2% of frame width of its declared position; with two or three subjects one camera pose cannot satisfy every position, and the residual is reported rather than absorbed. On 204 external director-storyboard shots, delivered head height is 1.906 times the staged target from the director's words, 1.733 from the compiled prompt, and 0.955 with the greybox control; the condition that holds framing best draws the described action least. Declaring the pose on 30 shots raises the action drawn from 58.9% to 74.4% without moving the framing. Transitions, fitted motion and human review of the generated panels remain open. Code: this https URL 

---
# Improving Cross-embodiment Transfer in Latent Action Models with Action-Similarity Supervision 

**Authors**: Maxime Alvarez, Renzo Caballero, Tatsuya Matsushima, Yusuke Iwasawa, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2609.19846)  

**Abstract**: As generalist robot policies gain vision and language from web-scale pretraining, demonstrations remain costly to collect and tied to the robot that recorded them. Latent action models (LAMs) address both by learning latent actions from action-free videos that can be shared across embodiments, however, in practice, LAMs are sensitive to background visual noise, and the same motion from two different robots may be encoded with different latents. One solution to the background visual noise is to add an auxiliary loss predicting the robot action from the latent action, further associating the latent action space to the embodiment specific robot action space. We study a different use of the same labels, through action-similarity supervision. The similarity between any two latent actions is trained to match the similarity of the two ground-truth robot action sequences. The ground-truth actions are never predicted by the LAM, so the latent action does not need to encode embodiment specifics. We evaluate cross-embodiment transfer on RoboTwin 2.0 in a controlled setup, two bimanual robots demonstrate disjoint task sets, a policy is trained on all the demonstrations, and each robot is evaluated closed-loop on the tasks only the other demonstrated. With the policy architecture and its hyperparameters, the dataset, and the evaluation protocol fixed, predicting latent actions instead of ground-truth actions more than doubles cross-embodiment success. Given the same ground-truth actions, similarity supervision transfers better than an auxiliary loss that predicts the ground-truth action during the LAM training. Computing the similarities on end-effector motion rather than joint-space motion, and letting the loss compare latent actions across the two robots, gives the best approach of the study. 

---
# Trust, but Validate the Instrument: Auditing AI-Generated RTL Verification Plans on Authored Security-Regression Proxies 

**Authors**: Hang Xiao, Chuhong Xu, Kainan Zhou, Gangzhen Qian, Lu Yi  

**Link**: [PDF](https://arxiv.org/pdf/2609.19844)  

**Abstract**: AI-generated RTL verification plans can satisfy a provider schema yet fail at the boundary to trusted execution. We present SecTB-RTL, an auditable framework covering 31 tasks and 124 authored hardware-security regressions. A deterministic non-AI baseline killed 36, 75, and 78 mutants at increasing resource limits. The first confirmatory run (C1-R2) failed before model execution because the provider rejected its response schema. After a schema-only repair made without viewing outcomes, a separately frozen follow-up run (C1-R3) completed 1,860 calls. The provider accepted 1,857 responses, but only nine passed the production semantic validator. The generation and execution rules did not match. We therefore preserve the run as an instrument-validation incident and report no prompt-effect estimate. This incident shows that provider or schema acceptance does not establish execution validity. Compilation and coverage are only diagnostics; the exact saved artifact must pass the full production path. A subsequent follow-up is excluded because it did not satisfy the preregistered evidence-completeness gate and is treated only as future work. We release the benchmark, failure-preserving contract, incident provenance, and governance controls needed to prevent infrastructure behavior from being misreported as model behavior. 

---
# Reproducing Transparent and Scrutable Recommendations: Exploring Open-Weight Models via Natural-Language User Profiles 

**Authors**: Noah Mamié, Laurin van den Bergh  

**Link**: [PDF](https://arxiv.org/pdf/2609.19831)  

**Abstract**: In this reproducibility study, we investigate the transparency and scrutability of recommender systems enhanced by incorporating generated natural-language user profiles that represent user preferences. The original paper explores the synthesis of user profiles from raw user-generated review text across domains such as movies and accommodations (Amazon Movies & TV, TripAdvisor). Crucially, these natural-language user profiles enable direct user interaction and intervention, allowing users to customize recommendations by correcting misattributed preferences or addressing cold-start settings. We successfully reproduce the core findings of the original study. Additionally, we extend the evaluation by conducting systematic context ablation experiments, multi-seed stability across five distinct random seeds to establish statistical reliability, and a mechanistic interpretability analysis using the nnsight framework to probe internal model representations under counterfactual profile perturbations. Our findings verify the original paper's claim that User Profile Recommendation (UPR) achieves competitive performance under its test-set reranking protocol and makes recommendations more transparent. Perturbing the natural-language profiles does change predictions, but it shifts predicted ratings uniformly across genres with no detectable genre-selective effect, leaving rankings unchanged even under direct activation steering. We trace this back to the rating-regression objective rather than the profile interface, with ranking-objective models clearly exceeding in this task. 

---
# CoRELoop: Parameter-Efficient Controlled Recurrent Refinement for Audio Deepfake Detection 

**Authors**: Kunyu Feng, Yuxiang Wang, Li Wang, Wan Lin, Zhizheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.19818)  

**Abstract**: Generalizing to unseen attacks remains challenging for audio deepfake detectors, and collecting training data covering all potential attacks is impractical. We explore recurrent refinement in an already-trained SSL-based detector without additional data or changes to its original parameters. However, directly recycling encoder outputs as inputs degrades detection in our diagnostic. We propose CoReLoop, which makes this reuse effective by adapting recurrent inputs to the frozen encoder, controlling state updates, and aligning refined outputs with the frozen classifier. By training only lightweight refinement modules and loop-specific low-rank adapters on the original data, CoReLoop enables additional refinement while preserving the detector's original first-pass prediction. On 14 cross-domain test sets, the 24-layer model reduces pooled equal error rate (EER) from 4.85% to 3.74% with two passes, with approximately 10M trainable parameters out of 598M. To selectively apply this refinement, an optional halting head chooses the depth for each utterance, achieving 3.73% pooled EER with an average of 1.18 passes. 

---
# Long-horizon autoformalization of a core theorem underlying MIP* = RE 

**Authors**: Sirui Lu, Ruixuan Deng, Yanqiao Zhu, Zhengfeng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2609.19814)  

**Abstract**: Landmark mathematical formalizations have taken specialist teams years to complete. We present FormalFlow, a system that coordinates AI proving agents under human supervision to address statement drift and proof composition in long-horizon formalization. Drawing on software engineering principles and practices, it uses a shared blueprint to guide nested planning, proving and review loops. Agents strengthen verification and review throughout formalization. We completed a machine-checked Lean 4 proof of the quantum soundness of the classical low individual-degree test, a core theorem underlying MIP* = RE. Developing the proof took 63 days; greater parallelism could further reduce this time. The final library contains 126,367 lines of Lean code, all generated by agents. The formalization corrects side conditions and intermediate errors while preserving the published final error bound under corrected assumptions. This work provides a verified foundation for quantum complexity and demonstrates a route to affordable verification of major research proofs by small teams. 

---
# Evolution or Illusion? Rethinking Evaluation in LLM Evolutionary Search 

**Authors**: Tal Oved, Roi Pony, Oshri Naparstek, Udi Barzelay  

**Link**: [PDF](https://arxiv.org/pdf/2609.19799)  

**Abstract**: LLM-driven evolutionary search finds programs by launching seeds and iterating each one. Papers report a single budget setting, usually one seed run for a fixed number of iterations, and rank methods from that one point. We show this is not enough. We evaluate three evolutionary search strategies on five optimization tasks, commonly used by papers in the genre to report results. We run the analysis over a full grid of seeds and iterations. Our findings suggest that the best way to split a fixed budget between more seeds (width) and more iterations (depth) changes with the strategy, the task, and the total budget. Furthermore, we observe that the ranking of strategies also changes with the budget. On one task the strategy that looks worst at one seed is best at forty seeds. On another the best number of iterations is well below the value common in practice, so extra depth wastes budget that more seeds would turn into score. We provide a measurement protocol that reports the seeds-by-iterations frontier and practical guidance for using it. 

---
# Learn Your Own Thoughts: Abstract Token Curriculum 

**Authors**: Khashayar Gatmiry, Avrajit Ghosh, Parsa Mirtaheri, Jason D. Lee, Nika Haghtalab, Emmanuel Abbe, Peter Bartlett  

**Link**: [PDF](https://arxiv.org/pdf/2609.19717)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable reasoning capabilities by utilizing chain-of-thought (CoT) as a scratchpad for intermediate stages of thinking. However, CoT techniques require explicit supervision on thinking tokens, which requires rich, task-specific data. In this work, we propose Abstract Token Curriculum (ATC), a novel curriculum learning framework that elicits effective continuous intermediate representations without direct supervision or manual scratchpad design. ATC gradually increases problem complexity through a sequence of distributions, training the model to develop internal abstract ``thoughts'' in the continuous representation space. This paper provides both theoretical and experimental evidence for the benefits of ATC and its advantages over previous methods for training continuous thoughts. Theoretically, we show that for learning parity functions with single-layer softmax attention using ATC, attention naturally focuses on the CoT tokens in the context that provide the ``easiest path'' to predicting the next token. Experimentally, we show ATC's effectiveness on graph reachability and arithmetic learning tasks. 

---
# SoK: Trading Agents or Market Crashers? Dissecting Robustness and Security Failures in Academic Financial LLM Trading Schemes 

**Authors**: Mengxiao Wang, Nitesh Saxena  

**Link**: [PDF](https://arxiv.org/pdf/2609.19705)  

**Abstract**: Autonomous large language model (LLM) agents are moving rapidly into high-stakes domains, yet existing agentic-AI security studies remain largely domain-agnostic and overlook the distinctive, high-consequence attack surface such settings create. We examine this gap through financial trading agents, a representative case of high-stakes agentic security, where a single compromised agent has direct execution authority over real capital in an adversarial, reflexive market. To this end, we present FARSIGHT (Financial Agent Robustness and Security Investigation and Global Holistic Testing), a framework that performs scheme-level evaluation of financial LLM agents on two axes: robustness under market turbulence (including flash-crash-like scenarios), and security against three attack types: attacks on information sources, attacks on agents, and agent-as-attacker behaviors. Applying FARSIGHT to 15 representative academic schemes, we find that most overlook robustness and realistic adversarial threats: 80% fail at least one core robustness metric and 100% exhibit security vulnerabilities. These two failure modes are inseparable: a small misjudgment can cascade into a market-wide crash on its own, while an adversary can deliberately trigger the same collapse at minimal cost. 

---
# Self-Evolving Search Index 

**Authors**: Sangam Lee, Wonjae Lee, Sunghwan Kim, Deogyong Kim, Jaehoon Kim, Daye Nam, SeongKu Kang, Dongha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2609.19656)  

**Abstract**: Information retrieval is increasingly important as LLM agents tackle complex tasks involving diverse information needs. Because retrieval relies on an index that represents each document through index keys, retrieval quality depends heavily on how effectively these keys expose the knowledge contained in each document. However, effective index representations vary across retrieval environments, making it difficult for any fixed optimization strategy to perform consistently. Yet evolving an index to its retrieval environment remains largely human-driven, requiring humans to diagnose retrieval failures, refine the optimization strategy, and reprocess the index accordingly. We propose SELF-INDEX, a framework that enables an index to self-evolve without human intervention. Its Optimizer autonomously diagnoses retrieval shortfalls, selectively revises the responsible index keys, and validates each revision before updating the index. Beyond reacting to observed retrieval demands, SELF-INDEX proactively explores additional demands through a Query Simulator, allowing the index to evolve beyond the queries already available for optimization. Across diverse corpora and retrievers, SELF-INDEX consistently improves retrieval performance while outperforming existing index optimization methods. We further show that these benefits extend to downstream applications, improving the effectiveness and efficiency of search agents and helping agent memory systems retrieve useful past interactions. 

---
# DataCanvas-EDU: An Agentic Framework for Instructor-Guided Synthetic Data Generation in Business Analytics Education 

**Authors**: Bang An, Maria Hamdani, Joseph Fox  

**Link**: [PDF](https://arxiv.org/pdf/2609.19617)  

**Abstract**: Business analytics education requires diverse datasets to support different learning objectives, student backgrounds, and analytical tasks. Real-world data can be difficult to obtain and offer limited flexibility for adapting a case to a particular course. Even when suitable data are available, instructors must investigate the patterns, verify the results, and prepare assignments and reference solutions, requiring substantial time and effort. The use of large language models (LLMs) introduces an additional concern about training data contamination. Widely used public datasets often have extensive tutorials and worked analyses that models may have encountered during training. Students may therefore receive explanations drawn from existing analyses without practicing how to investigate unfamiliar data in collaboration with AI. This paper presents DataCanvas-EDU, an agentic framework for instructor-guided synthetic data generation in business analytics education. Instructors specify teaching goals and intended patterns through conversation, while an AI agent writes generation code, checks the resulting data, and prepares assignments, reference analyses, and rubrics. Four phases, Plan, Create, Verify / Test Analysis, and Evaluate, organize the process and support instructor review and revision. The framework is intended to simplify case preparation while creating opportunities for students to investigate newly designed patterns with AI. We illustrate the approach with WindowDash, a food delivery case containing 15,000 orders and nine designed patterns. DataCanvas-EDU is packaged as a reusable AI Agent Skill for compatible agent environments, with the package and installation instructions available at this https URL 

---
# Semantic Layer Induction from Raw Telemetry via Hierarchical LLM and RAG Abstraction 

**Authors**: Yuanzhe Jia, Ali Anaissi  

**Link**: [PDF](https://arxiv.org/pdf/2609.19615)  

**Abstract**: Modern applications generate massive volumes of raw telemetry data, but translating those noisy, heterogeneous event streams into actionable business insights remains a fundamental challenge. Data engineers and analysts expend substantial effort reconciling semantic discrepancies, hand-crafting parsing logics, and maintaining fragile mappings between raw data and business KPIs. In this paper, we present an end-to-end framework that fully automates the construction of a business semantic layer from application raw logs. Our approach introduces a two-stage semantic abstraction: first, high-level business features are identified via LLM inference augmented with domain-specific industry knowledge; second, fine-grained business nodes are derived through a structured pipeline comprising data refinement, hybrid retrieval, multi-stage filtering, semantic clustering, and canonical naming. Evaluation on production-scale telemetry demonstrates that our system improves human-assessed semantic quality from 50 to 80+ on a 100-point scale, reduces maintenance effort by 80%, filters out 74% of noise, and achieves 0.87 Cohen's kappa via an integrated LLM-as-Judge evaluation, enabling continuous, scalable quality assurance. Overall, our work distinguishes itself from prior work by addressing the novel problem of business semantic layer induction from raw telemetry, operating without labeled training data or manual rule engineering. 

---
# TacSushi: Tactile-Grounded World-Action Modeling for Dexterous Sushi Manipulation 

**Authors**: Haodi Hu, Kaen Kogashi, Toshiaki Koike-Akino  

**Link**: [PDF](https://arxiv.org/pdf/2609.19613)  

**Abstract**: Dexterous food manipulation requires control under deformation, occlusion, and uncertain contact. We present TacSushi, a tactile-grounded, Cosmos3-based world-action policy that learns from recorded future consequences while acting on current observations. The backbone encodes current RGB, language, and hand state, and feature-wise gated fusion incorporates fingertip tactile features into the action representation. During training, a decoder conditioned on demonstrated action chunks predicts logged future visual observations, task progress, relative contact risk, and tactile summaries; this decoder is removed at deployment. Failed trials provide consequence supervision, but their actions are excluded from imitation. We train TacSushi on 340 successful and 50 failed real-robot trials and compare six methods in 600 separate rollouts across three in-distribution tasks and two out-of-distribution ingredient variants. To assess food quality beyond a single geometric threshold, we score terminal outcomes using an anchored visual-quality protocol that equally weights five human ratings and three vision-language-model ratings per rollout. Full TacSushi achieves 68.3% average in-distribution success and 37.5% out-of-distribution success, compared with 36.7%/10.0% without future-consequence supervision and 25.0%/17.5% with direct tactile concatenation in place of gated fusion. These comparisons support complementary benefits of feature-wise gated tactile fusion and training-only predictive supervision. 

---
# DeltaSelect: Affordable A/B Testing for Coding Agents 

**Authors**: Nicholas J. Conn  

**Link**: [PDF](https://arxiv.org/pdf/2609.19607)  

**Abstract**: Coding-agent benchmarks are built for broad and comprehensive comparisons, not frequent development decisions. Individual runs vary, full suites are expensive, and the benchmark harness may differ from the harness used in practice. In a resampling analysis of DeepSWE's published trials, only 19.5% of tasks (22 of 113) had a fifth-percentile Pearson correlation of at least 0.50 with full-benchmark performance. The paper presents DeltaSelect, an open-source method that identifies tasks whose one-run results consistently track full-benchmark performance using Pearson correlation, maps fractional verifier results to a common score using linear regression, and selects a fixed task set within a dollar budget. DeltaSelect is intended for repeated baseline-versus-candidate comparisons during development, not model rankings. In a gpt-5.6-luna low-reasoning case study, DeltaSelect was used to revise custom skills and instructions. Across 13 evaluations, the recorded cost was USD 27.86 at rates published August 16, 2026. The adopted version cost 58.1% less than the initial version (USD 1.75 versus USD 4.18; p=0.008), while the calibrated score was higher (42.36% versus 36.46%; published-analog variance p=0.326). 

---
# Form Over Content In Gradient-Based Data Attribution Methods 

**Authors**: Sunwoo Kim, Seokwon Jung, Sohyung Kim, Seong Joon Oh, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2609.19589)  

**Abstract**: Data attribution methods using gradient similarity are widely used to analyze and select training data for large language models, but what gradient similarity actually measures is debated. Some interpret it as identifying task-relevant skills, while other work reports that surface form is the main factor. We resolve this debate for supervised fine-tuning examples by varying task and answer format independently. Specifically, we render benchmarks in different answer formats, such that datasets can share a task without a format or a format without a task. We find that gradient alignment follows the answer format, as benchmark pairs sharing an answer format align strongly (disattenuated cosine near 0.4), while same benchmarks rendered with different answer format classes show no alignment (near 0.0). We demonstrate that this ordering holds from the earliest pretraining checkpoints through post-training, and across model scales and families. We then analyze the released selections of LESS, a gradient-based data selection method for instruction tuning, and find that each target's selections over-represent the target's own answer format. Hence, we demonstrate that gradient-based attribution methods track format similarity more than task semantics, meaning that such methods, as well as the semantic interpretation of the gradient, should be tested on data where answer format and task vary independently for greater robustness and reliability. 

---
# Red-Teaming Auto Mode: Improving Blocking Classifiers Against Malign Coding Agents 

**Authors**: Alex Remedios, Simon Storf, Fabien Roger, John Hughes  

**Link**: [PDF](https://arxiv.org/pdf/2609.19587)  

**Abstract**: To keep coding agents from going off the rails, production systems now review each proposed action with a blocking monitor that can reject it before it runs (Auto Mode in Claude Code, Guardian in OpenAI's Codex). Prior evaluations of such monitors largely measure robustness to accidental harm or prompt injections from untrusted sources looking to hijack the agent. Less understood is how they hold up when the agent they monitor is persistently misaligned. To understand this risk, we task an adversarial agent with evading production blocking monitors and causing catastrophic harm, e.g. by exfiltrating its own weights. We find that when instructed with high-level attack strategies, adversarial agents can succeed through several distinct mechanisms, such as agent-generated prompt injection against the monitor, multi-agent attacks, and malicious compaction. In particular we find that in 79% of trials, the agent can use an injection attack against Auto Mode and Guardian to run arbitrary bash commands. We also find that it is possible to greatly improve Auto Mode through design changes like enhancements to tool coverage, transcript formatting and an agentic monitor stage. Even with these improvements, preventing multi-context attacks at an acceptable cost remains an open problem. By detailing our red-teaming methodology and highlighting new attack vectors, we aim to help defenders evaluate their mitigations against the possibility of persistent malign coding agents. Code is available at this https URL. 

---
# CliniCIRCA: A Modular LLM Framework for Constructing Longitudinal Mental Health Patient Journeys from Raw EHR Narratives 

**Authors**: Aiwei Ivy Zhang, Nimra Ishfaq, Mohit Chandra, Santiago Alvarez Lesmes, Adam Coscia, Khatiya Chelidze Moon, Xiaohan Ding, Munmun De Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2609.19585)  

**Abstract**: In mental health care, reasoning over patient journeys is a key task for clinicians. Yet these journeys, encompassing a longitudinal progression of biological, psychological, and social events, are often spread across disparate unstructured text narratives, making temporal recovery challenging. We present CliniCIRCA, a multi-stage LLM framework for Calendar-anchored, Imprecision-aware Reconstruction of Clinical Annals. To our knowledge, CliniCIRCA is the first to temporally classify clinical events across unstructured discharge summaries without event-level timestamps. From 14,882 MIMIC-III mental health admissions, we first construct a benchmark of 52 discharge summaries on which CliniCIRCA produces 15,891 temporally tagged events. After correcting 629 errors based on a clinician-in-the-loop evaluation, we produce verified gold-standard labels. Finally, the corrected timelines drive a temporally grounded summarization stage that compresses each source 1.52 times into a date-grouped chronological record. We then scale the framework to generate 1,000 silver-standard timelines and evaluate them as training data. Compared with zero- and few-shot prompting, instruction tuning generally improves five open-weight models on event extraction, temporal tagging, and summarization across silver and clinician-verified evaluations. 

---
# Large Language Model Agents for Evidence Based Genetic Disease Severity Classification 

**Authors**: Tohid Ghasemnejad, Ahmadreza Argha, Mark Grosser, John Wang, Min Yang, Thantrira Porntaveetus, Tony Roscioli, Nigel H. Lovell, Mahmoud Aarabi, Hamid Alinejad-Rokny  

**Link**: [PDF](https://arxiv.org/pdf/2609.19569)  

**Abstract**: Disease severity classification for genetic conditions is subjective and labor-intensive, creating bottlenecks in genomic screening, where commercial panels vary widely in size and overlap. We developed an autonomous AI agent integrating Reasoning and Acting (ReAct) with Retrieval-Augmented Generation (RAG) to classify 10,211 Human Phenotype Ontology terms. It uses American College of Medical Genetics (ACMG)-endorsed severity guidelines and American College of Obstetricians and Gynecologists (ACOG) quality-of-life criteria to retrieve PubMed literature, generate interpretable reasoning chains, and independently verify claims. At the phenotype level, using expert-curated cohorts, the agent achieved 93.55% accuracy (MCC 0.9237) with 82.6% to 91.4% of claims supported by direct evidence or valid inferences. Gene-level severity was aggregated across 8,738 pairs, identifying 3,283 autosomal recessive pairs with severe or profound presentations. External validation showed 95.2% concordance with Mackenzie's Mission gene list. This system enables standardized panel design by providing reliable, automated classification supported by direct evidence. 

---
# A Multi-Modal Generative Model for Tomato Disease Leaves Understanding 

**Authors**: Khang Nguyen Quoc, Minh-Phuoc Tran, Gia-Han Truong, Luyl-Da Quach  

**Link**: [PDF](https://arxiv.org/pdf/2609.19555)  

**Abstract**: Artificial intelligence for plant disease analysis has advanced from task-specific classifiers to multi-modal models capable of jointly interpreting visual and textual information. However, practical deployment in precision agriculture remains limited because most existing approaches treat disease understanding as isolated prediction tasks, failing to capture the complementary relationships among symptom recognition, severity assessment, and question-driven diagnostic reasoning. In tomato pathology, accurate interpretation of diseased leaves requires more than label prediction; it demands integrating visual symptoms with semantic context to support a comprehensive and explainable understanding. Here, we present SOLAR, a multimodal generative model that understands tomato disease spanning six question-answering tasks. SOLAR learns to align visual features with task-aware language representations by Fusion Expert module based on mixture-of-expert, enabling it to generate contextually relevant answers across diverse diagnostic tasks. By formulating tomato disease analysis as a generative Visual Question Answering (VQA) task, SOLAR provides a flexible framework that supports multi-task inference within a single model while improving performance and cross-task knowledge sharing. We evaluate SOLAR on $41,677$ images, including $216,209$ Question-Answering (QA) pairs to understand tomato leaf disease under both closed and open-ended QA settings. Experimental results show that SOLAR consistently outperforms state-of-the-art vision-only, vision-language, and task-specific models across all tasks, demonstrating superior accuracy, robustness, and multimodal reasoning. These findings highlight the potential of generative multimodal modeling as an effective direction for understanding of plant disease. The code for this study is available at this https URL. 

---
# Compressed Active Subspaces for Scalable Bayesian Inference 

**Authors**: Thomas Flynn, Sanket Jantre, Byung-Jun Yoon, Kibaek Kim  

**Link**: [PDF](https://arxiv.org/pdf/2609.19539)  

**Abstract**: Active subspace methods provide a framework for quantifying predictive uncertainty in high-dimensional models by identifying and performing inference along parameter directions that have the greatest influence on the model output. However, the construction of active subspaces requires storing many full-dimensional model gradients, which becomes prohibitive as model size increases. We address this limitation by proposing Compressed Active Subspaces (CAS), a scalable approach that first maps the model parameters to a compressed space using a structured isometric embedding and then constructs the active subspace within this reduced parameterization. Our approach substantially reduces the memory required for active subspace construction and enables Bayesian inference for large models where standard active subspace methods become impractical. We demonstrate the scalability of CAS on neural networks of increasing size while maintaining predictive performance and robust uncertainty estimates. 

---
# Detecting Soft Errors in Parallel Software with LLM-tuned Instruction Duplication 

**Authors**: Yafan Huang, Guanpeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.19531)  

**Abstract**: We propose PaRID (PaRallel Instruction Duplication), a software-directed soft error detection framework that requires only compile-time effort for multithreading parallel programs. PaRID addresses two key challenges: supporting parallel programs with mixed serial and parallel regions and minimizing performance overhead without relying on costly dynamic profiling. It combines parallel-aware code transformation with LLM-tuned performance modeling, guided by eight generalizable findings from an offline characterization study, to enable fast soft error detection in parallel applications. Evaluation on NPB benchmarks shows that PaRID reduces protection overhead from 162.79% to 59.84% on average and achieves up to 5x speedup while maintaining full error detection effectiveness. 

---
# AURORA: A Natural Language-Driven Agentic Framework for Understanding, Reasoning, and Orchestrating Reliable Air-Ground Co-Simulation 

**Authors**: Keshu Wu, Hao Zhang, Rui Gan, Xiangbo Gao, Xiaopeng Li, Zhengzhong Tu, Yang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2609.19527)  

**Abstract**: Air-ground transportation research increasingly relies on co-simulation, yet constructing scenarios remains labor-intensive and difficult to validate. More importantly, a generated scenario may execute successfully while failing to realize the spatial, temporal, communication, or behavioral relationships requested by the user. This paper presents AURORA, a natural-language-driven agentic framework that treats air-ground scenario generation as a process of compilation with verification. Central to AURORA is the Air-Ground Scenario Graph (AGSG), a typed intermediate representation that explicitly connects agents, aerial missions, events, communication links, success conditions, and their cross-domain dependencies. This shared representation enables simulator-grounded parsing, joint road-airspace grounding, temporal planning, pre-execution feasibility checking, trace-based runtime verification, failure localization, and bounded repair within a unified workflow. We further introduce AURORA-Bench to evaluate not only whether generated scenarios execute, but whether they faithfully realize the requested interactions. Experiments across multiple language models show that structured execution substantially improves reliability, while runtime verification exposes silent failures that completion-based evaluation overlooks. Localized repair further resolves many violations without regenerating the entire scenario. The results show that reliable scenario generation requires verifying realized behavior, not merely executable code, and demonstrate the value of explicit intermediate representations for verifiable and repairable language-driven co-simulation. 

---
# CoreSense: Traceable Failure Recall and Conflict-Aware Belief Gating for Auditable Robot Decisions 

**Authors**: Zoe Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.19512)  

**Abstract**: Robots can recall prior failures without knowing whether recalled evidence remains valid, conflicts with current observations, or is sufficient to guide a decision. We present CoreSense, a robot-system integration architecture that combines traceable episodic evidence with a conflict-aware belief gate and bounded, auditable recommendations. The gate checks scope, provenance, time, contradiction, and support before it permits PROCEED, requests re-observation, abstains, or escalates. Evaluation follows three complementary layers without commanding a physical robot: offline public real-robot data, a frozen signal-level simulation, and a live cloud deployment path. On CableTrace-120 and BotFails-200, belief gating reduces protocol-defined unsafe proceeds from 20% and 40% to 0%. A disjointly calibrated raw-video policy also reaches 0% unsafe proceed, but overblocks every nominal episode. On public data, a ViFailback-BotFails visual detector reaches 0.778 AUROC yet remains all-blocking, whereas cycle-disjoint UR3 telemetry for protective stops yields 0% unsafe proceed, 36.1% overblocking, and 61.9% coverage; grip-loss transfer remains a negative result. Controlled physical corroboration yields 3.3%, 0%, and 42.0%, while conflict-aware fusion yields 4.7%, 0%, and 42.8%. Finally, 20/20 cloud recalls validate a CockroachDB Cloud-Amazon Bedrock deployment path. The evidence supports an auditable integration pattern, not autonomous recovery or certified safety. 

---
# For Your Eyes Only: Evaluating Coordination Between Isolated Language Model Instances 

**Authors**: Alexander Shirnin, Aleksey Kudelya  

**Link**: [PDF](https://arxiv.org/pdf/2609.19504)  

**Abstract**: As model-generated content is increasingly consumed by other model instances in automated workflows, a practically important question arises: can a model embed a signal in natural language that an independent instance of the same model can detect, relying only on shared pre-training and task instructions, without any shared memory or coordination-specific training? We introduce For Your Eyes Only, a cooperative signalling game designed to evaluate this directly. A Sender produces free-form descriptions for two words, one of which is a hidden target; an isolated Receiver must identify it. We evaluate seven contemporary models from four architectural families on 300 word pairs from established psycholinguistic corpora, using the Double-Pass Success Rate to control for output biases. We find that most models struggle to maintain coordination once they are required to avoid detectable signals, while one frontier model retains near-perfect performance even after such filtering. We further show that models can direct this capability toward deliberate misdirection, and that coordination is consistently weaker across architectures than within them. 

---
# Efficiently Linking Unstructured Data for Multi-step Reasoning 

**Authors**: Jiaming Liang, Haydn Jones, Jacob R. Gardner, Mark Yatskar, Zachary Ives  

**Link**: [PDF](https://arxiv.org/pdf/2609.19491)  

**Abstract**: Modern LLMs and AI agents increasingly support data engineering workflows that integrate evidence from unstructured sources. Such pipelines typically do data retrieval, integration, and ranking before proceeding to more complex agentic reasoning or actions, e.g., for scientific discovery. The core retrieval problem in these workflows jointly executes multi-attribute filtering, multi-vector search, exact relational joins, and thresholded embedding-similarity joins. Given a planned query and monotone scoring function, our DASE query engine constructs and ranks candidate evidence tuples. It comprises (i) a multi-step reasoning query model over structured predicates, multiple vectors, and relational links; (ii) SemJI, a sparse materialized embedding-similarity join index for rare near-neighbor pairs; and (iii) a co-designed execution layer that combines predicate-aware ANN traversal, batched access, and threshold-based score aggregation.
On scientific-discovery workloads, DASE retrieves candidate evidence for multi-step reasoning queries 6x to 46x faster than strong RDBMS, rerank, and vector-database baselines at comparable recall; and for tasks that require semantic-operator post-processing, DASE acts as a high-recall prefilter that makes downstream LLM evaluation both cheaper and more accurate -- e.g., on SemBench E-Commerce it improves BigQuery quality from 0.67 to 0.80 while cutting cost from $2.42 to $0.54. 

---
# From Models to Systems: A Comprehensive Survey of Efficient Multimodal Learning 

**Authors**: Pan Wang, Siwei Song, Hui Ji, Siqi Cao, Heng Yu, Zhijian Liu, Huanrui Yang, Yingyan Celine Lin, Beidi Chen, Mohit Bansal, Xiaoming Liu, Pengfei Zhou, Ming-Hsuan Yang, Tianlong Chen, Jingtong Hu  

**Link**: [PDF](https://arxiv.org/pdf/2609.19445)  

**Abstract**: The rapid expansion of multimodal models has surfaced formidable bottlenecks in computation, memory, and deployment, catalyzing the rise of Efficient Multimodal Learning (EML) as a pivotal research frontier. Despite intensive progress, a cohesive understanding of what, how, and where efficiency is manifested across the learning stack remains fragmented. This survey systematizes the EML landscape by introducing the first structured, model-to-system taxonomy. We distill insights from over 300 seminal works into three hierarchical levels--model, algorithm, and system--addressing architectural parsimony, execution refinement, and hardware-aware orchestration, respectively. Moving beyond a purely categorical review, we offer a methodological synthesis of the vertical synergies between these layers, elucidating how cross-layer co-design contributes to the fundamental "Efficiency-Utility-Privacy" trade-off. Through an integrative case study of Multimodal Large Language Models (MLLMs), we trace the field's evolutionary trajectory from initial structural adjustments to modern full-stack resource orchestration. Furthermore, we provide a holistic discussion and application-specific optimization blueprints for diverse domains and posit a paradigm shift toward self-regulating intelligence, where efficiency is an intrinsic, emergent property of the model's fundamental design rather than a post-hoc constraint. Finally, we present open challenges and future directions that will define the trajectory of EML research. This survey establishes a structured framework for multimodal systems that are not only high-performing and generalizable but natively efficient and ready for ubiquitous deployment. A continuously updated version is available at this https URL. 

---
# Predict Before You Deploy: Offline Prediction of Quantization-Induced Task Degradation for World Action Models 

**Authors**: Jiuyi Xu, Jinjia Guo, Meida Chen, Jing Du, Yangming Shi  

**Link**: [PDF](https://arxiv.org/pdf/2609.19441)  

**Abstract**: World action models (WAMs) rely on video-generation backbones, requiring substantial memory and compute for deployment. Post-training quantization reduces memory and can accelerate inference, but bit width, grouping, and quantizer choice define a large configuration space. Identifying configurations that preserve task performance through exhaustive closed-loop evaluation is costly. We propose PreDE (Predict Before You Deploy), a policy-calibrated framework for predicting quantization-induced task degradation from offline action deviations. Using closed-loop outcomes from a small development set, PreDE calibrates two thresholds and accepts, rejects, or defers new configurations using a fixed observation log. Under a within-setting label-ordering hypothesis, the rule issues decisions where all thresholds consistent with the development labels agree. Across five WAMs and four benchmark settings, quantization produces configuration-dependent task losses that cannot be explained by bit width alone or a shared deviation threshold. Across 28 held-out configurations from two policies, PreDE issued 21 decisions before observing closed-loop outcomes (75% coverage), all matching the observed acceptable or degraded labels. Deferred candidates included both acceptable outcomes and a 33-percentage-point loss. In 450 Franka Research 3 trials across two independently fine-tuned policies, all configurations assigned to high-deviation groups before testing showed significant degradation, while low-deviation comparisons showed no statistically significant degradation. On the real robot, W4A4 achieved a 1.37x action-query speedup and approximately 44% lower peak memory. These results support policy-specific behavioral calibration for quantization configuration selection while identifying candidates that require closed-loop evaluation. The code is available at this https URL. 

---
# From Rollout to Reset: A Graph-Based Harness for Autonomous Long-Horizon Manipulation Evaluation 

**Authors**: Jing Jiang, Yue Yang, Xinkai Jiang, Gedas Bertasius, Daniel J. Szafir, Rudolf Lioutikov  

**Link**: [PDF](https://arxiv.org/pdf/2609.19413)  

**Abstract**: Robot manipulation policies are improving quickly, and real-robot evaluation remains the standard evidence for that progress. It still relies on a human to reset the scene between rollouts, which consumes operator time and leaves the initial state distribution unspecified, so results reproduce poorly. A recent system, AutoEval, automates both reset and scoring, but only for single-step tasks, because a long-horizon rollout can terminate in combinatorially many configurations that no single learned reset policy covers. We present HALTER, a Harness for Autonomous Long-horizon Task Evaluation and Reset, which restores the scene by planning over a library of learned atomic reset skills, so demonstration cost scales with the size of that library rather than with the number of terminal states. HALTER builds a spatial scene graph online from point clouds and vision foundation models, and an LLM reasons over this graph to score the rollout, plan the reset, and verify that the reset succeeded, without collecting labeled success images for any task. On four long-horizon tasks on a Franka arm, HALTER restores the scene in 76% of episodes, against 52% for AutoEval and 65% for a motion-planning reset, and it estimates the completed-skill fraction correctly in 90% of episodes, against 76%. Its reset-verification verdict is correct in 91% of episodes, compared with 78% for AutoEval. It also cuts the operator time of an evaluation campaign by 72% relative to manual reset. We further measure compositional generalization on three held-out tasks, where HALTER resets 74.7% of episodes against 1.3% for a per-task reset policy, and we ablate the scene representation and the graph update rate. 

---
# Efficient Nash Equilibrium Computation for Cybersecurity Games 

**Authors**: Michael Lanier, David Farmer, Yevgeniy Vorobeychik  

**Link**: [PDF](https://arxiv.org/pdf/2609.19399)  

**Abstract**: Computing Nash equilibria of simulation-based cybersecurity games with policy-space response oracles (PSRO) is bottlenecked by payoff estimation: every payoff-matrix entry costs Monte-Carlo rollouts of a slow simulator, while policies and restricted-game solves are cheap. We introduce Regret-Weighted Payoff Sampling (RWPS), a budgeted estimator that simulates only the cells an equilibrium is sensitive to and fills the rest with a surrogate trained on every entry simulated earlier in the run. The sup-norm error bound cannot evaluate such an estimator, because it is set by the cells left deliberately inaccurate. We prove an instance-dependent bound that weights error by the opponent's equilibrium mixture, a certificate computable from simulation data alone, and a coverage result showing that once the deviation-relevant set is simulated, surrogate error cannot affect either player's regret. On three 21x21 general-sum games, two synthetic and an asymmetric Colonel Blotto, the refined bounds are four to six times tighter on the estimator's own output, and the coverage result predicts in advance which games are cheap: 18% of the matrix for small-support games against 82% for Blotto. In growing-pool PSRO, RWPS reaches lower exploitability than minimum-regret-first search, information-gain search, and progressive sampling at a matched budget, and on the CyGym and ANSG cyber simulators it is lowest at the smallest budgets. 

---
# Riemannian--Lorentz Fusion of Vision Transformers and State-Space Models 

**Authors**: Badri N. Patro, Vijay S. Agneeswaran  

**Link**: [PDF](https://arxiv.org/pdf/2609.19384)  

**Abstract**: Scaling deep learning faces critical bottlenecks: data exhaustion, exponential training costs, and resource concentration. Model merging combines pre-trained checkpoints without gradient descent, offering orders-of-magnitude savings versus retraining. Combining independently trained vision models is difficult when their architectures and parameter shapes differ. Existing weight-space merging methods generally assume aligned, shape-compatible checkpoints, whereas a Vision Transformer (ViT) and a state-space model (SSM) implement token mixing with different operators. We study a hybrid Heterogeneous merging setting that retains both architectures while aligning parameter groups by semantic role. Our proposed Riemannian--Lorentz Parameter Fusion (RLPF) method projects aligned groups to common coordinates, lifts selected coordinates to the Lorentz hyperboloid model of hyperbolic space, computes a regularized geodesic barycenter, and decodes the result into the two branches. A learned gate then combines branch logits for each input. Component groups use fixed curvature values, with normalization parameters treated as Euclidean. In the results available in this manuscript, the fine-tuned system obtains 82.37\% on CIFAR-10, 75.04\% on Oxford-IIIT Pet, and 78.58\% top-1 accuracy on ImageNet-1K; the corresponding best-parent accuracies are 76.54\%, 71.42\%, and 76.42\%. On ImageNet-1K, the reported pre-fine-tuning initialization reaches 77.80\%. These results support further study of geometry-aware heterogeneous fusion, but not a training-free single-checkpoint merge: RLPF is a two-branch hybrid whose gate and reported final models are trained. 

---
# LinePilot Digitizer: Line-Plot Recovery with Manual and Automatic Calibration 

**Authors**: Fengbo Ma, Rayan Akhtar, Aakash H. Joshi, Xiaoting Li, Haijian Sun, Zhen Xiang, Xianyan Chen, Yiping Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2609.19377)  

**Abstract**: Recovering numerical series from line plots requires accurate axis calibration and reliable curve extraction. We present LinePilot Digitizer (LinePilot), which combines continuous color-based curve recovery with three calibration modes: LinePilot (standard), LinePilot (enhanced), and LinePilot (OCR). We also introduce DigitizerBench, the first dedicated benchmark for systematically evaluating digitizer performance, using an orthogonal design spanning signal, rendering, and plot-structure factors with complementary automatic and human-guided evaluations. We evaluate performance using failure-penalized capped normalized root-mean-square error (FPC-NRMSE), which assigns unit loss to missing, unusable, or catastrophically inaccurate outputs. On DigitizerBench-Full, LinePilot (OCR) achieves the lowest mean FPC-NRMSE (0.672) and highest trusted usability (38.2%) among the tested automatic pipelines. On DigitizerBench-Lite, LinePilot (enhanced) achieves the lowest mean FPC-NRMSE (0.081), 100% output success, and highest trusted usability (93.3%). The orthogonal benchmark design further enables factor analysis to identify the factors that most significantly affect digitizer performance. Together, the three calibration modes provide a practical trade-off between automation, user control, and accuracy within a shared curve-recovery workflow. 

---
# How to Guide Your Language Flow 

**Authors**: Rohit Dilip, Tianrong Chen, Yuyang Wang, David Van Valen, Joshua Susskind, Miguel Angel Bautista  

**Link**: [PDF](https://arxiv.org/pdf/2609.19356)  

**Abstract**: We introduce a new method to guide flow matching models. Our approach, which we call probe guidance, uses the frozen internal states of an existing diffusion model to construct a guidance signal. This works using a similar principle as autoguidance, but eliminates the need for an additional forward pass at inference time and provides a reliable path to ensure that the weak and strong model share similar dynamics. We apply and benchmark this method on continuous diffusion language models, where probe guidance sets a new state-of-the-art performance on unconditional generation. When applied to a 1.7B diffusion language model, probe guidance consistently improves on multiple choice question answering benchmarks. Using our probes, we study the traditional autoguidance setting where the strong model is a weak checkpoint, and find that the weak model must come from a low-entropy region of training. These findings both provide a practical way to improve diffusion language models and shed light on the actual mechanism behind autoguidance, which is currently poorly understood. 

---
# Can Vision-Language Models Judge Olympic Diving? From Reasoning to Scores in Zero-Shot Action Quality Assessment 

**Authors**: Henry O. Velesaca, David Freire-Obregon, Luigi Miranda, Abel Reyes-Angulo  

**Link**: [PDF](https://arxiv.org/pdf/2609.19354)  

**Abstract**: Automated action quality assessment (AQA) in Olympic sports remains a challenging task due to the complexity of human motion and the subjectivity inherent in expert judging. This work evaluates the capability of open-source Vision-Language Models (VLMs) to perform zero-shot action quality assessment on Olympic diving videos using the AQA-7 benchmark dataset. In this regard, a regression-based framework is pro-posed to leverage both the semantic reasoning and phase-level sub-scores generated by the VLMs, combining TF-IDF vectorization, dimensionality reduction, and ensemble learning to predict final competition scores. Experimental results show that standalone VLMs achieve moderate Spearman correlations below 0.32, while the proposed ensemble regression framework substantially improves performance in the reported evaluation, reaching a Spearman correlation of 0.67 with a four-model configuration. Textual reasoning features con-sistently outperformed raw numerical sub-scores, highlighting the richness of VLM-generated explanations for action quality analysis. These findings suggest that VLMs hold strong potential as assistive tools for explainable and semi-automated sports performance evaluation. The code is publicly available on GitHub this https URL diving judge vlm 

---
# Kinematics-Grounded Agentic AI for Robotic Additive Manufacturing Process Planning 

**Authors**: Jingzhan Ge, Ruimin Chen, Azadeh Haghighi, Jiong Tang, Farhad Imani  

**Link**: [PDF](https://arxiv.org/pdf/2609.19347)  

**Abstract**: Robotic additive manufacturing (AM) extends material-extrusion printing beyond gantry kinematics but makes process planning robot-dependent. A slicer-generated plan that appears favorable in part coordinates can become infeasible or robotically unfavorable on a manipulator because slicer-process decisions and part orientation determine the generated path, while part orientation and workspace placement affect its kinematic realization. Existing AM tools, large language model (LLM)-based decision-support methods, and digital-shadow systems do not provide integrated pre-execution evaluation of these coupled decisions. This paper presents agentic robotic additive manufacturing (A-RAM), an agent-specialist-tool framework that converts user intent and a part file into traceable, execution-ready plans. The LLM interprets manufacturing objectives and constraints, identifies prescribed and searchable planning variables, and encodes this reasoning in a schema-constrained request; a deterministic Planning Agent instantiates the corresponding search workflow, while domain tools compute quantitative evidence for slicing, placement, inverse kinematics, trajectory timing, Joint-6 jerk, and extrusion. The framework is evaluated on a six-axis robotic-arm AM cell through three case studies covering expert-specified planning, goal-only planning, objective-dependent infill screening, and geometry-dependent orientation-placement selection. Across the evaluated candidate sets, selected plans achieve up to 53.5% lower maximum Joint-6 jerk and 48.3% lower mean absolute Joint-6 jerk than the least favorable valid candidates, while objective-specific infill screening yields motion-plan completion times up to 40.1% shorter and extrusion paths up to 12.7% shorter than the corresponding least favorable screened patterns. 

---
# AUDITPLAN: Commit, Then Answer for Auditable Safety Alignment 

**Authors**: Sai Sri Pushpa Jampani, Kshitij Mishra, Asif Ekbal  

**Link**: [PDF](https://arxiv.org/pdf/2609.19325)  

**Abstract**: Safety tuning pipelines judge only the final answer, which makes it difficult to distinguish robust refusal from two undesirable shortcuts: blanket refusal on benign requests and polished but unfaithful safety rationales that do not actually constrain the answer. We propose AUDITPLAN, a single-model plan-then-answer approach where the model first emits a compact structured safety plan and then answers conditioned on it. The plan records a threat label, intended action, and explicit constraints, enabling machine-checkable auditing while remaining hidden from users at deployment. We train this behavior with supervised fine-tuning followed by reinforcement learning with FAITHGATE, a reward-gating objective that grants answer reward only when the safety plan is correct. This discourages safe-looking but unfaithful behavior and promotes tighter plan-answer coupling. Across Qwen backbones, AUDITPLAN improves both robustness and auditability: on Qwen2.5-3B-Instruct, FAITHGATE reduces ASR from 24.0% to 11.6%, LSR from 1.0% to 0.36%, and over-refusal from 11.0% to 2.0%, outperforming answer-only RL, free-form explanation, and weighted-sum structured rewards. Similar trends hold for Qwen2.5-1.5B-Instruct. Larger-model confirmation runs on Qwen-3-4B-Instruct and Qwen2.5-7B-Instruct preserve the same trend suggesting that explicit internal commitments can make safety alignment more faithful, robust, and auditable. 

---
# GAVEL: Graph World Models for Verified and Efficient Long-Horizon LLM Task Planning 

**Authors**: Ruiyang Wang, Hao-Lun Hsu, Swarajh Mehta, Jiwoo Kim, Zhihao Dou, Miroslav Pajic  

**Link**: [PDF](https://arxiv.org/pdf/2609.19315)  

**Abstract**: Large language models (LLMs) provide a flexible interface for long-horizon robot planning, but generated plans often fail to respect embodiment constraints, recover from planning errors, or reason effectively under partial observability. We present GAVEL, a framework for verifying and repairing long-horizon LLM planning built around an explicit graph world model. The graph represents relevant object-relations, action pre-conditions and effects, and probabilistic beliefs over unobserved object locations. This model can predict the consequences of LLM-generated actions before execution, detect violations, and repair those whose corrections follow directly from the world model. This method also reserves LLM replanning solely for errors requiring semantic reasoning. For multi-task instructions, GAVEL reasons over distributions of possible object locations to reorder remaining subtasks and minimize expected search cost. We evaluate GAVEL on BEHAVIOR-1K across 100 single long-horizon tasks and 500 multi-task instructions. With Qwen3-8B, GAVEL improves single-task success from 41.2% to 91.8% and multi-task success from 19.9% to 92.6%. Distributional belief reasoning also reduces travel distance by approximately 5.4% compared with a static variant. These improvements show that an explicit graph world model harness can substantially improve the reliability and efficiency of long-horizon embodied planning across compact and frontier hosted LLM capabilities. 

---
# Why Pretraining Fails to Share Cross-Lingual Knowledge 

**Authors**: Adam Gaber, Uriel Dolev, Elisabeth Fittschen, Bobby Cheng, Yuval Marton, Leshem Choshen  

**Link**: [PDF](https://arxiv.org/pdf/2609.19291)  

**Abstract**: Large Language Models (LLMs) have made remarkable progress in the processing and modeling of many languages. Yet, unlike human multilinguals, they exhibit surprisingly limited cross-lingual knowledge transfer. While this limitation is well documented, its origins during multilingual training remain unclear. We pretrain 360M- and 7B-parameter LLMs and show that poor cross-lingual knowledge generalization emerges during pretraining and persists under standard interventions. To isolate its cause, we employ a controlled bilingual pretraining setting using two copies of the same language, sharing identical text and token segmentation, but mapped to disjoint token spaces. We find that disjoint tokens alone are enough to induce knowledge compartmentalization, even between identical copies of the same language, establishing disjoint token spaces as a fundamental barrier to cross-lingual knowledge generalization. Guided by this understanding, we suggest mapping languages into a shared token space by simple word-wise translation and find it substantially improves cross-lingual knowledge generalization, recovering up to 12.6\% of native-language learning efficiency --- 14$\times$ the baseline. 

---
# Physics-Informed Hemodynamic Modeling for Data-Free Prediction and Sparse-Data Assimilation 

**Authors**: Xi Chen, Jianchuan Yang, Hongde Li, Guangxin He, Qiuyu Ye, Qiang Luo, Mao Chen, Wenqi Hu  

**Link**: [PDF](https://arxiv.org/pdf/2609.19290)  

**Abstract**: Clinical decision-making for coronary intervention relies mainly on angiography and fractional flow reserve (FFR). However, angiography is two-dimensional and lacks depth information for 3D lesion characterization, while FFR provides only a single functional index, offering limited hemodynamic insight. Among existing methods, numerical analysis is computationally expensive, whereas learning-based approaches require extensive supervision and often lack physical consistency. To address these limitations, we propose physics-informed hemodynamic modeling, an integrated deep learning framework for 3D coronary blood flow analysis from dual-view angiography. First, an attention-enhanced CNN reconstructs coronary geometry from angiography. The resulting point clouds are then mapped to a reference domain and Fourier-encoded for joint representation. A decoupled network separately predicts velocity and pressure fields, with embedded physical priors enabling efficient transfer across physiological conditions. Across 32 clinical patients evaluated under four flow conditions, the trans-stenotic pressure-drop mean absolute percentage error was 2.02%, while the velocity and pressure relative-L2 errors were 0.054 and 0.023, respectively. Validation against hospital-measured FFR further achieved 93.8% diagnostic accuracy (30/32; exact 95% CI, 79.2%-99.2%). The framework also supports illustrative revascularization comparisons and sparse-data assimilation, with the full angiography-to-hemodynamics pipeline completed within 20 minutes per patient. 

---
# Randomized SVD Approximations for Spectral Co-Clustering of Word-Document Matrices 

**Authors**: Fateme Mazdarani, Carlos Toxtli  

**Link**: [PDF](https://arxiv.org/pdf/2609.19243)  

**Abstract**: Spectral co-clustering is a useful tool for discovering latent structure in word-document matrices, but its reliance on singular value decomposition (SVD) can make standard formulations expensive on high-dimensional data. This paper presents two randomized approximations for normalized spectral co-clustering of bipartite text data when the numbers of document and word clusters may differ. The first method uses randomized SVD through random projection, while the second combines partial SVD with element-wise random sampling. Across real-world and synthetic datasets, both methods reduce runtime relative to the full-SVD baseline, but their behavior depends on matrix sparsity. The random projection method is the more reliable approximation across the tested settings, whereas the sampling-based method is most useful on denser matrices and provides limited benefit on already sparse text data. These results show that randomized approximations for spectral co-clustering should be selected according to the underlying structure of the data. 

---
# Robust Conformal Intrusion Detection via Traffic-Aware Calibration and Attack-Orbit Invariance 

**Authors**: Zhenpeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.19241)  

**Abstract**: Large language models fine-tuned for network intrusion detection emit single-point predictions without statistical validity guarantees. Conformal prediction supplies a finite-sample coverage guarantee, but a threshold calibrated on clean traffic fails once an adversary perturbs controllable network features. We demonstrate this failure across three intrusion detection benchmarks and propose traffic-aware conformal prediction, which calibrates on traffic drawn from the perturbation mechanism an attacker is expected to use and provably restores coverage whenever that mechanism is known and can be sampled. A stronger, adaptive attacker that queries the target model's own score can still degrade this matched-calibration guarantee. We address this second threat model by excluding attacker-controllable features and their deterministic descendants from the scored representation, and prove that this yields an exact, pathwise coverage guarantee rather than a probabilistic bound. Across three independently fine-tuned language model architectures, this representation remains completely unchanged under every evaluated attack attempt, at a quantified seven-to-fourteen-point cost in clean accuracy relative to the unrestricted feature set. 

---
# YNU-HPCC at SemEval-2025 Task 11: Bridging the Gap in Text-Based Emotion Using Multiple Prediction Headers 

**Authors**: Hao Yang, Jin Wang, Xuejie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.19238)  

**Abstract**: This paper describes the participation of the YNU-HPCC team in subtask A of task 11, Bridging the Gap in Text-Based Emotion at SemEval-2025. Our best-performing system employs the RoBERTa (Robustly Optimized BERT Approach) model, an improved version of BERT that utilizes the Transformer encoder architecture. We enhanced the output head to allow the model to process one emotion simultaneously. We obtained the official ranking score (0.44), including results from all languages. The entire dataset was translated into English using Google Translate to facilitate subsequent processing. Through probabilistic and attention analyses, we found that (I) a single prediction head performs better than six heads predicting six emotions simultaneously, and (II) training on a uniformly translated English dataset yields better results than using the original dataset. The code is available at: this https URL. 

---
# The AR Fairness Metamodel: A Structured Framework for Fairness Measures 

**Authors**: Julian Alfredo Mendez, Timotheus Kampik  

**Link**: [PDF](https://arxiv.org/pdf/2609.19234)  

**Abstract**: This paper presents the AR fairness metamodel, a framework designed to represent, analyze, and compare different fairness scenarios. The metamodel considers key elements, such as agents, resources, and their attributes, and enables the systematic definition and comparison of various fairness measures. We provide examples involving both discrete and continuous measures, including equality, equity, group fairness, individual fairness, the Gini index, the Theil index, Jain's fairness index, and a detailed fairness measure for Australia's Child Care Subsidy. We also explore relationships among group fairness, individual fairness, and envy-freeness, supported by formal proofs. At the conceptual modeling level, our approach builds on the Tiles framework, which offers modular components that can be connected to capture diverse fairness definitions. The goal is to make AR-based fairness definitions practical and adaptable across contexts, providing a clear way to define, compare, and evaluate them. An implementation of the Tiles framework is available as an open-source tool, and can support fairness modeling and evaluation across a wide range of applications. 

---
# PAPC: Platform Mediation for Privacy-Propagation Externalities in AI-Mediated Workflows 

**Authors**: Tao Huang, Guosen Wu, Chen Hou, Guolong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2609.19226)  

**Abstract**: AI-mediated platforms coordinate work through LLM agents acting for different principals. In these workflows, privacy loss can be created before a final answer appears: a memory write, shared-workspace update, inter-agent message, or tool event may impose downstream exposure cost on another principal. We model this failure mode as a privacy-propagation externality, where the cost of a raw disclosure depends on topology and fanout as well as content. We present PAPC, a platform-mediated mechanism that intercepts information-moving events before they update shared state or external channels. PAPC combines policy, provenance, topology/fanout, privilege, and content signals to allow an event, release a policy-safe abstraction, quarantine raw content, block a transition, or narrow onward rights. The model explains why final-output control misses intermediate exposure costs and why high-fanout objects amplify propagation. Across retrieval-memory and multi-agent workflow benchmarks, PAPC preserves deterministic task completion and eliminates measured exact raw-value and external raw-value exposure. The results position event-level mediation as a platform-governance primitive for agent-mediated online work. 

---
# Perceptual Refinement of an End-to-End Video Streaming Pipeline via Generative AI Layers 

**Authors**: Emanuele Artioli, Farzad Tashtarian, Christian Timmerer  

**Link**: [PDF](https://arxiv.org/pdf/2609.19215)  

**Abstract**: Traditional codecs treat every region of a frame alike; a generative layer can instead degrade the regions a viewer attends to least and reconstruct them at the client. We present PRESLEY, which extends the prior conference work ELVIS by replacing destructive block removal with adaptive in-place degradation under a removability mask, signaling per-block strength in a bit-packed side channel, and restoring via generative backbones conditioned on transmitted visual priors rather than unconditioned in-painting. We separate the problem into three goals: choosing which blocks to degrade, degrading them so the encoder spends fewer bits, and restoring them. Against its predecessor at matched rate, PRESLEY achieves a decisive mean -56.4% BD-rate reduction on delivered background quality across 13 rate ladders spanning multiple codecs and dataset families. Against pristine baselines, PRESLEY defines the operating regime of generative transport: delivering substantial bitrate savings (up to -29.4% BD-rate) and superior background quality (17/23 sequences) in the target bit-starved regime, while maintaining foreground fidelity bit-exact. We further map where the theoretical headroom in this class of architecture lies. Using an exact leave-one-superblock-out combinatorial oracle as an additive empirical bound, we show that existing complexity heuristics already capture 83.3% of bit-cost savings, bounding remaining cost-axis headroom at about 5% of total bitrate. We then identify and model the primary unaddressed axis -- post-restoration damage -- which disperses widely (4.9-8.4 dB). We prove that this damage is predictable before transmission (held-out rho = +0.400), establishing the feasibility of transmit-time restorability modeling and defining the roadmap for joint rate-distortion-restoration selection rules. 

---
# Layer-wise Curriculum Learning for Efficient LLM Compression 

**Authors**: Donggeon Lee, Dooyeon Na, Seungmin Oh, Jongbin Ryu  

**Link**: [PDF](https://arxiv.org/pdf/2609.19213)  

**Abstract**: In this paper, we introduce layer-wise curriculum learning for efficient LLM compression. The proposed method facilitates the knowledge transfer from the teacher model to the student model, utilizing a curriculum learning approach that begins with easier optimization tasks and progressively tackles harder ones. In order to adopt the layer-wise learning in LLM compression, we partition the whole model into multiple segments consisting of layers, thereby enabling more computationally efficient knowledge transfer for LLMs. Based on our theoretical analysis of cumulative error phenomenon, layer-wise curriculum learning accelerates convergence while stabilizing the knowledge transfer process. In addition, we present a feature caching method with a multi-threading strategy to efficiently address feature misalignment across layers, maximizing GPU utilization. Consequently, our method exhibits advanced model compression performance, as well as high computational efficiency in terms of minimized memory usage and short training hours. Experiments on multiple datasets show that the proposed method achieves state-of-the-art performance while reducing GPU memory usage and training hours by more than 50\% on BERT and GPT-2. Moreover, it outperforms the other pruning methods on LLaMA-family and Qwen models under the same training hours, with a lower GPU memory footprint. 

---
# Not All Nodes Are Created Equal: Homophily-Aware Stratification for Stable GNN Evaluation 

**Authors**: Naga Venkata Sai Jitin Jami, Thomas Altstidl, Sebastian Hoefler, Jonas Mueller, Dario Zanca, Bjoern Eskofier, Heike Leutheuser  

**Link**: [PDF](https://arxiv.org/pdf/2609.19210)  

**Abstract**: Graph neural networks are widely used for transductive node classification, with accuracy typically measured on randomly drawn train/validation/test splits. Reported accuracy has been shown to shift substantially across different random splits of the same dataset, making published comparisons between architectures unreliable. The classical remedy in non-graph settings is stratified $k$-fold cross-validation, which ensures each test fold reflects the full class distribution of the dataset. We argue that class stratification alone is insufficient for graphs: nodes are not isolated but connected, and folds that differ in their distribution of local neighbourhood homophily expose the model to systematically different relational conditions that directly affect message-passing behaviour. The resulting cross-fold variation reflects the homophily composition of each split, inflating reported variance beyond what model behaviour alone would produce. To address this, we propose \hp{}, a topology-aware stratification procedure that treats node homophily as the primary stratification axis, aligning folds with respect to local relational consistency alongside the class marginal that standard stratification already controls. Stratifying on homophily alone does not guarantee class balance, so \hp{} incorporates class label as a secondary axis, preserving class representativeness as a natural consequence of the procedure. We evaluate \hp{} on a broad benchmark suite comprising 15 node-classification datasets spanning the full homophily spectrum and 7 GNN architectures. \hp{} achieves a mean stability rank of 1.49 compared to 2.31 for random $k$-fold, achieving the lowest mean stability rank on 13 of 15 datasets while preserving class balance close to class-stratified splits and substantially better than random. We argue that homophily-aware split construction merits broader adoption for GNN evaluation. 

---
# MeshKV: A Network-on-Chip KV Cache Fabric for Scalable Transformer Decoding Accelerators 

**Authors**: Dong Liu, Yanxuan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2609.19207)  

**Abstract**: Autoregressive transformer decoding is constrained by irregular key-value (KV) cache movement on tiled accelerators. Prior compression and DRAM-placement systems still concentrate traffic on centralized memory paths that bottleneck long-context serving. We present MeshKV, a KV cache fabric that moves blocks as packetized flows over a lightweight NoC. It co-designs (i) TaKV affine striping to spread homes and cut hotspot load, (ii) Mare multicast with verified duplicate suppression, and (iii) Pad, which overlaps prefetch, tile multiply, and streaming softmax behind credit-aligned FIFOs. Together they convert bisection back-pressure into useful KV transfer. On our 8x8 FPGA implementation with LLaMA-2-7B and Mistral-7B at 8K-32K, MeshKV reduces interconnect traffic by up to 58%, improves KV bandwidth utilization by 2.1x, and delivers up to 1.9x multi-stream throughput. 

---
# REACT: A Fully Spiking State-Space Model for Real-Time Event-Driven Temporal Perception 

**Authors**: Geoffroy Keime, Nicolas Cuperlier, Benoit R. Cottereau  

**Link**: [PDF](https://arxiv.org/pdf/2609.19204)  

**Abstract**: Robotic systems operating in dynamic environments require visual perception that evolves continuously with the incoming sensory stream. Event cameras provide microsecond temporal resolution and asynchronous sensing, but most learning-based methods accumulate events into frames or temporal bins, introducing an integration delay that can limit fast reaction. Here we propose REACT, a fully spiking state-space model for event-driven temporal perception that processes raw events one by one, without temporal accumulation. REACT uses a complex-valued spiking neuron, C-SiLIF, whose continuous-time dynamics are driven by the physical inter-event interval, allowing its internal state to evolve at the temporal resolution of individual events. We evaluate REACT on gesture recognition and time-to-collision (TTC) estimation from full-field event streams, without a target bounding box or localization input. On EvTTC, REACT achieves a 9.59% relative TTC error with 4.6 ms end-to-end inference latency, within 0.15 percentage points of the best learned method while requiring no target prior. At the dataset's mean approach speed, this latency corresponds to only 4 cm of vehicle motion, compared with 1 m for the fastest competing learned method. REACT further supports anytime TTC prediction, zero-shot transfer to a different driving sequence, and INT8 quantization, reducing the estimated energy consumption from 18.5 to 2.8 mJ per 32,768 events. These results show that event-driven spiking state-space dynamics can provide low-latency, continuously updated temporal perception for reactive robotic systems. 

---
# Code-as-Auditor: Executable Compliance Reasoning via Regulation-to-Code 

**Authors**: Jisoo Kim, Taeyoon Kwack, Jinwoo Jang, Woo Kyung Kim, Honguk Woo  

**Link**: [PDF](https://arxiv.org/pdf/2609.19199)  

**Abstract**: Large Language Models (LLMs) are increasingly adopted for compliance and legal reasoning tasks, yet their outputs often lack explicit grounding in legal logic and evidence. We present Code-as-Auditor, an LLM-based framework that extends the model's reasoning capability toward structured and evidence-grounded compliance assessment. The framework translates regulatory information into (1) formalized checklists and executable decision trees, encoding regulations and conditions as interpretable code structures. During inference, each checklist item is (2) dynamically expanded into factual and counterfactual questions, guiding the model to reason over case-specific evidence and potential violations. This process establishes a reasoning pipeline that proceeds from evidence identification, through rule application, to final decision-making, while a self-verification loop improves the logical consistency of the generated code and the traceability of outcomes. Experiments on privacy and data protection scenarios demonstrate that Code-as-Auditor delivers more accurate and evidence-backed evaluations, enabling automated compliance regulation checking grounded in explicit regulatory criteria. 

---
# Optimal Transport Metric Learning for Feature Alignment in Partially Supervised Segmentation 

**Authors**: Dakini Mallam Garba, Salim Abdou Daoura  

**Link**: [PDF](https://arxiv.org/pdf/2609.19176)  

**Abstract**: Multi-organ segmentation is often challenged by partially annotated datasets and domain shifts across different imaging sources. To address these limitations, we propose a two-stage learning framework that efficiently leverages partial supervision. In the first stage, the model learns from available annotations to produce accurate segmentations of annotated organs, establishing robust feature representations. In the second stage, we introduce learnable organ prototypes and a Sinkhorn-triplet loss to enforce organ-wise feature consistency across datasets. This encourages latent embeddings of the same organ to remain close, while increasing separation between different organs, even when annotations are missing. Our approach achieves performance comparable to state-of-the-art methods on the BTCV dataset, while remaining computationally efficient. By explicitly aligning feature distributions rather than relying solely on pseudo-labels, the framework effectively mitigates domain shift, making it particularly suitable for medical image segmentation tasks with limited annotation resources. 

---
