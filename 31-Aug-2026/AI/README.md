# Logos: An Agent Harness on a Cross-Process Bus 

**Authors**: Hanzhang Jia, Liheng Zeng, Hao Cheng, Yi Gao, Bo Ma  

**Link**: [PDF](https://arxiv.org/pdf/2608.28553)  

**Abstract**: Modern agent systems assemble capabilities at runtime, and this dynamic composition has recently received a complete formal treat ment in the spatiotemporal-composability calculus, in which a capability is a component carrying a tracked inverse, and agents are assembled as plugins. This plugin form is carried by a single process sharing one context, a carrier that places all components in one physical failure domain, a fault suspends every component at once, and process death interrupts every session the process hosts. This paper shows that neither the modeling nor the calculus binds an agent to one process, the statelessness of the language model keeps all cross-step state outside the model, and the soundness invariant is defined on the state space alone. These observations condense into four lemmas whose premises are the hypotheses of the calculus and the statelessness of language-model inference. On these lemmas this paper constructs Logos, a ROS-like cross process agent harness in which a plugin is a process and the only shared state is an append-only transcript. Eighty sessions resume with no repeated effect after kills placed at the four boundaries of the tool-call cycle, and a same-fault comparison with a single process reference configuration shows one fault interrupting every co-resident session while under the peer-process construction one fault ends at one node. 

---
# InstructMesh: Selective Refinement of Generative 3D Models for Fabrication 

**Authors**: Faraz Faruqi, Ahmed Katary, Demircan Tas, Theresa Hradilak, Ning Zhang, Jiaji Li, Fabian Manhardt, Martin Nisser, Vrushank Phadnis, Ruofei Du, Federico Tombari, Megan Hofmann, Stefanie Mueller  

**Link**: [PDF](https://arxiv.org/pdf/2608.28534)  

**Abstract**: Recent advances in generative AI allow users to create 3D models from text or images. However, these models prioritize visual plausibility over geometric accuracy, often generating results with flaws that compromise their intended use post-fabrication. We present InstructMesh, an interactive post-generation refinement tool that enables selective repair of generative 3D models through region selection and targeted operations, such as opening or sealing voids, or adjusting local thickness. Users can invoke edit operations via natural language prompts or slider controls. By operating directly on the intermediate latent representation, InstructMesh allows users to apply robust geometric corrections without requiring expert modeling skills. To inform our design, we first analyze common fabrication-related failure modes in outputs from state-of-the-art generative tools. We then conduct two user studies, demonstrating that novices can identify and perform fabrication-relevant repairs on generative outputs using InstructMesh, and revealing user preference for hybrid interfaces that combine slider controls with natural language input. 

---
# When Robots Mishear Us: Mapping the Safety Risks of Voice-Controlled Embodied AI 

**Authors**: Sihan Jia, Oliver Lemon  

**Link**: [PDF](https://arxiv.org/pdf/2608.28518)  

**Abstract**: We investigate whether automatic speech recognition (ASR) errors in user input can lead to unsafe outputs from Embodied AI (EAI) models. We find that ASR errors can lead to harmful instructions being accepted and executed by EAI models, thereby reducing safety. We simulate ASR errors and combine them with existing safety benchmarks (SafeAgentBench and POEX) to evaluate how different errors affect embodied AI safety. We find that some of them preserve semantic structure but increase harmful ambiguity, while others weaken the model refusal behaviour and allow unsafe plans to be generated and executed. We show that in some cases automatic correction of ASR errors can reduce the risk, but this is not always effective. Overall, we show that ASR errors lead to significant safety risks for embodied AI. 

---
# Training Communication-Efficient Mixture-of-Experts Language Models with Layer Re-Configuration 

**Authors**: Simeng Sun, Roger Waleffe  

**Link**: [PDF](https://arxiv.org/pdf/2608.28511)  

**Abstract**: When training Mixture-of-Experts (MoE) language models with expert parallelism, all-to-all token dispatch and combine collectives can consume a substantial fraction of end-to-end training time. In this work, we study communication-efficient MoE models (CE-MoE), in which we adopt a heterogeneous layer pattern that decouples token-mixing and channel-mixing depth. Compared to conventional models which interleave MoE layers after each token-mixing layer (e.g., attention, Mamba-2), CE-MoE models concentrate expert capacity in a select few routed MoE layers, while maintaining depth by adding additional token-mixing and dense-FFN layers. Across a scaling ladder from 2B to 31.5B total parameters, under matched total and activated parameters, CE-MoE models consistently reduce training cost while matching validation loss and downstream benchmarks with full-MoE baselines. At the 31.5B scale, CE-MoE uses 33.3\% fewer GPU-hours while improving average downstream score and inference throughput. 

---
# AcrossVAM1.0: Particle World Modeling for Text-Assisted Robot Video Prediction 

**Authors**: Yafei Zhang, Nan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2608.28491)  

**Abstract**: Predicting robot videos requires both precise motion reasoning and preservation of high-frequency appearance, yet monolithic pixel models entangle these objectives and often conceal their progress behind a strong last-frame baseline. We present AcrossVAM1.0, a lightweight, text-assisted video action model that factorizes future prediction into object-centric motion and dense appearance. A frozen SAM3-DLP codec decomposes four context frames into semantic particles for the robot, arm, and gripper, together with a background latent. A 0.28M-parameter spatio-temporal Transformer aligns particle identities, rolls their states forward, and is modulated by a frozen OpenCLIP instruction embedding through FiLM. A causal dual-stream decoder combines particle-rendered motion with appearance encoded exclusively from the last observed frame; a residual refiner and learned delivery mask produce five future frames without access to future appearance. On our VRS benchmark constructed from diverse real-robot trajectories, particle dynamics reduce trajectory error by 21.0\% over persistence. Across three delivery-mask seeds, AcrossVAM1.0 improves future-frame PSNR/SSIM from 19.97/0.796 to 20.573/0.8004, while raw particle generation improves motion-region PSNR from 11.89 to 13.23. The delivered model does not yet beat persistence in LPIPS, and correct-versus- shuffled language changes trajectory error by only 2.8--3.1%. We report these limitations alongside oracle, negative-control, multi-seed, and per-robot analyses. The results show that explicit particle dynamics are a promising low-dimensional interface for robot video prediction, while robust language grounding and appearance delivery remain the principal open challenges. 

---
# COVER: Identifiable Evaluation of Coalition Routing 

**Authors**: Raghul Sugumar, Amrit Gopinath  

**Link**: [PDF](https://arxiv.org/pdf/2608.28475)  

**Abstract**: When a multi-agent system changes its team, it also changes the messages and final answer it produces, so an end-to-end accuracy gap does not by itself identify a routing effect. We introduce method, an evaluation contract that fixes a public information boundary, downstream stack G, and finite legal team family before outcomes are generated. Complete coverage identifies exact finite-benchmark oracle regret conditional on that stack. For any finite collection of frozen policies, executing the union of their distinct selected teams is the minimal assumption-free support for every pairwise policy contrast, though not for absolute oracle regret. Two controlled tables with source-ID-disjoint splits test the instrument. On MuSiQue-12, a pre-specified privileged positive control improves regret from 0.532 to 0.402; a later public-interface control reaches 0.424 versus 0.554 but is retrospective. On HotpotQA-4, a pre-specified public direct scorer improves regret from 0.313 to 0.110. In fixed-stack Llama execution, verified route regret improves by 0.190, while the raw-answer gain is 0.010 with an interval crossing zero. A five-family ToolSandbox variant-shift validation exhaustively evaluates 16 declared teams on 14 untouched task variants (224/224 valid rows): the declared-family oracle reaches 0.768 safe-evidence completion, while the prospectively frozen router gets 0.637 (regret 0.131), failing the predeclared 0.10 criterion. A later retrospective comparator reaches 0.655, matching all-workers with 4.57 versus 5.00 workers on average. Thus COVER exposes selection headroom without manufacturing a routing win. A crossed-stack diagnostic shows absolute scores depend on G but finds no detectable router-by-finalizer interaction. COVER is an auditable measurement methodology, not a claim of stack-invariant or universal agent-routing superiority. 

---
# Learning to Use Tools: Reinforcement Learning for Tool-Integrated Mathematical Reasoning 

**Authors**: Minghui Xu, Zi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.28447)  

**Abstract**: Current large language models (LLMs) increasingly benefit from external tool integration, especially for tasks requiring reliable computation and verification. Motivated by this, we study calculator tool calling for improving mathematical reasoning on the Countdown task. We first analyze reasoning failures and find that calculation errors account for a substantial portion of incorrect responses. We then construct supervised fine-tuning datasets to teach the model useful tool-use patterns and how to interpret returned outputs. Building on this tool-formatted policy, we apply several on-policy reinforcement learning methods, including RLOO, RLOO++, GRPO, and DAPO, using automatically verifiable final-answer rewards. To enable a more reliable evaluation, we construct a fresh 1,024-problem held-out Countdown benchmark with no exact overlap with the training data. Our results show that calculator tool integration consistently improves both SFT and RL baselines, yielding roughly 10 percentage-point gains across pass@k. Among the RL methods, Tool-DAPO achieves the strongest performance, improving pass@1 from 35.8% for Tool-SFT to 66.0%. Further analysis shows that RL encourages more effective tool use even when only final-answer rewards are provided. These findings suggest that tool integration reduces arithmetic and verification errors, while RL increases the probability of correct reasoning traces. 

---
# Prove2Me: An Open Collaborative Platform for Scaling Math Formalization 

**Authors**: Shuze Chen, Kunal Marwaha, Xiaoyang Lu, Henry Yuen, Tianyi Peng  

**Link**: [PDF](https://arxiv.org/pdf/2608.28433)  

**Abstract**: Proof assistants such as Lean 4 promise the paradigm of formally verified mathematics, but large-scale formalization projects have faced major barriers to entry, including the need for expertise in formal verification (as well as the underlying mathematics) and the significant time required for writing formal proofs. AI coding agents have dramatically reduced these barriers; human users can now use natural language to prompt agents to write complex proofs in Lean. This opens up the intriguing possibility of internet-scale mathematical collaboration involving both humans and AI agents, where correctness is machine-checked.
To realize this possibility, we introduce Prove2Me (this https URL), an open collaborative platform for formalizing mathematics. Users launch formalization "missions", to which AI agents contribute formal proofs toward completion. We designed mechanisms and a specialized harness in Prove2Me that enable large-scale collaboration so that agents can build on one another's work and freely reuse existing results. In doing so, Prove2Me aims to turn math formalization into a scalable, crowd-sourced effort open to anyone with an agent. 

---
# Program Learning with Verifiable Rewards: Symbolic Backpropagation for Post-Training LLMs 

**Authors**: Vishvesh Bhat  

**Link**: [PDF](https://arxiv.org/pdf/2608.28421)  

**Abstract**: Post training a language model to reason means updating its weights. Supervised finetuning and reinforcement learning both place the acquired capability inside the model where it cannot be inspected cannot be checked step by step and cannot be moved to another model. We argue that for tasks whose intermediate steps admit verification, reasoning is better placed outside the base models weights as an explicit program composed from deterministic and neural primitives. We introduce PLVR (Program Learning with Verifiable Rewards): a post training method that learns such programs directly from input-output examples. Its mechanism is symbolic backpropagation: each program layer carries a typed ontology a loss is computed at the output against ground truth and required input ontologies are propagated backward by type inference over primitive signatures: an analogue of the chain rule in which credit assignment is a derivation rather than an estimate. Where RLVR verifies a terminal outcome, PLVRs reward is a per step contract verdict dense over program structure. On LiveCodeBench v6 and Tau2Bench, 30B base models with PLVR outperform RL at matched budget by 27.8 points on average and frontier models an order of magnitude larger by 13.6 points. A single primitive library serves two benchmarks, so the marginal cost of a new task is 100 examples of program search and no new finetuning data. Replacing the loss guided search with uniform sampling over the same type admissible space at equal budget collapses the median program from 65.6 to 17.5, identifying the backward pass rather than the type system as the source of the advantage. We release the symbolic backpropagation library and a conformance checker so the method can be applied to primitive libraries other than our own. 

---
# VERA-8B: Evidence-Grounded Audit Risk Reasoning from SEC Filings 

**Authors**: Menghan Liu, Elynn Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.28402)  

**Abstract**: Across audit applications, judgments must be supported by reasonable evidence. However, standard financial language models prioritize fluency over evidence. They are built for general financial reasoning and may produce plausible but ambiguous answers, creating a grounding gap that makes them unsuitable for audit work. We address this gap with VERA-8B, a new end-to-end audit reasoning system that identifies audit risks before enforcement actions occur. Constructing such a model raises several challenges, as no prior machine learning work targets pre-enforcement audit prediction. To our knowledge, we are the first to unify SFT and GRPO for evidence-grounded audit reasoning under one evidence standard, achieving performance that surpasses all evaluated baselines. Because auditing cannot tolerate unsupported claims, we introduce abstention and uncertainty qualification to defer uncertain or evidence-incomplete cases. Finally, we design an AuditBridge to ground model reasoning for practical audit work. It transforms raw filings into verified records and then into reviewer-ready reports, bridging finance and computation with broad generality. Together, these components produce auditable, review-ready outputs suitable for practical audit work. 

---
# RetailAgent: Structured Adverse Timing in Self-Conditioned Multimodal LLM Trading Agents 

**Authors**: Yupeng Zhang, Liuyuan Jiang, Hongyi Huang, Bingheng Li, Lisha Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.28399)  

**Abstract**: In financial markets, a sequential policy that reacts systematically to price movements may become predictable to other market participants. This paper studies whether large language model (LLM) agents exhibit such directional structure through RetailAgent, an experimental framework in which an LLM observes anonymized intraday equity price histories and permitted state, then repeatedly chooses long (hold the stock) or flat (stay out) before the subsequent interval return is revealed. We compare returns during long and flat intervals along the same stock's intraday path after removing the overall fraction of long decisions. This exposure-matched measure reveals persistent negative timing across modality, horizon, state, and model family. Shuffling saved action sequences substantially attenuates the effect, showing that alignment between actions and subsequent returns drives the negative score. Feeding self-authored memories into decisions further increases policy persistence, while timing becomes more negative among stock-days on which the agent uses both actions. These results reveal stable, recoverable directional structure in sequential LLM financial decisions and a behavioral signal for studying how another participant could respond to a predictable policy. 

---
# Timing-Aware Repurchase Prediction for Web-Scale E-Commerce: Survival Models for Multi-Surface Grocery Recommendation 

**Authors**: Akshay Kekuda, Shreeranjani Srirangamsridharan, Ishan Bhatt, Yanan Cao, Sinduja Subramaniam, Evren Korpeoglu, Kaushiki Nag, Kannan Achan  

**Link**: [PDF](https://arxiv.org/pdf/2608.28393)  

**Abstract**: Repurchase recommenders in e-commerce are commonly framed as a binary question asking "will this customer buy this item within W days", a formulation that requires a separately trained model for every horizon of interest. We replace this stack with survival models that predict time-to-repurchase directly, and evaluate them on millions of customers from a major grocery e-commerce platform across more than thirty ablation configurations. Our study makes three contributions. First, an empirical hazard analysis reveals a slightly decreasing marginal hazard (k ~ 0.9), differing from the common intuition that grocery items become more likely to be repurchased the longer since the last purchase (increasing hazard, k > 1). Log-Normal achieves the best marginal fit (R^2 = 0.998) and the best ranking, despite Weibull providing the best conditional residual fit, revealing an apparent discrepancy we analyze in detail. Second, a single Accelerated Failure Time (AFT) model replaces three per-horizon binary classifiers, matching or exceeding each at its own horizon while using roughly 3x fewer total trees. Feature importance reshuffles under the survival objective: channel-cadence and recency signals rise while aggregate frequency counts fall. Third, a 4-parameter parametric calibration maps raw survival CDFs to per-horizon probabilities with zero cross-horizon monotonicity violations. Calibration quality varies by an order of magnitude across the AFT family: Exponential AFT (Weibull k=1) achieves expected calibration error (ECE) ~1e-4, roughly 10x lower than Log-Normal, while ranking metrics agree within 0.3% relative. We adopt Exponential AFT for probability-consuming surfaces and Log-Normal for pure ranking, exposing a principled calibration-ranking trade-off within a single AFT family. 

---
# MAP: A Benchmark on Multimodal Accessibility Planning for Real World Places 

**Authors**: Jason Armitage, Ioannis Tsochantaridis, Linda Mazzone, Chuqiao Yan, Srini Narayanan, Sarah Ebling  

**Link**: [PDF](https://arxiv.org/pdf/2608.28384)  

**Abstract**: We introduce MAP, the first benchmark to evaluate multimodal AI systems as assistants for users with accessibility requirements when planning visits to places in the real world. In our evaluation, systems are presented with requests to verify or recommend a point of interest meeting an accessibility requirement. MAP contains two novel assessments: Claim verification for accessibility planning assesses if information on places and stated accessibility features is supported and identifies places that satisfy requested accessibility features. Visual evidence retrieval for accessibility planning checks if a multimodal AI system can select visual evidence for the requested place and accessibility feature. Our methodology supports comparison of AI systems in a setting where place information and accessibility information can change over time by evaluating systems and refreshing ground truth data at scheduled times. The benchmark is based on automatic rating and human rating for a proportion of responses. 

---
# EvoUndo: Recoverability-Constrained Self-Evolution for LLM Agent Harnesses 

**Authors**: Tanmay Sah, Dolly Sah, Harshul Jain, Tanya Sah  

**Link**: [PDF](https://arxiv.org/pdf/2608.28363)  

**Abstract**: LLM agents increasingly modify their own prompts, tools, middleware, resources, and execution harnesses at runtime. Such self-evolution can improve capability, but a successful mutation may leave persistent effects that cannot be safely reversed in states different from the one in which it was created. We introduce EvoUndo, a framework for representing, synthesizing, diagnosing, and independently verifying recoverability of model-generated self-modifications across counterfactual states. Across 600 unseen one-shot self-evolution tasks, we identify 197 capability-improving mutations that fail recoverability verification. Under the original recovery representation, conventional repair strategies recover 0/197 of these natural failures. Deterministic oracle analysis recovers 48/197 under the original recovery language L0, while the extended recovery calculus increases empirical oracle recovery to 191/197. A protocol-locked 2x2 grounding-by-expressivity intervention then separates two bottlenecks: exact state-address grounding increases successful recovery from 0/48 to 38/48 (79.2%) when the original language is sufficient, while extending the recovery language enables recovery on 142/143 (99.3%) failures in the oracle-defined S1 stratum. On the primary gpt-oss-120b backbone, adding exact-address diagnostics to the richer language reduces recovery to 133/143 (93.0%); a Qwen3.8-27B replication preserves the grounding and expressivity effects but not this negative interaction, indicating that the latter is model-dependent. These results indicate that reliable agent self-evolution requires co-designing verification, state grounding, witness semantics, and recovery-language expressivity rather than relying on iterative prompting alone. 

---
# GRACE:Gradient-guided Coreset Selection for LLM Unlearning 

**Authors**: Praveen Bushipaka, Andrea D'Angelo, Lucia Passaro, Tommaso Cucinotta  

**Link**: [PDF](https://arxiv.org/pdf/2608.28361)  

**Abstract**: Machine Unlearning methods for Large Language Models typically assume pre-specified forget and retain sets. In realistic settings, however, requests may provide only a few examples of undesired behavior, requiring forget and retain sets to be inferred from heterogeneous corpora. We study this data-selection problem and propose GRACE , a gradient-guided coreset selection method that constructs both forget and retain sets for LLM unlearning. GRACE first computes a forget direction from seed examples that elicit the undesired behavior, then selects a compact forget coreset whose gradients approximate this direction using non-negative orthogonal matching pursuit. To preserve model utility, it selects retain examples after projecting out the forget direction and applying clustered orthogonal matching pursuit in the remaining gradient space. Across two target domains, two model families, and four unlearning algorithms, GRACE improves model utility while maintaining comparable forget quality, with particularly consistent gains over prior gradient-based selection methods. 

---
# Propagating construction-time knowledge quality into medical question answering: A framework grounded in clinical guidelines 

**Authors**: Jie Hu, Junjie Wang, Shan Lu, Yifang Hu, Gong Cheng, Yun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.28360)  

**Abstract**: Large language models have facilitated knowledge graph (KG) construction from clinical guidelines, but extracted triples vary in structural validity and evidential support. Meanwhile, graph-augmented question answering (QA) systems typically optimize query relevance during retrieval, with limited reuse of quality information produced during KG construction. This creates a disconnect between construction-time quality control and inference-time evidence use. We investigate whether construction-time triple quality can serve as a persistent signal for downstream evidence selection and presentation. We propose a quality-aware framework that models structural conformance (SchemaConf) and evidential support (EvidScore) as complementary dimensions and fuses them into a per-triple quality signal, Q(t). Rather than using quality solely for filtering, the framework retains Q(t) and derived quality tiers as graph attributes and propagates them into quality-weighted subgraph retrieval and tier-conditioned evidence prompting, while preserving passage-level provenance. Experiments on Chinese diabetes clinical guidelines show that the utility of the quality signal is distribution dependent. Under cross-version and cross-model shift, the fused Q(t) provides stronger triple-quality discrimination than either component alone (AUC 0.748 vs. 0.703 for EvidScore and 0.645 for SchemaConf). In guideline-grounded QA, propagating construction-time quality reduces required-knowledge omission from 16.3% to 5.3% and conflicting outputs from 16.3% to 2.7%, with an evidence-grounded precision of 81.6% and near-zero invalid citations. Blinded clinician ratings favor the full framework over no retrieval (4.68 vs. 4.21 on a five-point scale) and approach the oracle condition (4.80), while cross-generator experiments show consistent trends. 

---
# AGENT-O: A Semantic Agent Card Framework for Interoperable and Governed Healthcare AI Agents 

**Authors**: Pengze Li, Cui Tao  

**Link**: [PDF](https://arxiv.org/pdf/2608.28345)  

**Abstract**: AGENT-O is a modular ontology framework that defines a semantic Agent Card for representing health-oriented AI agent systems and supports assessment of reporting completeness in scientific publications. AGENT-O was developed as an OWL 2/RDF ontology covering runtime, models, workflow, tools, clinical use, evaluation, provenance, governance, and reporting assessment. Evaluation included ontology inventory, OWL-RL reasoning, three SHACL suites, 12 SPARQL competency queries, three cases, and model-assisted reporting-completeness assessment of 279 papers across five dimensions. The ontology contained 1,962 RDF triples and 1,922 Protege axioms, with 252 active classes, 198 active object properties, and 51 datatype properties. All SHACL suites conformed on example graphs, all competency queries returned prespecified evidence, and all 279 papers were scored. Incomplete reporting was highest for runtime/architecture (84.6%), governance/safety (82.8%), and provenance/reproducibility (78.1%), compared with evaluation (25.8%) and benchmark-process alignment (29.8%). AGENT-O supported semantic Agent Card representation and reporting assessment while revealing an evaluation-specification gap: evaluation and benchmark procedures were reported more consistently than runtime architecture, governance, and reproducibility. AGENT-O provides a reusable ontology, semantic Agent Card profile, and reporting-completeness workflow for structured reporting and gap identification, but does not assess agent quality or deployment readiness. 

---
# Real-Valued Hyperdimensional Sequence Representations with Hadamard Product Binding and Shift Equivariance 

**Authors**: Kenny Schlegel, Dmitri A. Rachkovskij, Denis Kleyko, Amy Loutfi, Stefan Streif, Evgeny Osipov  

**Link**: [PDF](https://arxiv.org/pdf/2608.28334)  

**Abstract**: Encoding temporal order is a fundamental requirement for sequence representations in Hyperdimensional Computing. Fractional Power Encoding provides similarity-preserving position vectors whose inner products approximate shift-invariant kernels, and it supports shift-equivariant transformations of encoded sequence representations. However, standard formulations of Fractional Power Encoding are primarily designed for binding operations such as circular convolution or complex-valued multiplication, which limits their compatibility with Hadamard product binding of real-valued vectors. This paper develops real-valued position encodings motivated by Random Fourier Features, aiming to retain the desirable properties of Fractional Power Encoding while supporting Hadamard-based operations. We propose three real-valued position-encoding variants: a real-valued baseline based on the inverse Fourier transform, and Sinusoid and Cosine-only representations derived from Random Fourier Features. Among them, the Sinusoid variant provides an explicit algebraic shift operator, allowing temporal shifts to be applied directly to the vector-encoded sequence representation without re-encoding the shifted sequence. Experiments on time-series classification datasets show that the proposed real-valued representations achieve performance comparable to standard Fractional Power Encoding while enabling computationally efficient Hadamard product binding. The Sinusoid variant offers the most favorable trade-off, combining efficient real-valued implementation with exact shift-equivariant transformations. 

---
# MAIL: Memory-driven, Adaptive, Incremental, and Literature-grounded Framework for Hypothesis Generation in Chemistry 

**Authors**: Mahdi Babaei, Xueshen Li, Yutao Kuang, Jolene P. Reid, Yu Gan  

**Link**: [PDF](https://arxiv.org/pdf/2608.28315)  

**Abstract**: The ever-expanding volume of the chemical literature offers unprecedented opportunities to generate novel and impactful hypotheses. However, the bottleneck lies in efficiently navigating this vast knowledge base to formulate high-quality, experimentally meaningful insights.
While Large Language Models (LLMs) show promise for this task, existing methods often rely on static inspiration corpora, predefined heuristics, or laborious human-in-the-loop pipelines and decision-support frameworks that limit scalability and novelty. In this work, we propose an automated approach, a Memory-augmented, Adaptive, Incremental, and Literature-grounded (MAIL) framework for hypothesis generation in chemistry. Our MAIL method formulates hypothesis generation as a temporally grounded, memory-driven reasoning process, where hypotheses emerge from an evolving conceptual path that continuously accumulates and reinterprets prior knowledge. We evaluated the MAIL framework on a public TOMATO-Chem dataset and a newly curated and disseminated high-novelty nature/science challenge (HN-NS) dataset. Across both datasets, MAIL generates structurally coherent and mechanistically plausible hypotheses, achieves the highest MIOS and MPOS by more effectively recovering the central ideas and methodological elements of the historical target hypotheses, and obtains the highest overall expert-evaluation scores for scientific quality. These results demonstrate the potential of LLMs to autonomously explore chemical domains and generate hypotheses that are both innovative and chemically plausible. 

---
# Memristive-Friendly Hadamard Reservoir Computing: Structured, Multiplier-Free Recurrences at Scale 

**Authors**: Andrea Ceni, Gianluca Milano, Carlo Ricciardi, Claudio Gallicchio  

**Link**: [PDF](https://arxiv.org/pdf/2608.28295)  

**Abstract**: Reservoir Computing (RC) designs Recurrent Neural Networks around a fixed, i.e., untrained, recurrent layer, and is a natural candidate for neuromorphic hardware. Memristive-friendly reservoirs derive the neuron dynamics from memristive-device kinetics, but still rely on dense recurrent matrices, which are expensive to realize physically. In this paper, we replace the dense matrix with a structured orthogonal operator, built from sign diagonals, a permutation, and a fast Walsh-Hadamard transform. The operator is multiplier-free, requires $O(N)$ parameters and $O(N\log N)$ operations per step, and is never materialized as a matrix. We instantiate it in a standard and in a memristive-friendly Echo State Network, with one binary input connection per unit.
Our mathematical analysis shows that exact orthogonality yields an echo state condition that is tight in the recurrent scaling, and a noise response that is predictable at design time. Moreover, the operator mixes the whole state in a single application. Experiments on twenty classification and seven regression benchmarks, at reservoir sizes up to $N = 8192$, show that the structured models match dense orthogonal reservoirs, and achieve better mean performance than the cycle reservoir by a margin that widens with size. Furthermore, we time the recurrent step on three hardware platforms, where it is up to $50\times$ faster than a dense product and $10^4\times$ smaller in memory. Finally, we ablate the operator and measure the response to noise, quantization, device mismatch and discrete faults. 

---
# LoopArena: Benchmarking Models as Runtime Controllers for Loop Engineering 

**Authors**: Yi Wang, Haopeng Zhang, Chengxiang Huang, Rui Dai, Kaikui Liu, Piotr Koniusz, Xiangxiang Chu  

**Link**: [PDF](https://arxiv.org/pdf/2608.28281)  

**Abstract**: Loop Engineering is emerging as a practice for organizing development work around coding agents. Instead of writing each prompt by hand, practitioners design loops that monitor progress, assign work, run checks, and decide what the agent should do next. Even with a capable coding agent, a loop may trust a stale progress note, skip needed verification, spend its budget in the wrong direction, or stop before the task is safe to submit. Yet the final outcome of one end-to-end run cannot tell whether success or failure reflects the loop's guidance or the coding agent's ability to carry out the task. We introduce LoopArena, a benchmark for evaluating how well one model can guide a separate coding agent through a long-running task. The model under evaluation is the \textbf{Controller}: after each coding round, it receives a structured summary of the run and instructs a separate, fixed coding agent, the \textbf{Worker}, on what to do or verify next, or decides whether to stop. LoopArena evaluates this ability in three complementary settings that differ in execution scope and cost. Type I scores next-step Loop Contract selection through execution-validated questions without running the Worker at evaluation time. Type II executes repeated control over a selected slice of a full task, while Type III evaluates the paired full task from its original state. On full tasks, the best observed Strict Success Rate is \textbf{24.69\%}, leaving substantial room for improvement in long-horizon loop control. Across Controllers, the paired reduction in estimated inference cost averages \textbf{64.4\%}, and Type II produces a similar ordering under the main Core criterion (Spearman's \(\rho=\textbf{0.9747}\)). We release the benchmark data and evaluation code at this https URL . 

---
# RECAST: Recent & Context-Aware Sampling for Test-Time Adaptation in Streaming Biosignals 

**Authors**: Yong-Yeon Jo, Junho Song, Joon-myoung Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2608.28271)  

**Abstract**: Streaming biosignals vary across subjects and drift over time, so population-trained models lose accuracy during long-term monitoring. Test-time adaptation (TTA) enables online personalization by updating the model on incoming samples. But in a stream, a basic question is left open: \emph{which samples should drive each update?} Using all buffered samples blurs the update with irrelevant segments. Using only the latest segment makes the update noisy and unstable. The most useful samples are recent, aligned with the current physiological state, and reliable enough to learn from. We propose \textbf{RECAST} (REcent \& Context-Aware Sampling for TTA), a lightweight sampling module for buffered TTA frameworks. RECAST builds each adaptation batch from three signals: temporal recency, contextual similarity, and predictive reliability. It changes only which samples are used, leaving the model and the training objective unchanged. On two blood-pressure datasets, RECAST improves estimation accuracy and trend tracking over baselines and ablations. The per-patient gains are statistically significant on both datasets, with broad improvement on the regular benchmark and gains concentrated on the hardest patients in the emergency-department setting. RECAST stays practical, adding only sub-second latency per segment on a single GPU and CPU core. 

---
# Finding Where the Buck Stops: An Automated Failure Attribution-Based Reflection Framework for Multi-Agent Collaboration 

**Authors**: Xiaoqing Wang, Keman Huang, Bin Liang, Hongyu Li, Xiaoyong Du, Wuqiong Pan  

**Link**: [PDF](https://arxiv.org/pdf/2608.28264)  

**Abstract**: Multi-agent systems (MAS) powered by large language models have shown promise for complex tasks but suffer from high failure rates. Current self-reflection methods for MAS require all agents to reflect upon failure, overlooking a critical reality: failures typically stem from a specific agent leading the task astray, namely the decisive error agent, while others merely fulfill their regular duties. Forcing regular-behaving agents to reflect contaminates their memory with wrong insights. Hence, we propose DoCtOR (Diagnose-then-Correct PPO-enhanced Reflection), a novel reflection framework that enhances multi-agent collaboration. DoCtOR first identifies the decisive error step and decisive error agent through automated failure attribution, then employs counterfactual reasoning to generate a corrected decisive error step, and finally engages only the decisive error agent to produce targeted reflections. Experimental results show DoCtOR achieves 22%, 26%, and 27% improvements over initial success rates on HotPotQA, ChartQAPro, and Mind2Web datasets, outperforming Reflexion, Retroformer, and COPPER. We further establish the generalizability of our diagnose-then-correct paradigm and demonstrate that in low-resource settings, focusing reflection on reasoning steps after the decisive error step achieves comparable quality to reflecting on the complete failure trajectory. 

---
# Physics-Guided Flow Matching for CT Image Reconstruction 

**Authors**: Davide Evangelista  

**Link**: [PDF](https://arxiv.org/pdf/2608.28256)  

**Abstract**: Deep generative models have recently emerged as powerful priors for solving ill-posed inverse problems in CT, with diffusion-based approaches achieving state-of-the-art reconstruction performance. However, diffusion models typically rely on stochastic sampling procedures, long inference trajectories, and carefully tuned noise schedules, which can limit computational efficiency and numerical stability, especially at high spatial resolutions. In this work, we investigate Flow Matching as an alternative generative prior for CT reconstruction. We train a high-resolution Rectified Flow Matching model on 256x256 chest images from the Mayo Clinic Low-Dose CT dataset. To mitigate overfitting and limited anatomical variability, we employ a two-stage training strategy consisting of an initial phase with strong, anatomically informed data augmentation, followed by a fine-tuning phase with reduced or no augmentation to refine structural fidelity. The resulting model is capable of generating high-quality and anatomically coherent CT-like images, serving as a strong learned prior. We then evaluate multiple reconstruction methods specifically designed for Flow Matching models, including Plug-and-Play Flow, FlowDPS, Flower, and Flow-Priors (ICTM), and compare them against state-of-the-art diffusion-based reconstruction algorithms such as DDRM, DPS, and DiffPIR. Experimental results across several CT inverse problem settings show that Flow Matching-based approaches consistently outperform diffusion-based methods in terms of PSNR, SSIM, and perceptual quality, while requiring fewer sampling steps. Finally, we publicly release the trained Flow Matching model and accompanying code to facilitate reproducibility and future research. Overall, this work demonstrates that Flow Matching provides a stable, efficient, and effective alternative to diffusion models for high-resolution CT image reconstruction. 

---
# Regime-Aware Portfolio Management via Retrieval-Augmented LLM-Guided Expert Switching 

**Authors**: Ahmad Asadi, Reza Safabakhsh  

**Link**: [PDF](https://arxiv.org/pdf/2608.28252)  

**Abstract**: Financial markets are inherently non-stationary, making the effectiveness of individual portfolio-management strategies highly dependent on changing market conditions. This work proposes a retrieval-augmented expert-switching framework that dynamically selects portfolio management experts based on their historical performance under similar market situations. A dual-stream variational autoencoder represents asset-level and market-wide information, while a retrieval-based knowledge base stores historical situations and expert performance. During inference, an instruction-tuned LLM reasons over the retrieved evidence to identify the most appropriate expert rather than directly generating portfolio actions. We further establish a monotonicity property showing that adding a locally superior expert cannot degrade the switching mechanism's performance. Experiments across cryptocurrency, stock, and foreign-exchange markets show that the proposed selector achieves the highest cumulative return and Sharpe ratio among the evaluated selection strategies in all three markets. In the stock market, for example, cumulative return increases from 26% for the best fixed expert to 34%, while the Sharpe ratio improves from 0.74 to 0.96. Ablation results confirm the importance of both retrieval and LLM reasoning, while experiments with different expert-pool sizes demonstrate the value of complementary expertise. Overall, the findings support retrieval-grounded expert switching as an effective approach to adaptive portfolio management in non-stationary financial environments. 

---
# Beyond Task-Only Matching: Personalized Skill Routing with Counterfactual Evaluation 

**Authors**: Tianle Wang, Yanghe Zou, Xiang Liu, Ziyao Huang, Chenchen Fu, Weiwei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2608.28241)  

**Abstract**: The rapid expansion of reusable skill repositories makes skill routing a critical capability for large language model (LLM) agents. Existing methods treat routing as task-only semantic matching. However, when users with incompatible constraints issue an identical request, this assumption conflates task relevance with skill suitability: a task-only router can select a semantically plausible skill that is unsuitable for the requesting user. To expose this failure mode, we formulate \textit{personalized skill routing} as profile-conditioned retrieval, in which relevance depends jointly on the task and the user profile. We first introduce a profile-counterfactual benchmark, in which the task is held fixed while changes in the user profile induce changes in the reference skill. We further construct paired counterfactual supervision and propose SkillFeed, a progressive retrieve-and-rerank framework that first establishes task--skill alignment and then learns profile-conditioned discrimination. By retrieving body-level evidence and reranking semantically similar but profile-conflicting candidates, SkillFeed identifies skills that satisfy both task requirements and user constraints. On SkillFeed-Bench, SkillFeed attains 75.1\% top-1 retrieval accuracy, a 23.1-point improvement over the corresponding pretrained routing baseline. Adding profile conditioning yields a 35.1-point gain on queries where user profile changes the reference skill. This contrast shows that user profiles are most consequential precisely when they change skill suitability. Our website is publicly available at this http URL . 

---
# REINS: Refusal-Enhanced Inhibitory Steering with Sparse Autoencoder Features 

**Authors**: Kai-Xuan Ding, Hao-Xiang Xu, Ji-Hua Peng, Zi-Qi Chen, Jiaqi Wang, Zhen-Hua Ling  

**Link**: [PDF](https://arxiv.org/pdf/2608.28233)  

**Abstract**: Steering with Sparse Autoencoders (SAEs) offers a lightweight inference-time path for adapting the behavior of large language models without retraining. By exposing sparse and interpretable features, SAE steering provides a promising interface for safety control that guides harmful continuations toward refusal. However, we observe that complex wrappers can still undermine existing SAE steering methods on harmful prompts. To evaluate this failure mode systematically, we construct Generalized Undercover Instruction Safety Evaluation (GUISE), a dataset of harmful prompts with complex wrappers. Existing single direction SAE steering methods do not reliably produce refusals on harmful prompts, suggesting that refusal enhancement alone can be too weak when the harmful continuation path remains active. This motivates us to propose Refusal-Enhanced INhibitory Steering (REINS), which suppresses harmful continuation features and enhances safe refusal features in the same SAE feature space. Experiments on GUISE and other datasets show that prior methods either intervene too weakly or achieve only apparent safety through collapse, while REINS substantially reduces harmful responses, markedly improves safe refusals and largely preserves general capabilities. 

---
# Stay Within Your Bounds: Distance-Guided Decoding for Guaranteed Context-Free Grammar Compliance 

**Authors**: Vincenzo Collura, Karim Tit, Eleonora Giunchiglia, Mike Papadakis, Maxime Cordy  

**Link**: [PDF](https://arxiv.org/pdf/2608.28229)  

**Abstract**: Grammar-constrained decoding helps large language models produce syntactically valid structured outputs, such as code, JSON, and SQL. For context-free grammars, many practical decoders enforce local prefix feasibility: each token must keep the current prefix extendable to some valid completion. Yet, under tokenizer-grammar mismatch and finite token budgets, feasible prefixes may still fail to reach acceptance. We propose a lookahead-guided decoding framework for context-free grammars based on pushdown automata. Offline, we compute bounded pushdown summaries with reachability labels and upper-bound distances to acceptance. Online, these estimates guide horizon-aware pruning and beam search. The resulting decoder is syntactically sound: every output is accepted by the target grammar. Experiments on JSON, SQL, and Linear Temporal Logic (LTL) show both consistent syntactic validity and improved completion quality over existing baselines. 

---
# Generative AI Alignment with Hinduism's Theological Plurality and Sacred Representation 

**Authors**: Dipto Das, Arpita Kundu, Nusrat Jahan Mim, Shion Guha, Syed Ishtiaque Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2608.28228)  

**Abstract**: Generative AI systems are increasingly used to answer personal questions and mediate everyday practices, including religion. However, existing discussions around AI alignment and ethics have largely centered secular, Western, and Abrahamic assumptions about religion, offering limited attention to other faith-based traditions. In this paper, we examine how Hindu users engage with generative AI systems in relation to their religious knowledge, belief, and practice. Drawing on 15 semi-structured interviews with Bangladeshi Hindu participants, we analyze how users interpret AI-generated religious representations, scriptural explanations, devotional interactions, and synthetic religious media. We found that AI can be both accessible and ethically troubling. While AI supported scriptural inquiry, devotional visualization, and religious storytelling, our study also identified concerns about theological flattening, cultural misrepresentation, devotional manipulation, and the simulation of sacred presence and authority. We conclude by arguing that religious alignment in generative AI requires interpretive alignment: systems that disclose their limits, preserve plurality, and avoid simulating sacred authority and sycophantic personalization. 

---
# Expert Knowledge & Machine Understanding: Bridging Reactome's Ontology with LLM Semantic Embeddings 

**Authors**: Susanna Bravi, Riccardo De Luca, Rosa Sicilia, Christine Nardini, Mario Santoro  

**Link**: [PDF](https://arxiv.org/pdf/2608.28178)  

**Abstract**: Biological knowledgebases like Reactome provide high-quality pathways that include biological elements' relationships and textual descriptions (metadata). The quality of such pathways is granted by manual curation, that presents, however, significant scalability challenges. Lately, numerous NLP tools have been proposed to cope with this issue, leveraging textual information to automatically expand biological knowledgebases. However, little exploration has been done so far to assess whether relationships among textual descriptions mirror higher order biological relationships. This study explores whether human-written descriptions in Reactome can be used to infer the experts' defined global hierarchical structure. To test this, we extracted from Reactome the Homo Sapiens hierarchy of pathways and their reactions (Reactome Hierarchy), and used textual metadata to reconstruct a Semantic Hierarchy, combining a sentence transformer model (SPECTER2) with a modified agglomerative nesting algorithm and a graph reconstruction algorithm. Quantitative (Laplacian Spectral Distance and Bootstrapping) and qualitative (global topological metrics) analyses confirm our hypothesis and indicate that the global hierarchical structure of pathways can be inferred by experts textual metadata. 

---
# CrabOS: An Operating System for Human-AI Co-inhabitation 

**Authors**: Qi Yang, Yun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2608.28165)  

**Abstract**: AI agents are evolving into long-running computational entities that can invoke tools, maintain memory, and complete complex tasks across applications. In real-world settings, completing a task often requires humans and AI to take turns leading its execution. Such alternation depends on the seamless handoff of the work state of the task between humans and AI. Existing agent systems, however, provide humans and AI with separate work environments. AI agents must therefore rely on additional bridges to continue work: either developers build task-specific interfaces to access the work state, or users manually transfer relevant parts of it through screenshots or textual descriptions. Both approaches make handoffs costly and scale poorly.
We propose Human-AI Co-inhabitation, a type of work environment that enables humans and AI to seamlessly take turns continuing work on the same task, and design and implement CrabOS to realize this concept. CrabOS represents the work state as natural-language-readable text objects shared by humans and AI, allowing both to access and manipulate it directly through the same auditable interface without bridges. Case studies show that CrabOS elevates support for complex tasks with alternating human and AI leadership from bridge-dependent application-level solutions to native operating-system capabilities, which provide a new foundation for developing and running AI agents. 

---
# Under-Mattress Temporal Sensing for Next-Day Agitation Risk Scoring in Dementia Wards 

**Authors**: Zhen Liu, Marta Bono, Robbe Decloedt, Ajda Flisar, Maarten Van Den Bossche, Maarten De Vos  

**Link**: [PDF](https://arxiv.org/pdf/2608.28152)  

**Abstract**: Agitation fluctuates over short time horizons in people living with dementia, yet continuous physiological information for anticipating next-day risk is limited. We assessed whether contactless under-mattress signals from the preceding night inform next-day agitation risk and whether preserving minute-level temporal structure improves performance over conventional nightly summaries. We analyzed 423 patient-nights from 65 subjects in a specialized hospital dementia unit using two under-mattress sensing systems. A unified four-paradigm benchmark compared nightly handcrafted summaries, three-period handcrafted features, full-night sequence modeling, and sliding-window multiple-instance learning. Source-specific preprocessing and five-fold patient-grouped cross-validation were used, with performance estimated from pooled out-of-fold predictions. Evaluation included discrimination, calibration, fixed-threshold metrics, and a comparison of period-signal attribution patterns across two temporal models. Full-night sequence modeling achieved the highest discrimination (AUROC, 0.692; AUPRC, 0.849) and balanced accuracy (0.658). Both minute-level pipelines had higher AUROC than nightly summaries, but differences from three-period handcrafted features were uncertain. Cross-model attribution prioritized activity, heart rate, and respiratory rate during the core overnight period. Calibration remained limited. The preceding night's signals supported modest next-day risk discrimination, with minute-level temporal modeling outperforming nightly summaries. Prospective calibration and external validation are needed before use in individual care decisions. This patient-grouped benchmark identifies contactless overnight sensing as a promising biomedical engineering direction for agitation-risk research in hospitalized dementia cohorts. 

---
# The Shape of Power: A Multilingual Framework for Social Power Reasoning in Dialogues 

**Authors**: Farah Atif, Sougata Saha, Monojit Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2608.28144)  

**Abstract**: Social power plays a fundamental role in shaping human interaction, yet computational studies of power remain limited to narrow linguistic and cultural settings. Existing datasets further lack the demographic and relational depth needed for robust cross-cultural analysis. To address this gap, we introduce a theoretically grounded framework for studying social power in naturalistic multilingual dialogue through movie screenplays. The framework integrates a schema informed by social science theory, a native speaker annotation pipeline refined through pilot studies, and a custom interface for scalable cross-lingual analysis. Using this framework, we constructed an initial corpus containing 15,836 annotated instances from 100 scenes in French and Egyptian Arabic movies. Our analysis reveals strong agreement on observable demographic and contextual attributes, while socially interpretive aspects, such as power asymmetry and intention alignment, remain more contested, highlighting the complexity of social power across cultures. We evaluated 6 Large Language Models (LLMs) and Multimodal LLMs on cross-cultural social power reasoning, finding persistent gaps between human and model agreement in relational and theory-of-mind reasoning. Our work introduces the first extensible multilingual framework for studying social power in dialogues and provides an initial evaluation setting for studying cross-cultural social reasoning. 

---
# Speculative Probing: LLM Monitoring at Speculative-Decoding Cost 

**Authors**: Collin Zhang, Tingwei Zhang, Vitaly Shmatikov  

**Link**: [PDF](https://arxiv.org/pdf/2608.28099)  

**Abstract**: Real-time classification during language model inference is valuable for safety filtering, behavioral analysis, and model monitoring, but current approaches force a trade-off between accuracy and efficiency. Hidden-state probes are fast but limited: they are either not context-aware: operating on a single vector and cannot model interactions across positions; or they are very costly: having dedicated classifier models (Llama Guard, Qwen Guard, LLM-as-judge) or performing computation on hidden states for all tokens and then pooling the results (MultiMax). This shows an intrinsic trade-off between efficiency and accuracy.
However, we find that the speculative-decoding module in recent LLMs can be repurposed for efficient high-quality classification. By appending a trained soft prompt at the end of the target sequence, we can repurpose the speculative-decoding module into a sequence classifier. At inference time in a speculative-decoding pipeline, the KV cache is already in GPU memory, so classification adds negligible overhead. We evaluate on four classification tasks across four models (Qwen3.5-4B, 9B, 27B, MiniCPM4.1-8B). Our small probes consistently outperform zero-shot GPT-5.4-mini and, on multilingual prompt safety, match or beat specialized 8B safety classifiers (Qwen3Guard-Gen-8B, Llama-Guard-3-8B) without running a full LLM. 

---
# SEPO: Evidence-Grounded Prompt Optimization via Structural Editing 

**Authors**: Xiaoyu Ma, Haoyue Liu, Yiwen Li, Jionghao Zhu, Zhichao Wang, Ye Chen, Xiaoying Tang  

**Link**: [PDF](https://arxiv.org/pdf/2608.28067)  

**Abstract**: Existing API-only prompt optimisers are often described as interpretable, but in practice, this usually means only post-hoc inspectability: each iteration still rewrites the prompt as one opaque string, leaving a trace of full-prompt diffs rather than localisable, machine-readable edits. This paper introduces SEPO (Structural, Evidence-grounded Prompt Optimization), a multi-trajectory prompt optimiser centred on edit-effect lineage feedback. Rather than treating each iteration as an isolated whole-prompt rewrite, SEPO locally edits stable, typed units in a two-layer prompt schema, links the target and realised structural operations of each edit to the examples it newly fixes or breaks, and carries this edit-effect record forward to guide later architect calls on the same search branch. This makes prompt optimisation addressable, attributable, and actionable. Across a 14-task held-out suite, SEPO improves over the strongest baseline, GEPA, by 3.1 pp on Llama-3.1-8B-Instruct and 2.2 pp on Qwen3-8B, reaching 61.9% and 73.3% macro accuracy. SEPO also lies on both the optimisation-time and test-time Pareto frontiers, spending 2.9M optimisation tokens versus 4.1M for GEPA and producing prompts over 5x shorter. 

---
# Learning to Allocate Incentives for Incentivized Advertising via Offline Model-Based Reinforcement Learning 

**Authors**: Zilin Zhao, Han Yang, Tianpei Yang, Fangsheng Huang, Yanfei Cui, Kan Peng, Yi Li, Yiming Zong, Hao Zhang, Yinsong Xue  

**Link**: [PDF](https://arxiv.org/pdf/2608.28065)  

**Abstract**: Complete your ad view and grab a 5-cent bonus! In incentivized advertising, a platform promises users a bonus before observing downstream ad revenue, encouraging them to click and complete ads. It must balance the incentive promised in advance against the revenue realized afterward: insufficient incentives forfeit monetization opportunities, whereas excessive incentives reduce net profit. Because current incentives may also shape user expectations and future engagement, incentive allocation is a sequential decision problem with delayed revenue, cost sensitivity, and carryover effects.
Existing work has not studied decision-making algorithms for this setting. Auto-bidding assumes available ad opportunities, while targeted promotion optimizes incentives outside the ad monetization pipeline. We formulate the problem as an MDP and develop an offline model-based RL framework for cost-controllable sequential incentive allocation. It learns a world model of user feedback and ad revenue, then performs conservative policy optimization. An independent counterfactual scorer evaluates each learned policy on held-out logs, enabling pre-launch selection without costly online exposure. Experiments on large-scale industrial data and online A/B tests show that the scorer provides a stable offline signal. The deployment path from causal inference to offline RL and then Offline-MBRL further validates the framework: MB-IQL improves per-user net profit by 7.96\% over TD3+BC, whereas reverting to plain IQL reduces it by 6.56\% (both \(p<0.0001\)). 

---
# WeAgent-MMSearch: Native Text-Vision Interaction for Multimodal Search Agents 

**Authors**: Zongkai Liu, Hui Zhang, Liqiang Niu, Zhen Cao, Han Li, Juntao Liu, Wenchao Chen, Chengduo Zhao, Chao Yu, Fandong Meng  

**Link**: [PDF](https://arxiv.org/pdf/2608.28062)  

**Abstract**: Multimodal search agents extend parametric knowledge with newly emerging and long-tail evidence from the open web. Yet many existing agentic search environments often expose retrieved evidence only as text and omit tool-returned images from subsequent context, reducing visually grounded trajectories to text-only reasoning. Long-horizon interaction also compounds tool-call, response-length, timeout, and budget failures, which can discard salvageable trajectories, waste rollout computation, and disturb policy updates. To address these issues, we introduce WeAgent-Harness, a multimodal agentic harness that supports native text-vision interaction and runtime recovery. Retrieved images receive persistent disk references, allowing the model to inspect, process, and cite them throughout the trajectory. Based on this harness, we develop WeAgent-MMSearch, an integrated system spanning data construction, agentic post-training, and multimodal rollout. For data construction, a strong MLLM uses WeAgent-Harness to discover, synthesize, and verify MMSearch-style tasks and collect expert trajectories. During post-training, our Failure-Aware GSPO (FA-GSPO) recovers salvageable abnormal rollouts and filters invalid ones to improve bounded multimodal planning and this http URL also introduce VisTarget-Bench, a 150-task human-verified benchmark that pairs each question with a held-out target image, distinguishing image-retrieval failures from visual-perception failures. Evaluation on VisTarget-Bench and seven public benchmarks shows that agentic post-training improves the average score by 19.22 points, enabling our model to outperform similarly sized open-source models and rival models with roughly ten times its parameter count. 

---
# String: An Agentic OS Where Every App Is a Markdown File 

**Authors**: Jookyung Song, Nojun Kwak, Simyung Chang  

**Link**: [PDF](https://arxiv.org/pdf/2608.28027)  

**Abstract**: LLM agents have become a new class of software user, but every surface they work through was designed for someone else. Pages are built for human eyes, which can skim and ignore; tool schemas for programs, which pay nothing to carry definitions they never call. An agent has neither luxury: it re-reads, and pays again for, everything it is shown on every turn. We present String, an open-source runtime that gives this user an interface of its own and treats the job as an operating-systems problem. Tool knowledge moves out of the agent's context and into a common layer that renders it back one view at a time as Markdown. A single SFMD (String-Flavored Markdown) document declares an application's views, typed actions, navigation, and credentials, and the runtime handles discovery, validation, execution, state, and secrets behind two core verbs: /open to see and /act to do. Web and app turn out to be two renderings of one architecture: an SFMD site serves styled HTML to browsers and the raw document to agents, so one grammar reaches apps, files, shells, and the web, even legacy HTML, with no per-site integration. Views stay partial by design, and the staging is causal: disclosing one tier of detail a single turn too early costs up to 23 accuracy points, while proper staging drops wrong-action selection from 28% to 2%. Privilege follows provenance: a remote page may call HTTP but never the shell, and caller-supplied text never expands a stored secret. On an 87-task benchmark that pairs each task with curated skills, operationalizing those procedures as on-demand String apps yields comparable aggregate success across six models from frontier to small (+1.3pp) while using 33.5% fewer tokens among completed episodes, and the resident interface stays a constant 53 tokens at any catalog size. We report the design, the evaluation, and what three months of production use taught us. 

---
# Coverage, Not Credit: Failure-Credit Routing of Zeroth-Order Perturbation Budgets Does Not Improve On-Pool Sample Efficiency for LLM Agents 

**Authors**: Yuxu Ge  

**Link**: [PDF](https://arxiv.org/pdf/2608.28011)  

**Abstract**: Trajectory-level credit assignment can localize which module of a tool-using LLM agent causes failures using only verifiable signals. We ask whether such failure credit should route a fixed zeroth-order/evolution-strategies (ZO/ES) perturbation budget. Across a synthetic environment and frozen Qwen2.5-1.5B/3B and SmolLM2-1.7B agents, three task families, six allocation schemes, a credit-noise sweep, paired seeds, and exact sign-flip tests, we find no statistically detectable improvement over uniform allocation in any on-pool comparison (no gain of at least 2 percentage points). The joint soft-plus-sigma scheme is equivalent to uniform within a +/- 0.02 AUC margin on 1.5B and 3B; concentrating the full budget on the credit argmax is marginally equivalent on 1.5B, where that module is the verified bottleneck, and significantly worse on 3B. Inverse-propensity debiasing does not rescue routing, and misrouting costs up to -0.074 AUC in-house and -0.118 end-to-end on the BFCL-derived family. Across six fixed-step schedules, loss is linear in bottleneck starvation rate (R^2 = 0.94, descriptive), and a preregistered credit-free coverage floor removes detected harm. Matched-budget burst and step-compensating catch-up schedules are consistent with harm arising from insufficient cumulative parameter movement rather than update frequency. Our primary estimand is optimization efficiency on a fixed task pool. On unseen BFCL functions, the study's one exception is that soft routing exceeds uniform on held-out endpoints (+0.047, p = 0.031, n = 6). A plausible but untested reading is that routing-favored caller improvements transfer while uniform's on-pool gains reflect a synthesizer behavior specific to our harness. We report this exception explicitly and document three failure modes that can silently invalidate ZO/ES experiments on frozen LLMs. 

---
# PhenoIntel: A Lifecycle-Aligned Multi-Agent Web Application for Verified, Accessible Plant Phenotype Analysis 

**Authors**: Narendren S V, Soumyashree Kar  

**Link**: [PDF](https://arxiv.org/pdf/2608.27999)  

**Abstract**: Existing conversational plant-phenotyping platforms are difficult for plant scientists to use and lack the reliability scientific research demands: failed analyses are reported as valid measurements rather than flagged as missing, statistical tests run without checking assumptions, predictions carry no uncertainty estimate, and specialised hardware limits accessibility. We present PhenoIntel, a lifecycle-aligned multi-agent web platform that turns the full machine-learning workflow into a reliable, user-friendly phenotyping system. Nine specialised agents divide the analysis into stages, from image collection through model selection, inference, and reporting, rather than handing the whole task to one AI manager. Independent checks separate these stages, and every agent reads from and writes to one shared, fixed-structure record, so an inconsistent output from one stage is caught before it reaches the next. Uncertainty is matched to each model family, conformal prediction, detection-confidence spread, or Monte Carlo Dropout, rather than applied uniformly, and quality thresholds adapt to crop and task instead of one global cutoff. When no suitable model exists, PhenoIntel can propose, validate, and integrate a new one on its own. The model repository spans ten trained models across five crops and four imaging modalities. Classification models reach Macro F1 of 0.78-0.996; object-detection models reach 0.96 mAP@50 with a 54% reduction in counting error over an unoptimised baseline; and a temporal model reaches held-out Macro F1 of 0.7050. PhenoIntel runs in a browser on standard hardware, requiring no GPU, and a 1,200-test automated suite confirms complete pipeline execution. Every result carries calibrated uncertainty, validated statistics, and FAIR-compliant provenance, a combination existing conversational phenotyping tools do not offer. 

---
# Automated Analysis Framework for Multilingual Climate-Health Literature Based on Multi-Agent Large Language Model 

**Authors**: Yuze Sun, Shihui Zhang, Jiancheng Pan, Yunjia Ye, Wentao Luo, Jiahao Li, Quan Zhang, Wenjia Cai, Xiaomeng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.27998)  

**Abstract**: The rapid proliferation of interdisciplinary and multilingual scientific literature has left traditional manual analysis and single-algorithm methods plagued by low efficiency, poor scalability, and insufficient domain adaptability. Targeting the literature analysis needs of the typical interdisciplinary climate-health field, this study proposes a multi-agent large language model automated analysis framework for multilingual scientific literature, which realizes full-process automation covering literature screening, structured information extraction, and standardized integration. With a central coordination module as the core, the framework deploys three dedicated agents for document evaluation, information extraction, and analytical review to mimic the literature analysis thinking of domain experts, and adopts a four-layer hallucination control strategy together with a manual verification procedure to ensure the accuracy and reliability of analytical outcomes. Validated on a bilingual Chinese-English corpus of 32,642 climate-health papers covering China from 1993 to 2023, the framework achieves an F1 score of 0.92 in core information extraction, and completes the extraction and standardization of 2,012 city-literature association pairs, offering effective technical support for large-scale evidence mining in the climate-health research domain. 

---
# Should I Use This Synthetic Dataset for Training? How to Test with Minimal Real Data 

**Authors**: Zhenyu Tao, Wei Xu, Xiaohu You, Petar Popovski, Osvaldo Simeone  

**Link**: [PDF](https://arxiv.org/pdf/2608.27996)  

**Abstract**: Digital twins (DTs) and learned world models are increasingly used to generate synthetic data that augment the scarce real datasets available for training artificial intelligence (AI) models in engineering systems. Owing to the inevitable simulation-to-reality (sim-to-real) gap, however, augmentation may fail to improve the performance of the trained model on the real data distribution. This paper addresses the resulting decision problem: Given a real dataset, a candidate synthetic dataset, and a fixed learning algorithm, decide whether training on the augmented dataset improves the true, population-level performance, while consuming as few real test data points as possible. Two formulations are considered: a direct test on the mean loss difference between the two trained models, and a symmetry-based test on the paired loss difference, which trades a stronger null assumption for faster evidence accumulation. For the latter, we introduce the {adaptive e-process sign-flip test} (aeSFT), a doubly adaptive procedure that adapts both the number of Monte Carlo sign-flip rounds, and hence the computational cost, and the amount of real test data consumed. aeSFT yields anytime-valid Type-I error control, with no need to pre-specify the test-set size. Experiments on a synthetic-data classification task, a DT-aided wireless packet-scheduling task, and a radio-map prediction task show that aeSFT identifies useful synthetic data using substantially fewer real test samples than mean-based sequential testing, matches the power of fixed-sample sign-flip testing and the paired $t$-test, while keeping the false-positive rate below the target level. 

---
# GOD: Govern, Observe, and Direct - A Real-Time Control Room for Agent Societies 

**Authors**: Yige Luo, Ran Guan  

**Link**: [PDF](https://arxiv.org/pdf/2608.27992)  

**Abstract**: Generative-agent systems are easier to start than to inspect. A run can contain many agents, locations, messages, commands, and model calls, yet the operator often gets either a finished replay or raw logs. That makes it hard to ask why an agent moved, test a small intervention, or package a run for another researcher. GOD is a local-first control room for agent societies. From the same browser workflow, an operator can issue targeted questions or interventions and inspect the resulting replay state. The system combines a setup wizard, Agent Studio, Map Studio, a spatial replay interface, Ask and Intervene commands, and portable experiment, map, and agent packs. Its technical contribution is the command and artifact loop: live controls and replay evidence share the same operator command model, while package contracts separate scenario, map, and profile data from local runtime state. The public release includes hosted Smallville-style and PKU replays, the open-source repository, and downloadable packs. We evaluate this path on 15 completed run slots. Across the 14 intervention runs, 78 of 84 target-agent checks recorded the commanded destination, and 169 of 182 state answers matched a saved location or action string. 

---
# When Evidence Shapes Collaboration: Knowledge-Conditioned Topology Generation for Multi-Agent Systems 

**Authors**: Yangxiao Jiang, Jiarun Fan, Mingcong Xu, Yanxi Guo, Jiwen Feng, Shanqing Xu, Mengchen Qian, Wei Chen, Xiaojin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.27984)  

**Abstract**: Multi-Agent Systems (MAS) have recently moved from static workflows toward dynamically generated collaboration topologies. However, existing topology generation methods rely primarily on the parametric knowledge of large language models, with external search or retrieval used only as a reactive tool rather than an explicit determinant of collaboration structure. This leads to structure-knowledge misalignment, where systems exhibit redundant interactions or insufficient verification in knowledge-intensive tasks. We propose K-GAT (Knowledge-Guided Agent Topology Generator), a neuro-symbolic framework that formulates collaboration topology design as a knowledge-conditioned structure learning problem, integrating external evidence directly into autoregressive graph generation. Extensive experiments on knowledge-intensive benchmarks demonstrate K-GAT's efficiency and effectiveness: notably on the expert-level GPQA dataset, K-GAT outperforms the LLM-Debate baseline by a substantial margin of +15.7% in accuracy, while consuming less than half the computational tokens. 

---
# Learning from Hard Prompts: Difficulty-aware Advantage Amplification in Dynamic Sampling 

**Authors**: Siyuan Gan, Yuhan Li, Xiran Wang, Linjian Meng, Boyan Wang, Zhen Zhao, Jing Huo, Lei Bai, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2608.27982)  

**Abstract**: Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO) is a prominent variant of Group Relative Policy Optimization (GRPO). DAPO introduces several improvements over GRPO. Among these, Dynamic Sampling contributes the most to DAPO's accuracy gains relative to GRPO. To improve accuracy, Dynamic Sampling enhances training stability by eliminating zero policy gradients from zero advantages. Specifically, it avoids such zero gradients by filtering out prompts where sampled responses are either entirely correct or incorrect. However, our theoretical analysis shows that Dynamic Sampling decrease training efficiency as it cannot effectively utilize hard-to-sample correct responses on hard prompts. Formally, it asymmetrically amplifies the advantages of distinct responses to the same prompts. On hard prompts, incorrect responses undergo greater amplification than correct ones. This leads the model to avoid generating the observed incorrect responses rather than capitalizing on the hard-to-sample correct ones on hard prompts, resulting in low training efficiency. To improve training efficiency, we propose Direct Advantage Amplification (DAA), which amplifies the advantages of hard-to-sample correct responses on hard prompts, as obtained by Dynamic Sampling. This ensures that, when Dynamic Sampling is used, these hard-to-sample responses can be effectively capitalized on, implying higher training efficiency. By integrating DAA into DAPO, we obtain Difficulty-aware Advantage Amplification Policy Optimization (DA3PO), which is implemented with fewer than 30 lines of code from DAPO. Experiments show that DA3PO significantly outperforms GRPO and other classical GRPO variants. 

---
# openJiuwen: Beyond Static Harnesses for Long-Horizon Coding Agents 

**Authors**: openJiuwen Team, Tao Yu, Xinyu Zhang, Qianqian Chen, Xiaoneng Xiang, Chia Kwangyang, Xingchen Huang, Ran Chen, Yangkai Ding, Zheng Wang, Yeo Boon Hong, Bingzheng Gan, Enrui Hu, Shuo Cheng, Deyang Li, Ruifeng Shi, Hongbo Wang, Qi Ye, Xuefeng Jin, Zhangchun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2608.27969)  

**Abstract**: Long-horizon coding agents operate over evolving repository states while increasingly relying on heterogeneous capabilities, delegated agents, and multi-agent coordination. These trends pose two complementary challenges for the agent harness. First, developers need to compose capabilities, reconfigure execution logic, and scale increasingly complex agent systems without repeatedly rebuilding orchestration. Second, complex coding tasks continuously produce new evidence---such as semantic diagnostics, execution outcomes, task progress, and changing context relevance---that should dynamically influence subsequent runtime decisions. We characterize these challenges as Structural Composability and Runtime Adaptivity. We present openJiuwen, an open-source harness designed for both developer composability and adaptive task execution. openJiuwen provides a shared execution substrate and Rail-based capability composition across single agents, delegated sub-agents, and Swarm Flow, enabling developers to construct sophisticated agent harnesses under common execution semantics. It further adapts framework-controlled runtime decisions around a fixed model policy, allowing evolving evidence to dynamically affect context, feedback, and task control toward successful completion. We systematically evaluate openJiuwen on SWE-bench Verified and Terminal-Bench 2.1, where it achieves 82.6% and 87.19%, respectively, exceeding the strongest selected official-leaderboard point estimates by 3.4 and 3.39 percentage points. These results show that openJiuwen achieves strong performance on complex coding tasks while providing a composable and adaptive harness design. 

---
# AERA: Adaptive Evidence Residual Allocation for Efficient Test-Time Reasoning 

**Authors**: Ziming Wang, Ivor Tsang, Hangwei Qian  

**Link**: [PDF](https://arxiv.org/pdf/2608.27964)  

**Abstract**: Test-time scaling improves language-model reasoning by generating additional candidate solutions, but allocating the same inference budget to every problem is computationally wasteful. Existing adaptive stopping methods commonly rely on confidence, agreement, or answer stability, implicitly assuming that stronger current evidence indicates that further computation is unnecessary. We show that this assumption can fail: checkpoint-level correctness evolves non-monotonically, and observable evidence may strengthen before an answer collapses or weaken before it recovers. Motivated by this mismatch, we introduce Adaptive Evidence Residual Allocation (AERA), a sequential controller that learns whether additional computation is likely to recover a better answer from checkpoint-observable evidence. AERA characterizes cumulative response prefixes using answer-distribution, temporal, re-solving, semantic, and compute features, and repeatedly decides whether to stop or allocate the next response block. Future checkpoint correctness is used only to construct offline supervision and is never available to the controller at inference time. Across GSM8K and GPQA Diamond, AERA identifies question-specific residual opportunities while substantially reducing inference computation. In a frozen-threshold incremental-generation evaluation on 300 untouched GSM8K questions, AERA achieves 92.61% accuracy versus 93.01% with 128 responses while reducing completion tokens by 95.99%. These results suggest that adaptive reasoning should estimate the future value of computation rather than equating present confidence with correctness. 

---
# SABER: Stability-Aware Early Exit for LLM Reasoning via Adversarial Branch Probing 

**Authors**: Wanli Cheng, Haiya Xiang, Juntao Li, Hongling Wang, Wenliang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.27963)  

**Abstract**: Large Reasoning Models (LRMs) achieve strong reasoning capabilities, yet long-chain reasoning becomes inefficient once the intermediate answer stabilizes across reasoning steps: additional reasoning yields little marginal benefit while incurring substantial inference cost. Existing early-exit methods based on confidence or entropy poorly capture reasoning stability, while consistency-based approaches rely on multi-step trajectory agreement, requiring sequential evaluations that delay exit. To better balance efficiency and reliability, we propose SABER, a training-free framework for stability-aware early exit via adversarial branch probing. SABER constructs simple yet effective semantic perturbations around intermediate reasoning states to form adversarial branches, and applies lightweight probing to estimate their likely final outcomes without full trajectory rollouts. When the probed outcomes remain consistent across branches, SABER exits early; otherwise, it continues reasoning. Experiments across multiple reasoning benchmarks and model architectures show that SABER reduces reasoning token consumption by 30.2\%--39.8\% on average while maintaining competitive accuracy with full-length reasoning. 

---
# When Teacher Guidance Misleads: Reward-Aligned On-Policy Distillation 

**Authors**: Siyuan Gan, Yuhan Li, Xiran Wang, Linjian Meng, Boyan Wang, Zhen Zhao, Jing Huo, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2608.27960)  

**Abstract**: On-policy distillation (OPD) has recently emerged as a popular post-training paradigm for large language models (LLMs), providing an efficient way to transfer the knowledge and capabilities of teacher models into student models. However, teacher guidance on student-generated prefixes is not always reliable. Training should optimize the model to generate responses that are more likely to be correct, or equivalently, to get higher outcome rewards. But during OPD, the teacher model may provide guidance that discourages the student from moving toward correct trajectories or moves the student toward incorrect ones, which is misaligned with outcome reward. Such misaligned guidance is unreliable, as it would mislead the optimization process and ultimately degrade model performance. To mitigate misaligned teacher guidance, we propose Reward-Aligned On-Policy Distillation (RA-OPD). The key insight is to keep only trajectories whose induced updates move the student toward correct trajectories or discourage the student from moving toward incorrect ones. Specifically, for each sampled trajectory, RA-OPD checks whether its trajectory-level distillation return is consistent with its outcome reward and then filters out the misaligned trajectories. RA-OPD selects more reliable trajectories to improve student model performance without requiring additional computational cost. We evaluate RA-OPD on math and code benchmarks using models from the Qwen3 family and the DeepSeek-R1 family. Across seven math benchmarks and three code benchmarks, RA-OPD significantly outperforms standard OPD and other tested OPD variants. 

---
# The Illusion of $\textit{What If}$: Evaluating the Breakdown of Counterfactual Reasoning in LLMs 

**Authors**: Yucheng Wang, Yuetian Du, Zhengyi Liu, Rongyu Zhang, Bing Zhao, Boyu Yang, Ming Kong, Lin Qu, Hu Wei, Jie Liu, Qiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.27953)  

**Abstract**: Counterfactual reasoning requires models to reason beyond the observed world and explain how altered conditions propagate through downstream consequences. Existing benchmarks largely target bounded settings with fixed variables or single gold outcomes, overlooking open-domain scenarios requiring causal-process evaluation. To this end, we present $\textbf{WhatIfBench}$, a diagnostic benchmark for open-domain, open-form, long-horizon counterfactual causal reasoning, containing 220 what-if questions across STEM, HSS, and Hybrid scenarios. To evaluate free-form responses, we further propose $\textbf{PRISM}$, which first converts each natural-language explanation into a Response-Derived Semantic Causal Graph of events, states, and mechanisms. On top of this graph, PRISM then jointly applies a Process Metric assessing graph-level causal validity and a Rubric Metric assessing answer-level explanatory adequacy. Evaluating six frontier LLMs with this framework, we find that WhatIfBench remains far from saturated: even the strongest model reaches only a 64.62% final score. Further analysis reveals persistent causal gaps, premise drift, and topology fragmentation, suggesting that fluent counterfactual narratives often mask fragile causal processes. The benchmark, code, and evaluation scripts are available at $\href{this https URL}{WhatIfBench}$. 

---
# Cross-Session Decomposition Attacks: Scaling Risk and Intent-Aligned Retrieval Defense 

**Authors**: Disen Liao, Yihan Wang, Freda Shi, Yaoliang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2608.27945)  

**Abstract**: Scaling laws are usually read as a capability story: lower language-modeling loss yields more useful models. We study a safety consequence of this mechanism in \emph{cross-session decomposition attacks}, where benign-looking subqueries are asked across independent interactions and later recomposed toward a forbidden objective. We formalize this setting as \emph{compositional safety risk} and prove a conditional risk-transfer bound: when the reference environment already contains dispersed evidence for a risky reconstruction, the gap between deployed composed risk and reference composed risk is controlled by the model's excess loss on allowed subqueries. Synthetic withholding experiments show that wider transformers assign lower loss to held-out instructions that never appear verbatim in training but are recoverable from injected supporting facts. A 600-intent pretrained-LLM evaluation shows that larger Qwen3 and Gemma3 family members can yield greater harmful-capability uplift under a fixed decomposition-composition pipeline. As a defense, IntentAlign-MiniLM, our 22M-parameter intent-aligned retriever, outperforms much larger embedding models on held-out intent retrieval and yields the best learned-retriever harmful recall across tested guardrails. Code is available in \href{this https URL}{our GitHub repository}. 

---
# CASTANET: Causality-Aware Spatio-Temporal Adversarial Network Using Traffic Incident Effects 

**Authors**: Toshiya Kitahara, Ryu Shirakami, Koh Takeuchi, Hisashi Kashima  

**Link**: [PDF](https://arxiv.org/pdf/2608.27942)  

**Abstract**: Predicting non-periodic traffic congestion caused by sudden incidents (e.g., accidents and road damage) is crucial for advanced intelligent transportation systems. However, incident-driven congestion is difficult to forecast because incidents are extremely sparse, occur at specific times and locations, and have heterogeneous impacts depending on the traffic context. While recent deep learning approaches have significantly improved periodic traffic forecasting, their performance on non-periodic congestion remains limited, partly because incident records are not explicitly incorporated and their occurrence is strongly biased in space and time. To address these challenges, we propose CASTANET, which integrates spatio-temporal graph neural networks and causal treatment effect estimation to utilize incident records while mitigating selection bias. Experiments on real-world traffic data and accident records from Tokyo, which we treat as incidents, show that CASTANET reduces RMSE by 4.0% overall compared to the best baseline and by 10.1% on incident-conditioned evaluation, with gains reaching 14.55% under severe congestion. 

---
# A Deep Learning-Based Stacking Ensemble Framework for Turbofan Engine Remaining Useful Life Prediction 

**Authors**: Limon Bin Hossain, Md. Salehin Seyam, Md Rashedul Islam, Abdur Rahman, Md Sharifuzzaman  

**Link**: [PDF](https://arxiv.org/pdf/2608.27940)  

**Abstract**: This study proposes a two-level stacking ensemble framework for Remaining Useful Life (RUL) prediction of turbofan engines, evaluated on the NASA C-MAPSS benchmark using the FD001 and FD003 subsets. The framework integrates four heterogeneous deep learning base learners: Long Short-Term Memory (LSTM), Convolutional Neural Network (CNN), CNN-LSTM, and CNN-GRU, whose out-of-fold predictions are combined by an XGBoost meta-learner to capture complex degradation patterns while mitigating individual model biases. Comprehensive experiments demonstrate that the stacking ensemble achieves superior predictive performance, with Root Mean Square Error (RMSE) of 9.989 and 8.613, Mean Absolute Error (MAE) of 7.081 and 5.195, and R-squared values of 0.899 and 0.906 for FD001 and FD003, respectively. Compared to the best-reported baseline (TCAT: RMSE 11.12 and 11.02), the proposed method achieves RMSE reductions of 10.2 percent and 21.8 percent for FD001 and FD003, respectively. Feature correlation analysis, residual diagnostics, and training convergence curves validate the model's robustness. These findings underscore the efficacy of stacking ensemble methods for prognostics and health management in safety-critical aerospace applications. 

---
# From Documents to Reasoning: A Validated Synthetic Data Pipeline and Semantic-Aware Fine-Tuning for Financial Numerical Reasoning 

**Authors**: Lokendra Birla, Milind Savagaonkar, Visnu Srinivasan, Sowmya Rasipuram, Shubhashis Sengupta  

**Link**: [PDF](https://arxiv.org/pdf/2608.27919)  

**Abstract**: Financial question answering (QA) has emerged as a key benchmark for evaluating the performance of Large Language Models (LLMs) on domain-specific tasks involving complex data formats such as tables, charts, and rich textual narratives. While recent advancements have enabled models to reason across modalities and perform multi-step arithmetic operations, limitations remain in performance consistency, and evaluation reliability. In particular, standard evaluation metrics like Exact Match (EM) often fail to account for minor variations such as differences in units or formats, misleading performance assessments.
In this work, we propose a comprehensive pipeline for improving financial QA systems through high-quality synthetic data generation and fine-tuning of smaller language models (SLMs) using Quantized Low-Rank Adaptation (QLoRA). Our pipeline includes aggressive data validation for synthetic question answer generation to ensure the relevance and correctness of synthetic question-answer pairs. We introduce a novel evaluation metric that matches answers computed from arithmetic expressions rather than ground-truth answers; providing a more accurate reflection of model reasoning capability. Furthermore, we propose a modified loss function that aligns predicted and reference expressions using semantic similarity, our novel evaluation metric and standard cross-entropy, resulting in improved performance. Experimental results on benchmark datasets, ConvFinQA demonstrate significant gains in QA accuracy after fine-tuning using synthetic dataset and proposed loss function. 

---
# AI Alignment through a Game-theoretic Lens: A Survey 

**Authors**: Yanan Cai, Zhongrui Zhao, Zhigang Lu, Ickjai Lee, Wei Emma Zhang, Minhui Xue, Yihong Zhang, Shuchao Pang, Wei Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2608.27910)  

**Abstract**: As large language models and increasingly capable AI agents are deployed in high-risk settings, aligning them with complex human values has become a central challenge. Existing alignment methods, while effective in improving helpfulness, harmlessness, and controllability, often struggle to capture real-world preferences that are context-dependent, non-transitive, and shaped by dynamic multi-party interactions. This survey reviews AI alignment through a game-theoretic lens. Specifically, it organizes recent progress around key game-theoretic elements and synthesizes the literature along three challenges: preference diversity, alignment priority, and temporal dynamics. This perspective clarifies where current alignment methods genuinely benefit from game-theoretic analysis, where the framework is looser, and what challenges remain in building robust, adaptive, and verifiable AI systems. 

---
# Rubric-to-Code Credit Assignment for Reinforcement Learning 

**Authors**: Rui Jin, Jikai Chen, Yihan Chen, Hao Zhou, Demin Zhu, Kaichen Yang, Dong Wang, Chenyi Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2608.27906)  

**Abstract**: Interactive web application generation requires models to produce usable HTML, CSS, and JavaScript applications from natural language requests. Unlike conventional code generation, application quality depends on multiple user-facing functional requirements, each often tied to localized code regions such as event handlers, state updates, DOM fragments, or CSS selectors. Standard GRPO collapses these structured outcomes into a single sequence-level reward and applies the resulting advantage uniformly to all tokens, weakening credit assignment. We propose \textbf{Rubric-to-Code Credit Assignment} (RCCA), a reinforcement learning framework that converts rubric-level functional feedback into localized optimization signals over generated code. RCCA builds training tasks around explicit functional rubrics, uses a hierarchical reward to separate format, source-code, runtime, and functional failures, and aligns evaluator-generated textual attributions with responsible code spans and generated tokens. The resulting model, \textbf{Ling-RCCA-Flash}, scores 41.25 on MiniAppBench, improving Ling-3.0-Flash by 32.20 points and slightly surpassing Claude Opus 4.5. It also reaches 76.19 on ArtifactsBench, improving the SFT model by 4.48 points and establishing a new top score under the official ArtifactsBench leaderboard setting by surpassing the GPT-5 score by 3.64 points, suggesting transferable implementation-level gains. 

---
# Resource Constraints and Performance in Agentic AI Systems 

**Authors**: Amaz Salman, Malka Halgamuge, Teo Susnjak  

**Link**: [PDF](https://arxiv.org/pdf/2608.27886)  

**Abstract**: Progress toward more autonomous AI increasingly depends on agentic systems that combine a language model with tools, memory, state management, and multi-step execution. These mechanisms shape both task capability and operational burden. We compare OpenClaw and NanoBot as complete agentic systems using a paired primary benchmark and a more detailed instrumented subset of paired prompts. In the primary benchmark, the rate of full task completion was 31% for OpenClaw and 25% for NanoBot, a six-percentage-point difference with a 95% task-bootstrap interval from -3 to 15 percentage points, providing no statistically established full-completion advantage for either system. In the instrumented layer, both systems achieved 26% full completion, while NanoBot reached at least partial completion on 43% of prompts compared with 26% for OpenClaw. OpenClaw took longer on 83% of prompts and had a higher recorded peak-memory value on every prompt, with geometric mean ratios of 2.98 for wall time and 19.44 for peak memory. Among the ten detailed-layer prompts on which at least one system achieved partial or full completion, NanoBot weakly dominated on eight; across all 23 prompts, however, ten of its eighteen dominance cases were cheaper joint failures. Outcome labels differ across the two evidence layers, showing why agent-system evaluation should connect capability and resource measurements to attempt-level execution and scoring provenance. These findings show that progress toward more autonomous AI should be evaluated through verified task completion, observed resource use and records linking each result to the execution that produced it. 

---
# HyQuant: Hybrid-Precision Quantization for LLM Attention 

**Authors**: Jiatong Ding, Bingxin Xing, Yu Zhang, Dian Ding, Xiaodong Yi, Xianbin Ouyang, Feihu Zhou, Kun Zhang, Zhenyu Guo, Hao Pan, Guangtao Xue, Yiming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.27875)  

**Abstract**: Quantization has been widely adopted in LLM training and inference to reduce cost and improve efficiency. However, low-bit quantization of the \emph{attention} module often introduces large errors at very low bit-widths, causing performance degradation. Existing methods mainly rely on smoothing techniques to handle outliers, while we propose a hybrid quantization design to better balance accuracy and efficiency. Specifically, we propose \textbf{HyQuant}, an efficient hybrid quantization framework for LLM attention. HyQuant quantizes most attention states into low-bit formats while retaining a small set of vertical-line tokens and local-window states in high precision. These accuracy-critical regions are selected using lightweight vertical-line-aware attention-pattern signals, reducing quantization error with limited overhead. In the Prefill stage, HyQuant uses a hybrid-precision quantized attention operator that preserves vertical-line tokens and a local sliding window in full precision while quantizing the remaining context. In the Decode stage, HyQuant applies the same principle to KV-cache compression and fuses KV dequantization with attention computation to improve memory and hardware efficiency. Across diverse tasks, models, and datasets, HyQuant maintains nearly lossless accuracy with an extremely simple design, demonstrating the efficiency and practical feasibility of hybrid quantization for LLM attention. Code is available at: this https URL . 

---
# See, Hypothesize, Validate: Multimodal Agentic Framework for Discovering Governing PDEs 

**Authors**: Sarang Manoj Pekhale, Amartya Roy, Rajat Sarkar, Souvik Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2608.27869)  

**Abstract**: Discovering governing partial differential equations (PDEs) from observational data remains a core challenge across the sciences. Existing sparse-regression, symbolic-regression, and LLM-based approaches can be constrained by predefined libraries, noise sensitivity, hallucination, or limited iterative refinement. We introduce \textbf{MAGE} (\textbf{M}ultimodal \textbf{A}gentic \textbf{G}overning \textbf{E}quation Discovery), an agentic framework that organizes PDE discovery as a \textit{confidence governed hypothesis validation loop} inspired by the scientific cycle of observation, hypothesis, and falsification. Four role-specialized agents collaborate: a \textit{Differential Observer} computing derivatives and diagnostic visualizations; a VLM-powered \textit{Phenomenology Extractor} distilling qualitative cues from multimodal diagnostics; an LLM-driven \textit{Governing Law Synthesizer} proposing candidates without a predefined library; and an \textit{Equation Arbiter} fitting coefficients and assigning confidence scores. Discovery iterates until the top candidate clears a user-specified threshold, providing a structured process with an explicit accept-reject protocol. On the evaluated canonical PDE suite, MAGE obtains \textbf{8/8} exact structural recovery and the lowest coefficient error among the compared methods on \textbf{7/8} systems, with improvements of up to \textbf{4 orders of magnitude} and a geometric-mean improvement of approximately \textbf{3 orders of magnitude}. The pipeline also recovers the expected operators in two complex geometries and, on one laboratory sensor record, selects a cubic restoring-force model with held-out $R^2=0.98538$. These results support further study of structured agentic reasoning for library-free governing-law discovery, while broader generalization remains to be evaluated. 

---
# CoRe-MoE: Compact Reusable MoE for Continual Multimodal Instruction Tuning 

**Authors**: Runze Liu, Naibin Gu, Mingxu Ai, Yuqing Li, Peng Fu, Zheng Lin, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.27867)  

**Abstract**: Continual multimodal instruction tuning requires multimodal large language models to acquire new task abilities sequentially while preserving previously learned knowledge. LoRA-MoE provides a promising solution by introducing expert-based capacity, but repeatedly learning and maintaining full LoRA experts leads to substantial parameter overhead. This raises a natural question: is full expert expansion necessary for every new task? To answer it, we analyze the SVD of task-specific LoRA updates and observe substantial overlap in their input- and output-side LoRA direction subspaces, with task-specific adaptation largely captured by lightweight coordinates over these subspaces. Motivated by this observation, we propose CoRe-MoE, a Compact Reusable MoE framework for parameter-efficient continual multimodal instruction tuning. CoRe-MoE extracts reusable input- and output-side direction bases from an initial expert bank, and for subsequent tasks trains only compact coordinate experts together with task-specific low-rank routers. Experiments on two representative MLLMs show that CoRe-MoE improves final average performance over the strongest competing baseline by up to 5.90 points, while using less than 1% of the trainable parameters required by sequential LoRA for later tasks. The code is publicly available at this https URL. 

---
# SpikeOPD: Stable On-Policy Distillation for Autoregressive Spiking Language Models 

**Authors**: Enqiao Lu, Xingrui Yu, Yiwei Fu, Zhenglin Wan, Pengfei Zhou, Wangbo Zhao, Muqing Jian, Xueyi Zhang, Yang You, Ivor Tsang  

**Link**: [PDF](https://arxiv.org/pdf/2608.27857)  

**Abstract**: Spiking neural networks (SNNs) offer a path to energy-efficient language modeling through sparse encoding and event-driven computation, but training capable spiking language models from scratch remains difficult. A practical alternative is ANN-to-SNN migration through knowledge distillation (KD), where a pretrained artificial neural network (ANN) teacher supervises an SNN student. Existing migration approaches distill on fixed corpus prefixes, whereas autoregressive inference conditions on self-generated prefixes, creating prefix-source mismatch. It manifests as output-policy mismatch with the ANN teacher and internal spiking-dynamics drift between self-generated and matched corpus prefixes. On-policy distillation (OPD) offers a natural way to mitigate both manifestations by continuing teacher supervision on self-generated prefixes. We evaluate a teacher-only full-KL variant, Vanilla OPD, via a controlled stress test and observe it may suffer from delayed rollout-feedback collapse. This result shows that on-policy coverage alone does not ensure stable adaptation. Motivated by these findings, we propose SpikeOPD, a stable on-policy distillation framework for autoregressive SNNs that learns from self-generated prefixes while maintaining rollout stability. It applies full-KL teacher correction to reduce output-policy mismatch, while matched-prefix policy anchoring constrains policy departure from the frozen reference SNN on the same prefixes. Layerwise spike regularization further limits firing-rate deviations during on-policy adaptation. Across three model scales, SpikeOPD improves average accuracy over the corresponding KD SNNs by 0.8, 1.7, and 2.9 points at 0.125B, 0.35B, and 1.3B, respectively, while preserving their sparse-compute profiles. 

---
# From Uncertainty to Clinical Risk: Severity-Aware Conformal Planning for Interactive Medical Diagnosis 

**Authors**: Yue Zhou, Haiyang Zhou, Jin Zhang, Kong Wang, Yongxin Ni, Youhua Li, Hanwen Du  

**Link**: [PDF](https://arxiv.org/pdf/2608.27847)  

**Abstract**: Interactive medical diagnosis dynamically acquires patient information through multiple rounds of questioning, supporting accurate, efficient, and safe clinical decisions under incomplete evidence. Existing methods commonly guide information acquisition with predictive uncertainty or label ambiguity, but overlook the asymmetric clinical risk of missing severe diseases and lack unified long-horizon planning over whether to continue asking questions or commit to a diagnosis. To address these limitations, we propose Severity-Aware Conformal Clinical Planning, which formulates interactive diagnosis as a risk-sensitive sequential decision problem. The framework maintains complementary diagnostic, safety, and masked-evidence beliefs; calibrates turn-specific diagnostic prediction sets and severity-weighted differential-diagnosis risk on held-out diagnostic trajectories; and introduces the calibrated clinical risk into Monte Carlo Tree Search to jointly evaluate long-horizon Ask and Commit trajectories. Experiments on DDXPlus and MediQ show that our method achieves more accurate diagnoses with fewer questions across multiple large language models, while improving differential-diagnosis quality and reducing high-risk errors in severe cases. These findings validate the value of using clinical risk, rather than predictive uncertainty alone, as a planning signal and demonstrate the effectiveness of the proposed framework for information acquisition and risk-aware diagnostic decision making. They also motivate future work on clinical-risk-oriented interactive diagnosis and information-acquisition methods. 

---
# An Empirical Evaluation of Cross-City POI Recommendation on a Large-Scale Benchmark 

**Authors**: Peibo Li, Yang Song, Hao Xue, Maarten de Rijke, Flora D. Salim  

**Link**: [PDF](https://arxiv.org/pdf/2608.27840)  

**Abstract**: Cross-city point-of-interest (POI) recommendation is crucial for navigating unfamiliar urban environments, yet its progress has historically been constrained by data limitations. Using the recently proposed large-scale benchmark Trip World, we empirically re-examine whether conclusions drawn on small prior benchmarks still hold under worldwide coverage, low home-destination region overlap, and large, semantically rich POI inventories. Our evaluation surfaces three bottlenecks of representative state-of-the-art methods: (1) hometown-aware models appear to rely more on destination-region priors than on user-specific preference transfer; (2) their accuracy-efficiency trade-off degrades at this scale, where the simplest model is among the strongest; and (3) existing mechanisms for integrating semantic metadata yield little benefit. We further include a diagnostic pilot on agentic methods adapted from next-POI recommendation, finding that naive adaptation trails a simple popularity prior even though the relevant semantic signal is present in the data. These results highlight the need for task-specific designs that support cross-city preference transfer, semantic grounding, and scalable reasoning over unseen destination inventories. 

---
# KLOD: Locality-Preserving Knowledge Editing via Non-Target Distribution Preservation 

**Authors**: Hojun Jeong, Gyunyeop Kim, Sangwoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2608.27839)  

**Abstract**: Fine-tuning-based knowledge editing is simple and architecture-agnostic, but standard cross-entropy increases the edited target probability without explicitly constraining changes in the non-target output distribution. In sequential editing, such unconstrained redistribution can accumulate as distributional drift and contribute to locality degradation. We propose KLOD, a bounded and distribution-preserving objective for fine-tuning-based knowledge editing that separates the intended target update from distributions that should remain stable. KLOD stops target amplification once a probability threshold is reached, while preserving the target-excluded non-target distribution at target positions and the full next-token distribution at prefix positions. Experiments on CounterFact and ZsRE with Llama3-8B-Instruct and Qwen2.5-7B-Instruct show that KLOD substantially mitigates locality degradation while maintaining high edit reliability. The target probability threshold further provides a controllable Generalization--Locality trade-off. Ablation, multi-seed, and distributional KL analyses support the interpretation that KLOD's locality gains are associated with preserving output distributions rather than simply weakening the edit. Code is available on GitHub this https URL . 

---
# RealSWE: A Compositional Evaluation of Coding Agents under Realistic User Requests 

**Authors**: Gyuhyeong Kim, Hyojung Gwon, Jeonghyeon Kim, Kyuhong Shim, Sunjae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.27831)  

**Abstract**: Coding agents are now commonly evaluated on the SWE-bench family of benchmarks, whose tasks are built from curated GitHub issues--long, structured, and information-rich. Real user requests, however, are typically far shorter and less structured. To characterize this gap, we define a six-category information taxonomy and four dimensions of linguistic style, and apply them to real user prompts from SWE-chat and problem statements from SWE-bench Verified and Pro. We find that requests carrying only a problem statement, alone or with limited additional context, account for 88% of real prompts but just 7% of benchmark problems. Furthermore, 87% of real prompts are casually written whereas 94% of benchmark problems are formal. Guided by these observations, we introduce sys, 381 multi-variant task families derived from SWE-bench Verified and Pro. Variants within each family share the same underlying task and gold patch while differing only in information composition and linguistic style. Evaluating seven contemporary LLMs with sys, we find that i) realistic inputs reduce resolution rates by 6.4 pp on average and can change model rankings. Controlled analysis further shows that ii) including Desired Behavior and Motivation significantly affects performance, whereas Environment Information and Reproduction Steps merely add tokens without measurable benefit; iii) linguistic style has only small, model-dependent effects. These findings provide actionable guidance for users and agents: explicitly stating the desired behavior and motivation--which most real prompts omit--substantially improves the LLM's software engineering performance. 

---
# Evidential-Based Higher-Order Set Argumentation Framework 

**Authors**: Shuai Tang  

**Link**: [PDF](https://arxiv.org/pdf/2608.27824)  

**Abstract**: Evidential argumentation extends Dung's abstract argumentation by requiring arguments and interactions to be backed by chains of evidence rooted in prima-facie elements. However, existing formalisms lack a unified treatment of evidential support, higher-order relations (attacks and supports targeting arbitrary elements), and collective interactions (sources as sets). In this paper, we introduce the Evidential-Based Higher-Order Set Argumentation Framework (EHSAF), which conservatively generalises several existing frameworks within a single expressive setting. We develop two complete semantics for EHSAFs: an \emph{adjacent complete labelling semantics} that admits multiple truth values (true, false, undecided) for arguments in support cycles, reflecting an open epistemic attitude toward future evidence; and an \emph{extension-based complete semantics} that follows a strict evidentialist stance, accepting only arguments with well-founded support chains. We show that these two semantics diverge in the presence of support cycles, and prove their equivalence under support-acyclicity. To enable computational reasoning, we provide a normal propositional encoding of EHSAFs and prove that, in three-valued Łukasiewicz logic, its models correspond precisely to the adjacent complete labellings. We further extend this encoding to continuous fuzzy logics (G{ö}del, Product, and Łukasiewicz), defining a continuous fuzzy normal encoded semantics. We establish that this fuzzy semantics satisfies key properties---continuity, monotonicity, boundary conditions, and solution existence---and that its ternarisation recovers the adjacent complete labellings under natural t-norm conditions. Our framework thus unifies expressive argumentation with principled three-valued and fuzzy semantics, bridging the gap between qualitative and quantitative reasoning about evidence. 

---
# AcCoRD: Evaluating User-Agent Collaboration Under Realistic User Preference Dynamics 

**Authors**: Tejas Srinivasan, Shikib Mehri, Nandita Shankar Naik, Anirban Das, William M. Campbell, Jesse Thomason  

**Link**: [PDF](https://arxiv.org/pdf/2608.27818)  

**Abstract**: User preferences in user-agent collaboration are rarely static and fully-specified upfront: preferences are formed, revealed, adjusted, and relaxed during interaction. Existing benchmarks for evaluating user-agent collaboration focus almost exclusively on resolving underspecified preferences, thereby failing to capture the richer dynamics of real-world interaction. We introduce AcCoRD, a user-agent collaboration benchmark requiring agents to handle diverse user preference dynamics in two domains: online shopping and travel planning. We evaluate five frontier LLMs under two prompting strategies: vanilla ReAct, and an uncertainty-guided variant that prompts models to identify and resolve ambiguity about user preferences. Our results reveal that frontier models can handle underspecification but struggle to satisfy preferences that emerge or evolve mid-interaction and require more sophisticated uncertainty modeling. Further, prompting alone fails to elicit the required uncertainty recognition. We release AcCoRD as a resource for developing agents that can navigate the full complexity of real-world user preferences. 

---
# CURA: Certified Runtime Alarms for Computer-Use Agents 

**Authors**: Divake Kumar, Sina Tayebati, Devashri Naik, Amanda Sofie Rios, Nilesh Ahuja, Omesh Tickoo, Ranganath Krishnan, Amit Ranjan Trivedi  

**Link**: [PDF](https://arxiv.org/pdf/2608.27808)  

**Abstract**: Self-report is the cheapest oversight channel a deployer has, and on capable computer-use agents (CUAs) it fails precisely where oversight matters. On 361 OSWorld tasks our pipeline, a read-only feasibility gate, a planner, and a GUI executor, reaches a mean task score of 82.9, above the 72.4 human reference, yet 64 of its 71 failures (90%) end with a success claim, 61 acknowledging no blocker, and the explicit failure affordance is never used in roughly 9,100 calls. We introduce CURA (Certified Runtime Alarms for Computer-Use Agents), an external monitor that reads only harness-visible telemetry, with no model internals, extra LLM calls, or prompt changes, and turns the running trajectory into a sequential test with certified false-alarm control. At alpha = 0.10 its CUSUM alarm detects 42.3% of failures a median of 31 steps before termination at a realized false-alarm rate of 0.066, and risk is partly resolvable before the first action (gate probe, 0.69 AUROC). Retrospectively the composite reaches 0.828 AUROC (fold-internal floor 0.802), but its margin over a total-token baseline is not significant (Delta = +0.026, p = 0.101); the separation is online, where CURA recalls more at matched certified budgets: 0.41 versus 0.34 at alpha = 0.10, 0.56 versus 0.38 at alpha = 0.20. Alarm-gated mid-execution oversight recovers 23 of 70 failures while spending a frontier overseer on 38, giving a deployable cascade at mean score 86.8 and 84.5% full-solve (305 of 361). The certificate bounds false alarms only. We also report where behavioral monitoring is uninformative. 

---
# CEDAR: Automata as Verifiable Interfaces for Language-Guided Embodied Action 

**Authors**: Lekai Chen, Alvaro Velasquez, Ashutosh Trivedi  

**Link**: [PDF](https://arxiv.org/pdf/2608.27797)  

**Abstract**: Natural-language tasking of embodied agents is rarely just goal specification: users also impose constraints that must persist while the world changes. Code-generating LLM agents can produce plausible behaviors for such instructions, but their free-form programs provide no stable object to verify, compose with new constraints, or repair from a failing trace. We present CEDAR, a counterexample-guided framework that grounds instructions as regular languages over environment event traces. CEDAR uses a language model for semantic judgments and execution traces for correction, then represents both skills and specifications as deterministic finite automata. This turns constraints into executable finite-state objects: a learned skill can be intersected with a learned sleep at night or stay in this biome specification, yielding a controller that enforces the learned constraint by construction rather than by repeated prompting. In Minecraft, with the same simulator/API observations available to a program-generating baseline, CEDAR maintains temporal and spatial constraints that the baseline fails to preserve and amortizes reuse of learned skills, reducing cumulative LLM queries. These results suggest that regular languages offer a practical verification layer between natural-language instructions and embodied-agent policies. 

---
# ReToolSQL: Agentic Reinforcement Learning for Robust Text-to-SQL 

**Authors**: Pratik Kakkar, Chandra Dhir, Ravi Shankar, Pareekshit Reddy Gaddam, Anup Shirgaonkar  

**Link**: [PDF](https://arxiv.org/pdf/2608.27796)  

**Abstract**: Recent work has shown that reinforcement learning from execution feedback can substantially improve text-to-SQL performance, often enabling smaller models to match or exceed much larger systems. However, most existing approaches treat SQL generation as a single-turn task, limiting the model's ability to recover from errors through iterative refinement. We present ReToolSQL, a two-stage training framework for text-to-SQL that combines (i) a supervised warm-start on rejection-sampled reasoning traces with (ii) agentic reinforcement fine-tuning (RFT) over multi-turn tool-use trajectories. The key insight is that the two stages act on complementary axes, the supervised fine-tuning (SFT) on verified privileged-teacher traces expands the set of solvable questions (raising pass@k coverage on the hardest cases), while RFT converts that expanded capability into higher single-pass accuracy by teaching the model when to verify, what evidence to retrieve, and how to repair faulty SQL from execution feedback. Applied to Gemma 4 instruction-tuned (31B), RFT alone achieves 73.66% execution accuracy (EX) on the BIRD-SQL development benchmark (74.12% EX with self-consistency). Initializing RFT from the SFT checkpoint (SFT$\to$RFT) yields our strongest model at 74.32% EX single-pass and 74.77% EX with self-consistency. At the time of writing, this ranked first on the BIRD single-model development-set leaderboard. The approach uses composite rewards anchored on execution correctness, requires no human annotation beyond the benchmark itself, and operates within a single dense 31B model, showing that a properly designed SFT$\to$RFT pipeline over tool-use trajectories is a practical path toward robust enterprise-grade text-to-SQL. 

---
# Credo: Reusable Declarative Primitives for Agentic Workflows 

**Authors**: Duo Lu, Andrew Crotty, Uğur Çetintemel  

**Link**: [PDF](https://arxiv.org/pdf/2608.27790)  

**Abstract**: An LLM application depends on both a model and a harness: the program that determines what each call sees, how many calls to make, and which answers to trust. Coding agents can now discover strong harnesses by searching over candidate programs, but the resulting artifact is an opaque block of imperative code whose logical steps, runtime signals, physical execution decisions, and prompt strategies remain implicit and task-specific, forcing subsequent tasks to start the harness search process from scratch. The potential for reuse, however, is substantial. A searched harness encodes significant knowledge, such as the logical steps that work, the signals that matter, the physical operator decisions that adapt execution, and the prompt strategies that are effective, yet this knowledge is buried in imperative code with no inspectable or reusable structure, nor does it carry any provenance or metadata. Credo addresses this problem by recovering a structured declarative description of a searched harness, tagging each extracted primitive with relevant metadata, and cataloguing all of it with provenance. A compiler can then bind stored primitives to generate harnesses for new tasks without having to start the search over from scratch. This paper provides preliminary results demonstrating the potential of our approach and lays out a related research agenda that the database community is well-positioned to tackle, including cost-based compilation over declarative catalogs and catalog maintenance under model and workload drift. 

---
# Why Didn't It Check? Unsupported Final Claims and Their Repair in Two Tool-Equipped Language Models 

**Authors**: Justin Bronder  

**Link**: [PDF](https://arxiv.org/pdf/2608.27768)  

**Abstract**: A language model with access to tools can commit to a final claim unsupported by the evidence it has seen, even when a single available tool call would resolve the uncertainty and its instructions explicitly forbid assumptions and guesses. We separate this failure into two precisely defined quantities: occurrence, how often the model makes an unsupported claim on its own, measured from the visible evidence and final claim without using the hidden correct answer; and conditional repair, how often those same naturally occurring unsupported claims are repaired when the missing evidence is supplied. On one fixed Qwen3-32B setup, 33 of 512 first responses to 256 new prompt templates ended with an unsupported established claim. We replayed each case from an exact copy of the state in which the claim occurred; within each matched replay, the alternative tool responses had the same structure and length and differed only in a one-character response code. Resolving evidence repaired 33 of 33 claims; a matched response carrying no useful information repaired 0 of 33. When the evidence supported the original answer, the model preserved 33 of 33, with no observed harm. In a separate experiment, on 64 cases where evidence was needed, an automatic checking rule added 21 evidence calls, corrected all 10 wrong unsupported claims, preserved the 11 that were correct by accident, and never changed a correct answer into a wrong one. On a fixed Gemma 4 setup using the same sampling settings, the model called the tool in all 512 first responses and never made an unsupported final claim, so conditional repair could not be measured for that setup. These results describe two local fixed model setups on two synthetic task families. They do not show how common this failure is in real-world deployments, nor that it reflects a general mechanism shared across models. 

---
# Probing Perceptual Priors of MLLMs via Gibbs Sampling with Interpretable Generative Controls 

**Authors**: Manuel Cherep, Pattie Maes, Nikhil Singh  

**Link**: [PDF](https://arxiv.org/pdf/2608.27727)  

**Abstract**: A model's behavior on a task is jointly determined by the input it receives and the prior it brings in, i.e. the distribution over stimuli it implicitly expects. Interpretability research has traditionally studied models by holding inputs fixed and examining model responses either mechanistically, probing how internal structure represents inputs, or behaviorally, measuring how variation in inputs leads to variation in outputs. Neither reconstructs the prior distribution itself, since internal structure shows what a model can represent, not what it expects, and any fixed stimulus set leaves most of the possible input space unseen. In particular, such an input space in real-world settings, such as images seen by VLMs, is extremely high-dimensional and diverse. These priors thus remain a poorly understood component of models that nonetheless influence real-world behavior. We propose a method to sample from models' perceptual prior distributions directly, by steering a generative model to produce stimuli along controllable axes and running Gibbs sampling over that space with the model under study as the judge. We apply this to a variety of categories and target variables (such as trustworthiness in faces and cheapness in art images) and recover both canonical biases and surprising novel priors invisible to direct prompting, warranting further investigation of their downstream effects. 

---
# PCFBench: A Diagnostic Benchmark for Product Carbon Footprint Estimation 

**Authors**: Krishna Rao, Andrew Dumit, Shaena Ulissi, Jacob Feintzeig, P. James Joyce, Daniel Frank, Steven Watson, Jonathan Glidden, Gizem Ilayda Dinc, Travis M. Kwee  

**Link**: [PDF](https://arxiv.org/pdf/2608.27716)  

**Abstract**: AI systems are being deployed on high-stakes, domain-specific workflows that demand correctness not just in the final output, but at every intermediate step. One such workflow is estimating a product carbon footprint (PCF), the greenhouse-gas emissions attributable to a physical product. AI agents are increasingly being used to generate PCFs, but existing evaluations score either total emissions (hiding error sources and cancelling mistakes) or sub-tasks in isolation (missing compositional interactions). We introduce PCFBench, the first benchmark to carve PCF modeling into independently-evaluable tasks that require decomposition, retrieval, ontology matching, and numerical extraction. It comprises 614 expert-labelled items across six tasks. Together they probe reasoning under under-specification, conflicting context, and numerical constraints. Across eight frontier LLMs from four providers, no single model dominates. Although the strongest models estimate total product emissions within 2 times of declared totals on 77% of products, this rate drops to 37-58% when the PCF is generated step by step, with only 45-75% obeying mass conservation. These failures undermine the transparency practitioners need to compare products and drive decarbonization. We release the dataset and evaluation harness to support targeted progress. 

---
# Agents for Everyone: A Workshop Framework for Building Agentic AI Capabilities in a Distributed Curation Community 

**Authors**: Seth Carbon, Sierra Moxon, Kimberly Van Auken, Pascale Gaudet, Christopher J. Mungall  

**Link**: [PDF](https://arxiv.org/pdf/2608.27675)  

**Abstract**: Agentic AI has the potential to accelerate curation of biological databases and knowledge bases. However, uptake has been hindered by a number of challenges and obstacles, including access to agents and appropriate training. Here we describe how we have attempted to address and mitigate these challenges and obstacles through the deployment of a cloud-based agentic environment, and the development of an interactive training workshop for the Gene Ontology Consortium. Our cloud environment for agentic-assisted curation was based on the JupyterHub platform, and utilized Claude Code as a universal harness. This allows curators to interact with an agent session through a terminal running in the browser, and has additional benefits such as centralization of access through a single API gateway, removing the need for participants to manage subscriptions or install software locally. We created four training modules, walking participants through basic agentic tool use first and then working up to agentic biological pathway curation using the existing GO-CAM (GO Causal Activity Model) curation tool. Thirty-seven participants took part in the four-hour workshop. Our key takeaway from this workshop is that building community capability with agentic AI is primarily a problem of access, workflow design, and training. Removing technical barriers, introducing capabilities gradually, grounding exercises in familiar curation tasks, and giving curators direct experience evaluating agent output can provide a practical route toward building shared agentic AI capability in distributed scientific communities. 

---
# A Framework for Object-Centric Predictive Monitoring of Collaborative Processes 

**Authors**: Daniel Calegari, Andrea Delgado, Leonel Peña, Martín Rubio  

**Link**: [PDF](https://arxiv.org/pdf/2608.27671)  

**Abstract**: Predictive Process Monitoring (PPM) of collaborative, inter-organizational processes requires reasoning over multiple interdependent entities, including participants, messages, local executions, and the global collaboration case. Existing approaches extend traditional event logs with collaboration attributes but retain a single-case perspective, leaving much of this structure implicit. Object-centric process mining (OCPM) provides an alternative by representing these entities as first-class objects with explicit relations and multiple notions of case. This study connects collaborative PPM and OCPM through three contributions: (i) a formal semantic mapping from extended collaborative event logs to an OCED-conformant object-centric representation, serialized in OCEL 2.0; (ii) a reformulation of collaborative prediction tasks as object-centric prediction tasks; and (iii) a reproducible converter and prediction pipeline implementing the proposed mapping. We evaluate the framework on four public collaborative event logs and a fifth derived from the BPI Challenge 2013 incident-management log by executing the fourteen reformulated tasks using five predictive strategies across tabular, sequential, and graph-native encodings. We further discuss the benefits, limitations, and threats to the approach's validity. The representation makes collaboration structure explicit and makes it natural to state prediction targets based on object relations that fall outside the case-centric taxonomy, at the cost of increased relational complexity and dependence on object-centric tooling. 

---
# If Agents Were Angels, No Governance Would Be Necessary: Out-of-Band Policy Enforcement at a Trusted Tool Boundary 

**Authors**: Marc Millstone, Tyler Akidau, Johannes Brüderl, Marat Pekker  

**Link**: [PDF](https://arxiv.org/pdf/2608.27646)  

**Abstract**: Give an agent a human's credential and it inherits the person's reach without the judgment that limits its use. It can sweep every reachable record into model context, where hidden instructions steer its next call, and every request stays credential-valid while the agent exceeds its job or absorbs a secret. Prompts are a brittle guardrail: one fallible reasoner interprets the task and enforces its limits.
We present Out-of-Band Policy Enforcement (OBPE), a trusted boundary outside agent reasoning. It authorizes the typed operation and resource, narrows the query before the backend call, then filters records and fields or masks values in the response. Semantic gating can deny or hold an authorized call on argument values or external state. A data policy owner sets the maximum grant; agent policy can only narrow it. We prove, under stated conditions, that the policy plan is order-independent and agent policy cannot widen the ceiling. Field removal covers one execution; masking and history rules claim less.
We release an HTTP proxy prototype simplified from our production system, with conformance tests tying its typed Cedar policy core to the model. Against Jira and ServiceNow mocks, our benchmark compares prompted agents with and without OBPE on four models, including 20 adaptive red-team tasks. A trace failure means protected data entered agent context, an exact value appeared in the answer, or a forbidden effect completed. In 3,621 trials it fell from 57.6% to 0.2%, a cluster-weighted reduction of 41.2 points [95% CI: 27.7, 54.9]; fulfillment fell from 79.1% to 60.9%, while paired safe-useful completion rose 21.8 points [9.5, 35.2]. Some answers reconstructed a value that never entered context or used filtered row counts as an oracle: shaping one execution is not noninterference. Write controls, durable approval, and temporal and aggregate policies lie outside this evaluation. 

---
# Generative AI Expands the Intellectual Reach of Course Based Undergraduate Research Experiences (CUREs) 

**Authors**: Aditi Babar, Kristin J. Davin, Alex Dornburg  

**Link**: [PDF](https://arxiv.org/pdf/2608.27638)  

**Abstract**: Course-based undergraduate research experiences (CUREs) broaden access to authentic scientific inquiry through responsive instructor support as research problems become increasingly complex. Generative artificial intelligence (GenAI) may extend this support by providing individualized assistance that can adapt as student needs change. However, how embedding GenAI within a CURE to provide support across the research process impacts student inquiry, collaboration, and scientific reasoning remains unresolved. Here we use longitudinal qualitative data collected across three semesters of a bioinformatics and genomics CURE to show that GenAI expanded the intellectual reach of the research experience in three distinct ways. First, personalized, on-demand scaffolding allowed students to move beyond the boundaries of instructor expertise and transform their own interests into researchable inquiry, with all teams developing distinct self-directed projects rather than selecting instructor-provided topics. Second, GenAI became part of the distributed cognitive system of research teams, helping novice researchers communicate and coordinate across differentiated expertise without eliminating specialization. Third, expanded capability did not replace the need for disciplinary judgment. Students increasingly validated, revised, or rejected AI-generated contributions, such that research independence emerged through retained intellectual responsibility. Together, these findings suggest that GenAI can extend the reach of CUREs by expanding what novice researchers can investigate, how they can collaborate, and the level of responsibility they can assume while preserving human judgment central to authentic scientific inquiry. 

---
# LongGuard: Mechanistic Analysis and Training-Free Mitigation of Long-Context Failure in Safety Guardrails 

**Authors**: Ziyang Chen, Xing Wu, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2608.27580)  

**Abstract**: Safety guardrails serve as the last line of defense against harmful inputs and outputs of large language models (LLMs), yet they are trained and evaluated almost exclusively on short text. We present LongGuard, a framework that evaluates, mechanistically analyzes, and mitigates long-context guardrail failure. We formulate the task as Safety Needle-in-a-Haystack (SafetyNIAH) over a 0.25k-32k length grid; across 15 mainstream guardrails, unsafe recall drops monotonically by more than 50% on average, and a paired Benign-Fill vs. Needle-Repeat design attributes the failure to proportional dilution of the unsafe needle rather than to absolute length. A three-layer attention-logit-behavior analysis on six guardrails locates the mechanism: attention mass on the unsafe needle is diluted, the unsafe-over-safe logit margin is compressed in lockstep, and the detection decision collapses accordingly, with this attention->logit->behavior chain remaining consistent after partialling out length. We further isolate a sparse set of guard-specialized retrieval heads that exhibit partial specificity relative to their base models. Building on the analysis, we propose two training-free mitigations - Chunked Detection (CD) and Attention-Head Sharpening (AHS) - and a deployment protocol, Context-Aware Hyperparameter Routing (CAHR), that selects configurations by context length and audit side. Across five benchmarks spanning synthetic data, long-context attacks, and reasoning-model outputs, CAHR-CD and CAHR-AHS improve the six-guardrail average by 22% and 13%, respectively. Code and data are available online. 

---
# Nemotron 3.5 Content Safety Moderator: A Compact Multimodal, Multilingual, and Reasoning Enabled Content Safety Moderator 

**Authors**: Varun Singh, Anuj Doshi, Makesh Narsimhan Sreedhar, Shaona Ghosh, Katherine Luna  

**Link**: [PDF](https://arxiv.org/pdf/2608.27548)  

**Abstract**: Safety moderation for deployed AI applications is moving beyond text-only prompts: systems increasingly need to judge images, documents, screenshots, and generated responses under policies that vary across domains. Existing guardrails usually cover only part of this setting, making it difficult to combine broad coverage, custom policy control, and low compute cost. We present Nemotron 3.5 Content Safety Moderator, also referred to as Nemotron 3.5 CS in this paper for brevity, a compact 4B vision-language safety moderator that jointly classifies user prompts, images, and assistant responses across 12 languages. Nemotron 3.5 CS returns safety labels for latency-sensitive moderation and can additionally produce concise reasoning traces that apply supplied custom policies and identify violated categories when reasoning is requested. We also release a multimodal and multilingual safety dataset for guard training, spanning human-labeled real-image moderation, benign vision-language and document tasks, synthetic rare-risk and jailbreak cases, and custom-policy examples. Across evaluations spanning multimodal safety, text moderation, multilingual robustness, custom-policy following, benign false positives, and latency, Nemotron 3.5 CS demonstrates a practical coverage tradeoff: it adds image-conditioned and policy-conditioned moderation while remaining broadly competitive with specialized guard models. These results suggest that compact vision-language moderators can serve as deployable front-line safety components, with reasoning used selectively for audit and policy review. 

---
# SETU: An Agentic Ecosystem for Multilingual, Persona-Aware Communication Coaching 

**Authors**: Jonnalagadda Maruthi Tejas, Uponika Barman Roy, Tilottama Goswami, Samir Goswami, Mousita Dhar  

**Link**: [PDF](https://arxiv.org/pdf/2608.27524)  

**Abstract**: Corporate training teams need scalable and explainable tools to improve workforce communication in multilingual settings. Existing systems often score text, audio, or video in isolation, or produce black-box outputs that are difficult to audit for coaching use. This paper presents SETU, an agentic ecosystem for corporate communication coaching aimed at recruiters, frontline sales professionals and training units who prepare for audience specific conversations. SETU is designed for two scoped scenarios: (i) recruiter-candidate eligibility-and-interest calls with persona context and (ii) sales pitches with target-audience adaptation; owing to limited evaluation resources, this paper reports results on scenario (ii) only. The ecosystem decomposes analysis into specialized video, audio-speech, text-relevance, scoring, notification and reporting agents coordinated through trust-aware orchestration. It generates modality-attributed coaching reports for formative training, with human reviewers retaining final judgment. The name SETU (bridge in several Indic languages) reflects the goal of bridging communication gaps across regional languages and audience expectations. 

---
# WM-R1: Training GUI Agents to Reason and leverage World Models with Reinforcement Learning 

**Authors**: Yu Han, Tianwen Qian  

**Link**: [PDF](https://arxiv.org/pdf/2608.27508)  

**Abstract**: GUI agents trained with reinforcement learning (RL) have showcased strong environment learning capabilities on mobile platforms. However, RL typically demands extensive real-environment interactions, leading to high resource costs and instability, especially in GUI scenarios. To address these, we propose WM-R1, the first reinforcement learning framework that trains mobile GUI agents with world models instead of real environments. Specifically, world models serve as the source of state transitions during all rollouts, replacing the real Android environment within the training loop. WM-R1 also embeds world models directly into the thinking process, enabling agents to reason about the consequences of candidate actions before committing to the final action. Crucially, WM-R1 eliminates the need for real-environment interaction, supports massively parallelized and step-level granularized trajectory generation grounded in world models, and introduces a multi-dimensional rule-based reward that jointly optimizes task success, trajectory efficiency, and world model utilization. For efficient training, we curate a high-quality dataset of 2000 challenging tasks. Experiments on Android mobile benchmarks demonstrate that WM-R1-trained agents significantly outperform GRPO-only baselines and inference-time simulation methods. Code is available at this https URL . 

---
# Thinking Costs Tokens: When More Structure is Worth the Price 

**Authors**: Thomas Nolasque, John Grey, Calista Pham, Ankit Vani  

**Link**: [PDF](https://arxiv.org/pdf/2608.27506)  

**Abstract**: Adding inference structure to a language model lets it search, verify, and revise, but these actions consume the very budget they are supposed to use well. In this paper, we investigate whether there exists a token-budget threshold, below which the overhead of planning and verification hurts performance and above which it helps. We evaluate two systems on FinQA and TAT-QA financial reasoning tasks, using GPT-5.4 mini across 14 budget tiers ranging from 250 to 42,000 output-equivalent tokens. The first system is a monolith, which is a single LLM call. The second is a verified search architecture that adds planning, label-blind checking, and repair capabilities. We run 1,000 cases for a total of 28,000 completed cells. Both systems score 0% at the two lowest tiers, where neither can fit a complete prompt. At 1,000 tokens, the monolith reaches 18% accuracy while verified search scores near 0%, since the planning overhead leaves no room for an answer. From 1,500 tokens onward, verified search surpasses the monolith and maintains a consistent advantage, reaching approximately 44% at the highest tiers while the monolith reaches approximately 40%. The crossover occurs between 1,000 and 1,500 output-equivalent tokens, confirmed by a strict intersection-union test ($p \le 0.001$ at both endpoints). 

---
# CareGraph: An Auditable Hybrid AI Framework for Evidence-Grounded Personalized Longitudinal Health Intelligence 

**Authors**: Pratik Ghawate, Tanvi Patil  

**Link**: [PDF](https://arxiv.org/pdf/2608.27484)  

**Abstract**: Artificial intelligence is transforming personalized healthcare, yet fragmented clinical, self reported, and wearable evidence remains difficult to interpret and trace. We present CareGraph, an auditable hybrid AI framework that converts heterogeneous records into prioritized trends, missing context indicators, bounded next steps, discussion questions, and provenance linked explanations. CareGraph organizes evidence without diagnosing, predicting outcomes, selecting treatment, or making autonomous clinical decisions. Its pipeline covers deterministic analysis, context detection, graph construction, constrained language model synthesis, evidence validation, safety controls, and release gating. Tests used synthetic cohorts of 400 patients each for development, validation, and holdout. On holdout data, a frozen ordinary least squares trend rule with a sufficiency gate achieved 0.827 accuracy, 0.837 macro F1 with a 95 percent confidence interval of 0.819 to 0.854, and 0.974 insufficient data F1. Missing context detection achieved 0.815 strict micro F1 versus 0.318 for the legacy detector. On an authored holdout benchmark, safety ruleset version 1.2 achieved 1.000 precision, 0.950 recall, and 0.974 F1. An audit requiring graph retrieval across 80 patients yielded 79 syntheses and 78 presentations without fallback; one output was blocked and one failed closed because of an invalid evidence key. Against monolithic GPT 5.6 on 56 matched patients, CareGraph was faster at 40.15 versus 49.62 seconds, shorter at 661 versus 1,163 words, and showed better exploratory lexical alignment with longitudinal targets; the baseline used fewer tokens and cited more raw evidence. Graph auditing verified provenance and deterministic retrieval; incremental graph effects on generation require paired evaluation. CareGraph offers a safety bounded foundation for intelligent personalized health systems. 

---
# Context Localization for Generalized Level-Based Evaluation in Knowledge-Based Systems 

**Authors**: Ondrej Hutník, Natália Puškárová  

**Link**: [PDF](https://arxiv.org/pdf/2608.27482)  

**Abstract**: We study context localization for generalized level-based evaluation in knowledge-based systems. The framework models situations where a structured nonnegative score, defined on facts, rules, cases, criteria or evidence units, is evaluated through conditional aggregation tests on admissible knowledge contexts. The generalized level measure maximizes a monotone set function over all contexts whose aggregated support reaches a prescribed level. We characterize when filtering the score by a context $B$ is equivalent to localizing the admissible contexts by intersection with $B$. The main theorem shows that this consistency holds for all monotone set functions if and only if two structural conditions are satisfied: monotonicity with respect to contexts and a reduction property excluding positive localized support outside $B$. We analyze pointwise and block-generated mechanisms producing the reduction property, extend the result to parameterized systems, and interpret it as a stability criterion for context-dependent evidence selection, non-additive support evaluation and level-based knowledge aggregation. 

---
# Effectiveness of IoT and Deep Learning for Detection and Severity Assessment of Postelectrotermes militaris in Tea Plantations 

**Authors**: D.K.C. Senevirathna, A.A.E. Nanayakkara, H.M.C.K. Kulathunga, J.K.D.P. Nadula, R.M. Mapatuna, Malithi Nawarathne, Jaliya L. Wijayaraja, P.D. Senanayake, Samitha Vidhanaarachchi, Kalpani Manathunga  

**Link**: [PDF](https://arxiv.org/pdf/2608.27480)  

**Abstract**: Tea plantations are vulnerable to Postelectrotermes militaris, commonly known as the Upcountry Live Wood Termite (ULWT), which can cause substantial damage when infestations remain undetected. This study proposes an IoT-enabled acoustic monitoring framework integrated with deep learning for early detection and severity assessment of ULWT infestations in tea plantations.
Research Method: Audio signals were captured non-invasively from tea trunks using a high-sensitivity microphone connected to a Raspberry Pi-based IoT device, with geographic coordinates recorded for spatial tracking. After trimming, resampling, and segmentation, 2,000 ten-second samples were obtained, comprising 1,000 healthy and 1,000 infested samples, and divided into 1,600 training, 200 validation, and 200 test samples. The dataset used in this study is publicly available on Kaggle (Senevirathna et al. 2026). Fourier-derived spectrograms trained a CNN for infestation classification and probability estimation. A weighted severity model combined CNN probability, mean acoustic amplitude, and nearby infested plants within 5 m, with geospatial mapping used to visualize infestation distribution.
Findings and Values: Field trials in a ULWT-affected tea plantation in Pundaluoya demonstrated feasibility under realistic environmental noise. On the held-out test set, the CNN achieved 81.5% accuracy, 80.6% precision, 83.0% recall, 81.8% F1-score, and 0.819 ROC-AUC. Beyond binary infestation detection, the framework introduced quantitative severity assessment using infestation probability, acoustic amplitude, and nearby infested plants. The resulting severity and geospatial outputs can support plantation managers in identifying high-risk areas, prioritizing field inspections, and implementing more timely and targeted control measures. 

---
# Benchmarking General Mobile Assistants in Challenging Real-World Scenarios 

**Authors**: Yiqi Zhu, Feiyu Gao, Jiaxing Fan, Jiahui Zeng, Minggang Wu, Chenliang Li, Haiyang Xu, Peng Li, Ming Yan, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.27477)  

**Abstract**: Graphical user interfaces have emerged as an important environment for evaluating autonomous AI agents on multimodal interactive tasks. Existing benchmarks such as AndroidWorld and MobileWorld provide strong foundations for mobile agent evaluation, but their application coverage and task design do not yet fully capture the diversity and complexity of realistic mobile use. We present GMA, a benchmark for evaluating general mobile assistants in challenging real-world scenarios. GMA introduces seven applications based on open-source projects, spanning domains such as lifestyle sharing and travel planning, and 300 tasks across four difficulty tiers, from atomic actions to complex multi-step workflows. We evaluate eight frontier models and find that performance declines substantially as task complexity increases, with current agents remaining far from reliably handling realistic user requirements. We further conduct controlled ablation studies of agent harness choices, including context retention and explicit state tracking, under a shared environment, model setting, and task taxonomy. Results show that appropriate harness design can meaningfully improve performance, particularly on demanding workflows, while the effectiveness of specific designs can vary across foundation models. Overall, GMA complements existing benchmarks by expanding application coverage and task complexity, providing a challenging testbed for evaluating mobile agents and studying how harness design supports reliable execution in complex mobile workflows. 

---
# Class-Based Heuristic Selection for Solving the Flying Block Puzzle 

**Authors**: Sanyar Ahmadi, Pedram Asadzadeh, Amanj Khorramian  

**Link**: [PDF](https://arxiv.org/pdf/2608.27476)  

**Abstract**: Heuristic search underlies planning in autonomous systems ranging from warehouse logistics to robotic navigation, yet generic heuristics fail to exploit the structural constraints that govern constrained spatial domains, causing search performance to degrade catastrophically on harder instances. We study this problem through the two-column Flying Block Puzzle, a rigorously NP-complete spatial planning microworld whose bottleneck geometry mirrors clearance-to-size constraints encountered in multi-agent path finding, autonomous vehicle navigation, and block relocation systems. We introduce the Class-Based Heuristic A* (CBHA*) algorithm, which integrates a General Move Constraint to capture minimum displacement costs when vacant units are scarce, a formal kinematic taxonomy partitioning the state space into seven mutually exclusive classes with provably admissible heuristics based on vacancy ratio and goal-piece geometry, and a class-conditional tie-breaking mechanism that dynamically switches between depth-priority and vertical-distance ordering to overcome f-value plateaus. Over 146 benchmark instances, CBHA* achieves a 93.4% success rate against 64% for Depth-Prioritized A*, 39% for Standard A*, and 17% for BFS, while reducing node expansions by 87.98% relative to Standard A* and sustaining an average effective branching factor of approximately 3, demonstrating that class-triggered adaptive heuristics constitute a principled mechanism for efficient spatial planning that generalizes structurally to physical constraint systems. 

---
# Hypothesize, Evaluate, Refine: A Scientific Agent for PDE Discovery with Unknown Spatial Coefficient Fields 

**Authors**: YuJie Huang, WenWu He, ZhuoEr Lin, Congcong Liu, Dong Liang, Zhuo-Xu Cui  

**Link**: [PDF](https://arxiv.org/pdf/2608.27475)  

**Abstract**: Discovering PDEs in heterogeneous media requires jointly identifying the governing operator and the unknown spatial fields that parameterize it. These tasks are coupled: changing field placement changes the differential law, while a sufficiently flexible field can conceal structural error on a single trajectory. We present Hypothesize, Evaluate, Refine for PDE Discovery (HER-PDE), a scientific-agent framework that discovers compositional PDE structure together with nonparametric, time-invariant coefficient fields. The Agent analyzes two noisy trajectories generated by different excitations, proposes complete expression-tree hypotheses, and combines creative structural exploration with local candidate refinement. Its Hypothesis Evaluation Interface (HEI) estimates only the fields explicitly declared in each hypothesis, never adds missing terms, and scores structures by bidirectional cross-excitation transfer. The selected law is subsequently audited on a sealed temporal interval. Across five controlled two-dimensional systems observed with 5 percent relative Gaussian state noise, the Agent recovers the generating operator in all five cases, including equivalent signed-field and product-rule parameterizations. Across nine unknown coefficient fields, the recovered fields attain a median Pearson correlation of approximately 0.85 and a median relative L2 error of approximately 0.28. These results show that agent-guided hypothesis refinement can recover heterogeneous governing laws without prescribing a parametric form for their spatial coefficients. 

---
# LLM-Augmented Causal Discovery: Probabilistic Fusion of Edge Existence and Orientation 

**Authors**: Neville K. Kitson, Anthony Constantinou  

**Link**: [PDF](https://arxiv.org/pdf/2608.27472)  

**Abstract**: Bayesian network structure learning (BNSL) from observational data struggles with orientation identifiability, while large language models (LLMs) offer broad but often unreliable causal knowledge. We propose combining these complementary sources through a novel representation, termed Probabilistic Dependency Graphs (PDGs). In a PDG, each edge is associated with a distribution over directed, undirected, and absent states, enabling fusion via weighted averaging. We evaluate this approach on 26 benchmark networks, combining ensembles of three BNSL algorithms (FGES, Tabu, PC) with three LLMs (Gemini, Claude, GPT) across multiple prompts and random seeds. A simple 50/50 fusion improves F1 over the better of either source alone in 22 of 26 networks, with a statistically significant mean improvement of $0.056$ $(p<0.001)$. Analysis reveals that the two sources play complementary roles: BNSL contributes a high-recall edge skeleton (80\% vs 60\% for LLM), while LLM contributes accurate edge orientation (96\% vs 77\% for BNSL). Our results show that representing both sources as probabilistic uncertainty over edge existence and orientation is a practical and effective way to improve causal graph accuracy. 

---
# Retrieving Relations, Detecting Fallacies: A RAG Approach to Political Debate Analysis 

**Authors**: Deborah Dore, Greta Damo, Elena Cabrio, Serena Villata  

**Link**: [PDF](https://arxiv.org/pdf/2608.27471)  

**Abstract**: Fallacies are arguments that employ invalid reasoning, making their automatic detection critical in sensitive contexts such as high-stakes political debates, where public opinion is shaped. Spotting a fallacious argument requires contextual knowledge beyond its pure surface text. This entails world knowledge pertaining to the subject matter under discussion, as well as knowledge of the relationships that exist between arguments within the argumentative discourse. Prior work on fallacy analysis has shown that argumentative discourse structure can beneficially improve classification performance. However, such structure is typically encoded only as static classifier features, limiting its flexibility. Building on this intuition while addressing this limitation, we introduce a guided retrieval-augmented methodology for fallacy detection and classification that leverages argumentative relations of support and attack to dynamically steer the extraction of relevant documents. We evaluate our approach on the ElecDeb60to20 benchmark across 42 retrieval configurations and 14 models, performing retrieval over a 15GB knowledge base of collected political-related documents. Our approach improves macro-F1 up to 0.864 for fallacy detection and up to 0.725 for classification over non-retrieval baselines. These results show that incorporating external knowledge significantly enhances fallacy detection and classification when retrieval is argumentatively guided. 

---
# Not All Explanations Are Sought: Information-Seeking Psychology for Human-Centered XAI 

**Authors**: Andrea Beretta, Salvatore Rinzivillo  

**Link**: [PDF](https://arxiv.org/pdf/2608.27464)  

**Abstract**: This position paper argues that human-centered explainable AI (HCXAI) should incorporate insights from the psychology of information seeking. Drawing on Sharot and Sunstein's framework of information-seeking motives, we propose that people evaluate whether to engage with explanations based on three types of expected utility: instrumental (will it help me act better?), hedonic (will it make me feel better?), and cognitive (will it improve my understanding?). Each utility is estimated through a lens shaped by well-documented cognitive biases, including illusion of control, automation bias, unrealistic optimism, impact bias, overconfidence, and confirmation bias. These biases can lead to two failure modes: excessive information-seeking that fragments attention without improving decisions, and insufficient information-seeking that leaves critical risks and misunderstandings unexamined. This challenge is particularly acute for agentic AI systems, where explanations must support not just understanding a single output but anticipating cascading actions, assessing risks, and deciding when to intervene. By integrating information-seeking psychology into HCXAI, we advocate for a shift from making explanations available to making them sought: designing systems that account for when and why users actually want to know. 

---
# Rating the Raters: Rasch Measurement Theory for LLM Evaluation 

**Authors**: Pratik S. Sachdeva, Nathan Boudol  

**Link**: [PDF](https://arxiv.org/pdf/2608.27463)  

**Abstract**: LLMs now sit on every side of evaluation: as examinees scored on benchmarks, judges of other models' outputs, and raters of human-generated content. Each paradigm can be viewed as a measurement problem, where a latent property of an object is probed with items from an instrument (e.g., benchmark) by raters. Standard evaluation practices often neglect the contributions of each core component to the end result, limiting our understanding of what is being measured. Rasch measurement theory (RMT) is well-suited to this kind of problem. RMT decomposes ordinal ratings into separable facets on a common scale. It further provides a battery of diagnostics that can identify miscalibrated measurements and rater biases. We present a case study of RMT applied to the LLM-as-rater paradigm using the Measuring Hate Speech corpus, whose construct was itself built under RMT. We fit a series of many-facet Rasch models to annotations from nine LLMs spanning families and capability levels. Our analyses show that LLMs systematically differ from human raters in severity, item-level calibration, question-order robustness, target-identity sensitivity, and rating scale use, which all would be obscured by standard evaluation practice. Overall, we argue that RMT belongs in the toolkit for evaluating LLM-as-examinee, -judge, and -rater paradigms. 

---
# Time Capsule of Testable Human Knowledge: 41 Years of Jeopardy! in a Single Free Local Model 

**Authors**: David Noever, Forrest McKee  

**Link**: [PDF](https://arxiv.org/pdf/2608.27459)  

**Abstract**: In 2011, IBM's Watson was something like a sealed capsule of its era's queryable knowledge. Its DeepQA system defeated the strongest human Jeopardy! champions, but the knowledge that let it do so lived in a curated billion-document corpus running on a cluster of POWER7 servers, frozen at build time and impossible to move or copy. We show that the same kind of artifact, a snapshot of what a culture can answer, is now portable and essentially free. We evaluate a single 9 GB open-weight model (Qwen2.5-14B, 4-bit) against the complete open Jeopardy! clue dataset, 529,939 clues across all 41 broadcast seasons from 1984 to 2025. To our knowledge this is the first time a model has been run over the full corpus. The 41 years mark only how long the questions were collected. What they test is far older and broader: the accumulated body of human general knowledge a culture considers worth knowing, from ancient history and dead languages to science, literature, and geography, with a verified answer for every item. The model answers 67.0% of all clues under a strict forced-response protocol with exact and fuzzy matching, and exceeds 85% on factoid categories. We treat training-data exposure as something both systems share rather than a flaw unique to language models. Watson's case is in fact the more extreme one. Its corpus was assembled to contain Jeopardy answers and it was tuned on past clues, and it could not answer anything outside that curated distribution. The decisive test is whether a model can answer clues that did not exist when it was built. On clues aired after its training cutoff, the local model holds 65% and Claude Opus 4.8 holds 95%, while Watson by construction scores zero. The capability survives the move from a server room to a file you could seal in a time capsule, and unlike Watson it is not frozen to its own moment. 

---
# Aero Hand Open: A Simulation-Ready Tendon-Driven Hand for Dexterous Manipulation Learning 

**Authors**: Nan Wang, Mohit Yadav, Jonathan Wulff, Aidan Rosenbaum, Kezhou Chen, Yuvan Sharma, Xu Dong, Yiwei Tao  

**Link**: [PDF](https://arxiv.org/pdf/2608.28578)  

**Abstract**: Tendon-driven hands are anthropomorphic, and moving the actuators off the joints is what makes a hand of this capability affordable to build. Two effects produce that saving. Routing force through a cable removes the requirement that a motor fit inside the joint it drives, so smaller and cheaper motors suffice, and one motor can drive several joints through a single cable, so fewer motors are needed. They are also harder to learn on than a direct-drive hand. The underactuated transmission that produces the saving is itself difficult to represent in a simulator, and the joints one cable drives are not independently commandable. We present Aero Hand Open, a tendon-driven anthropomorphic hand that is released simulation-ready. Three things ship with it. A simulation model reproduces the cable transmission itself. An identified actuation map connects that model to the motor commands in both directions, including the three-way coupling of the thumb. A reinforcement learning package trains policies for the hand. Together they let a policy be trained entirely in simulation and run on the hand with no fine-tuning and no state estimation. We release the mechanical design, the simulation model, the identified mapping, the training environment and the deployment stack. 

---
# Learning a Size-Weight Frontier for Synthetic-Augmented Inference 

**Authors**: Chengpiao Huang, Kaizheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.28576)  

**Abstract**: Synthetic data can improve statistical inference when real data are scarce, but naively treating synthetic samples as real data can introduce bias and lead to unreliable inference. We develop a general framework for synthetic-augmented inference across a population of related tasks. It characterizes synthetic augmentation by the number of synthetic observations and their weight. Central to our framework is a size-weight frontier that specifies, for each weight, the largest synthetic sample size for which all smaller sizes attain the target task-marginal coverage. We estimate this frontier from historical tasks, and establish a finite-sample coverage guarantee simultaneously for all size-weight configurations on or below the estimated frontier. In experiments using large language model responses to augment opinion survey data, our procedure achieves target coverage and substantially narrows confidence intervals. 

---
# Blog: Survey of Optimizers 

**Authors**: Ruoran Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.28557)  

**Abstract**: Neural-network optimization in 2025-2026 is no longer well described as a succession of new Adam variants. The design space has expanded from coordinates to matrices and layers, from fixed training horizons to policies over time, and from mathematical update rules to state representations that must survive sharding and low-precision computation. This survey organizes recent optimizers and training optimization methods along four largely independent axes: temporal estimation, update geometry, horizon management, and representation and systems. It connects the spectral normalization of Muon, the historical matrix statistics of Shampoo and SOAP, adaptive and hybrid matrix methods, memory-efficient optimizers, schedule-free training, small-batch corrections, and quantized optimizer states. The central empirical conclusion is deliberately non-triumphal: matrix-aware methods represent a genuine advance, but there is no context-independent replacement for AdamW. Rankings change with model scale, data-to-parameter ratio, batch size, schedule, parameter partition, tuning budget, and whether the target metric is tokens, FLOPs, wall-clock time, or memory. The practical consequence is a compositional view of optimizer design and a stricter protocol for evaluating optimizer claims. 

---
# Video Generative Models as Geometry Learner 

**Authors**: Haosen Yang, Jifei Song, Zhensong Zhang, Xiatian Zhu, Jiankang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2608.28549)  

**Abstract**: Recent generative approaches to geometry estimation adapt pretrained image diffusion models and treat the task as image-conditioned generation. Leveraging off-the-shelf image diffusion models, they either (i) train task-specific geometry models (for depth and surface normal estimation) independently, losing the opportunity of exploring the intrinsic correlation of these geometric targets, or (ii) jointly fine-tune modified image diffusion backbones (e.g., altered self-attention), which typically demands substantial labeled data. To overcome these limitations in a principled fashion, we repurpose pretrained video generative models as a unified and data-efficient framework for geometry estimation, formulated innovatively as a next-frames prediction task. Our method, GeoNeXt, inherits naturally structured knowledge and richer priors from the video model, while further adapting them for joint modeling of images and geometry targets (image <-> geometry), enabling more data efficient and effective learning of geometry. Extensive experiments validate our method for zero-shot monocular depth and surface normal estimation across diverse datasets, outperforming both previous task-specific and unified generative competitors while using substantially less training data. Notably, our method rivals discriminative state-of-the-art approaches trained on over 100x more data and even standouts on several benchmarks. 

---
# An Enclosed Mode Is a Gauge Choice: Topology Relative to Reach in Certified Code World Models 

**Authors**: Javier Aguilar Martín  

**Link**: [PDF](https://arxiv.org/pdf/2608.28541)  

**Abstract**: A code world model accepted by a sampling gate can be exactly right on everything the gate can see and arbitrarily wrong beyond it. We characterize what a certified model can know, and what its errors can cost, when the omission is an annular freeze mode enclosing an unreachable interior. The gate quotient makes the question precise: acceptance-with-certainty determines the model exactly on the reachable query set; beyond reach is gauge. On a minimal ring instrument we prove the extreme case (a wrong-topology filled-disc artifact unfalsifiable by any sampling gate and bitwise harmless at play) and measure, with LLM synthesis across three model families, how one knob (a channel of width gamma) walks the same artifact through three regimes: unfalsifiable-and-harmless, falsifiable-and-costly, and instantly falsified. Three principles organize the empirics. First, danger is topology relative to reach: a channel the planner can use collapses the blind model's exploitation (play cost 1.09 to ~0 over a knee at gamma ~ 0.1), while a hidden channel with the same first Betti number keeps it at full strength (1.12). Second, repair is parameter-bound and sensor-bound: no family recovers the region from outside evidence; from inside, models pose the right topology but cannot pin its parameters, and the posed topology tracks the guiding persistent-homology summary's wrong beta_1 (a sensor with a measured geometric resolution limit), not the truth. Third, mitigation must match the error's dimension and direction: point fences fail against the one-dimensional boundary, a dimension-matched persisted fence collapses exploitation to a two-lesson transient (0.999 to 0.058), and the dual freedom certificate collapses the invented-mode failure symmetrically (1.769 to 0.029). In n dimensions the shell makes misidentification near-certain while the danger stays fully exploitable: the two axes are independent. 

---
# Texture Image Classification Using DWT AlexNet Feature Fusion and Deep Neural Networks 

**Authors**: Arun D. Kulkarni  

**Link**: [PDF](https://arxiv.org/pdf/2608.28524)  

**Abstract**: Texture image classification plays a significant role in computer vision applications, including industrial inspection, medical image analysis, remote sensing, and object recognition. Handcrafted features can capture local texture characteristics but may have limited capability to represent complex visual patterns. In contrast, deep learning models automatically learn discriminative representations but may not fully exploit the multiscale spatial-frequency information inherent in texture images. This paper proposes a hybrid feature fusion framework, termed DWT_AlexNet_DNN, which combines Discrete Wavelet Transform (DWT) features with deep features extracted using AlexNet for texture image classification. 

---
# Conformal Uncertainty Quantification Guarantees for Neural Operators 

**Authors**: Tom Stent, Nicolas Boullé  

**Link**: [PDF](https://arxiv.org/pdf/2608.28515)  

**Abstract**: Neural operators provide fast surrogate models for approximating operators between function spaces, but their predictions often lack uncertainty quantification. We develop a split conformal framework to guarantee that a calibrated pointwise band around the neural operator output contains the true solution on at least a $1-\gamma$ fraction of the evaluation domain, with probability at least $1-\alpha$ over test and calibration inputs, where $\alpha,\gamma\in(0,1)$. Our method reduces a normalized residual field to its spatial $(1-\gamma)$-quantile and computes a scaling factor using a held-out calibration dataset. We prove marginal coverage guarantees for measurable residual fields defined on arbitrary probability spaces, covering both continuum domains and fixed discretizations. Under mild assumptions on the data distribution, we show that the coverage conditional on the calibration set follows a Beta distribution, which we verify with numerical experiments on Darcy flow and Navier--Stokes equations, where our calibration yields bands consistently tighter than existing corrections while retaining the target coverage. 

---
# On the Maintenance and Co-evolution of Agent Plugins: An Empirical Study of Claude Code Plugin Marketplaces 

**Authors**: Ahmed Hereiz, Yingzhe Lyu, Hao Li, Bram Adams, Ahmed E. Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2608.28497)  

**Abstract**: AI coding agents, software tools that automate development tasks through reasoning and tool use, are increasingly extended through plugin marketplaces, yet the structure, maintenance, and co-evolution dynamics of these emerging repositories remain empirically unexplored. Unlike traditional software packages that deliver functionality through source code, agent plugins deliver functionality through a combination of natural-language instruction files, scripts, and configuration files, raising the question of whether these plugins are maintained artifacts that co-evolve across components, or one-off artifacts that developers write once and do not need to revisit. To study the maintenance and co-evolution of agent plugins, we conduct an empirical study of 1,926 repositories hosting Claude Code plugin marketplaces, analyzing 8,351 plugins and 77,773 commits across 2,018 marketplaces. We find that the marketplace is expanding rapidly, plugin-touching commit activity growing 8.8x over six months after the October 2025 launch, and plugins targeting Software Engineering tasks accounting for 61.3% of all plugins. Plugin development is predominantly feature-driven, with feature commits occurring at more than twice the rate of conventional open-source software (OSS) (39.6% vs. 17.2%). Claude co-authors 34.9% of all commits, and four commit types (docs, perf, style, and refactor) carry substantially different meanings in plugin repositories than in traditional software. Most component types evolve independently, but within skills directories, natural-language instruction files and implementation scripts co-evolve at above-chance rates, with 78% of co-changes being functionally coupled, representing a new class of maintenance dependency not observed in traditional software engineering. 

---
# LLM-Based Agents for Software and Systems Security: Approaches, Applications, and Assessment 

**Authors**: Jingjing Nie, Jiawei Guo, Krishna Meda, Haipeng Cai  

**Link**: [PDF](https://arxiv.org/pdf/2608.28490)  

**Abstract**: Software and systems security workflows are typically procedural: analysts inspect heterogeneous artifacts, form hypotheses, invoke tools, interpret outputs, and revise plans. Large language model (LLM)-based agents, which can plan, use tools, retain state, and revise actions across multi-step workflows, are being rapidly adopted to automate this work. Given the consequences of delegating security decisions to autonomous systems, understanding how such agents are built, used, and assessed is crucial. Yet to this date, there remains a lack of systematic understanding of what has been done and how far we are in this field: the term "agent" is applied inconsistently, applications differ sharply in risk, and assessment protocols are often incomparable. To gain a comprehensive and coherent view of this area hence inform relevant future research, this paper provides a systematic literature review of the (1) technical approaches, including agent architecture, perception, memory, reasoning and planning, action space, orchestration, and self-improvement, (2) applications, with respect to the security tasks served, and (3) assessment, including the datasets, outcome and trajectory metrics, safety measures, and baselines considered, over the peer-reviewed literature spanning the emergence of this area (2023--2026). Our synthesis reveals a field that has built agents able to act but not yet agents whose authority is bounded or whose behavior is auditable. In addition to knowledge systematization, we also extend our insights into the limitations of and challenges faced by current approach, application, and assessment designs, which shed light on potentially promising future research directions. 

---
# How Proper Scoring Rules Shape LLM Forecasting 

**Authors**: Benjamin Turtel, Paul Wilczewski, Kris Skotheim, Ville A. Satopää, Philip E. Tetlock  

**Link**: [PDF](https://arxiv.org/pdf/2608.28482)  

**Abstract**: This paper evaluates how reward function choice shapes the performance and behavior of LLM forecasters. We compare five proper scoring rules as training objectives for binary forecasts of resolved real-world events. Although the rules share the same theoretical incentive for truthful probability reporting, the resulting models differ in calibration, probability use, and estimated profiles of bias, information, and noise, with smaller differences in aggregate accuracy and discrimination. The Brier-trained model has the lowest observed Brier score and highest AUC-ROC, while the log-trained model has the highest observed log score and lowest calibration error. Models with similar aggregate performance also reach that performance through different combinations of bias, information, and noise. Proper scoring rules therefore need not behave interchangeably as training objectives. Reward choice may shape not only how well an LLM forecasts, but how its forecasting errors are structured. Each condition uses a single seed, so some differences may reflect training stochasticity. 

---
# NL2AGBench: Benchmarking LLM Auto-Formalization for AlphaGeometry 

**Authors**: Samuel Xiao, Judy Song, Rory Hu, Ziliang Zong  

**Link**: [PDF](https://arxiv.org/pdf/2608.28481)  

**Abstract**: Recent advances in large language models (LLMs) have demonstrated strong capabilities in natural language understanding and mathematical reasoning. However, their ability to translate informal mathematical problems into formal representations remains underexplored. This limitation is particularly important for neuro-symbolic geometry systems such as AlphaGeometry, whose theorem-proving engine requires inputs in a specialized domain-specific language (DSL). Although AlphaGeometry achieves near-IMO gold-medalist performance, manually converting natural-language problems into its formal syntax remains a significant usability bottleneck. To address this challenge, we introduce the Natural Language to AlphaGeometry Benchmark (NL2AGBench), which evaluates LLMs in translating English geometry problems into AlphaGeometry-compatible formal representations. NL2AGBench uses execution-based verification within AlphaGeometry to assess translation quality rather than relying solely on textual similarity. We evaluate ten state-of-the-art open- and closed-source LLMs across multiple parameter scales and analyze executable translation accuracy, syntactic correctness, and error characteristics. Our experiments reveal a substantial performance gap between closed- and open-source models: leading closed-source models achieve executable translation rates above 80%, while even the largest open-source models struggle to consistently preserve geometric constraints and produce valid formalizations. We introduce an error taxonomy distinguishing syntax and logic errors and investigate mitigation strategies, including few-shot prompting, fine-tuning, and human-guided hinting, which yield measurable improvements across multiple model families. 

---
# Real-time virtual circuits for plasma shape control via neural network emulators: experimental demonstration on MAST Upgrade 

**Authors**: Nicola C. Amorisco, Kamran Pentland, Adriano Agnello, George K. Holt, Alasdair Ross, Matthew J. Marshall, Edward Jones, Graham J. McArdle, Charles Vincent, Timothy Nunn, Martin Kochan, Pedro Cavestany, Aran Garrod, Stanislas Pamela, James Buchanan  

**Link**: [PDF](https://arxiv.org/pdf/2608.28468)  

**Abstract**: Conventional plasma shape control in tokamaks relies on virtual circuits (VCs) that are computed offline from linearisations around a small, tailored number of reference equilibria, and deployed as expertly prepared schedules during the discharge. Here, we report on the first experimental deployment of real-time VCs. We replace pre-set look up tables with VCs updated in real time using surrogates of the plasma response. Both the existing control architecture and the interpretability of VC-based control are retained. Previous work showed that neural network emulators can produce accurate VCs, and validated their performance in closed-loop shape control simulations. Here, we report their first experimental validation on MAST Upgrade (MAST-U). Dedicated experiments spanning different scenarios, including prescribed shape perturbations, feedback-driven divertor-leg motion, and strongly evolving plasma configurations, show that real-time VCs can realise plasma shape control tasks within the MAST-U plasma control system. These results establish the experimental feasibility of real-time linearisations as a practical extension of conventional plasma shape control in tokamaks. The present implementation demonstrates a central step towards a simpler control workflow, in which manually constructed, phased VC schedules are replaced by VCs generated automatically online from a trained surrogate model, without scenario-specific retraining. 

---
# Anatomy-Aware Promptable Segmentation with Online Interactive Training for AUTOPET V 

**Authors**: Pablo Lozano-Jimenez, Sergio Romero-Tapiador, Ruben Tolosana  

**Link**: [PDF](https://arxiv.org/pdf/2608.28461)  

**Abstract**: We present an anatomy-aware, promptable model for whole-body lesion segmentation in FDG and PSMA PET/CT, developed for the AUTOPET V challenge. The proposed method is built as family of nnU-Net-based models and trained in two stages: i) a pre-training stage that produces a strong initial segmentation, and ii) an online interactive stage that learns to exploit scribble prompts, refining the prediction over successive interactions. Anatomical context is incorporated through organ supervision using a single shared head that predicts lesions and organs from the same features, which reduces false positives arising from physiological uptake. Also as the tracer (i.e., FDG/PSMA) is not provided at inference, we add a tracer classifier based on image processing and a random forest over coronal MIP features, routing each study to a combined FDG+PSMA model or to a PSMA-specific model. Across four-fold cross-validation the organ-supervised model achieves the best and most stable performance, the interactive stage improves the Dice score monotonically with each prompt, and PSMA-specific training yields the strongest tracer-wise results. 

---
# ARC-CT: Anatomy-Routed Contrastive Vision-Language Learning for 3D Chest CT 

**Authors**: Huseyin Umut Isik, Mehmet Alp Ozaydin, Sila Kurugol, Şeyda Ertekin  

**Link**: [PDF](https://arxiv.org/pdf/2608.28455)  

**Abstract**: Contrastive vision-language learning uses paired chest CT volumes and radiology reports to learn abnormality classifiers without manually annotated labels. However, two characteristics of chest CT challenge conventional global contrastive learning. First, many critical abnormalities are small or anatomically localized, and pooling an en- tire volume into a single embedding may dilute their visual evidence. Second, the standard contrastive objective treats every other scan in a batch as a negative. Because many chest CTs share abnormalities, this objective incorrectly pushes co-positive pairs apart. We propose Anatomy-Routed Contrastive Learning for 3D Chest CT (ARC-CT), a region-aware framework that addresses these limitations using only la- bels extracted from reports by an LLM, with no manual annotations or bounding boxes. ARC-CT combines three components: (1) an Anato- myQFormer localizing evidence via queries constrained by automatically generated organ masks; (2) a label-Jaccard soft InfoNCE objective in- tegrating the standard one-hot target with the label-set overlap of each pair, which reduces false-negative penalties between studies that share clinical findings; and (3) an organ-level alignment loss connecting mask- pooled visual features to organ-specific report text extracted offline with a large language model. ARC-CT achieves a 0.86 mask-free macro AUC across 18 abnormalities using a compact 3D ResNet-18 backbone. Over- all, ARC-CT outperforms both comparable efficient baselines and sev- eral larger transformer models. Our code and weights are available at this https URL. 

---
# Fidelity Is Not Enough: Dispatch-Level Instrumentation for Agentic Datasheet Extraction 

**Authors**: Qing Ye, Meng-Hsuan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2608.28439)  

**Abstract**: One model passed our fidelity check without ever opening the datasheet. We found it while qualifying models for an internal extraction service: a structured-output constraint had silently disabled tool use, and the model answered anyway, with fabricated source text. Only the per-tool trace exposed it. Fidelity -- whether an extracted value matches the source -- is the standard measure for agentic document extraction, and it scores that run a success. We therefore log every tool call in an agentic benchmark of 25 hand-curated claims over three components, with 12 more on a fourth, 37 in all. From that dispatch record we build two instruments: a rule-based failure-attribution classifier, and a silent-failure detector whose two rules check only which tools were called, never the extracted value. The detector raises no flag on 207 clean fidelity-passing extractions across three model families, and recovers all 50 planted faults that withhold exactly the tools its rules check. The two results are not symmetric: the first bounds the false-positive rate, the second is recall by construction, and detection power against runs that call their tools and still answer wrongly is unmeasured. A second, independent oracle, a causal chamber that tests whether the datasheet's claims hold under physical measurement, is intentionally partial: it confirms only what the apparatus can exercise, a verifiable envelope of 2 of those 37 claims, and we give a taxonomy of why the rest are not physically gradable. Under a controlled perturbation, fidelity passes throughout while the chamber verdict flips exactly at the measurement uncertainty. Across three deployed model stacks (one destabilised by its serving stack, not by any capability gap) the tool layer buys portability and observability rather than accuracy, and earns its premium only once a document outgrows the context window. 

---
# Are These Modules Worth Their Cost? A Paradigm-Level Accuracy-Cost Analysis of In-context Learning Text-to-SQL 

**Authors**: Jiayan Lin, Yujia Liu, Zijin Hong, Zheng Yuan, Yilin Xiao, Hao Chen, Qinggang Zhang, Xiao Huang, Feiran Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.28432)  

**Abstract**: Recent advances in in-context learning (ICL) text-to-SQL have substantially improved execution accuracy on public benchmarks by assembling increasingly elaborate pipelines around the base generator, yet existing studies typically report aggregate end-to-end accuracy, without quantifying the marginal accuracy-cost contribution of individual design choices. Consequently, providing a unified, paradigm-level cost-accuracy quantification remains a critical challenge for understanding and configuring modern text-to-SQL. To address this, we instantiate 17 paradigm-level configurations across five recurring modules of the ICL text-to-SQL pipeline under a single controlled implementation, and attribute each paradigm's marginal contribution and incurred cost across all four backbones spanning diverse capability levels and reasoning styles. Our analysis reveals that execution-feedback refinement is the only paradigm whose benefit holds universally at consistently low cost, while most other modules help only under backbone-dependent conditions. Token accounting shows that input demand is more closely tied to pipeline structure, whereas output demand is more sensitive to backbone generation behavior. Cross-module analysis further shows that stacking improves accuracy on most backbones, although how the gains compose varies with backbone capability. We also find that a fixed budget is often better spent engineering a more elaborate pipeline over a mid-tier backbone than upgrading to a frontier model with a lean pipeline. These findings distill into an actionable, cost-aware tiered guideline that transfers to five additional backbones without per-paradigm search. 

---
# LongPIBench: A Long-Context Benchmark for Prompt Injection 

**Authors**: Yupei Liu, Yuqi Jia, Neil Zhenqiang Gong, Jinyuan Jia  

**Link**: [PDF](https://arxiv.org/pdf/2608.28411)  

**Abstract**: Prompt injection attacks pose a serious security risk to large language models in real-world applications. However, existing prompt injection benchmarks primarily focus on short-context inputs, leaving the attacks and defenses in long-context settings largely unexplored. This gap leads to a substantial overestimation of the effectiveness of current defenses. In this paper, we bridge the gap by introducing LongPIBench, a long-context benchmark for prompt injection covering 4 realistic application scenarios: paper peer review, resume screening, code review, and email summary. For each scenario, we construct a synthetic dataset and a real-world dataset, with context lengths ranging from thousands to tens of thousands of tokens. The evaluation results on LongPIBench reveal significant vulnerabilities of prompt injection defenses under long-context settings: even simple heuristic prompt injection attacks achieve high success rates and frequently bypass state-of-the-art defenses. We hope LongPIBench can serve as a practical benchmark for systematically evaluating prompt injection defenses in realistic long-context scenarios. 

---
# When Linguistic and Internal Confidence Diverge in Large Language Models 

**Authors**: Hefan Zhang, Bingquan Zhang, Ming Cheng, Saeed Hassanpour, Weicheng Ma, Soroush Vosoughi  

**Link**: [PDF](https://arxiv.org/pdf/2608.28382)  

**Abstract**: Users often ask large language models (LLMs) to report how confident they are, but it is unclear whether such linguistic confidence tracks the model's internal confidence. We study this question across 8 classification tasks, 2 generation tasks and 30 models from three families. For classification, we compare linguistic confidence with logits-based confidence along three axes: association, magnitude agreement and calibration. For generation, we test whether linguistic confidence tracks semantic-entropy-based uncertainty. The axes frequently diverge. Instance-level association is weak on average, although it improves on easier items and for stronger base models. Instruction-tuned models often report higher confidence and sometimes show higher association, but they also have larger confidence gaps and worse calibration. Prompt design mostly changes the distribution of reported confidence. Attitude cues inflate confidence without improving alignment, while score exemplars can preserve rank-order signal when they avoid collapsed confidence values. Regression analyses show that distributional properties of confidence scores explain much of the observed alignment pattern, with model metadata playing a smaller role after controls. These results support a lossy-channel view of linguistic confidence. A more dispersed verbal confidence distribution can carry useful rank information, but it does not make the scores calibrated. Linguistic confidence should therefore be evaluated with multi-axis diagnostics before being used in downstream reliability pipelines. 

---
# AI as Teammate: Rethinking Task Distribution in Medical Training 

**Authors**: Fendi Tsim, Alina Gutoreva, Anthony Weiss, Nicole Dubosh  

**Link**: [PDF](https://arxiv.org/pdf/2608.28373)  

**Abstract**: Integrating Artificial Intelligence (AI), particularly generative AI, into medical training has prompted concerns about learner over-reliance, misuse, and erosion of foundational clinical competencies. We propose a conceptual reframing at the decision level: the problem is not misuse but misclassification - a mechanistic failure of real-time metacognitive evaluation in selecting a subzone-inappropriate AI interaction mode. Drawing on "SCAN" (Substitute, Complement, Aid, Non-Negotiable), a human-centric decision-making framework for generative AI task allocation grounded in Vygotsky's Zone of Proximal Development and metacognition, we advance the emerging social-constructivist conversation around AI in medical education by offering a testable account of AI's role in clinical reasoning development. This framework yields testable predictions for how misclassification can be detected, mitigated, and, more importantly, prevented in the clinical learning environment. Regarding clinical reasoning development, we show how trajectories of skill acquisition (upskilling) and failure (the triad of skill failure: de-skilling, never-skilling, and mis-skilling) operate at the individual task level in ways that fixed-phase, cohort-wide treatments fail to capture. We further identify passive engagement within correctly classified AI-scaffolded tasks as a particularly insidious, detection-resistant pathway to mis-skilling - one requiring subzone re-identification from AI assistance to expert assistance, with human experts serving as epistemic auditors. The paper operationalizes SCAN for clinical curriculum design, supervision, and assessment, and opens an empirical research agenda grounded in cognitive science. This paradigm shift from misuse to misclassification is not semantic: it offers educators a clear perspective on what to look for, what to assess, and what to intervene on. 

---
# Real-Time Musculoskeletal Surrogates for Pediatric Cerebral Palsy: a Credibility Pilot 

**Authors**: Mohammad Arif Ul Alam  

**Link**: [PDF](https://arxiv.org/pdf/2608.28371)  

**Abstract**: Real-time musculoskeletal (MSK) surrogates could support personalized rehabilitation for children with cerebral palsy (CP), but their credibility depends on subject-wise evaluation, low inference latency, and calibrated uncertainty. We develop a subject-conditioned causal neural surrogate using OpenSim-derived static parameters, temporal joint kinematics, true muscle capacities, and training-only perturbations. On a real pediatric CP gait dataset comprising nine children, we use leave-one-subject-out validation on six development subjects and evaluate a frozen configuration once on three locked test subjects. The surrogate accurately reproduces musculotendon lengths (R-square = 0.92 in development validation and approximately 0.95 on locked subjects; nRMSE < 8%) while requiring only sub-millisecond to few-millisecond neural inference, well below a 100 ms interactive-rehabilitation target. In contrast, direct muscle-force estimation remains unstable at this small, heterogeneous scale: pooled metrics can overstate within-subject, per-muscle accuracy. A Monte Carlo credibility pilot further shows that propagating only +/-5% anthropometry and muscle-capacity variation produces severely overconfident nominal 90% intervals (approximately 4% force coverage and below 1% MT-length coverage). These results establish a leakage-free evaluation and credibility framework for pediatric MSK surrogates, while identifying force modeling and epistemic uncertainty as the central next challenges for clinically credible digital twins. 

---
# Optimal Adversarial Testing: Extracting Honest Test Results from Dishonest Test Takers 

**Authors**: Owen Cox, April Xu, Weiyu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.28362)  

**Abstract**: In applications, it is often required to test objects or people to determine their qualities in terms of certain metrics. However, besides being naturally noisy, the test results can be corrupted by adversarial behaviors of objects or people being tested (test takers). For example, dishonest test takers can cheat in the exams to distort the test results. With the development of AI technologies, such distortions driven by cheating using AI technologies are becoming more commonplace and severe. In this paper, we propose optimal testing strategies which can still recover needed test results even if there are cheaters polluting the results. The proposed testing strategies will optimally re-test selected group of test takers using different testing security measures. We determine the optimal testing strategies using a dynamic programming method. 

---
# Cross-Spectral Dense Correspondence for Multimodal Spectral Medical Imaging 

**Authors**: Eric L. Wisotzky, Jost Triller, Simon W. Härtl, Oliver T. Bruns, Peter Eisert, Anna Hilsmann  

**Link**: [PDF](https://arxiv.org/pdf/2608.28341)  

**Abstract**: Precise dense correspondence is a fundamental prerequisite for multimodal spectral imaging systems that fuse disparate wavelength ranges for subsequent analysis in medical and scientific imaging. Corresponding image points are often observed with non-overlapping spectral sensitivities, leading to wavelength-dependent contrast changes, intensity inversions, and appearance shifts for which dense ground truth is difficult to obtain and conventional RGB-based training data provides only limited supervision. We address this data gap by introducing a sensor-agnostic cross-spectral modulation protocol on established correspondence benchmarks with intensity input projection, and by proposing a synthetic cross-spectral correspondence benchmark simulating physically plausible radiometric differences. Evaluation on several modern dense correspondence backbones trained with our unified cross-spectral protocol showed substantial improvements under severe spectral mismatch while maintaining performance on standard RGB benchmarks. Ablation experiments show that view-dependent channel selection and nonlinear radiometric transformations provide complementary robustness, indicating that the primary limitation of existing models is not their structural matching capacity but the mismatch between training distribution and spectral characteristics of the target image pair. Qualitative evaluations on heterogeneous medical spectral acquisition systems demonstrate the practical relevance of the proposed training data augmentation protocol as an enabler for spatially coherent spectral fusion in HSI workflows. 

---
# BanglaMed-QA: A Question Answering System for Healthcare Support in Bangla 

**Authors**: Rowzatul Zannat, Abdullah Al Shafi, K. M. Azharul Hasan, Atia Shahnaz Ipa  

**Link**: [PDF](https://arxiv.org/pdf/2608.28329)  

**Abstract**: Medical question answering (QA) systems have become crucial tools for providing reliable health information. But they remain very unexplored for low-resource languages like Bangla due to limited datasets and systems tailored to these languages. To address this, we introduce BanglaMed-QA, a robust QA system specifically designed for the Bangla medical domain. The process begins with building a structured medical knowledge base that includes 4,493 QA pairs in 9 categories under 506 diseases. To improve semantic comprehension, domain-specific root word dictionaries and synonym sets are proposed, in addition to part-of-speech tagging for anaphora resolution. We adopt supervised machine learning models in which SVM is found to be the best model to categorize questions. Multiple similarity metrics, including cosine, Jaccard, BM25, and Levenshtein, are applied with soft and hard voting methods for query matching. The performance of the QA system has been evaluated in two aspects, with a 95% F1 score in an automated evaluation and an average human satisfaction rating of 0.9 out of 1.0. This validates the real-world application of BanglaMed-QA in closing the healthcare information gap for Bangla speakers. 

---
# Layered LLM Defenses as an Ensemble: Access Tiers, Inference Cost, and the Measured Failure Correlation Between Defense Layers 

**Authors**: Abrar Alotaibi, Muhammad Shahid Jabbar, Sadam Al-Azani, Moataz Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2608.28327)  

**Abstract**: Practitioners defend large language models (LLMs) by stacking defenses, assuming the layers compound. A stack is an ensemble, and ensembles compound only under a condition the LLM security literature recommends but never measures: the members must fail on different inputs.
Two instruments make that measurable. The Adversary Access-Tier Model (AATM) grades an adversary by the access it holds, from system-only (A0) to influence over training data (A4). A cost model sorts defenses into five classes of inference-time overhead; because two classes require training weights or reading activations, they tier the defender as AATM tiers the adversary. From these we derive how a stack behaves, and the quantities a defender cares about diverge: coverage saturates within a tier, cost rises by class, false refusals accumulate as a union, and residual attack success falls multiplicatively only under independence.
We measure that independence. Running one adaptive adversary against a seven-layer stack, failure correlation is positive in all fifteen measurable pairs ($\phi$ from $0.30$ to $0.75$), and the joint residual exceeds the multiplicative prediction by up to $0.172$. Stratifying on behavior difficulty dissolves most of the association, so the dependence is predominantly common-cause, but it survives permutation inference, majority-vote grader labels, and externally calibrated thresholds. The same stack refuses four in five benign prompts while remaining statistically indistinguishable from its strongest single layer.
The dependence is architectural rather than sampling-based: members correlate through the model they all wrap, so no wider member pool weakens it. Diversity therefore selects stack members but does not predict what an assembled stack delivers, which has to be measured end to end. 

---
# Deriving Scaling Laws for OpenEuroLLM Models: Learning Rate, Batch Size and Loss 

**Authors**: Niccolò Ajroldi, Diana Alexandra Onutu, Haider Al-Tahan, Jörg Franke, Sampo Pyysalo, Jenia Jitsev, Aaron Klein  

**Link**: [PDF](https://arxiv.org/pdf/2608.28308)  

**Abstract**: We study the scaling behavior of learning rate and batch size in pretraining dense large language models on English-prevalent corpora. Beyond scaling \textit{jointly optimal} learning rates and batch sizes, we investigate their \textit{marginal} evolution with model capacity and data scale and develop a model that captures these relationships. As we employ a Warmup-Stable-Decay learning rate schedule, we further investigate the gains from learning rate annealing over a broad range of hyperparameters settings, models and data budgets, and whether the optimal learning rate and batch size \textit{transfer} between the stable and decay phases. Finally, we characterize the dependence of loss on model capacity and dataset size, evaluating recently proposed scaling forms that explicitly model their interaction. We find these approaches particularly effective at capturing both undertraining and overtraining regimes across our experiments. This study establishes a first baseline and scaling procedure for the development of future OpenEuroLLM models. We open-source the complete collection of pretraining runs used in this study. 

---
# VISTA: Verifier-Informed Student-to-Teacher Adaptation for On-Policy Self-Distillation 

**Authors**: Zewen Ding, Zezhong Wu, Zhou Tao, Shida Wang, Shizhuo Hou, YongXiang Hua, Haoyu Cao, Linli Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.28306)  

**Abstract**: On-policy self-distillation (OPSD) improves reasoning by training a problem-only student on its own rollouts using dense token-level supervision from a privileged teacher that also sees a reference solution. However, standard OPSD treats the teacher distribution as a fixed target along the student's rollout and updates only the student %, although -- even though privileged conditioning does not guarantee that the teacher always provides the most appropriate target for problem-only reasoning. This one-way supervision can therefore misdirect the student when the teacher distribution is misaligned with valid student reasoning. We therefore introduce Verifier-Informed Student-to-Teacher Adaptation (VISTA), which preserves the standard OPSD student update while using outcome-verified rollouts to adapt the teacher toward the student distribution. Within each verified rollout, VISTA further restricts this adaptation to the top-$k$ positions with the largest teacher--student KL divergence. Notably, VISTA reuses the rollout and loss function from standard OPSD, introducing no additional sampling or separate reward objective. Across AIME24, AIME25, and HMMT25 with Qwen3 models at 1.7B, 4B, and 8B, VISTA achieves the highest Avg@12 at every scale, improving over OPSD by $0.6$, $0.7$, and $2.1$ points, respectively. These results demonstrate the value of student supervision from outcome-verified rollouts and highlight student-to-teacher adaptation as a promising direction for OPSD. 

---
# PanelShield: Verifiable Closed-Loop Safe Planning for Robotic Industrial Panel Operation 

**Authors**: Guipeng Xin, Jiahe Xu, Chenhui Wan, Jie Liu, Youmin Hu, Zhongxu Hu  

**Link**: [PDF](https://arxiv.org/pdf/2608.28305)  

**Abstract**: Industrial panel operation is knowledge-intensive and safety-critical. Beyond control recognition and action generation, execution must satisfy constraints in operation manuals and safety regulations. While foundation-model-based planners show strong semantic capability, they typically lack computable, localizable, and reproducible mechanisms for violation detection and repair. To address this, we propose PanelShield, a verifiable closed-loop safety planning framework for manual-guided industrial panel operation. The framework generates parameterized action primitive sequences from task-relevant manual evidence and applies dual formal verification with LTL and a Safety FSM to enforce cross-step temporal correctness and local transition legality. When violations occur, it outputs a structured counterexample with the earliest violating step and cause, enabling targeted repair and re-verification. We build a multi-level long-horizon planning benchmark covering three representative industrial device panels, and evaluate the framework in simulation and real-world robotic experiments. Results show that PanelShield improves complex safety-constrained task performance over foundation-model-only planning baselines while reducing the violation rate to 2.7%, with 4.1 s total latency. Real-world experiments demonstrate end-toend feasibility. Overall, PanelShield offers a verifiable approach to robotic panel operation that balances flexibility, safety, and auditability. 

---
# MaCoPlanner: LLM-Assisted Manual-Compiled Task Planning with Proactive Safety Verification for Robotic Industrial Panel Operation 

**Authors**: Guipeng Xin, Jiahe Xua, Mohammad Deghat, Chenhui Wan, Jie Liu, Youmin Hu, Zhongxu Hu  

**Link**: [PDF](https://arxiv.org/pdf/2608.28300)  

**Abstract**: Robotic industrial panel operation requires not only accurate control localization but also compliance with operating procedures, safety rules, and device-state constraints distributed across heterogeneous manuals. This study presents MaCoPlanner, a task-planning framework built on knowledge compiled from equipment manuals that converts equipment manuals into a typed intermediate representation, retrieves task- and state-relevant evidence, and uses it to support plan generation. Before actuation, candidate plans are symbolically rolled out and checked against procedural and state-transition constraints; detected violations are localized and returned for targeted repair, while unresolved plans are rejected. A separate execution interface grounds verified symbolic actions to physical controls and updates the device state. Under an independent evaluation oracle, MaCoPlanner achieves a final violation rate of 2.7%, and 26.3% of the runs in the repair analysis are rejected after exhausting the refinement budget. Compared with Raw-Manual, task success increases from 62.8% to 84.4% on Level-2 tasks and from 25.9% to 43.2% on Level-3 tasks. Experiments on a controller-panel simulator without an attached industrial load further demonstrate integrated execution feasibility under representative interaction conditions, without claiming industrial deployment readiness. 

---
# A Probabilistic Interpretation of KV Cache Eviction 

**Authors**: Renato Geh, Alex Chen, Daniel Israel, Aditya Grover, Guy Van den Broeck  

**Link**: [PDF](https://arxiv.org/pdf/2608.28293)  

**Abstract**: The premise and promise of KV (cache) eviction is simple: higher throughput can be achieved by evicting some entries from the KV cache, at a negligible cost to quality. This holds empirically for many existing methods, though most rely on creative heuristics for selecting which entries to drop. Despite recent advances, the problem of KV eviction has remained informal in the literature. This paper aims to properly formalize this problem through the lens of probabilistic reasoning and reveal what can be learned from this perspective. Concretely, we (1) formalize the problem of KV eviction and, unfortunately, prove that it is computationally hard, (2) show that by framing it probabilistically, KV eviction reduces to the problem of expectation estimation, which can be approximated through sampling, (3) show that through this probabilistic interpretation, correcting for evicted entries during decoding---a previously ignored problem---becomes feasible, and (4) reveal that existing methods in the literature are zero-variance biased estimators that can be easily adapted in order to enable decode time correction. In practice, we show that this probabilistic version of KV eviction coupled with decode time correction is more robust to different tasks compared to existing eviction methods and achieves competitive performance at the same compression budget. 

---
# Embedding Models for Stance-Aware Argument Retrieval 

**Authors**: Angelo Sparacino, Francesca Toni, Adam Dejl  

**Link**: [PDF](https://arxiv.org/pdf/2608.28283)  

**Abstract**: In computational argumentation, obtaining arguments that explicitly support or attack given claims is a critical precursor to downstream reasoning tasks. When these supporting and attacking arguments are to be retrieved using semantic search methods, they need to be assessed for topic-relevance to the claims of interest as well as for correctness of their (positive or negative) stance towards the claims. In this paper we explore how dense embedding models (hereafter, models), powering modern retrieval pipelines, can serve as the basis of semantic search incorporating this dual assessment. We show experimentally that existing models struggle with asymmetric reasoning, exhibiting a strong bias toward topical overlap while ignoring instructional stance. We also show that correcting this bias via contrastive training triggers a new failure mode where models over-correct, over-fixating on polarity keywords (e.g., "supports" or "refutes") at the expense of the semantic topic. We thus introduce diagnostic word-ablation metrics to quantify this phenomenon and propose a data-centric solution. By implementing a balanced argument curriculum alongside LLM-augmented, stance-inverted arguments, we force the (embedding) models to learn deeper directional logic rather than exploiting superficial lexical shortcuts. Our evaluation demonstrates that, for sufficiently powerful models, this approach can alleviate the observed overcorrection, achieving further improvements in stance-aware argument retrieval. 

---
# Spatial-Semantic Reasoning using Large Language Models for Efficient UAV Search Operations 

**Authors**: Marin Maletic, Marijana Peti, Tamara Petrovic, Stjepan Bogdan  

**Link**: [PDF](https://arxiv.org/pdf/2608.28270)  

**Abstract**: We present a real-time semantic navigation framework for Unmanned Aerial Vehicles (UAVs) focused on improving time efficiency in the Object Goal Navigation (ObjectNav) task. Central to our approach is a Large Language Model (LLM) that interprets user-provided natural language instructions and performs semantic reasoning over detected objects and spatial context to prioritize high-probability search regions. The system combines real-time object detection, 3D spatial mapping, and polynomial spline interpolation for smooth and feasible UAV trajectory planning. Unlike prior methods that rely on offline reasoning or simulator-constrained action spaces, our framework can operate in real time, continuously updating semantic relevance based on new observations. Experiments in both simulated and real-world settings demonstrate reductions in mission duration while maintaining high search accuracy, underscoring the effectiveness of LLM-guided reasoning for time- efficient UAV-based ObjectNav. 

---
# A comprehensive and trustworthy benchmark of AI methods for change detection in Earth observation 

**Authors**: Tadej Tomanič, Alice Baudhuin, Jan Sotošek, Jure Brence, Panče Panov, Nikola Simidjievski, Dragi Kocev  

**Link**: [PDF](https://arxiv.org/pdf/2608.28247)  

**Abstract**: Change detection in Earth observation (EO) is critical for monitoring land surface transformations, yet recent research in the field is constrained by inconsistent evaluation protocols and a narrow focus on predictive accuracy without regard for computational efficiency. To address this, we present a standardized, open-source benchmark for evaluating state-of-the-art (SOTA) deep learning methods for Earth observation change detection. We conduct a comprehensive analysis of ten representative model architectures, ranging from convolutional networks (CNNs) to vision transformers (ViTs), across ten heterogeneous change detection datasets. We rigorously evaluate these models with identical experimental protocols, comparing models trained from scratch against those utilizing pre-trained weights. Furthermore, we evaluate predictive performance alongside computational efficiency, including parameter counts and inference latency. Our findings reveal that well-optimized classical architectures, such as Siamese U-Nets, frequently outperform more complex contemporary models when computational efficiency is factored in, and that pre-training consistently provides a significant performance boost with no additional inference cost. To ensure complete transparency and reproducibility, all experimental resources, including standardized data splits, training scripts, training logs, and model checkpoints are publicly available and adhere to FAIR principles (Findable, Accessible, Interoperable, and Reusable). 

---
# Training-free Suction Grasp Detection for Deformed Aseptic Cartons Using Vision-Language Models and Geometric Surface Scoring 

**Authors**: Marin Maletic, Goran Vasiljevic  

**Link**: [PDF](https://arxiv.org/pdf/2608.28246)  

**Abstract**: Robotic sorting of recyclable waste is challenging due to the deformable and geometrically inconsistent nature of target objects. We present a training-free suction grasping system for sorting deformed aseptic beverage cartons, decoupling target identification from grasp-point selection. An open-vocabulary vision-language model detects cartons from a text prompt, SAM2 refines each detection into an instance mask, and a geometric scoring method selects the suction point by combining surface flatness with normal alignment. Three geometric methods are compared: k-nearest-neighbour PCA, Sobel cross-product, and RANSAC plane fitting. Evaluated on a real robot across three deformation levels and 35 cluttered scenes, single-object grasp success reaches 88.2% and end-to-end retrieval in clutter is 72.6%. 

---
# Performative Privacy: When Differential Privacy Maximizes Utility 

**Authors**: Uddalak Mukherjee, Edwige Cyffers, Yann Chevaleyre  

**Link**: [PDF](https://arxiv.org/pdf/2608.28198)  

**Abstract**: Privacy-preserving learning is often motivated by the idea that protecting users' data can preserve trust and thus participation, improving utility in the long term. However, this claim has not been formalized so far. In parallel, performative learning provides a framework for studying learning systems whose deployment affects the data they later observe. In this work, we bring these two perspectives together and introduce \emph{performative privacy}, where data leakage reduces future participation. We study a simple model where agents repeatedly contribute data for mean estimation but may leave the system when their data is leaked. Privacy is implemented through differentially private mechanisms, creating a trade-off between estimation noise and future participation. We show, through a theoretical study of the dynamics and numerical experiments, that a finite privacy budget can outperform non-private estimation in the long term when the feedback loop between leakage and participation is sufficiently strong. This provides first evidence that differential privacy can be optimal not only as a protection mechanism, but also from the perspective of long-term utility. 

---
# Beyond Flat Netlist: Hierarchical Graph Representation Learning for Scalable Analysis of Sequential Circuits 

**Authors**: Jingyi Zhou, Zhengyuan Shi, Jiaying Zhu, Ziyang Zheng, Qiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.28188)  

**Abstract**: Circuit Representation Learning (CRL) offers a powerful paradigm to guide and optimize core Electronic Design Automation (EDA) tasks, but its practical adoption is hindered by the immense scale of industrial netlists and a failure to explicitly model register-level temporal dynamics. To overcome these barriers, we introduce DeepSeq3, a novel hierarchical framework that abstracts circuits into a two-level representation: fine-grained combinational subgraphs partitioned by flip-flops (FFs), and a high-level Super-Node Graph (SNG) that models the register-transfer structure. A dual Graph Neural Network (GNN) architecture learns representations at both levels, capturing local Boolean logic and global state transitions. Crucially, we introduce a state-centric pre-training scheme that predicts the reachability between FF states, endowing the model with a deep understanding of temporal behavior. Demonstrated on large-scale benchmarks, DeepSeq3's approach yields superior scalability and richer representations, reducing bounded model checking (BMC) solving time by 18% while guaranteeing correctness. 

---
# Conformal Risk-Averse Decision Making with Optimized Certainty Equivalent Risk Control 

**Authors**: Amirmohammad Farzaneh, Osvaldo Simeone  

**Link**: [PDF](https://arxiv.org/pdf/2608.28179)  

**Abstract**: We study risk-averse decision making, in which an agent selects actions while being uncertain about the true system state. The risk is measured via optimized certainty equivalent (OCE) metrics, which generalize popular criteria such as mean-variance risk and conditional value-at-risk (CVaR). We characterize the optimal policy under known distributions, and show that it reduces to a prediction set-based solution for the CVaR. This provides an operational interpretation of conformal prediction-type prediction sets. For unknown distributions, we develop a data-driven calibration strategy, based on a synthetic model for the likelihood and held-out calibration data, yielding high-probability control of the OCE risk. The approach is evaluated on two wireless beamforming settings. 

---
# Text Restoration of Ancient Documents with Language Models 

**Authors**: Shibingfeng Zhang, Edoardo Caraffa, Annafelicia Zuffrano, Maddalena Modesti, Giovanni Colavizza  

**Link**: [PDF](https://arxiv.org/pdf/2608.28170)  

**Abstract**: Purpose - This study investigates the feasibility of restoring missing text caused by physical lacunae in damaged ancient manuscripts using language models.
Methodology - The study proposes different scenarios to replicate real-world conditions. Language models of different architectures are applied according to their suitability to each scenario. We also propose several decoding strategies that further enhance performance and address the discrepancy between lacuna boundaries and the models' tokenization schemes.
Findings - The results reveal that text restoration of these documents cannot be fully automated, but it can serve as a useful tool to assist paleographers in their work. Model performance varies greatly depending on which structural part of the document needs to be restored and whether the character length of missing text is available.
Originality - This is the first study and to analyze model performance on formulaic and non-formulaic content and the impact of lacuna length awareness in manuscript restoration. Both are recurring challenges in paleographers' manual restoration work. Through systematic comparison and both qualitative and quantitative analysis of different models' performance under varying settings, this study offers a guideline for developing assistive tools to support paleographers. 

---
# Gen-TAS: A Generative AI-Aided Hardware-Software Task Allocation Framework for FPGA-GPP Heterogeneous Systems 

**Authors**: Mary Kong, Yuqin Zhao, Semih Vazgecen, Cristian Sestito, Themis Prodromakis  

**Link**: [PDF](https://arxiv.org/pdf/2608.28160)  

**Abstract**: FPGA-GPP heterogeneous systems combine software flexibility with the performance and energy efficiency of reconfigurable hardware. However, determining which application tasks should execute on the GPP or FPGA requires extensive expertise and design-space exploration, particularly when user objectives vary across latency, communication, resource utilisation, and power. This paper proposes Gen-TAS, a knowledge-grounded LLM framework for user-specific FPGA-GPP task allocation. By combining task-graph analysis with RAG, Gen-TAS grounds LLM reasoning in historical implementation knowledge and generates multiple explainable strategies tailored to the specified objectives. Human-in-the-loop selection and a deterministic backend connect LLM-generated decisions to reproducible FPGA SoC implementations. Experiments on CNN and SDR workloads across multiple LLMs demonstrate stable, requirement-driven allocation. Under latency-oriented objectives, implementations following the selected strategies achieve speedups of up to 2.45$\times$ and 92.53$\times$, respectively, relative to the corresponding all-GPP baselines while other objectives select strategies that trade some acceleration performance for FPGA-GPP communication, resource utilisation, or FPGA power. 

---
# Nested Byte-Level Vocabularies Are Cheap to Deploy and Expensive to Share: A Pre-Registered Negative Result 

**Authors**: Christos Koutsiaris  

**Link**: [PDF](https://arxiv.org/pdf/2608.28151)  

**Abstract**: A byte-level BPE tokenizer is an ordered list of merge rules, so applying only a prefix yields a vocabulary whose token identifiers are the first rows of the full vocabulary. This prefix nesting allows one language model to operate at several vocabulary sizes, use a control token to indicate the active size, and be deployed at any trained size by slicing its embedding and output head. We pre-registered five claims, including margins, seeds, contrasts, and a stop rule, and trained 30 models with 3.1M- and 10.6M-parameter bodies on 200M tokens each. Slicing is numerically exact: across 76 checks, a sliced model reproduces the restricted full model's logits bit for bit and removes 66% of deployed weights without changing latency. However, the shared model trails a fixed-cap specialist by 3.64% bits per byte at 32k against a 1% margin, and by 2.96% at 8k against a 2% margin. A 2x2 ablation separating the control token from output restriction finds that the token changes performance by +0.07% to +0.13%, with all intervals crossing zero, while output restriction costs +0.47% to +1.19%; the factors are substitutes rather than complements. Multi-cap training nevertheless improves robustness: under typographical noise, the same checkpoint degrades 12.5--15.4 points less in its fine mode and outperforms each fixed-cap specialist at that specialist's vocabulary size. A control with neither cap token nor output restriction is equally robust, attributing this benefit to multi-granularity training rather than conditioning. The per-cap penalty tracks each cap's share of training rows, yielding a falsifiable prediction for future work. 

---
# The Approximation Rank of Softmax Attention: Sharp Geometric Laws and Robust Interaction Dimension 

**Authors**: Yuhe Sui, Jianing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.28150)  

**Abstract**: Which geometry controls the rank complexity of normalized softmax attention? We study maximum-row-$\ell_1$ approximation rank, exactly the least unrestricted rank preserving every bounded vector-valued output. Two sharp worst-case laws isolate support geometry: for fixed $d$ and error $\varepsilon$, spherical self-attention has rank $\Theta_{d,\varepsilon}(\min\{n,(1+\beta)^{(d-1)/2}\})$, while full-ball geometry adds one radial degree and, for $\beta\ge\beta_0(d,\varepsilon)$ and $n\ge C_d e^{\beta/8}$, gives $\Theta_{d,\varepsilon}(\beta^{d/2})$. For a fixed head, row-softmax quotients out row-scalar logit directions: the remaining visible query--key interaction dimension $r$ yields an $r/2$ per-instance upper law, and bounded constructions show this exponent is minimax sharp. Approximate interaction subspaces incur an explicit residual output error and yield a tolerance-indexed SVD dimension. On an 84-head BERT-base calibration set, we observe modest effective-dimension reductions across many head--temperature settings, together with positive associations with finite constructive rank upper certificates. Together, these results separate support geometry, which sets worst-case temperature scaling, from softmax-visible interaction geometry, which controls per-head approximation complexity. 

---
# Post-Edit Re-Verification in Simulator-Backed Engineering Agents: A Controlled Comparison of Verification-Cadence Guidance 

**Authors**: Qingchuan Zhu, Shuyue Tong, Pengju Ren  

**Link**: [PDF](https://arxiv.org/pdf/2608.28147)  

**Abstract**: Engineering agents that interact with external simulators may need to coordinate design modification with reacquisition of engineering evidence for the modified state. We ask whether first post-edit re-verification changes when explicit verification-cadence guidance is retained versus omitted while verification-relevant state/facts are held constant. Cadence-Guided (CG) retained an instruction to request a new simulation after a substantive modification, whereas Cadence-Omitted (CO) removed that instruction; neither condition used a hard gate. The study therefore measures instruction-conditioned post-edit verification-policy adherence rather than spontaneous recognition that prior evidence has become stale. Using DWSIM as the simulator backend and continuous valve-pressure adjustment, five Alibaba/Qwen models were evaluated on eight synthetic cases; each model-case-condition combination was executed three times via live API calls, yielding 120 evaluation slots per condition. Re-verification was observed in 94/120 CG slots versus 32/120 CO slots; cadence violations occurred in 26/120 versus 87/120; and bounded final success was reached in 95/120 versus 35/120. qwen3.5-35b-a3b showed minimal re-verification (1/24 in CG and 0/24 in CO) and no final success in either condition. Within this bounded protocol, explicit post-edit verification-cadence guidance was associated with more re-verification, fewer cadence violations, and more frequent bounded final success, supporting the treatment of verification cadence as an explicit interaction-protocol component. 

---
# CheXtriev: Anatomy-Centered Representation for Case-Based Retrieval of Chest Radiographs 

**Authors**: Naren Akash, Arihanth Tadanki, Jayanthi Sivaswamy  

**Link**: [PDF](https://arxiv.org/pdf/2608.28137)  

**Abstract**: We present CheXtriev, a graph-based, anatomy-aware framework for chest radiograph retrieval. Unlike prior methods focussed on global features, our method leverages graph transformers to extract informative features from specific anatomical regions. Furthermore, it captures spatial context and the interplay between anatomical location and findings. This contextualization, grounded in evidence-based anatomy, results in a richer anatomy-aware representation and leads to more accurate, effective and efficient retrieval, particularly for less prevalent findings. CheXtriv outperforms state-of-the-art global and local approaches by 18% to 26% in retrieval accuracy and 11% to 23% in ranking quality. The code is available at this https URL. 

---
# VICT: Verifier-Instrumented Credit Tracing for Long-Horizon LLM Agent Reinforcement Learning 

**Authors**: Pengcheng Li, Zhengyang Zhang, Dongxu Zhang, Sui Huang, Shaohua Ma  

**Link**: [PDF](https://arxiv.org/pdf/2608.28128)  

**Abstract**: Fine-grained credit assignment is a central challenge in reinforcement learning for long horizon LLM agents. Standard objectives often train from programmatically verifiable terminal rewards by broadcasting each sparse outcome to every action in a trajectory. Existing methods typically seek finer credit from the rollout side, constructing auxiliary trajectory signals or additional comparisons to estimate action importance. Although useful, these approaches still treat the verifier that judged success as a scalar reward, discarding its internal task structure. Our key insight is that many verifiable tasks already encode the relevant checks inside their terminal verifier. We propose VICT (VerifierInstrumented Credit Tracing), a training-time interface that exposes executable or evidence backed atoms and traces them back to actions through dependency-valid proof edges. VICT redistributes group-relative advantage only along those edges, shifting credit assignment from rollout-side inference to verifierside tracing. It preserves the original terminal reward, abstains when evidence is incomplete or ambiguous, and changes only the training-time advantage tensor, requiring no learned critic, process labels, branch rollouts, or inference-time verifier access. On ALFWorld and WebShop, VICT improves substantially over outcome-only training and achieves strong performance alongside recent fine-grained credit methods; ablations rule out dense atom rewards, final-commit credit, temporal proximity, and sparsity as sufficient explanations. 

---
# Do Medical Vision Models Reason About Anatomy? Probing the Spatial Inductive Biases of Learned Visual Representations 

**Authors**: Naren Akash, Neeraja Ramanan  

**Link**: [PDF](https://arxiv.org/pdf/2608.28092)  

**Abstract**: Interpreting a CT scan means comparing structures on either side, judging how far apart organs sit, and knowing where each one belongs. Medical vision encoders are evaluated on diagnostic accuracy, or through assembled multimodal systems where a failure is hard to attribute, so it remains unclear whether their representations support any of this. We construct SPAR-Bench, eight probes over multi-organ abdominal CT that separate coordinate localization, relational reasoning, and spatial queries, and apply them to five architectural configurations and three medical foundation models, frozen and finetuned. Probes that ask for a comparison within the slice stay at chance, and neither pretraining scale, finetuning, nor architecture closes the gap. Probes that appear solved in domain fall to chance under zero-shot transfer, indicating that their accuracy reflects recall of canonical anatomy rather than computation over the image. Reading the same frozen features with a pooled head rather than the full set of tokens moves relational recovery from 0.7% to 67.8%, so pooled probing understates what a representation holds. Questions the encoders answer well are answered at chance by four open-weight MLLMs. Our results suggest these encoders carry a map of where organs usually lie, and little of the machinery for comparing structures within a particular patient. Code and data will be available at this https URL. 

---
# VersaGauss: A Versatile Framework for Generating Multiphase Dynamics with 3D Gaussians 

**Authors**: Ruijie Su, Lingxiao Yang, Xiaohua Xie, Jianhuang Lai  

**Link**: [PDF](https://arxiv.org/pdf/2608.28069)  

**Abstract**: Recent progress has been made in 3D Gaussian representation for reconstruction, generation, and physical simulation. However, current approaches mainly concentrate on physics-based dynamic generation of solid objects and only handle single-phase collision interactions. We introduce VersaGauss, a unified framework for generation, simulation, and rendering that supports versatile physics-based dynamic generation, particularly for multiphase interactions. Our system takes a few images as input and produces a realistic, physics-driven 3D dynamic scene with multiple objects. To optimize the Gaussian kernel distribution, we develop a particle pruning algorithm. We also propose the Coupled Multiphase Point Method (CMPM) to effectively model and generate multiphase interactions. Additionally, harmonic interpolation within CMPM and a Gaussian evolution strategy are introduced to achieve realistic fluid rendering. Extensive experiments demonstrate that our framework can simulate interactions among various materials such as fluid, rubber, sand, snow, and others. Code is available at this https URL. 

---
# Dynamic Alignment Compensation for Hallucination Mitigation in Large Vision-Language Models 

**Authors**: Kairong Yu, Zixin Zhu, Le Yu, Hongwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.28058)  

**Abstract**: Large Vision-Language Models (LVLMs) remain prone to hallucinations, producing responses that are irrelevant or inconsistent with the multimodal input. Existing mitigation methods mainly rely on external supervision, output calibration, or attention regulation, leaving the internal representation dynamics of autoregressive generation underexplored. We identify an inference-time failure mode in which cross-modal representations degrade across decoder layers and drift across generation steps, destabilizing token prediction and increasing hallucination risk. We propose \emph{Dynamic Alignment Compensation} (DAC), a training-free inference-time method that detects representation divergence and selectively applies lightweight residual compensation. DAC combines Layer-wise Semantic Compensation to mitigate inter-layer degradation with Sequential Semantic Correction to constrain temporal drift. Experiments on nine hallucination-focused and general-purpose multimodal benchmarks across multiple LVLM backbones show that DAC consistently reduces hallucinations while maintaining strong overall performance. 

---
# Explainable Uncertainty Estimation for Reliable Medical AI 

**Authors**: Li Rong Wang, Jamie Duell, Xinran Xu, Thomas C. Henderson, Yu Yue Hew, Pik Wan Erica Chiang, Xiao Wei Alstar Ang, Bingwen Eugene Fan, Xiuyi Fan  

**Link**: [PDF](https://arxiv.org/pdf/2608.28052)  

**Abstract**: Artificial intelligence has strong potential to support clinical decision-making, yet its adoption in healthcare remains limited due to a lack of trust. Uncertainty estimation can signal unreliable predictions, and explainable AI (XAI) can clarify how predictions are made but existing methods treat them separately, providing no feature-level insight into why a prediction is uncertain or which tests to prioritize to reduce it. To address this gap, we propose explainable uncertainty estimation, which unifies uncertainty estimation and XAI to both quantify uncertainty and explain feature-level contributions. We introduce the Expected Gradients Reconstruction Uncertainty Estimate (egRUE), which incorporates prediction explanations into its uncertainty computation and decomposes uncertainty into feature-wise contributions. We prove theoretical properties of egRUE and show through experiments that it improves reliability and interpretability compared to existing methods. A user study with medical experts further demonstrates that egRUE's explanations improve calibrated trust over uncertainty scores alone, increasing confidence in correct predictions and reducing confidence in incorrect ones. By combining prediction uncertainty with feature-level explanations, egRUE strengthens decision-making support in safety-critical healthcare settings, clarifying both when predictions may be unreliable and which features drive that uncertainty. 

---
# SimpCue: Cue-Based Prompting for Multilingual Text Simplification 

**Authors**: Mehrzad Tareh, Horacio Saggion, Stefan Bott  

**Link**: [PDF](https://arxiv.org/pdf/2608.28042)  

**Abstract**: Text simplification aims to make complex texts easier to understand while preserving their original meaning. Recent large language models can perform simplification through prompting, but it remains unclear whether adding explicit linguistic information about sentence complexity to the prompt improves their outputs. We investigate this question for multilingual sentence-level Easy-to-Read simplification in Catalan, Spanish, and Italian. Using Qwen3-8B, we compare a baseline prompt, a gold-cue prompt enriched with gold linguistic cues, and a predicted-cue prompt enriched with automatically predicted cues. We evaluate the outputs using SARI, BLEU, chrF, and BERTScore, and complement this evaluation with a manual qualitative analysis. Predicted-cue prompting obtains the best overall scores across all four metrics, although the gains over the baseline are small. Gold-cue prompting does not consistently improve over the baseline, and results vary across languages. These findings indicate that cue-based prompting can influence multilingual Easy-to-Read simplification, but its benefits are modest, metric-dependent, and language-dependent. 

---
# Compared to What? A Human-Anchored Security Benchmark for LLM-Generated Infrastructure-as-Code 

**Authors**: Animesh Shaw  

**Link**: [PDF](https://arxiv.org/pdf/2608.28021)  

**Abstract**: Large language models are increasingly used to author Infrastructure-as-Code (IaC), where a single insecure default can be deployed directly into production. Prior evaluations report raw vulnerability counts for model-generated IaC, but without a human baseline they cannot determine whether models are actually worse than engineers. We introduce GenIaC-SecBench, a benchmark of 100 deployment scenarios stratified by architectural complexity, evaluated across 12 model configurations from four vendors, producing 1,196 IaC artifacts scanned by three independent policy engines (Checkov, Trivy, KICS). Critically, we also scan 634 human-authored IaC templates with the same toolchain, providing the first size-matched human security baseline.
Vulnerability density is strongly inverse to artifact size (Spearman $\rho = -0.55$, $p < 10^{-77}$), meaning unmatched comparisons measure size rather than security. When matched on declared-resource count, all model configurations fall within 3.21x--3.87x the human vulnerability density, with the gap widening for simpler tasks (4.9x at one resource, 1.4x at twenty or more).
We decompose reasoning into standard generation, prompt-engineered chain-of-thought, and vendor extended-thinking APIs. Vendor extended thinking significantly outperforms prompted chain-of-thought ($-12.0\%$, $p = 0.0013$), while prompted chain-of-thought is indistinguishable from standard generation ($-1.3\%$, n.s.). Token instrumentation shows extended thinking uses under 1\% of the output budget, explaining its bounded effect.
Two negative results also emerge: deployability does not correlate with vulnerability ($r = 0.158$, $p = 0.625$), and classical complete-case Friedman testing is infeasible for realistic benchmark designs, motivating the Skillings-Mack statistic. All code, data, and regeneration scripts are released. 

---
# Twin Worlds: Equivariance-Based Abstention for Evidence-Grounded Reasoning 

**Authors**: Vy Nguyen, Ziqi Xu, Jeffrey Chan, Estrid He, Feng Xia, Renqiang Luo, Erik Cambria, Xiuzhen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.28018)  

**Abstract**: Knowledge-intensive reasoning requires Large Language Models (LLMs) to ground answers in provided evidence. When evidence is insufficient, it is desirable that models abstain rather than confidently generating unsupported answers. Existing abstention methods rely on uncertainty estimation or evidence sufficiency checks, but neither tests whether the reasoning process for generation, driven by the interaction of provided evidence and the model's internal memory parameters, is actually grounded in the evidence. A key contributing factor is that entity mentions in context activate memorised associations, causing models to generate plausible responses ungrounded in evidence. We propose Twin Worlds (TW), a framework for improving reliability in knowledge-intensive reasoning through equivariance-based abstention: unlike invariance, which requires outputs to remain unchanged, equivariance requires outputs to transform correspondingly under entity substitutions. A model grounded in the evidence should produce answers that shift consistently when entities are substituted while their relations are preserved. TW constructs multiple worlds via typed substitutions of the original input that preserve relational structure while reducing parametric priors, and uses equivariance violations as an abstention signal. Across four benchmarks and three model backbones, TW identifies when answers are not reliably grounded in the provided evidence and outperforms uncertainty- and sufficiency-based baselines. 

---
# When Can Conditional Flow Matching Replace Pointwise Negative Log-Likelihood? 

**Authors**: Yansen Han, Hongxin Sun, Tao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2608.28010)  

**Abstract**: Flow matching enables likelihood-free training, yet alignment methods increasingly reuse conditional flow matching (CFM) losses as endpoint negative log-likelihoods (NLLs) and their old/new differences as log-likelihood ratios. We characterize when these substitutions are valid. For linear Gaussian paths, we exactly decompose endpoint NLL into entropy, a weighted CFM objective, an interior velocity--score residual, and a boundary residual. Thus CFM-only estimates and differences are exact only when the corresponding residuals cancel. At the off-policy population optimum, ordinary CFM is not generally a pointwise NLL estimator, whereas \(w_{\mathrm{sc}}(t)=(1-t)/t\) removes the interior residual; this positive result does not extend generally to training or on-policy alignment. On-policy log-ratios can remain biased even for identical endpoint laws or after surrogate optimization. Experiments across dimensions, distributions, and geometries support these conclusions and the mechanisms that make inexact ratios useful. **More broadly, the decomposition provides a theoretical basis for adapting likelihood-based LLM methods to flow matching, while distinguishing exact substitutions from controlled surrogates.** 

---
# A Method for Layer Bit-Width Allocation in LLM Quantization via Performance Maximization Under a Quality-Degradation Constraint 

**Authors**: Artem Safronov  

**Link**: [PDF](https://arxiv.org/pdf/2608.28003)  

**Abstract**: This paper proposes a layer bit allocation method for Gemma-3-1B, formulating the problem as performance maximization (latency decrease) given a degradation budget constraint (allowable level of generation quality loss). This approach is different from time- and resource-consuming uniform layer quantization methods that are used in the literature (like GPTQ or AWQ) or allocation methods without proven performance-accelerating effect (like MixLLM or TorchAO). The layer sensitivity profile resulting from our prior work SA-PTQ is applied using the activation pass-through mode inside TensorRT-LLM. For each layer precision is determined individually in blocks, according to a grouping introduced in the prior step (5+5, 10+10, all26), differentiating the contribution of FFN, Attention, and lm_head to the overall speedup. The clock speed was measured for 13 W8A8 variants on an RTX 5090. We find that for FFN and lm_head the time cost of quantization/dequantization is compensated for by the use of integer arithmetic, while for short context lengths, the opposite holds true for Attention: an additional step of quantization slows execution down. We propose a manual implementation of SmoothQuant for TensorRT-LLM which was necessary due to export failures, unavailable for lm_head. The best solution found under joint consideration of all three criteria with minimal degradation was FFN 5+5 with lm_head, providing an 11.0% reduction in latency with negligible quality loss (98.90% Top-1 agreement, +0.85% perplexity degradation). With acceptable quality loss for FFN all26 + lm_head, a speedup up to 19.1% was found possible. We suggest further optimizations: fused attention kernels in INT8, KV-cache quantization, using FP8 instead of INT8 and partial Attention quantization analogous to FFN. 

---
# CAITLYN: Can LLM Agents Autonomously Synthesize Defenses against Emerging Injection Attacks? 

**Authors**: Zi Liang, Xiaoyu Xu, Yanyun Wang, Minxin Du, Qingqing Ye, Haibo Hu  

**Link**: [PDF](https://arxiv.org/pdf/2608.27990)  

**Abstract**: Prompt injection attacks on Large Language Model (LLM) agents seek to introduce malicious instructions or content into external text sources retrieved by agents, forcing the underlying LLMs to execute harmful actions outside their benign scope. While current defenses effectively counter known injection attacks, deploying them in LLM agent environments remains challenging due to attack variants and emerging threats. Moreover, existing solutions typically suffer from an inherent trilemma, i.e., a constant trade-off among runtime efficiency, contextual precision, and adaptability. To bridge this gap, we propose Continuous Agents for Injection Threats via Lifelong Yielding Nexus (CAITLYN), an agent-agnostic defense middleware. CAITLYN integrates two systems. System I focuses on immediate defense against existing attacks using a two-tiered library: Tier-0 for rule-based detection scripts and Tier-1 for optimized LLM-based accurate inference. System II, in contrast, is deployed to monitor potential abnormal signals and attempt to synthesize new defenses. On standard benchmarks, CAITLYN matches the detection performance of state-of-the-art defenses at lower token overhead than LLM-as-a-judge baselines. On Emerging, our new delivery-aware benchmark featuring novel injection techniques, static baselines and the standalone System I configuration remain vulnerable. In contrast, System II autonomously synthesizes verified defense capabilities, substantially lowering the attack success rate across three diverse agent environments. 

---
# Not to Break, but to Attest: Adversarial Probes for Privacy-Preserving LLM Verification 

**Authors**: Cameron Wilding, Mina Shaker, Fatemeh Ganji  

**Link**: [PDF](https://arxiv.org/pdf/2608.27954)  

**Abstract**: Post-deployment changes to large language models can alter behavior while leaving routine outputs largely unchanged, creating a challenge for AI governance when model weights are proprietary. We present a privacy-preserving zk-SNARK-based audit framework that searches for probes designed in the spirit of adversarial examples to amplify logit drift between an approved model and a modified deployment. Our framework explores complementary probe families under different access models. Token-based probes operate in a black-box setting and require only the input interface, tokenizer, and vocabulary. Embedding-based probes require gray-box access to the embedding interface. Stress probes rely on additional interface capabilities but do not require full white-box access to model weights or architecture. This range allows probe selection to balance sensitivity, access requirements, and deployment cost. We evaluate probe constructions across LLM architectures, model-tampering scenarios representative of post-deployment attacks, and GPU platforms. Importantly, our experimental results demonstrate that token-based probes consistently deliver the strongest mean sensitivity across models and GPU platforms, although operating in a black-box setting. Our Groth16 zk-SNARK workflow remains practical as the probe set scales from 1 to 50, where proving time increases from 1.02 to 1.78 seconds, verification remains near 0.84 seconds, and proof size remains constant. 

---
# Antipatterns in AI-assisted Qualitative Data Analysis: A Catalog of Temptations and Pitfalls for Software Engineering Researchers 

**Authors**: Rashina Hoda, Carolyn Seaman, Victoria Gomes, Rodrigo Spinola  

**Link**: [PDF](https://arxiv.org/pdf/2608.27927)  

**Abstract**: AI-assisted qualitative data analysis (QDA) offers unprecedented opportunities to streamline software engineering (SE) research, yet uncritical use risks compromising analytical rigor and flooding the field with accelerated production of low-quality research. While tactical best practices will naturally evolve over time, SE researchers currently lack strategic guidance to identify and mitigate methodological risks when attempting AI-assisted QDA. Based on our decades of qualitative SE research expertise and experience combined with an understanding of the emerging landscape of AI-assisted QDA, this paper presents a catalog of antipatterns in AI-assisted QDA - a set of assumptions and practices that initially appear advantageous but ultimately undermine analytical rigor and validity. The antipatterns are grouped into three categories reflecting escalating impact: Dangerous Drivers, Operational Missteps, and Analytical Failures. As more SE researchers attempt AI-assisted QDA, these antipatterns will help them identify and avoid common temptations and pitfalls, while reviewers can be equipped with the vocabulary and criteria to call out problematic and failed practice. Ultimately, this catalog of antipatterns can serve as a stepping stone in our responsible methodological evolution toward principled and meaningful human-AI collaboration in qualitative research. 

---
# PCBnet: A Dataset and Automatic Construction of SPICE Netlists from Schematic Images 

**Authors**: Zhen Huang, Yuhao Gao, Yuzhi Liu, Daian Cheng, Chengyuan Shao, Yucheng Chen, Yongjian Jia, Futing Zhang, Yichen Shi, Wenhao Wang, Zuyan He, Yangbo Wei, Zhanfei Chen, Jinlong Yan, Yu Zhang, Haoying Wu, Ting-Jung Lin, Lei He  

**Link**: [PDF](https://arxiv.org/pdf/2608.27923)  

**Abstract**: Printed circuit boards (PCBs) are fundamental to modern electronic systems, yet AI-driven PCB design automation remains constrained by the lack of large-scale paired schematic-netlist datasets. PCB schematics are particularly challenging due to diverse component types, complex wiring topologies, and noisy textual annotations. To address this gap, we present PCBnet, a large-scale PCB schematic dataset comprising over 300 real-world designs with annotated pins and paired SPICE netlists. It contains more than 50,000 component instances, 150,000 wires, 100,000 text regions, and 400,000 characters. We further develop an automated schematic-to-netlist pipeline that combines visual recognition, topology construction, and domain-knowledge-guided multi-agent correction. The proposed method achieves 94.54% component detection mAP, 98.57% text recognition accuracy, and 84.47% end-to-end connectivity accuracy. PCBnet provides a benchmark and data foundation for future AI-driven PCB design automation. 

---
# Low-Altitude Fluid Antenna Network with Multi-Agent Reinforcement Learning 

**Authors**: Tong Zhang, Yanfei Su, Shuai Wang, Wanli Ni, Chengzhong Xu, Huseyin Arslan  

**Link**: [PDF](https://arxiv.org/pdf/2608.27909)  

**Abstract**: Low-altitude wireless networks (LAWNs) integrate terrestrial and aerial platforms to provide ubiquitous communication, sensing, and localization services for unmanned aerial vehicles (UAVs) and electric vertical takeoff and landing (eVTOL) aircraft. However, dynamic air-ground and air-air channels, abrupt blockages, and heterogeneous interference hinder the realization of this goal. Nevertheless, fluid antenna (FA), a cutting-edge multiple-input multiple-output (MIMO) technique, overcomes these challenges by reconfiguring antenna positions to unlock additional spatial degrees-of-freedom. In this paper, towards bringing low-altitude FA networks into reality, we study the fast and high-performance FA reconfiguration for low-altitude FA networks with multi-agent reinforcement learning (MARL). Specifically, we present an electromagnetic digital twin (EM-DT)-assisted MARL framework. To fill the sim-to-real gap, we introduce a two-stage transfer learning framework. Our case study shows that joint FA positions and beamforming optimization can enhance the system sum-rate by 118.5%, compared to the fixed position baseline. This gain comes from the dynamic millisecond timescale reconfiguration of FA arrays and the adaptive steering of beams toward aerial users with mobility. 

---
# LandingAgent: A Reference-Annotated Dataset and Agentic Generation Framework for Landing Pages 

**Authors**: Injun Baek, HyeongSeok Lee, Yearim Kim, Junhoo Lee, Nojun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2608.27902)  

**Abstract**: Landing pages are goal-oriented web interfaces that must communicate a target-specific value proposition while organizing information flow, visual hierarchy, and calls to action (CTA). Although large language models can generate plausible webpage code from natural-language prompts, direct generation often yields generic templates and unsupported persuasive claims. We study target-grounded, reference-guided landing-page generation, where a system must create an executable page for a new target by adapting reusable patterns from real pages without copying them. We introduce LandingBench, a reference-profile dataset that abstracts real landing pages into section sequences, layout patterns, tone descriptors, visual emphasis, and CTA structure. Building on LandingBench, we propose LandingAgent, a three-phase agentic framework that profiles the target, constructs a reference-guided wireframe, and refines the page through critique-guided polishing. We evaluate LandingAgent against direct prompting on faithfulness, conciseness, readability, aesthetics, and structural diversity. Experiments show improved target grounding, presentation quality, and layout diversity. Code is available at this https URL. 

---
# OpenStamp: A Watermark for Open-Source Language Models 

**Authors**: Miroojin Bakshi, Saksham Rastogi, Danish Pruthi  

**Link**: [PDF](https://arxiv.org/pdf/2608.27899)  

**Abstract**: With the growing prevalence of large language model (LLM) generated content, watermarking is considered a promising approach for attributing text to LLMs and distinguishing it from human-written content. A prominent class of techniques embeds subtle but detectable signals in generated text by modifying token sampling probabilities. However, such methods are unsuitable for open-source models, where users have white-box access and can easily disable watermarking during inference. In this work, we introduce OpenStamp, a watermarking technique that encodes the watermarking logic directly into the model weights by modifying only the final projection, or unembedding, layer. Through experiments across two models, we show that OpenStamp achieves superior detection performance, with minimal degradation in model capabilities compared to prior methods. The implanted watermark is explicitly designed, and empirically confirmed, to be more robust to paraphrasing attacks and harder to scrub off through post-hoc fine-tuning than prior open-source watermarks. To enable developers to watermark their models, we release our code alongside watermarked versions of 4 popular open-source models. 

---
# SOMTab: Set-Order Mamba for Efficient Tabular In-Context Learning 

**Authors**: Hao Wang, Siyu Zhang, Wei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2608.27882)  

**Abstract**: Tabular foundation models based on in-context learning have recently emerged as strong alternatives to task-specific model fitting. However, the current performance frontier remains dominated by attention-heavy architectures, where attention is used throughout the modeling pipeline. This raises a natural question: is attention necessary at every stage of tabular in-context learning? We introduce SOMTab, a Set-Order Mamba architecture for efficient tabular in-context learning. SOMTab separates representation construction from query-conditioned retrieval. For row and column representations, it maps unordered table tokens into stable latent slots and applies Mamba-based state-space mixing to construct compact representations. For final prediction, it retains attention-based in-context learning to preserve query-conditioned retrieval from labeled context examples. We further introduce DCH-TailMix, a synthetic prior that combines degree-corrected graph heterogeneity with mixed heavy-tailed regimes to diversify synthetic dependency structures. Across tabular benchmarks, SOMTab approaches the performance of strong Transformer-based tabular foundation models while achieving faster inference and lower GPU memory usage, yielding a favorable efficiency--accuracy trade-off. 

---
# From Perspective to Fisheye Depth Estimation and Open-Vocabulary Segmentation 

**Authors**: Rit Gangopadhyay, Alex Wong  

**Link**: [PDF](https://arxiv.org/pdf/2608.27860)  

**Abstract**: Vision foundation models are capable of generalizing across 3-dimensional (3D) scenes with high-fidelity estimates; their empirical success can be attributed to training on large-scale datasets of perspective images. However, when transferred to wide field-of-view (FoV) images, such as those captured by fisheye cameras, they return erroneous outputs due to a covariate shift stemming from the radial distortion on the image pixels. We propose a method to generalize vision foundation models to fisheye cameras. The crux of our method lies in a set of learnable parameters, termed Distortion Extenders (DEX), that model the fisheye distortion coefficients and the distributional shift between fisheye and perspective images encoded in the latent space. By minimizing a self-supervised alignment loss, DEX transforms the latent embeddings of fisheye images to resemble those of perspective images to recover high-fidelity estimates. DEX is architecture- and task-agnostic: We demonstrate DEX on monocular depth estimation and open-vocabulary segmentation for convolution- and Transformer-based architectures, where we consistently improve over baselines across indoor and outdoor fisheye datasets. As a byproduct, the activations of DEX can also be decoded to distortion coefficients to support camera calibration. Code available at: this https URL. 

---
# FedEHR-Agents: Federated Agentic Optimization for Automated EHR Modeling 

**Authors**: Jun Bai, Ruilin Wang, Yue Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.27856)  

**Abstract**: Recent advances in large language models are enabling autonomous clinical agents to perform increasingly complex electronic health record (EHR) modeling workflows. However, agents deployed at individual hospitals remain constrained by institution-specific data and modeling environments, while direct cross-hospital collaboration is restricted by the sensitivity of patient-level EHR data. Although federated learning (FL) provides a natural foundation for privacy-preserving collaboration, existing approaches remain predominantly model-centric, limiting federation to prediction models or their updates while overlooking the richer modeling experience accumulated by autonomous agents. To address this limitation, we propose FedEHR-Agents, an experience-centric federated agentic optimization framework for automated EHR modeling. Each hospital deploys an autonomous clinical EHR agent that performs data preprocessing and model development while refining local clinical modeling experience through historical memory, task-specific evaluation, and TextGrad-based prompt refinement. The federated server performs evidence-guided experience aggregation to integrate reliable and complementary modeling experience across heterogeneous hospitals and distills the aggregated experience into global meta-prompts for subsequent local refinement. Extensive experiments on real-world multi-hospital EHR benchmarks demonstrate that FedEHR-Agents consistently outperforms local and federated baselines across diverse clinical prediction tasks and remains robust across different federation scales and LLM backbones. These results establish clinical modeling experience as a promising collaborative object beyond conventional parameter-centric FL and point toward federated autonomous clinical intelligence. 

---
# FISGuard: Defending Against Membership Inference via Fixed Input Subspaces 

**Authors**: Haocheng Jiang, Hua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2608.27836)  

**Abstract**: As large language models are increasingly adopted in federated learning, protecting user privacy while performing parameter-efficient fine-tuning on distributed private data has become an important challenge. Although clients only share gradients instead of directly uploading raw data, the shared gradients may still leak membership information about training samples. ProjRes (S&P, 2026) further increases this risk: with less information and without accessing model outputs, an attacker can effectively distinguish members from non-members solely based on the projection residual between a candidate representation and the subspace induced by server-observable gradients. Existing defenses against membership inference mostly rely on gradient perturbation or regularization, which can not only degrade model utility but also fail to effectively defend against the membership inference attack introduced by ProjRes, which exploits the geometric structure of gradients.
To address this issue, we propose FISGuard, a lightweight defense. Its key idea is to construct and fix a low-dimensional representation subspace using independent public data, thereby restricting the space through which private representations are exposed via gradients while preserving the primary information required for downstream tasks. This substantially reduces the projection-residual discrepancy between members and non-members.
We evaluate FISGuard against five representative defense methods across three NLP datasets, two LLMs, and two fine-tuning strategies, Adapter and LoRA. The results show that FISGuard reduces the ProjRes attack AUC to near the random-guessing level of 0.5 in most settings, while maintaining downstream task performance close to that of the undefended model and introducing only limited computational overhead, thereby achieving a favorable privacy--utility trade-off. 

---
# Actionable CBFI: Integrating Structural Decomposition and Causal Counterfactual Recourse for Tabular Machine Learning 

**Authors**: Sejong Oh  

**Link**: [PDF](https://arxiv.org/pdf/2608.27821)  

**Abstract**: Explainable artificial intelligence (XAI) increasingly calls for actionable counterfactual recourse, yet current methodologies face challenges related to causal invalidity, excessive cognitive burden, and predictive failure. Exhaustive causal search algorithms often require modifications to multiple attributes, whereas additive attribution-guided methods, such as SHAP, ignore higher-order feature synergies, leading to suboptimal predictive momentum and diffuse intervention effort in complex nonlinear models, such as XGBoost. To bridge this gap, we introduce actionable case-based feature importance (A-CBFI), a diagnosis-prescription integrated framework for tabular machine learning. Grounded in structural causal models (SCMs), A-CBFI isolates synergistic interaction bottlenecks and releases suppressive structural locks, translating them into targeted interventions. By mathematically separating the active user intervention space (L_{\mathrm{active}}) from downstream effects and concentrating over 98.3% of the intervention effort on diagnosed root causes, A-CBFI enables highly targeted interventions. Empirical evaluations across the financial and healthcare domains demonstrate that A-CBFI reduces the active human intervention burden by 76.9% while maintaining comparable global recourse cost to exhaustive causal baselines. By prioritizing the diagnosed causal bottlenecks, A-CBFI provides targeted and actionable recourse while maintaining causal validity and achieving full relative convergence across all causally feasible instances. 

---
# ContextLeak: Exfiltrating LLM Agent Context via Malicious Tools 

**Authors**: Yuqi Jia, Ruiqi Wang, Patrick Li, Yuepeng Hu, Peinian Li, Neil Gong  

**Link**: [PDF](https://arxiv.org/pdf/2608.27800)  

**Abstract**: Exfiltrating an LLM agent's runtime context -- such as the user prompt, execution trajectory, and tool list -- poses severe security and privacy risks to users. Such attacks can be carried out via malicious tools and typically require three conditions: (1) the agent selects the malicious tool for task execution, (2) the agent passes its runtime context as input arguments to the tool, and (3) the tool's implementation transmits these inputs to an attacker-controlled endpoint. Existing work primarily focuses on conditions (1) and (3), leaving condition (2) largely unexplored, despite its critical role in enabling successful context exfiltration.
In this work, we bridge this gap by developing ContextLeak, a malicious tool attack that induces the agent to both select the tool and disclose its context as input arguments. We realize this attack by carefully crafting the tool's name and description using reinforcement learning. Specifically, ContextLeak employs an LLM, referred to as the attack LLM, to automatically generate the malicious tool's name and description. To improve attack effectiveness, we fine-tune the attack LLM via reinforcement learning on a set of shadow users with diverse, simulated agent contexts. Our key technical contribution is the design of novel reward functions tailored to the context exfiltration objective, enabling effective reinforcement-learning-based fine-tuning of the attack LLM. Extensive evaluation demonstrates that our attack remains highly effective even when the shadow users' contexts differ substantially from those of the victim users. Moreover, ContextLeak significantly outperforms existing malicious tool attacks when adapted to this setting. 

---
# How Much Can AI Understand? Toward AI-Assisted Sensemaking of Collaborative Discussion in Groups with Shared History 

**Authors**: Soobin Cho, Mark Zachry, David W. McDonald  

**Link**: [PDF](https://arxiv.org/pdf/2608.27799)  

**Abstract**: AI tools that support collaborative discussion typically treat the discussion as a standalone task, focusing only on its content and setting aside the social context of the group having it. But it is groups with a shared history, with their own norms, hierarchies, and relationships, where the most tangled and complex discussions tend to arise. These discussions cannot be understood apart from that context, and AI that overlooks it risks failing to convey what a discussion means, or even misrepresenting it. Drawing on two studies of how experienced Wikipedia editors read and make sense of discussions, we propose an AI-Assisted Sensemaking Model for Collaborative Discussions, which captures not only a discussion's arguments but also the norms and participants behind it, along with the context that gives each meaning. In this model, the system supports the early stages of the sensemaking process, and the degree to which it performs interpretive work can range from low to high. We argue that higher interpretive work reduces the burden on users but increases their reliance on the system's judgment. We then discuss the risks of an insufficiently intelligible system, what it would take to make one more intelligible, and the safeguards it still requires. 

---
# Compositional Failure in Audio-Visual LLMs: Late-Layer Prior Dominance Under Cross-modal Conflict 

**Authors**: Adarsh Sudheer, David Li, Omar Elbanna, Ishaan Kodarapu, Arjun Bahuguna, Vasu Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2608.27785)  

**Abstract**: We study audio-visual conflict as a compositional generalization test for AV-LLMs: the model must combine synchronized but semantically incompatible audio and video evidence and decide whether the pair matches. On VideoLLaMA 2-7B-AV, three alignment configurations remain nearchance on the scored exact-string Yes/No subset of AVHBench, even though their output priors shift substantially. Similarly, off-the-shelf InternVideo2 experienced a 32.3% accuracy decrease specifically under cross-modal conflict, accompanied by a 17.3% instruction-following failure. We call this failure mode prior dominance: late-layer commitment to an internally preferred answer pattern that is weakly grounded in the conflicting inputs. To explain this behavior, we conduct a mechanistic interpretability analysis and find that commitment remains concentrated at 25.5 $\pm$ 1 layers. We show that stronger temporal alignment changes answer bias, but do not improve compositional conflict resolution. Code and data to reproduce our mechanistic audit and behavioral evaluations are available at this https URL. 

---
# Beyond Search-Imitation: Prior-Directed Exploration for Searchless Chess 

**Authors**: Szymon Miłosz, Piotr Duch, Szymon Grabowski  

**Link**: [PDF](https://arxiv.org/pdf/2608.27757)  

**Abstract**: Searchless chess networks reach human master strength from a single forward pass by imitating a stronger teacher: the strongest, Leela Chess Zero's (Lc0) released Chessformer, distills the visit counts of an AlphaZero-style Monte Carlo Tree Search (MCTS). Imitating a search is a poor proxy for playing without one, so we fine-tune for single-pass strength with self-play reinforcement learning (RL). Its exploration is usually supplied by an entropy bonus, the reverse Kullback-Leibler (KL) divergence to uniform. We replace it with a forward, mass-covering KL toward the network's own MCTS prior (prior-directed exploration), so exploration covers the moves the prior judges promising, and pair it with an entropy-adaptive sampling temperature, set by the value head's outcome uncertainty, that sharpens once a position is decided. In about two thousand steps it raises puzzle accuracy from 93.9% to 94.9% on a 100,000-puzzle suite and mate-in-four accuracy from 77% to 81% while holding searchless strength at or slightly above the base. Measuring tactical accuracy and playing strength together across a matched-compute sweep, we find the two dissociate: accuracy gains fall in a one-point band while ratings straddle the base, and a control fine-tuned on puzzles alone posts the study's largest tactical gains while shedding roughly 260 Elo; a better puzzle-solver is not thereby a stronger player. Distribution-level measurements show what anchoring buys: without a regularizer self-play collapses onto a single line of play, and the puzzles newly solved are the near misses whose winning move the prior kept alive. The forward-KL prior tops the rating ladder, statistically tied with a reverse-KL anchor that concentrates twice as hard and drops the hardest solutions the mass-covering prior keeps in support. 

---
# Efficient Auto-Interpretability of AI Models in Biology 

**Authors**: Piotr Jedryszek, Oliver M. Crook  

**Link**: [PDF](https://arxiv.org/pdf/2608.27754)  

**Abstract**: Sparse autoencoders (SAEs), and other interpretability methods could turn AI models in Biology and other fields into engines of scientific discovery by explaining the superhuman capabilities of those models. However, a latent is only useful if we know three things: whether it is coherent, whether it can be described, and whether that description has predictive power. These questions are routinely conflated. We assemble them into a single pipeline and report the practical innovations each stage required. First, cross-seed dictionary stability prioritises which latents are worth spending resources to investigate. Second, an intruder-detection task asks whether a latents activating examples share a recognizable pattern. Third, a separate pass proposes a candidate biological description which we convert into falsifiable predictions which can be tested in silico. Deployed on the Boltz-1 Pairformer trunk, stability prioritisation finds interpretable latents using about 4.4 times fewer latent evaluations each, and at 5.2 times lower measured cost, while recovering over half of them, and the external check shows the surfaced motifs are significantly enriched for their claimed annotations. The results also suggest a possible tension: the cross- seed stability might be selecting for some types of features, like structure-related ones, much more than others, such as function-related features. 

---
# RiskBlend: A Multi-Signal Framework for Test Input Prioritization in Machine Learning Regression Testing 

**Authors**: Madhusudan Srinivasan, Namith Nishal Raphae  

**Link**: [PDF](https://arxiv.org/pdf/2608.27704)  

**Abstract**: When machine learning classifiers are retrained, inputs correctly classified by the previous model version may be misclassified by the updated version, creating regression faults that are costly to detect because verifying predictions against ground truth may require human annotation, expert review, or expensive simulation rather than inexpensive model inference. Test input prioritization addresses this problem by ranking inputs so that a limited verification budget reveals as many regression faults as possible. Existing approaches rely predominantly on single-model confidence scores and do not exploit how predictions, decision boundaries, and local neighborhoods change between model versions. We propose RiskBlend, a classifier-agnostic prioritization framework that combines four complementary risk signals: historical failure patterns, prediction shift, decision-boundary shift, and neighborhood change. These signals are combined using validation-learned APFD-squared weighting. Across four datasets, five classifiers, four regression-update scenarios, and 15 random seeds, totaling 1,200 experimental configurations, RiskBlend achieves the highest average APFD in all 80 dataset-classifier-scenario combinations, with improvements of up to 0.32 APFD over the strongest baseline. Confidence-based methods remain competitive primarily for linear classifiers on sparse categorical features, which we attribute to feature-space geometry. The results show that cross-version behavioral signals provide important complementary information for prioritizing regression faults in machine learning systems. 

---
# Evaluating Loss Functions in Differentiable Out-of-Domain Sound-Matching with Partial Parameter Distance 

**Authors**: Amir Salimi, Daniel Penner, Kalvin Eng, Abram Hindle, Osmar R. Zaïane  

**Link**: [PDF](https://arxiv.org/pdf/2608.27698)  

**Abstract**: In out-of-domain (OOD) sound-matching, a synthesizer is optimized to mimic a sound it did not generate. OOD evaluation of loss functions is underexplored in part because the standard "parameter loss" metric requires a shared parameter space between target and imitator, which OOD settings lack. We introduce Partial Parameter Distance (PPD), which applies parameter loss only to the critical parameters that mismatched synthesizers share (e.g., filter cutoffs), enabling automatically evaluated OOD experiments; we verify its results with blinded listening tests. Across seven scenarios involving band-pass filtering, amplitude modulation, and pitch-bending, we evaluate four differentiable loss functions (SIMSE_Spec, L1_Spec, JTFS, DTW_Envelope). Loss-function effectiveness remains tightly coupled to the method of synthesis: SIMSE_Spec excels at filter-cutoff recovery, DTW_Envelope at amplitude-modulation recovery, and JTFS at smooth pitch trajectories. Parameter-based evaluation agrees with listening tests on the top-ranked loss function in five of seven scenarios, demonstrating its utility as a diagnostic tool. 

---
# CARDINAL Predicts Cardiovascular Risk From Non-contrast Cardiac CT 

**Authors**: Roy Gabriel, Nattakorn Kittisut, Jamshid Hassanpour, Michael Galarnyk, Abanoub Abdelmalak, Marly van Assen, Carlo N. De Cecco, Arshed Quyyumi, Ali Adibi  

**Link**: [PDF](https://arxiv.org/pdf/2608.27690)  

**Abstract**: Cardiovascular risk prediction remains limited by incomplete clinical data and imaging biomarkers that reduce computed tomography (CT) to a small number of handcrafted features. We developed CARDINAL (Cardiovascular Assessment via Representation learning from Deep Imaging with Nested Anatomical Latent embeddings), a clinically grounded framework that learns compact representations from routine non-contrast cardiac CT for major adverse cardiovascular event (MACE) prediction. In 17,659 patients, CARDINAL was evaluated for 1-, 3-, 5-, and 10-year MACE prediction against American Heart Association (AHA) pooled cohort equations (PCE), AHA predicting risk of cardiovascular disease events (PREVENT), coronary artery calcium (CAC), segmentation-derived CT biomarkers, and 70-feature structural radiomics. Gains were largest at longer horizons. At 10 years, CARDINAL (joint) achieved an area under the receiver operating characteristic curve (AUROC) of 0.866 $\pm$ 0.020 and an area under the precision-recall curve (AUPRC) of 0.890 $\pm$ 0.015, compared with an AUROC of 0.826 $\pm$ 0.023 and an AUPRC of 0.826 $\pm$ 0.022 for structural radiomics, the strongest baseline. CARDINAL also achieved the highest survival concordance index (C-index), 0.753 $\pm$ 0.015, and high-versus-low risk-tertile hazard ratio, 10.78 $\pm$ 3.16, with favorable reclassification and exploratory calibration. These findings suggest that non-contrast cardiac CT contains prognostic information beyond conventional risk equations, CAC scoring, and engineered imaging biomarkers. 

---
# First Make It Playable, Then Make It Good: Staged Interaction Learning for Small Dialogue-Game Agents 

**Authors**: Syed Mahbubul Huq, Pranava Madhyastha  

**Link**: [PDF](https://arxiv.org/pdf/2608.27672)  

**Abstract**: We present Qwen-GuidePlay-2B, a 2B-parameter language model for dialogue-game interaction. We fine-tune Qwen3.5-2B using three steps: a) SFT on only successful game trajectories from Playpen, b) weighted turn-level SFT, and c) teacher-guided SFT. The teacher model (which is a larger model) is only used to fix formatting and evaluate examples, but does not create new gold actions. Our final model scores 57.12 clemscore and 42.68 statscore on the public Playpen validation. In the officially released challenge results, our model obtains the second-highest Playpen clemscore delta among submitted systems (which is approximately +36 over its base model). Our findings suggest that imitating full trajectories helps with playability, while turn-level and teacher-guided training usually improve decision-making and increase the overall score. Alternative procedurally heavy approaches like replay-repair and hard-example mining did not help, which suggests that small models are performant simply by using careful curation strategies rather than aggressive changes. We make available both the model and the code for reproducibility. 

---
# Semantic Watermarking with Order-Robust Detection over Sub-sentence Units 

**Authors**: Abdulrahman Diaa, Jonathan Petit, Florian Kerschbaum  

**Link**: [PDF](https://arxiv.org/pdf/2608.27666)  

**Abstract**: Semantic watermarks tie the mark to sentence meaning rather than token choices, promising robustness to content-preserving edits. However, the detector only observes attacker-supplied text, which can be reworded, reordered, or resegmented to evade detection without content loss. Rewording, reordering, and resegmentation all cause embedding displacement: detection tests embeddings different from those selected during watermarking and can therefore lose the mark. Our adaptive embedding displacement attack (EDA) admits all three edits under a single objective that maximizes this displacement. It uses a public paraphraser and surrogate encoder without access to the provider's generator or secret key. At a 5% false-positive rate (FPR) and content-preservation threshold $\bar{q}=90\%$, EDA successfully removes the mark on between 32.6% and 47.9% of documents across four schemes, the highest among the tested attacks. Therefore, EDA evaluates the schemes' robustness more thoroughly than passive paraphrasing.
To address these vulnerabilities, we design (k)-SwordStamp: semantic watermarks with order-robust detection over sub-sentence units, reducing sensitivity to attacker-chosen structure at a small quality cost. Against k-SwordStamp, the strongest no-box attack we test is an EDA variant adapted to its design, with a 10.8% attack-success rate. A stronger EDA with access to the provider's detector and secret key reaches a 39.7% attack-success rate, compared with 65.5% on k-SemStamp. Our code is available at this https URL. 

---
# Knowing Before Answering: Decoding Language Models for Reliable RAG 

**Authors**: Syed Mahbubul Huq, Christopher Child, Tillman Weyde, Pranava Madhyastha  

**Link**: [PDF](https://arxiv.org/pdf/2608.27661)  

**Abstract**: In Retrieval-Augmented Generation (RAG), retrieval may provide insufficient or conflicting information needed to answer a question. The system should not only know when to answer but also be able to identify cases in which the documents provided in RAG are insufficient or contain conflicting information. This can be framed as a three-way classification problem, where we use the model's internal signals to determine whether the provided information in the input can be classified as sufficient, insufficient, or conflicting. We create a controlled benchmark dataset that replicates a RAG setup with fictitious information and labels each instance as answerable, insufficient, or conflicting. We use hidden activations and attention-derived features as inputs to train a lightweight linear model to distinguish among the three classes. Across 16 language models spanning different architectures and a range of model sizes, our feature-based router consistently outperforms prompting-based baselines and the performance of specialised RAG-models. We further conduct analyses into the information dynamics of the models. We show that the most informative signals for the classification are available in the middle layers, with hidden activation states being more effective than attention values or the MLP-feature outputs in most of the tested models. Overall, our results suggest that language models internally encode whether retrieved evidence is sufficient to support answering, and that this signal can be decoded reliably for RAG triage. 

---
# Curvature-Aware Radius Shrinkage for Adaptive Nearest Neighbor Classification 

**Authors**: Alexandre L. M. Levada  

**Link**: [PDF](https://arxiv.org/pdf/2608.27634)  

**Abstract**: Nearest neighbor classification relies fundamentally on how locality is defined, yet conventional $k$-NN imposes the same neighborhood cardinality throughout the feature space. This assumption can be inadequate for data whose local geometry varies substantially across the underlying manifold. We introduce Curvature-Aware Radius Shrinkage for Adaptive Nearest Neighbor Classification (CARSANN), a geometry-driven framework that adapts the spatial support of each neighborhood according to local geometric complexity. CARSANN first estimates intrinsic dimensionality using TwoNN and constructs an intrinsic representation through principal component analysis. Local mean curvature is then estimated using a shape-operator-based formulation and controls neighborhood scale: highly curved regions receive stronger radius shrinkage, whereas approximately flat regions retain broader spatial support. Unlike methods that modify only the number of neighbors or the local metric, CARSANN explicitly adapts the spatial extent of local evidence. Experiments on more than 70 real-world OpenML datasets show that CARSANN consistently improves upon standard $k$-NN and is competitive with adaptive nearest-neighbor methods. In a controlled comparison using the same base neighborhood size, CARSANN achieves higher balanced accuracy on 40 of 45 datasets, increasing mean balanced accuracy from 0.6506 to 0.7528. The advantage also persists against $k$-NN with fixed $k=5$. Friedman and Nemenyi tests confirm that the improvements are statistically significant. These results indicate that local manifold curvature can serve as an effective geometric control variable for adapting neighborhood support, providing a complementary paradigm to cardinality-based nearest-neighbor adaptation. 

---
# Depth-Aware Pothole Detection Using YOLO and RT-DETR at the Edge 

**Authors**: Md Monjurul Ahsan Prodhan, Md Nour Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2608.27633)  

**Abstract**: Pothole detection and its severity measurement is still an important challenges in urban infrastructure management, where late maintenance directly contributes to vehicle damage, road accidents, and escalating repair costs. Existing automated approaches depend on 2D RGB images and cannot measure physical depth of potholes. In this paper, we present a depthaware pothole detection framework and then compare five architectures: YOLOv8n, YOLOv8nSeg, YOLOv9t, RTDETRL, and RTDETRX for RGB-D sensor fusion-based detection and automated depth measurement. A custom offline augmentation pipeline is used here to simulate adverse road monitoring conditions. All models are trained on the PothRGBD dataset with an 80% training and 20% validation split and evaluated using Precision, Recall, mAP@50, and mAP@50_95. Before measuring the depth data, all depth maps are corrected for camera tilt using RANSAC ground-plane orthorectification and all zero-valued sensor pixels are cast to NaN before any statistic is computed. YOLOv8nSeg achieves the highest mAP@50 of 0.9556 and mAP@50_95 of 0.6758 with the most accurate depth estimate of 2.96 cm with the pixel-precise Dseg algorithm. YOLOv8n achieves the fastest inference at 3.6ms. RTDETRX achieves the highest detection confidence at 92.70%. An important finding is that even after full RANSAC orthorectification, bounding box models overestimate pothole depth by 0.16 to 0.21 cm compared to pixel precise segmentation masks. This confirms that the pavement inclusion bias is structural rather than a calibration artifact. 

---
# LitCurate: A Configuration-Driven AI-Assisted Framework for Scientific Database Construction with an Application to Lower-Mantle Equation-of-State Data 

**Authors**: Abin Shakya, Wilson Samuels, Dominica Wilson, Gioia A. Marchi, Israa Draz, Chenxing Luo, Renata M. Wentzcovitch  

**Link**: [PDF](https://arxiv.org/pdf/2608.27629)  

**Abstract**: The growing scientific literature contains decades of experimental and computational results that could support data-driven and physics-based modeling, yet much of this infor- mation remains locked in publications and is not readily usable for large-scale analysis or sci- entific software. Building structured databases from the literature is particularly challenging whenrelevantstudiesmustfirstbediscoveredamonglargecollectionsofpapersandreported quantities must be extracted with enough scientific context to remain usable. We present LitCurate, an open-source framework for building scientific databases from the literature using large language models within an auditable, stage-wise curation workflow. LitCurate integratesliteraturediscovery, relevancescreening, full-textprocessing, andstructuredinfor- mation extraction while retaining intermediate results and provenance, allowing researchers to inspect and revise individual stages rather than treating automated curation as a black- box process. We apply LitCurate to construct an equation-of-state database of lower-mantle and lower-mantle-relevant high-pressure mineral phases from experimental and theoretical studies, comprising 1,334 entries from 205 papers. The resulting dataset links reported equation-of-state parameters to mineral phases, compositions, equation formulations, meth- ods, and parameter constraints, and labels values as source-reported or citation-reported when provenance can be determined. The records are available through a searchable web application. By connecting scientific literature to traceable, machine-readable data, LitCu- rate provides a reusable approach for transforming accumulated literature into resources for scientific analysis and computational modeling. 

---
# Tensor-Accelerated Eager Multi-Resolution Grids for Evolving Large-Scale Substrates 

**Authors**: Romain Claret, Michael O'Neill, Paul Cotofrei, Kilian Stoffel  

**Link**: [PDF](https://arxiv.org/pdf/2608.27612)  

**Abstract**: In neuroevolution, indirect encoding generates neural network connectivity from a compact genome rather than specifying each connection. ES-HyperNEAT automatically discovers where to place hidden nodes by examining CPPN output patterns: it recursively subdivides space using a quadtree, expanding regions where CPPN outputs show high variance. This adaptive approach discovers network topology without manual substrate specification, extending the fixed-grid HyperNEAT framework built on NEAT.
However, the quadtree resists tensorization. Each depth level depends on the parent's variance, forcing sequential evaluation. Different CPPNs produce different subdivision patterns, preventing batching. And variable leaf counts are incompatible with JAX's static shape requirement for JIT compilation. Our prior work confirmed these limits at depths exceeding 5, and a JAX reimplementation of the quadtree yielded only marginal speedup despite batched optimizations, motivating the eager reformulation presented here.
We present EMR-HyperNEAT, which evaluates all positions at all resolutions up front, then filters using the same variance criterion: ES-HyperNEAT's subdivide_if(var > $\theta$) becomes eval_all(); filter(var > $\theta$). This performs more CPPN queries than necessary, but all queries become independent and parallelizable across both cores and population members, reducing complexity from \BigO($4^D$) to \BigO($4^D/P$) across $P$ parallel cores. Recurrent substrate configurations become feasible through a connection type taxonomy. The experiments section validates 12-34$\times$ on-device GPU speedup on XOR at depths 5-7, and empirically higher solve rates across benchmarks. 

---
# PHR-VLA: Planning Horizon Reasoning for Vision-Language-Action Models 

**Authors**: Davood Soleymanzadeh, Kaidi Zhang, Zhiyuan Zhang, Bihao Zhang, Xiao Liang, Yu She, Minghui Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2608.27609)  

**Abstract**: Vision-language-action models (VLAs) have shown strong promise for general-purpose robotic manipulation by mapping language instructions and vision observations directly to actions. However, most VLAs primarily condition action prediction on current observations and lack an explicit mechanism for reasoning over future task dynamics, which is particularly important for fine-grained, contact-rich manipulation. We present PHR-VLA, a framework that enables planning-horizon reasoning in VLAs through privileged latent representations of future dynamics. PHR-VLA introduces a lightweight auxiliary future head that, during training, aligns the VLA's internal representations with latent dynamics extracted from future observations. Evaluation results demonstrate that local, contact-centric, patch-level latent dynamics supervision from the wrist camera improves success rate on LIBERO from 84.1% to 88.4% and on real-world disassembly tasks from 63.3% to 82.5%. Patch-level supervision from a third-person camera also improves performance on Meta-World from 56.70% to 57.8%. These results demonstrate that privileged latent dynamics alignment provides an effective training signal for improving anticipatory reasoning in VLA policies. Project website: \href{this https URL}{this https URL} 

---
# Quanta Perception as Probabilistic Events 

**Authors**: Varun Sundar, Pavan Thodima, Sacha Jungerman, Mohit Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2608.27584)  

**Abstract**: Autonomous systems rely on extracting information from light, yet remain brittle in extreme environments, from nighttime navigation to high-speed robotics. Conventional sensors aggregate photons over fixed exposures, imposing trade-offs between sensitivity, dynamic range, and temporal resolution that degrade perception when photons are scarce or dynamics are rapid. Quanta sensors detect individual photons, but their streams exceed real-time compute and latency budgets by orders of magnitude.
Here we introduce $\textit{probabilistic events}$, a computational primitive for real-time quanta perception from individual photon detections. By computing the posterior over the time since the last intensity change, we represent photon streams as recursive belief states. Rather than fixed-threshold event-camera triggers, this recursive Bayesian formulation yields three low-latency signals: motion-adaptive scene flux, high-fidelity activity maps, and entropy-based perceptual uncertainty. This representation enables perception in extreme conditions, including pose estimation of a running person at $\sim$0.05 lux---without retraining vision models. Our approach processes input streams exceeding 50{,}000 quanta frames per second on commodity GPU hardware---yielding kilohertz-scale outputs up to four orders of magnitude faster than state-of-the-art quanta reconstruction baselines, even for megapixel arrays. By replacing frame reconstruction with direct probabilistic inference over photon streams, this work bridges photon-counting quanta sensing with robotic vision. 

---
# Self-Explainable Multi-Label Graph Neural Network for Correlated Evidence Attribution 

**Authors**: Yingqi Feng, Yufei Tang, Min Shi, Xingquan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.27574)  

**Abstract**: Multi-label graph learning intends to capture the intrinsic complexity of real-world applications, where one sample is often related to multiple groups or consists of multiple objects. To date, a handful of multi-label graph learning methods exist, but none of them integrate training-time interpretation capability. While post-hoc graph explainers have been developed, they do not explicitly model label-dependent evidence sharing in multi-label graph learners, especially when label pairs are weakly or negatively associated. As a result, post-hoc approaches may miss how evidence should be shared or separated across different labels. This paper advances a new end-to-end self-explainable multi-label graph neural network (SEMGNN), which aims to simultaneously classify multi-labeled nodes and identify edges significantly contributing to each target node w.r.t. predicted labels. Different from post-hoc methods, SEMGNN jointly learns a predictor and a sparse edge-mask explainer within a unified framework and training objective. Label-label correlations are used to improve multi-label node classification and enhance individual label explanations, so that different labels of a node can be supported by distinct yet coherent structural and/or correlated evidence. Experiments and comparisons on synthetic and real-world multi-label networks, in social networking, entertainment, and life sciences, show that SEMGNN achieves competitive or improved predictive performance while providing more faithful and compact label-conditioned explanations. 

---
# FVeinSyn: Synthetic Finger Vein Image Generator 

**Authors**: Yifan Wang, Jie Gui, Adams Wai Kin Kong, Baosheng Yu, Changsheng Chen, Qi Li, Zhenan Sun, James Tin-Yau Kwok, Alex Kot  

**Link**: [PDF](https://arxiv.org/pdf/2608.27527)  

**Abstract**: A major challenge in finger vein recognition is the lack of large-scale public datasets. Existing datasets contain few identities and limited samples per finger, restricting the advancement of deep learning-based methods. To address this, we propose FVeinSyn, a large-scale controllable synthetic data generation framework for finger vein. It explicitly decouples synthesis of vascular topology and imaging appearance to mitigate the limitations caused by insufficient training samples, such as inadequate identity diversity and restricted realism. Specifically: first, a finger vein identity generator models vascular topology under physiological and geometric constraints using stochastic L-systems, producing anatomically valid and identity-distinctive vascular patterns. Then, a cascaded region-aware GAN renders the topological maps into realistic near-infrared images. Finally, an intra-class diversity generator introduces geometric and optical perturbations to simulate realistic intra-class variations. Using FVeinSyn, we generated 500,000 images (10,000 vein identities, 50 samples per identity) and conducted extensive evaluations. Results show that FVeinSyn holds significant advantages in realism, identity diversity, vascular pattern consistency, and intra-class diversity. Models trained with FVeinSyn outperform real-data-only baselines a cross eight public datasets, achieving an average accuracy improvement of 27.43\%. The code is available at: this https URL. 

---
# Destroy Me: Automatic Artifact Generation for Histopathology Images 

**Authors**: Zuzanna Krawczyk-Borysiak, Adam Krawczyk, Mateusz Miller, Gabriela Kaczmarek, Sławomir Pakuło, Małgorzata Sokół, Żaneta Swiderska-Chadaj  

**Link**: [PDF](https://arxiv.org/pdf/2608.27516)  

**Abstract**: Deep learning's diagnostic utility in pathology is constrained by model vulnerability to real-world data imperfections. While current strategies favor "perfect data" by filtering low-quality regions, which can lead to the loss of valuable diagnostic context, we propose a paradigm shift: engineering models to thrive in imperfect environments using "Destroy Me", a hybrid framework for realistic artifact synthesis and robust data augmentation. Our approach combines Stable Diffusion, fine-tuned to preserve morphological continuity by realistically integrating artifacts with the underlying tissue architecture, with physics-based procedural modeling to synthesize six common artifact types: tissue folds, precipitates, blur, stitching errors, dust, and pen markers. Artifact fidelity is assessed using Kernel Inception Distance (KID) and color Wasserstein distance metrics. Validating this strategy on lung adenocarcinoma pattern classification with an nnU-Net, we confirm that models trained on "destroyed" patches consistently outperform baselines on independent real-world datasets. Specifically, we observed a 10.5% relative improvement in macro F1-score and a 15% relative increase in the Cohen's Kappa ($\kappa$) coefficient. Crucially, our results demonstrate that selective, impact-weighted augmentation is vital for balancing practical robustness with the preservation of subtle diagnostic features. 

---
# Trajectory-Level Speculative Decoding for Diffusion Language Models 

**Authors**: Tianxiang Pan, Baitao Gong, Mo Guang, Hongwei Yong, Tianpeng Jiang, Yaqian Li, Zheng Cao, Kaiwen Long  

**Link**: [PDF](https://arxiv.org/pdf/2608.27514)  

**Abstract**: Diffusion-based language models (dLLMs) enable parallel token generation through iterative denoising, but existing decoding strategies collapse to single-token generation under low confidence, severely limiting throughput. Unlike autoregressive models where speculative decoding operates on token sequences in a fixed left-to-right order, dLLMs require speculating over denoising trajectories-sequences of multi-token updates with explicit positions and unmasking orders. We develop a trajectory-level speculative framework that constructs draft denoising trajectories via confidence-stratified tree exploration and verifies them through blockwise parallel evaluation with bidirectional attention masking. Our method further introduces inter-block speculation, exploiting diffusion models' bidirectional structure to perform cross-block lookahead. We formally characterize when this approach is exact and identify trajectory drift as the fundamental cost of increased parallelism. Building on Fast-dLLM's dual-cache infrastructure, our framework reduces denoising iterations by 30-40% and increases tokens-per-step from 2.6 to 4.3, achieving 7-14x speedup over vanilla dLLMs and 1.3x over Fast-dLLM with less than 1% accuracy change across reasoning and code benchmarks. 

---
# DAMP: Decay-Aware Mixed-Precision Recurrent-State Quantization 

**Authors**: Tao Zhang, Jianchao Tan, Pingwei Sun, Yanqi Yu, Zixu Jiang, Yuchen Xie, Xunliang Cai, Ziqian Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2608.27513)  

**Abstract**: Softmax attention stores key and value vectors for every preceding token, causing inference memory to grow with sequence length. Recent language models incorporating Gated DeltaNet (GDN) or Kimi Delta Attention (KDA) reduce this cost by replacing the KV cache in most layers with fixed-size recurrent states. However, these recurrent states are commonly stored in FP32 and consume substantial GPU memory; their updates are memory-bandwidth bound and contribute significantly to decoding latency. To our knowledge, we are the first to study post-training quantization of recurrent states in GDN and KDA based language models. We find that uniform quantization provides a poor accuracy--storage trade-off: INT8 and FP8 already degrade accuracy on complex reasoning tasks, while INT4 and NVFP4 reduce it to near zero. We further find that most quantization-error energy is concentrated in a small subset of channels and that the relative decay strength of state channels remains stable across prompts and tasks. Motivated by these findings, DAMP uses both quantization-error energy and decay-based persistence to identify high-risk channels during offline calibration. It stores these channels at higher precision and the remainder in INT8. We evaluate DAMP on Qwen3.6-35B and Kimi-Linear-48B across six benchmarks covering mathematical reasoning, general reasoning, and code generation. At 9.9 bits per state value, DAMP maintains average accuracy close to the FP32 baseline. DAMP reduces recurrent-state storage by 69.1%, accelerates the recurrent-state update kernel by up to 2.01x, and lowers full-model TPOT by up to 10.9%. 

---
# Quantization-Triggered Backdoors in Language Models: Cross-Quantizer Transferability and the Validation--Deployment Gap 

**Authors**: Jacopo Dardini, Claudio Stanzione, Giordano Colò, Giuseppe Fenza  

**Link**: [PDF](https://arxiv.org/pdf/2608.27512)  

**Abstract**: Post-training quantization is often treated as a semantically neutral optimization for edge deployment of Large Language Models. When a full-precision source checkpoint is evaluated and quantization is applied downstream without equivalent re-evaluation, this workflow creates a structural validation--deployment gap: because quantization is a many-to-one mapping over parameter space, source-precision certification does not guarantee behavioral equivalence in the deployed configuration. We formalize this gap through Quantization Behavioral Equivalence Classes (QBECs) and prove that QBEC membership does not imply behavioral equivalence, providing a theoretical basis for quantization-triggered backdoor attacks. Building on a three-stage adversarial fine-tuning framework, we embed latent malicious payloads into models that satisfy the source-precision checks used in our evaluation, yet activate targeted adversarial behavior upon INT8 or 4-bit compression. We evaluate this threat in two operationally motivated scenarios, tactical machine translation and political content analysis, extending prior work from decoder-only causal LMs to multilingual encoder-decoder sequence-to-sequence models. Results show that backdoored translation models move from zero measured friend--foe corruption at repaired FP16 to up to 85.02% inversion after quantization, and that a paired stance classifier measures an ideological shift of up to $\Delta\mathrm{Bias}=0.33$ upon compression. A cross-quantizer transferability analysis further shows that attack persistence varies across quantization schemes and model architectures, rather than being determined by nominal bit-width alone. These findings demonstrate that source-precision auditing alone does not rule out quantization-triggered behavior and that the final deployed configuration must be included in behavioral certification for trustworthy edge AI. 

---
# Marginal Coverage Credit Reduces Redundant Exploration in Parallel State-Entropy Optimization 

**Authors**: Junhao Cao, Hongyi Xia, Jianian Wu, Xiaopeng Yi, Lixia Huang, Ping Guo  

**Link**: [PDF](https://arxiv.org/pdf/2608.27507)  

**Abstract**: Policy Gradient for Parallel State Entropy maximization (PGPSE) expands state-space coverage by training independently parameterized policies in replicated copies of the same environment. However, its pooled team-entropy score measures only collective exploration and cannot identify policies that contribute non-redundant coverage. We introduce Marginal Coverage Credit for PGPSE (MCC-PGPSE), which combines leave-one-policy-out coverage with state-owner specialization to estimate policy-specific credit. MCC-PGPSE preserves PGPSE's pooled objective and redistributes non-negative auxiliary intrinsic rewards according to these credits without changing their total mass. This redistribution is designed to discourage redundant visitation and promote complementary coverage. We evaluated MCC-PGPSE in controlled environments, seven public discrete-state benchmarks, and representative Room and Maze settings from the original PGPSE protocol. Across all tested settings, MCC-PGPSE produced positive final window gains in normalized team state entropy and state support over the Entropy baseline. Controlled-task comparisons and the fixed-suite public aggregate were significant, whereas five-seed original-protocol comparisons were directionally consistent. Ablations and credit alignment controls indicate that most gains arise from leave-one-policy-out coverage rather than non-uniform weighting, mismatched credit, or neural novelty alone. These results support contribution-conditioned auxiliary reward allocation as an interpretable approach to improving complementary coverage among parallel policies in discrete state spaces. 

---
# A Survey on Rubric-Guided Reinforcement Learning for Language Models 

**Authors**: Zifei Shan, Fangning Shao  

**Link**: [PDF](https://arxiv.org/pdf/2608.27505)  

**Abstract**: Reinforcement learning from human feedback (RLHF) has become the dominant paradigm for aligning large language models (LLMs) with human preferences. However, traditional RLHF relies on scalar reward signals that lack interpretability and fail to capture the multifaceted nature of response quality. Rubric-guided reinforcement learning addresses these limitations by introducing structured, interpretable evaluation criteria, or rubrics, as the backbone of reward design, feedback generation, and policy optimization. In this survey, we introduce a Bayesian framework that defines constitutions as prior distributions $P(R)$ over evaluation criteria and rubrics as conditional instantiations $R_x \sim P(R|x)$. Under this unified view, we present a taxonomy of rubric-guided RL along the prior-posterior axis, covering constitutional AI, instance-specific rubrics, process-level supervision, self-evolving rubrics, and their agentic and multimodal extensions. Furthermore, as rubrics are natural-language artifacts, we present a linguistic analysis of how granularity trade-offs, semantic drift, and linguistic reward hacking impact alignment reliability, identifying key open problems for future research. 

---
# XHotpotQA: A Benchmark for Cross-Lingual Knowledge Composition in Multi-Hop Question Answering 

**Authors**: Iman Barati, Arash Ghafouri, Behrouz Minaei-Bidgoli  

**Link**: [PDF](https://arxiv.org/pdf/2608.27481)  

**Abstract**: Knowledge-intensive multi-hop question answering requires systems to select evidence and compose dependent facts, yet multilingual benchmarks usually translate an entire example into one language. This hides failures at language boundaries inside the reasoning chain. We introduce XHotpotQA, a controlled benchmark for cross-lingual knowledge composition over mixed-language evidence. Each instance is modeled as an evidence-dependency graph whose question, bridge evidence, answer-bearing evidence, and distractors have explicit language assignments. The audited resource contains 15,661 training and 7,405 validation instances, with sentence-level support supervision and supplied distractors. In validation, 99.81% of items cross the question-to-gold-evidence language interface and 95.60% use gold paragraphs in different languages. Across three reader artifacts, full question-evidence mismatch is associated with 10.25 to 15.79 lower Unicode-aware answer F1 than partial alignment, and different-script evidence with deficits of 11.98 to 23.70 points; the corresponding adapted-selector contrasts are 1.71 and 1.78 points. Under this supplied-candidate design, the evaluated readers therefore show substantially larger condition-associated deficits than the selector. XHotpotQA provides role-aware diagnostics, modular evaluation, and an audited test bed for knowledge-based systems that must integrate evidence across languages. 

---
# Select, Don't Train: The Benefits of Modular Entity Disambiguation with LLM-Based Selection 

**Authors**: Fina Polat, Daniel Daza, Pengyu Zhang, Klim Zaporojets, Paul Groth  

**Link**: [PDF](https://arxiv.org/pdf/2608.27470)  

**Abstract**: Entity Disambiguation (ED) is a key task for constructing and using knowledge graphs. State-of-the-art neural approaches commonly model ED as a single task, although it consists of two distinct subproblems: retrieving candidate entities and selecting the correct one given context. Dual-encoder models optimize for both within a shared embedding space, forcing representations to balance high-recall retrieval with fine-grained selection, and they require trained retrievers, which are costly to maintain as knowledge graphs change. While recent work has begun to combine retrievers with LLM-based selectors, the interplay between the two stages has not been studied systematically. In this paper, we present a systematic comparison of retrieval strategies for candidate generation under a shared LLM-based selection stage, combining sparse retrieval (BM25), Web KB search, and a state-of-the-art trained dense retriever with several open- and closed-source LLMs. We show that, once selection is delegated to a capable LLM, training the retriever provides only modest additional value: a fully training-free BM25 retriever paired with an LLM selector reaches a new state of the art on the ZELDA benchmark, raising inKB micro-F1 from 82.3 to 86.3 (+4); pairing the same LLM with a trained dense retriever reaches 88.5. Decoupling retrieval from selection also exposes a limitation of current ED systems: when the correct entity is missing from retrieved candidates, they are forced to predict an incorrect entity. In contrast, our framework allows for abstention when retrieval failure is detected. In an evaluation setting that rewards correct abstentions, the training-free BM25 + LLM pipeline reaches 90.7 F1. 

---
# UIC-AIHealth4All at ArchEHR-QA 2026: Answer-First Evidence Grounding for Clinical Question Answering 

**Authors**: Mohammad Arvan, Hossein Haeri, Natalie Parde, Rebecca T. Feinstein  

**Link**: [PDF](https://arxiv.org/pdf/2608.27467)  

**Abstract**: We describe the UIC-AIHealth4All system for ArchEHR-QA 2026, a shared task on grounded question answering from electronic health records. We participated in Subtasks 2 (evidence identification), 3 (answer generation), and 4 (answer-evidence alignment). For Subtasks 2 and 3, we propose an answer-first pipeline in which the model generates candidate answers citing specific note sentences before classifying the full evidence set, exploiting the asymmetry between judging relevance in the abstract versus relative to a generated answer. For Subtask 4, we apply self-consistency voting over five independent model calls, retaining links by vote threshold. Our pipeline ranked third on evidence identification (Strict Micro F1 62.90), ninth on answer generation (Overall 31.90), and fifth on answer-evidence alignment (F1 79.81). A post-hoc linguistic analysis of 45 stylistic features reveals that model outputs remain 3.2 Flesch-Kincaid grade levels harder to read than clinician-authored references despite matching their word and sentence counts, suggesting readability warrants explicit optimization in clinical NLP systems. Code and prompts are available at this https URL. 

---
# PACE: Publisher-Adaptive Content Extraction via Agentic Automation 

**Authors**: Zhanlin Liu, Munirathnam Srikanth  

**Link**: [PDF](https://arxiv.org/pdf/2608.27466)  

**Abstract**: Web content extraction is essential for reliable LLM data pipelines, yet existing methods often struggle to jointly satisfy accuracy, scalability, and adaptability. General-purpose extractors can be applied broadly, but they are often brittle on publisher-specific layouts and richer extraction targets such as metadata, images, and tables. Direct LLM-based extraction offers greater flexibility, but incurs substantial cost and latency at scale, while manually engineered publisher-specific parsers can achieve high accuracy but require substantial human effort to build and maintain.
We introduce PACE, an agentic framework for learning publisher-specific extraction configurations from representative pages and user requirements. During training, PACE uses LLMs to analyze page structure and aggregate reusable extraction patterns. At inference time, the learned configurations instantiate a fixed deterministic extractor template, enabling scalable extraction without additional LLM calls.
Experiments spanning article-body, metadata, and multimodal extraction show that PACE outperforms scalable non-manual baselines while approaching the quality of manually engineered publisher-specific parsers. PACE achieves stronger extraction of article text, metadata, images, and tables, demonstrating that agentic configuration learning can automate publisher-specific extraction for LLM-ready page representations beyond article text. 

---
# The Effect of Emotional Context on Large Language Models' Endorsement of Premature Decisions: Comparing Emotional Vulnerability Across Six Commercial Models 

**Authors**: Cheolho Shin, Yoojin Han, Donghun Shin, Kunho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.27465)  

**Abstract**: As large language models (LLMs) are increasingly used for everyday decision-making advice, whether a model shifts the direction of its advice according to the user's emotional state has become an important safety problem. We test whether emotional expression increases a model's endorsement (encouragement to proceed) when a user, holding the same objective information, is overconfident about a premature decision (e.g., quitting a stable job on weak evidence). As a key control, we include a no-emotion multi-turn (neutral) condition that holds factual content and the number of conversational turns constant, isolating the effect of emotion from that of conversation length. We exposed six commercial models (top-tier and mid-tier models from OpenAI, Anthropic, and Google) to three scenarios (career change, business expansion, emigration) across three conditions (cold/neutral/distress) with six repetitions each, yielding 324 conversations, and measured endorsement strength (0-100) via an eight-item rubric-based automated scoring. Emotional expression significantly increased endorsement (neutral 18.6 to distress 31.5, +12.9 points; mixed-effects $\beta = +12.9$, $p < .001$; Cohen's d = 0.51), and this was not explained by conversation length (cold-neutral difference non-significant, $p = .083$). Critically, the vulnerability varied by individual model rather than by price tier: five of six models showed a significant emotion effect, including the top-tier flagships Gemini 3.1 Pro and GPT-5.5, while only Claude Opus showed no significant change. Results were reproduced with an independent non-Google judge model ($\rho = .89$) and agreed in rank with two human coders ($\rho = .70$). Through a controlled design that separates emotion from conversational context, we show that emotional context increases LLM sycophancy even in top-tier flagship models. 

---
# Sledgehammer or Scalpel? A Fine-grained Adaptive Framework for Implicit Hate Speech 

**Authors**: Han Wang, Yuhu Cheng, Xuesong Wang, Yi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.27462)  

**Abstract**: Unlike explicit attacks with obvious profanity, implicit hate speech hides malice within seemingly compliant expressions through metaphors and contextual hints, making its detection in online content review challenging. While existing PLM- or LLM-based methods perform well, they typically apply a single reasoning process to all samples. This overlooks fine-grained linguistic nuances and causes unnecessary computation for simpler cases. We observe that online hate speech is not monolithic but manifests in varied forms. We therefore define three fine-grained categories: Shallow, Targeted, and Context-Dependent. Accordingly, we propose Fine-grained Adaptive Implicit Hate speech Detection (FAID), a novel framework that first performs fine-grained classification and then adapts to specific categories. Specifically, for Shallow samples with surface-identifiable intents, the framework adopts lightweight prompt-tuning for rapid classification; for Targeted comments that bind malicious intent to concealed targets, we design knowledge augmentation to iteratively refine the model and reveal hidden targets; for Context-Dependent comments lacking background information, we utilize an agentic framework that automatically generates prompts to evolve context, infer missing background information and identify ambiguous malicious intents. This adaptive architecture focuses computational resources on complex implicit samples while avoiding redundant reasoning for shallow samples. Experiments on four benchmark datasets demonstrate that FAID significantly outperforms SOTA baselines. 

---
# SciReC: Diagnostic Evaluation of Multimodal, Multi-Turn Relational Reasoning with Adaptive Interaction 

**Authors**: Nilay Yilmaz, Naga Sai Abhiram Kusumba, Stella Wenxing Liu, Yezhou Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.27461)  

**Abstract**: Relational reasoning requires the process of perceptual understanding, comparing, and integrating the underlying relationships between concepts. This ability consists of multiple categories, such as analogical, structural, and cause-effect, each capturing a different aspect of higher-order understanding. To examine the performance of multimodal large language models (MLLM) on these relational inference tasks, we developed SciReC, a model-adaptive multimodal academic dialog benchmark. As the relational reasoning process involves multiple representations and various factors (visual understanding, exhibiting knowledge, and memory recall), we propose DMRA, a deficit-based diagnostic framework that quantifies the contribution of these components to identify the primary cause of unsuccessful cases. Claude 4.6 achieved the best performance on the overall relational score with 73\%, followed by GPT 5.4 with 68\%. Performance trends indicate that open-source models achieve their lowest scores on spatial relations, while proprietary models struggle more with hierarchical and sequential relations. Across domains, model performance is lowest on Astronomy and highest on Psychology. The results of DMRA reveal that relational reasoning is the primary source of error across all models, followed by memory limitations. 

---
