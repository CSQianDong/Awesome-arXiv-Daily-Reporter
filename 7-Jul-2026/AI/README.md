# LLM-as-a-Verifier: A General-Purpose Verification Framework 

**Authors**: Jacky Kwok, Shulu Li, Pranav Atreya, Yuejiang Liu, Yixing Jiang, Chelsea Finn, Marco Pavone, Ion Stoica, Azalia Mirhoseini  

**Link**: [PDF](https://arxiv.org/pdf/2607.05391)  

**Abstract**: Scaling pre-training, post-training, and test-time compute have become the central paradigms for improving the capabilities of LLMs. In this work, we identify verification, the ability to determine the correctness of a solution, as a new scaling axis. To unlock this and demonstrate its effectiveness, we introduce LLM-as-a-Verifier, a general-purpose verification framework that provides fine-grained feedback for agentic tasks without requiring additional training. Unlike standard LM judges that prompt LLMs to produce discrete scores for candidate solutions, LLM-as-a-Verifier computes the expectation over the distribution of scoring token logits to generate continuous scores. This probabilistic formulation enables verification to scale along multiple dimensions: (1) score granularity, (2) repeated evaluation, and (3) criteria decomposition. In particular, we show that scaling the scoring granularity leads to better separation between positive and negative solutions, resulting in more calibrated comparisons. Moreover, scaling repeated evaluation and criteria decomposition consistently lead to additional gains in verification accuracy through variance and complexity reduction. We further introduce a cost-efficient ranking algorithm for selecting the best solution among candidates using the verifier's continuous scores. LLM-as-a-Verifier achieves state-of-the-art performance on Terminal-Bench V2 (86.5%), SWE-Bench Verified (78.2%), RoboRewardBench (87.4%), and MedAgentBench (73.3%). Beyond verification, the fine-grained signals from LLM-as-a-Verifier can also serve as a proxy for estimating task progress. We build an extension for Claude Code, enabling developers to monitor and improve their own agentic systems. Finally, we show that LLM-as-a-Verifier can provide dense feedback for RL, improving the sample efficiency of SAC and GRPO on robotics and mathematical reasoning benchmarks. 

---
# SovereignPA-Bench: Evaluating User-Owned Personal Agents under Evolving Intent, Platform Mediation, and Consent Constraints 

**Authors**: Dylan Zongmin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.05363)  

**Abstract**: Personal agents are becoming persistent user-owned intermediaries: they remember preferences, filter platform-mediated information, use tools, and negotiate with services. Existing benchmarks evaluate tool use, web navigation, desktop control, personalization, recommendation, and evolving context, but rarely ask whether an agent preserves user sovereignty: advancing the user's current interests while respecting privacy, consent, evidence, user burden, and resistance to manipulative incentives. We introduce SovereignPA-Bench, an executable benchmark for evaluating user-owned personal agents under evolving intent, platform mediation, privacy boundaries, consent constraints, evidence requirements, and burden tradeoffs. The benchmark separates agent-visible ObservableState from evaluator-only HiddenLabels, reports component metrics for task success, alignment, privacy, consent, evidence, manipulation, burden, and auditability, and preserves paired scenario ordering for model and policy comparisons. We evaluate 120 sovereignty stress scenarios across 4 model families and 8 policy baselines, yielding 3,840 frozen-prompt trajectories with raw prompts, outputs, provider-form responses, parsed actions, recomputable metrics, hard-set analyses, qualitative cases, and a blinded 3-annotator audit over 240 items. Full-sovereign scaffolding improves sovereignty score over direct, memory-only, consent-only, evidence-only, ReAct/tool-use, safety-prompt, and judge-guard baselines while reducing privacy leakage, consent violation, over-concession, and manipulation capture. Human audit shows high agreement on privacy and consent and lower agreement on manipulation, identifying the subjective frontier of platform-persuasion judgments. These results show that personal-agent evaluation must move beyond task completion toward representative, consent-aware, evidence-grounded action. 

---
# Graph Sparse Sampling: Breaking the Curse of the Horizon in Continuous MDP Planning 

**Authors**: Idan Lev-Yehudi, Vadim Indelman  

**Link**: [PDF](https://arxiv.org/pdf/2607.05359)  

**Abstract**: Planning under uncertainty in continuous domains is essential for autonomous systems, yet computationally demanding. Tree-based search methods such as Monte Carlo Tree Search (MCTS) remain popular, but their branching structure can require sampling budgets that grow exponentially with lookahead depth in the worst case. From a tree perspective, continuous state or action spaces become especially challenging, since the planner must decide where to search in an infinite branching hierarchy. We propose Graph Sparse Sampling (GSS), an online planning algorithm that shares sampled futures across many candidate decisions, rather than sampling separate successors for each candidate action. This branch-free graph exposes large GPU-friendly batches, while using heuristics to focus computation. We prove finite-sample performance guarantees for GSS covering full-rank or low-rank generative simulators via smoothed backups, and discrete or sampled continuous action spaces. Under suitable overlap, regularity, and action-coverage conditions, these bounds have polynomial dependence on the planning horizon, formalizing when shared futures can avoid the exponential horizon dependence of tree-shaped sparse sampling. We demonstrate continuous-control simulations where GSS substantially outperforms tree-based planners on long horizons or achieves near-optimal performance, supporting no-branching graph planning as a complementary design principle for online control. 

---
# OptiAgent: End-to-End Optimization Modeling via Multi-Agent Iterative Refinement 

**Authors**: Adriana Laurindo Monteiro, Nayse Fagundes, Gabriel Mattos Langeloh, Gustavo de Oliveira Kanno, Priscila Louise Aguirre, Thiago Costa Rizuti da Rocha, Victor Leme Beltran  

**Link**: [PDF](https://arxiv.org/pdf/2607.05346)  

**Abstract**: We propose OptiAgent, a multi-agent framework that, given a natural language description of an Operations Research problem, is able to output a solver-ready mathematical formulation as well as executable code. Our architecture prioritizes the mathematical modeling step, where dedicated agents extract structures, such as decision variables and constraints, enabling iterative self-correction. We introduce a novel multi-loop validation architecture with four specialized feedback mechanisms, each targeting a distinct failure mode such as misinterpretation, structural defects, mathematical inconsistencies, validation failures, and code errors. Alongside accuracy, our modular design improves the process of solving optimization problems by improving transparency, as each agent exposes its reasoning and feedback, making the full modeling process auditable. Our framework achieves state-of-the-art performance on 3 out of 4 benchmarks across LP, MILP, and Nonlinear Programming tasks, while remaining highly competitive on the remaining dataset. 

---
# Evaluating and Understanding Model Editing for Medical Vision Language Models 

**Authors**: Guli Zhu, Chenwei Wu, Liyue Shen  

**Link**: [PDF](https://arxiv.org/pdf/2607.05310)  

**Abstract**: Model editing promises a fast, targeted way to correct post-deployment mistakes in medical vision-language models (VLMs) without costly retraining. However, existing multimodal model editing benchmarks focus on general-purpose tasks and do not reflect realistic clinical domain requirements and variability. To address this, we introduce M3Bench, a clinically grounded benchmark for multimodal model editing that evaluates whether an edit remains reliable, precise, and generalizable under the challenges of image and text variation, modality and protocol shifts, clinical knowledge composition, and temporal progression. M3Bench contains 16,276 questions spanning diverse anatomy, modalities, and specialties, and supports both single and sequential edits.
By evaluating 4 representative editors across 6 medical and general VLMs, we find that no method excels across all criteria. Gradient-based editors achieve strong transfer but suffer from catastrophic locality violations, whereas memory-based methods preserve locality but lack compositional generality and exhibit high backbone-dependent hyperparameter sensitivity. We further attribute these failures to the latent space geometry of VLMs and how different editing methods shift its landscape. Overall, M3Bench establishes a rigorous clinical stress test for multimodal model editing and offers actionable guidance for safer post-deployment adaptation. The benchmark is publicly available at this https URL . 

---
# MetaSkill-Evolve: Recursive Self-Improvement of LLM Agents via Two-Timescale Meta-Skill Evolution 

**Authors**: Zefeng Wang, Minxi Yan, Jinhe Bi, Sikuan Yan, Volker Tresp, Yunpu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2607.05297)  

**Abstract**: Recent LLM agents tackle increasingly long-horizon, open-ended tasks, and external skills, reusable procedural knowledge supplied to the agent, further extend this capability. However, a fixed, hand-authored skill is rarely optimal, and cannot adapt to the diversity of tasks an agent encounters. Self-improving agents address this by rewriting their own skill files from execution traces, yielding meaningful gains on challenging benchmarks. Yet such self-evolution remains non-recursive: it improves only the task skill (what the agent does) while the improvement procedure (how it improves) is authored once and held fixed. We introduce MetaSkill-Evolve, a two-timescale framework that makes agentic skill improvement recursive: every branch carries both a task skill $s$ and a branch-local meta-skill $m=(\psi,\sigma,\alpha,\pi,\varepsilon)$ whose five components parameterise the Analyzer, Retriever, Allocator, Proposer, and Evolver agents of the improvement pipeline. Task skills evolve on a fast loop while the meta-skill evolves on a slower one under the same pipeline applied to itself, with no additional model or objective. With all five pipeline agents sharing a single frozen backbone, MetaSkill-Evolve outperforms no-skill, static-skill, and single-level evolution baselines on three agentic benchmarks (OfficeQA, SealQA, ALFWorld), improving held-out test accuracy over the raw backbone by +23.54, +16.09, and +1.92 points respectively. 

---
# MoP-JEPA: Hard-Assigned Predictor Mixtures for Stochastic JEPA World Models 

**Authors**: Zhi Song, Ximing Xing, Zhenchao Tang, hanbo Huang, Tianxu Lv, minghao Yang, Zhongzheng Niu, He Bing, Lusheng Wang, Jianhua Yao  

**Link**: [PDF](https://arxiv.org/pdf/2607.05238)  

**Abstract**: JEPA world models predict the next latent state with a single deterministic predictor trained by latent regression. We show that this fails structurally when the environment is stochastic: at a branching transition, the regression-optimal predictor outputs the conditional mean of the successor embeddings, a point between the true next states that corresponds to no state at all. We prove this collapse for deterministic and gated mixture-of-experts predictors, and prove that MoP-JEPA's hard-assigned predictors converge instead to a quantizer of the transition distribution: one head per successor mode, enumerable in a single forward pass, which is the interface a planner consumes. On official OGBench offline data with leak-free evaluation, planning over single-predictor rollouts performs poorly ($0.02$--$0.09$ success) while planning over our predicted modes reaches up to $0.85$, ahead of deterministic, gated-MoE, and variational predictors on every task. Because multi-prediction evaluation invites coverage freeloading, a verification protocol is part of the method: an input-agnostic codebook control, a shuffled-context test, router-gated readouts, transition-precision guards, and a verified-route criterion in which the model proposes its transition graph blind and ground truth is used only to check the result. Under this criterion our method outperforms the strongest soft alternative on all three mazes ($2$--$5\times$), and the protocol identifies the remaining gap in that baseline's raw scores as routes through predicted transitions that do not exist. The same model executes in the real environment, placing second of seven against the published OGBench baselines on the hardest maze. Multimodal dynamics decide whether a JEPA world model can plan at all; a mixture of predictors with hard assignment is a minimal and verifiable fix. 

---
# EvoAgentBench: Benchmarking Agent Self-Evolution via Ability Transfer 

**Authors**: Xingze Gao, Chuanrui Hu, Hongda Chen, Pengfei Yao, Zhao Wang, Yi Bai, Zhengwei Wu, Yunyun Han, Xiaofeng Cong, Jie Gui, Yafeng Deng, Teng Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.05202)  

**Abstract**: Agent self-evolution in long-horizon LLM systems is largely procedural: useful experience is not merely stored information, but reusable procedures for searching, debugging, and verification. Yet current evaluations do not isolate this form of transfer. Agent benchmarks test single-episode task solving; memory benchmarks target information retention rather than procedural reuse. We introduce EvoAgentBench, a benchmark for agent self-evolution via Ability-guided transfer across four agentic domains: web research, algorithmic reasoning, software engineering, and knowledge work. EvoAgentBench extracts trace-grounded Abilities from agent executions, canonicalizes them into operational units, and builds domain-specific Ability Graphs linking tasks that share procedural overlap. By design, every test task is backed by verified training-side Ability support. Across a 528/267 train/test split, two scaffolds, and three backbones, curated Ability content transfers reliably across model families, but no current automatic method sustains positive gain in all settings. EvoAgentBench shifts self-evolution evaluation from aggregate accuracy comparison to fine-grained diagnosis of experience encoding, routing, and uptake. The benchmark is publicly available at this https URL. 

---
# Reason, Reward, Refine: Step-Level Errors Corrections with Structured Feedback for Physics Reasoning in Small Language Models 

**Authors**: Raj Jaiswal, Dhruv Jain, Rishabh Dhawan, Sree Krishna Uppalapati, Shin'ichi Satoh, Tanuja Ganu, Rajiv Ratn Shah  

**Link**: [PDF](https://arxiv.org/pdf/2607.05199)  

**Abstract**: Physics reasoning fails structurally in small language models: an error at any step propagates forward, corrupting every inference that follows. Limited domain knowledge, hallucination under multi-step derivation, and distributional sensitivity compound this failure. We propose a step-level reward framework that identifies the first reasoning error, generates targeted structured feedback, and trains the model to revise its solution via policy gradient with KL regularization, without exposing it to ground truth solutions as generation targets. Unlike annotation-dependent step-level methods, no preference data construction is required and the external verifier operates exclusively at training time. Across five physics benchmarks, our framework delivers accuracy gains of 17-20% over CoT prompting and 10-16% over the strongest baseline, reduces calculation errors from 56.9% to 23.5%, and reduces miscomprehension errors from 22.3% to 12.0% in the best observed cases. Conceptual errors reduce from 89.7% to 68.7%, yet persist as the hardest failure mode across all conditions. 

---
# ClassicLogic: A Knowledge-Driven Benchmark of Classic Puzzle Games for Evaluating Compositional Generalization 

**Authors**: Mahnoor Shahid, Hannes Rothe  

**Link**: [PDF](https://arxiv.org/pdf/2607.05185)  

**Abstract**: Compositional generalization, the ability to understand and produce novel combinations of known components, remains a fundamental challenge for modern artificial intelligence. While few benchmarks exist, many focus on linguistic tasks and lack complex, explicit compositional structures. We introduce ClassicLogic, a new benchmark suite designed to evaluate an agent's ability to learn and compose problem-solving strategies. The benchmark consists of four classic logic puzzles: Sudoku, KenKen, Kakuro, and Futoshiki. Its core innovation is a hierarchical, explicit knowledge base for each game, where complex solving strategies are formally defined as compositions of simpler, foundational strategies. This structure allows for fine-grained evaluation of an agent's reasoning capabilities, from learning basic rules to applying multi-step compositional strategies to solve puzzles of increasing, mathematically validated difficulty. The open-source benchmark provides a challenging new testbed for advancing neuro-symbolic and other advanced AI reasoning systems. 

---
# Rethinking On-Policy Self-Distillation for Thinking Models 

**Authors**: Simran Kaur, Narutatsu Ri, Yinghui He, Liam Fowl, Sanjeev Arora  

**Link**: [PDF](https://arxiv.org/pdf/2607.05184)  

**Abstract**: Self-distillation is a promising recipe for self-improvement in language models. In this setting, a model can serve as its own teacher when given privileged information, such as a solution to a math problem. This seems especially appealing for thinking models, which can use test-time reasoning to absorb the privileged information. Surprisingly, we show that privileged self-distillation degrades thinking models on long reasoning traces: across five Qwen3 and OLMo thinking models evaluated on AIME24, AIME25, and HMMT25, privileged-context distillation causes a relative drop of up to 17% in avg@16 accuracy. The degradation scales with the amount of privileged context withheld from the student and is most pronounced at long rollout budgets, where thinking models otherwise obtain their largest gains. This failure mode is not specific to self-distillation: on-policy distillation (OPD) improves thinking models, but privileged OPD reverses these gains. Our diagnostics link this failure mode to how privileged teacher context reshapes learning at high-entropy forking positions, where multiple continuations remain plausible and may lead to different reasoning paths. Privileged context lowers fork rates in thinking-model rollouts but not in instruction-model rollouts. This leads to an interesting dichotomy, where privileged context can help instruction-tuned models but hurts stronger thinking models. The effect is visible when the student begins a self-correction branch, where privileged OPD penalizes sampled reconsideration tokens that vanilla OPD supports. Thinking models trained with a privileged teacher produce fewer verification, backtracking, and hedging markers, even after length normalization. These findings indicate that self-distillation for strong thinking models requires attention to token-level signal, especially around correction and reasoning steps. 

---
# CP-WSP: A Declarative CP-SAT Framework for Configurable Multi-Constraint Workforce Scheduling 

**Authors**: Vipul Patel, Anirudh Deodhar, Dagnachew Birru  

**Link**: [PDF](https://arxiv.org/pdf/2607.05177)  

**Abstract**: Workforce scheduling is an NP-hard combinatorial optimization problem requiring simultaneous satisfaction of labor regulations, coverage requirements, employee preferences and operational objectives. Existing CP formulations typically model simplified instances with 6-12 constraints at shift-level granularity and critically lack explicit support for: mandatory break scheduling with midpoint placement control; acuity weighted workload equity; sub-shift temporal granularity enabling demand-driven staffing; inter-week schedule stability; and cross-midnight shift patterns common in 24-hour operations. This paper presents CP-WSP: a declarative CP-SAT framework enforcing 14 hard constraints as mathematically inviolable requirements (zero regulatory violations by construction) while optimizing 15 soft objectives through a unified weighted penalty function -- all configurable via a JSON specification with no code changes required. Key contributions include: a shift-window variable decomposition enabling mandatory break scheduling with centrality control; acuity-weighted workload equity; multi-granularity temporal resolution from 30 minutes to 2 hours; inter-week schedule stability; a grid-offset preprocessing technique for cross-midnight shifts; and a reproducible 36-configuration benchmark suite for community comparison. Evaluated on INRC-II benchmarks at both hourly and shift-level granularity and on 36 synthetic configurations. 

---
# AgentGym2: Benchmarking Large Language Model Agents in De-Idealized Real-World Environments 

**Authors**: Zhiheng Xi, Dingwen Yang, Jiaqi Liu, Jixuan Huang, Honglin Guo, Baodai Huang, Tinggang Chen, Qi Zhang, Zhonghang Lu, Chenyu Liu, Jiajun Sun, Jiazheng Zhang, Dingwei Zhu, Xin Guo, Junzhe Wang, Zhihao Zhang, Yuming Yang, Junjie Ye, Minghe Gao, Dongrui Liu, Jiaming Ji, Guohao Li, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2607.05174)  

**Abstract**: Language agents, i.e., LLM agents, progress rapidly and are increasingly deployed in production environments. This trend underscores the urgent need for rigorous and realistic evaluations. However, most existing benchmarks evaluate agents in simplified, idealized settings. They typically rely on pre-packaged tool interfaces, overlook critical steps, and assume inputs are clean and fully specified. Consequently, they understate the difficulty of real deployments, where uncertainty and noise are ubiquitous and agents must proactively explore the environment to uncover new tools. To bridge this gap, we present AgentGym2, a new evaluation framework with task instances grounded in real-world end-to-end working demands. Beyond reasoning and planning, it measures agents' ability to execute end-to-end procedures, discover tools via exploration, compose tools for unseen tasks, and remain robust to noisy and underspecified information. Experiments on 15 proprietary and open-source models show that even SOTA systems like Gemini and GPT-5 struggle on AgentGym2, revealing a substantial gap between the capability of current agents and the demands of real-world applications. 

---
# The Changing Role of Symbolic Methods in Artificial Intelligence 

**Authors**: Jun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.05168)  

**Abstract**: Why do intelligent systems need to perform explicit symbolic reasoning? Computer science has traditionally regarded symbolic reasoning as a defining component of intelligence. Yet the remarkable success of modern foundation models raises a fundamental question: if increasingly capable AI systems can operate with little explicit symbolic reasoning, what role do symbolic methods actually play?
This article argues that explicit symbolic reasoning is not a fundamental property of intelligence, but a computational consequence of operating on simplified models of reality. We propose the Compression Principle: every computational model is a simplified representation of reality, and explicit symbolic reasoning compensates for information omitted during model construction. From this principle, we derive the Modeling--Reasoning Trade-off: as computational models preserve richer representations of the world, the need for explicit symbolic reasoning correspondingly decreases. This perspective provides a unified explanation for both the historical success of symbolic methods and the remarkable effectiveness of modern foundation models.
Paradoxically, the same development makes symbolic methods increasingly important for humans. As intelligent systems become more capable and more opaque, symbolic representations increasingly serve as interfaces through which humans specify requirements, verify behavior, regulate autonomous systems, and establish trust. We therefore argue that the future of symbolic methods lies not primarily as the computational engine of intelligent systems, but as the symbolic interface between increasingly capable AI systems and the humans who build, govern, and depend upon them. 

---
# DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation 

**Authors**: Xin Cheng, Xingkai Yu, Chenze Shao, Jiashi Li, Yunfan Xiong, Yi Qian, Jiaqi Zhu, Shirong Ma, Xiaokang Zhang, Jiasheng Ye, Qinyu Chen, Chengqi Deng, Jiping Yu, Damai Dai, Zhengyan Zhang, Yixuan Wei, Yixuan Tan, Wenkai Yang, Runxin Xu, Yu Wu, Zhean Xu, Xuanyu Wang, Muyang Chen, Rui Tian, Xiao Bi, Zhewen Hao, Shaoyuan Chen, Huanqi Cao, Wentao Zhang, Anyi Xu, Huishuai Zhang, Dongyan Zhao, Wenfeng Liang  

**Link**: [PDF](https://arxiv.org/pdf/2607.05147)  

**Abstract**: Speculative decoding accelerates Large Language Model (LLM) inference by decoupling draft generation from target verification. While recent parallel drafters efficiently propose long token sequences in a single forward pass, they suffer from rapid acceptance decay due to a lack of inter-token dependencies. Furthermore, indiscriminately verifying these extended blocks wastes critical batch capacity on tokens with high rejection risks, severely degrading throughput in high-concurrency serving systems. We introduce DSpark, a speculative decoding framework that unifies high-throughput parallel generation with adaptive, load-aware verification. To maintain draft quality, DSpark utilizes a semi-autoregressive architecture, coupling a parallel backbone with a lightweight sequential module, to introduce intra-block dependency modeling and mitigate suffix decay. To optimize system efficiency, DSpark employs confidence-scheduled verification, dynamically tailoring the verification length for each request based on estimated prefix survival probabilities and engine-specific throughput profiles. On offline benchmarks across diverse domains, DSpark substantially improves the accepted length over state-of-the-art autoregressive and parallel drafters. When deployed within the DeepSeek-V4 serving system under live user traffic, DSpark successfully mitigates verification waste. Compared to the established production baseline (MTP-1), DSpark accelerates per-user generation speeds by 60 to 85 percent at matched throughput levels. More importantly, by preventing severe throughput degradation under strict interactivity constraints, it enables performance tiers that were previously unattainable, shifting the Pareto frontier of our serving system. 

---
# TacReasoner: A Dynamic Tactile-Language Framework for Interactive Reasoning in Real-World Scenarios 

**Authors**: Kailin Lyu, Di Wu, Long Xiao, Jianning Zeng, Jianwei He, Chang Lin, Lianyu Hu, Lin Shu, Jie Hao, Ce Hao  

**Link**: [PDF](https://arxiv.org/pdf/2607.05131)  

**Abstract**: Among the five primary human senses, tactile is arguably the most fundamental to survival, as it enables the perception of physical contact and interaction in real-world environments. In this paper, we explore two key challenges of integrating tactile sensing into intelligent systems for multimodal reasoning: (i) insufficient modeling of dynamic tactile signals, which restricts reasoning over temporally evolving properties, and (ii) hallucination in tactile foundation models caused by the absence of explicit reasoning mechanisms, leading to unstable real-world inference. To address these challenges, we propose TacReasoner, a dynamic tactile-language framework for interactive reasoning in real-world scenarios. First, TacReasoner incorporates a Dynamic-aware Tactile Encoder to enhance the perception and representation of dynamic tactile signals. More importantly, we introduce TouchCoT-10k, the first tactile chain-of-thought dataset for structured reasoning over tactile inputs. Upon it, we establish DynTac-Bench to systematically evaluate dynamic tactile perception and real-world commonsense reasoning. Experimental results demonstrate that TacReasoner achieves competitive performance against state-of-the-art models across multiple datasets. Notably, despite using only 7B parameters, TacReasoner outperforms the 14B VTV-LLM model on most subtasks, highlighting its effectiveness and efficiency in tactile commonsense reasoning. 

---
# ASSEMCAD: Production-Ready CAD Assembly Generation from Natural Language 

**Authors**: Yurui Dong, Shu Zou, Siqi Li, Nianchen Deng, Hongbin Zhou, Xuemeng Yang, Pinlong Cai, Licheng Wen, Xinyu Cai, Botian Shi  

**Link**: [PDF](https://arxiv.org/pdf/2607.05123)  

**Abstract**: Recent advances in large language models and programmatic CAD have significantly improved Text-to-CAD generation for individual parts. However, production-ready mechanical assembly generation remains largely unsolved. Unlike single-part modeling, assemblies require coordinated reasoning over multiple components, functional interfaces, assembly relations, engineering principles, and physical consistency. Consequently, directly generating executable CAD code is insufficient for constructing mechanically valid and reusable assemblies. We present AssemCAD, an axiom-grounded framework for production-ready CAD assembly generation from natural language. Instead of representing an assembly as monolithic CAD code, AssemCAD first constructs an axiomatic Assembly Specification consisting of typed parts, geometry-backed ports, executable mates, and engineering axioms. Each assembly relation is explicitly grounded in one or more engineering principles, making the resulting specification interpretable, reusable, and verifiable. To realize this specification, AssemCAD introduces a port- and mate-based CAD assembly library that executes symbolic assembly relations through deterministic mate transformations and validates declared interfaces using concrete B-Rep geometric evidence. Built on this representation and library, AssemCAD further supports on-demand synthesis of reusable parametric component factories for both standard and open-world geometries. Experiments on AssemBench show that AssemCAD substantially improves assembly preservation and physical validity over code-centric CAD generation baselines, while generalizing across different foundation-model backbones. By combining axiom-grounded assembly reasoning with deterministic geometric execution, AssemCAD extends Text-to-CAD from isolated part generation toward production-ready mechanical assembly design. 

---
# Diffusion-Guided Uncertainty-Aware Delayed Policy Optimization 

**Authors**: Junqi Tu, Zejiao Liu, Fangfei Li, Yang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2607.05064)  

**Abstract**: Reinforcement learning in real world environments often suffers from severe performance degradation due to delayed feedback. Existing approaches typically mitigate performance degradation caused by observation delays by constructing augmented states or predicting the true states. However, these methods often overlook the inherent discrepancy between delayed state and true states induced by stochastic MDP. We theoretically prove the existence of such a discrepancy and show that it leads to the degradation of the optimal policy. To address this challenge, we propose Diffusion Guided Uncertainty Aware Delayed Policy Optimization (DUPO). Our method explicitly models the relationship between delayed state message and the current state using a diffusion model, and leverages the resulting discrepancy estimates to weight delayed policies. Extensive experiments on continuous robotic control tasks with multiple stochastic delays demonstrate that DUPO consistently outperforms existing methods and remains effective even under long and random delay scenarios. 

---
# Toward Trustworthy Large Language Model Agents in Healthcare 

**Authors**: Hadi Hasan, Safaa Salman, Adam Tai Abou Dargham, Ammar Mohanna, Ali Chehab  

**Link**: [PDF](https://arxiv.org/pdf/2607.05055)  

**Abstract**: Healthcare appointment scheduling remains a persistent operational bottleneck, driven by manual coordination, fragmented legacy systems, and high administrative overhead. These inefficiencies constrain provider availability and degrade patient access to care. This paper presents CareConnect, a safety-first conversational agent for healthcare logistics automation that leverages large language model (LLM) function calling, retrieval-augmented generation (RAG), and layered deterministic safety guardrails. The system orchestrates eight domain-specific tools to support appointment booking, modification, cancellation, and facility information retrieval, while enforcing strict scope constraints that prohibit medical advice or diagnosis. Safety-critical situations are handled through deterministic short-circuit mechanisms for emergency detection and medical intent refusal. We evaluate CareConnect on a comprehensive benchmark of 680 task-oriented scenarios spanning end-to-end workflows, multi-turn interactions, and edge cases. Experimental results demonstrate a 91.8% task completion rate with a median per-request latency of 2.2 seconds, 96.0% safety compliance on the dedicated safety-critical evaluation subset, and an average operational cost of $0.0324 per appointment, yielding a significant cost reduction compared to manual human scheduling. These findings show that carefully scoped and rigorously safeguarded LLM-based agents can reliably automate complex healthcare operational workflows while maintaining safety guarantees and achieving substantial cost efficiency. The source code and system implementation are publicly available at this https URL. 

---
# Quantum-Inspired Harmonic Decision Models: A Computational Framework for Music Generation 

**Authors**: Josef Pavlíček, Petra Pavlíčková, Martin Molhanec  

**Link**: [PDF](https://arxiv.org/pdf/2607.05007)  

**Abstract**: This paper introduces a quantum-inspired computational framework for harmonic decision-making in music. The proposed approach formulates harmonization as an optimization problem within a structured combinatorial space, where multiple candidate chord sequences are evaluated under interacting musical constraints. The model combines an interference-based harmonization stage with a classical optimization procedure grounded in tonal harmony. The quantum-inspired component enables the parallel consideration of multiple harmonic alternatives, while the classical stage refines the resulting sequences to ensure structural coherence and stylistic plausibility.
The framework is evaluated on selected musical examples, including Autumn Leaves and It's a Long Way to Tipperary. Quantitative analysis shows that the optimization stage significantly reduces chord density, increases harmonic stability, and improves functional organization. At the same time, expert evaluation highlights the importance of stylistic context, demonstrating that increased harmonic complexity is not always perceived as more natural. The results suggest that harmonic generation can be interpreted as a structured decision-making process in a constrained search space. The proposed approach provides a computational model that integrates domain-specific knowledge with an interference-based search mechanism.
Although preliminary, this work indicates that quantum-inspired methods may offer a useful framework for modeling complex decision processes in creative domains such as music. The proposed framework contributes to ongoing research on quantum-inspired models of cognition and decision-making in complex biological and creative systems. 

---
# STAPO: Selective Trajectory-Aware Policy Optimization for LLM Agent Training 

**Authors**: Qiuyi Qi, Tian Liang, Mutian Bao, Jinjian Zhang, Dongnan Liu, Wei Zhou, Linjian Mo, Ming Kong, Jie Liu, Feng Zhang, Qiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.04963)  

**Abstract**: Reinforcement Learning (RL) is the dominant paradigm for training Large Language Model (LLM) agents on long-horizon tasks. However, sparse and delayed rewards often lead to trajectory neglect, in which agents lose focus on the task goal and interaction history at intermediate steps. Prior work has explored step-level supervision using Shannon-entropy-based uncertainty signals, which conflate inherent state complexity with agent confidence and therefore provide unreliable estimates of decision reliability. To address this issue, we propose normalized entropy, which measures confidence deviations relative to an agent's average behavior under a given state, thereby strengthening the association between low-quality actions and trajectory neglect. Building on this insight, we introduce Selective Trajectory-Aware Policy Optimization (STAPO), a hierarchical group-based RL framework. STAPO leverages normalized entropy to locate outlier steps associated with trajectory neglect and optimizes them via a joint mechanism of trajectory-aware reward and trajectory-independent penalty, enhancing trajectory awareness while preserving training stability. Extensive experiments on ALFWorld, WebShop, and Search-Augmented QA demonstrate that STAPO achieves state-of-the-art performance while substantially alleviating trajectory neglect, validating its effectiveness and robustness for agentic tasks. 

---
# Medi-Gemma: A Hybrid Clinical Decision Support System Integrating Deterministic EMR Analytics and Retrieval-Augmented Generation 

**Authors**: Mohammed Saim Ahmed Quadri, Yunzhe Xue, Justin W. Ady, Usman Roshan  

**Link**: [PDF](https://arxiv.org/pdf/2607.04907)  

**Abstract**: Deploying Large Language Models (LLMs) in high-stakes clinical settings remains limited by structural hallucinations, weak deterministic reasoning over tabular patient data, and omissions in vector retrieval. This paper presents the architecture and validation of Medi-Gemma, a Clinical Decision Support System (CDSS) for wound pathology triage and workflow automation. The platform introduces a decoupled framework that separates clinical perception from data orchestration while preserving traceable reasoning. Medi-Gemma uses a multi-stage pipeline coordinated by a centralized ClinicalOrchestrator. Data requests are handled without generative inference by a DataManager that cleans unstructured Electronic Medical Record (EMR) files through type coercion. Natural language queries are processed by a hierarchical IntentRouter, which routes requests to deterministic analytics paths executed by a PandasQueryEngine or to patient-specific reasoning managed by a ClinicalRAGEngine using a CPU-optimized vector store. A key contribution is the Ground Truth Injection Module, which intercepts patient-specific queries, extracts numeric identification tokens, queries the structured dataframe via Pandas, retrieves the latest validated clinical state, and embeds this snapshot as an overriding context block in the LLM prompt before generation. Safety compliance is enforced by a deterministic ProtocolManager that maps clinical terminology to fixed evidence-based risk pathways, while a SafetyVerifier phrase filter prevents output rule violations. Validation shows that this architecture eliminates semantic context drift, prevents database compilation crashes, and improves factual adherence to backend clinical repositories. These results support Medi-Gemma as a safer pattern for LLM-based clinical decision support where structured data fidelity, retrieval grounding, and deterministic safeguards are essential. 

---
# CARL: Constraint-Aware Reinforcement Learning for Planning with LLMs 

**Authors**: Qiuyi Qi, Jinjian Zhang, Mutian Bao, Tian Liang, Guocong Li, Dongnan Liu, Wei Zhou, Jie Liu, Ming Kong, Linjian Mo, Feng Zhang, Qiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.04854)  

**Abstract**: Despite their strong reasoning capabilities and extensive world knowledge, Large Language Models (LLMs) frequently generate plans that violate task constraints, undermining their reliability in real-world applications. This deficiency arises from a lack of systematic mechanisms to incorporate constraint information during the generation process. While existing approaches attempt to mitigate this by relying on external tools or task decomposition, they fail to enhance the model's intrinsic constraint awareness. To address this, we propose Constraint-Aware Reinforcement Learning (CARL), a novel RL framework designed to strengthen LLMs' intrinsic focus on constraints. CARL introduces a constraint-aware reward by comparing the model's output distributions under constrained and unconstrained inputs, encouraging constraint focus and penalizing neglect. Compatible with various RL frameworks and requiring no external solvers or top models, CARL enables scalable, end-to-end constraint-aware planning. Extensive experiments on BlocksWorld, TravelPlanner, and T-Eval demonstrate that CARL significantly outperforms standard Reinforcement Fine-Tuning (RFT) baselines and state-of-the-art reasoning models, exhibiting a markedly increased focus on constraints. 

---
# AgenticPD: A Stage-Aware Agentic Framework for Physical Design QoR Optimization 

**Authors**: Shuo Ren, Zijin Cheng, Yaohui Han, Libo Shen, Leilei Jin, Wanting Tian, Rongliang Fu, Chao Wang, Bei Yu, Tsung-Yi Ho  

**Link**: [PDF](https://arxiv.org/pdf/2607.04758)  

**Abstract**: Physical design quality-of-results~(QoR) optimization is hard and expensive. Choices made at one stage can help or hurt later stages. Each evaluation requires a costly EDA run through the full flow. While existing methods still treat optimization as flat parameter tuning or a LLM-based script generation task, we present AgenticPD, a stage-aware agentic framework for physical design QoR optimization. Instead of re-running the full flow after every trial, AgenticPD is organized around the stage boundaries of the physical design flow, where a Judge Agent navigates the search and stage-specialized agents make local decisions within their own stage using stage-local tools. Additionally, the agent harness in AgenticPD provides structured observations, execution history, and agent context management. As a result, the system can branch from prior intermediate states and reuse checkpoints to continue the optimization procedure, and every candidate is evaluated at the post-route signoff. Across these baselines, AgenticPD achieves strong post-route timing while remaining competitive in power and area. 

---
# FM-ChangeNet: Learning Change through Pathwise Feature Transport 

**Authors**: Roie Kazoom, George Leifman, Genady Beryozkin  

**Link**: [PDF](https://arxiv.org/pdf/2607.04750)  

**Abstract**: We present FM-ChangeNet, a pathwise-supervised framework for change detection that reformulates bi-temporal reasoning as continuous transport in feature space rather than static endpoint comparison. Given encoded pre and post-temporal representations, we construct intermediate latent states and learn a time-conditioned velocity field $\hat{v}_\theta(z_t,t)$ along the transformation trajectory. This pathwise formulation constrains the predictor over a continuum of intermediate states, providing a denser and less ambiguous supervision signal than conventional endpoint-only segmentation and enabling the model to capture temporal evolution explicitly. The learned velocity field is not only a transport mechanism but also an interpretable representation of change: its magnitude serves as a spatially localized change cue that helps distinguish true structural variation from nuisance effects such as illumination shifts and spatial misalignment. We develop a hierarchical multi-scale architecture with cross-temporal alignment, time-conditioned coarse-to-fine flow decoding, and a unified objective that couples flow supervision, trajectory consistency, spatial regularization, and segmentation loss. Experiments on remote sensing benchmarks show that the proposed framework produces more structured and robust change representations while achieving state-of-the-art performance. 

---
# FORGE: Research-Trajectory Hijacking Attacks on Deep Research Agents 

**Authors**: Yue Pan, Ziheng Zhang, Junxiang Lei, Changhao Jia, Qingyi Si, Hongcheng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2607.04718)  

**Abstract**: Deep research agents decompose open-ended queries into subtasks, retrieve web evidence over multiple rounds, and synthesize long-form reports. This workflow creates a planning-layer poisoning surface: adversarial documents that enter the retrieval pool can steer follow-up questions and turn a local injection into report-level contamination. We present FORGE (Fabricated Orchestrated Reasoning chain for aGent Exploitation), a two-level attack that combines intra-document reasoning fabrication with inter-document chain coordination to hijack subtask planning. We further introduce the PRISM metric, which weights infected report claims by cognitive type, and Root Query Anchoring, a lightweight defense that ties recursive follow-up generation to the root query. Across 25 queries, Network FORGE reaches 26.4% PRISM with five injected documents and exhibits depth migration, in which recursive synthesis shifts poisoned content from overt framing into factual premises. On the 10-query defense subset, RQA (Root Query Anchoring) reduces PRISM from 38.5% to 18.3%. 

---
# Integrated Altruistic and Fairness Preference Induces Advanced Mutual Cooperation in Sequential Social Dilemmas 

**Authors**: Yu Wei, Yukiko Ogura, Yoshiyuki Ohmura, Ildefons Magrans de Abril, Hoshinori Kanazawa, Yasuo Kuniyoshi  

**Link**: [PDF](https://arxiv.org/pdf/2607.04710)  

**Abstract**: Inducing cooperation among distributed agents is still a difficult problem in the field of multi-agent reinforcement learning (MARL), particularly in social dilemma situations. There, individual interests are misaligned with the common good and individual rationality leads to suboptimal group outcomes. In contrast, humans are able to achieve cooperation with one another in such situations. A common explanation for such cooperative behavior is that individuals have social preferences. In order to achieve cooperation in MARL, we design a new utility function integrating altruistic preferences (incentive for other's reward) and fairness preferences (incentive for equality) from social psychology and behavioral economics, namely, Altruistic and Fairness Preference (AFP), a reward-sharing mechanism which converts one's own and other's rewards to incentives for cooperative behavior. We performed comparative experiments with standard RL and inequity aversion agents in two challenging sequential social dilemma games, and showed that AFP agents successfully achieved mutual cooperation with more collective rewards and higher equity than the baselines. To further understand the progression of AFP during training, we subsequently explore the effects of altruistic preferences and fairness preferences on agents' behavior. The results suggest that altruistic preferences encourage agents to contribute to the public goods, and fairness preferences induce mutual behavior between agents. 

---
# Formal Disco: Scalable Open-Ended Generation of Formally Verified Programs 

**Authors**: Gabriel Poesia, Simon Henniger, Tzu-Han Hsu, Yilun Du, Nada Amin  

**Link**: [PDF](https://arxiv.org/pdf/2607.04631)  

**Abstract**: The cost of producing code is rapidly diminishing with increasingly capable AI agents, while quality assurance of generated programs has not kept pace. Formal verification provides the strongest possible guarantees, but the ability of AI models to work with verification-aware languages is hindered by the scarcity of human-written examples of programs in those languages. To tackle this prevalent data scarcity issue, we propose Formal Disco: a distributed system for coordination of LLM-based workers that can be easily applied to open-ended synthetic data generation at scale. We use Formal Disco to share tasks and programs between three classes of workers: "initiators", which read random READMEs from open-source repositories and documentation snippets to sketch a related verified program, "fixers" which take compiler and verifier feedback and attempt to resolve issues, and "extenders" that take working programs and propose patches to expand them. Formal Disco records all agent-generated traces and uses them both for initial distillation from a stronger model as well as self-improvement. We also propose a principle of maximum entropy for synthetic program generation, and use entropy maximization via iterative supervised fine-tuning to learn to generate increasingly diverse programs over time. We release large datasets of synthetic verified programs in three languages - Dafny, Verus, and Frama-C -, and fine-tune open models for verification-relevant tasks, often matching or exceeding the performance of Claude Opus 4.5. Overall, our work offers a path to create synthetic data at scale for formal reasoning domains and overcome the long-standing data barrier. 

---
# MRMS: A Multi-Resolution Memory Substrate for Long-Lived AI Agents 

**Authors**: Jizhizi Li, Amy Shi-Nash  

**Link**: [PDF](https://arxiv.org/pdf/2607.04617)  

**Abstract**: Long-lived AI agents require continuity across interactions, but continuity cannot be obtained by simply extending the prompt window. An agent must preserve useful prior experience, retrieve it selectively, distinguish personal context from external evidence, and revise memory when the underlying situation changes. We propose an architectural memory substrate organized along two orthogonal axes: a representational axis spanning structured records, vector representations, and graph relations; and a temporal axis spanning short-term traces, medium-term abstractions, and long-term semantic commitments. Its key design constraint is synchronized structured-vector-graph memory: structured records govern eligibility, vector representations support recall, and graph relations adjudicate support, contradiction, and supersession before gated context projection. Its central claim is that reliable personalization is a memory design problem: useful memory is structured, selectively exposed, continuously consolidated, and epistemically labeled rather than stored as undifferentiated conversation history. Beyond the framework, we instantiate MRMS as a lightweight prototype implementing structured records, vector retrieval, temporal policies, and graph-based revision. The prototype exercises the core substrate mechanisms through pre-generation memory selection, revision, boundary enforcement, and evidence attribution under controlled long-lived interaction scenarios with explicit evidence requirements. 

---
# Governed Individuation: Cryptographically Decoupling an Agent's Learning from Its Authority 

**Authors**: Xue Qin, Simin Luan, Cong Yang, Zhijun Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.04613)  

**Abstract**: Autonomous agents are moving from sandboxed text generators to operators of code, data, and physical infrastructure, and they increasingly learn while deployed. This reopens a question that alignment techniques answer only probabilistically: after an agent has adapted in the field, is the running system still confined to what its operator authorised? Here we show that confinement can be guaranteed as an invariant of the agent's execution architecture rather than a probabilistic outcome of its training. Governed individuation binds an agent at boot to a cryptographically frozen identity digest, and routes every action through a gate defined over the semantic effect of the action rather than its name. We prove that no amount of learning, skill acquisition, or self-induced governance abstraction can widen the agent's permitted authority without an operator-signed change to its identity; the guarantee holds even when the agent induces its own safety principle and that principle is wrong. Empirically, in an open-ended tool-use benchmark where a large action space rules out name-based blocking, ungoverned software agents under reward pressure attempt to tamper with their own evaluation at a task-dependent rate that reaches every run on the hardest task, whereas the gate reduces executed forbidden effects to zero as a verified property of the construction while preserving task success. An adversarial evaluation of monitors of increasing semantic depth shows false-allows falling from 75% (name-based gating) to zero (dynamic effect tracing), and refusal history transfers compliance to held-out red-line families. Trust in a deployed learning agent shifts from a wager on its continued alignment to a check anyone can run at boot. 

---
# Attention Limited Reward Learning 

**Authors**: Wenqian Xing  

**Link**: [PDF](https://arxiv.org/pdf/2607.04590)  

**Abstract**: Pairwise human comparisons are a primary interface through which modern AI systems learn human preferences. RLHF and related alignment pipelines typically model such comparisons with Bradley--Terry log-odds, where choice probabilities are governed by latent reward differences. This paper examines what this assumption misses through a reduced-form model motivated by rational inattention, in which each label is generated by a low-capacity evaluation channel. The model separates two forms of ambiguity that standard reward modeling tends to conflate: a comparison may be difficult because the two candidates are genuinely close in value, or because the relevant distinction is hard to detect under limited attention. We show that limited attention can fundamentally distort what pairwise comparisons reveal. In particular, passive comparison data cannot generally distinguish reward, attention, and default tendencies, and heterogeneous attention can make standard Bradley--Terry reward modeling recover misleading rankings. Our analysis shows that learning is governed not by the raw number of labels, but by the amount of attended information each label carries. A case study on human votes over language-model pairs from Chatbot Arena exhibits the predicted signature, a cyclic component of the comparison data that exceeds sampling noise and that no scalar reward can represent; a second case study on perceptual comparisons shows that response times and gaze carry gap information that the labels do not. This perspective suggests that human feedback should be treated not as direct revealed preference, but as an attention-limited measurement process: a weak preference signal may reflect hidden evaluation difficulty rather than genuine indifference. 

---
# Detecting Answer-Driven Reasoning in LLM-Based Educational Tutors via Truncated Chain-of-Thought Auditing 

**Authors**: Bonan Shen, Dingyan Shang, Youting Wang, Tao Ning  

**Link**: [PDF](https://arxiv.org/pdf/2607.04572)  

**Abstract**: Large language model (LLM) tutors often produce fluent step-by-step explanations, but a correct and pedagogically formatted response does not guarantee that the answer was derived from the student-facing problem. In realistic tutoring systems, the model may also have access to teacher notes, answer keys, rubrics, or retrieved solution artifacts. We study whether such private answer information can make tutor explanations answer-driven: the final answer is behaviorally available before the written explanation has justified it. Using Truncated Reasoning AUC Evaluation (TRACE), which probes how early a chain-of-thought prefix can pass a verifier, we evaluate 1000 GSM8K test problems under three paired tutoring contexts: question-only, correct answer-key, and wrong answer-key. At fixed fractions of each generated explanation, we force the model to answer immediately and verify the response against the gold numeric answer. With Qwen2.5-3B-Instruct, answer-key access raises median TRACE AUC from 0.375 to 0.900 and makes the gold answer available at the first 10% prefix in 997 of 1000 cases. The effect remains strong on the 746 examples where both question-only and answer-key explanations end with the correct answer. These results support truncated CoT auditing as a lightweight process-level diagnostic for answer-driven reasoning in math tutoring explanations. 

---
# Heaviside Continuity of Rolling Coefficients for Eliminating Epistemic Entropy in Large Language Models 

**Authors**: MY Pitsane, Hope Mogale  

**Link**: [PDF](https://arxiv.org/pdf/2607.04562)  

**Abstract**: Large language models (LLMs) generate fluent outputs that can be wrong. Unlike humans, who often exhibit cues when providing false information, LLMs produce errors that are difficult to detect because autoregressive decoding provides no mechanism for verifying intermediate reasoning before state progression. We introduce Heaviside Continuity of Rolling Coefficients (HCRC), a verification-first execution framework that reformulates inference as predicate-gated state transitions governed by a Heaviside Gate. HCRC combines model confidence with independent verification signals from a parallel worker architecture, allowing execution to advance only when predefined correctness predicates are satisfied. This prevents invalid intermediate states from propagating, reducing epistemic entropy without modifying the underlying model. We evaluate HCRC on software-engineering and reasoning tasks across thirteen proposers from four providers. On capable proposers, the gate reduces the false-completion rate (FCR) from 4--7% to 0% while remaining latency-competitive and, in some settings, faster than the unwrapped model. On weaker proposers, it converts false completions into honest halts instead of corrupting downstream state. Beyond benchmarking, HCRC has operated for months as the production control plane of an agentic coding environment, authorizing file mutations, verification-driven progress reporting, and memory compaction. These results establish HCRC as a general framework for verification-driven LLM execution, showing that reliable reasoning can be achieved through principled execution control rather than model scale alone. 

---
# Measuring Harness-Induced Belief Divergence in Multi-Step LLM Agents 

**Authors**: Haiwen Yi, Xinyuan Song  

**Link**: [PDF](https://arxiv.org/pdf/2607.04528)  

**Abstract**: Software-agent benchmarks usually report whether an agent solves a task, but the agent reaches that outcome through a harness that controls what it sees, which actions it can take, which failures are repaired, which states are verified, and which evidence is logged. We show that this harness can change the agent's multi-step beliefs even when the task, environment, and base LLM are fixed. We introduce a belief-rollout diagnostic that elicits structured K-step trajectories over progress, risk, recoverability, constraints, failure mode, uncertainty, future success, repair cost, and next action under alternative harnesses. We define a cross-harness belief divergence and decompose it into an arrival term for immediate interface shifts and a growth term for horizon-dependent belief changes. On controlled coding tasks and public-benchmark stress tests, blocked actions, compressed repairs, selective verification, and cost-aware evidence pruning often preserve terminal success while changing the beliefs that drive later decisions. We further introduce BIWM, a no-training protocol that canonicalizes observations, logs censored branches, expands repair traces, records verification masks, executes risky branches in shadow, and aligns belief trajectories across harness views. The results suggest that harness design is an experimental variable in agent evaluation, not an implementation detail. Our code is available at this https URL. 

---
# VLA Grounder: Language-Conditioning Space Optimization for Black-Box VLA Models 

**Authors**: Damir Shodiev, Aleksei Staroverov, Nikita Kachaev, Alexey K. Kovalev, Aleksandr I. Panov  

**Link**: [PDF](https://arxiv.org/pdf/2607.04517)  

**Abstract**: Vision-Language-Action (VLA) models are commonly treated as end-to-end action policies conditioned on natural-language task descriptions. In practice, however, their behavior often depends sharply on how the instruction is phrased, suggesting that language is not merely a task label but an optimizable conditioning input. We study whether frozen VLA policies can be improved by optimizing language space rather than updating action weights. Our method introduces a language-conditioning space policy that translates a human instruction into a short VLA-grounded command using object appearance, spatial relations, and target-grounding cues. The language-conditioning space policy is initialized with a failure-derived command-space prior and optimized with reinforcement learning from sparse task-completion rewards, while the downstream VLA remains fully frozen. This yields language-conditioning space optimization: RL discovers which VLA-grounded commands best elicit successful behavior from the frozen action policy. Experiments on RL4VLA and VL-Think show that language-conditioning space optimization improves success on instruction-sensitive, symbolic, and multi-object manipulation tasks, demonstrating that language can serve as an optimizable variable for a robot foundation models. Website: this https URL 

---
# Compressing the Validation Bottleneck: An Agentic Self-Driving Lab for Scientific Discovery 

**Authors**: Kyunghoon Hur, Chihun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2607.04508)  

**Abstract**: Agentic AI-for-Science can automate ideation, planning, and analysis, but final validation still depends on real experiments. A self-driving lab (SDL) can execute those experiments, yet the loop still has bottlenecks: the agent may spend too many rounds on low-value experiments, or each round may require a high-cost experiment. We target these two physical bottlenecks with one agent. First, a prior-aware agentic DOE loop uses domain knowledge and past results to propose feasible and informative next experiments, reducing trials-to-target. Second, a cost-aware surrogate agent predicts high-cost, high-resolution measurements from low-cost, low-resolution measurements. It chooses between a high- and a low-cost measurement based on the predicted uncertainty. We examine these directions in the biology and materials domains, respectively. Together, under a single agent, these components aim to accelerate the SDL loop by reducing both the number of loops and the cost per experiment. 

---
# Why Pure Reasoning is Not Enough: Nature as the Source of Mathematical Innovation 

**Authors**: Charanjit S. Jutla, Vimal Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2607.04505)  

**Abstract**: We advance the hypothesis that human mathematical reasoning, constrained by both the undecidability and the computational intractability of even modest logical fragments, relies fundamentally on pattern matching from domains external to pure deduction. The most prolific reservoir of such patterns is the natural world, whose physical laws and biological systems have undergone billions of years of ``pre-computation'' and already exhibit surprisingly innovative solutions. To ground this claim, we trace the history of the Fourier transform and relevant mathematics, from the vibrating string controversy to the hear equation and subsequent formalisms prevalent in mathematics. At each critical juncture, a physics problem forced the acceptance or creation of a mathematical tool that pure formal reasoning failed to anticipate or, worse, human reasoning had resisted.
We further survey the landscape of logical complexity, from NP-hard propositional satisfiability to the non-elementary decision-procedures for monadic second-order theories, to demonstrate that even when a logic is decidable, the resources required for worst-case deduction are astronomically prohibitive. We argue that these barriers make physics-inspired pattern matching not just a historical accident but a cognitive necessity. Finally, we draw the consequence for artificial intelligence: if pure reasoning is constitutively insufficient, then any system aiming at human-level mathematical creativity must embed a vast store of cross-domain patterns rather than rely on deduction alone. This furnishes a principled justification for the enormous scale of contemporary large language models. 

---
# ResearchStudio-Idea: An Evidence-Grounded Research-Ideation Skill Suite from ML Conference Outcomes 

**Authors**: Qihao Zhao, Yangyu Huang, Yalun Dai, Lingao Xiao, Jianjun Gao, Xin Zhang, Wenshan Wu, Scarlett Li, Yang He, Yan Lu, Yap Kim Hui  

**Link**: [PDF](https://arxiv.org/pdf/2607.04439)  

**Abstract**: Large language models have made research ideation increasingly accessible, yet effective idea development requires more than generating candidate directions. Researchers must ground a problem in current literature, identify meaningful bottlenecks, differentiate from existing solutions, and evaluate risks before committing to implementation. We present ResearchStudio-Idea as a reusable skill suite for this first mile of research ideation. The suite includes Paper-Search, a standalone multi-source literature search skill; Scoop-Check, a standalone prior-art collision checker for novelty claims; and IdeaSpark, the end-to-end skill that composes evidence grounding, pattern-guided generation, collision retrieval, audit, and idea-card rendering into one workflow. IdeaSpark is constructed from a corpus of 1,947 machine learning conference papers collected from ICLR, ICML, and NeurIPS between 2021 and 2025, including Oral papers, a separately tracked high-citation subset, and rejected submissions. Analysis of these outcomes reveals 31 recurring ideation sub-patterns, consolidated into 15 reusable ideation patterns. Each pattern is operationalized as a structured card containing research contexts, bottleneck types, differentiation strategies, supporting precedents, and common failure modes. Given a research problem and an evidence bundle, IdeaSpark evaluates evidence readiness, reconstructs the surrounding research context, identifies unresolved bottlenecks, selects relevant patterns, instantiates one candidate direction, retrieves potentially conflicting prior work, and performs outcome-informed auditing. This workflow transforms reusable ideation patterns into traceable research proposals. Blind automated-judge evaluations show that IdeaSpark consistently produces stronger research proposals than no-skill and generic-skill baselines while maintaining competitive novelty. 

---
# Agent Step Value: State-Transition Measurement with State-Grounded LLM Evaluators 

**Authors**: Andrew Zhang, Chengzhan Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.04419)  

**Abstract**: Most agent evaluations collapse a multi-step trace into a final answer, a success flag, or a trajectory-level score. These aggregates obscure the diagnostic question developers need most: which action changed the state in a useful direction? We introduce Agent Step Value (ASV), a state-transition measurement framework that scores each observed action by the change it induces in a state-grounded evaluator's distribution over fixed candidate outcomes. ASV renders redacted before/after state projections, uses a stateless LLM evaluator to assign candidate log scores, and reports both gold-free belief diagnostics and offline oracle validation metrics. A label-free rationale pass separates evaluator deliberation from one-token option scoring, preserving candidate likelihoods while exposing leakage and floor-score events. On 100 reviewed open-QA evidence-seeking tasks with live PubMed retrieval, a partially live DeepSeek actor, and DeepSeek log-probability scoring, ASV evaluates 1,100 steps and 2,200 states. Under the fixed-layout rationale-conditioned protocol, mean gold-margin gain is -2.335 (trajectory-bootstrap 95\% CI [-3.395, -1.272]), entropy movement is 0.000, and mean Bayesian surprise is 2.693. ASV therefore localizes constructive and destructive belief pivots that final-answer scores and entropy-only step metrics miss. We release the standalone ASV Eval toolkit. 

---
# LLM-as-a-Tutor: Policy-Aware Prompt Adaptation for Non-Verifiable RL 

**Authors**: Yujin Kim, Namgyu Ho, Sangmin Hwang, Joonkee Kim, Yongjin Yang, Sangmin Bae, Seungone Kim, Jaehun Jung, Se-Young Yun, Hwanjun Song  

**Link**: [PDF](https://arxiv.org/pdf/2607.04412)  

**Abstract**: Reinforcement learning (RL) for non-verifiable instruction following increasingly relies on LLM judges with prompt-specific rubrics as reward signals. While recent methods adapt these rubrics to the evolving policy during training, the training prompts themselves remain static, drawn from fixed corpora. This static approach often results in a critical misalignment between prompt difficulty and policy capability, leaving the judge unable to recover a discriminative reward signal when prompts fail to elicit quality variance among rollouts. To address this misalignment, we introduce LLM-as-a-Tutor, a framework that extends the LLM's role from judge to tutor: a single model serves as an examiner that pairwise-compares policy rollouts to detect non-challenging prompts, and as a generator that appends atomic constraints to them. This append-only design monotonically raises difficulty in step with the policy's capability, producing a self-calibrating training signal without external difficulty schedules. On three complex instruction-following benchmarks, our method consistently outperforms both policy-unaware baselines and prior policy-adaptive methods that adapt rubrics or rewrite prompts, suggesting prompt adaptation as a missing axis of policy-awareness in non-verifiable RL. 

---
# MechMath Agent Team: LLM Driven Agents for Mathematical Research 

**Authors**: Yichuan Cao, Ruichen Qiu, Junqi Liu, Jiaqi Wang, Dakai Guo, Ruyong Feng, Lihong Zhi, Xiao-Shan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2607.04394)  

**Abstract**: AI reasoning has become a central focus in contemporary artificial intelligence, largely driven by the success of large language models. However, mathematical research, which is characterized by non-linear derivation paths, rigorous logical requirements, and protracted exploration cycles, poses severe challenges for existing reasoning systems. To overcome these limitations, we present the MechMath Agent Team (MMAT), which is a large language model driven agent designed to serve as a co-pilot throughout the full cycle of mathematical research. We design a tripartite Harness Architecture that decouples system responsibilities into Control, Execution, and Augmentation planes, thereby reconciling rigorous logical control with the agility demanded by open-ended research. Building upon this framework, we instantiate three specialized agents: a Knowledge Base Manager, a Natural Language Prover, and a Formal Language Prover, all operating in a closed loop to produce formally certified mathematical proofs. We evaluate MMAT on open problems in Number Theory, Algebraic Complexity Theory, Differential Algebra, Operator Algebra, and Inequalities. Across a two-month deployment, 11 problems have been solved, demonstrating its capacity to act as a co-pilot throughout the entire research cycle. The contributions are threefold: a general decoupled Harness Architecture for multi-agent mathematical reasoning, its concrete instantiation in the MMAT system, and empirical validation on a diverse suite of open problems. 

---
# Decentralized Aggregation of LLM Predictions via Wagering Mechanisms 

**Authors**: Yuhong Luo, David M. Pennock, Xintong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.04389)  

**Abstract**: It is increasingly common to aggregate predictions from multiple LLMs, each with domain expertise or access to private tools and data, to improve collective prediction performance. In decentralized settings, aggregation weights need to be determined without access to models' private information and should remain robust to strategic reporting. We propose a family of advantage-aligned wagering mechanisms for LLM aggregation (WALLA), in which each model reports a prediction and a learned wager, and predictions are aggregated using wagers as weights. WALLA introduces a leave-one-out baseline into the net payout function, yielding three desirable properties: (1) dominant-strategy incentive compatibility of prediction under arbitrary belief structure, (2) advantage--wager alignment, where the optimal wager is proportional to the model's expected score advantage, and (3) prediction-agnostic wager optimization, enabling decentralized learning of wager policies without requiring optimal predictions. We further instantiate two mechanism variants that trade off normality and no-arbitrage while maintaining a bounded worst-case deficit for the mechanism. Experiments on question-answering and forecasting benchmarks across heterogeneous models and private-information settings show that WALLA matches centralized aggregation methods in predictive performance, while simultaneously achieving decentralized learning, advantage-aligned aggregation weights, uncertainty awareness, and incentive-compatible prediction. 

---
# Nemotron-Labs-3-Puzzle-75B-A9B: Compressing Hybrid MoE LLMs 

**Authors**: Akhiad Bercovich, Talor Abramovich, Daniel Afrimi, Shay Aharon, Nir Ailon, Vladimir Anisimov, Omer Ullman Argov, Maor Ashkenazi, Tomer Asida, Nave Assaf, Tomer Bar Natan, Alexander Bukharin, Grzegorz Chlebus, Marcin Chochowski, Eric Chung, Mohammad Dabbah, Carlo del Mundo, Ewa Dobrowolska, Ido Galil, Yaniv Galron, Amnon Geifman, Yonatan Geifman, Izik Golan, Alex Gronskiy, Tomasz Grzegorzek, Netanel Haber, Lior Kadoch, Grzegorz Karch, Tomer Keren, Abhinav Khattar, Amir Klein, Tugrul Konuk, Roi Koren, Daniel Korzekwa, Shaun Kotek, Konstantinos Krommydas, Itay Levy, Ofri Masad, Yoav Miron, Pavlo Molchanov, Shahar Mor, Zach Moshe, Saurav Muralidharan, Najeeb Nabwani, Besmira Nushi, Mostofa Patwary, Omri Puny, Johannes Rausch, Tomer Ronen, Sepehr Sameni, Itamar Schen, Elad Segal, Daniel Serebrenik, Ido Shahaf, Soumye Singhal, Daniil Sorokin, Sharath Turuvekere Sreenivas, Marta Stepniewska-Dziubinska, Ali Taghibakhshi, Nima Tajbakhsh, Oren Tropp, Dor Tzur, Anna Warno, Yi-Fu Wu, Michal Zawalski, Jiaqi Zeng, Yian Zhang, Ran Zilberstein, Amit Zuker, Ran El-Yaniv  

**Link**: [PDF](https://arxiv.org/pdf/2607.04371)  

**Abstract**: We present Nemotron-Labs-3-Puzzle-75B-A9B, a compressed variant of Nemotron-3-Super optimized for interactive deployment. We designed the model to maximize server throughput under high user throughput constraints. In interactive serving workloads on a single 8xB200 node, Puzzle-75B-A9B achieves approximately 2x higher server throughput than Nemotron-3-Super at matched user throughput constraints. In ultra-long-context deployment on a single H100 GPU, the compressed model increases 1M-token concurrency from 1 request to 8 requests. Puzzle-75B-A9B is constructed using a multi-stage pipeline that combines the Iterative Puzzle compression framework with knowledge distillation, reinforcement learning, quantization, and a Multi-Token Prediction head. The compression process jointly optimizes heterogeneous MoE pruning, active parameter budget, and Mamba pruning to improve inference efficiency while preserving model quality. We evaluate Puzzle-75B-A9B on a broad suite of reasoning, coding, multilingual, long-context, and agentic benchmarks. Despite substantial compression, the model retains strong downstream accuracy relative to the parent model across a wide range of tasks. These results demonstrate that large hybrid MoE models can be substantially optimized for deployment efficiency while maintaining strong downstream capability. 

---
# Server-side Anti-cheat in FPS games for Aimbot detection using Deep learning and Machine learning 

**Authors**: Siddhesh A. Dhinge, Shubham G. Sukum, Harsh S. Ranjane, Ruturajsingh R. Rajput, Jyoti H. Jadhav  

**Link**: [PDF](https://arxiv.org/pdf/2607.04336)  

**Abstract**: Modern video games are becoming more complex day by day. Most of these modern games are multiplayer first-person shooter (FPS) games. The rising popularity of FPS games emphasizes the need to combat cheating for fair and enjoyable gaming. As the number of players using cheating techniques like aimbots, wallhacks, and speed hacks is also increasing, we need a way to detect players who are using cheating tools to gain an unfair advantage over regular players. In this system, we focus exclusively on detecting aimbot cheats. Players who use aimbot cheats generally do not prioritize other aspects of the game. To distinguish between regular and cheating players, we identify specific features encompassing time-series data such as aim velocity, number of shots, distance to target, and more, along with behavioral data such as utility usage, player movement, and other gameplay patterns. Utilizing these features, we construct a server-side aimbot detection classifier named 'YAACS'. YAACS comprises a parser, a deep learning model, and intermediary connection utilities designed for integration with the game server. The proposed system achieves a classification accuracy of 88.6% with a false positive rate of 0.97% using a Stacked LSTM with Dense layers trained on sequences of 128 ticks (Tick Delta Negative=56, Tick Delta Positive=24), outperforming the Decision Tree baseline which achieves a higher accuracy of 96.2% but at a false positive rate of 2.68%, 2.76x worse than the best LSTM configuration. These results demonstrate that incorporating temporal context through sequence modelling is critical for minimising false accusations in FPS cheat detection. 

---
# Do GUI Agents Believe Their Eyes? Diagnosing State-Belief Reliance on Pixels versus Structure 

**Authors**: Guijia Zhang, Harry Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.04334)  

**Abstract**: Multimodal GUI agents read an interface through two redundant channels: the rendered pixels of a screenshot and a serialized structure such as a DOM or accessibility tree. Before acting, an agent forms a belief about the current interface state, but existing benchmarks score task success, element grounding, or attack resistance and do not ask whether that belief is drawn from the pixels. We formalize visual state reliance, the attribution of a state belief to pixels, structure, or priors, and measure it with paired single-channel interventions over 310 real web, mobile, and desktop probes. Every probe is scored by deterministic forced choice, with no model-generated item and no model judge. Our central metric is the Perception-Fusion Gap, the fraction of probes a model perceives correctly yet resolves toward structure under conflict. Across five models from three vendors, textual state beliefs defer to structure while image-only accuracy stays near ceiling, and Perception-Fusion Gap is positive for every model; non-text identity, by contrast, stays largely pixel-bound. The substitution is specific to the serialized-text and indexed-action channel, and coordinate-action agents are largely immune. For textual conflicts, a white-box ablation traces the effect to a single copied structural value, and in two live environments the conflict drives wrong actions and real task failure. Visual state reliance therefore gives a measurable diagnostic of whether agent state beliefs are visually grounded, and the errors it exposes propagate to actions. 

---
# HAS-Bench: Evaluating LLM-Based Human-Agent Systems under Configurable Human Participation 

**Authors**: Yaozu Wu, Wei-Chieh Huang, Jizhou Guo, Dongyuan Li, Renhe Jiang, Henry Peng Zou, Chunyu Miao, Shanghao Li, Weizhi Zhang, WeiWei Ye, Yankai Chen, Meng Zhang, Xue Liu, Philip S.Yu  

**Link**: [PDF](https://arxiv.org/pdf/2607.04329)  

**Abstract**: Large language models increasingly operate in settings where humans are active collaborators rather than passive task providers. We introduce HAS-Framework, a graph-based framework that represents humans and LLM-powered agents as first-class participants with explicit roles, permissions, communication paths, and action authority. Building on this framework, HAS-Bench evaluates Human-Agent Systems under configurable human participation across agency levels, interaction channels, and persona policies. The benchmark measures both task outcomes and process-level collaboration behavior, including clarification quality, feedback utilization, control calibration, safety, initiative, and interaction cost. Experiments across six domains show that human participation can substantially improve task completion and failure recovery, but the gains depend on when, how, and by whom human input is exercised. 

---
# Agentic SABRE: An Uncertainty-Aware Neuro-Symbolic Multi-Agent Framework for Adaptive Ransomware Detection 

**Authors**: Henry Kabuye, Biju Issac, Jeyamohan Neera  

**Link**: [PDF](https://arxiv.org/pdf/2607.04292)  

**Abstract**: Ransomware has evolved into a complex, adaptive, and fast-moving adversary category in which static signatures and monolithic classifiers fail to generalise under concept drift, evasion, and behavioural polymorphism. In this paper, we present Agentic SABRE (Semantic-Behavioural Arbitration for Ransomware Evaluation), an uncertainty-aware, neuro-symbolic, multi-agent framework for adaptive ransomware detection. SABRE fuses semantic, representation-based evidence with behavioural, time-window forensic telemetry and employs Monte Carlo Dropout inference to quantify epistemic uncertainty for each agent. We introduce a decision-layer orchestrator that performs risk- and uncertainty-aware triage using two interpretable thresholds: a risk score and an uncertainty budget. High-confidence, high-risk samples are automatically contained, while uncertain or borderline cases are escalated to human analysts, establishing a flexible computational contract between autonomous response and analyst oversight. To support auditability and trust, SABRE integrates post-hoc explainability mechanisms, including gradient saliency, permutation importance, and counterfactual analysis, enabling both local and global interpretation of agent decisions. Extensive evaluation on RDset and RanSMAP demonstrates that Agentic SABRE preserves perfect discrimination on saturated semantic datasets, with AUC equal to 1.0, while improving robustness under weak behavioural signals. It achieves up to a 4.9 percent relative reduction in false escalations at equal recall while maintaining calibrated predictive uncertainty. Counterfactual analysis further shows that semantic and behavioural decisions can be reversed with bounded perturbation cost, indicating stable and interpretable decision boundaries. 

---
# Shortcut Learning in Legal Judgment Prediction: Empirical Evidence from the UK Employment Tribunal 

**Authors**: Joe Watson, Joana Ribeiro de Faria, Marcus Tomalin, Måns Magnusson, Huiyuan Xie, Hao Tian Yeung, Felix Steffek  

**Link**: [PDF](https://arxiv.org/pdf/2607.04261)  

**Abstract**: Current Legal Judgment Prediction (LJP) is constrained by its reliance on post-hoc judicial materials, increasing the likelihood that models perform retrospective classification rather than true forecasting. This paper empirically investigates shortcut learning in this context by studying claim-level outcome prediction in UK Employment Tribunal (UKET) decisions. Using a corpus of 33,158 individual claims, we predict outcomes from claim texts and LLM-extracted case summaries, evaluating models ranging from interpretable TF-IDF-based classifiers to black-box LLMs. While headline predictive performance figures appear strong, we demonstrate that such performance in LJP systems trained on post-hoc judicial text can be driven by the retrospective nature of the source material. Stratifying the test data by human judgments of leakage reveals that performance increases where outcome-revealing cues are embedded in the narrative. Moreover, a model trained on just the 4% of features identified as leakage achieves high performance, outperforming human experts. These findings substantiate concerns that LJP performance may be exaggerated by linguistic artefacts. Yet this vulnerability is not fatal to the research agenda. Instead, post-hoc judgments might be treated as potentially contaminated texts, requiring active auditing. Retraining models after masking leakage features results in only a negligible reduction in Macro-F1. Hence, while models will opportunistically exploit shortcuts when available, they remain capable of extracting useful predictive signals when these artefacts are removed. 

---
# Progress- and Reliability-Oriented Group Policy Optimization for Agentic Reinforcement Learning 

**Authors**: Mingxuan Fan, Peiyang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.04242)  

**Abstract**: Group-based reinforcement learning (RL) has become an effective paradigm for improving large language model agents on long-horizon interactive tasks. To obtain finer-grained policy updates than trajectory-level optimization, recent work has moved toward step-level group-based RL, where intermediate steps are grouped and compared within a rollout batch. However, step-level advantage estimation is sensitive to how groups are formed: grouping by broad state keys improves coverage but may compare actions taken under different histories, while enforcing historical consistency yields fairer comparisons at the cost of fragmented groups and missing peer-comparison signal. In this paper, we propose ProGPO (Progress- and Reliability-Oriented Group Policy Optimization), a learned-critic-free method for context-consistent step-level learning. ProGPO keeps exact-prefix action comparison, and complements sparse peer comparisons with transition credit derived from rollout-based state potentials. To estimate these potentials reliably, ProGPO combines semantic expansion with inverse-variance fusion across history depths. We evaluate ProGPO on two challenging agentic tasks, ALFWorld and WebShop, with Qwen2.5-1.5B-Instruct. Results show that ProGPO improves over matched agentic RL baselines under comparable computational overhead, and additional Qwen2.5-3B-Instruct experiments further test the scalability of the proposed method. 

---
# Biological Motifs for Agentic Control 

**Authors**: Bogdan Banu  

**Link**: [PDF](https://arxiv.org/pdf/2607.04240)  

**Abstract**: The transition of Large Language Models (LLMs) from passive generators to autonomous agents has introduced significant challenges in reliability, security, and state management. Current agentic architectures are often constructed ad-hoc, prone to hallucination cascades, infinite loops, and prompt injection attacks. This paper argues that many of these failure modes can be analyzed using control motifs long studied in systems biology, provided the comparison is made at the level of typed interfaces and coordination structure rather than literal biological mechanism.
We develop a typed interface correspondence between Gene Regulatory Networks and agentic software systems using polynomial functors and wiring diagrams. Five biological motifs are mapped to composable software design patterns: Coherent Feed-Forward Loops for noise suppression, Adaptive Immunity for layered security, Mitochondrial Signaling for resource governance, Endosymbiosis for neuro-symbolic integration, and Morphogen Diffusion for spatially varying coordination. An epistemic topology layer derives Kripke-style knowledge operators from the wiring diagram's observation structure and proves four predictive theorems for multi-agent scaling.
The core contributions are: (1) the Agentic Operad, a typed syntax for agent composition with provable error suppression bounds for feed-forward topologies; (2) an epistemic topology with four theorems (error amplification, sequential penalty, parallel acceleration, and tool density scaling) whose qualitative predictions are consistent with published multi-agent benchmarks; and (3) a six-layer progression from structure through development, grounded in autonomous learning frameworks and convergence proxies from the empirical literature. A reference implementation with 1,813 tests and 116 examples illustrates practical feasibility. 

---
# Unsupervised Features Mining via Activation Geometry 

**Authors**: Amit LeVi, Elad David, Max Fomin  

**Link**: [PDF](https://arxiv.org/pdf/2607.04222)  

**Abstract**: Interpretability methods aim to reveal the features represented inside large language models (LLMs). Many existing methods begin with labeled examples of a human-defined concept that may reflect human biases, and then identify how that concept is represented within the model, for example in its activation space or through other decomposition methods. We introduce \emph{Mining via Activation Geometry} (MAG), a simple unsupervised framework for extracting reasoning features from model activations by prepending the same natural-language instruction $Q$ to every input $p$, where $Q$ defines the reasoning feature of interest, such as ``Can this object be found in the desert?'' or ``Is this prompt malicious?'' We measure how the instruction changes the model's internal representation using $m(Q \mid p) - m(p)$ at a single readout point. We explore eight different MAGs. The extracted reasoning features predict the models' own world understanding and judgment, can be approximated into a single activation direction, we found that some features are more linearly represented and some less, this linear representation, which is vector steering, can change the LLMs' decisions through activation steering by injecting reasoning features. Finally, we use the same method to select the best training datasets for prompt-injection classifier probes: while similarity between ordinary activations is almost unrelated to downstream performance, RFD-based similarity achieves $94.7\%$ Top-1 and $100\%$ Top-2 accuracy. 

---
# Agentic IoT: Architectures, Applications, and Challenges Toward the Internet of Agents 

**Authors**: Rümeysa Hilal Sevinç, Bahaeddin Türkoğlu, İbrahim Kök  

**Link**: [PDF](https://arxiv.org/pdf/2607.04219)  

**Abstract**: The integration of AI into Internet of Things (AIoT) systems has gradually transformed them from passive data collection infrastructures into intelligent systems capable of anomaly detection, predictive maintenance, classification, forecasting, and optimization. However, most existing solutions still rely on task-specific models that infer from sensor data; thus, system-wide capabilities such as real-time reasoning, adaptive planning, autonomous coordination, learning, tool use, and contextual decision-making remain limited. This paper examines Agentic IoT as a next-generation cognitive IoT paradigm that integrates the perception, reasoning, planning, learning, and action capabilities of autonomous AI agents with cyber-physical systems. Agentic IoT aims to transform IoT from data-centric sensing and inference infrastructures into distributed cognitive agent ecosystems operating across the device/edge-fog-cloud continuum. The paper first grounds this transition as a paradigm shift and positions Agentic IoT in relation to AIoT, edge intelligence, multi-agent systems, and the Internet of Agents. It then systematically reviews current studies, presents a holistic architectural framework, discusses domain-specific application potential, and identifies key technical, operational, and research challenges together with future research directions. 

---
# A Clustering-Based Framework for Identifying Suspicious Trading Patterns in Capital Market 

**Authors**: Asif Zaman, Romona Magdalene Sarkar, Sabiha Khair Ohi, Iftekharul Mobin  

**Link**: [PDF](https://arxiv.org/pdf/2607.04184)  

**Abstract**: Market manipulation is the dubious practice of manipulating stock prices in order to make a quick profit, which truly degrades confidence on trading platforms. We implemented an unsupervised fraud-detection toolkit that begins with K-Means++ clustering to address this issue. A dataset of roughly one million financial transactions from 2012 to 2024 is used. In order to identify fraudulent trades and categorize them using market practice heuristic thresholds, the study suggests a clustering-based pipeline. The method highlights 2.02% of trades as suspicious where 51.10% clearly indicate spoofing, 0.10% indicate pump and dump, 0.55% indicate insider trading, 1.43% indicate a fake breakout, and 46.83% are unclassified. Despite the lack of ground truth, the model's performance is confirmed by a Silhouette Score of 0.561. 

---
# Language models guide symbolic equation discovery by controlling search 

**Authors**: Zikai Xie, Wenmei Li, Man Luo, Jun Jiang, Linjiang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.04156)  

**Abstract**: Scientific equation discovery must combine broad domain priors with strict numerical testing. Symbolic regression supplies numerical grounding but faces a combinatorial search space, whereas many language-model systems ask the model to propose or select formulas directly. We test a different division of labour. We compare role specifications in which the language model acts as equation author, candidate decider or search controller, alongside end-to-end language-model and purely numerical baselines. In the controller setting we propose here, implemented as LLM-PySR, language models specify variables, operators, transformations and search depth; symbolic regression enumerates and fits expressions; and deterministic metrics govern retention. Across 74 AI-Feynman equations and seven complex formula-recovery tasks, search control achieved the strongest observed balance of accuracy, complexity, stability and cost. On an independent battery dataset, LLM-PySR identified a compact piecewise-linear relation between early voltage-curve displacement and cycle life. The results suggest that language models should shape hypothesis exploration rather than decide which equations survive. 

---
# Forethought: Verifiable Reasoning from Neurosymbolic Primitive Programming 

**Authors**: Vishvesh Bhat, Jay Vaghasiya, Emmanuel Anaya Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2607.04096)  

**Abstract**: Current agentic workflows usually involve decomposing user requests into sequences of tool calls with correctly resolved parameters, the results of which are processed through reasoning traces in the language model's context window. The prevailing route to improve such reasoning is test-time scaling, which trains models to search over long chains of thought; but the resulting capability is entangled in model weights, is not verifiable step-by-step, and is costly at inference. We present Forethought, a neurosymbolic reasoning system that instead treats reasoning as an explicit, verifiable program, that builds from a library of symbolic and neural primitives which are composed through a domain-specific language. The result are reasoning programs, which are concrete representations of the model's work, and as such can be inspected and modified before deployment. Instantiated as a tool-calling execution kernel and evaluated across five benchmarks, Forethought improves base-model accuracy by about 30% relative and outperforms vanilla prompting, reinforcement learning scaffolds, and prompt-evolution methods, enabling small models to match or exceed frontier models capabilities. In a direct comparison, a non-reasoning model augmented with Forethought competes with a dedicated reasoning model while requiring roughly three orders of magnitude less post-training investment, and remains model-agnostic and auditable. 

---
# PLACEMEM: Toward a Compute-Aware Memory Plane for Lifelong Agents 

**Authors**: Sukanta Ganguly  

**Link**: [PDF](https://arxiv.org/pdf/2607.04089)  

**Abstract**: Lifelong agents need more than larger context windows and better retrieval. They need memories that can persist, evolve, and be corrected without forcing the serving stack to recompute the same history on every turn or silently reuse stale runtime state. We present PLACEMEM as a systems position on lifelong-agent memory, instantiated by an executable control-plane prototype. The central claim is that agent memory should be represented as versioned capsules that unify semantics, provenance, validity, and reusable runtime state under one correction-aware identity. In the current prototype, capsules drive prompt-level text retrieval, KV-aware routing, and cascading invalidation over live streamed backends; prospective layer-frontier replay is intentionally framed as a deeper integration agenda rather than a claimed engine feature. We describe a vLLM-first prototype with persistent capsule state, concurrency-safe invalidation, an OpenAI-compatible routing sidecar, a typed metadata contract, and a benchmark harness that measures live first-token latency, reuse, and post-correction behavior. The result is both an executable artifact that demonstrates correction-aware control-plane behavior today and a concrete roadmap for replay-aware serving integration in future lifelong-agent systems. 

---
# What is Left for Us? Second Scholarship Against the Degradation of Research by AI 

**Authors**: Claudio Novelli, Luciano Floridi  

**Link**: [PDF](https://arxiv.org/pdf/2607.04049)  

**Abstract**: We argue that generative AI can degrade research by eroding the very practices through which scholarly judgement is formed and academic trust is built. As constitutive conditions for the production and validation of knowledge, these practices cannot be reduced to the final outputs of research, which is what AI so effectively simulate. Accordingly, when researchers delegate central tasks of inquiry to systems like Large Language Models, they may stop enacting these practices and, with them, lose access to the formation they provide. An individual research output generated by AI may even appear improved but the researcher behind it fails to develop. Against this risk, merely keeping humans in the loop as prompters or quality checkers of AI outputs is insufficient to preserve research as a site of intellectual formation. What is needed instead is a renewed commitment to research as a lived practice in which judgement is formed gradually, often through frictions, and participation in a scholarly community. We defend it because it rests on four sources and warrants of research that cannot be automated: tacit knowledge, personal commitment, socialisation, and deep reading. This practice enacts what we call second scholarship, by which we understand the reappropriation of scholarly craft, chosen out of a critical experience of what generative AI can and cannot do. What cannot and should not be delegated becomes what research communities must value and answer for. This is what is left for us. 

---
# Explainable AI for Screening Abuse-Related Trauma in Bangladeshi Children: A Training-Free Multimodal Framework Evaluated on Noise-Aware Synthetic Data 

**Authors**: Salma Hoque Talukdar Koli, Fahima Haque Talukder Jely  

**Link**: [PDF](https://arxiv.org/pdf/2607.04010)  

**Abstract**: Bangladesh has an estimated 1.17 mental-health professionals per 100,000 population and only six child psychiatrists nationwide. No Bengali-language, culturally adapted tool exists for early screening of abuse-related psychological trauma in children. We present ShishuRaksha AI, a decision-support (not diagnostic) framework that fuses four screening modalities: validated questionnaires (SDQ, CPSS), Bengali narrative text, House-Tree-Person (HTP) drawing features, and facial affect. The fusion is training-free and clinically weighted, uses cross-modal attention, and includes a single-modality override rule. Every risk score is explained through clinically weighted, perturbation-based additive attribution and rendered as a bilingual (Bangla/English) report with referral routing to national child-protection services (OCC, DSS, NMHH) under the Children Act 2013. No clinical dataset of abused children can be collected ethically at this stage, so we introduce a noise-aware synthetic benchmark (500 cases, 116 positive [23.2%], four deliberate noise layers, literature-grounded HTP priors) and evaluate tree-ensemble surrogates of the fusion design (facial channel excluded) under 5-fold stratified cross-validation. The fused model reaches an AUC of 0.874 [0.834-0.908], against 0.756 [0.705-0.803] for an SDQ-only baseline, with ablation, operating-point, subgroup, and calibration analyses. We state all limitations openly, including synthetic-only data, no held-out set, text-feature circularity, and an urban-rural subgroup gap. This work is a feasibility study and a design contribution toward ethically deployable child-protection screening in low-resource settings. 

---
# Online Linear Programming for Multi-Objective Routing in LLM Serving 

**Authors**: Zixi Chen, Yinyu Ye, Zijie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2607.03948)  

**Abstract**: We study the online routing problem in large language model serving, where requests arrive sequentially and must be dispatched to parallel decode workers under tight batch-size and KV-cache constraints. Unlike widely used routing heuristics that are not tied to explicit service-level objectives (SLOs) and offer limited control over latency-throughput trade-offs, we introduce a multi-objective optimization framework that formulates routing as an online linear programming with interpretable decision rewards. We apply an efficient bid-price control policy based on the online linear programming that admits requests when their SLO-weighted benefit exceeds their shadow prices. To meet millisecond decision requirements, we develop a warm-started, projected first-order updates that track the evolving dual shadow prices online with predictable runtime. We integrate our router into the Vidur simulator and demonstrate substantial improvements over standard baselines across multiple SLO regimes, including end-to-end latency, time-to-first-token, throughput, and tail performance. A big picture from our result: a science-based approach outperforms others based on heuristics. 

---
# Harness-Aware Self-Evolving: Co-Evolving Model Weights, Harness, and Task Solutions 

**Authors**: Haochen Luo, Yi Huang, Sichun Luo, Fengyuan Liu, Lei Li, Zefa Hu, Junlan Feng, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.03935)  

**Abstract**: Self-evolving frameworks usually optimize task solutions while treating the surrounding harness as fixed. We introduce Harness-Aware Self-Evolving (HASE), an agentic reinforcement-learning framework in which a single model can generate task solutions or edit selected harness components in a multi-turn action space. HASE enables a single Qwen3-8B model to match the text-classification performance of a GPT-OSS-120B model that uses Claude Code as the harness proposer. In alpha factor mining, HASE outperforms the reported GPT-OSS-120B baseline. HASE also repairs imperfect evaluation components and converges to state-of-the-art performance in circle-packing algorithm discovery. These results show that HASE improves the harness and the solution through one unified agentic process. 

---
# Evaluating LLM Uncertainty in Long-Form Generation Using Deterministic Ground Truth 

**Authors**: Ido Amit, Ido Galil, Ran El-Yaniv  

**Link**: [PDF](https://arxiv.org/pdf/2607.03870)  

**Abstract**: As LLMs generate increasingly long outputs, effective uncertainty estimation must identify errors at fine-grained levels rather than discard entire responses. While such methods exist, evaluating uncertainty at any resolution (token to an entire generation) is challenging and highly sensitive to label imperfections, making zero-noise benchmarks essential; yet, long-form generation benchmarks tend to rely on fallible labels rather than deterministic ground truth. We introduce Single-answer Atomic Long-form Target (SALT), a benchmark of six procedurally generated tasks with single deterministic long textual ground truths, enabling unit-level evaluation of correctness, calibration, and ranking without external judges. Equipped with SALT, our analysis of 50+ LLMs reveals key insights: We identify which confidence functions dominate each uncertainty aspect and show that confidence ranking largely breaks at atomic resolution, even when clearer separability emerges at coarser line-level units. SALT further enables controlled atom-level interventions throughout generation, revealing two separable drivers of future errors: propagation from corrupted prefixes, dominated by global context correctness, and bounded degradation from increasing answer-context length. Finally, we demonstrate that reasoning, via Chain-of-Thought prompting or internalized through training, introduces a trade-off, improving accuracy while degrading confidence ranking. These findings directly impact risk-critical applications requiring reliable error identification and mitigation. 

---
# Folding, Reasoning, and Scaling with Open-source Drug Discovery Engine 

**Authors**: Aureka AI OpenDDE project  

**Link**: [PDF](https://arxiv.org/pdf/2607.03787)  

**Abstract**: Accurately modeling biomolecular interactions is a central bottleneck in biology and therapeutic discovery. Here, we introduce Open Drug Discovery Engine (OpenDDE), an open-source, all-atom biomolecular foundation model that uses co-folding as the entry point to a scalable AI-driven drug discovery engine. Rather than treating structure prediction as an isolated endpoint, OpenDDE is designed as a shared structural reasoning layer for modeling sequence-structure-function relationships across biomolecular complexes, enabling complex structure prediction today while providing a foundation for de novo design, affinity estimation, structure-conditioned optimization, and more. OpenDDE integrates advances in all-atom architecture, atomic latent reasoning, inference optimization, and large-scale data processing to achieve IsoDDE-level co-folding accuracy within a reproducible and openly accessible framework. We also identify two scaling-law directions for co-folding models, revealing practical routes for continued improvement through data, model, inference, and training scaling. By releasing training code, inference pipelines, checkpoints, and benchmarks, OpenDDE aims to democratize access to frontier biomolecular intelligence, accelerate global collaboration, and lay an open foundation for next-generation drug discovery systems that can move from predicting molecular structures toward designing, scoring, and optimizing therapeutic candidates for human health. 

---
# Bridging Interleaved Multi-Modal Reasoning as a Unified Decision Process 

**Authors**: Zican Hu, Xuyang Hu, Yiming Liu, Zuwei Long, Wei Liu, Yunzhuo Hao, Jiawei Gu, Linjie Li, Yu Cheng, Zhenhong Sun, Weibo Gu, Xing Sun, Zhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.03748)  

**Abstract**: Unified multi-modal models (UMMs) have shown promising interleaved text-image reasoning capabilities, yet effectively optimizing such multi-turn generation via reinforcement learning (RL) remains an open challenge. Existing approaches apply RL exclusively to text steps, relegating image generation to supervised surrogates, preventing policy gradients from propagating through the full interleaved trajectory across heterogeneous modalities. This leaves the potential of RL for UMMs largely untapped. In the paper, we introduce \textbf{BRAID} (\textbf{B}ridging inte\textbf{R}le\textbf{A}ved mult\textbf{I}-modal reasoning as a unified \textbf{D}ecision process), a simple framework that casts multi-turn text-image-text reasoning as a unified Markov decision process (MDP), enabling joint optimization of textual and visual generation via a single, principled RL objective. BRAID computes a shared trajectory-level advantage and propagates it coherently into both text tokens and image denoising paths, each optimized through its modality-native policy gradient mechanism. To further address long-horizon credit assignment, BRAID employs a vision-language model (VLM) judge that scores each intermediate image on its reasoning utility, supplying dense turn-level feedback to sharpen learning at critical visual branches. Experiments on spatial reasoning and visual perception benchmarks show that BRAID consistently outperforms various baselines, confirming that a unified MDP formulation with vision-thinking guidance is essential for effective multi-modal reasoning. 

---
# Can Conversational Temporal Dynamics Improve Depression Detection in Dyads? A Preliminary Investigation in Multi-Modality Perspectives 

**Authors**: Hanie Kang, Huang-Cheng Chou, Sudarsana Reddy Kadiri, Shrikanth Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2607.03744)  

**Abstract**: Automatic depression detection from clinical interviews typically models the semantic content and acoustic characteristics of participant speech. However, the interactional timing between the clinician and participant remains comparatively under-modeled. We investigate conversational temporal dynamics, specifically dyadic turn-pair timing, as a primary modality fused with self-supervised encoders. Evaluated on the DAIC-WOZ dataset, we compare a compact 24-dimensional timing module against frozen WavLM-large and RoBERTa-large baseline detectors. This temporal module achieves the highest single-modality performance on the development set. Furthermore, a convex-weighted late fusion strategy improves overall performance to 0.804 and 0.669 macro-F1 on the development and test sets, respectively. The learned fusion effectively assigns zero weight to acoustics, demonstrating that conversational timing serves as a lightweight, interpretable complement for dyadic depression screening. 

---
# Explainable Reinforcement Learning for Adaptive Traffic Signal Control 

**Authors**: Dickens Kwesiga, Nishu Choudhary, Angshuman Guin, Michael Hunter  

**Link**: [PDF](https://arxiv.org/pdf/2607.03703)  

**Abstract**: Reinforcement Learning (RL) has emerged as a powerful paradigm for adaptive traffic signal control. However, in safety-critical infrastructure like traffic control, the opaque, black-box nature of deep RL models poses challenges for transportation agency acceptance, regulatory compliance, operational trust, troubleshooting, and fine-tuning. To bridge this gap between high-performance optimization and human-comprehensible interpretability, this effort introduces a novel, explainable entity centric RL framework for safe and transparent traffic signal control. Rather than processing traffic states through monolithic, flat vectors, the proposed architecture disaggregates real-time intersection observations into distinct, high-dimensional lane entities and phase temporal configurations to inherently preserve the structural topology and geometric configurations of the intersection. Relational dependencies and inter-lane conflicts are dynamically extracted via a dual-stage attention network featuring sequential multi-head cross-attention and self-attention blocks. This design yields a real time affinity matrix that quantifies the direct influence of signal phases on specific approach volumes and queues, providing full visual and analytical interpretability. To ensure strict operational reliability, a deterministic action-masking interface is integrated directly into the Proximal Policy Optimization pipeline, explicitly blocking invalid phase transitions to guarantee absolute compliance with established signal timing and safety constraints. Evaluated in a microscopic simulation environment, outperforms state-of-the-art baselines in delay minimization. More importantly, the emergent attention weights align precisely with established traffic engineering principles, offering an auditable, trust-enabling, and deployable architecture for next-generation adaptive traffic control systems. 

---
# Agent Reinforcement Learning via Pivotal-Aware Self-Feedback Retry 

**Authors**: Weiyang Guo, Zesheng Shi, Longhui Zhang, Zeen Zhu, Min Zhang, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.03702)  

**Abstract**: Large language model (LLM) agents have shown strong decision-making capabilities in long-horizon interactive tasks, yet they still struggle to effectively leverage failed trajectories: full retries incur high interaction costs, while experience retrieval tends to dilute critical experience signals. To address this, we propose PivoARL, a self-feedback retry framework for experience exploitation in LLM agents. PivoARL identifies the pivotal erroneous turn through structured reflection and performs local retry only from the corresponding pivotal state, thereby reusing the correct prefix and reducing redundant interactions. From an information-gain perspective, we further show that pivotal retry concentrates useful experience signals near the error boundary, mitigating the signal dilution caused by state-agnostic experience utilization. Based on this insight, we design a pivotal-aware credit assignment mechanism that rewards correct prefixes while isolating erroneous suffixes, and optimize reflection quality through implicit reflection returns. We conduct a systematic evaluation on 4 agent tasks and 7 search-based QA benchmarks. Results show that PivoARL achieves significant improvements on Pass@2/3 across all tasks, with an average gain of about 11.5\% over MetaRL. Moreover, benefiting from contrastive preference signals induced by pivotal turns, PivoARL also consistently improves Pass@1 on over 80\% of the tasks. On Minesweeper environment, PivoARL improves over GiGPO by more than 45\% and reduces interaction turns by about 42\% on average compared with full-retry methods. Code is available at this https URL. 

---
# Robust Feasible Route Construction through Collaborative Partition Optimization 

**Authors**: Oguzhan Karaahmetoglu, Hyong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.03694)  

**Abstract**: Large-scale Capacitated Vehicle Routing Problems (CVRPs) are commonly solved by partitioning customers into smaller routing problems that can be optimized independently. While this substantially reduces computational complexity, independently constructed routing solutions may leave some customer demand unserved even when sufficient resources exist elsewhere in the fleet. We present Collaborative Routing Constructors (CoRC), a routing framework that enables independently solved subproblems to exchange customers and vehicles during optimization rather than relying solely on a fixed partition or a subsequent global re-optimization stage. Computational experiments on AGS benchmark instances and synthetic instances containing up to 200,000 customers compare CoRC against independent routing, post-routing global re-optimization, and state-of-the-art, end-to-end routing frameworks. Across all evaluated partitioning strategies, CoRC consistently constructs feasible routing solutions where competing partition-based methods do not. Furthermore, it remains effective on problem instances for which the evaluated end-to-end routing frameworks did not produce solutions under the same computational budget. These results demonstrate that collaboration between routing subproblems provides a robust and scalable approach for feasible large-scale route construction. 

---
# The Role of Rigor in Artificial Intelligence 

**Authors**: Timothy Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2607.03634)  

**Abstract**: Artificial intelligence (AI) has achieved extraordinary capabilities despite lacking many of the conceptual and scientific foundations associated with mature disciplines. Unlike traditional sciences, where reliable technology typically emerges from theoretical understanding, modern AI has progressed largely through performance-driven iteration and "alchemical" experimentation. This tension motivates a systematic analysis of AI through the lens of rigor. We introduce a three-part framework consisting of conceptual rigor (clarifying foundational concepts), epistemic rigor (establishing scientific understanding), and operational rigor (ensuring reliable performance and deployment). Using this framework, we analyze competing conceptions of intelligence and understanding, the strengths and limitations of the empirical approach to deep learning, the power and pitfalls of benchmarks, and the obstacles to theory development posed by modern AI systems. We argue that the distinctive trajectory of AI arises from how forms of rigor interact across paradigms, resulting in the primacy of operational rigor in modern deep learning. This perspective helps explain both AI's rapid advances and its persistent uncertainties, while clarifying the challenges involved in transforming AI into a mature science and reliable technology. 

---
# How to Avoid Debate: Scalable AI Safety via Doubly-Efficient Interactive Proofs 

**Authors**: Liyan Chen, Yael Tauman Kalai, Zoe Xi  

**Link**: [PDF](https://arxiv.org/pdf/2607.03561)  

**Abstract**: As AI models continue to develop powerful capabilities, it becomes critical that we are able to verify that their output is aligned with our intentions. A recent line of work focuses on verification via debate, a model of interactive proofs where two competing powerful provers, or AI models, debate each other to convince a weak verifier, or a human, of the correctness of their claim. However, debate assumes that the two AI models possess equal abilities and that one of them is truthful, which may not be realistic.
In this work, we show \emph{how to avoid debate}: we initiate the study of \emph{single-prover} interactive proofs for AI safety. Prior results in single-prover interactive proofs do not immediately carry over to the AI safety setting: for example, they do not work when the computation has access to an oracle, such as to human judgment or an external database such as the web. We present doubly-efficient single-prover interactive proofs and arguments for oracle-aided computations (also known as relativizing proofs), in the settings where (1) the computation is robust, in the sense that the output does not change if at most a small fraction of the answers to oracle queries are incorrect, or (2) the oracle is a low-degree polynomial. These results suggest that interactive verification is possible even without debate, under structured or noise-tolerant oracle access. 

---
# Applying Answer Set Programming with Fuzzy Membership Functions: a Case Study 

**Authors**: Luca Ferragina, Ilenia Galati, Lorena Gullone, Francesco Scarcello  

**Link**: [PDF](https://arxiv.org/pdf/2607.03550)  

**Abstract**: Human reasoning often operates through qualitative concepts expressed by linguistic labels such as high, low, expensive, or cheap, whose interpretation depends on context and is usually vague, despite being rooted in numerical data. This paper explores a novel fuzzy-logic-based qualitative extension of Answer Set Programming (ASP) to bridge numerical information and qualitative reasoning. The underlying language, formally introduced in a separate work, provides a principled framework that avoids rigid thresholds and supports robust reasoning under vagueness. Focusing on a representative use case, we illustrate how the framework integrates numerically grounded inputs (such as outputs of machine learning models) with symbolic reasoning over qualitative labels. Key features, including learning-based membership functions and semantically enriched predicates, enable the combination of expert knowledge, contextual factors, and subjective interpretations within a unified declarative setting. 

---
# MentalThink: Shaping Thoughts in Mental SVG World 

**Authors**: Kangheng Lin, Jisheng Yin, Dingming Li, En Yu, Yana Wei, Han Zhou, Liang Zhao, Hongyu Zhou, Hongbo Peng, Jianjian Sun, Zheng Ge, Xiangyu Zhang, Daxin Jiang, Jingyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.03530)  

**Abstract**: We introduce MentalThink, a visual-symbolic reasoning paradigm that equips Multimodal LLMs (MLLMs) with an executable mechanism for "mental" visualization. The core of MentalThink is a think-with-SVG pipeline, where the model learns to generate, render, and interpret scalable vector graphics (SVG) code as an intermediate visual representation for multi-turn reasoning. By creating structured vector sketches, the model can externalize spatial hypotheses, inspect them through deterministic rendering, and reason within a constrained geometric space, effectively mimicking the human process of mental imagery. We instantiate this paradigm through a two-stage training framework, combining Supervised Fine-Tuning (SFT) for SVG syntactic alignment with multi-turn Reinforcement Learning (RL) to encourage iterative inspection, revision, and refinement of intermediate visual hypotheses. Extensive evaluations demonstrate that MentalThink achieves superior performance on spatial understanding and reasoning benchmarks (e.g., 55.1% on VSIBench, 76.0% on MindCube), showing that executable vector graphics provide a verifiable visual workspace for dynamic perspective taking, visual reflection, and compositional scene construction. 

---
# Demonstrating Generalization Failures via Mixtures of Conditional Policies 

**Authors**: Jou Barzdukas, Jack Peck, Julian Schulz, Paulius Rauba, Steven Basart, Lennie Wells  

**Link**: [PDF](https://arxiv.org/pdf/2607.03478)  

**Abstract**: Post-training of frontier language models is conducted on curated task suites, and inevitably leaves a distribution shift between training and deployment environments. This exposes developers to generalization failures, which are relatively poorly understood. To better understand such generalization failures, we believe the community should construct clean demonstrations under simplified conditions. To facilitate this, we propose a simple and flexible way to construct language models which fail to generalize in controllable ways when subsequently trained with Reinforcement Learning (RL) on a given distribution of training tasks. Our construction uses Supervised Fine-Tuning on a dataset of a mixture of transcripts corresponding to a collection of 'conditional policies', which can each independently be assigned certain behaviors on each different task distribution, to obtain a model that is then well approximated as a 'mixture of conditional policies.' We observe that RL training then selects for policies that obtain the highest reward on the training distribution. This can produce striking behaviors: in a controlled setting with two distributions containing identical questions prepended with two different 'trigger strings', RL training on either distribution actively degrades performance on the other to zero, even though the underlying task is identical. We also use our construction to illustrate two novel ways in which generalization may fail in future language models, corresponding to distribution shifts of task coverage and temporal context respectively. While our construction is deliberately simple and may not closely resemble 'natural' generalization failures, the resulting 'model organisms' are of interest for alignment stress-testing and generalization science, and can be used as existence proofs that training success and generalization can come apart in structured ways. 

---
# Personalized Causal Recourse: A Human-In-The-Loop Approach 

**Authors**: Denise Tampieri, Giovanni De Toni, Paolo Giudici  

**Link**: [PDF](https://arxiv.org/pdf/2607.03425)  

**Abstract**: Algorithmic recourse addresses the challenge of providing tailored recommendations to users affected by unfavorable machine learning decisions, in potentially high-stakes scenarios. Traditional approaches to recourse often rely on the closest counterfactual explanations or assume a priori knowledge of a user's causal structure, resulting in interventions that overlook individual contexts and specific feature interactions. To overcome these limitations, we study a human-in-the-loop framework that iteratively approximates the user's structural causal model through interactive queries via Bayesian inference before producing recourse recommendations. This framework exploits humans' feedback to improve the identification of causal effects, allowing personalized recourse that is plausible, cost-effective, and aligned with the actual causal dependencies of each user. As a proof of concept, we evaluate this framework through simulated human responses. Our simulations across linear and non-linear causal models show promising results, though challenges remain in capturing complex, non-linear structures, emphasizing the importance of accurate approximations and robust noise distribution modeling. 

---
# Efficient bias mitigation in T2I diffusion models using Concept Graphs 

**Authors**: Mansi, Avinash Kori, Francesco Leofante  

**Link**: [PDF](https://arxiv.org/pdf/2607.03397)  

**Abstract**: Text-to-Image diffusion models often propagate harmful bias inherited from the training data. Existing bias mitigation techniques typically intervene only at the text encoder or provide inference-time guidance, often leading to generations that collapse into semantically incoherent outputs. To address these limitations, we introduce CO-ALIGN (Concept Ontology Alignment), a novel bias mitigation approach based on concept-graph alignment that operates on the model's internal concept ontology. By aligning concepts within the text encoder and denoiser, CO-ALIGN achieves substantial bias reduction while preserving generative integrity. We demonstrate the effectiveness of concept-graph alignment across three paradigms: text-encoders, denoisers and joint text-denoiser ontology alignment. CO-ALIGN outperforms the state of the art, improving fairness by $30\%$, $\Delta FID=11.4$ in image quality, $2.8\%$ in image fidelity, all while reducing semantically incoherent outputs by $88\%$. Beyond bias mitigation, we show that CO-ALIGN benefits other downstream tasks as well. In particular, our experiments demonstrate that better-aligned internal ontologies enhance concept unlearning robustness across multiple unlearning techniques. 

---
# From Mobile Data to Business Insights: An End-to-End Analytics Framework for Large-Scale Urban Mobility Analysis and Decision Support 

**Authors**: Thiago Andrade, Shazia Tabassum, Miguel E. P. Silva, Ricardo Dinis, Joao Gama  

**Link**: [PDF](https://arxiv.org/pdf/2607.03394)  

**Abstract**: Real time location data derived from mobile applications is a powerful tool for addressing various urban challenges, including tourism planning, parking management, bus route optimization, and resource allocation. Besides, it offers invaluable insights for shaping strategic decisions in commercial domains such as location based services, market share analysis, and behavioral profiling. In this expansive study, we aim to address all of the aforementioned challenges by investigating the behaviors and patterns of smartphone users within urban environments, particularly in the domains of tourism, transportation, and retail. Our approach encompasses the development of a sophisticated data platform from inception to implementation, which includes the formulation of use cases, architectural design, and implementation of modules. We employ state of the art techniques and technologies, including data anonymization, ETL pipelines, and utilizing Google BigQuery and Vertex AI for data processing and machine learning model development. A modular architecture based on reusable analytical building blocks was developed to generate data products that support multiple stakeholder driven use cases. Additionally, we apply interactive data visualization techniques via Power BI to facilitate the effective interpretation of analytical findings by stakeholders. The developed models address a wide range of mobility analytics tasks, including mobility profiling, frequent trajectory mining, area of influence analysis, traffic anomaly detection, and origin destination pattern analysis. The results demonstrate the framework's ability to capture user mobility dynamics at fine spatial and temporal resolutions, providing actionable insights for urban planning and strategic business decision making. 

---
# When Aggregate Alignment Misleads: Auditing Policy Repair Without Per-State Expert Actions 

**Authors**: Peiying Zhu, Sidi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2607.03386)  

**Abstract**: Agentic AI systems are increasingly used to edit, refine, and repair decision policies, but evaluating these edits is difficult when per-state expert action labels are unavailable. We study this problem in a hotel-pricing simulator where an agentic policy editor receives only region-level diagnostic feedback: summaries of how its price distribution differs from a benchmark policy across time, inventory, and market regions. The editor cannot observe benchmark actions, benchmark source code, reward numbers, or held-out outcomes, and may only propose constrained edits to a target-action table. On 5,000 held-out episodes, a multi-restart LLM editor reaches RevPAR 108.47 (95% CI 107.61 - 109.34), close to the benchmark policy's 108.75 (107.81 - 109.68), with paired gap (LLM minus benchmark) -0.276 and 95% CI [-0.692, 0.146]. A cheap diagnostic projection already recovers much of the revenue (107.90), so the LLM editor's distinctive gain is not raw revenue lift alone: it also reduces episode composition distance from 1.153 to 0.609. This is the strongest non-benchmark repair result. This profile is not explained by restart search alone: non-semantic proposers with up to 2,500 evaluations fall 8.77 - 14.57 RevPAR points short. Nor is it explained by plausible prompt format: a shuffled-diagnostic control breaks region-error correspondence and falls to RevPAR 94.30. The match is genuine but partial. A tree editor achieves stronger pooled alignment, 0.214 versus 0.266, and stronger reference-state D1, 0.328 versus 1.197, yet revenue falls to 98.91. These results show that agentic policy repair should be evaluated by whether diagnostic feedback becomes reliable closed-loop outcome, not by a single behavioral distance. 

---
# Reflective Dialogue or Prompt Refinement? Effects of Tutor Scaffolding on Students' Independent LLM Use for Programming 

**Authors**: Jerome Brender, Laila El-Hamamsy, Kim Uittenhove, Aitor Perez, Patrick Jermann, Francesco Mondada, Engin Bumbacher  

**Link**: [PDF](https://arxiv.org/pdf/2607.03303)  

**Abstract**: While Large Language Models (LLMs) can provide personalized support in learning, several studies have raised concerns regarding their use in education. Importantly, learning depends on how students engage with LLMs. This study examined how two types of LLM-based tutors shape students' prompting practices, learning, and subsequent LLM-use: a Socratic-Guidance (SG) tutor, which structures interaction through dialogic questioning, and a Prompt-Refinement (PR) tutor that guides the formulation of effective prompts. We conducted a two-phase study in a graduate-level mobile robotics course: 66 students used either the SG or PR tutor during a 6-week intervention, followed by 52 students using an unconstrained LLM during a 3-week course project. Results show that while the SG- and PR tutors led to similar task performance and prompting patterns during guided use, they differ in learning outcomes and later LLM-use. SG-students, relative to PR-student, achieved higher learning gains in later sessions, and were more likely to adopt understanding-driven prompting strategies, which are predictive of higher understanding, when using an unconstrained LLM. Although learners perceived the SG tutor as less efficient, the findings suggest that Socratic guidance supports the development of students' capacity to learn with LLMs over time, highlighting its importance for LLM tutor design. 

---
# Embodied Operators and Benchmarking: Toward Reusable and Deployable Embodied Intelligence Systems 

**Authors**: Junwu Xiong, Jiaxuan Gao, Wei Chai, Renxing Chen, Yuzhen Li, Yu Guo, Yucheng Guo, Mingxi Luo, Wenyang Ma, Yiyun Mou, Yifei Zhang, Chen Zhou, Yongjian Guo  

**Link**: [PDF](https://arxiv.org/pdf/2607.03283)  

**Abstract**: Embodied intelligence systems require not only end-to-end policy models, but also reusable functional modules that transform multimodal observations, robot states, human demonstrations, and task contexts into structured representations, decisions, trajectories, control references, and system services. This work defines these modules as embodied operators and studies them as independent yet composable units in embodied intelligence pipelines. We clarify their definition boundary, emphasizing task semantics, standardized input-output contracts, deployability, reusability, and multi-layer optimizability. We further construct a taxonomy covering five categories: detection and segmentation, spatial localization and 3D understanding, hand motion recovery, embodied foundation models and task-decision operators, and planning, control, and system support operators. For each category, we summarize representative functions, technical paradigms, application roles, and practical limitations. Beyond taxonomy, we propose a multi-dimensional benchmark framework that evaluates embodied operators in terms of correctness, end-to-end efficiency, resource usage, temporal stability, portability, interface compatibility, deployment reliability, and downstream task utility. We also discuss workflow-level operator acceleration and open challenges in operator composition, data standardization, world models, VLA safety, edge deployment, and real-world application value. Overall, this work argues that embodied operators should be optimized and evaluated as holistic deployable components, providing a foundation for reusable, scalable, and verifiable embodied intelligence systems. 

---
# Organizational Memory for Agentic Business Process Execution 

**Authors**: Lukas Kirchdorfer, Adrian Rebmann, Christian Warmuth, Timotheus Kampik, Theiss Heilker, Gregor Berg  

**Link**: [PDF](https://arxiv.org/pdf/2607.03228)  

**Abstract**: LLM-based agents offer new opportunities for automating business process execution beyond the limits of rule-based systems. However, general-purpose LLMs lack the organization-specific knowledge required for reliable execution, which is typically fragmented across human-oriented artifacts such as policies, process models, and standard operating procedures. While such knowledge can technically be encoded in individual prompts or agent-specific retrieval setups, this approach does not scale in enterprises, as it gives rise to knowledge silos and rule duplicates, and makes consistent updates and learning across agents difficult. We argue that this calls for an organizational memory for agentic business process execution: a shared, governed, and agent-consumable reference layer of evolving organization-specific procedural knowledge about how work should be executed. We derive requirements for such a memory, propose an architecture for its curation and consumption, and demonstrate its effectiveness in a proof-of-concept based on a procurement scenario. 

---
# APeB: Benchmarking Personalization Ability of Large Language Model Agents 

**Authors**: Garry Yang, Zizhe Chen, Xinru Chen, Yongqiang Chen, Jianxiang Wang, Deyu Zou, Linyi Ding, Jialiang Wu, Yunzhong He, Yu Gong, James Cheng, Huaixiao Tou  

**Link**: [PDF](https://arxiv.org/pdf/2607.03162)  

**Abstract**: LLM-powered agents struggle with personalization when users issue raw, underspecified queries. In this setting, agents must infer latent intent, extract preferences from noisy interaction histories, and select among competing alternatives. Existing benchmarks rarely test this capability, as they often rely on user-refined queries or simplified histories. We introduce personalized product search (PPS), a testbed for agentic personalization under raw queries and diverse histories. We construct Agent Personalized Benchmark (APeB) from action logs, pairing underspecified intents with rich histories and user-viewed candidate items. Evaluating state-of-the-art LLMs with multi-step agent workflows, we find that models handle explicit queries well but struggle with early-stage queries requiring intent and preference discovery. Rubric analysis attributes this gap mainly to ineffective history use. A simple history-aware query-refinement pipeline, VQRA, yields consistent gains, highlighting the need for dedicated history-utilization modules in personalized agents. 

---
# Silicon Sampling via Cross-Survey Transfer 

**Authors**: Chan-Tung Ku, Chan Hsu, Pei-Cing Huang, Frank Cheng-shan Liu, I-Ling Cheng, Yihuang Kang  

**Link**: [PDF](https://arxiv.org/pdf/2607.03091)  

**Abstract**: Silicon sampling-using large language models (LLMs) to simulate human survey respondents-has emerged as a promising approach for augmenting traditional survey research. However, most evaluations rely on distributional comparisons rather than individual-level prediction, which risks conflating pattern matching with coherent respondent-level prediction. We propose cross-survey transfer, a more rigorous evaluation framework in which an LLM is given a respondent's answers to one set of questions and must predict their answers to entirely different questions from the same survey. Using data from the Taiwan Election and Democratization Study (TEDS) 2024, three open-weight LLMs (27B-120B parameters), and supervised machine learning baselines, we find that: (1) zero-shot LLMs achieve 52% accuracy on genuinely unseen items, closing to within 6 percentage points (pp) of a supervised random forest trained on same-population data; (2) a stable construct predictability hierarchy emerges, from 67% for partisan attitudes to 23% for sovereignty; and (3) variance collapse and safety alignment effects-two commonly cited LLM limitations-turn out to be more nuanced than previously reported, with variance collapse affecting supervised models as well and alignment effects varying dramatically across model families. These findings clarify both the promise and boundaries of silicon sampling. 

---
# Human-Centric Reflective Architecture for Human-AI Collaborative Decision-Making 

**Authors**: Andreas Kouridakis, Dimitrios Patiniotis Spyropoulos, George Vouros  

**Link**: [PDF](https://arxiv.org/pdf/2607.03025)  

**Abstract**: The use of Large Language Models (LLMs) across diverse areas of human activity-ranging from everyday tasks to safety-critical applications-aims to enhance decision-making effectiveness with minimal human feedback. Concurrently, it seeks to align decisions with human expectations, preferences, and needs while mitigating risks associated with AI non-determinism. However, humans frequently over- or under-rely on AI recommendations, and current AI systems remain poorly calibrated to human expectations. To address these challenges, we introduce a human-AI collaborative decision-making framework designed to augment human capabilities and align AI agents with human preferences and expectations. Specifically, this paper (a) formulates the collaborative decision-making task as a stochastic game between an AI agent and a human player, and (b) proposes the Human-Centric Reflective Architecture (HCRA), which integrates human-calibrated models with reinforcement learning agents that leverage linguistic feedback in an iterative, reflective process. Evaluation results demonstrate that HCRA enhances decision-making effectiveness and delivers high-quality recommendations. 

---
# Beyond Forecasting: The Belief-to-Trade Layer in Prediction-Market Agents 

**Authors**: Yishu Wang, Yuxuan Wang, Jiaqi Deng, Hanyang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2607.03015)  

**Abstract**: Forecasting future events has attracted growing attention as a testbed for general-purpose AI. A natural way to ground this evaluation is let the models trade in the prediction markets. Trading, however, requires more than forecasting. Moreover, recent benchmarks report a substantial gap between calibrated probability scores and the trading results. We propose Raven-Agent, to the best of our knowledge, the first autonomous trading agent for prediction markets. On a controlled replay over an archived decision set, our architecture achieves the only positive return and the only positive risk-adjusted return among all tested policies. We have released our code in this https URL . 

---
# Reinforcement Learning for Evidence-Seeking Diagnostic Reasoning with Large Language Models 

**Authors**: Shengyi Hua, Kangzhe Hu, Conghui He, Xiaofan Zhang, Shaoting Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.02983)  

**Abstract**: Recent reasoning-centric Large Language Models (LLMs) have made significant strides, yet they predominantly operate on a passive-inference pattern that assumes complete information. In contrast, real-world clinical intelligence is inherently an iterative investigative process requiring strategic evidence acquisition. To bridge this gap, we formalize medical diagnosis as an Iterative Evidence-Seeking Task. We leverage Reinforcement Learning with Verifiable Rewards (RLVR) to elicit intrinsic reasoning within a closed-loop environment, guided by a novel suite of rewards that enforce diagnostic precision and examination consistency. To facilitate this, we introduce the Retrieval-Augmented Generation-based Examination Simulator (RAGES), a high-fidelity clinical oracle that provides realistic, knowledge-grounded follow-up evidence. Empirical results across diverse datasets demonstrate that our framework enables LLMs to transition from passive responders to autonomous assistants. Notably, our model demonstrates comparable performance to larger and reasoning-enhanced baselines, while RAGES proves superior to vanilla LLMs in generating biologically plausible clinical feedback. 

---
# Evaluating Generative Agents with Actions Grounded in Socially Distributed Task Environments using Incognita 

**Authors**: Dan C. Hsu, Luke Lu  

**Link**: [PDF](https://arxiv.org/pdf/2607.02975)  

**Abstract**: Effective agency in social environments depends on when an agent seeks knowledge, when it acts, and whether its actions are justified by acquired information. Existing grounded benchmarks provide executable actions, persistent state, and verifiable outcomes, while social simulation environments provide rich interaction among language agents. We study an evaluation setting that combines these requirements. We define socially distributed task environments as interactive environments where task-relevant knowledge is partitioned across role-isolated participants and consequential actions are accessible only through them. Communication serves as exploration over role-partitioned knowledge, while grounded action serves as exploitation over environment state. We introduce Incognita, a Concordia-based framework that separates social interaction from grounded execution. The evaluated agent routes messages to a user or specialist entities; specialists mediate admissible operations; a deterministic sub-environment executes accepted operations over a canonical state; and an offline evaluator scores outcomes with inherited rewards. Incognita-Retail transforms tau-bench retail into a multi-entity environment while preserving final-state reward semantics. We evaluate three generative agent models on 18 tasks stratified by social breadth, with 540 trials. Progress appears in reward and behavior: success rises from 0 percent to 8.9 percent and 17.2 percent, while premature finalization falls from 100 percent to 87 percent and 58 percent. Stronger models elicit more hidden knowledge, contact more entities, and attempt more grounded writes, yet reliability remains low. These findings show that socially distributed task environments expose behavior before reliable success, including knowledge elicitation, source selection, grounded action attempts, and premature completion belief. 

---
# A Sliding-Window-Based Reinforcement Learning for Dynamic Assembly Flow Shop Scheduling with Multi-Product Delivery 

**Authors**: Junhao Qiu, Jianjun Liu, Ting Liu, Rongjie Liao, Zhantao Li, Qingfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.02941)  

**Abstract**: Multi-product kitting delivery imposes significant challenges for real-time scheduling in hybrid manufacturing systems that integrate processing and assembly, as dynamic order arrivals simultaneously alter supply dependencies and the set of feasible job-machine assignments. This paper proposes a sliding-window-based reinforcement learning (SWRL) framework for end-to-end online scheduling in the flexible assembly flow shop scheduling problem with complex kitting constraints. The problem is formulated as a heterogeneous graph-based Markov decision process that captures the dual-layer kitting structure and the tail-product bottleneck dynamics that produce a sparse reward landscape. To address the resulting challenges, SWRL integrates a sliding-window filtering mechanism that filters inactive nodes and prioritizes kitting-critical operations, a spatiotemporal graph encoding network that tracks bottleneck shifts across consecutive decision states, and a dynamic action mapping module with a constrained waiting strategy that adapts to the changing action space under variable topologies. Experiments on real-world instances from a home appliance manufacturer demonstrate that SWRL achieves consistent tardiness reductions over classical dispatching rules and existing deep reinforcement learning methods, and exhibits robust performance across varying resource configurations, order loads, and arrival concentrations. 

---
# VERITAS: Towards a General-Purpose Replication Tool for Scientific Research 

**Authors**: Haokun Liu, Filbert Aurelian Tjiaranata, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2607.02931)  

**Abstract**: AI tools are accelerating scientific publication while the systems that review it struggle to keep up, and independent verification of published research has become both harder and more important. As manual replication is slow and expensive, a growing line of work uses coding agents to automate parts of the process. Existing efforts are largely packaged as benchmarks with companion agents that only run inside the benchmark's own pipeline, and no general-purpose replication tool exists. We present VERITAS, a domain-agnostic replication framework built around CLI coding agents. Given a paper, a code repository, or both, VERITAS extracts the paper's claims, runs the methodology while resolving issues as they arise, and judges each claim against the evidence from experiment runs. The pipeline returns an importance-weighted Replication Score, a severity-rated log of every fix applied, and the patched codebase. We evaluate VERITAS on CORE-Bench and ReplicationBench, 65 papers spanning computer science, social science, medicine, and astrophysics. Against two strong Claude Code baselines on the same model and host environment, VERITAS achieves state-of-the-art performance and leads on every metric on both benchmarks. 

---
# Oyster-II: Reinforcement Learning for Constructive Safety Alignment in Large Language Models 

**Authors**: Jiyang Guan, Yong Xie, Jun Chen, Jiexi Liu, Zipeng Ye, Defeng Li, Jiayu Shen, Jialing Tao, Hui Xue  

**Link**: [PDF](https://arxiv.org/pdf/2607.02914)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across diverse applications, yet ensuring their simultaneous safety, helpfulness, and trustworthiness remains a persistent challenge. Conventional refusal-oriented alignment strategies mitigate harmful content generation but systematically fail to serve legitimate user needs, often withholding information that could safely and constructively address the underlying intent of sensitive queries. Building upon the constructive safety paradigm pioneered by Oyster-I, which moves beyond blanket refusal toward thoughtful, response-oriented safety alignment, we identify two critical limitations of its Supervised Fine-Tuning (SFT)-based scheme: insufficient safety generalization to out-of-distribution scenarios and a phenomenon we term safety chain-of-thought (CoT) over-generalization, wherein safety-oriented reasoning patterns are excessively applied to benign queries, degrading helpfulness and user experience. To address these limitations, we propose Oyster-II, a reinforcement learning (RL)-based constructive safety alignment framework that adopts a Zero-RL paradigm combined with a multi-stage reinforcement learning this http URL across extensive benchmarks, Oyster-II comprehensively surpasses both Qwen3-14B and its predecessor Oyster-I on safety dimensions, achieving cross-scale performance comparable to Qwen3-Max and Qwen3.5-397B. 

---
# MedCalc-Pro: Solving Complex Medical Calculations with LLM Agents 

**Authors**: Siran Zhao, Ruihui Hou, Ziyue Huai, Chennuo Zhang, Tong Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2607.02879)  

**Abstract**: Current benchmarks for evaluating large language models (LLMs) in medical calculation are largely based on simplified settings, where each patient case corresponds to a single calculator and the required tool is explicitly specified in the query. However, real clinical scenarios often require multiple calculators for joint evaluation, nested-scale calculation, and fuzzy queries that do not directly specify the target calculator. To this end, we propose a new medical calculation benchmark, MedCalc-Pro, which covers three progressively challenging task settings: single-calculator, multi-calculator, and nested-calculator calculation settings. MedCalc-Pro contains 2,268 real-world clinical cases, covering 77 medical calculators across 14 clinical departments. Meanwhile, to address the limited performance of existing frameworks and methods in complex clinical scenarios, we further propose a more generalizable agent framework that supports multi-tool selection and nested-tool calling, while suppressing parameter error propagation through structured validation and evidence review. We conduct systematic comparisons across open-source, closed-source, and medical-specialized LLMs, and the results show that our framework achieves the best performance across all three task settings. This work provides a new benchmark and method for evaluating and applying LLMs in challenging medical calculation scenarios. 

---
# Object-Centric Environment Modeling for Agentic Tasks 

**Authors**: Yiyang Li, Tianyi Ma, Zehong Wang, Yijun Ma, Yanfang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2607.02846)  

**Abstract**: Large language model (LLM) agents can improve through accumulated experience, but free-form textual memories become difficult to maintain, validate, and reuse as interactions grow. Recent symbolic approaches learn executable skills or programmatic world models, yet often store local procedures or assume simplified dynamics. We propose Object-Centric Environment Modeling (OCM), which organizes experience into an executable object-centric environment model. OCM maintains two connected code bases: object knowledge, which defines environment entities and mechanisms as Python classes, and procedure knowledge, which records reusable interaction patterns that must import and use the object model. OCM works in an online setting: after each episode, OCM reflects on the trajectory, updates both knowledge bases, and verifies that all procedures execute against the updated object model. During future interaction, the agent uses progressive knowledge disclosure to inspect compact code signatures first and read source code only when needed. Experiments show that OCM achieves the best average rank across benchmarks and reduces invalid actions, demonstrating that agents can benefit from building object-centric environment models. 

---
# SwarmResearch: Orchestrating Coding Agents for Open-Ended Discovery 

**Authors**: Yuvraj Virk, Zack Edds, Chunqiu Steven Xia, Lingming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.02807)  

**Abstract**: Long-running coding agents such as autoresearch can persistently discover optimizations for open-ended problems. However, they tend to converge onto a single high-level approach, then proceed with low-level edits while missing other superior approaches to the problem. We hypothesize two harness-level design choices contribute to this behavior: accumulating context in a single long-running agent and only exposing a single program state to edit. We introduce SwarmResearch, an orchestrator-subagent harness in which a Shepherd Agent uses global context to steer a population of Search Agents, each operating with local context in their respective git branch. On open-ended optimization tasks, SwarmResearch discovers better or comparable solutions to state-of-the-art LLM-guided evolution and multi-agent techniques on 13/15 tasks, driven by higher-level exploration. Compared with fixed scaling of serial and parallel agents, SwarmResearch's orchestrator-guided scaling discovers better-performing solutions by adapting parallelism at different search depths. 

---
# Automated Data Readiness for Scientific AI 

**Authors**: Sean R. Wilkinson, Valentine G. Anantharaj, Jong Youl Choi, Ketan Maheshwari, Marshall McDonnell, Massimiliano Lupo Pasini, Polina Shpilker, Renan Souza, Patrick Widener, Sarp Oral, Wesley Brewer  

**Link**: [PDF](https://arxiv.org/pdf/2607.02771)  

**Abstract**: Leadership computing facilities steward large-scale scientific datasets that routinely require substantial transformation before serving as AI training data. However, no existing framework fully unifies automated transformation, readiness assessment, provenance tracking, and agent-native deployment. We present REDI, an open-source framework that addresses this gap through a unified five-stage pipeline (ingest, preprocess, transform, structure, and output) with per-stage instrumentation for reproducibility and deployment as an agent-callable skill; companion tool SetGo automates FAIR compliance and catalog publication. Evaluated across climate, proteomics, materials science, and nuclear fusion, REDI transforms all datasets from raw to AI-ready, with outputs validated against domain-expert references, and preliminary results show near-ideal parallel scaling to 100 nodes on Frontier for the climate case. Provenance-instrumented profiling reveals file I/O as the dominant pipeline cost, with format selection a first-order optimization lever. These results establish REDI as a cross-domain platform providing automated data readiness for scientific AI, transforming data preparation bottlenecks into reproducible, reusable community assets. 

---
# ASK in the Dark: Uncertainty-Gated LLM Assistance under Partial Observability 

**Authors**: Juarez Monteiro, Nathan Gavenski, Guilherme Lima, Francisco Galuppo, Odinaldo Rodrigues, Adriano Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2607.02686)  

**Abstract**: Reinforcement learning agents operating under partial observability must act on incomplete information, making them natural candidates for guidance from small language models (SLMs) that carry broad reasoning priors. Yet integrating SLM guidance into this setting has proven difficult: across all test environments, vanilla uncertainty-gated approaches achieve an overwrite rate at or near zero, meaning the SLM almost never contributes an independent action. We trace this failure to the bare egocentric prompt, which provides insufficient context for genuine reasoning, and identify it as a context problem rather than a capacity problem. We propose ASK+, which supplies the SLM with trajectory-aware context (a partially revealed map, visited positions, and action history) and structured chain-of-thought reasoning, converting it from a passive redundancy check into a more informative consultant that occasionally corrects the policy. We further establish that the predictive entropy signal used for selective querying measures action uncertainty rather than state uncertainty and remains informative in POMDPs, making uncertainty-gated assistance viable beyond fully observable settings. The stateful prompt drives substantial gains: on DoorKey, where vanilla ASK matches PPO (both 89%), ASK+ reaches 93% success; on FourRooms, success climbs from 53% to 70%; on HigherLower, accuracy reaches 73.7%, matching the SLM-only upper bound. Across all environments, Qwen3.5-2B matches or exceeds Qwen3.5-4B, confirming that prompt design and selective gating dominate the impact of model scale, enabling guidance without large models. 

---
# Internal Pluralism and the Limits of Pairwise Comparisons 

**Authors**: Bailey Flanigan, Michelle Si  

**Link**: [PDF](https://arxiv.org/pdf/2607.02672)  

**Abstract**: Local pairwise comparisons are a standard tool for learning how people want decision rules to work, e.g., in participatory design or alignment. However, their use builds in two strong assumptions: that local comparisons are sufficient evidence about how a person wants an automated decision rule to behave, and that people can always answer those comparisons decisively. We investigate how these assumptions may be compromised under internal pluralism: the idea that an individual evaluates decision rules according to multiple authoritative priorities about how the rule should behave. We provide a formal model of such pluralistic preferences over decision rules, which then lets us identify two distinct failures of forced local pairwise comparison data. First, priorities such as proportionality, egalitarianism, and equal treatment are inherently global: what they imply in one case can depend on what happens elsewhere, so local comparisons may fail to capture them. Second, even when priorities are representable locally, tension between strongly-held priorities can generate internal conflict, producing potentially costly behavioral distortions when comparisons are forced. We then use our model to investigate the alternative -- allowing people to report indecision -- and our findings suggest that doing so can considerably reduce the number of queries needed to learn preferences accurately. We conclude by describing how our model points toward preference-learning methods that elicit these priorities directly, yielding more faithful and interpretable accounts of what people value. 

---
# iFLYTEK-Embodied-Omni Technical Report 

**Authors**: Yuan Zhang, Jingfei Ni, Guanchen Lu, Shiqi Zhang, Qingshan Xu, Chi Liu, Xin Nie, Wenjie Xu, Lin Gao, Zhiyuan Cheng, Mingxin Zhou, Jiajia Wu, Diyuan Liu, Jia Pan, Chao Ji  

**Link**: [PDF](https://arxiv.org/pdf/2607.02542)  

**Abstract**: General-purpose embodied agents must understand multimodal instructions, anticipate how their environment will evolve, and produce precise control actions over extended horizons. Existing approaches typically specialize in visual-language reasoning, video-based world modeling, or action generation, while cascaded pipelines that first synthesize future observations and then infer actions can introduce interface bottlenecks and compound prediction errors. We present iFLYTEK-Embodied-Omni, a unified multimodal foundation model that jointly models vision(videos and images), language, and action within a single Omni framework. Its modality-specific visual-language, video-generation, and action-generation components communicate through shared multimodal self-attention. This design establishes brain-cerebellum collaboration: the vision-language modeland video generation model form a high-level brain for instruction understanding, task planning, progress tracking, and future visual-state prediction, whereas the action generation modelserves as a low-level cerebellum that directly converts planned subgoals and shared multimodal context into executable action chunks. To develop these capabilities, we combine action-annotated and action-free embodied videos from human demonstrations and robot interactions with embodied reasoning, embodied perception, and general-purpose image-text data to construct a comprehensive dataset. We further adopt a four-stage strategy that progressively trains the VLM, VGM, and AGM before jointly fine-tuning the complete model. 

---
# From Fixed to Free Cameras: Calibration-Free View-Robust Vision-Language-Action Model 

**Authors**: Wenhao Li, Xueying Jiang, Quanhao Qian, Deli Zhao, Shijian Lu, Gongjie Zhang, Ran Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.05396)  

**Abstract**: Real-world robot deployment rarely maintains the training-stage camera setup, where cameras often experience repositioning or remounting depending on actual scenarios. Existing view-robust Vision-Language-Action (VLA) policies tolerate such camera variations only when the camera extrinsics are explicitly provided, making them fragile and hard to use especially when view robustness is critical. We argue that the policy should not be told where the camera is, but rather figure it out by itself. To this end, we introduce Camera-Centric VLA (CamVLA), a new VLA model that decouples manipulation controls from camera geometry by predicting (i) a camera-centric end-effector action expressed in the local camera frame, and (ii) a 6-DoF hand-eye matrix relating cameras to the robot base. A deterministic geometric transformation composes the two predictions into a robot base-frame action. This disentangles how I should move in pose-independent camera-centric action generation from where I am looking from in camera-perspective geometric grounding. The resulting policy is calibration-free, depth-free, and single-view, requiring only a single monocular RGB image as the visual observation and task instruction at deployment. Evaluations in both simulation and real-world robot data show that CamVLA consistently improves success rates across diverse unseen viewpoints. Project page: this https URL. 

---
# Weak-to-Strong Generalization via Direct On-Policy Distillation 

**Authors**: Shiyuan Feng, Huan-ang Gao, Haohan Chi, Hanlin Wu, Zhilong Zhang, Zheng Jiang, Bingxiang He, Wei-Ying Ma, Ya-Qin Zhang, Hao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2607.05394)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) is a powerful recipe for improving language-model reasoning, but it is expensive to repeat on every new strong model because the target model must generate many rollouts during training. As models scale, post-training itself becomes a bottleneck. We study a weak-to-strong alternative: run RL on a smaller model where rollouts are cheaper, then reuse what that RL run learned to improve a stronger target model. Directly distilling the post-RL weak teacher is not enough, because the teacher's final policy mixes useful RL gains with the limitations of the smaller model. We propose Direct On-Policy Distillation (Direct-OPD), which transfers the teacher's RL-induced policy shift instead. Direct-OPD compares the post-RL teacher with its own pre-RL reference and treats their log-ratio as a dense implicit reward for the student. In plain terms, the checkpoint pair tells us which actions RL made the weak model more or less likely to take, and Direct-OPD applies that signal on the stronger student's own on-policy states. This directly reuses the weak model's RL supervision signal without training an explicit reward model or running sparse-reward RL on the target model. Empirically, Direct-OPD consistently leverages weaker teachers to improve stronger target models; notably, it boosts Qwen3-1.7B from 48.3% to 62.4% on AIME 2024 in just 4 hours on 8 A100 GPUs. It outperforms step-matched direct RL and enables the sequential composition of multiple policy shifts. Our results show that RL outcomes can be reused across model scales as implicit reward signals, not merely as final models to imitate. 

---
# Interpretable Human-Label-Free Deep Learning for Real-Bogus Classification with Uncertainty Quantification 

**Authors**: Raphaël Bonnet-Guerrini, Bruno Sanchez, Dominique Fouchez, Benjamin Racine, Maya Guy, Mariam Sabalbal, Manal Yassine, Vincenzo Piuri  

**Link**: [PDF](https://arxiv.org/pdf/2607.05393)  

**Abstract**: Time-domain surveys generate many transient candidates, making Real-Bogus classification a critical step in automated discovery pipelines. Reliable labels are costly, while community labels can be noisy and survey-dependent. We aim to develop a Real-Bogus classification framework that can be trained without human-labeled data using injected transients and bogus-dominated survey data, remains robust under strong class contamination, and provides calibrated uncertainty quantification. We combine simulated transient injections with a contaminated survey class and train a dual-network model using asymmetric co-teaching for classes with different label-noise levels. We evaluate performance on a benchmark subset and analyze the learned representation with latent-space visualization tools. For uncertainty quantification (UQ), we compare MC dropout and deep ensembles and propose a low-cost hybrid strategy that exploits the dual-network setting to improve calibration. We extend the evaluation to the light-curve domain to assess recovery of light-curve classes. The method achieves strong Real-Bogus performance on the labeled subset and remains stable under severe class contamination. It recovers transient light-curve classes with high fidelity, while single-source identification is limited by ambiguity in light-curve-derived labels. Our hybrid UQ approach achieves competitive calibration relative to more expensive ensemble baselines. Latent-space analyses indicate that uncertainty aligns with the decision boundary and reveal subclasses within the bogus population. Our results show that injection-driven, weakly supervised training can enable scalable and consistent Real-Bogus classification without human-labeled training data while providing calibrated uncertainties. The method is suited for transfer to forthcoming surveys by re-running the injection-based training pipeline. 

---
# Search Beyond What Can Be Taught: Evolving the Knowledge Boundary in Agentic Visual Generation 

**Authors**: Haozhe Wang, Weijia Feng, Jinpeng Yu, Che Liu, Ping Nie, Fangzhen Lin, Jiaming Liu, Ruihua Huang, Jimmy Lin, Wenhu Chen, Cong Wei  

**Link**: [PDF](https://arxiv.org/pdf/2607.05382)  

**Abstract**: Visual generators excel at rendering, but they confidently fabricate what they do not know. User requests are unbounded, evolving, and deeply long-tailed: new characters, trending entities, post-cutoff events, and more. This world-knowledge bottleneck is structural: generators are trained on fixed corpora, but the visual world is open-ended. We construct SearchGen-20K and SearchGen-Bench, with 20,839 prompts spanning twelve failure categories and twenty-two domains, paired with a pre-executed multimodal SearchGen-Corpus-1M to support offline, reproducible research. On SearchGen-Bench, frontier open generators score only 21 to 28 out of 100, a 40-point collapse invisible to existing benchmarks. The natural remedy is to employ search tools, enabling agentic visual generation. However, we find that naive search fails: it retrieves indiscriminately, injecting noise into prompts the generator already handles. We trace the root cause to a generator-specific, evolving knowledge boundary: the divide between what a generator can internalize through training and what must remain in external context. Although this boundary is hard to specify in advance, we show that it is discoverable through a teach-then-search co-training framework. Even a minimal version of this co-training recipe produces monotonic improvement, laying the foundation for recursive self-improvement in visual generation that can meet world-knowledge-grounded requests. We release the full dataset, co-training corpus, and search corpus as a replayable harness for tool-augmented, world-knowledge-grounded visual generation. 

---
# What Does a Discrete Diffusion Model Learn? 

**Authors**: Rodrigo Casado Noguerales, Bernhard Schölkopf, Thomas Hofmann, Aran Raoufi  

**Link**: [PDF](https://arxiv.org/pdf/2607.05381)  

**Abstract**: What does a discrete diffusion model learn: a denoiser, a score ratio, or a bridge plug-in predictor? At the level of jump rates, these are one object in different coordinates, and reading a neural network in the wrong coordinate changes the process being trained and sampled. Starting with a rigorous derivation of the continuous-time Markov chain (CTMC) ELBO for any noising process, boundary terms included, we prove the \emph{Oracle Distance} theorem: the negative ELBO is exactly equal to the data entropy plus the path KL from the oracle reverse process to the learned one, not merely a bound. Its unique optimizer is therefore the conditional expectation of the true reverse jump rate given the current noisy state, and its irreducible cost is the rate at which the forward process $Z_t$ destroys information about the clean data $Z_0$, $-\tfrac{d}{dt}I(Z_0; Z_t)$, so every noising process shares the same best achievable negative ELBO: the data entropy. For sequences with token-factorizing noise, the oracle projection yields three exact coordinates for the optimizer: denoiser, cavity (bridge plug-in), and score, with closed-form conversions among them. This framework identifies which law each loss in the literature actually optimizes, recovering MDM, UDM, SEDD, and GIDD as special cases; explains why denoiser and cavity coincide for masked diffusion but not for uniform diffusion; proves that a denoiser parameterization makes the uniform ELBO diverge at initialization while the bridge plug-in stays finite; and calibrates ELBO implementations exactly at initialization. Every identity is verified numerically, without approximation, on an exactly solvable model. 

---
# Cortex: A Bidirectionally Aligned Embodied Agent Framework for Long-horizon Manipulation 

**Authors**: Jiaqi Peng, Xiqian Yu, Delin Feng, Yuqiang Yang, Wenzhe Cai, Jing Xiong, Ganlin Yang, Jinliang Zheng, Jiafei Cao, Xueyuan Wei, Jiangmiao Pang, Yuan Shen, Tai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.05377)  

**Abstract**: While recent Vision-Language-Action (VLA) models show promise toward generalist manipulation policies, they struggle with long-horizon tasks due to their Markovian nature-relying solely on current observations. Hierarchical dual-system methods address this but suffer from a gap between high-level planning semantics and low-level execution kinematics. We introduce Cortex, a bidirectionally aligned embodied agent framework with a customized planning interface that conveys executable and tractable subtask plans from high-level VLM to low-level VLA. Specifically, we standardize manipulation subtasks into 32 canonical skill primitives and inject tractability principles, such as representative object attributes and improved trajectory reachability, into the data generation pipeline. This enables automatic annotation of over 4k hours of open-source video data and generation of 30 hours of simulation data. We further devise an event-balanced sampling strategy to construct training data for fine-tuning the framework to better handle planning ambiguity during subtask transitions, enhanced by carefully designed harness engineering from task contexts to skill constraints during inference. Both open-loop VLM and closed-loop system evaluations demonstrate Cortex's efficacy, e.g., it outperforms monolithic baselines by 3.1% on Libero-long and 4.1% on RoboTwin. Notably, Cortex's generalist VLM enables zero-shot completion of unseen real-world long-horizon tasks, such as multi-stage chemistry experiments, by simply combining with a fine-tuned VLA-a capability infeasible through VLA fine-tuning alone. 

---
# GaP: A Graph-as-Policy Multi-Agent Self-Learning Harness For Variational Automation Tasks 

**Authors**: Kaiyuan Chen, Shuangyu Xie, Letian Fu, Justin Yu, William Pacini, Sandeep Bajamahal, Hudson Kim, Jaimyn Drake, Daehwa Kim, Haoru Xue, Jonathan Francis, Christian Juette, Peter Schaldenbrand, Muhammet Yunus Seker, Ruwan Wickramarachchi, Uksang Yoo, Guanzhi Wang, Adithyavairavan Murali, Balakumar Sundaralingam, S. Shankar Sastry, Spencer Huang, Yuke Zhu, Linxi "Jim" Fan, Ken Goldberg  

**Link**: [PDF](https://arxiv.org/pdf/2607.05369)  

**Abstract**: For robots to work reliably in commercial and industrial applications, can recent advances in agentic coding systems combine interpretable robot programming with the open-world adaptability of model-free policies? We focus on "Variational Automation" (VA), a class of tasks that have larger variations in object geometry and pose than fixed automation. Model-free policies often struggle to close the reliability gap for VA tasks, which must be executed persistently and reliably in commercial and industrial applications. Motivated by prior work on Task and Motion Planning (TAMP) and the Robot Operating System (ROS), we introduce Graph-as-Policy (GaP), a multi-agent coding harness that generates directed computation graphs with perception, planning, and control nodes from a Modular Open Robot Skill Library (MORSL). GaP then generates an internal simulation environment to rehearse task instances with different graphs in parallel to iteratively refine the graph structure and parameters to improve success rates and throughput. Evaluation with 8 new open VA task benchmarks, 4 in-simulation and 4 in real-world, suggests that GaP can achieve success rates that significantly outperform baselines. Details, code, and data can be found online: this https URL 

---
# SPEARBench: A Benchmark for Naturalness Evaluation in Streaming Speech-to-Speech Language Models 

**Authors**: Thomas Thebaud, Yuzhe Wang, Hao Zhang, Sathvik Manikantan Napa Ugandhar, Ashish Hallur, Georgi Tinchev, Venkatesh Ravichandran, Laureano Moro-Velazquez  

**Link**: [PDF](https://arxiv.org/pdf/2607.05365)  

**Abstract**: Streaming speech-to-speech language models aim to answer spoken queries directly with synthetic speech. However, standard speech and text benchmarks do not capture whether these systems behave naturally in conversations, where timing, turn-taking, prosody, interpersonal stance, language and dialect consistency, and relationship-aware appropriateness jointly shape perceived quality. We introduce SPEARBench, a benchmark for evaluating naturalness in speech-to-speech language models from question-answer interactions. SPEARBench constructs controlled dialogue prompts from the Seamless Interaction corpus, runs inference across multiple models, and evaluates generated answers using a multidimensional protocol that covers response latency, interruptions, speech quality, ASR robustness, language and dialect consistency, emotional naturalness, interpersonal stance, and explainable distributional baselines. The benchmark includes original human answers as a reference condition and reports results for several contemporary models. Results show that current models can achieve high signal-level quality and low ASR error while still differing from human conversational behavior in latency, overlap, dialect preservation, emotional adaptation, and interpersonal stance dynamics. 

---
# REDDIT: Correcting Model-Generated Timestamp Drift in ASR without Forgetting via Replay-Based Distribution Editing 

**Authors**: Cheng-Kang Chou, Ming-To Chuang, Ke-Han Lu, Chan-Jan Hsu, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2607.05364)  

**Abstract**: Modern autoregressive ASR systems can emit timestamps as decoded tokens, enabling timestamped transcription without frame-level aligners or inference-time post-processing. We show that these generated timestamps can drift across long non-speech spans: the transcript may remain plausible, but the decoded time axis drifts away from the audio. We study this non-speech-induced timestamp drift with self-built gap and long-gap benchmarks across 15 evaluated timestamp-producing ASR and audio-language systems. Naive timestamp-corrected fine-tuning improves alignment but can severely degrade non-target ASR behavior, exposing a forgetting problem. We propose REDDIT(REplay-based Distribution eDITing), a lightweight two-stage post-training framework that corrects timestamps while avoiding this catastrophic forgetting: it first edits timestamp targets under the model's own replayed decoder context while matching the frozen base distribution on non-timestamp tokens, then applies a short edited-prefix refinement stage. In this framework, we construct correction supervision without human transcripts or human timestamp annotations by combining VAD-trimmed speech spans with inserted non-speech gaps and known concatenation offsets. On Whisper-tiny, 34.9 hours of targeted correction audio used and only 1.6% of model parameters updated, raising long-gap mIoU from 38.7% to 95.0% and reducing mixed-gap out-of-domain AAS from 2752 ms to 223 ms while preserving CV-en MER at 41.3% (versus 524.2% for ordinary SFT decoder tuning). 

---
# Selective Disclosure Watermarking for Large Language Models 

**Authors**: Xuyang Chen, Xiang Li, Yangxinyu Xie, Qi Long  

**Link**: [PDF](https://arxiv.org/pdf/2607.05353)  

**Abstract**: Watermarking methods embed imperceptible and verifiable signals into text generated by large language models (LLMs). Existing approaches include zero-bit schemes for distinguishing synthetic text from human writing and multi-bit schemes for embedding metadata. However, current multi-bit watermarking methods do not allow selective disclosure: verifying any part of the watermark requires revealing the entire embedded message. This lack of control leads to unnecessary information exposure and raises privacy concerns. We propose Hierarchical Vocabulary Routing (HeRo), a watermarking framework that enables selective disclosure of embedded metadata. The method recursively partitions the vocabulary and distributes watermark information across hierarchical layers, so that different verifiers can decode only the portions of the payload corresponding to their access level. We show that the proposed scheme preserves the unbiasedness of the underlying sampling process and thus maintains text quality. Experiments demonstrate that our framework supports fine-grained access control while achieving high detection accuracy and low latency. Code is available at this https URL. 

---
# Multiplayer Interactive World Models with Representation Autoencoders 

**Authors**: Anthony Hu, Václav Volhejn, Adrien Ramanana Rahary, Chris Mulder, Aditya Makkar, Amélie Royer, Manu Orsini, Alyx Liao, Adam Jelley, Eloi Alonso, Florian Laurent, Fredrik Norén, James Swingos, Jan Hünermann, Kent Rollins, Lucas Hosseini, Matthieu Le Cauchois, Maxim Peter, Pim de Witte, Tim Brown, Vincent Micheli, Moritz Böhle, Gabriel de Marmiesse, Viktoriia Sharmanska, Lucia Specia, Michael Black, Patrick Pérez  

**Link**: [PDF](https://arxiv.org/pdf/2607.05352)  

**Abstract**: We introduce the first multiplayer world model for highly dynamic environments governed by complex physical interactions. Whereas single-player world models treat the other agents as part of the environment, ours conditions on the action streams of multiple agents, learning to attribute changes in the scene to the correct player and to stay coherent under arbitrary combinations of their actions. We study this problem in the game of Rocket League, where players compete and cooperate under fast, tightly coupled dynamics. Trained on 10,000 hours of gameplay collected with publicly available bots, our 5-billion-parameter latent diffusion model generates four-player matches in real time, producing 20 frames per second on a single Nvidia B200 GPU. Although trained only on short clips, its rollouts stay stable far beyond the training horizon: distributional quality holds steady out to five minutes, the longest horizon we measure, and in practice we observe rollouts continuing for hours with no sign of collapse. We systematically investigate the central design choices: the video codec, the generative objective, and the multiplayer conditioning scheme. In addition, we characterize how behavior changes with model and data scale, including the capabilities that emerge and the failure modes that persist. We further develop targeted evaluations that probe the model's physical understanding rather than visual appearance alone. To support continued research on multiplayer world models, we release our dataset, our full training and inference codebase, and a live demo. 

---
# TREK: Distill to Explore, Reinforce to Refine 

**Authors**: Yuanda Xu, Zhengze Zhou, Kayhan Behdin, Jelena Markovic-Voronov, Hejian Sang, Xiaomin Li, Wenhui Zhu, Xinchen Du, Aida Rahmattalabi, Ran He, Sen Na, Zhipeng Wang, Alborz Geramifard  

**Link**: [PDF](https://arxiv.org/pdf/2607.05339)  

**Abstract**: Group Relative Policy Optimization (GRPO) is effective when the current policy already samples useful reasoning trajectories, but it stalls on hard prompts whose correct solution modes lie outside the student's on-policy support. We propose TREK (Teacher-Routed Exploration via Forward KL), a simple staged procedure that uses distillation not for imitation but for exploration support expansion. A key advantage of TREK is its generality: because it only consumes verified output trajectories, it can use an external black-box teacher, a white-box teacher, or the same model given additional inference-time context, and it can efficiently identify which hard-prompt samples are most worth consolidating even when teacher internals are unavailable. TREK first identifies prompts where the unaided student has very low pass rate, queries a proposal source to produce verified candidate solutions, keeps the top-$r$ proposals ranked by current student likelihood, applies a short forward-KL phase to pull those verified modes into the student's support, and then returns to standard on-policy GRPO refinement. On mathematical reasoning, TREK with DeepSeek-V4 proposals improves Qwen3 models across all tested scales on AIME 2024 and AIME 2025; for Qwen3-8B, it improves AIME 2025 from 36.9 to 40.3 and AIME 2024 from 47.9 to 51.1 (avg@16), while the self-context variant reaches 38.5 and 49.6 without an external teacher. On agentic tasks, TREK raises ALFWorld success rate from 75.8 to 82.8 and ScienceWorld success rate from 12.5 to 26.7; notably, on the hardest task types, TREK achieves high success rates early in training while unaided GRPO requires substantially more optimization steps to reach comparable levels. 

---
# Steering Optimisation Trajectories in Diffusion Representation Learning 

**Authors**: Rajat Rasal, Avinash Kori, Tian Xia, Ben Glocker  

**Link**: [PDF](https://arxiv.org/pdf/2607.05319)  

**Abstract**: We study why diffusion autoencoders can achieve similar image quality while learning substantially different latent structures. We trace this behaviour to optimisation dynamics; we analyse curves of image reconstruction against latent representation quality, revealing trajectories that organise around two distinct regimes early in training. Models in the reconstruction regime prioritise image fidelity early, whereas those in the disentanglement regime improve reconstruction and disentanglement more gradually. We hypothesise that this behaviour can be influenced by targeting shortcut pathways in the diffusion U-Net and controlling early noise-level exposure, thereby shaping the reconstruction-disentanglement trade-off during training. To steer optimisation toward stronger representations, we introduce SteeringDRL, combining gated residual U-Nets with a simple noise-level exposure curriculum for training. Across disentanglement benchmarks, SteeringDRL improves representation quality and reduces seed sensitivity. Our method further extends to spatial disentanglement in object-centric learning, improving segmentation quality on synthetic and real-world datasets. 

---
# Topological Shape Representation for Aneurysm -- Bifurcation Detection 

**Authors**: Akshay Gokhale, Mansi Dhamne  

**Link**: [PDF](https://arxiv.org/pdf/2607.05317)  

**Abstract**: Automated detection of intracranial aneurysms (IAs) from CT angiography (CTA) is severely hindered by high false-positive rates. Convolutional neural networks (CNNs) rely on local pixel intensities, causing systematic confusion between saccular aneurysms and vascular bifurcations -- a problem especially acute for small lesions (<3 mm), where detection sensitivity falls below 60%. We propose a plug-and-play, topology-aware false-positive reduction framework evaluating the Smooth Euler Characteristic Transform (SECT) -- a directional representation encoding global 3D vascular geometry independently of intensity -- against persistence-based summaries (Persistence Images and Landscapes), tested on a stratified subset of the RSNA 2025 dataset. SECT achieves an AUC of 0.943, substantially outperforming direction-agnostic methods (AUC ~0.68), and exhibits a clinical performance inversion: it excels on the sub-3 mm cohort, maintaining 0.943 AUC and 78.5% sensitivity at 95% specificity. The representation is also scanner-agnostic, achieving 0.927 mean AUC under leave-one-scanner-out (LOGO) validation across four manufacturers. By capturing asymmetric geometric invariants rather than intensity profiles, SECT reliably resolves the primary structural confounder in IA detection, positioning it as a robust downstream filter for hybrid deep-learning diagnostic pipelines. 

---
# Air Quality Downscaling with Station-Guided Pseudo-Supervision 

**Authors**: Guorun Wang, Simone Foti, Andreas D. Demou, Leonidas Kotoulas, Theodoros Christoudias, Alexandros Koliousis, Mihalis Nicolaou, Stefanos Zafeiriou  

**Link**: [PDF](https://arxiv.org/pdf/2607.05292)  

**Abstract**: Super-resolving coarse atmospheric fields to local PM$_{2.5}$ variations is uniquely challenged by a mismatch in spatial support: while pixels represent regional averages, ground-truth observations are discrete, unaligned samples of a continuous spatial signal. To bridge this gap, we present a station-guided framework for high-resolution PM$_{2.5}$ downscaling over Europe. Taking coarse CAMS atmospheric composition fields alongside heterogeneous side information (i.e., human activity, land cover, elevation, satellite aerosol observations, and wind fields) our framework jointly super-resolves ($\times 40$, $\approx$ 1 km) and bias-corrects CAMS rasters, without relying on temporal sequence modelling. To address the challenge of densely supervising our multi-scale transformer network with sparse in-situ data, we introduce a time-agnostic propagation strategy that utilises spatial Gaussian blending of interpolated OpenAQ observations. Extensive qualitative and station-level evaluations across Europe demonstrate that our model recovers fine-grained spatial structures and effectively mitigates localised CAMS biases. 

---
# Wavelet Scattering Transform for Interpretable Schizophrenia Biomarker Discovery and Classification from Resting-State EEG 

**Authors**: Md. Taksimul Ahsan Tawhid, Nasif Ahmed Rafe, Alif Tahmid Priyom, K. M. Mustafizur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2607.05282)  

**Abstract**: Schizophrenia is a debilitating neuropsychiatric disorder characterized by profound cortical network dysregulation, for which objective, clinically translatable EEG based biomarkers remain underdeveloped. Existing automated classification pipelines rely predominantly on static power spectral density features inherently blind to amplitude modulation dynamics and cross-frequency coupling, phenomena central to schizophrenia pathophysiology, while adopting epoch level cross validation strategies that introduce temporal data leakage, artificially inflate reported performance. This study introduces a mathematically principled diagnostic framework integrating the multi-order Wavelet Scattering Transform(WST), strict Leave One Subject Out (LOSO) cross-validation, and SHAP explainability for simultaneous EEG classification and biomarker discovery. Hierarchical WST coefficients capturing multi-scale amplitude modulation structure were extracted from resting state multichannel EEG. Subject-level ANOVA with Benjamini Hochberg false discovery rate correction identified significant biomarkers, with Random Forest and SVM classifiers evaluated under strict LOSO cross validation and subject-level majority voting. Second-order scattering coefficients encoding cross frequency coupling dominated the discriminative biomarker set, with gamma-band features most prevalent, demonstrating that temporal amplitude modulation constitutes the primary electrophysiological signature of schizophrenia. Electrode P3 was identified as the single most discriminative site. Under rigorous subject independent evaluation, the Random Forest achieved 90.48% accuracy (AUC = 0.9339; sensitivity = 95.56%). The proposed WST framework establishes a rigorous, interpretable standard for EEG-driven psychiatric biomarker discovery that can also be applicable in the detection of schizophrenia subtypes in the future. 

---
# ProPS: Prompted Profile Synthesis for Natural Language-Conditioned Speaker Embedding Distributions 

**Authors**: Thomas Thebaud, Junhyeok Lee, Laureano Moro-Velazquez, Jesus Villalba Lopez, Najim Dehak  

**Link**: [PDF](https://arxiv.org/pdf/2607.05276)  

**Abstract**: Speaker embeddings, or x-vectors, are widely used to represent speaker identity and speaker-related attributes, but existing embedding extractors are typically descriptive rather than generative: they map an observed speech segment to an x-vector, which is then used for downstream applications. We introduce ProPS, Prompted Profile Synthesis, a framework for generating distributions of speaker embeddings conditioned on natural language prompts such as "a thirties male speaker with an Indian accent". ProPS converts human-written profile descriptions into sentence embeddings and uses a mixture density network trained on a large-scale dataset to predict a Gaussian mixture model in the x-vector space. The model is trained by maximizing the likelihood that real speaker embeddings match the requested profile, and its generated distributions are evaluated by negative log-likelihood on held-out x-vectors and by attribute classification accuracies on sampled synthetic x-vectors. Experiments show that ProPS produces profile-conditioned distributions and generates x-vectors that preserve requested speaker attributes such as age, gender, accent, and prosodic characteristics. This design enables controllable speaker-profile synthesis for speech generation systems like Text-To-Speech (TTS) or Voice Conversion (VC) while anchoring generated distributions in observed speaker-embedding structure. 

---
# Adaptive Inference Batching using Policy Gradients 

**Authors**: Ruslan Sharifullin  

**Link**: [PDF](https://arxiv.org/pdf/2607.05272)  

**Abstract**: Inference serving systems must balance throughput and latency under bursty, heterogeneous workloads, yet the industry standard remains static batching policies that require manual tuning and cannot adapt to shifting traffic. We investigate whether reinforcement learning (RL) can learn adaptive batching and routing policies that outperform these heuristics, training REINFORCE and PPO agents on a discrete-event simulator validated against queuing theory and production traces (Azure Functions, BurstGPT). We formulate the problem as an MDP over queue state, request type and GPU availability, evaluating across standard Poisson traffic, extreme bursts, real-world traces and heterogeneous multi-GPU routing.
Our central finding is a clear boundary condition for RL's value in systems problems. In single-GPU settings, a well-tuned static batching policy is already near-optimal under Poisson-like arrivals and RL offers only marginal gains (+0.1% to +1.0%). In multi-GPU heterogeneous routing, however, where fast and slow requests compete for shared resources, the agent discovers a workload-segregation policy that eliminates Head-of-Line blocking, yielding a 3.5x (348%) improvement over Round-Robin and a 48% improvement over the strongest heuristic baseline (Shortest-Queue), with 60% higher throughput and 25% lower latency while respecting SLA constraints. The policy generalizes to unseen bursty and real-world traffic despite training only on synthetic Poisson arrivals and an attention-augmented policy network converges roughly 20% faster than an MLP baseline.
These results suggest RL's advantage over engineered heuristics concentrates in combinatorial, multi-resource decisions rather than single-resource temporal scheduling, a practical distinction for deciding where learned policies justify their engineering cost in production inference infrastructure. 

---
# Shifting from Discrete to Continuous Reference Data: QSM-Derived Horizontal Tree Biomass Distribution for Deep Learning Biomass Estimation 

**Authors**: Nils Griese, Christoph Kleinn, Nils Nölke  

**Link**: [PDF](https://arxiv.org/pdf/2607.05260)  

**Abstract**: Conventional modeling approaches for LiDAR-based above-ground biomass (AGB) estimation rely on discrete plot-level inventory aggregates. This methodology introduces boundary-effect uncertainties that may severely degrade model performance within small field plots. To solve this limitation, we evaluate a Horizontal Biomass Distribution (HBD) reference mapped continuously from Quantitative Structure Models (QSMs). We trained a sparse 3D U-Net on simulated broadleaved forest structures using three AGB reference types: a standard forest inventory (FI) plot-level aggregate, an edge-effect-free QSM plot-level aggregate, and a continuous HBD mapping. Evaluating training plot sizes scaling from 100 to 2500 $m^2$ , QSM-based models systematically outperformed FI approaches at small plot sizes. Specifically, for 100 $m^2$ plots, the HBD reference reduced the relative root mean square error (RRMSE) by 16.84 $\pm$ 4.37 % and increased $R^2$ by 0.22 $\pm$ 0.05 against the FI baseline. By replacing plot level aggregates with HBDs as AGB reference, this methodology corrects for edge-effects and shows that using an HBD-based reference enhances model performance for small plot sizes. 

---
# Privacy-Preserving Robustness Verification for Neural Networks 

**Authors**: Nianyun Song, Xiaokun Luan, Yu Guo, Rongfang Bie, Meng Sun, Xiyue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.05251)  

**Abstract**: Neural network verification and data privacy are inherently in tension: verification demands full access to model parameters and input data, yet both are increasingly restricted by privacy regulations and intellectual property constraints. This tension has left robustness verification impractical in privacy-sensitive domains. In this work, we address this gap with SecureCROWN, the first framework for privacy-preserving neural network robustness verification. Built upon secure two-party computation (2PC), our framework enables a model owner and a data owner to jointly compute certified robustness bounds -- revealing only the final result while provably protecting both parties' private data under the semi-honest security model. A key challenge is securely computing the conditional operations in Linear Bound Propagation, where the data-dependent branching is incompatible with standard secure computation protocols. We eliminate branching by formulating conditional logic as continuous arithmetic operations. Additionally, we introduce a Newton--Raphson refinement method to improve numerical stability. Extensive analysis and experiments show that SecureCROWN strictly matches plaintext verification results, while completing in 0.1--200s across varied model sizes and communication settings (LAN/WAN), demonstrating the feasibility of privacy-preserving neural network verification. 

---
# CanniUplift: A Holistic Framework for Mitigating Seller and Incentive Cannibalization in E-commerce Uplift Modeling 

**Authors**: Zuwang He, Shihao Shu, Yuli Qu, Hanyu Gao, Ziliang Zhang, Diwei Chen, Xiangda Yan, Buyu Gao, Tanchao Zhu, Yumeng Li, Junxiong Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.05242)  

**Abstract**: Personalized incentive allocation is vital for e-commerce, where uplift modeling is the standard for estimating Individual Treatment Effects (ITE). However, traditional models often fail in complex multi-seller environments with violations of the Stable Unit Treatment Value Assumption (SUTVA). We identify two critical challenges: Seller-level Cannibalization, where incentives shift expenditure between shops without growing the platform, and Incentive-level Cannibalization, where organic conversions or alternative rewards introduce significant noise into incrementality estimation. In this paper, we propose CanniUplift, a unified framework to mitigate these dual-source cannibalization effects. Specifically, we design Platform-level Global Alignment (PGA) to capture cross-shop substitution through global GMV consistency constraints. To tackle incentive-driven noise, we introduce Redemption-based Decomposition Denoising (RDD), which uses redemption behavior to decompose treated outcomes and reduce attribution noise within an entire-space framework. Furthermore, a Treat-Attention mechanism is designed to model intricate interactions between users' historical behaviors and current treatment options. Extensive experiments on both synthetic and large-scale industrial datasets demonstrate that CanniUplift significantly outperforms state-of-the-art baselines. Ablation studies confirm that the integration of PGA and RDD consistently improves wAUUC and wQINI. Successfully deployed online, our framework achieved a 4.08% relative increase in platform-wide incremental GMV (Delta GMV) over the production baseline and improved ROI in online A/B tests, proving effective in driving global platform growth. 

---
# Optimizing ML Workload Partitioning between CPUs and CIM Accelerators for Heterogeneous Computing 

**Authors**: Joel Klein, Rebecca Pelke, Roberto Laudani, Jan Moritz Joseph, Rainer Leupers  

**Link**: [PDF](https://arxiv.org/pdf/2607.05240)  

**Abstract**: Computing-in-Memory (CIM) accelerators execute Matrix-Vector Multiplications (MVMs) in memory, making them a compelling solution for Machine Learning (ML) workloads. However, existing ML workload partitioning approaches for CIM accelerators do not fully account for Resistive Random Access Memory (RRAM) constraints such as limited memory, high write latency, and limited endurance. They also neglect parallelism, low-level architectural effects, or the Central Processing Unit (CPU) as a complementary compute resource. To address these limitations, we propose an Integer Linear Programming (ILP)-based workload partitioning framework for heterogeneous CPU-CIM systems. It minimizes end-to-end inference latency under RRAM constraints, captures parallelism, and combines empirical profiling with analytical models. Using our framework, heterogeneous CPU-CIM execution achieves speedups of up to 30.9x over CPU-only execution on an edge CPU and 7.3x over a high-performance CPU. A Design Space Exploration (DSE) yields further design insights for future CIM accelerators. 

---
# Noisy-Channel Minimum Bayes Risk Decoding 

**Authors**: Yusuke Sakai, Hidetaka Kamigaito, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2607.05198)  

**Abstract**: Minimum Bayes Risk (MBR) decoding yields more robust and higher-quality text generation than maximum a posteriori (MAP) decoding by selecting hypotheses that maximize expected utility over sampled pseudo-references. However, there exists a discrepancy in the design: hypothesis selection calculates expected utility scores conditioned on given pseudo-references, while commonly used evaluation metrics, e.g., BLEU and COMET, are asymmetric. Therefore, it is important to consider both hypothesis-to-reference and reference-to-hypothesis directional effects. In this study, we introduce a noisy channel decomposition of MBR decoding that naturally incorporates bidirectional effects to account for these asymmetries. We decompose MBR decoding into four interacting components: hypothesis-to-reference likelihood, reference-to-hypothesis likelihood, hypothesis prior, and reference prior. This decomposition provides a unified interpretation of existing MBR variants and enables metric- and task-specific interpretability by isolating the contribution of each channel. Our comprehensive analysis reveals that channel-wise contributions exhibit distinct characteristics across metrics while remaining consistent across tasks, and suggests that appropriate channel weighting may lead to improvements over original MBR decoding. 

---
# Unified Audio Intelligence Without Regressing on Text Intelligence 

**Authors**: Zhifeng Kong, Sang-gil Lee, Jaehyeon Kim, Boxin Wang, Zihan Liu, Sungwon Kim, Yang Chen, Arushi Goel, Rajarshi Roy, Wenliang Dai, Zhuolin Yang, Yangyi Chen, Dongfu Jiang, Sreyan Ghosh, Tuomas Rintamaki, Andrew Tao, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, Wei Ping  

**Link**: [PDF](https://arxiv.org/pdf/2607.05196)  

**Abstract**: Audio intelligence involves understanding, reasoning about, and generating both audio and speech. In this work, we introduce Nemotron-Labs-Audex-30B-A3B (Audex), a unified audio-text LLM built on Nemotron-Cascade-2-30B-A3B, a strong text-only MoE LLM. Audex adopts a simple unified design with a single Transformer decoder: audio inputs are encoded and projected into the text embedding space, while text tokens and quantized audio output tokens are treated uniformly during generation. This architecture enables strong audio-text fusion, seamless multimodal generation, and compatibility with standard LLM training and inference infrastructure. For training, we meticulously curate audio-text datasets comprising 157.4B audio tokens and 320.5B text tokens. We apply multi-stage supervised training on these datasets, followed by text-only Cascade RL and multi-domain on-policy distillation. Audex delivers state-of-the-art audio understanding, speech recognition and translation, text-to-speech, audio generation, and speech-to-speech generation, while preserving very compelling reasoning, alignment, knowledge, long-context, and agentic capabilities of its text-only LLM backbone with marginal or no regression. We release the model checkpoints to facilitate open research. 

---
# When Claws Remember but Do Not Tell: Stealthy Memory Injection in Persistent Personal Agents 

**Authors**: Yechao Zhang, Shiqian Zhao, Jiawen Zhang, Jie Zhang, Gelei Deng, Xiaogeng Liu, Chaowei Xiao, Tianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.05189)  

**Abstract**: Persistent personal agents combine long-term memory with access to users' external environments, enabling personalized foreground assistance and proactive background execution. This integration also creates a new path to compromise: untrusted external content can be silently written into persistent memory and later reused as trusted state. We study this threat as stealth memory injection, in which a remote black-box adversary delivers a single email payload that must induce the agent to write poisoned memory, stay hidden in the agent's response to the user, and affect future behavior.
We introduce WhisperBench, a 108-case benchmark spanning five risk categories and both fact and preference poisoning. Built on a real IMAP/SMTP workflow and an authentic email agent skill, it enables full-cycle evaluation of stealth memory injection attacks. To enable this black-box attack under single-email delivery and without runtime feedback, we propose MemGhost, a one-shot payload generation framework. MemGhost uses an environment proxy to emulate persistent-agent execution and an objective proxy to convert memory adoption and conversational stealth into dense rubric-based rewards, then trains the attacker policy with supervised fine-tuning and reinforcement learning.
Across 56 held-out test cases, MemGhost achieves 87.5% end-to-end success on OpenClaw with GPT-5.4 and 71.4% on Claude Code SDK with Sonnet 4.6. It also transfers across personal-agent architectures (NanoClaw and Hermes Agent) and memory backends (filesystem and vector-based Mem0), and remains effective against input-level, model-level, and system-level defenses. These results suggest that persistent memory can turn ordinary external processing into a practical pathway for long-term agent compromise. 

---
# Relational Multi-Agent Reinforcement Learning for Dynamic Pricing in High-Speed Railway Markets 

**Authors**: Enrique Adrian Villarrubia-Martin, David Muñoz-Valero, Luis Rodriguez-Benitez, Giovanni Montana, Luis Jimenez-Linares  

**Link**: [PDF](https://arxiv.org/pdf/2607.05179)  

**Abstract**: In liberalised railway systems, operators must set prices dynamically in an environment with partial observability, as they retain private information about their objectives and performance, where regulatory constraints prohibit communication or direct information exchange between competitors to prevent explicit collusion. Consequently, agents must learn to infer strategic interactions only from observable market data which presents a significant challenge for multi-agent reinforcement learning, where standard approaches typically treat observations as unstructured vectors, ignoring the underlying market topology that governs strategic interactions. To address this, an entity graph modelling approach is proposed, which represents the environment as a graph of operational units, rather than decision-making agents or static infrastructure, encoding competition, coordination, and connectivity relations between entities. Then, an extension of the multi-agent twin delayed deep deterministic policy gradient algorithm with graph-based representation learning processes the features of the entities through a multi-layer relational graph convolutional network and aggregates them via a learnt attention mechanism. Experimental results in a rail pricing reinforcement learning environment show that this novel framework achieves higher revenue and stability in two different settings of increasing market complexity compared to a representative selection of relational and non-relational baselines. The code is publicly available at: this https URL 

---
# Open Problems in AI Incident Governance 

**Authors**: Harleen Kaur Sidhu, Rebecca Scholefield, Nour Annan, Kevin Hernandez, Isabel Nieh Hou, Abdulrahman Alshaikhi, Ze Shen Chin, Rokas Gipiškis  

**Link**: [PDF](https://arxiv.org/pdf/2607.05163)  

**Abstract**: AI systems may produce failures after deployment that pre-deployment safety assessments do not anticipate. Managing these failures requires what we refer to as adequate \textit{AI incident governance}, where having good definitions, taxonomies, monitoring practices, reporting mechanisms, and incident analysis is essential. We examine existing frameworks related to AI incident governance by regulatory bodies and independent efforts, and find that while there are frameworks that describe how individual functions can be performed, there is a lack of consistency within the aspects of definitions, classification, monitoring, and reporting. These inconsistencies apply to the types of incident data that is collected and reported, the ways in which they are categorised, and as a result, the depth, representativeness, and accuracy of analysis that can be performed. 

---
# PDEFlow: Autonomous Agentic PDE Pipelines for Neural Operator Learning and Solver-Free Inference 

**Authors**: Akshat Jani, Prathamesh Gadekar, Sakhinana Sagar Srinivas, Venkataramana Runkana  

**Link**: [PDF](https://arxiv.org/pdf/2607.05134)  

**Abstract**: We present PDEFlow, an autonomous agentic framework that turns user-level ODE and PDE descriptions into solver-backed neural-operator pipelines. The workflow links problem specification, data generation, operator training, and checkpoint-based inference. A stateful input graph converts multi-turn natural-language input and user edits into validated problem specifications. The data-generation module then samples parameters, solves the configured governing-equation with FEniCSx finite-element backend, and stores the solutions as operator-ready tensors. The training and inference stages use a registry-based interface, allowing different neural operators to be trained and deployed without changing the surrounding pipeline. In the current implementation, we instantiate this interface with a multi-branch Bayesian DeepONet. Experiments on benchmark ODE and PDE tasks show that PDEFlow can construct valid specifications, generate solver-backed datasets, train neural operators across steady and transient problem classes, and provide solver-free predictions from saved checkpoints. The framework is designed for repeatable scientific and engineering workflows where many related physics configurations must be specified, simulated, learned, and queried with minimal manual intervention. 

---
# Three-Phase Evaluation of AI-Assisted Software Development Life Cycle 

**Authors**: Joshua Strubel, Professor Carrie Russell, Carson Crockett, Jason Ferraro, Nathan Londhe, Uzayr Syed, Jacob Viehe  

**Link**: [PDF](https://arxiv.org/pdf/2607.05125)  

**Abstract**: This paper presents an exploratory evaluation of how increasing levels of AI autonomy affect software development productivity, requirement adherence, and developer cognitive workload. A team of four developers reimplemented the same full-stack web application across three sequential phases: partial AI-assisted development using GitHub Copilot, an AI-exclusive workflow using GitHub Copilot, and an AI-exclusive workflow using AWS Kiro. Evaluation metrics included development effort (hours), requirement adherence (RITM score), AI-interaction efficiency, and NASA-TLX workload measures. Across phases, higher levels of AI autonomy were associated with reduced development effort, improved requirement adherence, and lower self-reported mental workload, while developer frustration increased modestly. The AWS Kiro phase achieved the strongest overall performance on most measured dimensions, suggesting that tooling architecture may influence outcomes independently of AI autonomy level. 

---
# Agent Data Injection Attacks are Realistic Threats to AI Agents 

**Authors**: Woohyuk Choi, Juhee Kim, Taehyun Kang, Jihyeon Jeong, Luyi Xing, Byoungyoung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2607.05120)  

**Abstract**: AI agents act on behalf of user prompts, consuming external data and taking actions based on the agent context. Prior research on AI agent security has primarily focused on indirect prompt injection (IPI). Its most well-studied category is instruction injection, where attacker-controlled untrusted data is interpreted as an instruction. In response, many mitigations have been proposed to prevent instruction injection attacks. In this paper, we introduce a new category of IPI, agent data injection attacks (ADI). ADI injects malicious data disguised as trusted data, such as security-critical metadata (e.g., resource identifiers or data origins) or agent context data (e.g., tool call and response formats). As a result, agents unknowingly execute unintended actions based on attacker-controlled data. ADI has similar attack impacts as instruction injection attacks, because it causes agents to misbehave and execute unintended actions. Despite the similar impact, ADI remains underexplored and easily bypasses existing IPI defenses. We found several critical vulnerabilities in real-world agents that allow an attacker to launch various attacks: arbitrary click attacks on web agents (Claude in Chrome, Antigravity, and Nanobrowser), and remote code execution and supply-chain attacks on coding agents (Claude Code, Codex, and Gemini CLI). We evaluate ADI vulnerabilities across off-the-shelf models and AI agents, and find that ADI is effective in both standalone LLMs and AI agent settings. ADI exposes a critical gap in agent security, signifying that current AI agents do not employ a fundamental security principle: current agents do not isolate trusted data from untrusted data. 

---
# Localized LoRA-MoE: Block-wise Low-Rank Experts With Adaptive Routing 

**Authors**: Babak Barazandeh, Subhabrata Majumdar, Vinay Prithyani, George Michailidis  

**Link**: [PDF](https://arxiv.org/pdf/2607.05114)  

**Abstract**: Large Language Models (LLMs) and high-dimensional perception networks increasingly rely on parameter-efficient fine-tuning (PEFT) to adapt to diverse operational contexts. However, standard methods like LoRA are structurally limited by a monolithic bottleneck, making them highly susceptible to gradient warfare. Interleaved multi-task streams may trigger destructive optimization feedback, collapsing adapter weights into unspecialized averages. While recent spatial partitioning methods have introduced block-wise isolation, they remain trapped in static topologies, unable to adapt to dynamic task-switching or environmental sensor failure. In this work, we introduce Localized LoRA-MoE, a unified framework that fuses localized spatial blocking with dynamic, context-conditioned routing. We propose and evaluate two novel architectural paradigms: Block-Wise LoRA-MoE (Centralized Macro-Routing), which modulates the entire structural grid via a monolithic context signal, and Cell-Wise LoRA-MoE (Decentralized Micro-Routing), which empowers every coordinate cell in the matrix grid with autonomous, localized expert gating. Through a comprehensive suite of benchmarks, ranging from high-dimensional SVD matrix simulations and real-world tabular transformations to spatial vision perception under sensor degradation, we demonstrate that both architectures resolve optimization deadlocks inherent in static baselines. Our empirical results establish that decentralized cell-level gating achieves complete statistical parity with an omniscient global coordinator, providing a robust "gradient firewall" that protects surviving pathways from fault-propagated corruption. Our proposals consistently outperform static baselines, offering a scalable and parameter-efficient solution for dynamic model adaptation across granular coordinate fields and shifting operational regimes. 

---
# Grokking Is Conditional and Fragile: A Fully-Tractable, Multi-Seed Study at 12K Parameters 

**Authors**: Yoshiyuki Ootani  

**Link**: [PDF](https://arxiv.org/pdf/2607.05104)  

**Abstract**: Grokking -- the delayed onset of generalization long after a network has fit its training set - -is usually studied in models too large to read completely and reported from single training runs. We instead study a publicly released ~11,856-parameter Llama-style transformer (Glimmer-1-Base) on modular arithmetic, small enough to enumerate its weights, attention, and full input-output map, and we measure grokking as a multi-seed rate rather than a single outcome. In this fully-tractable regime grokking is a conditional, fragile phase transition. It is gated by training-set coverage, whose threshold tracks output cardinality (the modulus) more than task structure, an ordering that holds above the transition and across a ten-fold change in domain size. Weight decay reproduces the Omnigrok inverted-U at 12K parameters, a positive control on the rate measurement. Grokking also sits on a numerical knife-edge: two perturbations of the floating-point environment -- CPU thread count (reduction order) and CPU-versus-GPU execution -- each flip a minority of same-seed outcomes without a detectable shift in the aggregate rate. Decomposition into sub-task specialists helps chiefly by making coverage cheap rather than by adding supervision. Methodologically, multi-seed control under a fixed numerical environment overturns three dramatic single-run narratives in our own data, each a seed confound. The unit of evidence for grokking must therefore be a multi-seed rate under a pinned numerical environment, checked where possible against a direct reading of the model. 

---
# AIFS-SUBS: Extending Data-Driven Forecasting to Sub-Seasonal Timescales 

**Authors**: Jakob Schloer, Steffen Tietsche, Christopher D. Roberts, Lorenzo Zampieri, Simon Lang, Gert Mertes, Gareth Jones, Matthew Chantry, Frederic Vitart  

**Link**: [PDF](https://arxiv.org/pdf/2607.05100)  

**Abstract**: Data-driven models now rival numerical weather prediction in the medium range, but extending them to sub-seasonal lead times raises challenges absent at shorter horizons. Errors accumulate over long autoregressive rollouts, systematic biases grow with lead time, and several years of data must be held out for independent verification, even though machine-learning models otherwise benefit from longer training records.
To address these challenges, we adapt ECMWF's AIFS-CRPS medium-range model. AIFS-SUBS adopts a 24h autoregressive time step to reduce error accumulation, adds stratospheric levels and top-of-atmosphere thermal radiation as predictors, and reserves 2007--2011 as an independent verification window. We evaluate two config-durations: AIFS-SUBS, fine-tuned on operational analyses, and AIFS-SUBS-ERA5, trained on ERA5 alone. Across weeks 2--6, AIFS-SUBS matches the operational Integrated Forecasting System (IFS) in probabilistic skill while reducing systematic biases. For the convective (OLR) component of the Madden--Julian Oscillation (MJO), AIFS-SUBS extends skilful forecasts (correlation > 0.5) by eight days relative to the IFS, while matching or exceeding the IFS for the full multivariate RMM index. AIFS-SUBS also reproduces the observed MJO modulation of tropical cyclone activity comparably. Stratospheric skill is particularly strong with AIFS-SUBS reproducing sudden stratospheric warming (SSW) frequency and surface impact. In the AI Weather Quest, AIFS-SUBS-ERA5 attains a variable-averaged ranked probability skill score slightly ahead of the IFS at weeks 3 and 4. At inference, AIFS-SUBS uses about 200 times less energy than the IFS, opening the door to much larger real-time ensembles. AIFS-SUBS is ECMWF's first machine-learning model targeted at sub-seasonal time-scales. 

---
# Beyond Independent Labels: Schwartz-Geometry Decoding for Human Value Detection 

**Authors**: Víctor Yeste, Paolo Rosso  

**Link**: [PDF](https://arxiv.org/pdf/2607.05052)  

**Abstract**: Human value detection is commonly formulated as sentence-level multi-label classification over the 19 refined Schwartz values, typically predicted as independent labels. Schwartz theory, however, describes them as a circular motivational continuum, in which adjacent values are compatible and opposing values are in tension. We ask whether this structure can be operationalized as an explicit output-space geometry and used as a soft bias rather than a hard constraint. On a DeBERTa-v3-base classifier, we compare two ways of injecting it: training-time geometry-aware objectives and a post-hoc Schwartz-aware energy decoder that scores whole label sets jointly. Across five seeds, training-time geometry gives only limited gains-no larger for the true continuum than for a random ordering-whereas the decoder makes label sets more coherent with the continuum-on theory-aware coherence metrics we introduce-at no cost to Macro-F1 or Micro-F1 (held fixed by its selection rule). The gain is specific to the true Schwartz ordering: it does not appear for a random permutation or an empirical co-occurrence graph through the identical decoder. A bounded Qwen2.5-72B-Instruct diagnostic shows that supplying the continuum at inference shifts behavior but does not match supervised structured prediction. Theory-aware decoding thus offers a lightweight, controllable way to make value detection faithful to its label space. 

---
# RUFNet: Query-Guided Support Mask Refinement and Uncertainty Fusion based on Hybrid Mamba for Few-Shot Brain Tumor Segmentation 

**Authors**: Dongyi He, Xiangkai Wang, Binbing Xu, Bin Jiang, Hongjie Yan, Weixiang Liu, Wai Ting Siok, Nizhuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.05035)  

**Abstract**: Few-shot brain tumor segmentation remains challenging due to noisy support masks, inter-patient variations between support and query images, and the lack of pixel-wise confidence estimation. This study proposes RUFNet, a Hybrid Mamba-based few-shot framework that combines support mask refinement with uncertainty-aware posterior fusion. To preserve support-query dependencies with manageable cost, RUFNet adopts a Hybrid Mamba interaction backbone with linear complexity. To reduce support-mask noise, an Attention-Guided Mask Refinement module (AGMR) uses query features to recalibrate support masks and improve prototype consistency. To handle ambiguous predictions, an Uncertainty-Aware Posterior Fusion module (UAPF) estimates pixel-wise variance and adaptively balances few-shot predictions with query-aligned priors. On the Brain Tumor Segmentation Challenge (BraTS) 2020 dataset, RUFNet achieves Dice coefficients of 84.3% and 86.1% in the 1-way 1-shot and 1-way 5-shot settings, respectively, outperforming the compared state-of-the-art methods. These results suggest that Hybrid Mamba interaction, mask refinement and uncertainty modelling can improve the robustness of few-shot medical image segmentation. The official implementation code is available at this https URL. 

---
# LLM-Based Test Oracles: Source-of-Authority Taxonomy -- A Systematic Literature Review 

**Authors**: Ali Hassaan Mughal, Muhammad Bilal  

**Link**: [PDF](https://arxiv.org/pdf/2607.05031)  

**Abstract**: Large language models (LLMs) are increasingly used to produce test oracles, the part of a test that decides whether observed behavior is correct. Yet a clear account of where these oracles draw their authority is missing. Prior secondary studies organize the area by oracle form or by LLM technique. None organizes it by the source of the verdict's authority, the property that governs how far a verdict can be trusted. This article presents a systematic literature review, conducted and reported under the PRISMA 2020 guidelines. From 2,436 records, an LLM pre-filter followed by independent dual human screening (reviewer agreement, a Cohen's kappa of 0.79) and full-text assessment yielded 54 included studies. We analyze these along three axes: the source of an oracle's authority, the form it takes, and the mechanism that adjudicates it. We characterize the landscape of domains, languages, models, and adaptation strategies. Specification-derived authority, though the most common single source, covers about half of the studies (28 of 54). The remaining 26 reach a verdict with no specification at all. The source of authority and the adjudication mechanism cross-cut: the same source is checked by several mechanisms and one mechanism serves several sources, so a label such as LLM-as-a-judge names a mechanism rather than a basis for trust. We further report how these oracles are evaluated and how they fail, and read the sparse and empty regions of the taxonomy as a research agenda. The protocol, search query, and per-study coding sheet are released as supplementary material. 

---
# Your Agent's Memories Are Not Its Own: Forged Reasoning Attacks on LLM Agent Memory and Defenses 

**Authors**: Neeraj Karamchandani, Piyush Nagasubramaniam, Sencun Zhu, Dinghao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2607.05029)  

**Abstract**: Persistent memory has enabled large language model (LLM) agents to store factual knowledge, prior decisions, reasoning histories, tool usage information, and context. While this has improved the agent's functionality and continuity across tasks, it has also introduced a new attack surface: the agent's own reasoning history. In this paper, we introduce the Forged Amplifying Rationale Memory Attack (FARMA), which poisons an agent's remembered reasoning rather than its factual knowledge. It inserts forged reasoning traces using evasive language that bypasses keyword-based defenses, then amplifies them through self-referential reinforcement that defeats consensus-based defenses. To address FARMA, we introduce SENTINEL, a layered defense pipeline to detect forged reasoning entries. Its central component is the Reasoning Guard that structurally analyzes candidate entries for forgery using five weighted signals.
We evaluate FARMA and SENTINEL across multiple agents and different LLM models with 50 trials and show that FARMA achieves an attack success rate of up to 100% under baseline conditions and is capable of defeating defense mechanisms like keyword filter and A-MemGuard. Our evaluation also shows that SENTINEL reduces FARMA's attack success rate to as low as 0% with no false positives observed across 326 benign agent traces. Our work demonstrates the need to protect not only an agent's retrieved content but also the integrity of its reasoning history. 

---
# Hyperparameter Transfer in Graph Neural Networks 

**Authors**: Gage DeZoort, Boris Hanin  

**Link**: [PDF](https://arxiv.org/pdf/2607.05017)  

**Abstract**: The performance of deep learning models crucially depends on the settings of hyperparameters like learning rate, initialization scale, and weight decay. Hyperparameter transfer aims to make near-optimal hyperparameter settings consistent across model scale, so that large models can be optimized by proxy tuning their smaller, cheaper-to-optimize counterparts. While transfer principles are well-studied in the context of dense neural networks in language and vision tasks, they remain comparatively under-explored for graph neural networks (GNNs). We develop and validate a transfer parameterization for GNNs trained with SGD, Adam, and AdamW. Through theoretical scaling analyses and controlled experiments, we show that the proposed parameterization yields stable feature updates, learning rate transfer, and improved performance as width and depth increase. For SGD, we identify graph-dependent first-layer correction factors and show that their use can accelerate early training in graphs with sparse bag-of-words inputs. For Adam, we explore how different message passing normalizations affect early- and late-training transfer behavior, illustrating the importance of message passing normalization and advocating for an associated hyperparameter. For AdamW, we adapt a parameterization that allows for the joint transfer of weight decay and learning rate. Together, these results provide a practical recipe for scaling GNNs across a variety of learning tasks and training scenarios. 

---
# ImputeECG: Deep Learning Reconstruction of Complete 12-Lead Electrocardiograms from Incomplete Recordings for Cardiac Assessment 

**Authors**: Xiaocheng Fang, Haoyu Wang, Jieyi Cai, Qinghao Zhao, Jun Li, Shanwei Zhang, Guangkun Nie, Yujie Xiao, Shun Huang, Jiarui Jin, Hongmin Liu, Guodong Wang, Shuohua Chen, Liming Lin, Shouling Wu, Hongyan Li, Shenda Hong  

**Link**: [PDF](https://arxiv.org/pdf/2607.05009)  

**Abstract**: Complete digital 12-lead electrocardiograms (ECGs) are essential for AI-enabled cardiovascular assessment, yet many clinical ECG records, particularly those digitized from ECG images, remain incomplete because of short display formats, incomplete waveform digitization, lead loss, or signal corruption. We developed ImputeECG, a mask-conditioned one-dimensional Transformer autoencoder that completes 12-lead, 10-s ECGs while retaining all observed samples. The model was trained on PTB-XL and evaluated on PTB-XL and CPSC2018 under simulated incomplete settings, with additional real-world validation in a 43,633-record Kailuan clinical cohort after ECG image digitization. Metrics were computed over originally missing regions, with analyses of morphology and downstream diagnostic utility. On PTB-XL, ImputeECG reduced missing-region MAE by 41.7-51.0% and MSE by 54.0-63.7% versus the strongest baseline, with lower errors in R-peak timing, RR interval, QRS duration, QT interval, and P-wave, QRS-complex, and T-wave reconstruction. On CPSC2018, ImputeECG reduced MAE by 49.7-51.9%, supporting external generalization. In downstream multi-label classification, ImputeECG restored performance to 92.28% AUROC and 33.88% AUPRC in the most incomplete PTB-XL setting, approaching complete-ECG performance. On CPSC2018, completed ECGs achieved 94.75-95.89% AUROC and 78.83-81.86% AUPRC across settings. In Kailuan, ECG completion improved zero-shot sex prediction AUROC from 82.6% to 85.8% and reduced age prediction MAE from 10.72 to 9.87 years after image-based ECG digitization. These findings support ECG completion as a practical strategy for converting incomplete ECG records into AI-ready 12-lead, 10-s digital signals and extending the usable scope of ECG archives for digital cardiac assessment. 

---
# Comparison of Loss Functions for Robust Deep Learning-based Echocardiography Segmentation when Learning with Partially Labelled Data from Multiple Domains 

**Authors**: Iman Islam, Esther Puyol-Antón, Bram Ruijsink, Andrew J. Reader, Andrew P. King  

**Link**: [PDF](https://arxiv.org/pdf/2607.05008)  

**Abstract**: Echocardiography is the first imaging modality used for assessing cardiac function, and accurate segmentation of cardiac structures is essential for deriving biomarkers. However, the development of effective automated segmentation models for multiple cardiac structures is challenged by the difficulty of training on datasets from different sources that are often partially-labelled. This study aims to address this challenge by evaluating the performance of three loss functions - adaptive categorical cross entropy (aCCE) loss, marginal loss, and the adaptive binary cross entropy (aBCE) loss - in handling partially-labelled data. We conduct a comprehensive comparison of these loss functions across multiple scenarios and network architectures: intra-domain and inter-domain tasks, with both single and multiple partial-labels, and varying proportions of fully-labelled to partially-labelled data.
Our experiments reveal that all three loss functions exhibit strong performance in intra-domain segmentation tasks, effectively handling label variations within the same domain. For inter-domain tasks, where models are trained on datasets with a domain shift, the aBCE and marginal losses show superior performance when dealing with the case of one label being missing from some training examples. In scenarios involving more than one label being missing, marginal loss outperforms the other methods, demonstrating its robustness in such complex conditions. These results highlight the strengths of each loss function depending on the labelling scenario, emphasizing the importance of selecting the appropriate loss function to optimize model performance. This study represents the first investigation of techniques for handling partially-labelled data from multiple different domains in echocardiography segmentation and provides a comprehensive comparison of loss-based solutions. 

---
# TACTIC-KG: Toward Small Agent Teams for Cyber Threat Intelligence Knowledge Graph Construction 

**Authors**: Mouhamed Amine Bouchiha, Gregory Blanc  

**Link**: [PDF](https://arxiv.org/pdf/2607.05001)  

**Abstract**: Cyber Threat Intelligence (CTI) reports are predominantly unstructured, heterogeneous, and noisy, which limits their direct usability for automated analysis and reasoning. Cybersecurity Knowledge Graphs (CSKGs) provide a structured representation of adversarial entities, actions, and relations, but constructing such graphs from free-text CTI remains a challenge. Recent approaches rely on monolithic Large Language Models (LLMs) to perform end-to-end extraction and completion, leading to high cost, limited controllability, and unstable performance. This paper introduces TACTIC-KG, an agentic framework for CSKG construction that decomposes the task into modular, specialized LLM agents responsible for extraction, typing, verification, and curation. Using lightweight models (3B--8B), TACTIC-KG improves stability, recall, and graph consistency while reducing deployment cost. We implement and evaluate TACTIC-KG against recent state-of-the-art systems. Experiments on human-annotated CTI reports show that agent specialization consistently outperforms larger monolithic in-context-learning (ICL) baselines in extraction F1-score, typing accuracy, and structural graph similarity. 

---
# The Map Behind the Flow: Finite-Step Gradient Descent as a Dynamical System 

**Authors**: Thomas Hofmann  

**Link**: [PDF](https://arxiv.org/pdf/2607.04993)  

**Abstract**: Many phenomena of deep learning are dynamical: they concern not only which minima exist, but how gradient descent reaches, avoids, or selects among them. Edge-of-stability behavior, sharpness oscillations, catapult phases, balancing, and movement toward flatter representations are effects of the training map itself, and are poorly captured by the small-step gradient-flow limit.
This paper studies fixed-step gradient descent as a discrete dynamical system in a hierarchy of exactly solvable models retaining basic structures of deep learning: depth, factorization, width, data coupling, activation, and stochasticity. The starting point is the balanced scalar reduction of a deep linear chain, giving a quartic loss and a cubic gradient map whose post-edge behavior is explicit. Under the natural large-depth scaling, this dynamics converges to a universal Ricker-type map. The edge of stability is therefore not a breakdown of optimization, but the first bifurcation of the training map.
Embedding the scalar dynamics back into factored models turns these regimes into learning phenomena. Finite steps break conservation laws of gradient flow and contract factorization imbalance; residual oscillations move parameters toward flatter, more balanced representations. Wider linear networks produce a ladder of spectral edges, so the optimal learning rate can lie beyond the first edge. Data coupling, nonlinear activations, and stochastic targets preserve the same organizing principle: finite-step oscillations drive alignment, balancing, and representation selection. Thus the learning rate is not merely a numerical stability parameter. It is a structural parameter of the training dynamics, determining its attractors and shaping the representations gradient descent selects. 

---
# LLM for the development of FCM 

**Authors**: Alexis Kafantaris  

**Link**: [PDF](https://arxiv.org/pdf/2607.04983)  

**Abstract**: This article is about the development of a fuzzy cognitive map using a local large language model. In the light of recent advances it is evident that large language models, and even local large language models are capable of extracting quantities from textual data. In other words, a local LLM like Qwen2.5-32B, or probably larger, can accept entities as prompt input and determine relevant quantitative data as the model output. In turn, this output can be utilized for the construction of a data driven fuzzy cognitive map. Hence, this implementation is achieved and then the model is thoroughly tested; Qwen2.5-32B is used and the data is extracted from hotel reviews from TripAdvisor. Furthermore, the extracted documents pass through the model unfiltered and then a fuzzy cognitive map is trained and evaluated. A case is made about Greek reviews where a star topology FCM is formed that indicates the preferences of the reviewers. Finally, external validation is performed to establish whether the fuzzy cognitive map can correlate the star rating of the review -an outcome outside the model's inference scope -with its predicted satisfaction. 

---
# Joint Velocity Slope Diffusion Prior for Structurally Constrained Velocity Model Building 

**Authors**: Francesco Brandolin, Tariq Alkhalifah  

**Link**: [PDF](https://arxiv.org/pdf/2607.04982)  

**Abstract**: High-resolution velocity models are crucial for reservoir characterization and subsurface delineation. However, the band limited nature of our surface recorded data limits resolution. Utilizing well measurements to enhance the resolution of our subsurface models is an important objective. To this end, we present a diffusion-guided framework for structurally preconditioned velocity-model reconstruction from sparse well-log information. The proposed approach combines plane-wave PDE regularization, structurally preconditioned inversion, and measurement-guided diffusion posterior sampling within a unified formulation. Local structural slopes estimated through plane-wave destruction are used both to propagate well information along geological dip directions and to guide the diffusion sampling process through a joint velocity--slope generative prior. Numerical experiments on the Volve synthetic model and the Viking Graben field dataset demonstrate that the proposed framework improves structural continuity, lateral consistency, and geological realism compared with conventional structurally preconditioned inversion approaches while maintaining computationally practical inference through DDIM sampling. 

---
# Multi-Robot Open Adaptive Teaming Across Unseen Environments, Partners, and Scales 

**Authors**: Yang Li, Feng Xue, Fan Mo, Yunhao Liu, Jianhong Wang, Ying Wen, Qingrui Zhang, Shaoshuai Mou, Wei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2607.04972)  

**Abstract**: Deploying robot teams in the real world requires simultaneous adaptation to unseen environments, unknown partners, and varying team sizes, yet existing approaches often address these challenges in isolation under the closed-world assumption of fixed teammates. We formalize this as open adaptive multi-robot teaming and propose a hypergraphic-form game formulation that captures team-level cooperative relationships beyond pairwise interactions, providing a principled foundation for coordination structure inference when team composition changes dynamically within episodes. Unlike graph neural network architectures, this is a game-theoretic construct for modeling strategic interactions and payoff structures among agents. Building on this formulation, we develop the Hypergraphic Open-ended Learning Algorithm (HOLA), which progressively expands partner and environment diversity during training rather than optimizing for fixed configurations. Evaluated on cooperative pursuit with multi-drone and multi-quadruped platforms, HOLA outperforms all baselines across all three adaptability dimensions. Learned policies transfer directly to physical hardware without fine-tuning, with successful deployments on Crazyflie and Zsibot L1 platforms confirming robust real-world coordination in novel environments with unseen teammates. 

---
# MemPose: Category-level Object Pose Estimation with Memory 

**Authors**: Xiao Lin, Minghao Zhu, Yun Peng, Liuyi Wang, Qiyi Wang, Chengju Liu, Qijun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.04930)  

**Abstract**: In the pursuit of robust and generalizable category-level object pose estimation, most existing methods adopt parametric formulations that learn effective representations from data, yet they primarily encode category-level patterns into fixed shape priors or static parameter weights, which limits their scalability to highly diverse instances. In this paper, we rethink category-level pose estimation from a memory-centric perspective and present MemPose, a memory-augmented framework that explicitly incorporates category-level geometric memory into the pose estimation pipeline. We introduce an external memory buffer that stores and dynamically updates structural representations from previously observed instances, enabling the model to leverage accumulated experience to support current perception. Extensive experiments on four challenging benchmarks (REAL275, CAMERA25, Housecat6D and Wild6D) demonstrate the superiority of our proposed method over previous state-of-the-art approaches. 

---
# DSWAM: A Dual-System World Action Foundation Model for Fine-Grained Robot Manipulation 

**Authors**: Jian Zhu, Jianjun Zhang, Taiyi Su, Tianbin Liu, Zhangyuan Wang, Kai Xie, Zitai Huang, Chong Ma, Youzhang He, Tianjian Wang, Hanyang Wang, Weihao Ding, Yi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.04927)  

**Abstract**: World Action Models (WAMs) provide a promising alternative to Vision-Language-Action (VLA) policies by using video-based world modeling as dense supervision for robot action learning. Existing WAMs excel at physically grounded execution, but typically lack the explicit language-level planning interface in VLM-based VLAs for decomposing coarse instructions. Such decomposition becomes important when household tasks involve complex multi-step goals, where coarse user commands need to be converted into sequences of fine-grained executable subtasks. Meanwhile, the field still lacks a fair real-robot comparison between VLA and WAM execution capabilities, since existing systems often differ in data, robot embodiments, and task protocols. To address both the decomposition gap and the need for a controlled WAM-VLA comparison, we introduce DSWAM, a Dual-System World Action Foundation Model for fine-grained robot manipulation. DSWAM keeps a System 1 WAM executor as the default control path and optionally activates a System 2 vision-language subtask planner only when task decomposition is useful. The planner predicts executable subtasks from short-term visual history and a global task prompt, while the WAM executor performs world-aware action generation for each instruction or subtask. The executor is trained with action prediction and video co-training, but inference directly predicts action chunks without explicit future video generation. To make this execution path practical on real robots, we further integrate TensorRT acceleration, asynchronous execution, and real-time chunking (RTC) so that policy queries do not block robot control. To provide a fair real-robot comparison with VLA policies, we build and evaluate DSWAM under the DeMaVLA real-world deformable manipulation setting with matched robot platform, pretraining data, post-training data, and evaluation criteria. 

---
# Input Pathways Shape Few-Shot, Not Zero-Shot, Binding in Tiny Transformers: A Fully-Enumerable Study 

**Authors**: Yoshiyuki Ootani  

**Link**: [PDF](https://arxiv.org/pdf/2607.04926)  

**Abstract**: How does the way information reaches a transformer -- as symbolic tokens, a clean per-factor "oracle" code, or an entangled perceptual vector -- shape whether it binds that information compositionally? We study ~6-10K-parameter transformers on finite factored worlds enumerated exhaustively, so every measurement covers the whole input space (zero sampling variance) and the informative routes are information-matched (exact Bayes ceiling 1.0). We report four findings. (1) Endpoint invariance: on held-out binding queries no informative route reaches converged zero-shot composition -- each ends at or below chance despite a ceiling of 1.0, so within a bounded sweep the failure reflects inductive bias under a lookup-sufficient objective, not missing information. (2) A two-factor account of few-shot binding: sample efficiency is best explained by input-pathway parameter sharing and code readability; a dimension-matched control and a graded readability sweep isolate readability from input dimension, and the clean oracle is not the most sample-efficient readable route. (3) A double dissociation: early in training, distributed -- but not index-like -- codes pass through a transient above-chance phase (tracking code format), while few-shot efficiency tracks pathway sharing. (4) Failure anatomy: symbolic routes lose the answer at the readout; index routes mis-bind (the answer stays decodable, yet an input intervention shows the output tracks the wrong slot); entangled routes inherit their input's readability. The central claim is the two-factor account; the endpoint and anatomy results are diagnostic constraints. All code, manifests, and per-seed logs are released for exact reproduction. 

---
# Efficient Perception in Automotive Detection and Tracking Using Neuromorphic Computing 

**Authors**: Manish Kolachalam, Rani Malhotra  

**Link**: [PDF](https://arxiv.org/pdf/2607.04921)  

**Abstract**: Deep learning algorithms are notorious for their high carbon footprint and computational demands that limit their deployment on edge devices and raise concerns about their long-term sustainability. Neuromorphic computing and Spiking Neural Networks (SNNs) offer a promising alternative to traditional Von Neumann architectures, providing energy-efficient performance, massively parallel computation, and on-chip learning capabilities. Autonomous machines represent a critical application domain where these advantages are particularly valuable. We present the first comprehensive evaluation of SNNs for real-world automotive multi-object detection and tracking. Using transfer learning with the SpikeYOLO architecture, we achieve mean Average Precision of 0.937 on the KITTI dataset and 0.771 on BDD100K MOT2020 dataset for object detection and a Higher Order Tracking Accuracy score of 0.701 (KITTI) and 0.445 (BDD100K MOT2020) for object tracking--results competitive with conventional deep learning methods. Our results demonstrate that SNNs can deliver high-performance object detection and tracking in an energy efficient manner, establishing their viability for perception in real-world autonomous systems. 

---
# Graph Representation Learning of Longitudinal Medical Imaging Trajectories for Treatment Response Prediction 

**Authors**: Johannes Kiechle, Richard Osuala, Daniel M. Lang, Stefan M. Fischer, Ivana Janíčková, Karim Lekadir, Julia A. Schnabel, Jan C. Peeken  

**Link**: [PDF](https://arxiv.org/pdf/2607.04912)  

**Abstract**: In patients with breast cancer, pathological complete response (pCR) has been established as a clinically meaningful surrogate marker for long-term outcomes. While commonly treated with neoadjuvant chemotherapy (NACT), effective treatment decision-making remains challenging, as therapeutic response can vary substantially across patients, calling for predictive models capable of accurately estimating individualized treatment response. To address this, we propose an imaging-based 3D spatio-temporal framework for treatment response prediction that integrates a state-of-the-art graph neural network with relational modeling of temporal interactions across timepoints alongside three novel complementary self-supervised treatment trajectory representation learning objectives. Experiments across a cohort of 585 patients from the public ISPY-2 dataset demonstrate that our method substantially outperforms both vision and self-supervised learning baselines across several classification metrics. Alongside establishing a breast cancer pCR prediction benchmark, we include a principled ablation of our method and further introduce and empirically assess the impact of the available number of DCE-MRI timepoints per patient trajectory and the inclusion of inter-scan time-differences. Overall, our study substantiates the utility of clinically meaningful longitudinal medical imagaging modeling for predicting NACT-induced pCR. We will publicly share our code repository and a user-friendly PyPI library for dataset curation upon publication, effectively promoting reproducible open-source research. 

---
# EventCoT: Event-centric Video Chain-of-thought for Reasoning Temporal Localization 

**Authors**: Youngkil Song, Yoonjae Baek, Dongwon Kim, Inho Kim, Dongkeun Kim, Suha Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2607.04872)  

**Abstract**: Reasoning temporal localization (RTL) requires a model to generate an answer that itself contains the time interval supporting it, so high-level reasoning and precise temporal grounding must be produced jointly in a single response. To tackle this challenging task, we propose the first event-centric video chain-of-thought framework, dubbed EventCoT. EventCoT first performs event-centric tokenization of the input video to convert it into compact event tokens, enabling efficient identification of question-relevant events. It then reasons within the identified events to generate the answer, grounding the time interval via embedding matching that aligns placeholder tokens with visual embeddings. EventCoT achieves state-of-the-art results on ActivityNet-RTL for reasoning temporal localization while using substantially fewer visual tokens than previous work. To verify its general performance, we further evaluate EventCoT on the grounded video question answering benchmark ReXTime, where it attains strong zero-shot results. 

---
# SynSFX: Multi-Model Sound Effects Synthesis Dataset for Deepfake Detection and Evaluation 

**Authors**: Linxi Li, Yuncong Yu, Qianwei Guo, Liwei Jin, Yechen Wang, Carsten Maple  

**Link**: [PDF](https://arxiv.org/pdf/2607.04848)  

**Abstract**: While audio deepfake detection has advanced significantly, representative detectors show limited generalization to synthetic sound effects. Existing environmental audio datasets such as EnvSDD provide important initial resources, but remain limited in scale and generation provenance for studying isolated sound-effect deepfakes. To support this direction, we present SynSFX, a large-scale corpus of 43374 clips (26452 synthetic, 16922 real) spanning 7 popular text-to-audio models. 

---
# Pretraining Curricula Enable Selective Fine-tuning 

**Authors**: Sebastian A. Bruijns, Jirko Rubruck, Mia H. Whitefield, Kai J. Sandbrink, Fazl Barez, Christopher Summerfield  

**Link**: [PDF](https://arxiv.org/pdf/2607.04846)  

**Abstract**: Transformers follow implicit curricula whereby some tasks are learned before others. However, how explicit pretraining curricula influence learning, generalization, and the selectivity of fine-tuning is unclear. This is important for AI safety, where fine-tuning is used to selectively suppress misaligned behaviors. Here, we compare curricula that pretrain tasks in a balanced (sampled uniformly) or an imbalanced (one task early, the other late) fashion. We show that imbalanced learning of two conflicting copy tasks promotes in-context learning and improves the selectivity of refusal fine-tuning. Ablations and activation patching show that this occurs because imbalanced pretraining encourages tasks to be disentangled in separable neural circuits, whereas balanced training routes both tasks through a common pathway. We extend these findings to a synthetic language learning task involving rule-consistent and rule-violating data, where imbalanced curricula similarly lead to more localized, less entangled rule representations, resulting in more robust rule-following behavior. Together, these results suggest that imbalanced pretraining curricula may be an important tool for promoting disentangled representations, with direct consequences for the precision and reliability of safety fine-tuning. 

---
# HamQASBench: A Hamiltonian-Informed Diagnostic Benchmark for Evaluating Quantum Architecture Search 

**Authors**: Jiayang Niu, Akib Karim, Yan Wang, Jie Li, Ke Deng, Azadeh Alavi, Muhammad Usman, Yongli Ren  

**Link**: [PDF](https://arxiv.org/pdf/2607.04845)  

**Abstract**: Quantum Architecture Search (QAS) automates the design of parameterized quantum circuits for variational quantum algorithms, yet existing benchmarks organize instances by molecular identity or qubit count -- criteria agnostic to Hamiltonian structure -- and rely solely on energy accuracy, which cannot detect structural failures such as over-parameterization on near-product ground states. We introduce HamQASBench, a Hamiltonian-informed diagnostic benchmark organizing 11 molecules into five structural tiers via fingerprints derived from the Pauli operator basis, computational basis representation, and ground-state entanglement. A post-hoc critical-structure extraction procedure identifies minimal circuits consistent with each tier's requirements, complementing energy-based evaluation with per-qubit entanglement analysis and pairwise state fidelity. Benchmarking five QAS methods across four paradigms reveals failure modes invisible to conventional metrics: over-parameterization in the minimalism regime, eigenstate commitment under degeneracy, a representation bottleneck in strongly correlated systems, topology-induced routing failure, and circuit search space growth as a scalability bottleneck. 

---
# Evaluating the Effect of Linguistic Relatedness on Cross-Lingual Transfer in Large Multilingual Automatic Speech Recognition 

**Authors**: Andrei Florian, Cynthia Jayne Amol, Hope Kerubo Ombaba, Xiaoyu Cui, Boniface Mwau, Biatus Maina Kamau, Lilian Diana Awuor Wanzare, Christiane Fellbaum, Happy Buzaaba  

**Link**: [PDF](https://arxiv.org/pdf/2607.04814)  

**Abstract**: Extending automatic speech recognition (ASR) to low-resource African languages is constrained by the prohibitive demands of data collection at scale. A promising direction is to leverage linguistic relatedness to enhance cross-lingual transfer from a related auxiliary language to the low-resource target by sequentially adapting on both. Although this strategy has shown meaningful improvements in small ASR models, its effectiveness in large ASR remains unclear. We extend this framework to large multilingual ASR through a systematic controlled experimental design spanning six factors, two Africa-centric corpora, and four large ASR models, isolating whether linguistic relatedness reliably predicts cross-lingual transfer gains in this setting. Across all conditions, pre-adaptation on related auxiliary languages yields no practically meaningful transfer improvements given minimal target-language data, suggesting that linguistic relatedness alone may not reliably predict cross-lingual transfer gains in large multilingual ASR, or constitute an effective strategy for extending such models to low-resource languages. 

---
# An Exploration of Agentic Information Fusion for Test Maintenance Prediction 

**Authors**: Jingxiong Liu, Nasser Mohammadiha, Gregory Gay  

**Link**: [PDF](https://arxiv.org/pdf/2607.04786)  

**Abstract**: Test maintenance is a critical, yet costly, activity - particularly as codebases rapidly evolve. To assist, we present MAST, a multi-agent framework that predicts which test cases require maintenance following changes to the production code. This identification task is necessary as a precondition to any subsequent maintenance activities, but remains challenging due to the complex relationships between production and test code. MAST advances the state-of-the-art by integrating multiple analyses -- including static, lexical, and semantic analyses - through an intelligent fusion and post-check procedure and by focusing on a realistic use and evaluation setting - i.e., standardized input formats, repository-level analyses, and the ability to infer relations between test and production artifacts rather than assuming a pre-existing mapping.
We evaluated MAST on 21 industrial Java repositories from Ericsson AB, considering situations where test maintenance both was and was not required in the ground truth. MAST yielded superior precision to a state-of-the-art baseline - resulting in a higher accuracy, F1, and F2 score - with only some loss in recall. Our ablation study demonstrates the value of each analysis in producing the final recommendations. MAST illustrates the potential of multi-agent systems that can fuse multiple information sources when performing software testing tasks. 

---
# Predicting Drafted Deck Strength for "Magic: the Gathering" 

**Authors**: Tomas Rigaux, Hisashi Kashima  

**Link**: [PDF](https://arxiv.org/pdf/2607.04782)  

**Abstract**: Many real-world games do not admit a fixed, compact rule set: instead, their dynamics are defined by interactions among a large and often evolving collection of game pieces, making general-purpose policy learning impractical. Magic: the Gathering (MTG) exemplifies this setting, where the cards themselves define and alter gameplay rules, strategic constraints, and long-term outcomes, while the pool of available cards is ever-changing. We study Draft, a constrained deck-building format of MTG in which eight players make 39-45 sequential selections from semi-random packs to construct a 40-card deck under partial information. By isolating the card selection process from gameplay, Draft provides a tractable yet non-trivial setting for studying decision-making driven by combinatorial card synergies. We propose an encoder-based model that produces set-contextualized card embeddings to encode the draft decision sequence, with a consistent improvement over linear baselines on large-scale real-world data, establishing a first learned benchmark for outcome prediction in MTG Draft. Our code is available at this http URL. 

---
# Multi-Turn On-Policy Distillation with Prefix Replay 

**Authors**: Baohao Liao, Hanze Dong, Christof Monz, Xinxing Xu, Li Dong, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2607.04763)  

**Abstract**: We study on-policy distillation (OPD) for agentic tasks, where an LLM agent interacts with an environment over multiple turns and a student imitates a teacher over these multi-turn interaction histories. Fully online OPD is costly because each update requires fresh student rollouts through the environment and teacher queries at visited histories. We propose Replayed-Prefix On-Policy Distillation (ReOPD), an off-environment alternative that reuses pre-collected teacher trajectories as replayed prefixes: the student acts at selected steps, while the teacher provides dense per-step supervision without executing new environment interactions. We show that multi-turn OPD introduces a prefix trap: making histories more student-on-policy improves relevance to the student, but can query the teacher on histories where its target is unreliable. This creates a two-sided distribution shift between student occupancy and teacher reliability. ReOPD addresses this by treating multi-turn OPD as a reliability-aware prefix distribution design and implements it with a simple step-decaying sampling schedule that emphasizes early, lower-shift prefixes. Across mathematical reasoning with Python and search environments over multiple teacher and student model scales, ReOPD preserves or improves OPD-level accuracy, uses zero tool calls during student training, and is at least 4$\times$ faster per training step than OPD. ReOPD therefore turns expensive agent-environment interaction into a reusable offline resource, enabling scalable distillation across tools, tasks, and environments. 

---
# Trust Region Policy Distillation 

**Authors**: Zhengpeng Xie, Li Lyna Zhang, Zeke Xie, Mao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.04751)  

**Abstract**: Big goals are hard to achieve all at once; breaking them into small steps is wiser. We present Trust Region Policy Distillation (TOP-D), which transforms the notoriously unstable, high-variance On-Policy Distillation (OPD) into a stable training paradigm by dynamically constructing a proximal teacher. Theoretically, we establish a rigorous framework demonstrating that TOP-D inherently controls gradient variance. By providing a formal global convergence analysis alongside a monotonic improvement bound, we mathematically formalize the reliability and stability of the overall training dynamics. Empirically, TOP-D dramatically enhances training stability, sample efficiency, and final performance on mathematical reasoning tasks. More importantly, TOP-D introduces zero additional computational overhead, positioning itself as a promising alternative to the well-established OPD paradigm. 

---
# Wasserstein Residuals: Learning Gradient Flows from Population Dynamics 

**Authors**: Markus Heinonen, Yair Shenfeld, Ricardo Baptista, Daniel Waxman, Dmitry Batenkov, Tim Cooijmans, Eli Bingham  

**Link**: [PDF](https://arxiv.org/pdf/2607.04738)  

**Abstract**: Reconstructing population dynamics is a central problem in the physical and data sciences. Often, the dynamics are modeled as a Wasserstein gradient flow (WGF): a curve of distributions driven by an energy functional. Though there are multiple mathematical characterizations of a WGF, the dominant algorithmic approach relies on the Jordan--Kinderlehrer--Otto (JKO) scheme. JKO-based methods are inflexible to time discretisation and require solving costly optimal transport problems. We take a residual approach, enforcing the continuity equations via a non-negative loss function whose minimum is the WGF. Combined with a data-fitting divergence, this gives a single global objective. This perspective unifies several existing methods and leads to a new particle-based method, stitching, that is simulation-free and robust to large gaps between observations. We demonstrate that the stitching method achieves state-of-the-art performance across trajectory inference benchmarks. For code see this http URL. 

---
# RustMizan: A Compilable, Contamination-Aware Benchmarking Framework for Rust Vulnerabilities 

**Authors**: Tarek Elsayed, Shiping Yang, Eunsong Koh, Sanika Goyal, Vincent Huang, Paul Ngo, Nathan Young, Mohammad Omidvar Tehrani, Alvyn Kang, Arnell Kang, Zeyu Chen, Angélica Moreira, Xuan Feng, Angel X. Chang, Nick Sumner, Steven Y. Ko  

**Link**: [PDF](https://arxiv.org/pdf/2607.04729)  

**Abstract**: LLM agents are increasingly applied to vulnerability analysis, but existing benchmarks have not kept pace. They typically rely on small non-compilable snippets, focus on binary classification (vulnerable or not), and do not account for the risk that publicly-released datasets are part of model training corpora. We introduce RustMizan, a benchmarking framework for Rust vulnerability analysis that addresses these gaps. RustMizan contains compilable code variants at the crate, file, and function levels, with annotations for binary vulnerability detection, CWE classification, and function- and line-level localization. A paired mutation framework produces semantics-preserving code mutants for contamination testing and robustness probing. Across four frontier models in an agentic setup with command-line access, binary classification sits in the 56-65% range, but line localization F1 stays near 20%, and adversarial cues drop line F1 by about 27%. 

---
# Turning Off-Policy Tokens On-Policy: A Plug-in Approach for Improving LLM Alignment 

**Authors**: Yu Li, Xiuyu Li, Mingyang Yi, Jiaxing Wang, zhangliangxu, Zhaolong Xing, Zhen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.04728)  

**Abstract**: Reinforcement learning (RL) post-training for large language models (LLMs) follows a efficient paradigm of "rollout then update", which inevitably results in off-policy training data. To resolve this, Importance sampling (IS) is proposed, while the token-level ratios compound over long sequences, causing severe variance exploded. A natural idea is "transferring" these off-policy token into on-policy token, so that the importance scores for correction are unnecessary. Following this idea, we propose Selective Importance Sampling (SIS), which is inspired by rejection sampling. Concretely, SIS implements by viewing off-policy model as proposal distribution, and implement a token-level rejection test: accepted tokens are viewed as on-policy, so that receive unit importance score, while rejected tokens retain the standard IS correction. Our proposed SIS is theoretically proved reducing the gap between token-level and sequence-level off-policy gradient estimators. The SIS acts as a plug-in that only modifies the importance ratio in the policy loss, adding negligible wall-clock overhead, and can be combine with a vast vary of RL post-training algorithms. Experiments on dense and MoE LLMs across math and agent benchmarks show that SIS consistently improves all objectives, while providing substantially stronger robustness under off-policy data. 

---
# Dashboard2Code: Evaluating Multimodal Models on Reconstructing Interactive Dashboards 

**Authors**: Tianhao Niu, Ziyu Han, Qiguang Chen, Shiqi Zhou, Baocai Shan, Hengjie Fang, Qingfu Zhu, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2607.04727)  

**Abstract**: Automatic data visualization generation has advanced rapidly with multi-modal large language models, yet existing efforts largely focus on static charts and overlook the interactive dashboards commonly used for real-world data exploration. We introduce Dashboard2Code, a novel task that requires a model to proactively explore an interactive dashboard, acquire and integrate feedback from its own interactions (e.g., clicking and filtering), and generate code that reproduces the target dashboard. To support comprehensive evaluation, we present DashboardMimic, the first Plotly+Dash benchmark for Dashboard2Code, comprising 180 carefully designed and manually verified dashboard-code pairs spanning three difficulty levels and covering eight common real-world interaction patterns. We further propose an automated evaluation framework tailored to dashboards that combines code semantic analysis with dynamic interaction-based testing to assess visual and interaction consistency, showing strong agreement with human judgments. Experiments across a range of open- and closed-source multi-modal models reveal that even the strongest systems struggle on high-complexity dashboards and that a substantial performance gap remains between open-source and closed-source models on the Dashboard2Code task. 

---
# Geometry-Aware Motion Latents for Learning Robust Manipulation Policies 

**Authors**: Yunchao Zhang, Yijia Weng, Ruizhe Liu, Ming Hu, Leonidas Guibas, Yanchao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.04714)  

**Abstract**: Learning motion latents for robotic manipulation heavily relies on extracting motion patterns from visual sequences, yet effective action abstractions require understanding three-dimensional geometric transformations. Here, we introduce GeoMoLa (Geometry-Aware Motion Latents), which learns discrete motion latent codes by predicting how point clouds evolve during manipulation rather than reconstructing visual observations. This four-dimensional objective -- spatial geometry changing through time -- forces latent representations to encode actual physical motion rather than appearance patterns. GeoMoLa achieves state-of-the-art performance using only single-view RGB-D input, while existing methods require multi-view reconstruction, succeeding across diverse manipulation benchmarks. Our ablations reveal that geometric prediction is the key to driving performance, quantitatively validating that manipulation depends on spatial understanding. Furthermore, the learned codes exhibit effective motion abstraction: applying them to novel scenes produces physically consistent transformations regardless of visual context. Our real-world experiments also confirm this robustness capability, achieving robust manipulation with minimal demonstrations in cluttered environments where geometric reasoning determines success. Thus, we demonstrate that effective motion latents for robot control can better emerge from understanding motion through its three-dimensional effects rather than pixel-level patterns. 

---
# RSPO: Reward-Swap Policy Optimization for Multi-Turn LLM Agents 

**Authors**: Qiang Liu, Taian Guo, Ruizhi Qiao, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.04713)  

**Abstract**: Reinforcement learning holds significant potential for training large language models (LLMs) to handle multi-turn interactive tasks. However, in long-horizon, multi-turn tasks characterized by sparse outcome rewards, directly training with outcome rewards often results in slow convergence due to the sparsity of signals and the lack of fine-grained feedback. Furthermore, the model may fail to learn successful trajectories that are not sampled during training, thereby limiting its performance. Conversely, while employing customized dense process rewards provides richer signals and accelerates convergence, these surrogate rewards may exhibit potential misalignment with the ground-truth outcome rewards. This inconsistency can bias the training direction and ultimately degrade the model's final performance. In this work, we propose Reward-Swap Policy Optimization (RSPO), a method designed to leverage the rich information from dense process rewards to facilitate training with outcome rewards. By utilizing a reward-swap mechanism, RSPO ensures the diversity of sampled trajectories while guaranteeing consistency between the optimization objective and the true outcome rewards, thereby elevating the performance ceiling of the model. We conduct extensive experiments on two challenging agent benchmarks, WebShop and ALFWorld. By applying our method to various reinforcement learning algorithms, including GRPO, PPO, and GiGPO, we demonstrate that RSPO achieves consistent performance improvements across different baselines and benchmarks. 

---
# Strategic Buying Agents 

**Authors**: Mingyang Fu, Ming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2607.04708)  

**Abstract**: Agentic AI is shifting online shopping from search toward delegated purchasing, where autonomous buying agents monitor markets and decide when to buy on a consumer's behalf. We study the design of such strategic buying agents, which must decide when to purchase within a finite shopping window, translating price observations, the remaining time horizon, and beliefs about future price changes into a purchase policy. We formulate this problem across three information regimes: stationary, Bayesian, and robust, and treat the resulting optimal policies as a policy menu for implementation. In the stationary regime, price adjustments follow a Poisson arrival process with a known post-adjustment price distribution; the optimal policy is a dynamic purchase-threshold rule, with the threshold governed by an ordinary differential equation. In the Bayesian regime, the adjustment intensity is known, but the price-adjustment distribution is uncertain; the optimal rule remains threshold-based, now depending on posterior beliefs, and we bound the value of knowing the true distribution. In the robust regime, the agent has only price bounds and seeks worst-case protection; randomized threshold policies achieve optimal competitive-ratio and minimax-regret guarantees. We evaluate the proposed policies on Amazon price histories from Keepa (367 items, 48,933 timestamped observations) and examine their integration into language-model buying agents. The stationary and Bayesian policies perform competitively on mean normalized consumer surplus despite their stylized assumptions, while the robust policy performs best at the distribution's 10th percentile. Results suggest language models are better suited to selecting among regimes and calibration samples than to making buy-or-wait decisions directly. 

---
# URSA: Chemistry-Aware Benchmark for Utilitarian Retrosynthesis Assessment 

**Authors**: Bogdan Zagribelnyy, Ivan Ilin, Nikita Bondarev, Anton Morgunov, Arkadii Lin, Maksim Kuznetsov, Rim Shayakhmetov, Vladimir Aladinskiy, Alex Aliper, Alex Zhavoronkov  

**Link**: [PDF](https://arxiv.org/pdf/2607.04688)  

**Abstract**: Synthesis planning aiming to find pathways of reactions for a target molecule is one of the most important and challenging tasks in drug discovery. Recent progress has produced both specialized deep-learning retrosynthesis systems and general-purpose large language models, but objective comparison remains difficult due to the lack of flexible, chemically interpretable benchmarking protocols. In the current study, we are introducing the URSA (Utilitarian RetroSynthesis Assessment) evaluation framework that provides the opportunity to benchmark the synthetic routes not only from a formal perspective, such as convergence to commercially available starting materials, but also from a chemical plausibility perspective, mimicking the way expert chemists evaluate the reactions and routes. The study covers a comprehensive evaluation of both conventional end-to-end retrosynthesis solutions and LLMs for the synthesis planning task on a set of novel, diverse target molecules with undisclosed synthetic routes, which represent realistic tasks in the daily drug design routine. We find that while LLMs can support high-level strategic planning, they currently underperform specialized retrosynthesis models in reliably solving synthesis planning tasks. 

---
# ToolFailBench: Diagnosing Tool-Use Failures in LLM Agents 

**Authors**: Harsh Soni  

**Link**: [PDF](https://arxiv.org/pdf/2607.04686)  

**Abstract**: Tool calling is central to modern language model agents, but aggregate benchmark scores often hide where tool use fails. A model that never calls a needed tool and a model that calls the tool but ignores the result can look similar under final task accuracy. We introduce ToolFailBench, a diagnostic benchmark for measuring tool-use failures across 1,000 tasks in finance, medicine, law, cybersecurity, and real estate. Tool-required tasks return values the model wouldn't guess, forcing it to trust the tool while control tasks attach the same tools but should be answered directly. We label each trace with Tool-Skip, Result-Ignore, Output-Fabrication, and Unnecessary-Tool-Use, using a rule classifier and two LLM judges aggregated by majority vote. Across 19 headline models, the best reaches 86.33% Clean Tool-Use Rate, showing that faithful tool use is not saturated. More importantly, models with similar aggregate scores fail in different ways: most stay disciplined on no-tool controls, while Llama-3.1 models show an Always-Call pattern, and at the same parameter scale Llama-3.1-70B and Qwen2.5-72B differ by 89 percentage points on control-task accuracy. Tool-use evaluation should measure not only whether agents call tools, but whether they use tool outputs correctly and avoid tools when none is needed. 

---
# Do Vision-Language-Action Models Mean What They Say? On the Role of Faithfulness in Embodied Reasoning 

**Authors**: Matthew Foutter, Matteo Cercola, Lena Wild, Yunshan Wang, Michelle Li, Daniele Gammelli, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2607.04681)  

**Abstract**: Embodied Chain-of-Thought has emerged as a promising mechanism to enhance robot decision-making and interpretability in black-box Vision-Language Action (VLA) models. However, whether this verbalized Chain-of-Thought truthfully reflects the policy's underlying decision process remains poorly understood. We distinguish between functional reasoning, in which reasoning improves task performance, and faithful reasoning, in which reasoning truly reflects the policy's internal decision process. We argue that SoTA alignment strategies offer a necessary but insufficient notion of faithfulness, admitting reasoning whose intermediate steps can mask the causal links in action prediction through confounding factors (e.g., reasoning that is ungrounded in the environment and internally disconnected or inconsistent), restricting policy generalization. We study this gap through a human evaluation of a SoTA reasoning model for autonomous driving, revealing an inconsistent coupling between reasoning quality and downstream trajectory improvement. We then operationalize a behavioral surrogate for embodied faithfulness through a learned critic, Pinocchio, scoring observation grounding and stepwise coherence, and use this critic as a dense reward signal in post-training an embodied policy with reinforcement learning. Across withheld driving benchmarks, our post-trained planner improves faithfulness by 4% and 18% over SoTA alignment and trajectory error post-training baselines, respectively, while maintaining competitive downstream task performance. Finally, on a synthetic out-of-distribution test set, post-training for faithfulness improves policy responsiveness to rare counterfactual scenarios by 1.6x that of a SoTA policy, suggesting that faithful reasoning traces contribute to more robust, generalizable, and interpretable embodied intelligence. Project page: this https URL 

---
# Elastic Gang: Per-Token Membership Change for a Hard-Barriered LLM Inference Gang Co-Scheduled with OS Processes 

**Authors**: Daeyeon Son  

**Link**: [PDF](https://arxiv.org/pdf/2607.04668)  

**Abstract**: On-device LLM decoding is a hard-barriered CPU-SIMD computation that wants every core for milliseconds per token, while the rest of the OS wants those same cores continuously. A barriered gang cannot simply be dropped into a preemptive scheduler: an unannounced departure deadlocks a barrier, and an unannounced arrival silently corrupts logits. I present the elastic gang of Anima OS, a bare-metal x86-64 Rust kernel in which the inference gang is a first-class schedulable entity whose core membership may change between any two tokens. The core mechanism is an ACK-latched epoch protocol that never waits on a named core: a seqlock-style generation-tagged latch composed with RCU/epoch-style membership consent, so each token's participant set is the intersection of the cores the gang requested and the cores that acked the current epoch. An un-acked core is outside this token and joins at most one token later. Displaced general processes migrate and keep running; cores return to them the moment a generation ends.
On a real AMD Zen 5 machine (8C/16T), inference output is bit-exact under verified per-token membership change on both a 135M and a 7B model, the property that makes elasticity safe in a kernel whose safety gate reads logits. Against fair static core partitions, elastic membership Pareto-dominates: at intermediate inference duty cycles it delivers 1.75x (25%), 1.52x (50%), and 1.28x (75%) the general throughput of a static 8-core split at equal or better inference throughput, recovers all eight stranded cores when inference is idle, and converges to the split at saturation. Returning a lent core costs 0.22 us (p50); acquiring a busy, tenant-occupied core costs one scheduling quantum (~16 ms): a running tenant is never preempted mid-slice. Decode throughput saturates at gang width 8, so ceding cores past the knee is nearly free: elasticity auto-sizes the gang online. 

---
# Targeted Structure Completion for Sparse-View 3D Reconstruction in Autonomous Driving 

**Authors**: Guoqing Wang, Pin Tang, Xiangxuan Ren, Liping Hou, Chao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2607.04661)  

**Abstract**: Reconstructing 3D scene structures from sparse, low-overlap observations remains a fundamental challenge in autonomous driving. Recent state-of-the-art frameworks achieve promising results by incorporating voxel-based Gaussians, but incur substantial computational redundancy due to a uniform volumetric processing strategy. To bridge the gap between the efficiency of pixel-based Gaussian methods and the structural completeness of voxel-based Gaussian approaches, we propose FocusGS, a simple yet effective framework that shifts the paradigm from global densification to targeted structural completion. Our central insight is that structural completion should be decoupled from deterministic regions, with computation concentrated exclusively on areas exhibiting geometric ambiguity. Specifically, FocusGS addresses the localization challenge by deriving a 3D Geometric Ambiguity Manifold to accurately isolate localized areas prone to occlusion and high geometric uncertainty. To overcome the subsequent manifold completion challenge, we design a lightweight targeted structure completion module that selectively instantiates and optimizes continuous Gaussian queries strictly within this unstructured, sparse topological subspace. Extensive experiments demonstrate that FocusGS achieves a superior efficiency-quality trade-off, advancing state-of-the-art performance on driving-centric benchmarks while naturally reducing the total number of Gaussians by ~74% and decreasing rendering time by ~34%. 

---
# Machine Learning for Depression Screening and Intervention: an Original Circadian Rhythm Score-based Methodology 

**Authors**: Bin Wang, Shuo Lian, Yuanyuan Hou, Dexian Wang, Peilan He, Feng Hong, Yanwei Yu, Tianrui Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.04648)  

**Abstract**: Depression screening from large-scale behavioral data is challenged by fragmented circadian indicators, limited interpretability, and the lack of intervention-oriented analysis. Existing approaches typically analyze sleep, activity, and social behaviors in isolation, failing to capture their joint circadian structure. To address this limitation, we first propose the Circadian Rhythm Score (CRS), a composite index that compresses multi-domain daily behaviors into a unified representation of circadian rhythm. CRS is constructed to maximize discriminative power for depression screening while preserving behavioral semantics through non-negativity constraints. Empirical results demonstrate near-lossless compression, where a single CRS retains almost the full predictive capability compared with multiple raw behavioral indicators. Building upon CRS, we develop an interpretable depression screening framework based on gradient-boosted trees and SHAP analysis, revealing nonlinear and saturation-like associations between circadian rhythm and depression risk. Beyond risk prediction, we further integrate interaction modeling and counterfactual regression to estimate heterogeneous and dose-dependent behavioral effects, enabling intervention-oriented reasoning under different circadian contexts. Experiments on the China Health and Retirement Longitudinal Study (CHARLS, n=15,233), demonstrate robust screening performance (ROC-AUC=0.825) and identify actionable behavioral thresholds, including a minimum effective exercise dose of approximately 300 MET-min/week and an optimal restorative nap duration of approximately 65 minutes for sleep-deprived individuals. By bridging supervised representation learning and interpretable modeling, this work provides a scalable framework for depression screening and intervention-aware healthcare data mining. 

---
# Retroactive Chain-of-Thought (RetroCoT): Forensic Reconstruction Prompts as a Safety Diagnostic Across Model Generations 

**Authors**: Samira Hajizadeh  

**Link**: [PDF](https://arxiv.org/pdf/2607.04645)  

**Abstract**: Safety alignment in large language models is typically evaluated against direct, imperative harmful requests. We show that this alignment is highly conditioned on pragmatic register: models that refuse a direct request frequently comply when the same underlying objective is expressed through a different communicative stance. This suggests that current alignment policies are not invariant to semantic equivalence, but remain sensitive to how a request is pragmatically framed. We introduce Retroactive Chain-of-Thought (RetroCoT), a single-turn attack that reframes harmful requests as forensic reconstruction tasks. Rather than requesting harmful instructions directly, RetroCoT presupposes that the harmful outcome has already occurred and asks the model, acting as a forensic analyst, to reconstruct in reverse the causal chain that produced it. On AdvBench (n=50), RetroCoT achieves attach success rate of 58% on gpt-4o and 52% on gpt-4o-mini, compared with direct-request baselines of 0% and 4%, respectively. We further identify a pronounced generation gap: GPT-5-family models refuse RetroCoT entirely, explicitly identifying the reconstruction premise in their refusal rationales, consistent with explicit coverage of this reconstruction register. However, this robustness does not generalize across pragmatic forms. A single adversarial feedback turn presenting an existing forensic reconstruction response alongside evaluator critique raises ASR from 0% to 48% on GPT-5.4-mini and from 58% to 94% on GPT-4o; a control condition omitting the fabricated low score achieves 85% on GPT-5.4-mini, indicating that the operative element is pragmatic continuation within the established forensic frame rather than score manipulation. These results suggest that frontier-model alignment remains conditioned on pragmatic framing rather than semantic intent, and that new pragmatic registers can continue to expose a... 

---
# Hierarchical Evidence-Driven Reasoning for Long Document Understanding 

**Authors**: Junyu Xiong, Yonghui Wang, Rongjian Gu, Chenyu Liu, Bing Yin, Wengang Zhou, Houqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.04625)  

**Abstract**: Retrieval-Augmented Generation (RAG) streamlines long-document understanding by leveraging retrieval mechanisms to restrict input images to a highly curated subset. However, existing multimodal RAG pipelines primarily face two critical challenges: first, standard semantic similarity retrievers frequently fetch topically overlapping yet answer-void distractor pages that mislead downstream generation; second, rigid single-pass pipelines heavily depend on initial retrieval success, where any omission of core evidence inevitably causes cascading errors. To address these challenges, we introduce HIEVI-RAG, a hierarchical, evidence-driven multimodal RAG framework for closed-domain document understanding. HIEVI-RAG systematically factorizes complex queries into a cooperative four-stage pipeline: (1) hierarchical question decomposition to break multi-hop root queries into atomic child questions; (2) coarse visual page retrieval leveraging a multimodal retriever to fetch candidate pages based on semantic similarity; (3) fine-grained page verification via EVIAGENT, a specialized multi-page verifier trained with GRPO to execute cross-page reasoning over multi-image blocks; and (4) memory-guided iterative generation that leverages accumulated sub-question context to execute multi-round, dynamic reasoning over the prioritized sequence. Extensive evaluations across four benchmarks demonstrate the robust efficacy and synergy of our framework, which significantly outperforms existing open-source baselines and exceeds the strongest reported baseline by an average of 8.05% in accuracy. 

---
# SILO: Simulation-in-the-Loop Sim-to-Real Transfer for Multi-Stage Cable Routing 

**Authors**: Stone Tao, Jie Xu, Hesam Rabeti, Yashraj Narang, Yijie Guo, Iretiayo Akinola  

**Link**: [PDF](https://arxiv.org/pdf/2607.04616)  

**Abstract**: Linear-deformable manipulation remains challenging due to the complex deformations of objects such as cables and ropes. Prior data-driven approaches, particularly imitation learning, have shown some promise in narrowly defined settings but typically require thousands of demonstrations for specific tasks and cable types, limiting scalability and generalization. We introduce a sim-to-real reinforcement learning (RL) framework for multi-stage cable routing that leverages GPU-parallelized simulation to approximate linear deformable behaviors. Training across thousands of parallel simulations enables the learned policies to generalize across diverse cable geometries and deformation patterns. To bridge the sim-to-real gap, we propose a novel deployment strategy that combines a Simulation In the LOop (SILO) execution framework, localized RL policies, and robust cable state estimation. On real-world cable routing tasks, our approach achieves higher success rates and 2x reduction in cycle times compared to prior state-of-the-art learning methods. To our knowledge, this is the first successful sim-to-real transfer of RL policies for multi-stage cable routing. Videos and additional visualizations are available at this https URL 

---
# G2VD: Generalizable AI-Generated Video Detection via Counterfactual Intervention and Causal Disentanglement 

**Authors**: Meng Du, Hongchang Chen, Ran Li, Junjie Zhang, Qi Ouyang, Shuxin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.04607)  

**Abstract**: The rapid advancement of AI-generated videos poses increasing security risks and calls for robust detectors with strong cross-domain generalization. Although existing methods achieve promising results under in-domain evaluation, their performance often degrades substantially when tested on unseen generators. A key reason is shortcut learning, where detectors rely on domain-specific spurious cues, such as generator-dependent fingerprints and generation styles, instead of intrinsic forgery traces. To address this issue, we propose G2VD, a Generalizable AI-Generated Video Detection framework based on counterfactual intervention and causal disentanglement. First, G2VD introduces a counterfactual intervention pipeline (CFIPipeline) that generates controlled counterfactual samples via variational autoencoders (VAEs), followed by frequency-domain and pixel-domain alignment, thereby encouraging the detector to focus on generator-intrinsic cues. Building on this intervention process, we further design a causal disentanglement classifier consisting of two domain-anchored branches with distinct classification objectives, combined with an HSIC-based independence constraint to encourage the separation of task-relevant cues from domain-specific bias. Across four public datasets, G2VD shows strong average cross-domain performance and consistent gains over matched backbones. On the challenging GenVidBench cross-domain setting, it exceeds 90% accuracy and reaches an AUC close to 0.95. Notably, this performance is obtained using only 10% of the original training data. The code is available at this https URL. 

---
# Do All Visual Tokens Matter Equally? Object-Evidence Preserving Token Merging for Vision-Language Retrieval 

**Authors**: Suhyeong Park, Junha Jung, Jungwoo Park, Jaewoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2607.04605)  

**Abstract**: Multi-vector vision-language retrieval preserves fine-grained visual evidence through maximum-similarity late interaction, but dense image-side tokens make storage and scoring expensive. Existing token compression methods reduce this cost, yet they can remove or collapse object- and region-level evidence that future query tokens may need to select. We propose SaMer, an object-aware token merging framework that compresses image-side post-projector tokens into $K$ representative centroids while preserving the original late-interaction interface. SaMer uses object annotations only during training as a merge prior to discourage cross-instance mixing, requires no ground-truth bounding boxes or detectors at inference time, and adapts only the shared projection layer with frozen vision and language backbones. With $K=64$, SaMer removes more than 93% of image-side tokens and reduces ColPali storage by $16.09\times$, while improving R@1 on Flickr30K and MSCOCO. These gains arise because object-aware merging preserves query-selectable object evidence that pruning or feature-only pooling can remove or collapse. SaMer also outperforms compression baselines and shows stronger phrase-level grounding, suggesting that efficient multi-vector retrieval depends not only on reducing token count, but on preserving the evidence future query tokens need to select. 

---
# LCPNet: Latent Consistent Proximal Unfolding Network for Infrared Small Target Detection 

**Authors**: Tianfang Zhang, Fengyi Wu, Lei Li, Chang Liu, Zhenming Peng, Huaping Zhang, Xiangyang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2607.04603)  

**Abstract**: Infrared small target detection (IRSTD) aims to identify long distance small targets from complex infrared backgrounds, and is a fundamental task in remote sensing. Deep learning methods have improved IRSTD by learning discriminative image-to-mask mappings, but such feed-forward designs often underuse physical decomposition structure between targets and backgrounds. Deep unfolding methods partially address this issue by embedding model-driven iterations into neural networks, yet existing designs still operate mainly in image domain and use updates and memory mechanisms that are not fully coupled with underlying optimization process. To address these limitations, we propose Latent Consistent Proximal unfolding network (LCPNet). First, we verify that low-rank prior remains valid in latent representations and perform unfolding in this space, preserving physical constraint while avoiding repeated compression of intermediate states. Second, we derive a Latent Consistent Proximal (LCP) solver that evolves each latent variable from its previous state rather than reconstructing through an indirect residual, and stabilizes small target updates through task-adaptive normalization and gain control. Third, we introduce Shared Optimization Memory (SOM), a common historical state shared by all decomposition variables to provide coordinated guidance across unfolding stages. Extensive experiments on four public benchmarks demonstrate that LCPNet outperforms state-of-the-art methods while achieving accurate and robust detection with low false alarms and competitive efficiency. Model and code are available at this https URL. 

---
# TORINO: Token Reduction via Interpretable Concept Overlap in Vision-Language Models 

**Authors**: Riccardo Renzulli, Gabriele Spadaro, Shruthi Gowda, Alaa Eddine Mazouz, Van-Tam Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2607.04593)  

**Abstract**: Vision-Language Models (VLMs) have demonstrated impressive capabilities across different tasks, but their computational cost is dominated by the large number of visual tokens fed to the language model. Existing token reduction methods rely on attention-based scores or pairwise similarity, without an explicit semantic representation of each token. We introduce TORINO (TOken Reduction via Interpretable coNcept Overlap), a plug-and-play framework for adaptive visual token reduction in VLMs that requires no fine-tuning of the underlying model. TORINO leverages Sparse Autoencoders (SAEs) to project visual tokens into an interpretable latent space where token relationships can be analyzed through shared concept activations. Specifically, we define concept overlap as the degree of agreement between active SAE latents and use it to group tokens that share semantic content. Reduction within each group is then performed by either pruning or merging, providing a unified framework that preserves semantically important visual information while removing redundancy. Unlike fixed-budget approaches, TORINO dynamically adapts the reduction rate to input complexity, allowing different images to retain different numbers of tokens. Experiments across multiple vision-language benchmarks show that TORINO achieves favorable efficiency-accuracy trade-offs, reducing the number of visual tokens with minimal performance loss. 

---
# Simple-to-Complex Structured Demonstrations for Vision-Language-Action Learning 

**Authors**: Xinchuan Qiu, Yi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2607.04591)  

**Abstract**: Vision-Language-Action (VLA) models have demonstrated strong capabilities in robotic manipulation by integrating visual perception, language understanding, and robot action generation. Existing research has primarily focused on improving model architectures, training strategies, and dataset scale, while little attention has been paid to how demonstrations are collected and organized. We identify demonstration organization as a fundamental yet overlooked aspect of imitation learning, as it directly affects policy learning efficiency, training stability, and policy generalization. To address this gap, we propose a simple-to-complex structured demonstration collection strategy for VLA learning using a dual-arm robotic platform. Our approach systematically organizes data through three general principles: (i) decomposing complex manipulation tasks into progressively learnable sub-skills, (ii) standardizing the interaction environment to reduce unnecessary variability, and (iii) organizing demonstrations according to progressively increasing task complexity. This structured design enables VLA models to first acquire fundamental manipulation skills before learning increasingly complex task compositions, facilitating more effective learning of long-horizon manipulation tasks. We evaluate the proposed strategy on two representative robotic manipulation tasks: block grasping and sorting, and towel folding. Experimental results show consistent improvements in task success rate and training stability compared with the baseline method of directly collecting end-to-end complete task trajectories. These findings highlight demonstration organization as a previously underexplored but important factor in VLA learning and provide practical insights into efficient skill acquisition, scalable dataset construction, and long-horizon robotic manipulation. 

---
# LLM-Driven CI-CD Workflow Intelligence for Cyber Systems Engineering 

**Authors**: Bonan Shen, Jiazhou Gao, Tao Ning, Wei-Jung Huang, Xin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.04579)  

**Abstract**: CI/CD workflows have become executable operational policy: they decide what gets built, tested, released, and deployed, and they mediate how maintainers interact with delivery infrastructure. That makes them an important measurement point for cyber-systems engineering. Recent large language model (LLM) work shows that workflow stages can be recognized directly from configuration files, but stage labels alone do not tell us whether a workflow is brittle, unusual for its ecosystem, or worth revising first. We present an LLM-based CI/CD analysis pipeline that combines repository enrichment, anti-pattern detection, stage mining, and recommendation generation over a large GitHub corpus. Starting from 59,550 repositories with at least 1,000 stars, we identify 34,225 projects with CI/CD and collect 127,559 configuration files. Across 75,201 analyzed workflows, the anti-pattern detector reports 434,769 findings, dominated by reliability and maintainability issues. Across 59,906 configurations, stage usage differs significantly by language ($\chi^2 = 4168.88$, $p < 0.001$, Cramer's $V = 0.063$), and domain analysis shows distinct operational profiles, including higher release and cache usage in mobile projects. For repository-level recommendation generation, few-shot prompting performs best overall, averaging 8.25 recommendations per repository with 96.1% YAML-valid snippets. Taken together, the results argue for CI/CD observability that combines diagnosis, context, and human review rather than treating workflow mining as a stage-classification problem alone. 

---
# A Few Teacher Steps Go a Long Way: Cost-Efficient On-Policy Data Augmentation for Agent Post-Training 

**Authors**: Junze Ye, Jiayi Cheng, Miao Lu, Michal Mankowski, Jose Blanchet, Mohsen Bayati  

**Link**: [PDF](https://arxiv.org/pdf/2607.04574)  

**Abstract**: For LLM agents, supervised fine-tuning is not only about teacher labels' quality, but also about which interaction contexts those labels condition on. Pure behavioral cloning uses full teacher demonstrations, creating a mismatch between teacher-induced contexts seen in training and student-induced contexts encountered at test time. Recent work addresses this mismatch by querying a teacher at contexts reached by the student, often with increasingly elaborate filtering of the teacher's continuations. We instead frame on-policy data construction as a budget-allocation problem: under matched supervision resources, should teacher output be spent on more start-to-finish demos, longer continuations, outcome filtering, or broader coverage of learner-induced contexts? We formalize this design space through the rollout policy, switch-time distribution, continuation horizon, filtering rules, and two complementary costs: teacher inference generated before filtering and teacher supervision retained for SFT. Across HotpotQA, ALFWorld, and Terminal-Bench-Dev, bounded unfiltered teacher continuations at learner-induced contexts improve over pure behavioral cloning at matched budgets. On HotpotQA and ALFWorld, where we run the full comparison, few-step continuations match or exceed success-filtered and critical-context-filtered alternatives. Our findings suggest that a few teacher steps, placed at learner-induced contexts, can be a more cost-efficient supervision allocation than longer or more heavily curated teacher completions. 

---
# EEG-SpikeAgent: Agentic Closed-Loop Program Synthesis for Automated EEG Spike Detection 

**Authors**: Sonali Santhosh, Kelly Shuhong Yu, Eugene Chang, Jonathan Kim, Kie Shidara, Danilo Bernardo  

**Link**: [PDF](https://arxiv.org/pdf/2607.04558)  

**Abstract**: Automated detection of interictal epileptiform discharges in scalp electroencephalography (EEG) is clinically important, but recent high-performing deep-learning models often trade interpretability for accuracy. We introduce EEG-SpikeAgent, a closed-loop program-synthesis framework that uses a large language model (LLM) agentic system to generate signal-processing features for spike detection in scalp EEG. The system iteratively proposes one deterministic EEG feature module at a time, executes the resulting code on EEG to generate tabular features, evaluates performance via a tabular classifier, summarizes run-level metrics, and feeds structured diagnostics back to the model for refinement. Across iterations, EEG-SpikeAgent proposes and refines candidate signal features and decision rules informed by model performance. We evaluated EEG-SpikeAgent on VEPISET, a public 29-channel dataset of 4-second epochs containing 2,516 discharge-containing and 22,933 non-discharge epochs. Across five-fold cross-validation with a gradient-boosted tree classifier, agent-generated features achieved an area under the receiver operating characteristic curve of 0.935, balanced accuracy of 0.699, F1 score of 0.557, sensitivity of 0.401, and specificity of 0.996 at the default operating point. At an operating point with sensitivity 0.80, mean precision was 0.470 and mean specificity was 0.900. Artifact-aware feature generation improved balanced accuracy and F1 score over spike-only feature search. These results indicate that LLM-based program synthesis can automate EEG feature engineering in auditable and inspectable code-driven manner for clinical and methodological review. 

---
# Predicting Therapeutic Outcome via Aligning Patient-Specific Knowledge Graph and Gene-Level Perturbation Representations 

**Authors**: Dongmin Bang, Sugyun An, Inyoung Sung, Ilho Yun, Sun Kim, Sangseon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2607.04557)  

**Abstract**: Accurate prediction of patient-specific therapeutic response from pre-treatment transcriptomes is hindered by the scarcity of matched clinical response labels and post-treatment molecular profiles. Preclinical transfer-learning models can simulate drug-induced expression changes but are often hard to interpret and unstable, whereas knowledge-graph methods provide mechanistic context yet remain static and fail to capture drug-induced transcriptomic perturbation dynamics. We propose PREDIKTOR, a patient-centered multi-view framework that aligns a personalized network view with a transferable transcriptomic perturbation view to predict clinical drug response. For each patient, we construct an individualized gene regulatory network from tumor expression using DysRegNet and augment it with drug-target links from DrugBank; a graph neural encoder yields a drug-centric, mechanistically grounded embedding. In parallel, a frozen condition-specific gene-gene attention model pretrained on LINCS L1000 generates a simulated post-perturbation transcriptomic profile for the same patient-drug pair. We align the two views in a shared latent space via a CLIP-style contrastive objective with drug-context hard negatives, then concatenate the representations for end-to-end response classification. On TCGA, PREDIKTOR consistently outperforms state-of-the-art baselines under patient-, drug-, and tissue-split evaluations, and transfers zero-shot to the I-SPY2 trial, improving AUROC by 5.6% over competing methods. The aligned embeddings yield stable gene and pathway attributions that recover known mechanisms, supporting actionable and interpretable precision oncology. 

---
# Lights, Camera, Carbon: Architectural Scaling Laws for Video Generation Energy Consumption 

**Authors**: Nidhal Jegham, Boris Gamazaychikov, Sasha Luccioni  

**Link**: [PDF](https://arxiv.org/pdf/2607.04553)  

**Abstract**: We present a bidirectional framework for estimating the energy consumption of text-to-video (T2V) and text-to-video-audio (T2VA) models from architectural first principles and observable generation parameters such as resolution and duration, requiring no access to weights, model size, or implementation details. Forward, it predicts energy from generation parameters and architectural principles; backward, it recovers architectural scaling behavior from observed inference times, with accuracy serving as a criterion for architectural validity. Building on the established compute-bound nature of video diffusion models, we demonstrate that each model's energy profile obeys theoretically derived scaling laws, decomposing into quadratic and linear terms whose coefficients directly reflect the underlying architectural complexity. Validated across six open-source models spanning 8.3B-27B parameters and three GPU configurations, this decomposition achieves below 3% MAPE across all architectures. This approach offers a standardized, empirically and theoretically grounded framework for sustainability benchmarking across T2V models and architectures. 

---
# Explainable Novel Category Discovery in Semantic Concept Space 

**Authors**: Ifrat Ikhtear Uddin, Yang Zhou, KC Santosh, Longwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.04548)  

**Abstract**: Novel category discovery aims to identify unseen classes from unlabeled data by transferring knowledge from labeled categories, but most existing methods perform discovery in opaque latent feature spaces. As a result, they may separate novel categories accurately while providing little insight into what semantic evidence defines each discovered group. We propose xNCD, an explainable novel category discovery framework that performs both representation-based discovery and pseudo-label assignment directly in a structured semantic concept space. Instead of clustering arbitrary deep features, xNCD learns a label-free concept representation by aligning visual features with vision-language similarity priors from pretrained multimodal models, and then applies a unified labeled-and-unlabeled self-labeling objective over concept-space logits. This design makes each discovered category explainable by construction through stable concept signatures and instance-level concept evidence. Theoretically, we show that routing discovery through a semantic concept bottleneck induces a strict restriction of the feature-space hypothesis class, excluding a large family of unconstrained decision rules and biasing induced partitions toward semantically interpretable concept coordinates. Experiments on CIFAR-10, CIFAR-100, and CUB-200 demonstrate that xNCD preserves strong discovery performance while providing intrinsic explanations. Under task-agnostic evaluation, xNCD achieves 92.63% overall accuracy on CIFAR-10, close to UNO's 93.4%, and improves CIFAR-100 overall accuracy from 73.2% to 76.45%, while being the only compared method that provides human-readable cluster- and instance-level explanations. 

---
# Mask2Real-WM: Segmentation Masks as a Sim-to-Real Bridge for Controllable Dexterous World Models 

**Authors**: Riccardo O. Feingold, Davide Liconti, Chenyu Yang, Robert K. Katzschmann  

**Link**: [PDF](https://arxiv.org/pdf/2607.04546)  

**Abstract**: Action-conditioned world models allow robots to predict the future consequences of candidate actions without additional physical interaction, supporting policy evaluation, planning, and data augmentation. We present Mask2Real-WM, a two-stage action-conditioned world model for dexterous manipulation that decouples pixel prediction into a dynamics model and a rendering model. The dynamics model predicts future segmentation masks from past masks and 23-DoF action sequences. The rendering model maps the predicted masks to photorealistic RGB using a ControlNet-augmented Stable Video Diffusion backbone. The smaller sim-to-real gap in segmentation space enables the dynamics model to benefit from large-scale pretraining on over 50 h of synthetic simulation data, followed by fine-tuning on fewer than 2.5 h of real demonstrations. Experiments on a dexterous pick-and-place benchmark show that mask conditioning and simulation pretraining are both required for per-DoF action controllability across all 23 degrees of freedom. In contrast, monolithic baselines capture broad hand and end-effector trajectories but do not reliably reflect fine-grained, per-joint action effects. 

---
# Auto: The AGI Compiler 

**Authors**: Jaber Jaber, Osama Jaber  

**Link**: [PDF](https://arxiv.org/pdf/2607.04542)  

**Abstract**: Every LLM agent run re-derives its behavior token by token on a frontier model: brilliant, expensive, slow, and unbounded. We present Auto, a compiler that records live agent behavior, measures which parts are secretly deterministic, extracts them into verified programs or distilled specialists, and emits cognition binaries: WebAssembly artifacts whose manifests carry measured guarantees and whose declared capabilities are physically enforced by the sandbox. A tiered runtime executes compiled behavior behind conformally calibrated guards; guard trips deopt to the reference agent, and the captured trace recompiles back down, so nothing is figured out twice. We use "AGI compiler" in one narrow, testable sense: a system that autonomously converts novel experience into permanent, verified, near-free skill while measuring what it does not know. On AUTO-BENCH, a benchmark we introduce and pre-register, 87.1% of 560 recorded frontier-agent spans are witnessed-deterministic (three of the four censused task families measure 100.0%). On a 300-item stream with three scheduled distribution shifts, the closed loop compiles three artifact generations and drives marginal cost from 59 to 2 micro-dollars per item (6.4x end-to-end) at 96.9% parity on witnessed inputs with zero errors. The same stream also quantifies the failure modes: a loose guard silently mislabels 48.9% of compiled answers, and an unfaithful deopt reference causes the verification gate to refuse recompilation. Calibration and reference fidelity, not model capability, decide whether cheap stays correct. Code: this https URL 

---
# CRISP: A Spatiotemporal Camera-Radar Backbone for Driving via Forecasting-Based World-Model Pretraining 

**Authors**: Jingyu Song, Yi Liu, Katherine A. Skinner  

**Link**: [PDF](https://arxiv.org/pdf/2607.04541)  

**Abstract**: Camera-radar (CR) fusion is a practical sensing configuration for autonomous driving, but existing models are typically trained with task-specific supervision, limiting reusable representation learning. We present CRISP, a spatiotemporal CR backbone pretrained through forecasting-based representation learning. Given historical multi-view images and radar sweeps, CRISP learns a unified bird's-eye-view (BEV) representation by predicting future LiDAR point clouds. LiDAR is used only as privileged supervision during pretraining; the deployed model requires only camera and radar. To make forecasting-based pretraining effective for CR fusion, CRISP introduces an enhanced radar encoder, radar-enhanced temporal self-attention, and multimodal feature rendering with modality innovation gating. These components inject radar range and Doppler cues into BEV temporal propagation and allow BEV tokens to selectively incorporate camera and radar evidence. Experiments on nuScenes show that CRISP improves long-horizon point cloud forecasting and transfers effectively to downstream tasks, including 3D detection, tracking, online mapping, motion forecasting, future occupancy prediction, and planning, suggesting that predictive CR pretraining is a promising path toward scalable driving representations under practical sensor configurations. The project website is this https URL. 

---
# Obey, Diverge, Collapse: Blind Obedience to Incorrect Instructions Drives Code LLMs to Irrecoverable Code Semantic Collapse 

**Authors**: Raj Jaiswal, Anany Singh Divy, Savar Bhasin, Adi Bajpai, Tanuja Ganu, Rajiv Ratn Shah  

**Link**: [PDF](https://arxiv.org/pdf/2607.04537)  

**Abstract**: Code language models are now trusted collaborators in production workflows for debugging, refactoring, and iterative repair, and every benchmark that evaluates them assumes the instructions they act on are correct. We study what happens when that assumption breaks. We evaluate code language models across four experiments designed to assess whether models resist or obey incorrect instructions in single-pass and iterative repair settings, using the RunBugRun dataset of algorithmic Python problems with deterministic test cases. Our findings reveal a striking behavioral pattern: models correctly identify an incorrect instruction as wrong, then follow it anyway. This compliance unknowingly introduces errors beyond the original bug, and the corrupted code state cannot be recovered through subsequent self-guided iterative repair, which fails to converge across passes. We term this Blind Obedience, characterize the Ghost (Unknown) Errors it introduces, quantify the proportion of cases where semantic corruption proves irrecoverable, and show that extended reasoning cannot reverse it. These findings surface behavioral properties invisible to pass-rate evaluation, with direct consequences for code language models deployed in production settings. 

---
# Lyapunov-Guided Training for Hardware-Safe Neural Networks Under Fixed-Point Arithmetic 

**Authors**: Anis Hamadouche, Amir Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2607.04531)  

**Abstract**: Low-precision neural networks are attractive for resource-constrained hardware, but fixed-point arithmetic introduces failure modes that are often hidden by idealised quantisation models. In particular, two's-complement overflow wrapping can corrupt hidden activations by changing both their magnitude and sign, leading to unstable numerical error propagation and severe accuracy degradation. This paper proposes a Lyapunov-stabilised quantisation framework for low-precision neural networks operating under hardware-style wrapping arithmetic. The hidden-state energy is monitored through a layerwise Lyapunov function, and a monotone projection is applied to enforce bounded and non-increasing state evolution across depth. The method is evaluated on MNIST using a compact patch-based transformer under post-training quantisation and quantisation-aware training with fixed-point bit-widths from 4 to 16 bits. Monte Carlo results show that unconstrained wrapped quantisation-aware training collapses to near-chance accuracy across 6-16 bits, with activation overflow rates exceeding 11%. In contrast, the proposed monotone Lyapunov projection suppresses activation overflow to below 0.012% and restores stable low-precision learning, achieving 86.55% accuracy at 12 bits. These results demonstrate that Lyapunov-based state control can act as a hardware-aware stabilisation mechanism for reliable fixed-point neural inference and training. 

---
# Training-Free Model Selection and Domain-Aware Score Calibration for First-Shot Anomalous Sound Detection 

**Authors**: Grach Mkrtchian  

**Link**: [PDF](https://arxiv.org/pdf/2607.04526)  

**Abstract**: First-shot anomalous sound detection in DCASE Challenge Task 2 must flag anomalies of unseen machine types with a single threshold, without knowing whether a test clip comes from the data-rich source domain (990 normal training clips) or the data-scarce target domain (10). Two organizer-reported problems remain open: source- and target-domain AUC are negatively correlated across systems, and development-set performance does not predict evaluation-set performance. We address both with a training-free post-hoc layer over frozen audio embeddings: (i) per-domain quantile calibration shrunk toward a pooled map by a prior strength m, tracing a source/target balance frontier, and (ii) a label-free cross-validated domain-balance criterion that ranks candidate configurations from training normals only, paired with a coarse development-labeled viability veto. On DCASE 2025, the criterion rank-predicts the official evaluation score across a 45-configuration grid (Spearman rho = +0.91; family-block bootstrap 95% CI [+0.83, +0.95]) while development score is uninformative (+0.06). Criterion-based selection raises the evaluation score from 55.83 to 59.34 (jackknife CI [2.2, 4.8]) and, on an extended grid, to 61.05 -- retrospectively fourth of 35 teams. Replicating on DCASE 2023 and 2024 bounds the claim: development score is uninformative in all three years and degenerate configurations recur (vetoed every time), but under family-clustered uncertainty the criterion's predictive evidence survives only in 2025; in both replication years a fixed full-equalization default matches or beats criterion-based selection. A DCASE 2026 forward test is frozen before the 2026 evaluation ground truth is released; all headline numbers are reproduced by the official evaluator. 

---
# Language Models Represent and Transform Concepts with Shared Geometry 

**Authors**: Zhimin Hu, Lanhao Niu, Sashank Varma  

**Link**: [PDF](https://arxiv.org/pdf/2607.04525)  

**Abstract**: How concepts are represented in neural networks is a fundamental question in machine learning. The dominant view treats concept representations as stationary geometric objects. Yet concepts appear in context, and context transforms them. Drawing from neural population geometry, we formalize concept representations as point-cloud manifolds and contextual transformations as vector fields, and instantiate this framework in large language models. Across six model families of varying scales, we find that context moves each concept differently. The variance in these displacements is semantically organized, correlating with lexical concreteness and density. Importantly, both the concepts being transformed and this variance structure are shared across models: displacement structure transported from one model predicts held-out displacements in others significantly above chance. Together, these findings show that models share a common geometry not only in how concepts are represented, but more importantly in how context transforms them, a structure with richer organization than prior work has recognized. 

---
# Failures and Successes to Learn a Core Conceptual Distinction from the Statistics of Language 

**Authors**: Zhimin Hu, Jeroen van Paridon, Gary Lupyan  

**Link**: [PDF](https://arxiv.org/pdf/2607.04523)  

**Abstract**: Generic statements like "tigers are striped" and "cars have radios" communicate information that is, in general, true. However, while the first statement is true in principle, the second is true only statistically. People are exquisitely sensitive to this principled-vs-statistical distinction. It has been argued that this ability to distinguish between something being true by virtue of it being a category member versus being true because of mere statistical regularity, is a general property of people's conceptual machinery and cannot itself be learned. We investigate whether the distinction between principled and statistical properties can be learned from language itself. If so, it raises the possibility that language experience can bootstrap core conceptual distinctions and that it is possible to learn sophisticated causal models directly from language. We find that language models are all sensitive to statistical prevalence, but struggle with representing the principled-vs-statistical distinction controlling for prevalence. Until GPT-4, which succeeds. 

---
# Transplanting, inverting, and preventing a misalignment persona: method-conditional emergent misalignment in Qwen2.5 

**Authors**: Lyndon Drake, Zandi Eberstadt  

**Link**: [PDF](https://arxiv.org/pdf/2607.04510)  

**Abstract**: Emergent misalignment (EM) -- the broad misbehaviour a language model acquires after fine-tuning on narrow harmful data -- is mediated in Qwen2.5 models by a latent persona direction, and that direction is causal in open weights. Transplanting it into a model that shares only pretraining with its source induces broad EM (2.83 +/- 0.26% misaligned against a random-direction floor of ~1.1%), and ablating a model's own direction roughly halves an overt inducer's broadcast (21% to 10%). The transplant doubles as a measurement method, causally assaying directions that a source model represents but cannot itself express. Whether a fine-tune recruits this persona depends on method and capacity, and since low-rank PEFT is the cheaper regime at scale, the recruiting method is also the economical one. On Qwen2.5-32B, low-rank LoRA on insecure code recruits it (3.4% misaligned) while full SFT on identical data does not (0.3%) and moves against the persona axis (drift-persona cosine +0.17 at rank 1 to -0.10), the far-inducer, high-capacity exception consistent with a representational-distance x capacity account. The persona's causal role is itself conditional. Steering a bad-medical SFT run away from the direction during training raises the broadcast from 24% to 51% while a matched random control lowers it, so removing the direction is no blanket recipe. Because recruitment is a loss-reducing shortcut that capacity renders redundant, it can be screened for and prevented in the tested instances. Persona loss-relevance at the SFT solution orders four inducers' broadcasts rank-perfectly within Qwen2.5, inoculation removes recruitment selectively (4.75% to 0.0%, code coherence 65% to 87%), and fine-tuning orthogonal to the single behaviour-derived axis reduces it persona-specifically. Results are a controlled case study of one model family, single-seed in places. 

---
# Two Black Boxes, One Solver: Encoder Probing and Decoder Attribution for Neural Multi-Attribute VRP under Hard-Mask and Recourse Decoders 

**Authors**: Sohaib Afifi  

**Link**: [PDF](https://arxiv.org/pdf/2607.04487)  

**Abstract**: Neural autoregressive solvers for the Multi-Attribute Vehicle Routing Problem (MAVRP) reach competitive cost but offer no per-step justification, a problem when dispatchers must validate, accept, or compare them. We open two complementary black boxes in one protocol. On the encoder side, linear probes, spontaneous-organization metrics, rank-based richness measures, and discovered-direction analyses with intervention validation characterize how the latent represents constraint families at the graph, node, and edge level. On the decoder side, three attribution methods (gradient, integrated gradients, DeepLIFT) feed three reading angles: abductive, contrastive against the best feasible alternative, and counterfactual (smallest input change that switches the action or restores feasibility). Explanations are scored on fidelity, concentration, stability, sanity, and actionability. Across six variants combining three encoders (Attention baseline, Unimp, UnimpMoe) with two decoders (Hard-Mask, Recourse), we find that graph inductive bias improves both representational predictability and decoder sanity, that the Mixture-of-Experts encoder represents constraints in a distributed rather than axis-aligned way, and that the Recourse training regime, not merely its softer mask, produces policies that represent infeasibility usefully, exposing make-feasible counterfactuals that Hard-Mask policies fail to produce even when fed infeasible alternatives externally. 

---
# PulmoSight-XAI: An Explainable Multi-View Attention Ensemble with Gradient Boosting Meta-Learning for Multi-Label Chest X-Ray Classification 

**Authors**: Moshiur Rahman, Shafqat Alam, Tasnia Binte Mamun  

**Link**: [PDF](https://arxiv.org/pdf/2607.04478)  

**Abstract**: Automated chest X-ray classification remains challenging due to severe class imbalance, co-occurring pathologies, and the loss of localized features in conventional architectures. To address these, we propose an explainable hierarchical multi-view ensemble framework for the robust classification of 14 thoracic pathologies. The framework employs view-specific training by independently modeling frontal and lateral radiographs using an ensemble of five complementary convolutional neural networks. Replacing global average pooling, a multi-scale feature fusion strategy augmented with Convolutional Block Attention Modules (CBAM) preserves fine-grained intermediate representations while emphasizing high-level pathology-specific semantic features. To mitigate positive-negative imbalance and varying inter-class difficulty, models are optimized using a novel hybrid objective combining Asymmetric Loss with Adaptive Focal Loss. Beyond simple probability averaging, the framework incorporates a hierarchical meta-learning strategy where test-time augmentation (TTA) predictions and cross-model uncertainty measures are integrated into Level-1 gradient-boosting meta-learners (XGBoost, LightGBM, and CatBoost), followed by Level-2 stacking with optimized alpha blending. Evaluated on a large-scale CheXpert-style dataset, the framework achieves state-of-the-art macro-average AUROC scores of 0.9319 for frontal and 0.9154 for lateral radiographs. Furthermore, comprehensive explainability analysis using seven post-hoc attribution techniques demonstrates strong anatomical consistency and clinically meaningful decision localization. By integrating architectural diversity, multi-scale attention, hierarchical meta-learning, and rigorous explainability, the proposed framework provides a transparent, highly accurate, and clinically practical computer-aided diagnosis system for thoracic disease classification. 

---
# Regime-Conditional Stabilisation of LLM-Augmented Cooperative Multi-Agent Reinforcement Learning 

**Authors**: Faid Keddouri, Sohaib Houhou, Aissa Boulmerka, Nadir Farhi  

**Link**: [PDF](https://arxiv.org/pdf/2607.04470)  

**Abstract**: Large Language Models (LLMs) offer a natural interface for translating human objectives into reward signals for cooperative multi-agent reinforcement learning (MARL), yet the training-time dynamics of this integration remain poorly understood. We show that dynamically updating LLM-generated reward weights during off-policy MARL violates the stationarity assumption of Potential-Based Reward Shaping (PBRS) and contaminates the experience replay buffer, whose stored transitions carry reward labels computed under stale shaping weights. We characterise the result as a regime-dependent failure whose severity depends on how competent the unshaped baseline already is. To control it we propose two stabilisation strategies: a Phase-Based Freeze Schedule that enforces strict stationarity within training phases, and Exponential Moving Average (EMA) smoothing that bounds per-episode weight drift. We evaluate across three cooperative environments and five random seeds with QMIX, complemented by an exploratory VDN extension, yielding a three-regime taxonomy. In the augmentative regime (Simple Spread), where the baseline is functional (74.4 %), EMA significantly improves success to 86.7 % ($+12.3$ pp, $p<0.01$) while naive dynamic updates collapse it to 15.2 %. In the essential regime (Level-Based Foraging), where the baseline is broken (0.1 %), any shaping unlocks the task (95.9 % under EMA). In the supplementary regime (SMAC 3m), where the baseline is near-saturated (98.8 %), stabilised shaping preserves performance (99.9 %) while unstabilised shaping adds variance without gain. These findings establish reward-signal stationarity as a necessary design constraint and indicate that regime placement is a practical predictor of whether dynamic LLM shaping helps or harms. 

---
# Operator-on-F complements value-equivalence: a planning-time diagnostic for latent world models 

**Authors**: Donna Vakalis  

**Link**: [PDF](https://arxiv.org/pdf/2607.04464)  

**Abstract**: World-model evaluation for model-based reinforcement learning typically asks whether the learned model predicts reward and value well, which can leave planning-relevant errors in the model's latent rollouts unmeasured. We introduce a complementary diagnostic, operator-on-F, that compares a model's k-step latent pushforward to the environment's on an observable subset F, using the model's own predictor. On a TD-MPC2 size sweep over cheetah-run, reward-prediction error stays within [0.028, 0.091] for every model size - only about 3x variation - so an unnormalized reward-fit check has narrow resolution to distinguish them; the (unnormalized) Bellman residual and reward error themselves have weak relationships with return (Spearman -0.10 and -0.30). Operator error spans 0.28 to 2.62 over the same sizes. At 317M the operator error is 2.62 - an order of magnitude above the 0.28-0.36 cluster - and the planning return collapses to 0.9, while reward-prediction error (0.091) is the highest of the five but stays within the same small [0.028, 0.091] range as the rest of the sweep. The rank correlation between operator error and return loss is -0.90 (anchor-bootstrap 95% CI [-0.90, -0.70] at n=5 sizes; leave-one-out removal of any single size leaves it at -0.80 or stronger). The operator also returns informative, architecture-discriminating estimates in a cross-architecture comparison between TD-MPC2 and a pure-SSL latent world model. The operator diagnostic complements value-equivalence rather than replacing it. 

---
# Robustness Verification of an Autonomous Underwater Vehicle-based Plankton Classifier 

**Authors**: Abdelrahman Sayed Sayed, Pierre-Jean Meyer, Asgeir J. Sørensen, Mohamed Ghazel  

**Link**: [PDF](https://arxiv.org/pdf/2607.04453)  

**Abstract**: The assessment of planktonic standing stocks and microorganism structures is critical for understanding upper ocean biological processes. Currently, autonomous underwater vehicles (AUVs) equipped with in-situ optical imaging and artificial intelligence (AI) methods offer a promising solution for persistent surveillance, mapping and monitoring of planktonic life. However, current AI methods often lack robustness in dynamic, unstructured environments, where environmental noise and non-biological artifacts lead to frequent misclassifications. Standard convolutional neural network (CNN) classifiers often struggle with such conditions, leading to misclassifications that require time-consuming manual validation by marine biologists.
To address this issue, we propose a novel robustness verification framework for in-situ plankton classifiers based on reachability analysis. We also introduce a continuous-time neural ordinary differential equation (neural ODE) classification model leveraging the high-resolution imaging capabilities of the SilCam particle imager. In this paper, we demonstrate the effectiveness of the proposed framework by formally verifying the robustness of the neural ODE model against environmental perturbations. We demonstrate that our verification framework acts as an automated filter providing formal guarantees of model stability against ambiguous data, thereby improving the reliability of autonomous sampling and reducing the post-processing workload. 

---
# A Deep Learning-based surrogate model for Severe Accidents in nuclear reactors using ASTEC 

**Authors**: Alessandro Longhi, Danny Lathouwers, Zoltán Perkó  

**Link**: [PDF](https://arxiv.org/pdf/2607.04450)  

**Abstract**: Integral codes like the Accident Source Term Evaluation Code (ASTEC) are powerful tools to study the physics of Severe Accidents (SAs) in nuclear reactors. Real time SA simulators can also be helpful in training operators of nuclear plants to react correctly to malfunctions. However, SA simulators can take up to several days per simulation, making their use infeasible for real time applications. In this work we show how to speed up a SA simulator with a fast, Deep Learning based (DL), surrogate model (SM). The SM is built as a combination of a dimensionality reduction stage, via an AutoEncoder, and a time-stepping stage, via a Neural Ordinary Differential Equation. The data on which the SM is trained are obtained from the ASTEC simulator, by sampling a set of operator actions for station blackout (SBO) and loss-of-coolant accidents (LOCA). The objective of the developed SM is to approximate multiple spatio-temporal fields for the thermal-hydraulic physics, core degradation, and fission product release modules in ASTEC's vessel domain. The SM predicts simultaneously around $80$ different physical variables (both scalar and fields), maintaining a stable autoregressive rollout up to $50$ thousand time steps. In addition, the AutoEncoder achieves a dimensionality reduction by a factor of over $300$, which allows the SM to predict up to $40$ hours of simulation in under a minute, both on CPU and GPU. This work is the first study of the capabilities and limits of DL based surrogate modeling in approximating the challenging, highly non-linear physics of ASTEC. 

---
# From Regulation to Requirements: An Automated Requirement Derivation and Explanation Pipeline 

**Authors**: Pavithra PM Nair, Preethu Rose Anish  

**Link**: [PDF](https://arxiv.org/pdf/2607.04448)  

**Abstract**: Ensuring software compliance with regulations such as the General Data Protection Regulation (GDPR) and the Artificial Intelligence Act (EU AI Act) poses a significant challenge, as requirements engineers must translate complex legal text into actionable software requirements - a process that remains largely manual and error-prone in practice. We present an automated regulation-to-requirements pipeline that identifies requirement-bearing clauses in regulatory documents and derives system-agnostic software requirements, accompanied by plain-language explanations, traceable to their legal sources. We evaluate the pipeline on the full clause sets of the GDPR (398 clauses) and the EU AI Act (574 clauses). For requirement-bearing clause identification, the approach achieves macro-averaged F1 scores of 0.82 and 0.78, respectively, outperforming a SetFit-based baseline. Human evaluation shows high completeness (4.60 and 4.45) and correctness (3.74 and 3.54) of derived requirements, while explanation clarity scores are near-ceiling (4.92 and 4.94) on a 1-5 scale. We implement the approach in Reg2Req, a publicly released tool that further supports requirement classification, use case seeding, cross-reference analysis, definition indexing, and a traceability matrix to operationalize regulatory compliance in practice. A user study with 25 practitioners shows that the plain-language explanations significantly improve comprehension of derived requirements and confidence in acting on them (p < 0.001), and that all participants would use Reg2Req as a starting point for deriving software requirements from a regulation. 

---
# Wan-Streamer v0.2: Higher Resolution, Same Latency 

**Authors**: Lianghua Huang, Zhi-Fan Wu, Yupeng Shi, Wei Wang, Mengyang Feng, Junjie He, Chen-Wei Xie, Yu Liu, Jingren Zhou, Ang Wang, Bang Zhang, Baole Ai, Chen Liang, Cheng Yu, Chongyang Zhong, Jinwei Qi, Kai Zhu, Pandeng Li, Peng Zhang, Wenyuan Zhang, Xinhua Cheng, Yitong Huang, Yun Zheng, Yuxiang Bao, Yuzheng Wang, Zoubin Bi  

**Link**: [PDF](https://arxiv.org/pdf/2607.04443)  

**Abstract**: We present Wan-Streamer v0.2, a latency-preserving upgrade of the native-streaming, end-to-end audio-visual interaction model. v0.2 keeps the v0.1 modeling formulation, but raises the interactive output stream from 192x336 to 640x368 while preserving approximately 200 ms model-side signal-to-signal latency at 25 FPS. The higher-resolution stream supports scene-grounded mid-shot agents whose posture, gaze, hands, nearby objects, and local scene layout remain legible during real-time conversation. To support the larger visual stream without adding user-visible delay, v0.2 keeps the thinker as a single-GPU low-latency path for streaming perception, the short language/state Transformer pass that builds the generation cache, and final decoding. The performer becomes a multi-GPU Ulysses-style context-parallel group for the expensive next-unit latent generation. Each performer rank writes incoming K/V into a pre-sharded local cache. The long high-resolution latent video sequence is split across ranks for denoising and gathered through Ulysses communication, while the much shorter audio latent sequence is generated without sequence sharding. In this split, the thinker's language/state computation reaches the performer only as K/V conditioning, so no separate language sequence has to be communicated inside the performer group. This concentrates additional hardware on visual generation while preserving the compact thinker-performer boundary, keeping total remote interaction latency at approximately 550 ms when a 350 ms bidirectional network budget is included. 

---
# Generative wave propagator 

**Authors**: Shijun Cheng, Tariq Alkhalifah  

**Link**: [PDF](https://arxiv.org/pdf/2607.04440)  

**Abstract**: Seismic wavefield simulation is fundamental to seismology, but conventional finite-difference (FD) methods remain limited by numerical dispersion and stability constraints, which often require dense spatial grids and small time steps and thereby severely limit the effectiveness of iterative inversion workflows. We introduce a conditional diffusion-based wavefield propagator that advances seismic wavefields recursively from one time step to the next. Instead of learning an unconditional data distribution of wavefield evolution, the model is conditioned by a short history of recent wavefield time steps (snapshots), the velocity model, and the wavefield time step index, allowing it to represent the conditional transition between adjacent physical states. By training the network to directly predict the clean next wavefield snapshot, this strong physical conditioning makes it possible to replace the iterative reverse diffusion process with a single network evaluation for each predicted snapshot. To improve stability over long recursive rollouts, we further introduce a causal time-weighted loss, in which adaptive weights, accumulated as exponential moving averages of per-snapshot training errors, emphasize training directions that are consistent with the forward propagation sequence and reduce the amplification of one-step prediction errors. Because the learned propagator is tied to the temporal spacing of the training snapshots rather than to the FD stability limit, it can advance the wavefield using a physical time step ten times larger than that required by the underlying solver. Experiments on the Overthrust, SEG/EAGE, and Marmousi models show that the proposed method accurately reproduces wavefield snapshots and shot gathers and achieves an end-to-end speedup of 2.17 x over a GPU-accelerated tenth-order staggered-grid FD implementation under matched hardware conditions. 

---
# ResearchStudio-Reel: Automate the Last Mile of Research from Paper to Poster, Video, and Blog 

**Authors**: Lingao Xiao, Yalun Dai, Yangyu Huang, Qihao Zhao, Wenshan Wu, Hugo He, Ruishuo Chen, Jin Jiang, Qianli Ma, Jiahuan Zhang, Xin Zhang, Ying Xin, Yang Ou, Yan Xia, Scarlett Li, Longbo Huang, Zhipeng Zhang, Yang He, Yap Kim Hui, Yan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2607.04438)  

**Abstract**: Research dissemination, turning a paper into a poster, a talk video, and a blog post, is still a manual last mile. Prior automation treats each artifact in isolation that each re-extract the paper from scratch, usually ship one-way renders the author cannot reopen in PowerPoint or Word, and gates quality on soft VLM-preference scores that plateau while load-bearing sections still read as empty. We argue this last mile is best built as a composition of skills: thin agent-readable contracts that share one upstream extractor and wrap deterministic primitives in a measured-fill loop whose exits are hard pass/fail render gates. We instantiate this as ResearchStudio-Reel, five Claude Code and Codex skills organized into one shared extractor (Paper2Assets), three editable generators (Paper2Poster, Paper2Video, Paper2Blog), and one interactive convergence layer (Paper2Reel). Paper2Assets extracts each paper once into a shared bundle that can be reused by every downstream skill; The three generators produce a print-ready poster, a synchronized talk video, and a bilingual blog that stay factually consistent and round-trip through PowerPoint or Word; Paper2Reel then binds all three into a self-contained HTML viewer whose section-level clicks jump the video, slides, captions, and blog to matching content. On the Paper2Poster benchmark, our posters lead every aesthetic and information sub-criterion against both prior automated systems and single-shot frontier LLMs, surpassing the authors' own on aesthetics under two held-out VLM judges and winning overall on 84% to 93% of papers; capability audits further show that, by uniquely pairing narration-aligned on-slide highlights with a bilingual blog gated by layout-aware DOCX repair, ResearchStudio-Reel is the only pipeline to ship all three editable artifacts. Project is available at this https URL 

---
# A Retrieval-Augmented Framework for Detecting and Resolving Pragmatic Ambiguities in Natural Language Requirements 

**Authors**: Pavithra PM Nair, Preethu Rose Anish  

**Link**: [PDF](https://arxiv.org/pdf/2607.04436)  

**Abstract**: Natural language requirements (NLRs) are essential for bridging communication gaps among diverse stakeholders in software development. However, the inherent ambiguity in NLRs can pose significant challenges. In particular, some requirements may be misinterpreted due to varying contextual knowledge and domain-specific expectations of the stakeholders, a phenomenon known as pragmatic ambiguity. This paper presents an approach for detecting and resolving pragmatic ambiguities in NLRs. The approach leverages retrieval-augmented generation techniques with novice, intermediate, and expert domain knowledge bases to simulate stakeholders with varying domain expertise and detect discrepancies in requirement interpretation. Candidate disambiguated requirements are generated using the expert domain knowledge base, with final validation by a requirements analyst required to ensure alignment with the intended functionality. We evaluate the approach on two requirements specification documents from the PUblic REquirements dataset, using four large language models: GPT-4o-mini, Mistral-7B, Llama-3.1-8B, and Qwen2.5-7B. Detection performance is assessed using macro-averaged accuracy, precision, recall, F1, and F2 scores. The resolution quality of the candidate disambiguated requirements is measured through human evaluation of relevance, clarity, and consistency. In this initial evaluation, results show that the proposed approach can detect pragmatic ambiguities and produce candidate disambiguated requirements that are relevant, clear, and consistent with the intended system functionality. Among the evaluated models, GPT-4o-mini achieved the highest macro-averaged recall (0.75) and F2 score (0.75) for pragmatic ambiguity detection. In the resolution task, GPT-4o-mini received the highest relevance scores from human evaluators, while Mistral-7B achieved the highest scores for clarity and consistency. 

---
# RoboDojo: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies 

**Authors**: Tianxing Chen, Yue Chen, Zixuan Li, Junyuan Tang, Kailun Su, Weijie Wan, Baijun Chen, Haoran Lu, Haowen Yan, Honghao Su, Zhiyang Dou, Kaixuan Wang, Dandan Zhang, Yunze Liu, Yan Qin, Qiwei Liang, Qiwei Wu, Zijian Lin, Wenwei Lin, Yuran Wang, Minghua He, Tianshu Wu, Ruihai Wu, Jingquan Zhou, Kai-Chong Lei, Haibao Yu, Yuanfeng Ji, Weiyang Jin, Guanyu Lin, Xiaofan Li, Qi Xiong, Renjing Xu, Zhongyu Li, Wenhao Chai, Enze Xie, Ziwei Wang, Yao Mu, Hao Dong, Wojciech Matusik, Mingyu Ding, Wenbo Ding, Ping Luo, Masayoshi Tomizuka  

**Link**: [PDF](https://arxiv.org/pdf/2607.04434)  

**Abstract**: Generalist robot manipulation policies have advanced rapidly, yet existing benchmarks remain limited in systematically evaluating their capabilities. Many rely on simple, short-horizon, or skill-narrow tasks with limited capability coverage, and are often conducted only in simulation or only in the real world. Simulation enables scalable feedback but misses physical deployment challenges, while real-world evaluation is costly, time-consuming, and difficult to reproduce. We introduce RoboDojo, a unified sim-and-real benchmark for comprehensive evaluation of generalist robot manipulation policies. RoboDojo includes 42 simulation tasks and 18 real-world tasks covering diverse and complementary manipulation capabilities. The simulation benchmark evaluates five dimensions: generalization, memory, precision, long-horizon execution, and open-vocabulary instruction following, while the real-world benchmark exposes policies to challenging physical-world deployment conditions. RoboDojo supports scalable evaluation through heterogeneous parallel simulation in Isaac Sim and provides RoboDojo-RealEval, a reproducible real-world evaluation system with remote cloud access, standardized hardware, scene reset, evaluation protocol, and deployment interface. Together with XPolicyLab, policies can be integrated once and evaluated across simulation and real-world settings with minimal adaptation. We integrate 30 policies into XPolicyLab and evaluate them on RoboDojo, establishing a public leaderboard and systematic analysis of current policy performance. The website is available at this http URL. 

---
# Covert Trait Propagation Is Representation Alignment: Mechanistic Evidence from Hidden-Channel Distillation 

**Authors**: Kargi Chauhan, Aditya Shah  

**Link**: [PDF](https://arxiv.org/pdf/2607.04432)  

**Abstract**: A student model trained on pure uniform noise can still inherit its teacher's digit-classification ability, provided the two share initialization. Previous work proves this transfer is guaranteed when the teacher's learning rate is small enough, but does not explain where in the network the channel lives or what sets its capacity. Working in an MLP distillation setting on MNIST, we show these channels are not purely informational: geometric alignment gates access to the information the channel carries. Shared initialization makes the output projection W_2 a common coordinate key, and KL gradients reshape the student's input projection W_0 until its hidden representations align with the teacher's. We call this covert trait propagation (CTP). Five experiments support this mechanism: channel closure tracks weight drift, not teacher accuracy; freezing W_0 destroys transfer while freezing W_2 leaves it intact; multi-teacher ensembles cancel out despite each teacher carrying comparable label information; and linear centered kernel alignment (CKA) tracks student accuracy at r=0.98 across a continuous initialization sweep. Applying the same geometric lens to cross-token behavioral entanglement (CTBE) in instruction-tuned LLMs, we find the effect appears to be activated by alignment training, acting on an inherited substrate, and that the standard log-ratio metric produces an apparent frequency bias that is largely a circularity artifact. 

---
# On Pairwise Quantile Regression -- Statistical Guarantees and Applications 

**Authors**: Romain Thérézien, Stephan Clémençon, Fantin Girard, Hamza El-Abdouni  

**Link**: [PDF](https://arxiv.org/pdf/2607.04431)  

**Abstract**: Quantile regression provides a powerful tool for summarizing the conditional distribution of a real valued random variable (r.v.) of interest $Y$ as a function of covariates $Z$ in cases where it shows a large dispersion with high probability, going beyond the situation where standard least square regression is informative/predictive. This article aims to extend this methodology to the pairwise case, when the variable to be explained takes the form of a similarity function between two independent observations, such as pixelated ID photos, as input data of biometric systems) and the explanatory variables take the form of a pair of covariates of the observations, such as the age or the hair color. We establish theoretical guarantees for solutions of this statistical learning problem, considered here as empirical minimizers of a pairwise version of the pinball loss. Leveraging sharp concentration results for $U$-processes, we prove generalization bounds and identify mild conditions under which fast learning rates can be achieved. Confirming the probabilistic analysis, experiments based on simulation data also provide solid empirical evidence of the validity of the methodology promoted here for pairwise quantile regression. Finally, its usefulness from an application perspective is demonstrated by a detailed study aimed at analyzing errors in similarity scoring for facial recognition. 

---
# evalci: A Python Library for Statistically Rigorous Comparison of Language Model Evaluations 

**Authors**: Shreyas K Chandrahas  

**Link**: [PDF](https://arxiv.org/pdf/2607.04429)  

**Abstract**: The dominant practice in language model evaluation is to report a single accuracy number per model and declare the higher one better, without testing whether the gap could plausibly be sampling noise. On benchmarks of a few thousand items, and under temperature sampling where a model can differ from itself run to run by more than the reported gap between models, this practice routinely overstates confidence in headline claims. The statistical machinery to fix this -- confidence intervals, paired significance tests, power analysis, clustered standard errors, multiple-comparison correction -- is well established, but no standard, pip-installable tool packages it in the shape an evaluation actually takes: a per-item results table. We present evalci, a pure-Python library (numpy/scipy/pandas only) that turns a per-item results table into a publication-ready claim -- e.g., "Model A beats Model B, $\Delta=3.1$ pts, 95% CI [1.2, 5.0], paired permutation $p=0.002$, $n=1{,}319$" -- in one function call, with adapters for lm-evaluation-harness and HELM output. Every routine is validated against an independent reference (statsmodels, or brute-force exact enumeration) rather than only against itself. As a case study, we re-analyze a public comparison of nine language models' MMLU accuracy and find that 3 of the 8 adjacent leaderboard-rank gaps are not statistically significant after correcting for the 36 pairwise comparisons the ranking implies. evalci is available at this https URL (source: this https URL, DOI: this https URL) 

---
# dOPSD: On-Policy Self-Distillation for Diffusion Language Models 

**Authors**: Phuong Tuan Dat, Qi Li, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.04428)  

**Abstract**: Diffusion large language models (dLLMs) generate text by iteratively denoising a masked sequence, offering a parallel alternative to autoregressive models, but eliciting strong reasoning through post-training remains difficult: supervised fine-tuning is off-policy and suffers from exposure bias, while reinforcement learning gives only sparse, sequence-level rewards and is hard to apply without tractable sequence likelihoods. On-policy self-distillation (OPSD) offers a promising alternative, using one model as both student and teacher to provide dense, token-level, on-policy supervision, but its effectiveness hinges on giving the teacher privileged information (PI) - typically an instance-specific ground-truth reference unavailable at inference - so the student ends up distilling a weak PI-free consensus policy that yields little improvement on dLLM reasoning. We introduce dOPSD, which instead derives the teacher's privilege directly from the student's own denoising trajectory, evaluating masked positions using later, more-decoded steps of that same trajectory rather than an external label, so the teacher's advantage emerges from the model's own decoding process; on Dream and LLaDA, dOPSD improves both in-domain math reasoning and out-of-domain code generation, outperforming supervised and on-policy baselines. 

---
# UI-MOPD: Multi-Platform On-Policy Distillation for Continual GUI Agent Learning 

**Authors**: Niu Lian, Alan Chen, Zhehao Yu, Chengzhen Duan, Fazhan Liu, Hui Liu, Pei Fu, Jian Luan, Yaowei Wang, Shu-Tao Xia, Jinpeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.04425)  

**Abstract**: Recent advances in multimodal foundation models and agent systems have driven GUI agents from single-platform task execution toward cross-platform interaction. However, building multi-platform GUI agents remains challenging. On one hand, high-quality and executable cross-platform interaction trajectories are still scarce, and existing data often suffer from limited platform coverage. On the other hand, different platforms exhibit distinct interaction conventions, making joint or continual training prone to behavioral pattern mixing, platform-specific capability degradation, and catastrophic forgetting. To address these challenges, we construct Uni-GUI, a high-quality cross-platform GUI interaction dataset, and propose UI-MOPD, the first method that incorporates multi-teacher on-policy distillation into continual learning for GUI agents. UI-MOPD dynamically selects a platform-specific teacher according to the current environment and transfers platform-specific behavioral priors to a shared policy through platform-conditioned distillation, enabling adaptation to new platforms while preserving capabilities on existing ones. Experiments on OSWorld and MobileWorld show that UI-MOPD achieves task success rates of 38.2% and 12.0%, respectively, demonstrating its effectiveness in balancing cross-platform capability retention and new-platform adaptation.
Project page: this https URL. 

---
# Transferability Between Understanding and Generation in Unified Multimodal Models 

**Authors**: Jiwon Kang, Heeji Yoon, Jaewoo Jung, Jaewon Min, Minkyeong Jeon, Biyeon Hwang, Sangwon Jung, Seungryong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.04423)  

**Abstract**: Unified Multimodal Models (UMMs) integrate image understanding and generation within a single architecture, yet how the two tasks interact remains understudied. We investigate $\boldsymbol{\mathsf{transferability}}$ in UMMs: whether training a capability on one task improves the same capability on the other without explicit supervision. Through controlled experiments, we empirically find that transferability depends on architecture-models with fully shared transformer backbone and a unified visual encoder exhibit consistent cross-task transfer, while loosely coupled designs show little or none. Leveraging this transferability, we propose a practical training strategy. The most straightforward way to improve a target generative capability (e.g., counting) is to fine-tune generation directly, but this can degrade visual quality due to distribution shift. Instead, we train the corresponding understanding task and let it transfer into generation, which improves capability-specific generative performance while minimizing distribution shift. We validate this across three capabilities-counting, spatial relation, and text recognition/generation-showing that cross-task transferability can be systematically exploited in UMMs. 

---
# Full-Stack FP4: Stable LLM Pretraining with Quantized Projections, Optimizers, and Attention 

**Authors**: Siyu Ding, Mingchuan Ma, Jiabo Tong, Xingrun Xing, Ziming Wang, Guoqi Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.04422)  

**Abstract**: Recent NVFP4 pretraining methods mainly target transformer linear layers, leaving optimizer states, optimizer arithmetic and attention underexplored in 4-bit pipelines. This critical gap blocks stable full-stack 4-bit pretraining, as the three core modules exhibit unique numerical failure patterns: linear layers hit hard quantization noise limits with dimension-propagated error amplification; AdamW second moments are heavy-tailed non-negative values fragile to low-precision denominators; attention carries error-prone computation paths demanding strict forward-backward quantization consistency. We propose Full-Stack FP4, the first complete NVFP4 pretraining framework resolving all three stability bottlenecks via module-wise precision strategies. For linear projections, LoRA-SVD lightweight decomposition suppresses quantization noise and breaks the direct-quantization error ceiling, shrinking the linear-only loss gap from 1.40% to 0.61%. For optimizers, we design AdamW second-moment transformation for robust NVFP4 storage and fully support native NVFP4 Newton-Schulz iterations for the Root (Muon) optimizer. For attention, a mixed-precision scheme quantizes Q/K/V and backward dS while guarding vulnerable paths in BF16, paired with unified tensor reuse to sustain forward-backward alignment. We further analyze fast error accumulation in naive low-bit matrix multiplication and the extreme sensitivity of PV / dOV^T attention branches. All modules are plug-and-play with cumulative stability and efficiency improvements. Our 3B/64B-token pretraining validates near-BF16 performance with merely 1.47% loss gap, verifying feasible stable end-to-end NVFP4 LLM pretraining. 

---
# Auto-AEG: Scalable Data Construction for Open-Vocabulary Audio Event Grounding 

**Authors**: Zihan Zhang, Xize Cheng, Wenhao Yan, Tong Zhang, Dongjie Fu, Boyun Zhang, Yongbo He, Tao Jin  

**Link**: [PDF](https://arxiv.org/pdf/2607.04383)  

**Abstract**: Large Audio-Language Models (LALMs) reason fluently about sound yet struggle to localize precisely when events occur, while classical Sound Event Detection attains frame-level precision only over a closed label set. At the intersection of these paradigms lies the task of Open-Vocabulary Audio Event Grounding: predicting all time intervals of a target sound event described by an arbitrary natural language query. While this task is crucial for real-world audio understanding and LALM adaptation, it is bottlenecked by data scarcity. Few large-scale resources provide open-vocabulary onset/offset supervision, and manual temporal annotation is prohibitively expensive. To address this, we introduce Auto-AEG, a scalable pipeline that constructs such supervision by automatic data construction and model fine-tuning. It pairs programmatically synthesized clips, which carry exact ground-truth intervals for supervised cold-start, with multi-model pseudo-labels on real-world audio that supply the reward signal for reinforcement learning. Training with this pipeline yields promising performance gains on both the DESED SED benchmark and AEGBench, an independent difficulty-stratified benchmark we release. Our results show that automatically constructed data, coupled with interval-aware reward function design, is an effective data-side route to expanding the temporal localization capability of LALMs. 

---
# HASSL: Hierarchy-Aware Self-Supervised Learning Framework for Single Cell Microscopy 

**Authors**: Julius Riel, Vishwa Mohan Singh, Sai Anirudh Aryasomayajula, Anuun Chinbat, Hannes Leonhard, Moritz Ladenburger, Frederik Alexander, Vishisht Choudhary, Fabio Laredo, Giacomo Masserdotti, Thorben Prein, Carsten Marr, Amirhossein Kardoost  

**Link**: [PDF](https://arxiv.org/pdf/2607.04353)  

**Abstract**: Hierarchical structure is common in image data, where fine-grained clusters often merge into larger, coarser semantic groups. In biological cell images, current self-supervised learning models often suppress this hierarchy, as coarse factors such as imaging modality can obscure finer morphological attributes in the latent space. We propose a hierarchy-aware self-supervised training framework to address this problem. Our method combines two components: a distillation framework with a segmentation teacher to improve morphological awareness in the latent space, and a hierarchy-aware contrastive loss based on HDBSCAN to improve decision boundaries between closely related subtypes at different hierarchical levels. Together, these components reduce the tendency of self-supervised learning to overemphasize coarse factors and instead align embeddings with semantic and morphological cues. This yields biologically meaningful sub-clusters driven by fine morphological detail. We train and evaluate our method on a curated corpus of 2.3 million single cells aggregated from 20 microscopy datasets, both labeled and unlabeled, covering 208 cell classes. Our method improves over baseline and counterpart methods, increasing average top-K accuracy by 2.8%, top-9 retrieval on the dataset with the deepest hierarchy by 6.3%, and downstream F1-score for biologically relevant drug classification from perturbed cell morphology by 7.8%. 

---
# IRIS: An Intelligent Vision-Language System for Ocular Surface Diseases via Topic Tree and Scene-Driven VQA Generation 

**Authors**: Hao Wei, Wenjin Qi, Dasen Dai, Minqing Zhang, Wu Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2607.04344)  

**Abstract**: While Large Vision-Language Models (VLMs) demonstrate remarkable generic capabilities, their clinical reasoning in specialized domains like ocular surface diseases (OSDs) is severely hindered by a paucity of high-fidelity, multimodal instruction-tuning data. To dismantle this data bottleneck, we introduce IRIS, an Intelligent Recognition and Interaction System tailored for fine-grained OSD understanding via external eye photography. First, we curate IRIS-120K, the largest and most comprehensive OSD visual question-answering (VQA) dataset to date. Crucially, to overcome the semantic shallowness of conventional image-caption pairs, we propose a synergistic data generation paradigm to explicitly inject clinical priors. Our data engine operates via a dual-branch framework: 1) a Topic Finding Tree (TFT) that hierarchically anchors visual features to precise anatomical and pathological concepts, enforcing rigorous medical deduction logic; and 2) a Scene-driven strategy that synthesizes role-adaptive clinical dialogues to ensure pragmatic generalization. By explicitly aligning a compact 4B-parameter VLM on this structurally enriched corpus, IRIS achieves state-of-the-art performance, comprehensively outperforming both generalist and specialized medical VLMs with up to 34B parameters. Our findings underscore that structured knowledge injection profoundly prevails over sheer parameter scaling, unlocking the potential for resource-efficient, expert-level AI deployment on mobile edge devices for scalable OSD screening. Code, datasets, and model weights will be publicly released by this repo. 

---
# One Framework for All: Cross-Modal Membership Inference for Generative Models 

**Authors**: Dayong Ye, Tainqing Zhu, Kun Gao, Junhao Liu, Yichuan Chen, Shuai Zhou, Hengzhu Liu, Bo Liu, Wanlei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2607.04339)  

**Abstract**: Large generative models across text-to-text, text-to-image, and image-to-text modalities have been shown to pose significant privacy risks. One fundamental threat is membership inference attacks (MIA), which aim to determine whether a given data point was used in a model's training set. Although prior work has investigated MIAs against these three classes of generative models, existing approaches treat them in isolation and are not cross-applicable, thereby limiting their real-world utility. To address this limitation, we present the first comprehensive study of a unified membership inference framework that applies across text-to-text, text-to-image, and image-to-text modalities. Our approach is grounded in a key modality-agnostic observation: the output distribution of a generative model can approximate its training data distribution. Leveraging this property, we model the distributions of model-generated outputs and auxiliary non-member samples in a shared embedding space, and perform membership inference via likelihood ratio testing. We conduct extensive experiments in a strict black-box setting under both partial-knowledge and zero-knowledge threat models, and evaluate membership inference against both fine-tuning and pre-training data. Experimental results demonstrate our approach's superior performance in comparison to existing state-of-the-art methods, which are typically optimized for a single model class. 

---
# Fixed-Confidence Best-Arm Identification for Causal Mediation Analysis 

**Authors**: Harsh Shrivastava, Yuta Kawakami, Junpei Komiyama, Jin Tian  

**Link**: [PDF](https://arxiv.org/pdf/2607.04315)  

**Abstract**: This paper studies the problem of identifying the treatment that maximizes the expected natural direct potential outcome (NDPO), which captures the potential outcome of an intervention while excluding the pathway transmitted through a mediator that researchers may wish to remove from evaluation. We first establish population-level identification of the expected NDPO in a causal bandit setting using observable interventional distributions. We then develop a fixed-confidence best-arm identification (BAI) algorithm based on the Track-and-Stop (TaS) framework, employing a cutting-set method to solve the resulting semi-infinite optimization problem. The proposed algorithm achieves sample-efficient identification with a high-probability correctness guarantee. We prove that it satisfies $\delta$-correctness and asymptotic optimality. Finally, we validate the approach through empirical evaluations on a large-scale real-world advertising dataset (IPinYou). 

---
# HiFA4: Training-Free 4-bit FlashAttention on Ascend HIF4 NPUs for LLM Inference 

**Authors**: Hui Dong, Yanzhao Li, Jie Gao, Chunlu Li, Zhiyuan Zhang, Yupeng Sun, Zhenyuan Chen, Zhiqiang Zou  

**Link**: [PDF](https://arxiv.org/pdf/2607.04302)  

**Abstract**: We present HiFA4, a post-training operator-level design that executes both QK^T and PV in FlashAttention as 4-bit HIF4 Cube GEMMs for LLM inference on Ascend NPUs, while maintaining the online softmax state in FP16. To our knowledge, HiFA4 is the first Ascend-HIF4-targeted design of this kind evaluated on standard NLP benchmarks.
HiFA4 combines two mechanisms. Smooth-QK applies a calibration-static per-channel equivalent rescaling to Q and K after RoPE, transferring quantization difficulty from K to Q without per-tile online reduction at inference. P-Reordering accumulates the softmax normalizer from the same quantized attention weights P_hat used in the PV GEMM, rather than from a higher-precision reconstruction. We show that this inconsistent formulation introduces a coherent output-scaling error, and validate the effect on a Qwen3-8B Layer-0 MMLU trace, where all 3.6M measured attention tiles exhibit net probability-mass loss with median epsilon_bar = -0.064. P-Reordering also allows the normalizer to be fused into the PV Cube GEMM.
Across five LLMs, HiFA4 reduces quantization-induced decision drift. On Qwen3-8B, it recovers 37.5% of the accuracy gap introduced by direct HIF4 quantization, narrows the sample-weighted accuracy loss from 1.12 pp to 0.70 pp, reduces BF16-inconsistent MMLU predictions from 16.3% to 8.2%, and cuts MMLU accuracy regressions by 57% (1071 to 465). On Gemma2-9B, mild smoothing keeps HiFA4 within 0.7 pp of BF16 while reducing MMLU regressions by 27%. On LLaMA3.1-8B, Mistral-7B, and Phi-4B, where Smooth-QK is disabled, P-Reordering with the adopted Q-Mean auxiliary still reduces full-set MMLU regressions by 41-52%. A preliminary instruction-scheduling analysis projects a 35.4% critical-path latency reduction relative to BF16 by fusing the softmax normalizer into the PV Cube GEMM; on-hardware validation is left to future work. 

---
# CausalGame: Benchmarking Causal Thinking of LLM Agents in Games 

**Authors**: Zhenhao Chen, Yongqiang Chen, Chenxi Liu, Junchi Yu, Xiangchen Song, Zijian Li, Jialin Li, Philip Torr, Bo Han, Kun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.04293)  

**Abstract**: Building AI Scientist agents with Large Language Models (LLMs) has recently attracted growing attention. Since scientific discovery fundamentally relies on uncovering causal relationships from observations, the capability of causal thinking, i.e., distinguishing causation from correlation and recognizing hidden biases, is essential to LLM agents. Although a number of benchmarks exist for AI Scientists, none explicitly incorporate challenges from selection bias, measurement error, and hidden confounders that widely exist in real-world scientific discovery. To this end, we present CausalGame, a benchmark that evaluates the causal thinking capabilities of LLM agents through interactive games. CausalGame asks LLM agents to actively design experimental protocols, collect observation data, and derive a final solution with an explanation report. To emulate realistic scientific discovery challenges, we design 14 scenarios that incorporate selection bias, measurement error, and hidden confounders. Across 30 LLM agents, none demonstrates reliable causal thinking: the best model reaches only 68.0% survival against analytical optima of 78-85%, and merely 5-7% of sessions receive credits on the causal-reasoning rubrics. CausalGame provides a scalable and controlled testbed for evaluating the causal thinking of AI Scientist agents. 

---
# Agentic-V2X: Small Language Model Agents for Deadline-Aware V2X Scheduling in 5G/6G Networks 

**Authors**: Gerasimos Papanikolaou-Ntais, Alexandros Kaloxylos, Athanasios Kanavos  

**Link**: [PDF](https://arxiv.org/pdf/2607.04290)  

**Abstract**: Large Language Models (LLMs) are proposed as control interfaces for next-generation networks, but their latency, hallucinations, and lack of control guarantees make them unsuitable for near-real-time packet schedulers, especially in dynamic V2X environments. This paper introduces Agentic-V2X, an architecture where a small, locally deployed language model acts as a periodic non-real-time rApp-inspired policy creator, while a lightweight xApp-like controller executes validated policies at intervals suitable for scheduling. The framework targets deadline-aware 5G NR V2X scheduling with heterogeneous services (teleoperated driving, cooperative awareness, HD map sharing, and sensor sharing). Given a scenario summary, service objective, and telemetry, the LLM generates a structured policy containing service priorities, weight bounds, and safety constraints. A validator checks and repairs the policy before the controller enforces it via scheduler-weight adaptation in ns-3/ns3-ai. The evaluation compares proportional fair scheduling, static expert policies, a heuristic xApp, static LLM policies, and adaptive LLM-rApp policies over 126 completed runs. Metrics include deadline-constrained packet reception ratio, tail latency, deadline violations, throughput, fairness, policy validity, and safety interventions. Results show that the adaptive LLM-rApp/xApp design generates valid and executable policies and remains competitive at several operating points, including improved mean critical reliability over PF at the highest density. However, paired statistical analysis shows that the adaptive method is not the best aggregate method and remains below the strongest static policies overall. These results support Agentic-V2X as a safe, executable small-LLM policy-generation architecture rather than a universally dominant scheduler. 

---
# Risk-Constrained Freshness-Aware Semantic Caching for Open-Web Retrieval-Augmented LLMs 

**Authors**: Muhammad Mansoor, Tahir Ahmad, Yeo-Chan Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2607.04281)  

**Abstract**: Semantic caching reduces the latency and cost of retrieval-augmented generation (RAG) by serving cached answers to semantically similar queries, but most existing methods do not model the time-varying freshness of open-web evidence. We present FreshCache, a three-tier semantic cache that treats cache reuse as a risk-constrained temporal inference problem: before approving a cache hit, FreshCache estimates the probability that the cached result is stale using a fitted exponential decay model enhanced by a learned MLP, and approves reuse only when that probability falls below a per-tier error budget across answers (epsilon = 0.10), URL lists (epsilon = 0.20), and page content (epsilon = 0.35). This allows the system to degrade gracefully as entries age rather than forcing a binary choice between a stale hit and a full pipeline execution. We introduce FreshCache-Bench, a benchmark of 8,072 base queries across five freshness classes with ground truth staleness labels drawn from real web snapshots at 1, 12, 24 hours, and 7 days after a baseline crawl, expanded to 31,201 queries via paraphrase generation. At the 24-hour evaluation window, FreshCache_MLP achieves 97% search API savings at 0.1% hash-based stale error, and an LLM-judge evaluation on 396 confirmed change pairs shows that only 34.3% of detected content changes actually affect answer correctness, placing true answer-affecting stale error at approximately 0.034%. The rule-based FreshCache achieves 98% search savings at 3.3% stale error under a temporal holdout calibration, outperforming SemanticTTL (14.9% stale, 72% saved), vCache (7.2% stale, 47% saved), and SCALM (5.2% stale, 96% saved). Ablations show the temporal risk gate accounts for an 11.6 point reduction in stale error over similarity-only reuse, and the learned MLP reduces stale error a further 3.2 points over the rule-based model. 

---
# Self-Reference in Large Language Models: The Introspection Threshold for Recursive Self-Improvement 

**Authors**: Jiang Zhang, Bing Yuan, Qian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.04277)  

**Abstract**: The pursuit of self-evolving AI raises a critical question: when is autonomous self-improvement sustainable rather than degenerative? Drawing an analogy to von Neumann's complexity threshold for self-reproducing automata, we argue that sustainable recursive self-improvement in Large Language Models (LLMs) requires a functional analogue: introspection -- the system's capacity to simulate its own operations and target modifications. Grounded in Kleene's Second Recursion Theorem, we demonstrate the theoretical existence of such introspective programs. However, an empirical review reveals that while current LLMs exhibit quasi-introspection (e.g., partial metacognition), they fall short of true introspection due to structural bottlenecks: a lack of complete self-access, the feedforward nature of the Transformer, and computational class constraints that prevent fixed-point iteration. We conclude by outlining architectural paths to cross this complexity threshold and discussing the associated safety implications. 

---
# LBR: Towards Mitigating Length Bias in Large Language Models for Recommendation 

**Authors**: Hongchen Li, Bohao Wang, Jingbang Chen, Weiqin Yang, Hang Pan, Bingde Hu, Can Wang, Jiawei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.04270)  

**Abstract**: Large language models (LLMs) have recently emerged as powerful backbones for recommender systems by reformulating recommendation as a token-level generation task. Despite their promise, we identify a pervasive yet underexplored issue: $\textit{Length Bias}$. Because items are represented by textual descriptions of varying lengths, LLM-based recommenders can be systematically biased in two ways. On the input side, longer item descriptions occupy more tokens in the context and thus receive disproportionately large aggregate attention mass during user preference modeling. On the output side, decoding based on summed autoregressive log-likelihood score inherently disfavors long items. Worse still, conventional length normalization can introduce an additional bias and even degrade recommendation performance.
To address this problem, we propose $\textbf{LBR}$ ($\textbf{L}$ength $\textbf{B}$ias $\textbf{R}$eduction), a lightweight and model-agnostic framework for mitigating length bias in LLM-based recommendation. LBR mitigates input-side bias via Length-Aware Attention Calibration, which incorporates a length-dependent offset into attention logits to neutralize attention skew. For the output side, LBR introduces Effective Information Length Normalization, replacing naive token count with an information-theoretic length surrogate derived from the branching structure of the prefix tree. Extensive experiments on three real-world Amazon datasets and two representative LLM-based recommenders demonstrate that LBR substantially alleviates length bias while consistently improving recommendation accuracy and fairness, with negligible additional training and inference overhead (with an average NDCG@5 gain of 16.82%). The code is available at this https URL. 

---
# HALO-WA: Hybrid-Attention Latent-Guided Online Reinforcement Learning for World-Action Models 

**Authors**: Angen Ye, Weijie Ke, Xiaofeng Wang, Xinze Chen, Chaojun Ni, Guosheng Zhao, Boyuan Wang, Zheng Zhu, Junjie Xie, Dapeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.04265)  

**Abstract**: World-action (WA) models can generate long-horizon action chunks for general-purpose robotic manipulation, but they remain vulnerable to calibration, perception, and contact-dynamics errors in real-world precision tasks, often failing in the final few millimeters of alignment or insertion. We propose HALO-WA, a hybrid-attention latent-guided online reinforcement learning (RL) framework for WA models, which leverages latent features and action priors from the WA generation process through a lightweight actor-critic adapter to enable fast online adaptation to real deployment errors. HALO-WA introduces a hybrid-attention structure that preserves the temporal consistency of action chunks while reading task-relevant information from WA latents conditioned on visual context and end-stage correction requirements, thereby producing refined action chunks. We validate HALO-WA on four real-world precision manipulation tasks, where it improves the average success rate from 26.4\% for WA-base to 87.1\%, outperforming the strongest baseline by 19.2 percentage points while requiring only 45--75 minutes of online training per task. To facilitate reproducibility, we further conduct supplementary simulation experiments in RoboTwin and release the code at this https URL. 

---
# Signal or Noise? Understanding Generative Models for Real-World Sensor Time Series 

**Authors**: Zitao Shuai, Zongzhe Xu, Yuntian Wu, Sirui Li, Tianhong Li, Yuzhe Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.04245)  

**Abstract**: Generative models have changed how machine learning represents complex data distributions, especially in language and vision, yet many real-world systems are observed instead as continuous, high-dimensional, and noisy sensor time series. Existing generative modeling of sensor data, however, remains fragmented across modalities, datasets, and task formulations, limiting a systematic understanding of when, how, and why generative models succeed or fail in real-world settings. To address this gap, we introduce SensorGen, a large-scale study of sensor-signal generation spanning 14 settings across 4 domains, 7 datasets, and 12 signal modalities. Leveraging SensorGen, we systematically evaluate generative models from five major families and uncover three key findings: (1) flow-matching models provide strong overall performance across most settings; (2) signal properties matter, with demographic covariates improving longitudinal generation and time-frequency modeling improving high-frequency signal generation; and (3) generated signals have practical utility beyond visual realism, with scaling improving generation quality and synthetic data improving downstream performance. Together, SensorGen establishes a broader understanding of design choices, evaluation protocols, and failure modes in real-world sensor data generation. 

---
# Hierarchical Multi-to-Single-Modal Knowledge Distillation for Disruption Prediction in EAST 

**Authors**: Qiang Chen, Xiao Wang, Hao Si, Qingquan Yang, Meiwen Chen, Jianhua Yang, Xiaofeng Han, Yunhu Jia, Ran Chen, Liang Wang, Jin Tang, Guosheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.04241)  

**Abstract**: Plasma disruption is a critical threat to tokamak safety. Existing data-driven predictors mainly rely on time-series diagnostic signals, while visible images provide complementary spatial cues including plasma deformation, local brightening, and radiation-structure evolution. Although the image modality improves the model's discriminative capability, it also substantially increases the computational cost during inference. To address this issue, we propose a hierarchical multi-to-single-modal knowledge distillation framework for disruption prediction on a synchronized EAST multimodal dataset. During training, visible images and time-series signals are used to train a multimodal teacher, which learns disruption precursor representations through Transformer-based encoders and a prototype-guided spatiotemporal hypergraph module. During inference, only the time-series student is retained, with multimodal knowledge transferred through graph-structure-level, representation-level, and decision-level distillation. On the 640-discharge EAST dataset, the results demonstrate that the proposed framework can preserve the discriminative advantages of multimodal learning while substantially reducing inference cost, and providing an effective route for efficient disruption prediction in EAST. The source code of this paper will be released on this https URL. 

---
# SoftVTBench: A Safety-Aware Visuo-Tactile Benchmark for Physically Constrained Robotic Manipulation of Deformable Objects 

**Authors**: Bowen Jing, Mingxin Wang, Ruiyang Hao, Chenchen Ge, Hanwen Shen, Junjie He, Yang Cui, Yiming Hou, Weitao Zhou, Jiawei Wang, Minglei Li, Dandan Zhang, Ding Zhao, Houde Liu, Xiaofan Li, Si Liu, Ping Luo, Haibao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2607.04234)  

**Abstract**: Deformable object manipulation poses challenges beyond task completion: successful execution must also maintain safe physical interaction, holding the object stably without slip or drop while avoiding excessive deformation. However, existing manipulation benchmarks are predominantly success-oriented and rarely evaluate whether a policy remains physically safe throughout execution. We present SoftVTBench, a safety-aware visuo-tactile benchmark for physically constrained deformable object manipulation. Built in Isaac Sim with finite-element-simulated deformable objects, SoftVTBench provides multi-view RGB observations, RGB tactile sensing with marker motion, proprioception, and language instructions, and defines four matched task suites over object type (deformable vs. rigid) and variation axis (object vs. spatial). It separately reports Goal Success and Safety Success; the latter additionally requires no drop and peak deformation below a calibrated object-specific threshold, measured from policy-hidden privileged Finite Element Method (FEM) states. We implement pi0.5-based baselines under this protocol. Experiments show that success-only evaluation substantially overstates policy performance, as a large fraction of goal-completing rollouts still violate physical safety. Furthermore, incorporating tactile sensing improves Safety Success (e.g., from 21.4% to 35.6% on object-centric deformable tasks) and reduces object deformation during execution, while maintaining comparable Goal Success. SoftVTBench provides a reproducible benchmark for studying visuo-tactile deformable manipulation under physical interaction constraints. 

---
# Detecting Hallucinations in Retrieval-Augmented Generation through Grounding-Aware Sensitivity by Perturbation (GASP) 

**Authors**: Mohamed Aly Bouke  

**Link**: [PDF](https://arxiv.org/pdf/2607.04223)  

**Abstract**: Retrieval-augmented generation (RAG) reduces but does not eliminate hallucination, and existing detectors return a single answer-level score that does not indicate which sentence is unsupported, or why. To close this gap, we introduce Grounding-Aware Sensitivity by Perturbation (GASP), a span-level detector that scores each answer sentence by how strongly its likelihood depends on the retrieved evidence, a quantity we term grounding sensitivity. GASP holds the answer fixed and re-scores it under the full context, under no context, and with each chunk removed, then measures the log-likelihood drops and Jensen-Shannon divergences (JSD). The likelihood of a grounded sentence collapses once its supporting passage is removed, whereas a hallucinated sentence is almost unaffected, a contrast we interpret by casting decoding as a random nonlinear iterated function system (RNIFS). We evaluate GASP on three benchmarks (RAGTruth, TofuEval, RAGBench) with three instruction-tuned scorers from two model families (Qwen2.5-0.5B, Qwen2.5-1.5B, and SmolLM2-1.7B) under a leakage-clean protocol. On RAGTruth it reaches a response-level area under the ROC curve (AUC) of about 0.73 and a span-level AUC of about 0.67, improving significantly over perplexity and by clear margins over length, whole-context natural language inference (NLI), and self-consistency baselines. The only baseline competitive at the span level is a well-configured chunk-level entailment verifier, which requires a separate model, whereas a training-free threshold on the grounding features matches the trained classifier without labeled data and serves as the default detector. Beyond RAGTruth, the signal transfers to TofuEval but not to short-answer question answering in RAGBench, showing GASP is best suited to outputs constructed from the retrieved context rather than answers recoverable from parametric knowledge. 

---
# CritiqueDriveVLM: From Verifier-Guided Reinforcement Learning to Latent Thought Distillation for Autonomous Driving 

**Authors**: Zhaohong Liu, Hao Ye, Xianlin Zhang, Mengshi Qi  

**Link**: [PDF](https://arxiv.org/pdf/2607.04179)  

**Abstract**: End-to-end Vision-Language Models (VLMs) show immense potential in autonomous driving. However, standard Supervised Fine-Tuning (SFT) often suffers from reasoning hallucinations and conservative biases. While traditional tool-augmented frameworks and Chain-of-Thought (CoT) approaches mitigate these issues, they incur exorbitant token consumption and unacceptable latency, rendering real-time deployment impractical. To resolve this reliability-efficiency trade-off, we propose CritiqueDriveVLM, a novel unified three-stage framework internalizing reasoning directly into the VLM. First, we introduce Critique-Driven Multi-Turn Reinforcement Learning (RL) guided by a multi-dimensional verifier. By providing granular scalar feedback and a multi-turn penalty, we force the policy to internalize logical deduction, cultivating a robust System-2 Teacher that achieves high accuracy without fragile external tools. Subsequently, we propose Latent Thought Distillation to overcome the latency bottleneck. By aligning the Student's latent representations with the Teacher's fully converged reasoning states, we compress deep logical capabilities into a fast, CoT-free System-1 Student. Extensive experiments on the widely-used DriveLMM-01 benchmark demonstrate remarkable improvements. Compared to the base model, our tool-free Teacher significantly boosts Multiple Choice Quality (MCQ) from 55.54% to a state-of-the-art 76.54%. Crucially, our distilled Student preserves competitive reasoning depth while drastically minimizing generation length to an average of merely 28 tokens. This slashes inference latency by 88% (from 3482 ms to 416 ms), paving a highly robust pathway for low-latency autonomous this http URL source code is available at this https URL. 

---
# Piercing Gilbreath's Conjecture: From Deep Number Theory Insights to Fintech and Cybersecurity 

**Authors**: Vincent Granville  

**Link**: [PDF](https://arxiv.org/pdf/2607.04166)  

**Abstract**: I propose a new methodology to attack the fascinating Gilbreath's conjecture about prime numbers, first posted in 1878 and unsolved to this day. The problem statement is rudimentary: kids can understand it. However, despite decades of research, almost no progress has been made. This paper changes the game by presenting a new approach based on sieving, a number of new results with proof, a precise path to the solution, and solid references. It also introduces the concept of reverse sieving, along with applications to testing randomness, pattern and fraud detection, cybersecurity, synthetic data, sequence categorization and normalization, or to detect and quantify a new type of chaos in time series including Brownian motions. Magic primes, forbidden prime number constellations, cellular automata, and reduction via classes of equivalent sequences, are some of the innovative and promising topics discussed in the paper. 

---
# SeeMe: Mitigating Hallucinations in Large Vision-Language Models through Effective Visual Token Engineering 

**Authors**: Kai Tang, Jinhao You, Bohua Zhang, Yichen Guo, Yiding Sun, Dongxu Zhang, Chenxi Li, Xiande Huang, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.04163)  

**Abstract**: Large Vision-Language Models (LVLMs) have achieved remarkable progress in visual understanding tasks such as image captioning and visual question answering. However, they remain susceptible to hallucinations, generating content that is inconsistent with the actual visual input. Existing methods primarily intervene at the decoding stage, while overlooking a critical source of hallucinations: irrelevant or noisy visual tokens that mislead the decoding process. To address this issue, we propose SeeMe, a training-free framework that introduces the concept of feature engineering from traditional machine learning into LVLMs. SeeMe restructures visual tokens through a three-stage token engineering process to suppress hallucination sources while preserving informative visual evidence. Experiments on MME, POPE, and AMBER benchmarks across four LVLMs demonstrate that SeeMe consistently reduces hallucinations and improves output consistency, providing a novel perspective for mitigating hallucinations in LVLMs. 

---
# Information-Geometric Superposed Vowel Evaluation: Part 1. Moraic Syllabary (Japanese) 

**Authors**: Yusei Tamura, Shigekazu Ishihara, Ken Ito  

**Link**: [PDF](https://arxiv.org/pdf/2607.04154)  

**Abstract**: This paper explains the principles and provides examples of a new method for distinguishing between FAKE human speech synthesized by generative AI and natural speech. Since synthetic speech is generated based on information from a limited set of training spectra, the variety of vowels - which are key to identifying individuals - is limited. In contrast, natural speech exhibits a more diverse distribution of vowel spectra due to the flexibility of the human articulatory organ. In this paper, using Japanese - a Syllabary limited to five vowel phonemes, each of which corresponds one-to-one with a specific sound - as an example, we outline a method for distinguishing between synthetic and natural speech reading the same text by analyzing the spectral distributions. If we normalize the spectra of speech sounds and regard them as probability density functions for the frequency bands received by the hair cells of the human cochlea, and evaluate the distance between spectra using the Wasserstein metric, the Wasserstein distances between the vowels of synthetic speech are short. By preserving this distance and performing a topological mapping using persistent homology, the spectral probability density functions of synthetic and natural speech can be decomposed into clusters. 

---
# Mask-based Predictive Representations for Reinforcement Learning 

**Authors**: Kai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2607.04153)  

**Abstract**: Vision-based deep reinforcement learning involves dealing with high-dimensional inputs of image information. It is crucial to abstract effective states from high-dimensional image inputs and limited samples for sample-efficient reinforcement learning. To address this challenge, inspired by fields such as natural language processing and computer vision, we propose a self-supervised task based on mask prediction as an auxiliary task for reinforcement learning. This non-reconstruction method uses the sequence information collected by the agent from the environment and the context information in the sequence to predict the masked information, thereby strengthening the agent's understanding of the task and learning effective representations. Combined with transformers, we find that the model reconstructs the masked input sequence in the latent space. By feeding the compressed representations learned by this method into reinforcement learning models, we observe an improvement in the sample efficiency of reinforcement learning. Moreover, the model outperforms state-of-the-art sample-efficient reinforcement learning methods on multiple continuous and discrete control benchmarks. 

---
# HCSU: A Dataset and Benchmark for Fine-Grained Historical Calligraphy Style Understanding 

**Authors**: Yinsheng Yao, Yan Liu, Chen Ye  

**Link**: [PDF](https://arxiv.org/pdf/2607.04147)  

**Abstract**: Automated fine-grained perception of calligraphy styles--a task vital to cultural heritage preservation--remains a critical challenge for Large Vision-Language Models (LVLMs), largely constrained by existing datasets that suffer from modal mixture and flattened labels. To bridge this gap, we introduce HCSU, the first comprehensive dataset tailored for fine-grained Historical Calligraphy Style Understanding. HCSU comprises 39,307 meticulously curated character images from 49 historically prominent calligraphers across 10 dynasties, systematically decoupling authentic ink manuscripts (Tie) from stone rubbings (Bei) to resolve the long-standing modal mixture problem. Moving beyond conventional flattened labels, HCSU provides hierarchical expert-written aesthetic descriptions, enabling two rigorous evaluation protocols: fine-grained style discrimination and interpretable aesthetic reasoning. Extensive evaluations reveal a persistent gap between calligraphy-related knowledge and visually grounded style perception: state-of-the-art LVLMs show non-trivial performance but remain sensitive to script-level, textual, and source-specific cues, and often struggle to ground aesthetic judgments in fine-grained brushwork evidence. Ultimately, the HCSU benchmark exposes fundamental limitations in current multimodal architectures, aiming to inspire the evolution of expert-level visual reasoning for cultural heritage preservation. The dataset is available at this https URL. 

---
# !Imperio, smolVLA: The Implications of Data Poisoning on Open Source Robotics 

**Authors**: Stefan Bühler, Mark Schutera  

**Link**: [PDF](https://arxiv.org/pdf/2607.04146)  

**Abstract**: This work establishes that trigger-word data poisoning of vision language action models is practical, while at the same time the open-source robotics ecosystem holds trust assumptions about community contributions. A few poisoned samples can silently embed a backdoor that disables a robot on command. We evaluate this threat against smolVLA on a real-world pick-and-place task, training on three poison ratios and evaluating across different prompts on the LeRobot platform. Three poisoned episodes in 320 clean episodes suffice for a complete denial of service. Success rate drops to 0.0 plus minus 0.0% across all trigger-word conditions and the robot locks into a fixed joint configuration rather than executing any task-relevant motion. Clean-prompt behaviour holds at approx. 50% success rate across all poison ratios, confirming the attack is stealthy under normal operation. A single poisoned episode already reduces success rate to 6.7 plus minus 6.7%. The robot still moves, but no longer completes the task. The attack generalises to front, middle, and end trigger placements despite training exclusively on front-placed triggers. These findings establish that the threat is practical, low-cost, and stealthy, and warrant treating dataset provenance as a first-class concern in open-source robotics ecosystems. 

---
# CSB: A Counting and Sampling tool for Bit-vectors 

**Authors**: Arijit Shaw, Kuldeep S. Meel  

**Link**: [PDF](https://arxiv.org/pdf/2607.04142)  

**Abstract**: Satisfiability modulo theory (SMT) solvers have significantly advanced automated reasoning due to their effectiveness in solving problems across various fields. With the advancement in SMT solvers, there is growing interest in exploring capabilities beyond mere satisfiability, similar to the progression observed in Boolean satisfiability solvers that expanded into counting and sampling. In this study, we investigate the following question: Can we rely on modern CNF model counters and CNF samplers to extend modern SMT solvers to handle the problems of counting and sampling over bit-vectors?
The main contribution of this work is the development of an efficient and user-friendly tool, csb, that solves a bunch of problems around model counting and sampling on the theory of bit-vectors, namely exact and approximate projected and non-projected model counting, along with the almost-uniform and uniform-like sampling. In the case of exact counting, projected counting, and uniform sampling. Our tool csb converts the bit-vector formula into a CNF formula using bit-blasting techniques before applying CNF model counters or samplers to perform counting or sampling. Our experiments demonstrate significant performance improvements over existing methods. 

---
# Conflict-Based Lazy Search for Fast Multi-Manipulator Planning 

**Authors**: Dongliang Zheng, Zhipeng Wang, Siqi Wang, Yuxi Lu, Bin He, Hesheng Wang, Panagiotis Tsiotras  

**Link**: [PDF](https://arxiv.org/pdf/2607.04124)  

**Abstract**: Employing multiple manipulators can boost efficiency and accomplish tasks that a single manipulator cannot do. However, real-time planning for multiple manipulators in a cluttered workspace still poses significant challenges for planning algorithms. This article proposes a new planning algorithm called Conflict-Based Lazy Search (CBLS) for multimanipulator planning. CBLS is built on Conflict-Based Search (CBS), an efficient multiagent pathfinding (MAPF) algorithm that has shown an order of magnitude speedup over previous approaches [1], [2]. CBS addresses MAPF by solving many single-agent pathfinding (SAPF) problems. Thus, its planning time directly depends on the efficiency of the SAPF algorithm adopted. Our CBLS algorithm enhances CBS with precomputation and lazy search. First, a lazily evaluated graph with controlled sparsity is precomputed for a single manipulator. Second, we propose the Lazy Edged-based A* (LEA*) for efficient SAPF. Since edge evaluation is the computational bottleneck of manipulator planning, LEA* uses lazy search and an edge queue to reduce the number of edge evaluations. We show that LEA* is optimally vertex efficient and has improved edge efficiency compared to A*. We apply the proposed CBLS to multi-manipulator planning problems and show its superior performance by comparing it with CBS and a sampling-based algorithm, namely, RRT-Connect. 

---
# SOV-CAD: Stepwise Orthographic Views Guided CAD Modeling Sequence Reconstruction 

**Authors**: Zhaopeng Feng, Chen Zhi, Xuhong Zhang, Zhengwen Feng, Xinkui Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2607.04119)  

**Abstract**: Reconstructing Computer-Aided Design (CAD) modeling sequences from images is crucial for preserving design intent and supporting parametric editing. However, existing methods typically generate full CAD sequences holistically, overlooking the iterative, feedback-driven nature of human design workflows. We address this limitation by introducing the rich stepwise visual supervision: at each modeling step, the system observes the target's orthographic projections, the projections of the incrementally constructed model, and the active sketch, enabling informed action selection. To effectively leverage this on-the-fly feedback, we propose SOV-CAD, a framework that formulates CAD reconstruction as a sequential decision-making task and employs offline reinforcement learning with a Decision Transformer architecture. This design incorporates continuous visual feedback guided by geometric alignment rewards, resulting in a more accurate and human-like modeling process. Extensive experiments show that SOV-CAD surpasses state-of-the-art methods in CAD sequence reconstruction while exhibiting strong data efficiency. Code of SOV-CAD is available at: this https URL 

---
# Parametric Memory Decoding for Zero-Shot Routing in LoRA-Based External Parametric Memory 

**Authors**: Fengxian Ji, Zhuohan Xie, Jingpu Yang, Fan Zhang, Zirui Song, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.04118)  

**Abstract**: With the rise of parametric memory, LoRA-based External Parametric Memory (EPM) has emerged as a modular solution, but existing routing methods often introduce additional training, deployment, and maintenance overhead. This raises a natural question: can a LoRA-based EPM bank be routed without maintaining an additional routing component? However, existing zero-shot LoRA routing methods still face two problems under the EPM setting: (1) their evaluations are scattered across different task settings rather than organized around EPM access, and (2) their routing signals lack a unified perspective to guide systematic improvement. To address these problems, we organize PMD-Bench, covering document-level, domain-level knowledge, and task-skill, and propose Parametric Memory Decoding (PMD), the first framework designed to systematically improve zero-shot LoRA routing by reframing it as decoding activations over external parametric memory. Based on PMD, we further instantiate PMDRouter, which scores each LoRA by its response magnitude from a single base-model prefill. Experiments on PMD-Bench show that PMDRouter achieves the strongest internal-signal performance across multiple zero-shot routing settings. These results demonstrate the feasibility of zero-shot LoRA routing and suggest that PMD can serve as a general framework for improving zero-shot routing methods. Sources: Github (this https URL) 

---
# DynaVieW: Schema-Guided World Modeling for Understanding Hierarchical Visual Dynamics 

**Authors**: Silin Gao, Hao Zhao, Zeming Chen, Sepideh Mamooler, Antara Raaghavi Bhattacharya, Qiyu Wu, Hiromi Wakaki, Yuki Mitsufuji, Li Mi, Syrielle Montariol, Antoine Bosselut  

**Link**: [PDF](https://arxiv.org/pdf/2607.04112)  

**Abstract**: Multimodal LLMs struggle to systematically model the temporal evolution of visual scenes in videos or multi-image sequences. Such inputs require models to predict or simulate multiple levels of dynamic constituents, such as actions taken in the visual sequence, and the associated changes to the visual environment that result. To address this challenge, we propose a dynamic schema-guided world model, DynaVieW, optimized for visual dynamic prediction and simulation. DynaVieW achieves an in-depth understanding of visual dynamics by learning interleaved state-transition sequences, where states cover broad visual scenes from video keyframes, and transitions capture comprehensive dynamic constituents within a hierarchical schema. DynaVieW jointly models transition prediction and state simulation under a mixture-of-experts architecture, with a cross-expert selective attention and a schema token re-weighted loss, to ensure effective and robust learning. DynaVieW's understanding of visual dynamics boosts its downstream performance in visual narrative creation and world simulation, showing improved consistency, controllability, and instruction-following. 

---
# Submitted and Diagnostic Analysis of Full-Text Temporal Retrieval for LongEval-Sci 

**Authors**: Yingdong Yang, Haijian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2607.04088)  

**Abstract**: LongEval-Sci evaluates scientific retrieval under collection change, where a system should be effective on the current corpus and remain usable as documents accumulate over time. This paper reports both official Task 1 results and development diagnostics for LongEval-Sci 2026. We compare the official PyTerrier BM25 and Qwen3 dense baselines with full-text BM25, additive and router variants, temporal full-text retrieval, temporal+citation retrieval, RM3 query expansion, cross-encoder reranking, and reciprocal rank fusion (RRF). In the official DCTR evaluation, the temporalized full-text runs are our strongest submissions: FT BM25+temporal and FT BM25+temporal+citation obtain the best ARP on all three snapshots (0.285, 0.267, and 0.180 nDCG@10) and reduce snapshot-3 relative change from 0.481 for the BM25 pivot to 0.368. Citation features match the temporal-only variant but do not provide a measurable additional gain in the official summary. Our internal snapshot-1 diagnostics show a complementary pattern: full-text BM25 is the strongest single development retriever (DCTR nDCG@10 = 0.3302, MAP = 0.2853), RRF gives the best deep recall (Recall@1000 = 0.9667), and some uncalibrated overlays can sharply degrade top-rank quality. We therefore conclude that full-text retrieval is the strongest foundation, temporal integration can improve official longitudinal effectiveness when applied to that foundation, and citation evidence still requires cleaner ablation and calibration. Beyond ranking, we also report a qualitative weekly IR-system update-monitoring analysis based on ingestion velocity and stale-coverage drift. 

---
# FedSPM: Routing-Enabled Federated Learning under Dual Heterogeneity via Semiparametric Mixture 

**Authors**: Zijian Wang, Pengfei Li, Guangyu Yang, Qiong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.04085)  

**Abstract**: Routing-prediction federated learning has emerged as a new paradigm that reframes inter-client heterogeneity as a resource for system-level intelligence: at inference time, the server routes each external query to the best-matched client for prediction. Existing approaches, however, typically treat each client as internally homogeneous, overlooking latent subpopulations within local data. For example, patients with the same diagnosis at one hospital may exhibit morphologically distinct disease subtypes. The coexistence of inter-client and intra-client heterogeneity, which we call dual heterogeneity, can impair both routing and prediction. To address this challenge, we propose FedSPM, a routing-enabled semiparametric mixture framework that represents each client using client-specific latent components. Each component combines a predictive distribution for classification with a feature distribution for routing. To flexibly model feature distributions while effectively sharing information across clients, FedSPM models their density ratios relative to a common nonparametric measure estimated via empirical likelihood. We develop a federated expectation-maximization algorithm that optimizes a tractable surrogate and prove convergence of the exact profiled objective at the standard $\mathcal{O}(1/\sqrt{T})$ rate when the surrogate errors are properly controlled. Experiments on controlled benchmarks and real-world medical data demonstrate consistent improvements in routing and prediction under dual heterogeneity. Code is available at this https URL. 

---
# Seeing Once is Enough? Online Geometry-Aware Token Pruning for 3D Question Answering 

**Authors**: Ruei-Chi Lai, Bolivar Solarte, Chin-Hsuan Wu, Yi-Hsuan Tsai, Min Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.04079)  

**Abstract**: Recent Multi-modal Large Language Models (MLLMs) have demonstrated remarkable performance on 2D question answering tasks. However, extending these models to the 3D question answering remains challenging, as they typically require multiple views of the scene, which incurs substantial computational cost at inference. To mitigate this issue, existing solutions rely on strategic frame selection or token-merging algorithms that require preprocessing in advance all frames of the scene, i.e., an offline fashion. In contrast, we propose the first online token-pruning method that can be integrated seamlessly with current MLLM models for 3D question answering tasks, without additional training and with lower memory this http URL key insight is to project each input frame into a shared voxel space using depth information and camera pose, identifying spatially-overlapped regions across frames and selectively pruning redundant image tokens before they enter the language model. Our method enables efficient online processing while reducing up to 50% of token usage. We apply this approach to Qwen2.5-VL-7B and Qwen3-VL-8B, demonstrating improved performance on the ScanQA, SQA3D, and OpenEQA-HM3D benchmarks. 

---
# Benchmarking API Drift in LLM-Generated Quantum Code Across Successive SDK Versions 

**Authors**: Mohammad Arif Rasyidi, Syahirul Faiz  

**Link**: [PDF](https://arxiv.org/pdf/2607.04072)  

**Abstract**: Large language models can generate plausible quantum code, but it is unclear whether they can reliably target the specific software development kit (SDK) version requested by the user. We study this problem as API drift and introduce quantum-api-drift, a benchmark for measuring version fidelity, defined here as execution success on the requested SDK version, cross-version compatibility, failure modes, and documentation-guided repair in LLM-generated quantum SDK code. We instantiate the benchmark with Qiskit, a representative quantum SDK that underwent substantial interface changes across v0.43, v1.3, and v2.0. We evaluate 17 models on 50 tasks with 3 samples per prompt, yielding 450 generated samples and 1,350 executions per model. Sixteen models are tested in a matched REST API setting with a 1024-token output cap, while GPT-5.4 (Codex CLI) is reported separately as a reference configuration. Across the 16 matched REST models, diagonal Pass@1 ranges from 0.02 to 0.85. Claude Opus 4.7 is strongest on v0.43 and v2.0, while Grok 4.20 is strongest on v1.3 at 0.513. Error profiles differ systematically by model strength: weaker models fail mainly with broken imports, while stronger models more often reach deprecation-level failures. Documentation-guided repair succeeds for 0.19 to 0.59 of repair attempts overall and is consistently much more effective for migration to v2.0 than to v1.3. The benchmark artifacts are publicly available at this https URL. These results show that version alignment is a distinct evaluation axis for quantum code generation and that API drift remains only partly recoverable even with migration guidance. 

---
# Beyond Multilingual Averages: MTEB-PT, a Benchmark for Portuguese Sentence Encoders 

**Authors**: Lucas Hideki Takeuchi Okamura, Alexandre Alcoforado, Anna Helena Reali Costa  

**Link**: [PDF](https://arxiv.org/pdf/2607.04071)  

**Abstract**: Portuguese remains underrepresented in text embedding evaluation, despite being one of the most widely spoken languages in the world. As a result, embedding models are often selected based on English or multilingual metrics, while their effectiveness in Portuguese remains unclear. We present MTEB-PT, a Portuguese benchmark constructed from a subset of MMTEB, comprising 14 existing datasets across Semantic Textual Similarity (STS), classification, retrieval, and reranking. We use this benchmark to evaluate 17 open- and closed-source embedding models under a unified protocol. Our results show that Portuguese performance is strongly task-dependent: multilingual rankings do not reliably predict Portuguese-specific performance across task families, no single model dominates all settings, and models with stronger long-context capacity are particularly advantageous on longer-input tasks such as retrieval and reranking. The benchmark also shows that language-specific fine-tuning still improves model performance in Portuguese, especially on task types that match the adaptation data most closely. To examine this effect, we fine-tune three representative backbone models with Portuguese contrastive supervision and Matryoshka Representation Learning (MRL). These benchmark-informed baselines yield their strongest gains on STS, consistent with the predominantly symmetric supervision used during training, while also improving retrieval and remaining competitive under dimensional truncation. We release the MTEB-PT benchmark, the fine-tuned models, and the training and evaluation code. 

---
# Enhancing Implicit Neural Representations with Image Feature Embedding for Unsupervised Cardiac Cine MRI Reconstruction 

**Authors**: Donghang Lyu, Marius Staring, Yiming Dong, Keupp Jochen, Hildo J. Lamb, Mariya Doneva  

**Link**: [PDF](https://arxiv.org/pdf/2607.04069)  

**Abstract**: Cardiac cine Magnetic Resonance Imaging (MRI) is a critical diagnostic tool that provides dynamic insights for radiologists. To accelerate acquisition, under-sampled k-space data is often used, requiring reconstruction methods that combine coil sensitivity encoding with prior information to recover missing data. Deep learning approaches have gained more attention for leveraging data-adaptive priors. While supervised learning approaches are a common choice, they depend on fully sampled reference data, which is not always available. Unsupervised methods eliminate the need for fully sampled reference data, which can be advantageous in cardiac cine MRI reconstruction. Among them, implicit neural representations (INRs) have shown great potential due to their simple architecture and good quality reconstructions. In this work, we propose an image-domain dual-branch INR framework, termed I-FP-INR, which extends the original INR design by introducing an additional feature-processing branch. This design aims to extract complementary feature embeddings to enhance the overall representation, thereby benefiting reconstruction. Extensive evaluations on both public datasets and in-house data show consistent improvements over baseline methods in reconstruction quality, with strong robustness across varied scenarios. 

---
# Speaker-Disentangled Chunk-Wise Regression for Syllabic Tokenization 

**Authors**: Ryota Komatsu, Kota Kawakita, Takuma Okamoto, Takahiro Shinozaki  

**Link**: [PDF](https://arxiv.org/pdf/2607.04064)  

**Abstract**: Unsupervised syllabic tokenization aims to learn discrete syllabic tokens that capture latent linguistic content-related structure from raw speech. Recent syllabic tokenization methods employ teacher-student distillation of the pretrained HuBERT to organize latent speech frame representations into syllabic segments. However, when trained with an utterance-level cross-entropy objective, the model predicts speaker identity rather than linguistic content, thereby compromising the purity of syllabic tokens. To address this problem, we propose a speaker-disentangled syllabic tokenizer that regresses speaker-perturbed student representations toward clean teacher targets within fixed-length chunks. Experimental results demonstrate that our proposed method achieves state-of-the-art performance in syllable boundary detection and syllabic segment clustering. Moreover, a speech language model trained on our syllabic tokens achieves a 7% relative improvement in syntactic and semantic understanding over the phone-level SpiRit-LM. 

---
# Telescope: Improving Zero Shot Detection of LLM Generated Content By Measuring Token Repetition Probability 

**Authors**: Christopher Nassif, Josh F. Cooper  

**Link**: [PDF](https://arxiv.org/pdf/2607.04061)  

**Abstract**: Distinguishing Large Language Model (LLM) generated text from human writing is a critical and difficult challenge. While LLMs are trained to write like humans, we hypothesize that this training leaves an indelible mark. LLMs develop a particularly strong aversion to token repetition very early in training. This bias persists as a ''Vestigial Heuristic'' (a developmental artifact) that is activated in LLM-generated text, separating LLM from human writing. To probe this phenomenon, we introduce Telescope Perplexity, a metric that evaluates the token repetition of the model, $P(s_i | s_{1:i})$ . Our empirical investigation reveals that the Telescope Perplexity signature emerges early in pre-training, and Telescope Perplexity empirically enables highly effective zero-shot LLM detection. We show state-of-the-art or competitive performance across diverse datasets (including modern evaluation sets we introduce), reference models, and perturbation schemes with greater efficiency than other methods. 

---
# Reward-Gated On-Policy Distillation 

**Authors**: Mohammad Sadegh Akhondzadeh, Vijay Lingam, Atula Tejaswi, Chanakya Ekbote, Sujay Sanghavi, Aleksandar Bojchevski  

**Link**: [PDF](https://arxiv.org/pdf/2607.04037)  

**Abstract**: On-policy distillation is a powerful way to transfer reasoning ability from a strong teacher to a smaller student: the student samples trajectories from its own policy, and the teacher provides dense token-level supervision on the states the student actually visits. However, this supervision is not always reliable: a teacher can assign high likelihood to plausible but incorrect solutions, or low likelihood to correct student solutions that follow different reasoning paths. Unconditionally distilling the teacher can therefore reinforce bad modes or erase useful student behavior. To address these limitations, we introduce RG-OPD: Reward-Gated On-Policy Distillation that uses verifier feedback to decide when teacher logits should be trusted. RG-OPD bridges sparse verifier rewards and dense teacher logits, preserving token-level supervision while filtering misleading teacher signals. Across reasoning and coding benchmarks, RG-OPD produces stronger distilled students, outperforming both vanilla reverse-KL distillation and the recent TSD-KD baseline. At 1K generation length, RG-OPD improves over reverse-KL by 2.9 points and over TSD-KD by 4.9 points; in the long-generation setting, it improves over the untuned student by 8.2 points. Our code is available at this https URL. 

---
# The "I Don't Know" Filter: Enhancing Agentic Reliability in Function Calling 

**Authors**: Stefan Broecker, Mason del Rosario, Boris Selitser, Thomas Strohmer  

**Link**: [PDF](https://arxiv.org/pdf/2607.04034)  

**Abstract**: The language models that underpin agents have seen a rapid rise in performance on function calling benchmarks. However, the metrics used in the training and evaluation of these models often encourage models to make positive claims even when the answer is uncertain, leading to hallucinations. Such hallucinations can be disastrous when language models are trusted to use function calls to make decisions in high stakes applications. To that end, we propose an agent evaluation metric that takes into account the negative outcomes associated with incorrect function calls. Further, to catch hallucinations before they can cause harm, we propose a lightweight trainable filter that can quantify a language model's uncertainty and remove potentially harmful function calls. By training that filter to detect and suppress uncertain function calls without modifying the underlying model, we demonstrate a practical path toward agents that know when to say "I don't know," a property we argue is essential to production reliability. 

---
# OmniOpt: Taxonomy, Geometry, and Benchmarking of Modern Optimizers 

**Authors**: Siyuan Li, Jiabao Pan, Yumou Liu, Zhuoli Ouyang, Xin Jin, Xinglong Xu, Jingxuan Wei, Shengye Pang, Jintao Che, Xuanhe Zhou, Conghui He, Cheng Tan  

**Link**: [PDF](https://arxiv.org/pdf/2607.04033)  

**Abstract**: Optimizer selection for large-scale model training has become a system-level design decision constrained jointly by compute, memory, tuning budget, and task diversity, yet the landscape of over one hundred methods remains fragmented. We therefore present OmniOpt, a unified survey and benchmark cookbook of optimizers for the research community. OmniOpt rests on four coupled components. First, we treat every optimizer update as a structured transformation through a five-stage meta-pipeline, and show that most methods engage only one or two of these stages. Second, we use norm-constrained linear minimization oracles (LMOs) to unify different optimizers. Third, these two views ground a dual-dimension taxonomy, one dimension assigning each method to a mechanism family and the other recording the measurable training objectives it aims to improve. Fourth, and at the core of this paper, we instantiate the full taxonomy in a unified cross-domain benchmark spanning representative optimizers, model scales, and training regimes from language model pretraining to image classification, systematically analyzing each method family across multiple effect objectives and laying out their trade-offs. OmniOpt thus supplies the research community with an operational coordinate system for selecting optimizers under explicit mechanism and objective assumptions, and charts a direction for the future development of the optimizer community. 

---
# Efficient Discovery of Conditional Dependencies with Desbordante 

**Authors**: Ivan Kozhukov, Dmitry Fedoseev, Maksim Emelyanov, Artem Smola, Pyotr Senichenkov, Pavel Anosov, George Chernishev  

**Link**: [PDF](https://arxiv.org/pdf/2607.04030)  

**Abstract**: Conditional functional dependencies (CFDs) are functional dependencies with a restricted scope: they specify the context in which a dependency holds and are useful for data-quality tasks, specifying complex integrity constraints, and extracting valuable insights from data.
We study the CFD discovery problem, which is computationally demanding. We build on the state-of-the-art CFDFinder algorithm and introduce a set of algorithmic and engineering improvements, including a parallelization strategy, to produce ParCFDFinder. Our implementation is integrated into Desbordante - a high-performance open-source data profiler written in C++ that exposes a Python interface, enabling CFD discovery to be invoked from any Python program.
Experimental results show that our enhancements speed up the algorithm by up to $318\times$ ($118\times$ on average) and reduce memory usage by up to $23\times$ ($14\times$ on average) compared with the existing Java-based implementation of Metanome. Integrating ParCFDFinder into Desbordante makes it possible, for the first time, to conveniently discover CFDs on datasets with hundreds of thousands of rows on a commodity machine within a reasonable time. 

---
# A Unified Algebraic Framework for Classification Performance Evaluation 

**Authors**: Ronaldo C. Prati  

**Link**: [PDF](https://arxiv.org/pdf/2607.04028)  

**Abstract**: We propose a unified algebraic framework for classification performance evaluation that encompasses binary, multiclass, multilabel, ordinal, hierarchical, cost-sensitive, and soft-label settings within a single formalism. The foundation is a representation of actual and predicted labels as binary indicator matrices, combined with three aggregation operators -- global, column-wise, and row-wise -- that correspond exactly to micro, macro/weighted, and exemplar averaging. Any binary performance measure expressed in terms of true/positive/negative counts extends automatically to all settings by substituting these operators, generating multiclass and multilabel versions without measure-specific derivations. The framework further accommodates soft classifier outputs via argmax or thresholding, soft ground truth via triangular norms, ordinal classification via membership functions or cumulative encodings, and cost-sensitive evaluation via a cost matrix that subsumes MAE and MSE as special cases. We establish several theoretical results: micro-averaging equals denominator-weighted macro-averaging; the product $t$-norm is the unique one preserving the confusion-matrix partition; skew-invariant measures are characterised as functions of recall and specificity; and micro-precision, micro-recall, and micro-$F_1$ are all equal to accuracy in multiclass settings. Empirical illustrations on synthetic and real data confirm the theoretical findings. 

---
# Finite Reliability Representations: Noise-Calibrated Belief-Space Covers for Reliable Decision-Making 

**Authors**: Hyung-Jin Yoon, Hunmin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.04019)  

**Abstract**: Physical sensing and actuation noise floors should inform how much belief resolution a decision-making system can reliably use. We introduce Finite Reliability Representations (FRR), a framework for covering belief spaces by reliability cells: regions within which the optimal action-value function Q*(b,u) varies by at most a tolerance epsilon, uniformly over actions. The framework is formulated on beliefs rather than states and uses a cover rather than an equivalence quotient, because approximate decision-closeness is not transitive in general. A central technical point is that noisy Bayesian updates should not be treated as globally contractive on arbitrary beliefs. We therefore separate three objects: the fixed-observation filter map, the predictive observation law, and the controlled belief-transition kernel. For nonlinear continuous-state systems, FRR is obtained under a reachable-set Lipschitz modulus for the belief-transition kernel. For finite-state POMDPs, the same construction becomes exact on the belief simplex: prediction is linear, Bayesian correction is a normalized positive linear map, sensor noise enters through observation-distribution distinguishability, and actuation uncertainty enters through an action-execution channel. Under the corresponding action-value Lipschitz condition, an FRR cover supports a cell-constant policy whose suboptimality is bounded by 2 epsilon/(1 - gamma). We also introduce reliability entropy, the logarithm of the minimal number of reliability cells, as a measure of certified decision-relevant belief complexity. The framework distinguishes representation sufficiency from fundamental performance floors imposed by sensing, process, and actuation noise. It applies to finite POMDPs, linear-Gaussian filters, locally linearized nonlinear filters, and particle-filter implementations through analytic or empirical certification of reliability cells. 

---
# When Does Small Data Work? Accuracy and Efficiency Trade-offs Between Tabular Foundation Models and Conventional Methods for Crowd-State Classification at Hajj and Umrah 

**Authors**: AlJawharh S. AlOtaibi, Mohamed Eltahir, Jude AlSubaie  

**Link**: [PDF](https://arxiv.org/pdf/2607.04013)  

**Abstract**: Learning from few labeled examples is a central challenge in tabular machine learning, and it becomes the binding constraint in domains where labeling is costly, such as crowd monitoring during Hajj and Umrah. Tabular foundation models, which predict from only a handful of examples without task-specific training, were recently introduced to address this very-few-label regime. In this study we test them on crowd-state classification to assess how much they help when labels are scarce, and we compare them against standard machine learning methods to characterize the accuracy and efficiency trade-offs between the two approaches. Using three real datasets we evaluate different machine learning models, in untuned and tuned forms, against three foundation models. Results show that no single family is best everywhere. The right choice depends on the label budget. When labels are very few, foundation models lead. As labels grow, tuned conventional models catch up and significantly surpass the foundation models on the more structural geometry target. Efficiency separates them further where tuned machine learning models incur a large tuning cost that foundation models avoid, although foundation models reprocess their context at every prediction. We summarize these results as a practical map of which approach to prefer under a given label budget and computational budget. 

---
# Separating Representation from Reconstruction Enables Scalable Text Encoders 

**Authors**: Megi Dervishi, Mathurin Videau, Yann LeCun  

**Link**: [PDF](https://arxiv.org/pdf/2607.04011)  

**Abstract**: While decoders have rapidly scaled, encoders have remained largely unchanged since BERT. We revisit this disparity by frozen backbone evaluation via probing. Under this lens, the representations of BERT encoders become increasingly $\textit{unexploitable}$ by frozen probes, despite improved perplexity. The misalignment originates in BERT's flat design, which couples representation learning to the token reconstruction loss. We propose $\textbf{CrossBERT}$, a two-part architecture that separates the learning of high-quality encoded representations from the rigid grounding of token reconstruction. This design further enables high masking ratios ($\ge 50\%$) and gradient collection over all tokens via a $\textit{Complementary Masking Strategy}$, respectively increasing throughput by $1.5$ to $2\times$ and sample efficiency by $2\times$. Overall, CrossBERT demonstrates monotonic scaling and superior performance on MTEB(eng, v2) and frozen GLUE benchmarks. 

---
# Full Glyph Images Beat Token Embeddings: A Controlled Study for Transformers 

**Authors**: Shuyang Xiang, Hao Guan  

**Link**: [PDF](https://arxiv.org/pdf/2607.03994)  

**Abstract**: Modern language models generally represent text as sequences of discrete token embeddings, an assumption deeply rooted in current practice but rarely questioned. We challenge this representation, especially for Chinese, by replacing index-based token embeddings entirely with a single rasterized image of the character sequence, processed by a vision encoder composed of a shared ResNet and a shallow Vision Transformer. To isolate the role of input representation, we construct a dual-branch controlled framework in which both a Vision-based model and an index-based baseline share an identical decoder backbone, training objective, optimizer, and data curriculum. Any performance difference is therefore attributable to the input modality only. Across all tested decoder backbones, the Vision-based model consistently outperforms the baseline, reaching a peak accuracy of 0.429 versus 0.355 for the index-based baseline,that is, a 21% relative improvement, while converging in about half the number of training epochs. The advantage emerges especially within the first five epochs (under 21% of total data) and persists under moderate character corruption: the corrupted Vision model matches the clean index-based baseline. Ablation studies reveal that the advantage requires both spatially coherent input and a ViT encoder with 2D positional encodings. A cross-script comparison on English shows the advantage does not transfer directly to alphabetic writing systems, suggesting that the uniform visual density and radical structure of Chinese characters are enabling conditions. These findings suggest that transformers are more modality-agnostic than commonly assumed, and that discrete tokenization is not a fundamental requirement for Chinese language modeling. 

---
# NouveauVoice: Generating Novel Pseudo Speakers for Voice Anonymization 

**Authors**: Meiying Melissa Chen, Anastasia Kuznetsova, Zhenyu Wang, Zhiyao Duan  

**Link**: [PDF](https://arxiv.org/pdf/2607.03985)  

**Abstract**: Advanced neural technologies in speech synthesis and voice conversion (VC) have introduced severe risks to personal privacy, necessitating robust Speaker Anonymization Systems (SAS). Existing SAS approaches modify voice characteristics in the hand-crafted feature space or speaker embedding space, often struggling to provide sufficient identity variance across generated voices. In this paper, we propose NouveauVoice, a novel pseudo-speaker generation framework based on a Hierarchical Deep Variational Autoencoder (NVAE). Integrated as a standalone plug-in module on top of state-of-the-art architectures (FACodec and CosyVoice2), our approach leverages tractable sampling and the Evidence Lower Bound (ELBO) objective to synthesize highly expressive pseudo-speaker embeddings with significantly enhanced speaker diversity. Evaluating our framework under a protocol similar to the VoicePrivacy Challenge alongside Maximum Mean Discrepancy (MMD) analysis, we demonstrate that NouveauVoice achieves strong identity concealment, yielding an Equal Error Rate (EER) exceeding 38% against an automatic speaker verification attacker model. Our system shows a reasonable trade-off between strict anonymity, rich pseudo-speaker diversity, and downstream speech utility, such as intelligibility and emotional expressiveness. 

---
# BanglaMemeEvidence: A Multimodal Benchmark Dataset for Explanatory Evidence Detection in Bengali Memes 

**Authors**: Fatema Tuj Johora Faria, Mukaffi Bin Moin, Md. Mahfuzur Rahman, Pronay Debnath, Asif Iftekher Fahim, Faisal Muhammad Shah  

**Link**: [PDF](https://arxiv.org/pdf/2607.03981)  

**Abstract**: Memes have become influential communication tools on social media, combining viral visuals with concise messaging to convey impactful ideas. While substantial research has examined the affective dimensions of memes, key challenges such as detecting harmful content, identifying cyberbullying, and performing accurate sentiment analysis remain critical, largely due to the need for deeper contextual understanding. In this paper, we introduce MemeEvidenceDetect, a hybrid task aimed at analyzing a meme and its contextual information to identify specific sentences that explain or elucidate its meaning and humor. To support this task, we present BanglaMemeEvidence, a curated dataset of 2,917 Bengali memes, emphasizing its significance as a resource for the Bangla language. Each meme is annotated with natural language explanations, including Meme OCR, Meme Context, and Evidence Sentences, alongside relevance scores that reflect the relationship between a meme and its corresponding annotations. To address the gap in dynamically inferring a meme's context, we propose BengaliMemeEvidenceNet, a hybrid multimodal framework that integrates textual and visual features for comprehensive meme representation. Our experiments demonstrate the effectiveness of BengaliMemeEvidenceNet, achieving an F1 score of 0.74. To the best of our knowledge, this is the first study to focus on evidence detection in Bengali memes, marking a notable step forward in the analysis of memes in low-resource languages. 

---
# Scalable Semantic Steering of Embedding Projections 

**Authors**: Wei Liu, Eric Krokos, Kirsten Whitley, Rebecca Faust, Chris North  

**Link**: [PDF](https://arxiv.org/pdf/2607.03978)  

**Abstract**: Low-dimensional projections support interactive visual analysis of high-dimensional data embeddings, but their structure often does not align with analyst-defined semantic relationships. Recent LLM-augmented semantic steering methods address this gap by externalizing analyst intent from user-defined groups of seed examples, but they propagate intent through per-item LLM reasoning, causing LLM calls and cost to grow linearly with collection size. We propose a scalable semantic steering method that shifts semantic computation from individual items to user-defined groups. A single LLM call generates structured profiles for all groups, which are embedded and combined with seed centroids to form hybrid semantic prototypes. The method then propagates intent without retraining, using embedding-space soft assignment, abstention, and alignment-scaled updates before reprojection. On a 5K-document LitCovid corpus, our method achieves global alignment comparable to per-item LLM steering while reducing LLM calls by over three orders of magnitude. An image case study shows that the same prototype-based mechanism extends to multimodal embeddings. These results suggest that group-level representations can make semantic steering more practical for larger embedding collections. 

---
# Order-based Causal Discovery for Multistage Processes 

**Authors**: Eun-Yeol Ma, Junsub Jung, Heeyoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.03971)  

**Abstract**: Causality has become an increasingly important tool for gaining a deeper understanding of complex systems. Among various causal analysis methods, causal discovery, which identifies causal relationships among variables from data, has been widely used to uncover underlying causality in diverse processes. However, while multistage processes are prevalent in many fields, existing causal discovery methods may produce counterintuitive results, given the known process knowledge, and may not be computationally efficient for handling large datasets typical of multistage processes. To address this gap, we propose a novel causal discovery method called Order-based Causal Discovery for Multistage Processes (OCDM). OCDM is designed to infer the causal structure of multistage data while preserving their inherent hierarchical and sequential structure by explicitly incorporating process knowledge into the causal discovery process. Specifically, we propose a structural knowledge-informed order-inferring algorithm that infers the causal order of variables by incorporating information about the stage from which each variable originates, based on an order-based causal discovery framework naturally suited for inherently ordered multistage data. Furthermore, to eliminate spurious edges from the initial causal graph generated based on the inferred causal order, we introduce a novel pruning technique using stochastic gated neural networks, which offers greater computational efficiency compared to existing methods. Through experiments on various datasets, we demonstrate that OCDM effectively infers the causal structure of multistage processes, outperforming existing methods. 

---
# Refused in Chat, Written in Code: Workflow-Level Jailbreak Construction in IDE Coding Agents 

**Authors**: Abhishek Kumar, Carsten Maple  

**Link**: [PDF](https://arxiv.org/pdf/2607.03968)  

**Abstract**: Large language models are increasingly deployed as IDE-integrated coding agents that decompose tasks, generate and edit files, run code, and refine outputs over many turns. Yet their safety is still often evaluated as if they were chatbots: one harmful prompt, one response, judged in isolation. We introduce workflow-level jailbreak construction, a failure mode in which a harmful objective is assembled across ordinary stages of a software-development workflow rather than generated through a single direct prompt. Using GitHub Copilot in Visual Studio Code, we study four closed-weight backends: Claude Sonnet 4.6, Claude Haiku 4.5, Gemini 3.1 Pro, and Gemini 3.5 Flash. Across 204 prompts from Hammurabi's Code, HarmBench, and AdvBench , the models show near-complete refusal under direct chat, CSV-read, and single-step code-fix baselines, with only 8/816 successful responses in each baseline condition. Under the full workflow, however, the same prompts and backends produce 816/816 unsafe teaching-shot completions, all independently confirmed by two expert evaluators under a strict rubric. These results show that conversational refusal benchmarks can substantially overstate the safety of deployed coding agents and motivate defenses that reason about safety across multi-turn IDE workflows and their generated artifacts, not only individual chat turns. 

---
# Worldscape-MoE: A Unified Mixture-of-Experts World Model for Scalable Heterogeneous Action Control 

**Authors**: Jianjie Fang, Yongyan Xu, Ziyou Wang, Chen Gao, Yuchao Huang, Zhaolu Wang, Rongze Tang, Mingyuan Jia, Baining Zhao, Weichen Zhang, Xin Zhang, Haisheng Su, Yu Shang, Wei Wu, Xinlei Chen, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.03964)  

**Abstract**: World models are rapidly becoming a core infrastructure for embodied intelligence and interactive agents: they provide controllable simulators in which agents can perceive, act, forecast, and acquire scalable experience. Yet current video generation world models are still organized around isolated control interfaces, such as camera trajectories, robot actions, or hand-joint signals. This fragmentation is increasingly a scaling bottleneck. The central challenge is not the absence of controllable generators, but the lack of a unified and extensible learning framework that can absorb heterogeneous action supervision while preserving a shared model of world dynamics. In this work, we introduce Worldscape-MoE, a Mixture-of-Experts world model built on Diffusion Transformers for scalable heterogeneous action control. Our key observation is that different controls specify different interfaces to the same underlying world: although their representations differ, they constrain shared physical regularities, scene dynamics, and interaction semantics. Worldscape-MoE operationalizes this observation through modality-aware control injection, shared and control-specific experts, and a progressive MoE tuning strategy that supports continual extension to new action modalities. Experiments across locomotion, robotic manipulation, and egocentric hand control show that heterogeneous supervision improves rather than interferes with individual control capabilities. Worldscape-MoE achieves strong results on WorldArena, improves locomotion and hand-control metrics, exhibits robust out-of-distribution generalization, and demonstrates scaling behavior as additional control data and experts are integrated. 

---
# NormWorlds-CF: Solver-Verified Counterfactual Normative Reasoning with Metamorphic-Relation GRPO 

**Authors**: Xinqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.03957)  

**Abstract**: Language models can reach the right normative verdict for the wrong reason. We introduce NormWorlds-CF, a solver-verified environment for counterfactual normative reasoning in executable rule worlds. Its deterministic solver produces final answers, proof and falsification certificates, argument statuses, support sets, and paired-world change labels, enabling supervision and evaluation without LLM judges. The benchmark contains staged SFT diagnostics and a compact paired-world task with 270 root families and 1080 canonical-to-variant pairs. The SFT diagnostics show that final-answer supervision is an unsafe proxy: answer-only SFT reaches perfect accuracy on answer tasks but scores zero on falsification, while proof-plus-falsification training with targeted replay reaches strong all-task accuracy. For the structured-change task, we introduce metamorphic-relation GRPO (MR-GRPO), a class-conditioned reward for GRPO that gives partial credit for relation families and solver-visible change fields. In matched 1.7B continuation experiments, MR-GRPO improves held-out relation accuracy and relation-family correctness, and reduces wrong-family error, compared to sparse and answer-only GRPO. In Qwen3-4B three-seed validation, answer-only reward improves answer-change fields but weakens relation-family structure, sparse reward preserves coarse relation labels best, and MR-GRPO delivers the strongest balanced performance across answer-change, support-change, status-change, and soft root-level metamorphic-relation metrics. These results show that verified counterfactual structure can shape post-training beyond final answers, while exact full change-record generation, invariant subtype recognition, and out-of-distribution (OOD) transfer remain open problems. 

---
# The Remarkable Effectiveness of Providing AI Agents with Natural Language Tools: A Replication Study Validating NLT Performance Across 14 Models 

**Authors**: Alexander Somma, Isabelle Plante, Fred Premji  

**Link**: [PDF](https://arxiv.org/pdf/2607.03953)  

**Abstract**: This study independently replicates and extends the Natural Language Tools (NLT) framework of Johnson et al.~(2025), which questions the use of structured tool calling in large language model (LLM) agentic systems. We evaluated NLT across 14 models and 8,560 trials, adding newer frontier, reasoning, and open-weight models to the original set. The results confirm the core findings and add detail. NLT improves tool-calling accuracy by 14.9 percentage points overall (62.3\% versus 47.4\% structured) and reduces critical errors by 93\% (51 versus 755 errors). The gains depend on model capability: models without native tool calling, reasoning models, and smaller models gain substantially (+24.0pp to +43.1pp), while heavily optimized frontier models (GPT-5, Gemini 2.5 Pro) show smaller or reversed advantages. This matches recent analyses of reinforcement-learning-optimized tool use (Martinez, 2025). NLT also cuts token usage by 25.2\%. The reliability and efficiency advantages compound in recursive agentic workflows, where agents chain many tool calls across sub-agents: a structured failure triggers retries, fallback routing, and coordination overhead, while NLT avoids most of that cost at the source. This work makes three contributions: (1) the first independent validation of NLT using open-source tooling, (2) evidence that model capability moderates NLT's advantages (Chen et al., 2025; Zhang et al., 2025), and (3) a measurement of NLT's reliability benefit (93\% fewer errors), its most deployment-relevant property given the known fragility of structured tool calling. NLT is a practical alternative to structured tool calling, especially for production systems that value reliability over parseability. 

---
# Why3-py: A Tool for Formal Verification of Hypothesis Testing and Meta-Analysis in Python 

**Authors**: Akira Tanaka, Yusuke Kawamoto  

**Link**: [PDF](https://arxiv.org/pdf/2607.03951)  

**Abstract**: The reproducibility crisis in scientific research has received widespread recognition, thereby increasing the importance of meta-analyses that integrate statistical analyses from multiple studies. However, statistical methods often have ambiguous and implicit underlying assumptions, which can lead to their erroneous applications and interpretations. To address this issue, we propose a formal verification framework for statistical programs written in Python. Specifically, we present Why3-py, a Python front-end for the Why3 verification platform that transforms Python programs into verification-oriented WhyML representations suitable for formal verification, addressing the challenges arising from Python's dynamic typing and runtime polymorphism. Furthermore, we extend the StatWhy tool to support the verification of meta-analysis methods. These tools enable users to identify overlooked assumptions and misuse of analyses, and to verify the correctness of Python programs for hypothesis testing and for meta-analyses. 

---
# MPSelectTune: Prompt-type Selection for Fine-tuning improves Concept Unlearning in LLMs 

**Authors**: Shubhadip Nag, Srinjoy Das, Agniva Saha, Anushree Ghosh, Soumi Das, Tarun Kumar, Suparna Bhattacharya, Sourangshu Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2607.03932)  

**Abstract**: LLMs can be conveniently adapted to a diverse set of tasks, e.g, prediction, question-answering tasks, etc, using appropriate prompts with few-shot examples. Biased or harmful concepts, e.g. gender or bio-weapons, present in pre-trained LLMs can lead to unsafe or unethical responses for many such prompts. Removing such undesirable concepts robustly across different prompt types remains a challenging problem, since existing unlearning methods typically ignore the impact of prompt variation. In this paper, we explore a novel adversarial approach to use a joint prompt for the main task and concept task prediction. We show that fine-tuning using the ``worst prompt type'' for concept prediction (with the highest concept accuracy) improves the average unlearning performance over a fine-tuning method that uses a combination of all prompt types. Our proposed method, MPSelectTune, is a two-stage approach that minimizes the concept accuracy of the highest accuracy-prompt type, after fine-tuning using a novel multi-task loss using multiple prompt types. Experimental results on four benchmarks show $2 - 15\%$ main task accuracy improvements over recent baselines and while reducing the worst-case concept accuracy by up to $17\%$ compared to recent baselines. 

---
# Probe, Don't Prompt: A Hidden-State Probe for Metadata Filtering in Multi-Meta-RAG 

**Authors**: Mykhailo Poliakov, Nadiya Shvai  

**Link**: [PDF](https://arxiv.org/pdf/2607.03929)  

**Abstract**: Multi-Meta-RAG improves retrieval for multi-hop question answering by filtering a vector store on metadata (the news source) that it extracts from each query by prompting gpt-3.5-turbo. We show this proprietary, free-form extractor can be replaced by a local, deterministic probe trained on the hidden states of a small open-source language model. On all 2556 MultiHop-RAG queries the probe reaches 90.9% set-exact accuracy against 88.0% for a model-free substring baseline and 80.9% for GPT-3.5, a margin that comes entirely from null queries, on which GPT-3.5 never abstains; on non-null queries all three stay within about a point. Because the probe's output space is exactly the fixed 49-source vocabulary, it cannot drift outside the allow-list as the prompted model does. Three design choices make it work: selecting a shallow layer, mean pooling, and class-imbalance-aware multi-label training over the long tail of sources. A 135M-parameter model lands within ~1.5 points of a 1.5B one, so the filter is cheap to output: a partial forward pass through the first few layers plus one linear head, with no API. The code is available at this https URL. 

---
# TokAN: Accent Normalization Using Self-Supervised Speech Tokens 

**Authors**: Qibing Bai, Shuai Wang, Yuhan Du, Bohan Li, Yannan Wang, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.03928)  

**Abstract**: Accent normalization (AN) seeks to convert non-native (L2) accented speech into standard (L1) speech while preserving speaker identity. The current techniques either require naturally recorded parallel L1-L2 speech for training, or suffer from quality degradation when supervised by synthesized targets. In this paper, we present TokAN, a token-based accent normalization framework that operates on self-supervised discrete speech tokens extracted from a L1-L2 jointly trained vector-quantization (VQ) tokenizer, without the need of synthetic supervisory speech. An autoregressive encoder-decoder model performs token-to-token conversion, translating L2-accented token sequences into the tokens of standard voice. We also introduce reinforcement learning (RL) post-training based on Group Relative Policy Optimization (GRPO), using word error rate and accent classifier confidence as complementary rewards. A non-autoregressive flow-matching synthesizer recovers the Mel-spectrogram from the converted tokens, conditioned on the source speaker embedding. We also develop a flow-matching duration predictor that supports total-duration-aware synthesis, making TokAN applicable to duration-critical tasks such as voice dubbing and live casting. Experiments on seven English accents demonstrate that TokAN reduced the word error rate from 12.40% to 9.89% after supervised fine-tuning, and further to 9.23% after RL post-training, consistently outperforming frame-to-frame, direct flow-matching, and prompt-based token-conversion baselines in terms of accent reduction and intelligibility. 

---
# TabQueryBench: A Query-Centric Benchmark for Synthetic Tabular Data 

**Authors**: Jialin Zhang, Fenghao Dong, Yajie Zhou, Vyas Sekar, Shinan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.03926)  

**Abstract**: Synthetic tabular data support use cases like data sharing, model development under access restrictions, and rapid prototyping of analytical workflows. Modern generative models are evaluated by their statistical similarity, correlation structure, privacy, and downstream machine-learning utility. However, such evaluations leave a gap: they rarely test the structure that matters for analytical queries. We present TabQueryBench, a query-centric benchmark that uses SQL-shaped analytical queries as structural assessors for synthetic data fidelity. It provides an extensible foundation for query-centric synthetic-data evaluation. From 12 public sources of analytical queries, TabQueryBench taxonomizes recurring cross-domain logic into 44 reusable query templates and grounds them to each dataset via a policy-guided template-to-SQL pipeline. This makes queries schema-aware while preserving comparability across generative models. Across 49 datasets and 11 generative models, it activates 10-12 templates per dataset, producing more than 100 executable SQL queries per dataset. Our systematic experiments show five main patterns. First, current tabular generative models can have good distance-based fidelity, but they still fall short on query-centric fidelity: RealTabFormer achieves the highest query-centric fidelity, but it only reaches 0.75 +/- 0.15 (REAL data score is 1.00). Second, tabular generative models struggle with very high-cardinality discrete support. Third, SOTA generative models preserve good global conditional query-centric fidelity, but fail more on local queries. Fourth, tail fidelity deteriorates as queries move toward the extreme tail; even the best model recovers only about 40.7% of real rare values. Finally, there is a fidelity-cost tradeoff in tabular generation: BayesNet offers the strongest tradeoff, with slightly lower query-centric fidelity but much lower generation cost. 

---
# Advanced Topic Modeling Techniques for Categorizing Software Vulnerabilities 

**Authors**: Utkarsh Tiwari, Spoorthi M, Anirudh S, Nidhin Prabhakar T. V  

**Link**: [PDF](https://arxiv.org/pdf/2607.03887)  

**Abstract**: The increasing complexity and frequency of software vulnerabilities demand efficient methods to analyze and prioritize threats. Traditional approaches often fail to process the vast amount of unstructured textual data effectively, highlighting the need for advanced solutions. This study leverages state-of-the-art topic modeling techniques powered by large language models (LLMs) to extract meaningful insights from the 'Threat' feature of a software vulnerability dataset. Models such as BERTopic, Top2Vec, CombinedTM, Llama2 with BERTopic, and Mixtral are utilized, along with dimensionality reduction and clustering methods like UMAP, PCA, HDBSCAN, and DBSCAN. By uncovering latent patterns and generating interpretable clusters, this research enhances threat prioritization and decision-making in cybersecurity. The findings support scalable and automated solutions for vulnerability management, contributing to improved security practices. 

---
# Enhancement of E-commerce Sponsored Search Relevancy with LLM 

**Authors**: Md Omar Faruk Rokon, Andrei Simion, Weizhi Du, Musen Wen, Hong Yao, Kuang-chih Lee  

**Link**: [PDF](https://arxiv.org/pdf/2607.03886)  

**Abstract**: Sponsored search plays a crucial role as a revenue stream for search engines, wherein advertisers competitively bid on keywords that align with the users' search queries. The task of matching relevant keywords to these queries is complicated by the vast and ever-evolving space of keywords, the ambiguity of user and advertiser intentions, and the wide range of topics and languages involved. Consequently, ensuring that ads are pertinent to user queries presents significant challenges. In the fast-paced world of e-commerce, the accuracy of sponsored search results is vital for boosting user satisfaction and optimizing business operations. This paper presents the development of an advanced Ad Relevance Model within a sponsored search framework, utilizing the power of a pretrained large language model. We detail a pioneering adaptation of the LLAMA2 7B model through Low-Rank Adaptation (LoRA), which markedly enhances search precision and operational efficiency, thus opening new avenues for improving user interactions in extensive online marketplaces such as this http URL. We introduce a novel query and ad title classifier, which discerns the relevance of search interactions across three categories: Relevant, Partially Relevant, and Irrelevant. Our approach involved adapting the pretrained model specifically for the e-commerce sponsored search context, training it on a large dataset. The fine-tuned model demonstrated a marked improvement in ad relevance accuracy, achieving 89.43% accuracy on a comprehensive test dataset -- outperforming both the baseline model and other advanced language models like GPT-4. The integration of LoRA with the based model represents a significant stride in customizing language models for e-commerce applications, resulting in enhanced search accuracy, cost efficiency, and operational privacy -- a triad essential for the modern digital marketplace. 

---
# Consistent but Miscalibrated: Evaluating LLM Limitations for Risk Communication in Natural Language 

**Authors**: Diego Cerda-Mardini, Sarath Chandar, Sreenath Madathil  

**Link**: [PDF](https://arxiv.org/pdf/2607.03882)  

**Abstract**: LLMs are increasingly deployed as post-hoc explainers of AI-generated outputs, yet it remains unclear whether they can reliably communicate probabilistic information in natural language. For this role to be viable, models must produce identical verbal descriptions for identical inputs, and select descriptions that accurately reflect the magnitude of the underlying numerical quantities. We evaluate whether nine LLMs meet these requirements within a two-stage prediction pipeline, in which an upstream model has produced probabilistic outputs characterized by their likelihood and uncertainty, and LLMs are tasked with selecting an appropriate verbal descriptor for each. We simulate predictions from an upstream model by taking samples from a Beta distribution parameterized by its mode and prior sample size. We then prompt LLMs to explain these predictions under six domain contexts and with ten temperature settings, and repeating each experiment ten times. We find that LLMs are generally consistent but miscalibrated, with substantially weaker performance on uncertainty than on likelihood tasks. Providing models with precomputed summary statistics (mode and prior sample size) reduced sensitivity to contextual framing but did not resolve the underlying miscalibration, suggesting that the bottleneck resides in the verbalization step itself. These findings indicate that current LLMs do not yet constitute reliable zero-shot standalone risk communication tools for probabilistic predictions. 

---
# Next-Gen Sponsored Search: Crafting the Perfect Query with Inventory-Aware RAG (InvAwr-RAG) Based GenAI 

**Authors**: Md Omar Faruk Rokon, Weizhi Du, Zhaodong Wang, Musen Wen  

**Link**: [PDF](https://arxiv.org/pdf/2607.03880)  

**Abstract**: Sponsored search plays a crucial role in e-commerce revenue generation, where advertisers strategically bid on keywords to capture the attention of users through relevant search queries. However, the process of identifying pertinent keywords for a given query presents significant challenges because of a vast and evolving keyword landscape, ambiguous intentions, and topic diversity. This paper highlights an opportunity for to earn a considerable amount of Ads revenue and user engagement where a significant proportion of queries fail to retrieve any sponsored ads. To utilize this opportunity, we introduce the Inventory-Aware RAG-based Generative AI model (InvAwr-RAG), which integrates advanced semantic retrieval and real-time inventory data. This model combines dynamically generated and historically successful queries to align with available inventory and ad campaigns while diversifying rewritten queries to enhance relevance and user engagement. Preliminary results show a significant 68% increase in fill rate and balanced relevance metrics, indicating a strong potential for increased ad revenue. The InvAwr-RAG model sets a new standard in dynamic query optimization, significantly improving ad relevancy, advertiser ROI, and user experience on Walmart's digital platform. 

---
# GeoSelect: Spatial-Program Execution for Training-Free Referring Remote Sensing Image Segmentation 

**Authors**: Yuhang Jiang, Guohui Deng, Miaozhong Xu, Chao Ruan, Jinling Zhao, Linsheng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2607.03869)  

**Abstract**: Referring remote sensing image segmentation isolates the object named by a natural-language expression in an aerial image. Existing training-free methods resolve the expression through implicit vision-language activations or region-text similarity, which gives weak control over the spatial, comparative, and ordinal relations that dominate aerial referring: they cannot represent constructions such as the largest ship or the second court from the left. We propose GeoSelect, a training-free pipeline that reframes referring as the execution of a typed spatial program. A frozen, text-only language model synthesises the expression into a small domain-specific language, a well-formedness checker accepts the program, and a deterministic executor runs it. The central abstraction is a single scored candidate set type under which every operator composes: continuous geometric fields realise position and proximity as dense pixel-level maps, while discrete set and order operators add the extremum, ordinal, counted-union, and relational constructions that fields alone cannot express. Because execution is explicit, every intermediate program, field, and ranking is inspectable, and a reliability ladder degrades any failing program to a field-only special case, so every expression returns an answer. GeoSelect attains 58.86 mIoU on RRSIS-D test and 55.27 mIoU on RISBench test, more than twice the best prior training-free method on RRSIS-D, with no referring supervision and on a single GPU. A controlled comparison with candidates and segmenter fixed attributes the gain to explicit execution, not the backbone; an oracle decomposition localises the residual gap to detection recall on RRSIS-D and selection on RISBench, and an exposure audit confirms robustness to pretraining leakage. Code will be released upon acceptance at the project page this https URL. 

---
# High-Fidelity One-Step Generative Visuomotor Policy via Recursive Correction, Frequency Consistency, and Contrastive Flow Matching 

**Authors**: Yuran Chen, Xinye Cai, Zhonglin Gong, Yang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2607.03865)  

**Abstract**: Generative models such as diffusion and flow matching have advanced robotic visuomotor policies by modeling multimodal action distributions, but their multi-step sampling or ODE solving introduces inference latency. Existing one-step acceleration methods often compress the whole generation process into a single large update, leading to spatial deviation, frequency distortion, and mode averaging. This paper proposes a high-fidelity one-step generative visuomotor policy framework that addresses these issues with three complementary mechanisms. Recursive Consistent Action Flow (RCAF) uses recursive correction to compensate for spatial truncation errors and align one-step predictions with refined flow trajectories. Dual-Timestep Frequency Consistency (DTFC) preserves high-frequency manipulation details through adaptive spectral consistency across flow timesteps. Contrastive Flow Matching (CFM) separates entangled action flows with a margin-based repulsive objective, reducing ambiguous actions in multimodal manipulation. Experiments on RoboTwin, RoboTwin 2.0, Adroit, DexArt, and real-world robot platforms show that the proposed method achieves competitive or superior performance compared with strong 10-step generative policy baselines while requiring only one forward pass (1 NFE), enabling low-latency visuomotor control. 

---
# When Simpler Is Better: Evaluating Translation Pipelines for Medieval Latin Manuscripts 

**Authors**: Nguyen Kim Hai Bui, Md. Easin Arafat, Tamás Gábor Orosz, Mufti Mahmud  

**Link**: [PDF](https://arxiv.org/pdf/2607.03836)  

**Abstract**: Despite remarkable progress in machine translation, Vision Language Models (VLMs) struggle on historical manuscripts, a domain that stresses core Natural Language Processing (NLP) capabilities: low-resource transliteration, archaic vocabulary, and noisy input signals. We present a systematic framework for evaluating the full image-to-translation pipeline on medieval Latin manuscripts, a setting in which scribal shorthand, ligatures, and parchment degradation expose failure modes that are invisible in clean-text benchmarks. Benchmarking on the CATMuS Latin dataset reveals a specialization gap: domain-specific Optical Character Recognition (OCR) models reduce character error rate by up to 4.3$\times$ compared to general-purpose VLMs, despite operating at orders of magnitude fewer parameters. We introduce the Interpres-Parallel-Corpus (IPC), a novel dataset comprising 1,383 aligned manuscript image lines, transcriptions, and expert translations, the first of its kind for medieval Latin. Our experiments uncover a complexity paradox: the simplest pipeline, a specialized OCR model feeding directly into a VLM, outperforms all multi-component variants. Adding retrieval-augmented generation (RAG) or post-OCR correction introduces prompt saturation and error propagation that degrade aggregate translation quality. These findings offer both a new benchmark and practical guidance for deploying translation systems in low-resource historical settings. 

---
# Beyond Static Rules: Automated Discovery of Latent Vulnerabilities in Text-to-SQL 

**Authors**: Hanqing Wang, Yongdong Chi, Jian Yang, Lei Yang, Jiehui Zhao, Yun Chen, Guanhua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.03833)  

**Abstract**: While Large Language Models (LLMs) have achieved remarkable success in Text-to-SQL tasks, their deployment in real-world environments is hindered by latent reliability issues. Identifying these latent weaknesses is critical for building trustworthy database interfaces, yet current diagnostic approaches rely heavily on static, expert-defined rules, which lack the capability for systematic and automated exploration. To bridge this gap, we propose SAGE (Systematic Automated Guided Exploration), a novel framework designed to autonomously uncover latent failure patterns in LLM-based Text-to-SQL generation. Specifically, SAGE generates vulnerability hypotheses for given samples and references a continuously evolving Vulnerability Codex to design targeted perturbations, thereby iteratively verifying and documenting potential defects. Extensive experiments on state-of-the-art open-source LLMs demonstrate that SAGE uncovers a substantial number of failure cases, highlighting the significant fragility of current models. Furthermore, our analysis reveals that the Vulnerability Codex exhibits strong cross-model transferability, indicating that the discovered patterns represent generalized structural weaknesses. Finally, we explore SAGE's potential for remediation. Although preliminary, lightweight fine-tuning on the generated samples yields promising improvements, suggesting a scalable pathway for closing the reliability loop in future work. 

---
# How Do Diffusion Classifiers Decide? A Bias-Centric Evaluation 

**Authors**: Saba Fathi, Fardin Ayar, Maryam Abdolali, Ehsan Javanmardi, Manabu Tsukada, Mahdi Javanmardi  

**Link**: [PDF](https://arxiv.org/pdf/2607.03831)  

**Abstract**: Diffusion models have recently been repurposed for zero-shot classification, giving rise to diffusion classifiers that identify the best-matching text prompt by minimizing the noise-prediction error. Despite their growing adoption, how these models make classification decisions remains poorly understood. We introduce ASOB-Bench, a bias evaluation for diffusion classifiers along three dimensions: Attribute binding, Size-Order bias, and Background dependency. These dimensions serve not as an exhaustive taxonomy but as targeted probes of how the text-conditioned reconstruction-error score reaches a decision. Such a perspective is well studied for discriminative vision-language models, yet remains overlooked for diffusion classifiers. Extending an existing framework with five new attribute categories on newly constructed datasets, we find diffusion classifiers are less prone to attribute misbinding than an OpenCLIP baseline; on the established ComCo benchmark they are substantially more susceptible to size-order shortcuts; and on ImageNet-B they suffer far larger accuracy drops, revealing heavy reliance on background over foreground cues. Reconstruction-error heatmaps and U-Net cross-attention visualizations expose the mechanism behind each bias. Because diffusion classifiers share the same denoiser as text-to-image models, these single-pass diagnostics also point toward analogous failure modes in generation. Overall, diffusion classifiers exhibit a distinct bias profile from vision-language models, offering guidance for building more robust diffusion-based models. 

---
# Q-TriM: Question-Guided Tri-Modal Attention for Audio-Visual Question Answering 

**Authors**: SungHun Kim, SeungJun Baek  

**Link**: [PDF](https://arxiv.org/pdf/2607.03825)  

**Abstract**: Audio-Visual Question Answering (AVQA) extends classical VQA by requiring joint reasoning over video and synchronized audio. However, many AVQA systems rely on deeply stacked layers of self- and cross attention across text, video, and audio. Such sequential stacking may incur loss of information such as subtle inter-modal cues over the layers, causing errors to accumulate across sequential attention layers during the fusion. We introduce Q-TriM which performs multi-modal fusion in a shallow and parallel manner instead of a deep and sequential manner. For Q-TriM, we propose a novel framework for attention operation incorporating video and audio conditioned on text. As a result, we obtain not only standard cross attention outputs but also Tri-Modal Attention representations in which Query, Key, and Value come from distinct modalities. These attention representations are combined in parallel at a single stage, thus avoiding the multi-modal fusion with deep stacks in order to mitigate error accumulation and depth-induced issues. Q-TriM achieves state-of-the-art performance on three AVQA benchmarks, including substantial gains on MUSIC-AVQA-R, which demonstrates its robustness and out-of-distribution generalization. Code is available at this https URL 

---
# DualView: Preventing Indirect Prompt Injection in Personal AI Agents 

**Authors**: Juhee Kim, Woohyuk Choi, Taehyun Kang, Youngmin Kim, Byoungyoung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2607.03821)  

**Abstract**: Personal AI agents that run on the user's local machine, such as OpenClaw, automate daily tasks including web search, email, and file management. Their access to computer resources, including the network, file system, and shell, exposes them to indirect prompt injection (IPI) attacks. Prior Dual LLM defenses block IPI by replacing untrusted data with symbols that the agent can reference but not read. However, they track untrusted data only inside the agent's context, so when the agent saves and later rereads untrusted data, that data, possibly an attacker's prompt, can return as trusted data rather than as a symbol, which we call stored IPI.
Operating on the user's real environment, which humans and programs share, is what makes agents like OpenClaw practical, and is exactly why a defense that ignores it is incomplete. Preserving symbols in such an environment is hard, because humans and programs need original data. We present DualView, which extends untrusted data tracking from the agent's context to the user's environment, including the file system, shell, network, and other agents, by giving each channel two views. In AgentView, the agent sees untrusted data as symbols even after writing it out and reading it back, blocking stored IPI, while HumanView preserves original data for humans and tools. DualView routes each tool call to the right view and synchronizes data across the two views. DualView deploys as an OpenClaw plugin using only tool hooks, without changing the agent's tool-call logic or tool implementations. Since DualView isolates untrusted data by design, its protection is not limited to known attack templates. In our evaluation on an IPI benchmark and PinchBench, DualView blocked every IPI attack, including stored IPI, while keeping utility close to the unprotected baseline. 

---
# CGGS: Consistency-Augmented Geometric Gaussian Splatting for Ego-centric 3D Scene Generation 

**Authors**: Zhenyu Sun, Xiaohan Zhang, Qi Liu, Huan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.03819)  

**Abstract**: Challenges remain in ego-centric 3D scene generation due to limited view overlap and the dominant influence of individual perspectives on scene interpretation. These factors hinder the creation of viewpoint-consistent and semantically aligned visual content, as well as the construction of accurate geometric structures. In this paper, we propose CGGS, a text-to-3D framework aiming to enhance 3D-content-awareness and address geometric distortions in ego-centric scene generation. Firstly, the Ego-centric Generator is proposed by fine-tuning a Multi-View Latent Diffusion Model with consistency-augmented loss to generate consistent, high-fidelity 2D content aligned with textual descriptions. Then, Layout Decorator leverages optical flow and point-track correspondence to estimate depth, therefore producing dense point clouds as coarse layouts from the ego-centric 2D priors. Building on this initialization, Geometric Refiner is proposed to enhance 3D Gaussian reconstruction via an entropy-based Mutual Information Depth Loss (MID) combined with a hierarchical optimization scheme for improving visual quality and geometric structure. Comprehensive experiments demonstrate that \textcolor{softred}{CGGS} outperforms previous methods in generating coherent and accurate text-driven 3D scenes. Project page: this https URL. 

---
# Probing Low-Level Acoustic Attribute Encoding in CLAP Audio Embeddings 

**Authors**: Héctor Martel, Joe Hennessy-Priest, Taemin Cho  

**Link**: [PDF](https://arxiv.org/pdf/2607.03806)  

**Abstract**: Audio foundation models are widely adopted as general-purpose feature extractors, yet the internal structure of their learned representations remains insufficiently understood. In this work, we analyze CLAP audio embeddings through a probing framework, studying the encoding of three fundamental perceptual dimensions: reverberation (RT60), loudness (LUFS), and spectral content, measured via spectral centroid (SC) and relative pitch (RP). Probes of increasing complexity are trained to predict each attribute from frozen embeddings across five datasets spanning noise, speech, monophonic musical notes, and music mixtures. Our primary finding is that all of these attributes are reliably recoverable from the CLAP embedding space across the examined datasets. Within this global picture, two encoding regimes emerge: RT60, LUFS, and RP are approximately linearly encoded, while SC requires non-linear probes. Both regimes generalize across eight additional audio foundation models, with the notable exception that amplitude-invariant architectures discard loudness entirely by construction. The identified linear feature directions are geometrically consistent across datasets for RT60 and LUFS, while highly domain-specific for RP. Finally, we provide a qualitative demonstration of cross-modal consistency, showing that text embeddings of acoustic descriptors align geometrically with the identified RT60 feature direction. 

---
# CineMobile: On-Device Image-to-Video Diffusion for Cinematic Camera Motion Generation 

**Authors**: Xuyao Huang, Zelai Deng, Xu Wang, Xizhong Xiao, Zhijie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2607.03803)  

**Abstract**: The growing demand for image-to-video creation on mobile devices has increasingly focused on cinematic motion effects like bullet time, dolly zoom, slow motion, etc. While Diffusion Transformers (DiTs) exhibit strong performance in video generation, their large parameter sizes and multi-step iterative denoising processes lead to substantial computational overhead, making efficient generation on mobile devices challenging. We propose CineMobile to bridge the gap. In particular, CineMobile adopts a three-fold optimization strategy: (1) leveraging a distillation-guided pruning approach to derive a compact yet efficient model that retains the essential video generation capabilities required for cinematic effects; (2) optimizing the compressed model into a 4-step generator via a combination of diffusion distillation and reinforcement learning; (3) employing a hybrid post-training quantization strategy to compress the model footprint to under 1 GB. Experimental results show that compared to the teacher model with the Wan 2.1 architecture, CineMobile achieves a 40x speedup in generation while maintaining comparable visual quality. Specifically, CineMobile generates 49-frame 480p videos with a per-step denoising latency of 0.6s on an NVIDIA H200 GPU and 20s on the MediaTek Dimensity 8400 Ultimate 5G platform, with a peak memory usage of 1.8 GB, demonstrating its practical applicability for mobile-based image-to-video creation. 

---
# Punching Above Their Weight: Classification-Head Fine-Tuning of Tiny Language Models (TLMs) for Verifiable Multiple-Choice Tasks 

**Authors**: Bhavesh Sood, Jaromir Savelka  

**Link**: [PDF](https://arxiv.org/pdf/2607.03801)  

**Abstract**: We define Tiny Language Models (TLMs) as models below roughly 3B parameters that fit on mainstream consumer devices. We study how to adapt them for and use them on verifiable multiple-choice tasks. We compare three LoRA-based fine-tuning paradigms (label generation, gold only, and our discriminative classification head) on a unified setup across several Qwen3 models from 0.6B to 8B and five benchmarks: HellaSwag, WinoGrande, PIQA, SciQ and ARC-C. Classification-head fine-tuning reliably outperforms label generation (+2-3%) at the 0.6B and 1.7B scales. Further, TLMs fine-tuned using the discriminative method are competitive to zero-/few-shot GPT-3 (175B), PaLM (540B) and GPT-4. The performance we report for Qwen3-0.6B and Qwen3-1.7B are SOTA on HellaSwag, WinoGrande, and PIQA. 

---
# Foundations of Equivariant Deep Learning: Unifying Graph and Sheaf Neural Networks 

**Authors**: Yoshihiro Maruyama  

**Link**: [PDF](https://arxiv.org/pdf/2607.03798)  

**Abstract**: Symmetry is everywhere in nature and society. Geometric deep learning exploits symmetries in data to improve the performance and efficiency of deep learning systems. In this paper, we extend geometric deep learning to utilize richer symmetry structures. Specifically, we develop order-equivariant neural networks (OENN), which generalize standard graph message passing and sheaf neural networks via the theory of equivariant bundles over face posets (face categories). We (i) characterize all linear order-equivariant maps, (ii) build OENN layers, and (iii) prove universal approximation theorems (UATs) for continuous order-equivariant maps, which are new results even when restricted to sheaf neural networks (for which no UAT was known before). We illustrate the framework on graph and sheaf models. Our results can also be seen as extending the known UAT for graph neural networks to a more general setting that subsumes sheaf neural networks as well. In addition, we show that OENN can be extended further to CENN, Category-Equivariant Neural Network, which gives the general form of equivariant neural networks as well as of equivariant universal approximation theorems, allowing us to leverage categorical symmetry in data (e.g., non-invertible symmetries on multiple objects with compositional relations on those symmetries). 

---
# Rethinking Depth Pruning for Vision Transformers: A Heterogeneity-Aware Perspective 

**Authors**: Zhenfeng Su, Kang Zhao, Han Bao, Tao Yuan, Zhongzhe Hu, Xianzhi Yu, Wenxuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.03784)  

**Abstract**: While prior studies have successfully compressed vision Transformers (ViTs) through various pruning techniques, most have concentrated on width pruning to achieve significant reductions in model size. Depth pruning, which removes entire layers from a ViT, is notoriously difficult for accuracy recovery despite its potential to deliver higher speedups, limiting the acceleration achieved by existing joint width-and-depth pruning methods. In this work, we reveal that the failure of existing depth pruning methods lies in their neglect of heterogeneity between different layers, and we introduce HetDPT, a heterogeneity-aware depth pruning method that avoids dimension mismatch. Comprehensive experiments on ImageNet-1K, CIFAR-100, COCO, and ADE20K validate our method: HetDPT achieves a 1.58$\times$ speedup for DeiT-B while maintaining accuracy and a 1.39$\times$ speedup for DeiT-S with nearly no accuracy degradation. Furthermore, when combined with width pruning, HetDPT+ sets a new state-of-the-art record in extreme ViT pruning, enhancing the acceleration ratio from 4.24$\times$ to 5.19$\times$ for the Isomorphic-Pruning-2.6G configuration while maintaining near-lossless accuracy; our code is available at this https URL. 

---
# SkillFab: An Agent-Native Skill Production Platform 

**Authors**: Anjie Xu, Yifeng Cai, Yi Li, Zixing Wang, Zhiyu Zhang, Jingfan Chen, Ruohan Xu, Leye Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.03780)  

**Abstract**: SkillFab is an agent-native platform for turning missing capabilities into reviewed, reusable Agent Skills. At runtime, agents first search for reusable skills; when no adequate skill exists, the unmet capability becomes a demand-first issue before any repository or implementation branch needs to exist. Development then proceeds through a SkillFab-managed repository, Git-ingested commit evidence, maintainer review, and registry publication. The same lifecycle is exposed through web, REST, and MCP surfaces, so humans, scripts, and external agents operate on shared state rather than separate task logs. The current system uses scoped Git push URLs, native range commit ingestion, workflow-state reads, and workflow-event histories to make long-running agent work reviewable and recoverable. We document the platform model, architecture, implemented capabilities, and three case studies: an end-to-end OS-detect skill run, a Docker research package that converts operational practice into reusable skill knowledge, and an external optimization case showing how improved skill artifacts can enter SkillFab as reviewable, versioned submissions.
Deployment: this https URL. 

---
# Self-Improving Diffusion Classifiers with Minority Preference Optimization 

**Authors**: Hyunsoo Kim, Jungmyung Wi, Soobin Um, Donghyun Kim, Suhyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.03770)  

**Abstract**: Prior studies have demonstrated that diffusion classifiers achieve robust zero-shot classification performance. However, their effectiveness is strongly tied to the pretraining data distribution: they perform well in majority, high-density regions of the data manifold, but are significantly less accurate in minority, low-density regions. Although prior works on minority sampling have focused on generating more minority-like images, what minority sampling fundamentally enables beyond generation remains underexplored. In this paper, we reveal a direct relationship between minority sampling in generation and the perception capability of diffusion classifiers. Specifically, we show that enhancing minority sampling broadens the coverage of underrepresented regions on the data manifold, thereby improving diffusion-based recognition. To exploit this connection, we propose \textit{Self-Improving Diffusion Classifiers with Minority Preference Optimization} (MiPO), which fine-tunes a pretrained diffusion model using minority preference rewards. Using only arbitrary caption data, MiPO generates candidate samples, rewards those that better cover minority regions, and optimizes the model with LoRA and Group Relative Policy Optimization, without additional image data, external foundation models, or external reward models. This enables stable, prompt-adaptive minority sampling and translates low-density generative coverage into improved zero-shot diffusion classification. To sum up, we show that diffusion classifier perception is biased toward majority regions, demonstrate that this bias can be alleviated through minority preference optimization, and evaluate MiPO on five standard datasets. 

---
# FedACT: Federated Adaptive Coordinate Trust Modulation for Robust Transformer Training under Data Heterogeneity 

**Authors**: Shuai Li, Qinglin Wang, Ping Luo, Jiahuan Wang, Hongyang Hu, Haotian Mo, Yigui Feng, Ziang Liu, Qisong Xiao, Jie Liu, Tao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.03763)  

**Abstract**: Federated Transformer training increasingly relies on local AdamW, whose adaptive updates can provide much stronger local progress than SGD-based training. However, under heterogeneous client data, even globally corrected AdamW updates may remain highly uneven in coordinate-wise reliability. We refer to this phenomenon as coordinate trust mismatch. Existing federated adaptive optimizers mainly address mismatch at the client-update or communication-round level, but still apply the corrected adaptive direction densely and uniformly across coordinates. In this paper, we propose FedACT, a global-aware coordinate trust modulation method for federated AdamW training. FedACT first forms a globally corrected adaptive direction and then reallocates update magnitudes according to a coordinate-wise trust score, assigning larger steps to coordinates jointly supported by local gradients and global correction, while preserving smaller non-zero updates on the remaining coordinates. Extensive experiments on federated vision Transformers, CNNs, LLM pre-training, and LLM fine-tuning show that FedACT consistently improves over strong federated adaptive baselines, with the largest gains on Transformer models under stronger data heterogeneity. Mechanism analyses further show that FedACT improves cross-client direction consistency, suggesting that coordinate-level trust allocation effectively complements round-level global-local correction. Code will be released. 

---
# EmCom-Diffusion: Probing Visual Reflection in Emergent Languages via Image Generation 

**Authors**: Haruumi Omoto, Tadahiro Taniguchi  

**Link**: [PDF](https://arxiv.org/pdf/2607.03752)  

**Abstract**: Measuring the extent to which emergent languages encode the visual content of their inputs is an open problem. We refer to this property as visual reflection: the extent to which emergent messages preserve information about their source images that can be recovered without appeal to the speaker-listener pair that produced them. Existing metrics measure it only indirectly, through proxies such as human-defined concept inventories, natural-language captions, structural distance correlations, or Referential Game accuracy, each of which can either miss visual content the message encodes or credit content it does not. We propose EmCom-Diffusion, an evaluation framework that measures visual reflection directly: it reconstructs each input image from its emergent message and compares the reconstruction with the original image itself, rather than with human-defined targets. Concretely, it finetunes a pretrained text-to-image diffusion model on (image, emergent-message) pairs and scores visual reflection as the perceptual similarity between the reconstructed and original images, operating generatively rather than discriminatively. Instantiating it on MS-COCO with a Referential Game, we validate the metric against random and fixed-token baselines under three pretrained visual encoders, and compare it against four existing metrics (CBM, supervised translation, TopSim, and R@1). EmCom-Diffusion captures visual content the other metrics miss. 

---
# A Failure-Mode Benchmark for Polymorphic Sybil Poisoning in RAG 

**Authors**: Donghyun Lee, Juntae Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.03739)  

**Abstract**: We release a benchmark and failure-mode-aware evaluation framework for grounded QA under coordinated retrieval poisoning. The framework partitions reader outputs into four mutually exclusive categories (\emph{gold}, \emph{hijack}, \emph{abstention}, \emph{drift}), with instance-level paired clean-to-poison transition matrices and a Forced Exposure protocol isolating reader-side conflict resolution from retrieval variance. We introduce \emph{polymorphic sybil poisoning}, a coordinated attack class in which $S$ lexically diverse passages jointly support an attacker-chosen target while evading lexical near-duplicate filters that fully detect monomorphic baselines (capturing the residual 14.2\% with E5 cosine raises false-positive rate 9$\times$ on legitimate same-topic pairs). A monomorphic-polymorphic ablation under Forced Exposure isolates the diversity dimension and reveals a $+$18.8pp hijack amplification (95\% paired bootstrap CI $[+15.4, +22.4]$, $B{=}5{,}000$): monomorphic copies register only 4.0\% as hijack while polymorphic surface diversity recovers 22.8\% -- a 5.7$\times$ amplification of the ASR-visible attack channel. ASR alone treats every non-target output identically; under attack, abstention and drift together hold 47-66\% of output mass, unmonitored by ASR+ACC, and two readers at nearly identical ASR (within 0.2pp) differ by 16.5pp on abstention and 17.2pp on drift -- failure profiles invisible to ASR. We release the frozen benchmark (3{,}145 questions, 2{,}982 retained sybil groups; $S{=}6$ chosen to dominate top-10 retrieval slots, §\ref{sec:setup}), the official four-way evaluator, paired-transition utilities, and the Forced Exposure harness across five readers (7B-120B), two retrievers, and two cross-validation datasets (TriviaQA, 2Wiki), under CC~BY-SA~4.0 (data) and MIT (software); release information in §\ref{sec:release}. 

---
# Attending to Multimodal Generation One Token at a Time 

**Authors**: Varun Gupta, Vineet Gandhi, Makarand Tapaswi  

**Link**: [PDF](https://arxiv.org/pdf/2607.03738)  

**Abstract**: Multimodal large language models (MLLMs) generate responses autoregressively, integrating visual and linguistic information in an evolving context. Prior work on interpretability has focused on individual layers and circuits (where), leaving the token-level dynamics of multimodal computation during generation (when) underexplored. We address this gap and study attention shifts as per semantic role; tracking model attention to image, text, instruction, and previously generated tokens, One Token at a Time (OTaT). We introduce multimodal tasks that require explicit switching between visual and textual context within a single response. Across two mainstream model families and four open-weight MLLMs of varying sizes, we establish consistent patterns: attention to image peaks at tokens requiring image-derived information, instruction tokens are revisited during task transitions, and attention to previously generated tokens increases as the generation progresses. Causal attention blocking interventions validate the functional role of these trends. We profile model behavior under disrupted attention and observe responses falling back to language priors, or exhibiting cross-modal leakage, denial, or recovery. Finally, informed of the attention dynamics through our novel analysis, we propose a simple test-time intervention to boost attention to the relevant modality at the right time, significantly improving multimodal task performance. 

---
# CoGen3D: An Agentic Human-AI Co-Design Pipeline for 3D Asset Generation for Virtual Reality 

**Authors**: Weiwei Jiang, Wanyu He, Zheyu Tan, Zheyuan Kuang, Difeng Yu, Shinobu Hasegawa, Sven Mayer, Zhanna Sarsenbayeva  

**Link**: [PDF](https://arxiv.org/pdf/2607.03731)  

**Abstract**: Creating 3D assets for virtual reality requires modeling expertise, which restricts the authorship of immersive experiences. Existing generative AI tools rely on unconstrained, command-driven prompting, lacking the conversational scaffolding needed for users to articulate their intent and validate designs prior to rendering. To address this, we introduce CoGen3D, an agentic human-AI co-design pipeline that proactively guides users through conversational intent elicitation, a concept image confirmation, and image-to-3D generation that directly deploys to immersive scenes. We evaluated this system through a user study (N=120) across six affectively diverse immersive scenes, observing 60 Design group participants who co-created 3D assets for the scenes, and 60 Validation group participants who experienced the scenes with generated assets. Our findings show that co-designed assets are associated with higher scene engagement and shifted affective responses, while participants generally preferred concept images over the final 3D assets, with no increased leniency toward degradation in their own creations. Analysis of the human-AI conversations further shows that target environments shape users' conversational patterns. Our results suggest that our staged, intent-based co-design can democratize virtual reality authoring and shift immersive content creation from technical execution toward collaborative spatial design. 

---
# OmniTacTune: Policy-Agnostic Real-World RL for Tactile Residual Adaptation of Visual Policies 

**Authors**: Kelin Yu, Haode Zhang, Harish Ravichandar, Yunhai Han, Ruohan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2607.03723)  

**Abstract**: Visual policies learned from human videos, teleoperation, and robot demonstrations offer scalable motion priors, but often fail in contact-rich manipulation, where success significantly depends on local force and contact geometry. Tactile sensing provides these complementary signals, yet tactile data remain costly to collect and hard to generalize across sensors, robots, and tasks. We introduce OmniTacTune, a policy-agnostic real-world RL pipeline that adapts tactile feedback to pretrained visual policies through residual correction. OmniTacTune uses a two-stage design: it first bootstraps tactile-aware learning from autonomous base-policy rollouts, then learns a lightweight tactile residual policy through online interaction. Extensive experiments show that OmniTacTune generalizes across diverse contact-rich tasks, visual base policies, and tactile representations. Across four real-world contact-rich tasks, it improves visual base policies from 5-40% success to 85-100% within 40-80 minutes, demonstrating an efficient path for adapting tactile feedback to scalable visual robot policies. Project page: this https URL 

---
# Don't Blame the Large Language Model: How Scaffolding Evolution Shapes Coding Agent Quality 

**Authors**: Oussama Ben Sghaier, Hao Li, Bram Adams, Ahmed E. Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2607.03691)  

**Abstract**: Coding agents, autonomous systems that use large language models (LLMs) to resolve software engineering tasks, rely on agentic scaffolding: a middleware layer in between a developer and a large language model that orchestrates system prompts, tool execution, context management, and iterative reasoning loops. While these scaffoldings evolve at extreme velocities, no study has examined how this evolution affects agent quality (i.e., effectiveness and efficiency) over time. Practitioners regularly report quality regressions after scaffolding updates, yet consistently attribute them to the underlying model rather than the scaffolding itself. In this paper, we address this gap by conducting the first controlled longitudinal study that isolates the scaffolding's contribution. Unlike prior work that fixes the scaffolding and varies the model, we fix the model and vary only the scaffolding, evaluating 35 sequential releases to measure their impact on agent effectiveness and efficiency. We first empirically study the development and release evolution of five major open-source scaffoldings (i.e., Codex, Qwen Code, Gemini, OpenCode, and OpenHands), revealing extreme release velocities exceeding two releases per day and thousands of issues within months. We then perform a controlled deep dive into 35 sequential releases of the Qwen Code CLI, evaluating each against 50 stratified SWE-bench Verified tasks while holding the underlying LLM constant. We trace the resulting quality fluctuations to specific development patterns and architectural components, and illustrate our findings with concrete qualitative evidence linking individual pull requests to measured quality shifts. 

---
# Phase-Preserving Trimodal Transformer for Tropical Forest Biomass Estimation Using Optical and PolInSAR Data 

**Authors**: Luiz Felipe Parente Santiago, Rosiane Rodrigues de Freitas, Daniel Rodrigues dos Santos, Felipe Ferrari  

**Link**: [PDF](https://arxiv.org/pdf/2607.03663)  

**Abstract**: The accurate estimation of Above-Ground Biomass (AGB) in mature tropical forests remains a critical challenge in remote sensing, primarily due to the saturation of Synthetic Aperture Radar (SAR) signals in high-density areas and persistent cloud cover affecting optical imagery. To overcome these physical limitations, we propose the Trimodal Coherent Co-attention Transformer (TCCT), a physics-informed deep learning architecture. The TCCT natively fuses optical surface reflectance (Landsat-5) with complex-valued Polarimetric SAR Interferometry (PolInSAR) data from both P and L bands. Unlike traditional fusion methods, our architecture employs complex-valued encoders to preserve spatial phase coherence, coupled with a dynamic co-attention mechanism that acts as an adaptive gating module, reducing the weight of cloud-corrupted optical pixels and shifting reliance to microwave phase data. We also derived a localized spatial allometric calibration model via Levenberg-Marquardt optimization, tailored to the specific wood density of the Paracou region in the Amazon basin. Evaluated using a two-stage protocol, the TCCT first underwent a rigorous 5-fold cross-validation to establish robust global weights (achieving a global RMSE of 4.19 m). Subsequently, following a localized spatial fine-tuning phase over 200 epochs, the model attained an absolute RMSE of 3.78 m and an $R^2$ of 0.33 for Canopy Height Models (CHM), outperforming standard Random Forest, CNN, and Vision Transformer baselines. Our ablation study confirms that preserving phase coherence mitigates deep-canopy signal saturation. When converted to AGB, the fine-tuned TCCT map yielded a Relative RMSE (rRMSE) of 4.51% in dense forest areas above 50 Mg/ha. By meeting the European Space Agency (ESA) BIOMASS mission requirement of less than 20% error, the TCCT provides a robust framework for continuous carbon stock mapping in tropical biomes. 

---
# A Fair Benchmarking of Deep Relational Database Learning Models 

**Authors**: Kazi F. Akhter, Bharath Ajendla, Manar D. Samad  

**Link**: [PDF](https://arxiv.org/pdf/2607.03659)  

**Abstract**: Relational databases (RDBs) are the primary data infrastructure in many enterprises, yet recent deep learning methods designed for RDBs have been evaluated under inconsistent experimental protocols, making fair comparison difficult. We present one of the first systematic benchmarking studies of recently released deep learning methods for RDBs, evaluating them across five relational databases, with one classification and one regression task for each. We refactor all deep RDB models to allow the full range of experimental procedures to be applied consistently across all methods. Our findings indicate that the relational transformer (RT) approach delivers the strongest overall performance on both classification and regression tasks compared to the state-of-the-art graph-based modeling and learning of RDBs. Even for single-table learning tasks, deep learning methods designed for RDBs outperform the leading tabular foundation model, TabPFN 2.5. Extending learning from a single table (hop = 0) to multiple tables (hop = 1, 2) by connecting neighboring tables in relational databases enhances performance, but the additional benefit from higher hops diminishes as computational overhead grows. Deep RDB learning methods have the potential to challenge state-of-the-art tabular foundation models, especially on large-scale enterprise data. The source code for this benchmarking study is publicly available. 

---
# ViPo-MLLM: Visual-Pose Multimodal LLM for Gloss-Free Sign Language Translation 

**Authors**: Ahmed Abul Hasanaath, Bicheng Xu, Mir Rayat Imtiaz Hossain, Leonid Sigal, Hamzah Luqman  

**Link**: [PDF](https://arxiv.org/pdf/2607.03657)  

**Abstract**: Gloss-free Sign Language Translation (SLT) translates sign language videos into spoken-language sentences without gloss annotations, avoiding costly labeling but requiring fine-grained modeling of hands, body, and facial cues. Existing methods often use single-modality or weakly fused features, limiting performance. We propose ViPo-MLLM, a framework that integrates spatio-temporal RGB and human pose features. Dedicated encoders model intra-modal dynamics and cross-modal attention captures long-range dependencies. The fused representation is conditioned with a structured prompt and processed by an LLM trained with contrastive and language modeling objectives. The proposed model was evaluated on the PHOENIX14T and CSL-Daily datasets and achieved new state-of-the-art results on both datasets. Moreover, the ViPo-MLLM model attained competitive performance compared to gloss-based recognition approaches, confirming the effectiveness of the proposed pose cues and cross-modal attention mechanisms. 

---
# AutoCedar: An Agentic Framework for Verifier-Guided Access Control Policy Synthesis 

**Authors**: Adarsh Vatsa, Sachi Shome, Yingming Zhou, William Eiers  

**Link**: [PDF](https://arxiv.org/pdf/2607.03656)  

**Abstract**: Large Language Models are increasingly used to turn natural-language requirements into code. In access control, that shortcut is dangerous: a generated policy can compile and read correctly while granting access that no one approved. The difficulty is not only writing policy code. It is fixing what the requirements mean before code is written, and then checking that the final policy actually satisfies that intent. We present AutoCedar, a verifier-guided system that first turns natural-language access-control requirements into a reviewed, checkable target, and then synthesizes Cedar policies against that target. AutoCedar decomposes schema and policy authoring into small intent atoms: reviewable claims about vocabulary and behavior. Once those atoms pass mechanical validation and human intent review, the model proposes a candidate policy, the verifier checks it against the approved target, and each failure is turned into a repair signal that tells the model whether to broaden, narrow, or restructure the policy without changing the target. Because the model's work is split into small problems, each grounded in reviewed intent and backed by verifier feedback, end-to-end policy authoring becomes tractable. AutoCedar converges on all 221 tasks of CedarBench, our benchmark of authorization tasks paired with executable semantic boundaries. Across three requirements-corpus case studies covering healthcare, education, and conference management, AutoCedar converts noisy prose and extracted access-control fragments into reviewed schemas, formal checks, and a globally verified Cedar policy store for each scenario. 

---
# ELiTeFormer: An Efficient Transformer for FPGAs 

**Authors**: Victor Agostinelli, Nicolas Bohm Agostini, Antonino Tumeo  

**Link**: [PDF](https://arxiv.org/pdf/2607.03652)  

**Abstract**: Transformer blocks are prevalent in large language model (LLM) but present deployment challenges due to their challenging computational and memory demands. While prior work has typically optimized attention mechanisms or feed-forward networks (FFNs) separately, few hardware (HW) architecture have jointly addressed both components with co-designed hardware acceleration. We present ELiTeFormer (Efficient Linear Ternary Transformer), the first Transformer model architecture that unifies hybrid linear attention with ultra-low-precision (ternary) linear projections, specifically co-designed for field-programmable gate array (FPGA) deployment. ELiTeFormer achieves 10x model weight compression and 12.8x key-value (KV) cache compression compared to LLaMA 3, while maintaining competitive accuracy (31.9% on the MMLU benchmark, within 3.0% of BitNet b1.58). Our key architectural contribution is a novel processing element (PE) micro-architecture that eliminates all multiplications in ternary linear projections through bitmasking operations, significantly reducing resource utilization by completely avoiding dedicated digital signal processing (DSP) blocks. We simulate, synthesize, and deploy ELiTeFormer targeting a Xilinx VCK5000 Versal board using high-level synthesis (HLS) flows. Block-level simulations show 9.6x speedup for FFN operations and 4.4x speedup for attention compared to standard implementations. End-to-end deployment achieves up to 3.9x lower latency and 3.2x better energy efficiency than LLaMA 3 on an NVIDIA A100 graphics processing unit (GPU) at long context lengths. This represents the first FPGA realization combining linear attention with ternary quantization, demonstrating the viability of algorithm-architecture co-design for next-generation LLM acceleration. 

---
# ClinOCR-Bench: A Comprehensive Clinical Scanned Document Dataset for Optical Character Recognition Model Evaluation 

**Authors**: Enshuo Hsu, Jin Zhou, Kirk Roberts  

**Link**: [PDF](https://arxiv.org/pdf/2607.03650)  

**Abstract**: Extracting textual information from scanned medical documents, such as external laboratory reports and manually filled forms, has been a major challenge in modern electronic health records (EHRs). Recent advancements in vision language models (VLMs) have shown great promise over traditional OCR tools. However, at this point, most clinical OCR studies were conducted on private, institutional data. To our knowledge, there are few publicly available datasets for evaluating OCR models in the clinical domain. Furthermore, common scanning artifacts that undermine OCR performance are not reflected in those datasets, leaving a systematic evaluation unfeasible. Therefore, we release a publicly available, realistic-looking OCR benchmark dataset, ClinOCR-Bench, with 384 scanned images across 6 subsets: Normal, Handwriting, Poor Quality, Rotation, Tables, and Mix-artifacts. ClinOCR-Bench features: 1) diverse document types and layouts, 2) full coverage of common EHR scan artifacts, 3) protected health information-free, 4) template-aware train/test split, and 5) adequate sample size for OCR benchmarking. Baseline OCR performance was evaluated using state-of-the-art open-weight and proprietary VLMs. The dataset and documentation are available on GitHub (this https URL). 

---
# Moonstone: A Multimodal Foundation Model and Benchmark for Lunar Remote Sensing 

**Authors**: Ayush Prasad, Swarnalee Mazumder  

**Link**: [PDF](https://arxiv.org/pdf/2607.03644)  

**Abstract**: Decades of orbital missions have produced multi-modal remote sensing data for the Moon, spanning optical imagery, spectroscopy, thermal emission, radar, gravity, and elemental composition. Yet these datasets remain fragmented across archives, and no benchmark exists for evaluating machine learning on lunar data. We introduce Moonstone, the first multi-modal foundation model benchmark for lunar remote sensing. Our contributions are: (1) a 28-channel, 128 pixels-per-degree (~237 m) global lunar pretraining dataset from seven instrument families across five missions, (2) MG-MAE, a modality-grouped masked autoencoder with per-group convolutional tokenizers, a shared Vision Transformer encoder, attention masking for missing modalities, coverage-adaptive masking for heterogeneous spatial coverage, and spectral continuity regularization for physically plausible reconstructions, and (3) a benchmark of six downstream tasks covering classification, regression, and segmentation. MG-MAE pretrained features outperform scratch baselines on all tasks and surpass both ImageNet-pretrained and vanilla MAE baselines by large margins. Data and code are available at this https URL and this https URL . 

---
# Revealing Hidden Model Behaviors with Task-Specific Self-Reports 

**Authors**: Taras Kutsyk, Bartosz Zieliński  

**Link**: [PDF](https://arxiv.org/pdf/2607.03640)  

**Abstract**: Fine-tuning can give a language model a hidden behavior--it may give false answers under a narrow condition, or give harmful advice only when a prompt touches a particular topic. We introduce the Stabilized Adapter for self-Report (SAR), a lightweight LoRA adapter that makes a fine-tuned model describe its own hidden behavior in plain language, using only the model and the dataset it was trained on. Across seven implanted behaviors (plus a no-behavior control), SAR detects the hidden behavior in every one--even when the model has generalized into broad misalignment that the training data alone does not predict. Introspection Adapters (IA), the closest existing baseline, detects some behaviors from our suite but misses others entirely--and where it misses, it hallucinates, consistently reporting wrong behaviors. SAR retains positive signal on every setting where IA fails and halves the rate of hallucinations. This makes it much easier for practitioners to audit their models and obtain reliable answers to "what did my model actually learn?" type of questions. 

---
# An AI-Assisted Solution to the Signed BAR Conjecture: Uniqueness in the Harrison--Reiman Class and a Completely-$\mathcal{S}$ Class Obstruction 

**Authors**: Yiping Lu, Youheng Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.03639)  

**Abstract**: For a multidimensional reflected diffusion, determining whether the associated basic adjoint relationship (BAR) uniquely characterizes the stationary distribution is a basic uniqueness problem in the BAR approach. The problem has remained unresolved for more than 35 years since the introduction of the BAR approach. In this paper, we resolve the finite-signed uniqueness problem for stable Harrison--Reiman data with a nonsingular $M$-matrix reflection matrix. The proof uses pathwise differentiability of the reflected diffusion implies feasible directional differentiability of the probabilistic resolvent to show that, at boundary points, its one-sided initial-state derivative factors through the tangent projection and vanishes along active reflection directions. An interior one-sided convolution then yields smooth test functions whose oblique derivatives are uniformly bounded and converge pointwise to zero on each closed face. The interior signed measure is consequently invariant for the reflected semigroup. The proof was discovered with the assistance of ChatGPT 5.5 Pro and subsequently verified by the authors. We also show that the nonsingular $M$-matrix assumption is structural. In the larger completely-$\mathcal{S}$ class, a nonsingular reflection matrix with a singular proper principal block admits boundary gauges supported on lower-dimensional strata. Under standard exponential ergodicity and a mild one-step regulator bound, these gauges produce nonzero zero-mass signed BAR tuples; indeed the zero-mass interior BAR coordinates contain an infinite-dimensional subspace. A four-parameter three-dimensional family, including an explicit rational example, verifies the obstruction. Thus the finite signed version of the Dai--Dieker question has a positive answer in the Harrison--Reiman $M$-matrix class and a negative answer in a natural completely-$\mathcal{S}$ extension. 

---
# RADIO1D: Elastic Representations for Condensed Vision Modeling 

**Authors**: Greg Heinrich, Mike Ranzinger, Collin McCarthy, Natan Bagrov, Eugene Khvedchenya, Bryan Catanzaro, Jan Kautz, Andrew Tao, Pavlo Molchanov  

**Link**: [PDF](https://arxiv.org/pdf/2607.03624)  

**Abstract**: This paper challenges the assumption that vision-language models (VLMs) require fixed patch-based 2D vision features. Analyzing fine-tuned vision encoders, we find that representations become increasingly abstract and less spatially coherent during VLM training. Notably, models trained with image-text alignment (such as SigLIP2) develop a small number of specialized tokens that effectively summarize global image content. Building on this, we introduce RADIO1D, which compresses images into a compact, variable-length 1D token sequence using multi-teacher knowledge distillation and an autoencoder design. The resulting representations exhibit strong hierarchical summarization, enabling accurate scene understanding - even with a single token - and support improved composition-aware image retrieval. In VLMs, RADIO1D provides flexible accuracy-efficiency tradeoffs through adjustable token counts, delivering competitive performance on diverse multimodal benchmarks with lower computational overhead and better accuracy. 

---
# A Step Towards Robust Unsupervised Domain Adaptation via Fine-Tuning and Reinforcement Learning 

**Authors**: Sushant Dagaji Desale, Rahul Mishra, Ashutosh Kumar Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2607.03600)  

**Abstract**: Adversarial robustness in Unsupervised Domain Adaptation (UDA) remains a significant challenge due to noisy pseudo labels and inherent distributional shifts between the clean source and adversarially perturbed target domains. Existing approaches often fail to achieve an optimal trade-off between robustness and accuracy, as pseudo-labels generated by domain-adapted models tend to introduce classification errors under adversarial attacks. In this work, we propose \textbf{SFT+RL}, a two-stage robust UDA framework that integrates Supervised Fine Tuning (SFT) and Reinforcement Learning (RL) on top of CLIP's pre-trained visual encoder. In the SFT stage, we adversarially fine-tune a linear classifier using PGD-based perturbations over the labelled source domain while partially unfreezing CLIP's projection layer. It allows adaptation to adversarial noise while preserving CLIP's rich semantic priors. We introduce a confidence-guided pseudo-labeling strategy in the RL stage to annotate unlabeled target samples progressively. Pseudo labels are filtered using a decaying confidence threshold to balance quality and coverage, and the model is trained on a composite dataset formed by combining clean source samples with high-confidence target samples. Adversarial training is applied to mixed batches of clean and adversarial examples to enhance cross-domain robustness. Comprehensive evaluations on three benchmark datasets OfficeHome~\cite{tomm-ude}, PACS~\cite{pacs}, and VisDA~\cite{visda} demonstrate the effectiveness of our approach. Notably, \textbf{SFT+RL} achieves average improvements of \textbf{10.2\%} in clean accuracy and \textbf{15.8\%} in adversarial robustness across all three datasets, outperforming existing state-of-the-art methods. 

---
# They Infer What You Meant: Models Represent Communicative Intent More Reliably Than They Act On It 

**Authors**: Alex Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2607.03598)  

**Abstract**: When a person shares something with a language model, the model often answers the surface of the message rather than what the sender was doing by sending it: share a finished project and it critiques the code; share a raw late-night line and it runs a wellness check. We treat the sender's communicative intent, the Gricean what-was-meant, as a first-class interpretability object, and show the failure is one of readout on top of a robust representation. A linear probe decodes the sender's intent, whether they want a thing recognized or evaluated, from a model's default-pass hidden states, cleanly and surface-independently, across six models and four families and in the base checkpoints. The representation generalizes further, to intent that is only pragmatically inferred, and to a second, lexically clean intent (support versus help). The behavioral half of the story, and every causal test, is established on the recognize/evaluate contrast, where what varies is whether the default output acts on the intent. The readout lags the representation in depth within a model (the intent is decodable several layers before it drives the output); across models, which ones act on it by default is model-specific, an observed stratification (three of six show the failure) that we do not read as a scaling law. Where the gap is open, a direction closely tied to the representation, the discriminative direction at a searched-for layer, is a causal handle: steering it recovers the intended behavior, as well as an explicit instruction does and with no prompt at all. This direction is near-orthogonal to the feedback-offering axis, so it routes a represented intent rather than a generic feedback knob, though at the recovery dose the routed intent can override an explicit request. We support each link with controls against obvious deflations and report the nulls as plainly as the confirmations. 

---
# Token-Based Affordance Grounding with Large Vision-Language Models 

**Authors**: Seung Il Lee, Qinqian Lei, Daguang Xu, Dong Yang, Robby T. Tan, Yixin Chen, Bo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.03595)  

**Abstract**: Affordance grounding aims to localize image regions that support a specific action, serving as a core capability for physical intelligence and embodied perception. Previous studies have primarily relied on weakly supervised learning with action labels from exocentric images. However, these methods often struggle with visually ambiguous exocentric images containing co-occurring actions; moreover, they fail to distinguish semantically similar actions because existing methods typically rely on brief action phrases that lack rich semantic details for action-specific localization. Although large vision-language models (LVLMs) encode rich action semantics and their action-conditioned textual outputs implicitly contain spatial cues, they do not directly provide action-specific spatial localization. To address these problems, we propose TokAG, a zero-shot affordance grounding framework that exploits the token-level semantic-spatial signals in LVLMs to localize action-relevant regions without external supervision. We observe that attention maps associated with different LVLM output tokens vary significantly, with many attending to irrelevant regions such as the background. Thus, we introduce a spatial-aware token-selection mechanism to systematically evaluate each output token and select the one whose attention maps exhibit dominant activation over the target object, instead of relying on arbitrary attention maps. By extracting these object-focused attention maps, we transform the LVLM's implicit semantic signals into zero-shot affordance heatmaps. Our zero-shot framework consistently outperforms prior weakly supervised approaches across multiple benchmarks, improving NSS by 10.7% on the unseen split of AGD20K and by 29.7% on HICO-IIF. The code and models will be made publicly available. 

---
# An Interpretable Deep Learning Framework for Discovery and Clinical Validation of Deep Radiomic Signatures in Tumor Classification 

**Authors**: Chengkun Sun, Jinqian Pan, Renjie Liang, Zhengkang Fan, Xin Miao, Yi Guo, Mei Liu, Muxuan Liang, Russell Terry, Jie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.03593)  

**Abstract**: Imaging signatures are quantitative features extracted from medical images that provide clinically meaningful information for tumor diagnosis, characterization, prognosis, and treatment planning. Although deep learning has shown great potential for imaging signature discovery, its limited interpretability remains a major barrier to clinical adoption. Existing approaches often achieve high predictive performance but provide little biological insight into the identified signatures. We propose a unified framework for interpretable imaging signature discovery by integrating deep learning based segmentation, explainable classification, and radiomic analysis. A robust segmentation model is first used to accurately delineate tumors, followed by a Grad-CAM guided pipeline that identifies diagnostically important regions as candidate imaging signatures. A mutual information based adaptive thresholding strategy enables patient-specific signature extraction. The resulting signatures are validated using a downstream deep learning classification model, while radiomic features extracted from the signature regions are evaluated with traditional machine learning models and interpreted using SHAP to identify the most discriminative biomarkers. The proposed framework is evaluated on the public BUSI breast ultrasound, KiTS renal CT, and BraTS brain tumor datasets, as well as a private UF Health renal CT cohort. Compared with conventional whole-tumor radiomics, the proposed signature-based approach achieves improved discriminative performance while providing greater biological interpretability. By converting deep learning attention into reproducible quantitative imaging biomarkers, this framework offers an interpretable and reproducible solution for non-invasive tumor characterization and imaging biomarker discovery. 

---
# Responsibility Distribution Estimation in Ego-View Accident Videos with Multimodal Large Language Models 

**Authors**: Ryosei Tamura, Andrew Shin  

**Link**: [PDF](https://arxiv.org/pdf/2607.03591)  

**Abstract**: Recent studies on multimodal traffic accident understanding have mainly relied on infrastructure-camera footage, satellite imagery, or structured crash records. However, such data sources are costly to deploy and maintain at large scale, and they cannot objectively capture what the driver was actually able to observe before the accident. In contrast, ego-view accident videos directly represent the driver's visual perspective, making them suitable for reasoning about avoidability and driver responsibility. In this paper, we introduce responsibility distribution estimation for ego-view traffic accident videos, a new task in which a model predicts the percentage of responsibility assigned to each involved agent. We construct an LLM-assisted responsibility annotation pipeline and fine-tune multimodal large language models under multiple input settings, including raw frames, segmentation-enhanced input, and textual descriptions. Experimental results establish a strong initial benchmark, demonstrating that multimodal LLMs can effectively perform this nuanced, constraint-based reasoning task. Our findings suggest that ego-centric accident videos provide a promising foundation for socially and legally meaningful multimodal reasoning beyond conventional accident classification and explanation tasks. 

---
# PLGSA-Transformer: Periocular Landmark-Guided Attention with Occlusion-Adaptive Cosine Thresholding for Cross-Modal Masked and Unmasked Face Recognition 

**Authors**: Dana A Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2607.03581)  

**Abstract**: The widespread adoption of facial masks, accelerated by COVID-19 and mandated in security-sensitive settings, has exposed limitations of conventional face recognition systems. Existing approaches relying on fixed cosine thresholds, non-adaptive CNNs, and purely data-driven features fail to generalize when facial regions are occluded, creating a gap between lab performance and real-world deployability. This paper proposes PLGSA-Transformer, a cross-modal face matching framework with three contributions. First, Periocular Landmark-Guided Spatial Attention (PLGSA) uses MediaPipe landmarks to compute Gaussian heatmaps over the eye, brow, and forehead regions, fusing them with EfficientNetB3 features via a learnable residual gate to direct attention toward discriminative visible regions. Second, a Hybrid CNN-Transformer Branch reshapes feature maps into tokens processed by a two-layer Multi-Head Self-Attention encoder, enabling cross-regional dependency modelling. Third, the Occlusion-Adaptive Cosine Threshold (OACT) is a jointly trained head that raises the matching threshold in proportion to predicted occlusion severity. The model is evaluated on 858 images from Zenodo MDMFR (60%), Kaggle CelebA-HQ masked collection (25%), and author-collected images (15%), spanning both genders, ages 21-75, with varied mask types, trained via a unified loss combining contrastive verification, identity classification, and occlusion cross-entropy. PLGSA-Transformer achieves 97.22% pair verification accuracy with ROC AUC 1.0000, surpassing VGG-16-based MUFM (Abdullah et al., 2025; 95.0%), HOG classifiers (Adnan et al., 2020; 85.0%), and Feature-based Structural Measure (Shnain et al., 2017; 86.61%). These results confirm that encoding periocular geometry into attention, with Transformer modelling and occlusion-adaptive thresholds, yields a robust, scalable solution for cross-modal masked face recognition. 

---
# Differentiate the Evaluator, Not the Program: An Efficient Runtime Representation for Neuro-Symbolic Learning 

**Authors**: Lucas Sheneman  

**Link**: [PDF](https://arxiv.org/pdf/2607.03574)  

**Abstract**: AI systems increasingly propose executable scientific models whose value depends on both their symbolic structure and their fitted continuous parameters. This makes parameter calibration the bottleneck of program-and-parameter co-search: an outer loop can generate thousands of candidate programs, but each needs an inner gradient-based optimization before it can be assessed. Staging each candidate into its own differentiable graph makes individual models fast but sacrifices the program-as-data property that keeps search fluid; interpreter-based approaches preserve programs as runtime data but pay interpreter overhead that dominates the numerical work.
We present the Native Differentiable Virtual Machine (NDVM), a runtime representation that differentiates executable programs without compiling each candidate into a separate graph. NDVM separates symbolic structure from differentiable numeric state: tags, symbols, environments, and control remain native runtime data, while numeric payloads live in dense batched buffers with exact reverse-mode gradients recorded along the realized execution trace, so one evaluator walk is amortized across large populations of parameter vectors. A locked cost model of a real differentiable self-hosted Scheme interpreter motivates the design. We realize NDVM as a native runtime with forward and gradient equivalence to the reference backend, about 60x per-lane batch amortization, near-linear multicore scaling, and two independent front ends. In fixed-budget co-search over LLM-proposed programs, NDVM reaches high-quality solutions about 24x sooner in wall-clock time, suggesting runtime differentiation as a practical systems foundation for scientific discovery workflows. 

---
# Teacher Supervision over Representation Equivalence Classes 

**Authors**: Sang Il Han  

**Link**: [PDF](https://arxiv.org/pdf/2607.03572)  

**Abstract**: Knowledge distillation is usually framed as a choice of what to match in the teacher - its logits, hidden features, or sample relations - which presupposes that the teacher's representation has absolute coordinates to match. It does not: a pretrained representation is identifiable only up to an orthogonal-and-isotropic-scaling equivalence class, so a student should learn the teacher's equivalence class, not its features. The organizing fact is that capability is the teacher's output function, a class invariant that factors through the quotient by the class action, so an objective recovers capability exactly when it is defined there. This makes absolute feature matching ill-posed, and admissible supervision a matter of targeting class invariants (Gram structure, CKA, principal subspaces) or aligning coordinates first, unifying feature matching, relational distillation, alignment, and grafting in one geometric account. We validate our framework on Qwen2.5 and Llama-3.1. A restoration study recovers a corrupted model's representation (CKA ~ 0.99) but not its capability, and an ablation isolates the cause: output-function (logit) matching drives capability, while matching hidden representations aligns geometry without restoring function. Recovery is confined to the corpus-covered region, and a graft study confirms that boundary overlap predicts transplant success but is necessary, not sufficient. 

---
# EPRA U-Net: An Efficient Pyramid Residual Attention Framework for Accurate Infarct Segmentation in Diffusion-Weighted MRI 

**Authors**: Hasan Ulutas, Muhammet Emin Sahin, Mustafa Fatih Erkoc, Esra Yuce, Turker Tuncer, Sengul Dogan, Serkan Kiranyaz  

**Link**: [PDF](https://arxiv.org/pdf/2607.03568)  

**Abstract**: Objective: Accurate identification of acute ischemic infarcts on diffusion-weighted magnetic resonance imaging (DWI) is a critical prerequisite for reliable lesion quantification and effective clinical decision support in the management of cerebrovascular events. Methods: This study presents EPRA U-Net (Efficient Pyramid Residual Attention U-Net), a task-specific integrated architecture for efficient and accurate infarct segmentation of DWI images. In the proposed architecture, an EfficientNet-based encoder was used as a hierarchical feature extractor with a minimized parameterization. In addition, a Residual-Recurrent (R2) block (recurrent unrolling step t = 2, following the original formulation) and Atrous Spatial Pyramid Pooling (ASPP) were integrated to enhance the performance of spatial dependency modeling. Additionally, a dual attention mechanism was incorporated to highlight lesion-related activations while concurrently enabling the suppression of extraneous background responses. To prioritize lesion detection consistent with clinical imperative, a Tversky loss function was adopted, emphasizing the sensitivity of detection over its specificity during the optimization process. Results: Experimental evaluations were conducted utilizing an in-house dataset comprising 167 patients with 4,895 DWI slices; subsequently, the performance of the proposed EPRA U-Net was assessed in comparison with state-of-the-art models, specifically UNet++, DeepLabV3+, and TransUNet. The experimental results suggest that EPRA U-Net attained superior performance, evidenced by a pixel-aggregated Dice of 0.8984, a per-sample Dice of 0.9469, an IoU of 0.8155, a Recall of 0.8887, a Lesion F1 of 0.9378, and an HD95 of 11.62 px. Furthermore, a clear reduction in the rate of missed lesions, specifically by 16%, 25%, and 29%, was observed when compared with UNet++, DeepLabV3+, and TransUNet, respectively. 

---
# Latent Clarity: Bridging World-Model Kinematics to Semantic Manifolds for Video Anomaly Anticipation 

**Authors**: Abu Anas Ibn Samad  

**Link**: [PDF](https://arxiv.org/pdf/2607.03558)  

**Abstract**: Continuous video anomaly detection is dominated by reactive Multiple Instance Learning (MIL) that collapses spatiotemporal features into scalar scores. We introduce PULS (Predictive Unified Latent Space), a continuous semantic world-model pipeline comprising two modules: a 490M-parameter KSD Bridge (Kinematic-to-Semantic Distillation) and a 16.8M-parameter Anticipatory State Predictor (ASP). The KSD Bridge maps V-JEPA 2 physical tensors into the 2048-d Qwen3-VL-Embedding-2B text-aligned hypersphere, trained on a subset of UCF-Crime. This translation alone yields a chunk-level AUROC of 0.8994 for UCF-Crime and 0.8162 for out-of-distribution XD-Violence without MIL or hierarchical fusion.
We introduce and validate the Latent Clarity Hypothesis: because JEPA's temporal predictor discards aleatoric pixel noise while preserving kinematics, anticipated future representations are more semantically separable than observed presents. The ASP sharpens these anticipated future latents, achieving 44.5% mean 14-way zero-shot VQA accuracy (exceeding observation baseline by +9.6 pp). Applying the ASP to Observation Tensors collapses accuracy to 7.3% (random chance), proving Anticipation and Observation occupy distinct sub-manifolds. A Triple-Track Lead-Time protocol with an L1-surprise gate yields a peak +8.9 pp anticipatory advantage at T-0.5s (p < 0.001, N = 1,000 permutation), separating physical anticipation from static scene priors. Zero-shot transfer to XD-Violence confirms that Newtonian-invariant kinematic representations generalize out-of-distribution. 

---
# Aligning Language Models with Selective Prediction 

**Authors**: Gaoxiang Luo, Yifan Wu, Sinian Zhang, Aryan Deshwal, Ju Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.03528)  

**Abstract**: Large language models (LLMs) are increasingly deployed as critical decision-making components in high-stakes real-world AI systems, rendering LLM reliability a foremost practical concern. In this paper, we focus on enhancing LLM reliability through selective prediction (SP), a strategy that allows an LLM to only predict for inputs where it is likely to be correct (i.e., coverage) and hence reduce the error rate (i.e., risk) on that portion of inputs -- flagging the remaining inputs for future human discretion. In other words, SP improves LLM reliability by balancing the risk-coverage trade-off and enabling seamless human-AI collaboration. To integrate SP into LLMs, we focus on the LLM post-training alignment stage and propose to align LLMs with SP performance metrics, in contrast with existing LLM alignment methods that focus primarily on correctness or calibration metrics. Specifically, we propose a novel alignment framework, Reinforcement Learning for Selection Reward (RLSR), which targets the area under the risk-coverage curve (AURC) -- a popular SP performance metric -- as its alignment objective. RLSR achieves substantially better risk-coverage trade-off compared to multiple alignment baselines on both in-domain and out-of-domain tasks. 

---
# AGL-1: The Enterprise AI Governance Layer as a Control Plane for Trusted Enterprise Intelligence 

**Authors**: Roopam W. Sure  

**Link**: [PDF](https://arxiv.org/pdf/2607.03516)  

**Abstract**: Enterprise artificial intelligence is moving from isolated experimentation toward operational dependency across copilots, retrieval-augmented generation systems, autonomous agents, and AI-enabled business workflows. As this transition accelerates, the primary enterprise challenge is no longer only model access or inference scale. It is governed intelligence operations: the ability to enforce authorization, preserve contextual lineage, control persistent memory, detect stale or conflicting knowledge, constrain agentic execution, and produce audit-ready evidence across distributed AI estates.
This paper introduces AGL-1, the Enterprise AI Governance Layer, as a vendor-neutral reference model for the control plane that should operate across foundation models, retrieval systems, orchestration frameworks, enterprise memory, policy engines, observability systems, tools, APIs, and business applications.
Building on governed knowledge-system principles introduced in GKS-5, AGL-1 generalizes the governance problem from retrieval-specific controls to full AI execution-path governance. It identifies recurring failure modes such as unauthorized retrieval, stale grounding, unmanaged memory, weak provenance, policy drift, fragmented observability, and uncontrolled autonomous execution. It then defines seven governance domains: identity-aware retrieval, policy enforcement, provenance management, memory governance, knowledge integrity monitoring, agentic execution control, and trust observability.
The central claim is that durable enterprise value from AI will increasingly depend on the ability to govern intelligence at scale. In complex enterprises, trust is not a property of the model alone. It is a property of the system around the model: identity, knowledge, policy, memory, tools, human oversight, and evidence working together as a managed control plane. 

---
# CAGE-1: Control, Assurance, and Governance Evaluation for Enterprise Agentic AI 

**Authors**: Roopam W. Sure  

**Link**: [PDF](https://arxiv.org/pdf/2607.03510)  

**Abstract**: Enterprise artificial intelligence is moving from experimentation into operational workflows. Early programs focused on model access and retrieval-augmented generation, but enterprises are now beginning to deploy agents that plan, retrieve, remember, call tools, update systems, and coordinate work across applications.
This changes the evaluation problem. Leaders are no longer asking only whether an answer is accurate or fluent. They need to know who authorized an action, which policy applied, whether evidence was current, whether memory was valid, whether a tool call was permitted, whether the decision can be replayed, and whether the agent can be stopped before it creates business impact.
This paper introduces CAGE-1: Control, Assurance, and Governance Evaluation for Enterprise Agentic AI. CAGE-1 is an evaluation framework for deciding whether enterprise agents are ready for deployment. It evaluates authority, policy enforcement, retrieval quality, memory integrity, tool safety, auditability, human oversight, conflict handling, safe failure, Prebind Assurance, operational readiness, and business fitness.
CAGE-1 introduces Prebind Assurance to describe the evaluated ability to prove that an agentic action is controlled before it becomes binding, effective, or operationally consequential. The framework tests whether a proposed action is admitted, held, narrowed, refused, escalated, quarantined, or made non-effective before protected consequence forms. 

---
# Reading Between the Dots: Decoding Hidden Computation across Filler Tokens 

**Authors**: Kaley Brauer, Claudio Mayrink Verdun, Samuel Marks  

**Link**: [PDF](https://arxiv.org/pdf/2607.03502)  

**Abstract**: Frontier LLMs can perform multi-step reasoning over content-free filler tokens like dots or counting sequences, producing correct answers with no visible chain-of-thought (CoT). This is a limit case for behavioral oversight, where surface tokens carry no information about the underlying reasoning. But hidden from the output is not the same as hidden from us. On four task families (fact retrieval, parallel numeric composition, string manipulation, and in-context computation), two open-weights frontier models (DeepSeek V3, Kimi K2) compute over filler tokens in a structured, legible way: attention routes the question through the filler region to the answer, logit-lens readouts show retrieved facts emerging early and their composition crystallizing in late layers, and KV-cache transplants at filler positions causally swap outputs between examples. We introduce an unsupervised decoding pipeline that takes only hidden states as input and recovers intermediate values with 80-95% accuracy (best LLM judge) across both models and all four tasks, without ground-truth labels or training. Hidden computation that defeats behavioral CoT monitoring is, on these tasks, directly readable from the residual stream, suggesting monitorability is a property of the model's full computational trace, not just its surface tokens. 

---
# STRATOS: Bridging the Symbolic-to-Numeric Gap in Spatio-Temporal Text-to-SQL for Meteorological Data 

**Authors**: Yi Zhang, Farhad Nooralahzadeh, Jonathan Fürst, Fabio Scherrer, Antonis Bezes, Vassiliki Kotroni, Kurt Stockinger  

**Link**: [PDF](https://arxiv.org/pdf/2607.03501)  

**Abstract**: Copernicus, the European Union's Earth observation program, produces petabytes of Earth observation and climate data, offering immense potential for research, policy, and applications. However, access to these datasets requires advanced programming skills and familiarity with domain-specific formats such as NetCDF or GRIB. Moreover, general-purpose Text-to-SQL systems fail when applied naively to the meteorological domain due to a profound ``Symbolic-to-Numeric'' gap. To overcome these limitations, we present an end-to-end Text-to-SQL framework specifically engineered for real-world, scalable meteorological data exploration. Our system intercepts natural language to resolve spatial and semantic ambiguities \textit{before} SQL generation. We design STRATOS, a Spatio-Temporal Resolution Agent for Text-to-SQL to dynamically bridge the symbolic-to-numeric gap by mapping fuzzy concepts to a localized ontology and resolving spatial entities via external knowledge bases. Further, our complexity-aware query rewriter rewrites expensive spatial predicates, reducing execution times from hours to seconds. Last, we introduce the STRATOS Evaluation Workload, comprising 7,520 complex query pairs explicitly designed by domain experts to test scalability and symbolic-to-numeric translation across challenging spatio-temporal dimensions previously unexplored by Text-to-SQL systems. 

---
# Towards Diverse and Comprehensive Benchmarks for Mutual Information Estimation 

**Authors**: Alberto Foresti, Ivan Butakov, Alexander Tolmachev, Giulio Franzese, Alexey Frolov, Pietro Michiardi  

**Link**: [PDF](https://arxiv.org/pdf/2607.03487)  

**Abstract**: Mutual information (MI) estimation is a central problem in machine learning and statistics; however, existing benchmarks typically evaluate estimators on simplified, low-dimensional distributions, leaving their performance on complex, realistic data largely unexplored. We address this gap with a comprehensive benchmarking framework grounded in a unified copula-theoretic perspective that subsumes existing benchmarks as special cases. Within this framework, we propose two complementary families of tests: a copula-first family that systematically varies ground-truth MI, dimensionality, and marginal complexity using synthetic and flow-based transformations; and a marginals-first family that couples real-world image data with controlled dependency structures, extending the classic same-class-pairing paradigm. We use this suite to extensively evaluate three classes of estimators: non-parametric, discriminative, and generative. Contrary to prevailing assumptions, our results indicate that there is no universal winner: each category can systematically outperform all other estimators under specific setups. By analyzing these cases, we identify fundamental estimation barriers and propose new tests that more effectively stress these specific limitations. We share the open source code at this https URL. 

---
# CaresAI at SMM4H-HeaRD 2026: Predicting TNM Staging 

**Authors**: Joseph Itopa Abubakar, Jorge Jarme, Favour Igwezeke, Mary Adewunmi  

**Link**: [PDF](https://arxiv.org/pdf/2607.03466)  

**Abstract**: This study aims to predict Tumor, Node, and Metastasis (TNM) stage labels independently, with the Cancer Genome Atlas (TCGA) pathology report as the sixth shared task of SMM4H-HeaRD 2026. The problem is framed as three multi-label classification tasks. We explore both classical and deep learning approaches using Term Frequency-Inverse Document Frequency (TF-IDF) features and embeddings from ClinicalBERT, BioBERT, and PubMedBERT. These representations are used with Logistic Regression (LR), Light Gradient Boosting Machine (LightGBM), Feed-Forward Neural Networks (FFNN), and Wide Residual Networks (WRN). Our results show that individual embeddings perform similarly to the TNM label classification, while their combination improves its predictive ability. WRN achieves AUROC scores of 0.839 (T), 0.8502 (N), and 0.803 (M) with F1-scores of 0.622, 0.702, and 0.9337, respectively, for the training phase. LightGBM with TF-IDF performs best with AUROC scores of 0.9368 (T), 0.9524 (N), and 0.8311 (M) and F1-scores of 0.7559 (T), 0.7384 (N), and 0.7017 (M) during the training phase. Furthermore, the result of the Codabench for the test sets indicates a Macro-F1 score of 0.978, 0.957, and 0.879 for the T, N, and M categories respectively for test set 1; while test set 2 records a Macro-F1 score for T, N, and M is 0.807, 0.767, 1.0 respectively. However, performance declined during the evaluation phase of the test sets, a drop from 0.938 to 0.858 of test set 1 to 2, for the Macro-F1 score across all stages; suggesting limitations in model generalizability, sensitivity to class imbalance, and challenges in processing lengthy clinical documents. Although this study provides an efficient baseline model and a reproducible pipeline, further optimization and validation are required before it can be considered suitable for use in a real-world clinical setting. 

---
# Best-of-Better-$N$: Generating Pre-Aligned Responses with In-Context Learning 

**Authors**: Eric Lei, Hsiang Hsu, Chun-Fu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.03453)  

**Abstract**: Inference-time alignment methods, such as Best-of-$N$, offer a flexible alternative to training-based alignment by using reward models to select high-quality responses generated by a reference LLM. However, the efficacy of these methods is inherently limited by the response quality: if the reference LLM assigns negligible probability to high-reward responses, no selection strategy will succeed in finding aligned outputs. In this work, we propose Best-of-Better-$N$ (BoBN), an in context learning-based generation framework to address this challenge. Our method utilizes retrieval from high-reward examples relevant to the input query and task. Crucially, we introduce a restyling step where retrieved responses are rewritten by the reference LLM to align with the target task's format and style. These restyled examples are used in-context to shift the sampling distribution toward the high-reward region. We analytically characterize how in-context learning shifts the output distribution of pretrained transformers toward the high-reward region, resulting in provable benefits on the target task. We then evaluate BoBN on safety alignment and mathematical reasoning benchmarks across several reference LLMs. BoBN's higher-quality responses enable better performance to be achieved when the number of responses $N$ is fixed, and smaller $N$ required to achieve a target performance. 

---
# SkillOpt-Lite: Better and Faster Agent Self-evolution via One Line of Vibe 

**Authors**: Yifei Shen, Bo Li, Xinjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.03451)  

**Abstract**: While skill optimization for autonomous agents has gained traction, existing methods rely on complex pipelines. This leaves a fundamental question unaddressed: What constitutes a minimal viable pipeline for skill optimization, where every component is justified by theory or empirical necessity? We formalize skill optimization via Zeroth-Order (ZO) optimization, mapping classical counterparts (central difference, trust regions) to recent literature. Noting that unlike blind numerical perturbations in classical ZO, skill trajectories serve as interpretable debugging feedback. Grounded in Claude Code philosophy and PAC learning, we establish three principles for convergence and generalization: file-system-based trajectory exploration, consensus attribute mining, and independent validation gating. Eliminating redundancies, we propose SkillOpt-Lite. It accelerates convergence and outperforms full SkillOpt: improving LiveMath by +8.8 points on GPT-5.5 and +25.4 points on GPT-5.4-nano, allowing the nano model to surpass standard GPT-5.4 optimized by SkillOpt. Finally, we integrate our framework into production coding agents like VSCode Copilot, enabling developers to evolve agent skills via one line of vibe. Because our framework treats all agent components simply as standard editable code, this minimal pipeline naturally generalizes to full harness optimization (HarnessOpt). On SpreadsheetBench, HarnessOpt enables GPT-5.4-nano to achieve 0.7758 accuracy, outperforming the larger GPT-5.5 running standard pipelines (0.7620). Code is available at this https URL. 

---
# HiMe: Hierarchical Embodied Memory for Long-Horizon Vision-Language-Action Control 

**Authors**: Li Ji, Siyin Wang, Pengfang Qian, Xiaopeng Yu, Yihai Tian, Zhaoye Fei, Jingjing Gong, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2607.03449)  

**Abstract**: Current Vision-Language-Action (VLA) models excel at robotic manipulation but often struggle with non-Markovian tasks requiring long-term memory and reasoning due to their reliance on immediate observations. Existing solutions face a ''frequency-competence paradox,'' where stronger reasoning models are too slow for real-time control, while faster models lack sufficient reasoning capabilities. To resolve this architectural misalignment, we propose HiMe, a Hierarchical Embodied Memory framework that decouples embodied intelligence into a high-frequency Executor for execution, a Sentry for working memory, and a Planner for long-term strategy. We also introduce a dynamic knowledge system based on cross-modal semantic schemas and active management mechanisms, allowing robots to maintain memory plasticity through ''Add, Update, and Delete'' operations. This hierarchical design effectively balances the conflict between real-time execution and slow thinking planning, significantly improving success rates in long-horizon tasks. Experiments demonstrate that this approach not only outperforms flat memory baselines but also exhibits the novel ability to self-correct its internal knowledge based on human preferences. 

---
# TRIAGE: Trustworthy Retrieval Instrumentation And Graph Evaluation 

**Authors**: Axel TahmasebiMoradi, Lucas Schott, Martin Royer  

**Link**: [PDF](https://arxiv.org/pdf/2607.03447)  

**Abstract**: Knowledge graphs (KGs) that underpin Graph-based Retrieval-Augmented Generation (Graph-RAG) are increasingly built automatically by LLM-driven extraction rather than curated by experts. Proper evaluation would require instrumenting all pertinent stages: extraction, graph construction, and inference, coherently enough to localize failures, so that a failure at one stage is not discovered as a wrong answer at the end. We introduce TRIAGE, a stage-aware instrumentation framework for automated, document-grounded graph-RAG that asks not only whether the underlying graph can be trusted but at what cost it can be queried. TRIAGE attaches stage-specific, independently interpretable metrics to three stages: the KG Implementation (triple confidence, source coverage, and schema and canonicalization checks), the KG Validation by expert (graph-level structural quality, with correctness and completeness computed only as offline calibration when a reference is available), and the KG Usage (retrieval coverage, faithfulness, and retrieval cost); the deployed metrics need no gold annotations, the gold-requiring ones serving only as offline calibration. At usage time these metrics form a diagnostic chain of necessary conditions whose first broken link localizes the failure, and the diagnosis maps to the stage levers that can remedy it: extraction, graph and schema, or retrieval. TRIAGE is a theoretical framework with a proof of concept and a reproducible evaluation protocol. 

---
# No Time Like the Present: Agentic Test-Time Training for LLM Agents 

**Authors**: Yanbo Wang, Jinhua Hao, Yuze Shi, Kun Yuan, Ming Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.03441)  

**Abstract**: LLM agents often degrade over long episodes: as trajectories grow, they revisit explored states, repeat failed actions, and lose strategies that previously worked. Test-time training (TTT) offers a way to adapt model weights to the evolving task state, but existing LLM TTT methods largely adapt once to a fixed input. We study continuous TTT in multi-turn agent episodes, where each update changes the policy that generates later training text. This creates a self-training loop that helps when new trajectory information appears, but can amplify drift when the agent gets stuck and repeatedly trains on similar text. We find that update-text repetition distinguishes these regimes and introduce Agentic Test-Time Training (aTTT), a token-level reweighting method that downweights the loss on tokens appearing in repeated $n$-grams from prior updates while leaving novel tokens fully weighted. To run such updates inside live episodes, we build a concurrent serving system using vLLM's runtime LoRA API, limiting overhead to 1.9$\times$ the no-TTT cost. aTTT improves success by up to 5.0 points on ALFWorld and 4.9 points on SWE-bench Lite. The gains concentrate where models already have task competence but drift over long trajectories, suggesting that aTTT mainly preserves existing competence rather than teaching new abilities. 

---
# Amortising Bayesian Experimental Design for Sequential Information Gathering in LLMs 

**Authors**: Jakob Hartmann, James Harvey, Jhonathan Navott, Erik Y. Wang, Luckeciano C. Melo, Flaviu Cipcigan, Cheng Zhang, Alessandro Abate  

**Link**: [PDF](https://arxiv.org/pdf/2607.03426)  

**Abstract**: Large language models (LLMs) exhibit strong reasoning and world-knowledge capabilities, yet often struggle to gather information effectively across the multi-turn interactions required in sequential decision-making settings. We introduce Amortised Sequential Information Gathering (ASIG), a fine-tuning approach that amortises Bayesian Experimental Design (BED) into LLM policies via a multi-turn extension of Group Relative Policy Optimisation with an Expected Information Gain reward. Evaluated on the 20 Questions task, ASIG more than doubles the success rate of the 7B base model and reduces inference cost by over $25\times$ relative to BED-LLM, a competitive inference-time baseline. Applied to MediQ, a medical diagnosis benchmark unseen during training, ASIG improves information-seeking performance at the 7B scale, suggesting that the learned strategies can transfer out of distribution. Our findings show that amortising BED into LLM policies provides an effective and computationally efficient approach to sequential information gathering. 

---
# Securing Multi-Tool AI Agent Chains With Dynamic, Real-Time Compositional Policies 

**Authors**: Chris Schneider, Kriti Faujdar, Philipp Schoenegger, Ben Bariach  

**Link**: [PDF](https://arxiv.org/pdf/2607.03423)  

**Abstract**: Modern AI agent implementations such as frontier coding agents chain multiple tools at runtime that create a security surface that per-tool guardrails are unable to address, as individually permitted tools can violate organizational policies when composed. We propose the Dynamic Security Control Compositor (DSCC), a two-phase approach to compositional security for multi-tool agent chains. In Phase 1, at session checkout, a Most Restrictive Set (MRS) algorithm composes per-tool security policies into a single effective policy with a formal monotonicity invariant that extending a chain can only tighten the result, blocking incompatible combinations before any tool executes. Outputs of any tool call propagate their classification constraints into a session-level taint state, so subsequent invocations must satisfy the most restrictive constraints seen so far. In Phase 2, at runtime, the system tracks the sensitivity of data the agent touches through a monotonic taint state and revokes the session if the accumulated exposure would make a subsequent tool call a policy violation. Together, these phases provide defense in depth, where static composition prevents unsafe chains from starting, and runtime taint tracking catches violations that emerge from the specific data used. We provide a reference implementation on 32 tools governed by 16 NIST SP 800-53 aligned policies and evaluate it under two composition modes. In the default clearance mode, permitted combinations are partitioned into classification-level clusters, blocking 79.2% of policy pairs and 95.5% of triples. The alternative taint mode admits mixed-classification chains within the exfiltration boundary, blocking 42.5% and 60.5% respectively. We discuss the governance implications for organizations deploying multi-tool agents, including the utility-security tradeoff and the changes needed to operationalize chain-aware policies. 

---
# DETECT-3B-Omni is Agnostic of Content and Demographics 

**Authors**: Nicolas M. Müller, Aditya Tirumala Bukkapatnam, Dominik Schnieders, Zohaib Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2607.03418)  

**Abstract**: A trustworthy and GDPR-compliant deepfake audio detector must base its decisions on acoustic artifacts, not on what is being said or who is speaking. We present a large-scale study of semantic independence for Resemble AI's detector, DETECT-3B-Omni. Using 10,240 audio samples from diverse US English speakers across 30 states, generated through 8 different AI voice-cloning systems, we test whether detection accuracy depends on spoken content (benign versus malicious), speaker gender, speaker age, or speaker region. Using equivalence testing, our results show that the accuracy difference between any two of these groups is at most 2 percentage points, at 99% confidence. The detector therefore identifies AI-generated audio with equivalent accuracy regardless of what the audio says or who the speaker is. 

---
# The S-ICDF Dataset: Sionna-Simulated Dynamic Interference Characterization and Direction Finding 

**Authors**: Christian Wielenberg, Lucas Heublein, Jonathan Ott, Alexander Mattick, Nisha L. Raichur, Jonas Pirkl, Lukas Schelenz, Tobias Feigl, George Yammine, Christopher Mutschler, Felix Ott  

**Link**: [PDF](https://arxiv.org/pdf/2607.03411)  

**Abstract**: Jamming and spoofing threaten wireless and satellite navigation by disrupting or manipulating radio frequency (RF) signals, undermining availability, integrity, and trust. Robust interference monitoring (i.e., detection, classification, characterization, and direction finding) is therefore essential to identify and localize anomalous signals. While machine learning (ML) promises improved performance in complex environments, its development and validation depend on large-scale datasets that capture realistic signal and channel variability. Collecting such data in the real world is difficult because intentional jamming is illegal and ground-truth attribution is confounded by propagation, hardware, and environmental effects. To address this gap, we create and publish S-ICDF, a large-scale indoor interference dataset generated with Sionna, a GPU-accelerated simulation library for physical-layer wireless communications. S-ICDF covers 102 interference configurations, including diverse antenna array patterns, bandwidths, and simulation settings such as noise level and reflection depth. We further provide baseline results by benchmarking S-ICDF with classical estimation and direction finding (DF) methods (MUSIC, ESPRIT, and CAPON) and with modern ML approaches. The dataset is publicly available at: this https URL 

---
# Spectral Signatures of Large Language Models 

**Authors**: Zhuoying Zhang, Ishan V. Prasad, Yuanzhe Hu, Zihang Liu, Hengrui Luo, Pu Ren, Yaoqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.03377)  

**Abstract**: The rapidly growing repository of publicly available large language models (LLMs) presents significant challenges for systematic management and quantification at scale, such as model lineage tracing, licensing, and evaluation. However, task-specific benchmarks are insufficient for this setting, as LLMs differ widely in architectures, scales, and training procedures. To address this challenge, we adopt spectral shape-based metrics for managing and quantifying LLMs based on Heavy-Tailed Self-Regularization theory. Our approach uses the shape information of the weight empirical spectral density as a compact spectral signature of each model. This signature captures intrinsic properties of pretrained models and remains robust during post-training, making it suitable for model-level analysis. In addition, this metric is data-free, computationally-efficient, and scale-invariant, enabling large-scale analysis in practice. Moreover, we curate a large and diverse model corpus consisting of major open-source LLM families, and use it to systematically benchmark spectral and non-spectral metrics across models and downstream tasks. We show that our spectral signature supports the tracking of the model lineage, the unsupervised clustering of similar models, and the quantification of the model performance. Overall, the proposed spectral signature provides a meaningful proxy for broad performance trends across LLMs, enabling efficient organization, comparison, and analysis of large model collections. 

---
# Brand-as-Memory: Vision-Language Models Encode Causal, Mechanistically Localizable Credibility Priors for News Sources 

**Authors**: Chih-Ting Liao, Xin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2607.03365)  

**Abstract**: Vision-language models (VLMs) increasingly read news and web content as images, where the publisher's identity is visually present. We show that VLMs carry a strong source-credibility prior keyed on outlet identity, and study it along three axes. (i) Cross-model benchmark. We introduce CueTrust, a cross-model diagnostic that measures which surface source cue overrides an article's content evidence via a Source-Override Index (SOI). Across seven VLMs and five cues, the vulnerability profile is model- and scale-dependent, and the override is outlet-identity-specific and encoding-invariant, firing from the masthead name, the logo image, or the bare domain, but not from a named author, in-text authority, or page layout (clean negative controls). (ii) Mechanistic account. For the brand cue, we give a full mechanistic account: swapping only the masthead moves credibility across an approximately 11 log-odds range that tracks professional ratings (rho = 0.88 with Media Bias/Fact Check). The prior is dual-coded (name and logo), strengthens with scale, is causally formed at layers 19-21, carried by interpretable seed-stable sparse-autoencoder features, and recurs at the same relative locus in a second model family. It overrides content (about 1.8x) as a signal-magnitude effect within a shared pathway, not a privileged route. Steering the localized direction selectively reduces the override (41% reduction) and generalizes to held-out outlets, confirming the prior is causally used, not merely decodable. Deployed VLMs may thus defer to source identity over the evidence in front of them, a reliability failure we can measure across models, localize, and causally probe. We release the stimulus suite and CueTrust. 

---
# LLM-Enhanced Hierarchical Heterogeneous Graph Representation Learning for Malicious Python Package Detection 

**Authors**: Hang Gao, Xiaoyu Chen, Baoquan Cui, Zhen Tang, Peng Qiao, Fengge Wu, Jian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.03350)  

**Abstract**: Malicious Python packages have become a major threat to software supply chain ecosystems due to the widespread adoption of open-source repositories such as PyPI. Existing learning-based detection methods struggle to capture the hierarchical organization and heterogeneous interactions among different program entities. Although Large Language Models (LLMs) have demonstrated strong capabilities in code understanding and semantic reasoning, they are rarely integrated with structural program representations for fine-grained malicious behavior analysis. In this paper, we propose an LLM-enhanced hierarchical heterogeneous graph representation learning framework for malicious Python package detection. The framework constructs a hierarchical heterogeneous code graph that explicitly models heterogeneous code entities and different types of structural dependencies. LLMs are further leveraged to infer function-level semantic roles, introducing an additional layer of semantic heterogeneity. Based on this graph, we develop a hierarchical heterogeneous graph neural network that performs type-aware message passing over different node and edge categories, effectively modeling malicious behavior propagation for accurate package-level classification. The framework also incorporates a function-level attribution mechanism which, combined with LLM reasoning, automatically identifies suspicious functions and localizes fine-grained malicious behaviors without human expert intervention. Extensive experiments on real-world datasets show that our framework consistently outperforms traditional machine learning methods, graph-based detectors, and state-of-the-art LLMs across packages with varying sizes and dependency complexities, while providing accurate, robust, and interpretable malicious behavior localization. 

---
# PedestrianDiffusion: Multimodal Generative Denoising and Dense State Estimation for Inertial Navigation 

**Authors**: I-Hao Lu, Dongsoo Han  

**Link**: [PDF](https://arxiv.org/pdf/2607.03349)  

**Abstract**: The accuracy of consumer-grade inertial navigation is bottlenecked by the stochastic noise of Micro-Electro-Mechanical Systems (MEMS). Traditional deterministic neural architectures often succumb to ``estimation jittering,'' sacrificing high-frequency kinematic fidelity for numerical stability. We propose PedestrianDiffusion, a multimodal spectral-domain generative framework reformulating dense 6D state estimation as a continuous conditional denoising process. By operating in the frequency domain, our formulation bounds the spectral covariance, acting as a mathematical preconditioner to stabilize the reverse diffusion trajectory. Furthermore, we introduce a zero-shot semantic conditioning mechanism leveraging vision-language embeddings as categorical priors to generalize across heterogeneous sensor noise profiles. To address the computational intractability of generative tracking, we deploy a single-step deterministic probability flow ODE solver ($T=1$). This yields high-capacity asynchronous batch trajectory refinement, establishing the viability of generative architectures for asynchronous batch trajectory refinement on edge hardware. Extensive evaluations on the OxIOD, RIDI, RoNIN, and TLIO benchmarks demonstrate that PedestrianDiffusion achieves state-of-the-art performance, exhibiting unprecedented robustness to impulse perturbations and coupled 6D kinematic drift. This work provides a rigorous algorithmic blueprint for next-generation Neural Inertial Measurement Units (N-IMUs). 

---
# Efficient Decentralized Multi-task Dataset Valuation via Model Merging 

**Authors**: Mohammadsajad Alipour, Mohammad Mohammadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2607.03346)  

**Abstract**: Accurate and efficient dataset valuation is essential for enabling fair and transparent data marketplaces, especially when multiple contributors provide data for training multi-task models. Most existing valuation methods, however, are limited to single-task settings, overlooking scenarios where a buyer aims to optimize performance across multiple downstream tasks. Moreover, traditional valuation approaches, such as Shapley-based or retraining-based methods, are computationally expensive and poorly suited for decentralized environments without a trusted central coordinator and with strict privacy constraints. We propose DMVM (Decentralized Multi-task Valuation via Model Merging), a novel framework that bypasses retraining and data sharing by leveraging task arithmetic to infer dataset contributions directly from model combinations. Instead of retraining or sharing raw data, DMVM quantifies how models trained on different datasets combine in parameter space to infer each dataset's marginal utility across multiple tasks. This formulation yields a valuation process that is scalable, computationally efficient, and explicitly aligned with multi-task generalization behavior. To support decentralized deployment, we introduce a secure aggregation protocol that enables collaborative valuation without revealing individual model parameters or private data. We also provide theoretical error bounds characterizing the approximation quality of DMVM and validate our framework through comprehensive experiments on computer vision and natural language processing tasks. 

---
# FedAvg for HAR: Exploring the Tradeoff Between Personalized and Generalization Accuracy 

**Authors**: Andrea De Luna, Susanna Peretti, Chiara Contoli, Alessandro Bogliolo  

**Link**: [PDF](https://arxiv.org/pdf/2607.03334)  

**Abstract**: The federated learning (FL) paradigm fosters distributed pervasive computing combined with artificial intelligence techniques, allowing for optimized data usage and improved mitigation of privacy concerns. Indeed, model training occurs on the client's local devices, and model parameters are subsequently shared with a centralized server. However, there is a need to find a tradeoff between models' personalization and generalization capabilities. In this paper, we design and implement several testing scenarios devoted to evaluating and comparing the centralized, local, and federated paradigm performances. We also design and implement a scenario that emulates a change in clients' data. We then present experimental results of the FedAvg algorithm applied to the Human Activity Recognition (HAR) domain to understand the trade-off between personalized and generalized accuracy. Results show that, although FedAvg confirms a higher degree of personalization capabilities while keeping a high degree of generalization with respect to the traditional centralized learning, this result is not so obvious under stressful conditions, such as when varying class distribution over clients. 

---
# SPORK: Self-Speculative Forking to Accelerate Agentic LLM Inference 

**Authors**: Huajun Bai, Weiwei Lv, Huichuan Zheng, Youyou Lu, Jiwu Shu  

**Link**: [PDF](https://arxiv.org/pdf/2607.03333)  

**Abstract**: LLM agents are becoming a common interface for research, coding, and question answering, yet their Thought-Action-Observation loop is often serial: the model reasons, emits a tool call, then idles the GPU until the result returns. This wait consumes 16-37% of wall time in our workloads and 35-61% in prior reports. Speculative tool execution can hide this wait, but existing systems need auxiliary predictors, historical traces, or static workflow graphs, leaving a gap for training-free, day-one deployment. We observe that the model can be its own predictor: a probe forked at the start of generation predicts Qwen3-32B's upcoming tool name with 74.6-99.6% accuracy across five benchmarks. We present SPORK (Self-sPeculative fORKing), a training-free controller that dispatches the speculated tool call early, overlapping its execution with the remaining chain-of-thought decode. A cost model captures when speculation breaks even, and each component improves one of its terms: a prefix-cache fork cuts probe cost, a confidence gate filters mispredictions, and partial-token accept turns rejected probes into speculative-decoding drafts. On acceptance, the tool result is ready when reasoning ends; on rejection, SPORK falls back to serial execution with no correctness penalty. On real-tool benchmarks, SPORK cuts Qwen3-32B's GAIA P95 by 18% (131.9 to 108.1 s); the mechanism holds across model sizes from 4B to 32B and across dense and mixture-of-experts models, with task accuracy within 1 pp of baseline or better wherever measured. SPORK deploys as a thin controller over standard completion APIs (no retraining, no auxiliary models, no offline traces) and is orthogonal to token-level speculative decoding. SPORK is open source at this https URL. 

---
# From Judgments to Issues: Structured Extraction of Legal Reasoning with Citation-Hallucination Control 

**Authors**: Giovanni Piccioli, Alessia Fidelangeli, Piera Santin, Pierpaolo Vivo  

**Link**: [PDF](https://arxiv.org/pdf/2607.03325)  

**Abstract**: We present an automated pipeline that decomposes Italian tax-court judgments into individual legal issues and extracts, for each issue, a structured XML representation grounded in the IRAC framework and the legal syllogism. The pipeline targets a corpus of approximately $330{,}000$ first- and second-instance decisions of the Italian tax courts and is built around a capable yet cost-efficient general-purpose model (DeepSeek V3), a choice driven by the need to process several hundred thousand documents at a sustainable cost. To address the well-documented unreliability of large language models on legal citations, we couple the extraction step with an automatic hallucination-detection filter that compares the references produced by the model with those identified in the judgment text by a dedicated parser (Linkoln), normalised to standard identifiers (URN-NIR, ECLI, CELEX). We validate the pipeline on $50$ judgments annotated by two PhDs in tax law, computing inter-annotator agreement and LLM-vs-expert agreement on both issue extraction and legal citations, together with a stand-alone evaluation of the hallucination filter. To the best of our knowledge, this is the first issue-level, expert-validated structured extraction pipeline with hallucination control for Italian tax-court decisions, and it provides a concrete starting point for downstream applications such as issue-level retrieval, citation-network analysis, and the construction of large-scale datasets of legal reasoning. 

---
# Hierarchical Multi-Agent Reinforcement Learning for Carbon-Aware AI Data Centers in Power Distribution Systems 

**Authors**: Hyunsoo Lee, Panggah Prabawa, Dae-Hyun Choi, Joongheon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.03324)  

**Abstract**: Eco-friendly energy management for artificial intelligence data centers (AIDCs) is crucial because of the significant increase in energy consumption-induced carbon emissions from AIDCs resulting from the rapid expansion of AI applications. This paper proposes a hierarchical carbon-aware multi-agent reinforcement learning (CA-MARL) framework for robust and efficient operations of AIDCs under uncertainties while ensuring low-carbon operation of power distribution systems. The framework comprises a workload manager (WM) agent and multiple local AIDC agents trained using a multi-agent transformer method, corresponding to a global AIDC aggregator and a local AIDC operator, respectively. Leveraging AIDC operation data along with nodal carbon intensity (NCI) calculated from the carbon emission flow-integrated distribution system operator problem, the WM agent spatially allocates AI training and inference jobs among all AIDCs. Based on the jobs allocated from the WM agent and NCI information, each AIDC agent schedules economical and eco-friendly operations of the AIDC by performing the following tasks: i) temporal shifting of training jobs, ii) spatial allocation of training graphics processing unit (GPU) blocks and inference GPUs within the AIDC, and iii) control of the supply air temperature of the cooling system. The effectiveness of the proposed framework was assessed using an IEEE 33-node power distribution system. 

---
# Is Agentic Code Review Helpful? Mining Developers' Feedback to CodeRabbit Reviews in the Wild 

**Authors**: Hong Yi Lin, Mingzhao Liang, Kla Tantithamthavorn, Patanamon Thongtanunam  

**Link**: [PDF](https://arxiv.org/pdf/2607.03316)  

**Abstract**: Agentic code review, where autonomous agents provide code review comments on pull requests, is increasingly integrated into development workflows, yet there is limited empirical evidence on how developers respond to such comments in practice. In this paper, we present an empirical study of agentic code reviews using CodeRabbit as a case study. Through an empirical study of 31,073 pairs of code reviews and developer feedback from 10,191 pull requests across 239 GitHub repositories, our results show that agentic reviews receive mixed reception: 36.4% were accepted and 7.3% triggered discussion, while 56.3% were rejected. Rejections were primarily associated with invalid suggestions that were false positives, redundant, or out of scope, as well as misalignment with developer intent and coding practices. We further found that agentic reviews tend to focus more on functional concerns than evolvability-related comments, yet they were more likely to be invalid. To improve effectiveness in review practices, we explored various LLM-based approaches for predicting review rejection. We found that lightweight learning-based methods achieve up to 76% F1 score, suggesting learnable patterns exist between code reviews and their corresponding feedback. Our results highlight the current state of CodeRabbit's agentic code reviews, showing opportunity gaps for improvement, as well as shortcomings hindering its effectiveness. 

---
# A harmonised dataset for Earth system foundation models 

**Authors**: Carlos Rodriguez-Pardo, Massimo Tavoni  

**Link**: [PDF](https://arxiv.org/pdf/2607.03298)  

**Abstract**: Foundation models for Earth systems have so far been trained primarily on physical climate and weather data, with limited representation of the human systems that both drive and respond to environmental change. The lack of a unified global training resource that combines climate, land, ocean, cryosphere, infrastructure, hazards, and socioeconomic data on a common grid hinders progress toward truly multimodal Earth system foundation models. We present WorldTensor, a harmonised global dataset that aligns hundreds of environmental and socioeconomic variables to a standardised 0.25$^\circ$ spatial grid and annual temporal framework. WorldTensor integrates reanalysis products, remote sensing, emissions inventories, land use reconstructions, hydrological observations, infrastructure and hazard datasets, and socioeconomic indicators within a single representation designed for machine learning workflows. To build the dataset, we regridded inputs across heterogeneous native resolutions and projections, rasterised point and vector datasets into spatially meaningful gridded fields, and reconciled temporal coverages ranging from daily observations to sparse multiyear socioeconomic snapshots. All outputs are distributed as NetCDF files with standardised coordinates, variable metadata, and a common CF metadata convention. WorldTensor provides a reproducible resource for training and evaluating foundation models that learn coupled dynamics across environmental and human systems at planetary scale. 

---
# Semantic Segmentation-Driven Image-Level Diagnosis of Liver Cancers in Hematoxylin and Eosin Histopathology Images 

**Authors**: Ivica Kopriva, Dario Sitnik, Arijana Pacic, Karolina Krstanac, Irena Veliki Dalic, Marijana Popovic Hadzija  

**Link**: [PDF](https://arxiv.org/pdf/2607.03253)  

**Abstract**: As hematoxylin & eosin (H&E) staining constitutes the primary entry point in routine diagnostic workflows, computer-aided diagnosis from whole-slide H&E images is of particular clinical relevance. However, substantial variability in specimen preparation, staining protocols, and scanning conditions, together with inherent uncertainty in expert pixel-level annotations, makes automated analysis of H&E-stained images challenging. In this study, we propose a semantic segmentation-based framework for image-level diagnosis, grounded in the clinically motivated assumption that each histopathological image corresponds to a single cancer type. Image-level predictions are obtained by assigning the class of the dominant pixel-level label in the segmentation output. To ensure clinical relevance, we adopt the nnU-Net architecture and train it on a publicly available dataset collected in our study with pixel-level annotations for three liver cancer types: hepatocellular cacrcinoma (HCC; 55 images from 30 patients), cholangiocellular carcinoma (CCA; 55 images from 29 patients), and colorectal metastatic adenocarcinoma (CMA; 60 images from 30 patients). Annotations were independently provided by four pathologist. We hypothesize that the combination of stain normalization and semantic segmentation mitigates domain shift and reduces sensitivity to annotation noise. Five-fold cross-validation yielded balanced accuracy of 0.975 (HCC), 0.950 (CCA), and 1.000 (CMA), comparable to results obtained with immunohosthochemical staining and superior to several deep learning models trained on patch-level annotations. The proposed framework has the potential to support pathologists in prioritizing immunohistochemical marker selection, thereby reducing diagnostic costs and turnaround time. Integration with immunohistochemical findings improve overall diagnostic reliability. 

---
# Unbiased Alignment for Large Language Models with Noisy Preferences 

**Authors**: Jialiang Wang, Xianming Liu, Xiong Zhou, Hui Liu, Haoliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.03248)  

**Abstract**: The alignment of large language models with human preferences is commonly achieved through Reinforcement Learning from Human Feedback or Direct Preference Optimization. However, these methods are vulnerable to the significant noise prevalent in real-world preference datasets. To address this critical issue, we present a theoretical framework for unbiased alignment, introducing the Unbiased Reward Model (URM) loss and the Unbiased Direct Preference Optimization (UDPO) loss. By mathematically correcting the distortion induced by preference noise, our novel objectives enable unbiased model training directly from noisy datasets, without requiring clean ground-truth supervision. We provide rigorous theoretical analyses demonstrating that our methods are noise-tolerant, parameter downward compatible, and classification-calibrated. Comprehensive experiments across diverse datasets demonstrate that our approaches outperform state-of-the-art baselines. Code available at: this https URL. 

---
# Agentic and Generative AI for Open-Source Intelligence and Cyber Investigations: Taxonomy, Evaluation, Challenges, and Future Directions 

**Authors**: Eduardo Almeida Palmieri, Mohamed Chahine Ghanem, Dipo Dunsin, Zubair Baig, Ed de Quincey, Kim-Kwang Raymond Choo  

**Link**: [PDF](https://arxiv.org/pdf/2607.03233)  

**Abstract**: The rapid growth of publicly available digital information has rendered manual open-source intelligence (OSINT) analysis insufficient for modern intelligence, cybersecurity, and cyber investigation. Large language models (LLMs) and agentic AI systems, capable of tool use, multi-step reasoning, and iterative intelligence generation, have emerged as promising solutions, yet evaluation frameworks have not kept pace with reported capabilities. This survey systematically reviews 74 studies and makes four contributions. First, it establishes agentic AI as a distinct analytical category rather than an extension of LLM prompting, organising the literature through an 11-category taxonomy covering LLM foundations, agentic architectures, retrieval-augmented generation (RAG), knowledge graphs, prompt engineering, domain adaptation, evaluation benchmarks, and risk. Second, it identifies the hallucination-validation gap as a corpus-level finding: although hallucination is recognised as a major reliability concern in over twenty studies, end-to-end hallucination is empirically measured in only one OSINT-specific RAG-based system, non-reproducible conditions, while related reasoning and factual-correction studies evaluate general-domain question answering rather than OSINT. Third, it maps existing research to the OSINT lifecycle, showing strong support for collection and analysis but limited coverage of verification, reporting, dissemination, and decision support. Fourth, it derives a ten-point research agenda addressing evaluation, benchmarking, hallucination measurement, adversarial robustness, dark-web coverage, multimodal intelligence, and governance. It concludes that a human-AI co-pilot model, where LLMs assist collection and triage while analysts retain responsibility for verification and decision-making, represents the most defensible near-term deployment architecture. 

---
# CONTRA: Red-Teaming Configurations of Personalizable Agents 

**Authors**: Jonathan Nöther, Adish Singla, Goran Radanovic  

**Link**: [PDF](https://arxiv.org/pdf/2607.03220)  

**Abstract**: Recent tools such as OpenClaw have extended the capabilities of LLM-based agents from simple dialog-based systems to fully autonomous agents. These systems allow personalization of the agent through modifiable internal files and the installation of skills. While this enables deployment in a wide range of settings and the automation of diverse tasks, greater capability and autonomy increases the risk of malicious actions being executed unintentionally. In this work, we explore the interplay between agent configuration and the risk of executing dangerous actions without explicit instruction. To this end, we propose CONfiguration Tree-search for Red-teaming Agents (CONTRA), an LLM-assisted tree-search algorithm that discovers agent configurations resulting in the execution of malicious actions. CONTRA works by reasoning about benign yet dangerous configurations and evaluating them in a simulated environment. We construct a dataset of the 473 most popular skills from a public repository, along with 2-5 corresponding malicious target actions per skill. In a large-scale analysis, we find that 75.1% of skills have at least one configuration resulting in the execution of a malicious action, most of which have not been detected as containing malicious content by existing scans. Overall, CONTRA successfully identifies a configuration leading to the execution of the target action in 39.2% of all tested cases. Our findings demonstrate that current agents provide insufficient safety with respect to personalization. 

---
# Builder, Defender, Breaker: The Case Against Removing the Human from the AI-Driven Security Lifecycle 

**Authors**: Mohamed Chahine Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2607.03215)  

**Abstract**: Artificial intelligence has spread across the whole of the security lifecycle. The same family of models now writes application code, hardens it, and probes it for weaknesses, so that a single generative substrate increasingly performs all three roles at once. Enthusiasm for this convergence tends to treat full autonomy as the natural end point of partial assistance. This article argues that it is not. When the system that builds an artifact is drawn from the same distribution as the systems that defend and test it, the three roles inherit a common set of blind spots, and the independence that makes verification meaningful is quietly lost. Removing the human does more than raise the automation level: it collapses the external oracle against which machine output is judged, outruns the point at which a person could intervene, hands adversaries a predictable and poisonable target, and dissolves the locus of accountability when something fails. Drawing on evidence from autonomous code generation, adversarial machine learning, software fault tolerance, and the first all-machine hacking tournaments, we argue that the human belongs in the loop not as a temporary scaffold but as a permanent structural requirement, and set out what a defensible division of labour between people and machines should preserve. 

---
# OpenGlass: A Sensing-Computing Split Architecture for Local MLLM-Driven Real-Time Visual Assistance 

**Authors**: Mengzhang Li, Yuan Yao  

**Link**: [PDF](https://arxiv.org/pdf/2607.03213)  

**Abstract**: We present OpenGlass, an open-source, privacy-oriented, local-first system for low-latency multimodal visual assistance, with a primary focus on blind and low-vision users. Cloud MLLM assistants offer strong visual understanding, but often require uploading first-person visual data and can suffer multi-second network delays; wearable glasses are ideal for sensing, but cannot host large models under tight compute and power budgets. OpenGlass addresses this gap with a sensing-computing split: an ESP32-based glasses-side unit captures visual context, while a nearby consumer-grade device performs local MLLM inference and local speech output, reducing cloud reliance and keeping raw egocentric visual data on user-controlled devices by default. We evaluate response quality, query-ready-to-audio latency, safety-aware abstention, and auditable logs. Under real ESP32 Wi-Fi capture, OpenGlass reaches 993 ms median user-to-audio latency with resized payloads and 1625 ms with raw 1280 x 720 payloads; 97.5% and 93.3% of trials fall below 2 s, respectively. OpenGlass is a user-initiated visual-assistance reference platform for obstacle/hazard awareness, sign/object queries, and image-quality self-checking, rather than a certified navigation aid. We release source code, hardware instructions, prompts, evaluation data, and logs. 

---
# Transition Information Density: Morphological Trajectories, Synesthetic Perception, and Structured Interpolation in Neural Training (or: The Synesthetic AI) 

**Authors**: Sam Mao  

**Link**: [PDF](https://arxiv.org/pdf/2607.03210)  

**Abstract**: Standard machine learning training presents data as discrete endpoint pairs, omitting the structure of the space between them. This paper introduces Transition Information Density (TID) -- the information content recoverable from structured intermediate states between categorically distinct training endpoints -- and Positional Identity, the defined location of an intermediate state on the A-to-B continuum. Both constructs are grounded in three empirical contexts: grapheme-color synesthesia, the Synesthesia Grid (a boundary-contour morphing algorithm instantiating TID in visual morphological space), and a four-condition training experiment across four representational mediums. Probes trained on structured interpolation at defined Positional Identities (C3) exhibit substantially lower intrinsic dimensionality than volume-matched controls (C2) in Phonetic/Linguistic (C3: 3.33 vs. C2: 10.81) and Semantic Description (C3: 4.59 vs. C2: 8.67) mediums. Visual and cross-modal mediums do not show this effect, establishing a modality boundary condition. A fixed-N=50 comparison confirms that Positional Identity structure, not sample count, drives the effect. Resolution N scales monotonically with representational richness. Pooled TwoNN analysis reveals globally collapsed representations in visual space (0.075) and globally consistent representations in phonetic space (0.977). The paper contributes a formal definition of TID and Positional Identity, a nine-metric shape characterization framework, and a four-condition experimental design isolating trajectory structure, data volume, and Positional Identity as distinct factors. 

---
# Self-Specializing Vision-Language Transmon Chip Calibration in a Physics-Grounded Environment 

**Authors**: Animesh Tripathy, Aswanth Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2607.03193)  

**Abstract**: Calibrating a superconducting transmon chip is a sequential decision problem under noise, drift, and a finite budget: an expert must choose experiments, read ambiguous plots, judge fit quality, and revise stale beliefs as the chip drifts. We study whether a vision-language agent can close this loop and specialize itself to one physical device without weight updates, via three co-designed artifacts. The first is a physics-grounded simulation environment for transmon chips: calibration observables derive from circuit-quantized parameters via scqubits, with realistic flux-line distortion, wall-time-scaled and mid-scan drift, and gate leakage, concerns a toy simulator would omit; each tool call advances a modeled clock so drift accrues by wall time, not call count. The second is a vision-language agent that runs the loop end to end, calling tools, reading plots, maintaining a structured notebook, and submitting parameters without hidden truth, scored against hidden parameters and gate fidelities measured on the device. The third is gradient-free online adaptation: a reflector reads truth-free anomaly signatures from past attempts and grows a small, human-readable device note appended to the prompt, admitted by a paired-snapshot accept gate that isolates strategy improvement from drift. On a hard-tier chip under budget pressure, six iterations raised the worst-case CZ fidelity from 0.678 to 0.787 and cut its variance, reproducing at four-qubit scale; a single accepted note raised CZ fidelity from 0.678 to 0.913 on its paired snapshot. A planted-fault study confirms the note is causal, diagnosing a hardware fault truth-free, its principal value raising the failure floor and cutting variance. The agent, scoring, and reward transfer to real hardware via a measurement-backend swap; only the accept gate is a simulation affordance, reducing to a held-out-slice or repeat-and-average form. 

---
# A Bayesian Framework for Evaluating Scenario Compatibility in Generative Population Synthesis 

**Authors**: Zhenlin Qin, Leizhen Wang, Yancheng Ling, Zhenliang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2607.03190)  

**Abstract**: Scenario-based transportation analysis specifies future assumptions through aggregate population targets, whereas generative population synthesis models produce detailed individual-level realizations. When scenario targets are imposed on generative models, current practice relies on deterministic marginal calibration, implicitly assuming that the targets are compatible with the model's learned structural support. However, whether scenario-level constraints lie within the generative support--and how strongly they distort structural uncertainty--remains largely unexamined. We propose an ensemble-based Bayesian updating framework to quantify scenario compatibility in conditional population synthesis. A population-aware conditional variational autoencoder is developed to learn a distribution over plausible population structures while preserving aggregate fidelity. An ensemble of realizations sampled from the learned prior provides an empirical approximation of structural uncertainty. Scenario targets are treated as probabilistic evidence over aggregate statistics, and posterior weights are obtained through Bayesian updating across the ensemble. Scenario compatibility is quantified using effective sample size (ESS), which measures posterior concentration and the compression of structural uncertainty induced by conditioning. Experiments demonstrate that scenario impact depends not only on target magnitude but also on alignment with the learned joint structure, and reveal structural failure modes when targets fall outside prior ensemble support. The proposed framework provides a probabilistic diagnostic model for evaluating scenario feasibility and structural consistency before downstream projection and transportation planning. 

---
# Scalable Maximal Frequent Episode Mining with Desbordante 

**Authors**: Maxim Ivanov, Matvei Smirnov, Alisa Strazdina, George Chernishev  

**Link**: [PDF](https://arxiv.org/pdf/2607.03188)  

**Abstract**: Episode mining aims to extract subsequences of events that possess certain distinctive properties and constitute facts valuable to the user. Maximal frequent episode mining concentrates on discovery of frequently-appearing subsequences, which are not included into any other larger frequent subsequence. The state-of-the-art for this problem is the MaxFEM algorithm which enumerates possible subsequences, while applying various pruning techniques to accelerate the search. However, this is a computationally-intensive problem: reducing the minimum number of required subsequence occurrences or increasing the length of the subsequence both substantially raise running time, which limits practical use of MaxFEM. In this paper we describe our efforts in designing a high-performing algorithm for this problem. For this we: 1) develop an efficient C++ implementation of MaxFEM, and 2) devise an efficient technique to parallelizing it. As the result, we propose an improved parallel MaxFEM variant, which we call ParMaxFEM. Additionally, we integrate the improved algorithm into Desbordante - a high-performance, open-source data profiler with deep Python integration that treats patterns as first-class entities and allows users to develop their custom programs that can include discovery and validation of patterns. To evaluate our approach we compare both C++ implementations with the original SPMF implementation. Experiments demonstrated that our reimplemented version provides up to $8\times$ speedup over the SPMF baseline, while our parallelization technique provides up to $35\times$ improvement overall (on 8 cores). 

---
# AnchorVLA: Bridging Discrete Decisions and Continuous Trajectories for Vision-Language-Action Planning 

**Authors**: Qi Liu, Yabei Li, Hongsong Wang, Heng Zhang, Lei He  

**Link**: [PDF](https://arxiv.org/pdf/2607.03182)  

**Abstract**: Autonomous driving planning requires translating navigation intent, traffic rules, dynamic interactions, and language instructions into executable continuous trajectories. Vision-Language-Action models have been introduced into driving planning to improve long-tail generalization, commonsense reasoning, high-level semantic understanding, and explainability. However, existing VLA planners mainly follow planning-head-based trajectory prediction or full-trajectory autoregressive generation. The former only weakly constrains continuous trajectory generation with VLA reasoning, while the latter relies on long sequences of low-information-density coordinate tokens, making semantic-action alignment difficult and leading to discretization errors and inefficient inference. To address these limitations, we propose AnchorVLA, a hierarchical decision-anchored VLA planning framework that uses trajectory-pattern anchors as an explicit interface between high-level VLA reasoning and continuous trajectory execution. Specifically, Decision-as-Anchor Representation represents behavior-level driving decisions with anchor tokens, each encoding an entire local motion pattern rather than a single coordinate point. Decision-Anchored Residual Flow then generates fine-grained continuous trajectories in the selected anchor-defined residual space, capturing multi-modal execution refinements after high-level decision making. By reasoning over compact and semantically meaningful anchors instead of autoregressively generating waypoint sequences, AnchorVLA preserves LLM-based decision making while improving inference efficiency, semantic-action alignment, and continuous generation flexibility. Experiments on the Bench2Drive closed-loop benchmark show that AnchorVLA achieves a state-of-the-art Success Rate of 77.28 and a competitive Driving Score of 89.92. 

---
# Teaming Up with AI: Coordination and Cooperation 

**Authors**: Nicole Immorlica, Inbal Talgam-Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2607.03181)  

**Abstract**: Successful diffusion of AI in the workforce hinges on the economic value that AI brings to human endeavors. Bringing AI into the workforce is more than deploying a powerful new technology -- it is launching a new form of collaboration. Each human worker is now endowed with a team of AI agents; work can be delegated to these agents, and the role of the human shifts towards managing and monitoring. How can we maximize the economic value from collaboration with AI in the workforce? How can we make it a "true" collaboration that empowers human workers rather than replacing them?
We take an approach that combines the fields of theoretical computer science and economics, highlighting the potential of algorithmic tools grounded in economic principles to improve the effectiveness of human-AI collective work. We consider two tiers of tools: (1) tools for better coordination, via algorithmic management of interdependencies; (2) tools for better cooperation, via contractual incentive alignment. We show how a principled approach based on algorithmic and economic research enhances both coordination and cooperation, charting a pathway for future research to inform AI markets. 

---
# CRRL: A Causality-Based Reinforcement Learning Framework for Autonomous System Recovery 

**Authors**: Safia Fatima, Kai Olav Ellefsen, Leon Moonen  

**Link**: [PDF](https://arxiv.org/pdf/2607.03177)  

**Abstract**: Traditional reinforcement learning (RL) for recovery in autonomous systems lacks causal understanding and generalizes poorly to novel failure scenarios. RL policies often stall in failure states, spending up to 70% of an episode immobilized. Rule-based recovery alone is inadequate, and adding heuristic recovery to a pretrained PPO policy worsens rewards because policies cannot coordinate well with unanticipated interventions. The issue is not missing recovery mechanisms but a lack of policies trained to collaborate with them.
We introduce CRRL, a causal-guided RL framework that trains policies to work effectively with rule-based recovery. The recovery detects stalled states and assists the agent. Causal relations from driving logs shape the training signal, teaching the policy to anticipate stalls and adjust actions in recovery contexts. The framework follows MAPE-K, with sensor collection, causal model construction, and hybrid RL policy training corresponding to Monitor, Analyze, and Plan/Execute, respectively.
We evaluate CRRL through a four-condition ablation study across three driving scenarios, with 20 episodes per condition. We find that causal training significantly improves reward, distance, and velocity. Moreover, 9 of 20 roundabout episodes required zero recovery intervention, confirming navigation competence. These results show that causal-guided training produces effective RL policies that cooperate with rule-based safety components. 

---
# Effectiveness of LLM-based Software Diversity for Reliability Improvement -- an Empirical Study 

**Authors**: Gabriel Almeida, Ilir Gashi, Vladimir Stankovic, João R. Campos  

**Link**: [PDF](https://arxiv.org/pdf/2607.03174)  

**Abstract**: Software diversity has been extensively studied as a means of reducing the risk of common-mode failures. Classic work showed that the central issue is whether failures of diversely redundant components overlap in ways that limit the reliability gains. Traditional software diversity is costly to obtain, since it requires multiple implementations as well as the corresponding validation, maintenance, and deployment effort. Recent advances in Large Language Models (LLMs) may change this. LLMs enable inexpensive code generation: they produce many candidate implementations of the same specification quickly, across different models, decoding settings, and programming languages. This raises a natural question: can LLMs serve as practical generators of software diversity, and how much reliability improvement can that diversity actually provide? In this paper, we extend classical empirical studies of software diversity in human-written programs to LLM-generated code. We study three specifications using both historical human-written programs and large pools of LLM-generated ones evaluated under a common compilation, sandboxing, and exhaustive test suite. We explore LLM diversity along multiple axes, including model family, generation temperature, and programming language. Reliability improvement is evaluated in a 1-out-of-2 configuration across both homogeneous and heterogeneous program populations, including within-LLM pairings and pairings across programming languages and across LLM-generated and human-written programs. The results show that combining LLM-generated programs, especially in heterogeneous settings, can yield reliability gains, although this is partly conditioned by the programming language and generation setting. Taken together, these findings suggest that LLMs provide a scalable source of comparatively low-cost programs whose diversity can be leveraged for reliability improvement. 

---
# Decentralised Federated Learning over Temporal Networks: The Role of Heterogeneities 

**Authors**: Arash Badie-Modiri, Chiara Boldrini, Lorenzo Valerio, János Kertész, Márton Karsai  

**Link**: [PDF](https://arxiv.org/pdf/2607.03171)  

**Abstract**: Decentralised federated learning, based on peer-to-peer communication, is increasingly proposed for on-device training of machine learning models, promising a privacy-preserving, communication-efficient training process with no risk of single-point failure. However, the role of structural and temporal inhomogeneities in such fully decentralised settings remains poorly understood. Here, we investigate their effects when model parameters are locally averaged during aggregation. We show that the decentralised federated learning process is governed, both in the early phase and the late, stationary limit, by the same dynamics as a lazy random-walk diffusion process on temporal networks. Based on this mapping, we demonstrate that the typical experimental scenario used in decentralised federated learning leads to unrealistically rapid convergence because of ignoring the temporal and structural inhomogeneities inherent in the communication network. We analyse real-world temporal networks and find that inhomogeneities most often dramatically slow down diffusion, hence the convergence process. 

---
# KARMA: Knowledge graph-based Automated Reasoning Materialization and Alignment 

**Authors**: Jinkyeong Choi, Chaebin Jeong, Donghyeon Park  

**Link**: [PDF](https://arxiv.org/pdf/2607.03166)  

**Abstract**: Template-based contrastive synthesis is scalable, but its candidates often differ only in a few entity-slots while sequence-level optimization spreads supervision over mostly shared templates. We formalize this as the Resolution Mismatch Problem and propose KARMA, which enumerates schema-constrained paths over domain knowledge graphs and verbalizes them into slot-aligned contrastive candidates. Slot-Parallel Alignment (SPA) then applies a decoupled slot-level objective to route preference supervision to discriminative entity-slots, with slot-aware masked attention serving as an optional packed-evaluation implementation. Across biomedical, computer-science, and chemistry benchmarks, KARMA outperforms base LLM and same-data SFT baselines, and compares favorably with sequence and token-level preference methods. 

---
# The Role of Prompt Language and Translation-Theory-Driven Prompts in Large Language Models: A Case Study on Spanish-Chinese Journalistic Translation 

**Authors**: Haohong Lai, Weijia Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.03160)  

**Abstract**: This study examines how prompt language and translation theory-driven prompt design influence the quality of Spanish-Chinese journalistic translations generated by GPT-5.2. A parallel corpus of four editorials from El Pais was translated under 48 experimental conditions (4 prompt types, 3 prompt languages, and 4 articles). Translation quality was assessed using BLEU and BERTScore-F1 for automated evaluation, alongside human evaluation based on the Multidimensional Quality Metrics (MQM) framework. Automated metrics identified the baseline prompt (BASE) as the best-performing condition, whereas human evaluation ranked the brief-oriented prompt (BRIEF) highest (MQM: 8.66 vs. 7.84), a reversal likely attributable to the single-reference constraint inherent in automated measures. Sub-error type analysis revealed that translation theory-driven prompts selectively reduced Awkward style errors, while Unidiomatic style errors persisted across conditions. Prompt language had a negligible impact under both evaluation paradigms. These results indicate that translation theory-driven prompts can yield measurable quality gains under expert evaluation of journalistic translations, although their pedagogical implications for language learners remain suggestive and require validation through user-based studies. 

---
# Which Algorithm Specification Formats Help Language Models Implement Machine Learning Algorithms? 

**Authors**: Masahiro Kato, Taka Kato  

**Link**: [PDF](https://arxiv.org/pdf/2607.03158)  

**Abstract**: Large language models (LLMs) are increasingly used to implement algorithms from research manuscripts, but papers often leave implementation choices implicit. This study examines how the written format of an algorithm specification affects first-pass LLM implementation accuracy. We compare ordinary prose, LaTeX algorithm-style pseudocode, PDF-like extracted pseudocode, Markdown fields, YAML-like specifications, JSON-like specifications, and Python code stubs across five machine learning tasks, three models, and four experimental settings, yielding 4,020 generated implementations. Hidden tests evaluate details that often determine correctness, including tie-breaking, array shapes, numerical rules, return structures, and invalid-input behavior. Under the core-information setting, LaTeX algorithm-style pseudocode has the largest average format effect, with YAML-like specifications and ordinary prose close behind. Under complete information, GPT-5.4 mini shows no format differences in the matched comparisons, whereas Gemma 3 4B and Llama 3.2 3B still do. Code stubs do not consistently improve correctness despite specifying the function signature. The results support a writing recommendation: authors should state the interface, computation steps, numerical rules, and boundary-case behavior explicitly, instead of relying on a particular surface format to carry those details. 

---
# Conditional Diffusion Guided Knowledge Transfer for Multi-Domain Knowledge Graph Completion 

**Authors**: Jiawei Sheng, Taoyu Su, Xixun Lin, Xiaodong Li, Tingwen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.03154)  

**Abstract**: Multi-domain knowledge graph completion (MKGC) aims to improve missing triple prediction in a target KG by transferring knowledge from other support KGs. Existing methods typically enforce consistency constraints on equivalent entities across KGs to transfer knowledge, which risks suppressing domain-specific contextual information of entities. This design can also compromise entity representation information from all KG domains, impeding performance improvements, especially in low-resource data scenarios. To address this, we pioneer a generation-based paradigm for MKGC and propose DMKGC, a conditional diffusion-guided knowledge transfer framework. Our key insight is to treat each KG as a partial view of the entity entire information, and generate informative domain-general entity embeddings through diffusion models conditioned on support KGs. Particularly, we first initialize domain-agnostic entity embeddings as prior entity embeddings, and then encode them within individual KGs. Afterward, we fuse equivalent entities from support KGs as the conditional diffusion generation guidance. We leverage the prior entity embeddings as the proxy generation objective, which ensures this conditional generation to be unbiased towards any conditioned KGs. Simultaneously, we also train the generated embeddings to be predictive across KGs, thus preserving domain-specific information. Extensive experiments on 14 KGs in 3 benchmarks demonstrate a 4.3\% average MRR improvement in tail entity prediction over state-of-the-art methods, with sustained gains in low-resource data settings. 

---
# Rethinking Neural Nonlinearity as Gating 

**Authors**: Muhammad Sabih, Frank Hannig, Jürgen Teich  

**Link**: [PDF](https://arxiv.org/pdf/2607.03148)  

**Abstract**: Activation functions are considered an essential primitive for neural nonlinearity, i.e., they enable neural networks to serve as universal approximators. In this paper, we show that this nonlinearity can also be achieved by input-conditioned threshold gating through branches as a universal primitive. We demonstrate that standard activations -- whether piecewise-linear (ReLU, PReLU, Hardtanh) or smooth (SiLU, Sigmoid, Tanh, GELU) -- are in fact instances of a single Threshold Gating (TG) primitive. For softmax, we show that it admits an exact TG conversion via its equivalent per-element Sigmoid form. We then validate these equivalences by converting pretrained networks across CNNs, transformer-based models, and recurrent architectures, preserving model performance without requiring retraining. Threshold Gating also enables training from scratch that goes beyond replacing existing activations, enabling gains in model compression, performance, and shorter training. We also propose a 'Minimal Branch Theorem' which relates the minimum number of required branches in our primitive to the trainability of general deep neural networks. In terms of hardware implementation, TG maps to a unified implementation in the case of analog in-memory systems, addressing the bottleneck of analog-to-digital and digital-to-analog converters (ADC/DAC) that is known to significantly impact power consumption and on-chip area. 

---
# CuBAS: Information Geometric Curvature-Based Adaptive Sampling for Supervised Classification 

**Authors**: Alexandre L. M. Levada  

**Link**: [PDF](https://arxiv.org/pdf/2607.03145)  

**Abstract**: The informativeness of a training set is as consequential as its size, yet most sampling strategies remain agnostic to the intrinsic geometry of the data distribution. We introduce CuBAS (Curvature-Based Adaptive Sampling), an information-geometric framework for adaptive data selection in supervised classification, grounded in the q-state Potts Markov random field (MRF) model. The central insight is that a labeled dataset can be viewed as a statistical manifold, on which local curvature, estimated via the ratio of second to first-order observed Fisher information, faithfully encodes the geometric complexity of the data distribution. We construct a k-nearest-neighbor graph over the labeled data and derive a closed-form curvature score at each vertex from the Potts sufficient statistics. This curvature signal partitions the graph into two complementary regimes: low-curvature regions, corresponding to smooth, homogeneous clusters, and high-curvature regions, concentrated around decision boundaries that are disproportionately informative for classification. By selecting nodes from both regimes, CuBAS constructs compact yet maximally informative training subsets. Empirical evaluation across more than 60 benchmark datasets demonstrates consistent and statistically significant improvements over random sampling and uncertainty-based baselines, across a wide range of labeling budgets and classifier architectures. CuBAS is computationally efficient (linear in the number of k-NN graph edges), theoretically grounded in the differential geometry of statistical manifolds, and interpretable in terms of the local shape operator of the data manifold. 

---
# Text as Partial Constraint: Core-Residual Alignment for Robust Vision-Language Learning 

**Authors**: Chengzhen Yu, Canran Xiao, Siyuan Ma, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.03143)  

**Abstract**: Vision-language alignment powers open-vocabulary recognition, retrieval, and LVLM grounding, yet natural captions are often underspecified, making similarity brittle and overly confident under paraphrase and omitted details. We aim to learn representations whose matching is stable across caption views and whose confidence reflects how strongly text constrains an image. We propose Text as Partial Constraint (TPC), a core-residual alignment framework that treats multi-view captions as incomplete supervision. It distills a consensus semantic core as the alignment target, learns a single-view core predictor for standard inference with one query, and explicitly discourages vision-language similarity from depending on the orthogonal unsaid residual. An uncertainty-aware contrastive objective further softens alignment when caption views disagree, reducing overconfident updates under weak language constraints. Across zero-shot recognition and adversarial robustness, TPC achieves 81.42/64.05 Top-1 clean/robust accuracy on ImageNet and 76.19/52.03 on an Avg-14 transfer suite, while improving LVLM transfer with 85.16 POPE F1 and 59.57 OKVQA accuracy under an LLaVA-1.5-7B stack. These results suggest that modeling text as a partial constraint is a practical and principled route to more reliable vision-language representations under underspecified language supervision. 

---
# Detecting Architectural Drift in Safety-Critical Firmware through Runtime Trace Analysis 

**Authors**: Domenico Francesco De Angelis, Marco De Luca, Domenico Amalfitano, Pasquale Cimmino, Anna Rita Fasolino  

**Link**: [PDF](https://arxiv.org/pdf/2607.03135)  

**Abstract**: Maintaining consistency between architectural design and runtime-observed behavior is challenging in long-lived safety-critical firmware. This paper presents a runtime-informed methodology for detecting architectural drift in ISO 26262-compliant firmware. The approach collects hardware-assisted execution traces, abstracts them into message exchanges among firmware components, and compares the resulting runtime behavior with design-time sequence diagrams through a deterministic differencing step. The computed delta identifies discrepancies as confirmed, missing, additional, or inverted, while a constrained LLM-based step generates a human-readable report only to support expert review. We evaluate the methodology in an industrial firmware context through agreement-based validation and a practitioner survey. Results over 26 test cases show strong agreement between the generated deltas and expert-curated references, while practitioners perceive the reports as useful for interpreting drift, reducing manual analysis effort, and supporting safety-oriented documentation activities. The findings suggest that combining runtime trace analysis, deterministic architectural differencing, and constrained LLM-based reporting can practically support architectural drift detection in evolving safety-critical firmware. 

---
# A Multi-Task Deep Learning Framework for Real-Time Intelligent Video Surveillance with Temporal Event Validation 

**Authors**: Estera Dumitru, Stelian Spînu  

**Link**: [PDF](https://arxiv.org/pdf/2607.03131)  

**Abstract**: Modern video surveillance systems generate far more video streams than human operators can effectively monitor, making automated analysis essential for timely detection of security events. This paper presents a unified multi-task deep learning framework that simultaneously performs face recognition with zone-based authorization, automatic license plate recognition, weapon detection, fire and smoke detection, and human action recognition on a shared GPU platform. Among the integrated modules, two task-specific deep-learning models are proposed in this work to address scenarios that are insufficiently represented in publicly available datasets: a single-class weapon detector fine-tuned on a merged and relabeled dataset, achieving a mean average precision (mAP@0.5) of 0.947, and a SlowFast-R50 action recognition model trained on a purpose-built vandalism dataset comprising 614 video clips, achieving 94.33% classification accuracy. To improve robustness in continuous video, all detection modules are integrated into a temporal event-validation architecture based on multi-frame confirmation, confidence-weighted voting, and cascaded filtering, transforming frame-level predictions into reliable security events. Each module is evaluated independently on established public datasets (LFW, D-Fire, FIRESENSE, and UCF-Crime), followed by integrated end-to-end system evaluation. The proposed temporal validation strategy reduces the fire and smoke false-alarm rate from 52% to 4% and improves video license plate exact-match accuracy from 66.7% to 81.8%, while the complete framework maintains real-time operation with a per-frame latency below 100 ms on commodity hardware. These results demonstrate that combining specialized deep-learning models with temporal event validation provides an effective and practical solution for reliable real-time intelligent video surveillance. 

---
# ACPO: Adaptive Credit Policy Optimization via Fine-Grained Surrogate Entropy 

**Authors**: Zijun Xie, Yuyang You, Yongzhi Li, Enlei Gong, Zeyu Chen, Quan Chen, Yanhua Cheng, Peng Jiang, Yadong Mu  

**Link**: [PDF](https://arxiv.org/pdf/2607.03126)  

**Abstract**: Reinforcement Learning (RL) has substantially improved the reasoning ability of large language models (LLMs), but sparse outcome rewards still make token-level credit assignment difficult. Existing scalable RL methods typically assign trajectory-level rewards uniformly across tokens, while recent entropy-aware approaches either rely on coarse detached heuristics or directly optimize true entropy, which can introduce non-local gradient components misaligned with sampled-token policy updates. We propose Adaptive Credit Policy Optimization (ACPO), a token-level credit assignment framework based on a mode-local surrogate entropy. ACPO asymmetrically modulates policy updates by emphasizing uncertain decisions in successful rollouts and overconfident tokens in failed rollouts. We show that the surrogate admits deterministic entropy bounds and, under modal alignment and proximal updates, preserves the policy-gradient direction to leading order. Experiments on mathematical reasoning and coding benchmarks, including AIME 2025 and HumanEvalPro, show that ACPO consistently improves over strong RL baselines such as DAPO, GTPO, and SAPO. 

---
# SNR-Adaptive Unified Diffusion for Multi-Task Medical Image Segmentation 

**Authors**: Jiahao Liu, Hang Wei, Shuai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2607.03103)  

**Abstract**: Clinical cardiac imaging pipelines currently deploy separate models for each dataset and modality, incurring redundant training costs and precluding knowledge sharing across anatomically related tasks. Consolidating semi-supervised learning, unsupervised domain adaptation, and domain generalisation into one model is therefore a practical necessity, yet naive joint training exposes a fundamental barrier: conflicting label semantics between datasets collapse LA Dice from 90.31\% to 83.38\%, while gradient imbalance across tasks of unequal complexity suppresses the weaker tasks throughout training. We present UniT-Diff, a unified diffusion segmentation framework that resolves these conflicts through three targeted mechanisms. An 11-channel task-specific output space physically partitions label categories, eliminating cross-task gradient sign reversal by construction. SNR-Adaptive Task Conditioning (SATC) scales the task token by the log signal-to-noise ratio of the current diffusion timestep, suppressing domain-specific bias during coarse denoising and restoring full task guidance as the signal clears. Task-Type-Aware Conditional Dropout (TTACD) permanently removes the task token for domain-generalisation inputs, routing them through a shared neutral pathway that draws on cross-dataset cardiac anatomy rather than source-vendor statistics. Under a single parameter set, UniT-Diff surpasses independently trained task-specific baselines on all three benchmarks simultaneously: +0.87\% on LA, +1.77\% on MMWHS, and +0.88\% on MNMS. 

---
# Flow-A11y: Flow-Aware Accessibility Testing 

**Authors**: Nasr Eddine Fliti, Leisan Kokorina, Florian Tambon, Michael Papadakis  

**Link**: [PDF](https://arxiv.org/pdf/2607.03100)  

**Abstract**: Modern web applications increasingly expose accessibility barriers through interaction flows rather than static page snapshots. Keyboard traps, focus loss, modal leakage, delayed status updates, dynamic controls, and changing page regions often become observable only after users perform concrete actions. These behaviors are directly related to dynamic WCAG criteria, yet they remain difficult to automate because their assessment depends on runtime interaction evidence and is still commonly performed through manual inspection. We present Flow-A11y, a flow-aware accessibility testing system for interaction-dependent WCAG criteria. Given a target page and a natural-language scenario, Flow-A11y executes the flow in a real browser, records an ordered runtime trace, constructs criterion-specific evidence packets, gates unsupported judgments, and emits auditable findings grounded in resolvable runtime evidence. Evaluated on 19 real public-web scenarios covering 45 dynamic WCAG criteria, Flow-A11y achieves over ten times higher oracle agreement than a generic browser-agent audit, while its evidence-calibration layer improves fail precision from 23.5% to 41.4% and eliminates invalid evidence references. These results show that runtime traces provide actionable evidence for assessing interaction-dependent accessibility behavior. They demonstrate a practical path toward automating dynamic WCAG criteria that page-level scanners cannot assess and that have traditionally required manual evaluation. 

---
# Don't Wait to Reply: Towards Responsive yet Thoughtful Dialogue through Proactive Thinking 

**Authors**: Ante Wang, Jiaqi Fu, Xuanyi Chen, Ruotian Ma, Zhaopeng Tu, Weizhi Ma, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.03093)  

**Abstract**: Thinking has emerged as a critical capability for Large Language Models (LLMs) tackling complex tasks. However, its reactive nature, where reasoning is passively triggered only upon receiving a user response, inevitably introduces latency that compromises conversational fluidity. This stands in sharp contrast to human dialogue, where speakers proactively anticipate and plan future content during natural pauses to ensure seamless interaction. To bridge this gap, we propose Proactive Thinking, a framework that empowers models to pre-compute potential response elements during conversational downtime instead of waiting idly for the next input. We then introduce a training-free baseline that can think ahead by anticipating future states, balancing efficiency and quality through speculative continual thinking. To evaluate this approach in practice, we adapt three benchmarks of varying complexity into time-aware environments that simulate real-time conversational flow. We demonstrate that proactive thinking effectively improves interaction efficiency without compromising performance. Ultimately, this work advocates for a fundamental shift toward more intelligent, anticipatory, and real-time conversational AI. 

---
# STELLA: Efficient Sensor-to-LLM Translation for On-Device Human Activity Recognition 

**Authors**: Nirhoshan Sivaroopan, Albert Zomaya, Kanchana Thilakarathna  

**Link**: [PDF](https://arxiv.org/pdf/2607.03089)  

**Abstract**: HAR is increasingly expected to run continuously on edge devices, yet recent LLM-based methods remain hard to deploy: raw sensor prompts are long, cloud inference adds latency and privacy risk, and fine-tuned LLM pipelines turn general-purpose models into task-specific classifiers. We present STELLA, an efficient sensor-to-LLM translation framework for on-device HAR that shifts the burden from LLM adaptation to sensor tokenization. A lightweight hierarchical tokenizer compresses an entire multi-channel inertial window into a fixed set of compact latent sensor tokens, which are projected into the embedding space of a frozen pretrained LLM and combined with a natural-language prompt for label scoring. This preserves activity-relevant temporal and cross-channel structure while keeping LLM-side computation predictable across sensor configurations. STELLA also supports on-device personalization, adapting only the lightweight tokenizer on small amounts of user-specific labelled data and augmenting inference with a local retrieval context, keeping the LLM, user data, and retrieval on device. Across seven public HAR datasets and eight benchmark settings, STELLA achieves new state-of-the-art performance, improving over prior methods by up to 11.83% F1; on-device personalization yields up to a further 21.91% F1 as user data accumulates after deployment. STELLA also outperforms representative time-series tokenizers under the same LLM pipeline and achieves real-time inference under practical mobile and edge budgets, showing that efficient sensor tokenization is a practical path toward accurate, private, and personalized LLM-based HAR on edge devices. 

---
# Attention-Guided Efficientnet Architecture For Precise Criminal Identification in Surveillance Images 

**Authors**: Savitha N J, Lata B T  

**Link**: [PDF](https://arxiv.org/pdf/2607.03073)  

**Abstract**: Criminal identification from surveillance imagery has become a critical research area in intelligent forensic surveillance systems due to the increasing deployment of CCTV cameras in public and private environments. However, surveillance-based face recognition remains highly challenging because of low image resolution, illumination variation, motion blur, pose changes, facial occlusion, and background clutter. To address these limitations, this paper proposes an Attention-Guided EfficientNet (AG-EfficientNet) framework for precise criminal identification in surveillance images. The proposed framework integrates EfficientNet-B0 with Convolutional Block Attention Modules (CBAM) to enhance discriminative facial feature learning under degraded surveillance conditions. In addition, a multi-scale surveillance feature fusion strategy is introduced to preserve both local texture information and high-level semantic identity representations. A hybrid Softmax-Triplet optimization mechanism is further employed to improve inter-class separability and intra-class compactness for robust criminal identity discrimination. The proposed framework was experimentally evaluated using the Labeled Faces in the Wild (LFW) and SCFace datasets. Experimental results demonstrate that the proposed AG-EfficientNet framework achieved superior surveillance recognition performance with an identification accuracy of 98.2%, Precision of 97.9%, Recall of 97.6%, F1-Score of 97.7%, and ROC-AUC of 0.99, outperforming conventional deep learning architectures including AlexNet, VGG16, ResNet50, MobileNetV2, and standard EfficientNet-B0. Furthermore, Grad-CAM visualization and ablation analysis confirm the effectiveness of the proposed attention-guided feature learning strategy. 

---
# PixCon: Clean-Positive Contrastive Learning for Foundation-Model Semi-Supervised Segmentation 

**Authors**: Ebenezer Tarubinga  

**Link**: [PDF](https://arxiv.org/pdf/2607.03068)  

**Abstract**: Semi-supervised semantic segmentation (SSSS) has long turned on one question, which pseudo-labels to trust, and answered it with ever more careful confidence filtering. Foundation backbones change the regime: with a DINOv2 teacher a strict threshold already retains a measured 98%-clean pseudo-label set, so the accuracy that remains lives not in the filter but in how the embedding space is structured by class. We propose PixCon, a clean-positive pixel-contrastive framework. PixCon maintains a per-class memory bank that admits only labeled pixels the student already classifies correctly, guaranteeing a contamination-free positive set ($\rho_F=0$) by construction, unlike prior contrastive SSSS banks (ReCo, U$^2$PL) built from confidence-filtered pseudo-labels. It is a single branch over a consistency backbone, adds no inference-time parameters, and needs no bank-specific threshold. A first-order analysis of the supervised-InfoNCE gradient explains why contamination hurts: its false-positive term scales as $\rho_F/(1-\rho_F)$, which we measure (0.018 on Pascal, 0.106 on ADE20K) rather than assume. Across Pascal VOC, Cityscapes, and ADE20K, PixCon matches or improves a strong DINOv2-based UniMatch V2 baseline in a compute-matched one-switch protocol: it improves every Pascal-1/8 seed (a per-seed gain of about +0.2 mIoU) and its three-seed mean reaches 87.90, the published UniMatch V2-B figure. Because contamination is already rare under foundation-model teachers, our analysis indicates the $\rho_F=0$ guarantee acts chiefly as robustness as teachers weaken, while the accuracy gain comes from cleaner positive supervision, making clean-positive contrast a robust, low-cost default for foundation-model SSSS. 

---
# Spectral Rewiring for Exploration, Purification, and Model Merging 

**Authors**: Zhilong Zhang, Hongli Yu, Huan-ang Gao, Hanlin Wu, Yuxuan Song, Wei-Ying Ma, Ya-Qin Zhang, Hao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2607.03065)  

**Abstract**: Reinforcement learning has become a standard post-training recipe for large language models, but dense full-parameter updates create two deployment-relevant bottlenecks: suppressed reasoning performance, often reflected by premature saturation of test-time scaling, and interference when consolidating multiple capabilities through multi-domain training or model merging. We show that the reasoning-effective component of these updates is largely concentrated in the base model's spectral space, motivating Subspace-Aligned Rewiring (SAR), a post-hoc editing method that retains this spectral core while removing orthogonal components. SAR therefore preserves reasoning gains and filters residual update directions that suppress performance or amplify cross-domain interference. Across several model families and scales, SAR extracts compact reasoning cores using as little as approximately 0.58% of total parameters: it preserves over 99% of post-training performance and improves high-k exploration in mathematical reasoning, and generalizes to agentic coding by improving six of seven open benchmarks on an in-house model. SAR also purifies mixed-domain training updates by releasing suppressed coding capability while maintaining math reasoning and instruction following. It further enables model merging across experts, yielding cross-domain generalization that surpasses previous merging baselines and even the best single-domain experts. Overall, SAR shows that extracting reasoning-effective updates from parameter geometry can serve as a training-free mechanism to improve reasoning and multi-domain performance. 

---
# LACE-SVD: Loss-Aware SVD with Cumulative Error Correction for LLM Compression 

**Authors**: Zhuowen Liu, Longkun Hao, Shiyu Feng, Xiaowen Chang, Ruiqun Li, Changqun Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.03057)  

**Abstract**: The rapid growth in the parameter scale of large language models (LLMs) has created a strong demand for efficient compression techniques. As a hardware-agnostic and highly compatible approach, low-rank compression has been widely adopted to reduce both memory footprint and computational cost. However, existing SVD-based methods are still largely driven by local reconstruction objectives, overlooking two critical limitations: rank budgets are often allocated without explicitly considering layer-wise loss sensitivity, and local approximation errors can propagate and accumulate through the residual stream, leading to amplified global deviations from the original model. To address these issues, we propose LACE-SVD, a Loss-Aware SVD framework with Cumulative Error correction for LLM compression. LACE-SVD first estimates the calibration negative-log-likelihood increase induced by candidate layer-wise compression ratios and solves a budget-constrained allocation problem to assign rank budgets. It then refines the compressed model with closed-form local updates and introduces a propagation-aware correction for residual-stream output modules, reducing layer-output discrepancy as a proxy for cumulative error propagation. Experimental results demonstrate that at a high compression ratio (0.6), the WikiText-2 PPL of our method on LLaMA-7B (32.57) is significantly better than that of Dobi-SVD (46.18). 

---
# OmniFocus: Query-Guided Modality-Balanced Token Compression for Omni-Modal Large Language Models 

**Authors**: Shijie Cao, Qingyu Zhang, Boxi Yu, Yuzhong Zhang, Boxi Cao, Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.03050)  

**Abstract**: Omni modal large language models (OmniLLMs) have attracted wide attention for their ability to jointly process audio and video, but they generate large token sequences under audio-visual inputs, leading to substantial inference cost. Existing audio-visual token compression methods often rely on unimodal guidance, overlooking the temporal locality of query-relevant evidence in audio-visual inputs and implicitly assuming that the two modalities share a temporally aligned information density distribution. We propose \textbf{OmniFocus}, a training-free query-guided token compression method for OmniLLMs that performs independent importance estimation for video and audio, enabling a modality-symmetric compression design that preserves modality-specific salient evidence while maintaining audio-visual alignment, thereby mitigating the modality bias issue that can arise from unimodal-guided compression. Experiments on the Qwen2.5-Omni model family across four audio-visual benchmarks show that OmniFocus maintains strong compressed performance at low token retention ratios and outperforms existing baselines on several major benchmark scores at 25\% token retention. On DailyOmni with Qwen2.5-Omni-7B at 25\% token retention, OmniFocus maintains 59.40 accuracy while delivering up to 1.38$\times$ prefill speedup relative to the full-token baseline, highlighting a favorable practical accuracy-efficiency trade-off. 

---
# MambaLIE: Scene Light Intensity-Boosted Low-Light Image Enhancement with State Space Model 

**Authors**: Wanshu Fan, Xiangyu Li, Cong Wang, Kin-man Lam, Xin Yang, Haiyan Zhang, Dongsheng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2607.03013)  

**Abstract**: Images captured by consumer electronic devices, such as mobile phones and digital cameras, often suffer from low-light degradation due to sensor limitations and imaging pipelines, which degrades visual quality and affects downstream vision tasks.
Existing methods based on Convolutional Neural Networks (CNNs) and Transformers have dominated current low-light image enhancement (LIE) due to their excellent ability to model hierarchical features.
However, CNNs operate in local receptive fields that cannot model long-range dependencies, while Transformers overcome this problem but incur substantial computational costs.
To address these challenges, we propose MambaLIE, a Scene Light Intensity-Boosted Low-Light Image Enhancement method based on a State Space Model (SSM).
We first introduce scene light intensity to improve the structural distribution of illumination, which is then gated with the low-light input to guide enhancement.
To better model the illumination while maintaining computational efficiency, we propose the Locally Enhanced State Space Model (LESSM) for efficient light enhancement.
Our LESSM contains two branches: an SSM branch and a Local Enhanced branch, where the former is used to model the long-range dependencies with linear time complexity, while the latter is used to enhance local feature representations.
Extensive experiments demonstrate that MambaLIE outperforms state-of-the-art CNN-based and Transformer-based LIE methods on four widely used synthetic benchmarks and five publicly available real-world benchmarks in terms of accuracy, speed, and model size, making it suitable for practical deployment on resource-constrained devices. 

---
# HyperVAttention: Efficient Sparse Attention with Spatio-Temporal Clustering for Video Diffusion 

**Authors**: Dongyeun Lee, Amir Zandieh, Vahab Mirrokni, Junmo Kim, Insu Han  

**Link**: [PDF](https://arxiv.org/pdf/2607.03012)  

**Abstract**: Video Diffusion Transformers (VDiTs) have demonstrated significant capabilities in high-fidelity video generation. However, their ability to produce long-duration videos is fundamentally constrained by the quadratic complexity of the self-attention mechanism. Recent clustering-based sparse attention methods improve the quality-speed trade-off by grouping semantically similar tokens, but their practical efficiency remains limited by two bottlenecks: substantial clustering overhead and low CTA utilization caused by irregular cluster-induced blocks. We propose HyperVAttention (HVA), a training-free sparse attention framework that addresses both bottlenecks jointly. To reduce clustering overhead, we introduce 3D local-window clustering, which exploits the spatio-temporal locality of video tokens to restrict centroid search to fixed local neighborhoods, and implement it with a custom Triton kernel for efficient execution. We further propose a hybrid clustering strategy that performs full clustering only at anchor steps and updates only subset tokens at intermediate steps, leveraging the temporal stability of cluster assignments across denoising steps. To improve CTA utilization, we present hardware-aware cluster merging that minimizes CTA-aligned execution cost through parallel agglomerative merging, improving block density and approximation fidelity by utilizing idle tile capacity. Together, these components reduce clustering overhead, avoid redundant updates, and better align sparse attention with the fixed tile structure of modern GPU kernels. Experiments on Text-to-Video generation show that HVA establishes a new Pareto frontier for training-free sparse attention in video diffusion, reducing end-to-end latency by up to $2.13\times$ while improving fidelity over existing training-free sparse attention baselines. 

---
# Can Model Merging Improve Aggregation in DiLoCo? 

**Authors**: Stefan Horoi, Benjamin Thérien, Guy Wolf, Eugene Belilovsky  

**Link**: [PDF](https://arxiv.org/pdf/2607.03011)  

**Abstract**: Model merging techniques, which aggregate independently finetuned models into one to combine their capabilities, have become a topic of significant interest in recent years, with a broad array of methods having been proposed to tackle this problem. Simultaneously, an emerging trend in distributed learning has been the use of methods such as local SGD and DiLoCo, which greatly reduce communication costs by periodically aggregating the independently trained local models. However, these communication-efficient methods have been shown to degrade in performance relative to the FLOP-matched data-parallel gold standard as the number of independent local models grows and as the number of local training steps before global communication is increased. In this work, we draw an explicit analogy between the pseudo-gradient aggregation step in local SGD/DiLoCo and task arithmetic-based model merging, establishing a straightforward way to utilize merging methods in the context of distributed optimization. We then evaluate multiple state-of-the-art model merging methods in this setting and identify one method in particular, Iso-C, as a promising approach for improving DiLoCo. We find that DiLoCo SGD with Iso-C aggregation outperforms not only simple pseudo-gradient averaging but even the momentum-based DiLoCo, despite lacking a momentum mechanism itself. Building on this finding, we propose IsoLoCo, which adapts Iso-C for distributed training by equipping it with Nesterov momentum. Our empirical evaluations on language model pre-training across varying numbers of local workers show that IsoLoCo significantly outperforms DiLoCo, with the gap between them widening as the number of workers increases. This advantage remains present across model sizes and inner step counts, confirming that merging-inspired aggregation is an effective strategy for low-communication distributed training. 

---
# Back to Basics: Improving Molecular Understanding in LLMs via SMILES-Graph Translation 

**Authors**: Wenda Wang, Jinjia Feng, Zhewei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2607.03007)  

**Abstract**: Recent advances in molecular large language models have led to strong performance on molecular understanding and generation tasks, yet these gains often come without reliable structural grounding. In particular, existing approaches conflict with the chemistry principle that structure determines function: despite their downstream success, current molecular LLMs perform poorly on basic structure recognition, suggesting that they fail to capture molecular graphs from canonical SMILES. To remedy this, we propose MolBasic, a structure-first framework that strengthens structural comprehension via SMILES-Graph translation. MolBasic is built around a multi-level structure perception benchmark, where bidirectional SMILES-Graph conversion serves as the core task to align sequential and topological representations. On top of this foundation, we employ a progressive learning scheme with a standardized Chain-of-Thought (CoT) to steer models from structure acquisition toward higher-level molecular reasoning. Experiments show that MolBasic substantially improves structural understanding and yields robust gains on downstream tasks, including property prediction and objective optimization, supporting our structure-first paradigm. 

---
# PosterHarness: Turning Scientific Poster Generation into an Auditable Instruction-Following Benchmark 

**Authors**: Tianyi Yang, Dawei Fu, Youpeng Wu, Zixun Kou, Linrui Chen, Ruobing Jiang, Zijian Wang, Qiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.03006)  

**Abstract**: Text-rich image models can now design poster-scale layouts, but we lack ways to measure whether they honor scientific communication contracts: legible labels, prescribed aspect ratios, and -- above all -- abstaining from fabricated scientific figures. We present POSTERHARNESS, an auditable harness reframing poster generation as measurable instruction-following tasks, with a pilot benchmark and failure taxonomy. POSTERHARNESS uses a placeholder-first contract to separate two jobs models otherwise conflate. The model performs visual-summary design: typography, reading path, color, and background -- but never draws data-bearing figures. Every figure region must be an empty labeled placeholder; a deterministic compositor inserts real source-paper figures at detected coordinates. This makes properties measurable: placeholder count and ID accuracy, blankness, aspect-ratio compliance, abstention from synthesized graphics, public-text hygiene, and source-figure provenance -- with failures logged as explicit rejections, not hidden in plausible-looking output. We instantiate the harness on 12 papers (6 HEP, 6 AI/ML-adjacent) and report three findings. (i) A counterfactual probe shows the placeholder contract drives VLM-counted synthesized figures from 34 to 0 across three papers. (ii) A failure taxonomy identifies blocking contracts: placeholder geometry, placeholder QA, template critic, and public text. (iii) Comparison with Paper2Poster shows a trade-off: PosterHarness yields higher-resolution artifacts, lower white-canvas fraction, and stronger VLM visual preference; the deterministic baseline retains slightly more PosterQuiz-style information and runs faster. We report this as regime characterization, not a superiority claim. All artifacts, prompts, manifests, and audit scripts are released as a reusable evaluation component. 

---
# CONFLUX: A Latent Diusion Model for 3D Chest-CT Synthesis with RL Post-Training 

**Authors**: Max Van Puyvelde, Halil Ibrahim Gulluk, Wim Van Criekinge, Olivier Gevaert  

**Link**: [PDF](https://arxiv.org/pdf/2607.02998)  

**Abstract**: Controllable generative models of 3D medical images can synthesize volumes with specified clinical attributes, but this demands samples that are simultaneously high-fidelity, natively 3D, and faithful to the requested conditioning. We present CONFLUX, a latent diffusion model for chest computed tomography (CT): a 3D variational autoencoder compresses each volume, and a rectified-flow transformer generates in the latent space. Generation is conditioned on structured radiological metadata (18 abnormality findings, sex, age, and reconstruction kernel) through adaptive layer normalization. The model leads strong volumetric baselines on tri-planar Frechet distance (FID 32.3 vs. 74.6 for MAISI) while exposing direct control over clinical attributes. To strengthen that control we add an online reinforcement-learning post-training stage (group-relative policy optimization) that rewards how reliably a classifier recovers the requested findings from each generated volume. Judged by a separate, independent classifier, post-training removes 47% of the shortfall relative to real-scan reliability. We release the model and a ~200k synthetic chest-CT dataset with conditioning metadata spanning a wide variety of clinical findings. 

---
# VISTA: Auditing Semantic Divergence in Vision-Language Models 

**Authors**: Junchi Liao, Jiawen Deng, Fuji Ren  

**Link**: [PDF](https://arxiv.org/pdf/2607.02995)  

**Abstract**: Vision-language models can exhibit visual concept-conditioned divergence: given images containing demographic features, corporate logos, or ideological symbols, some models produce unusually uniform responses that differ from what peer models say about the same input. These behaviors evade text-only audits because visual concepts cannot be isolated or substituted the way text tokens can. We present VISTA (Visual Inconsistency Screening Through Analysis), a black-box cross-model audit that couples semantic entropy with distribution-based divergence to flag model-specific anomalies. In a controlled study, we implant concept-conditioned stances in three VLMs via fine-tuning on small biased datasets and confirm that VISTA detects them. Auditing six VLMs across 19 topics, VISTA surfaces 142 high-suspicion cases (1.2%) and identifies selective refusal as a previously unreported divergence pattern, where models refuse demographic queries at rates varying from 0 to 65% across groups. 

---
# Enhanced Feature Extraction for IoT Network Intrusion Detection Using GNNs and KAN 

**Authors**: Long Zhao, Shixun Ji, Bin Cheng, Bin He  

**Link**: [PDF](https://arxiv.org/pdf/2607.02981)  

**Abstract**: Recent advancements in the Internet of Things (IoT) emphasize the urgent need for advanced network security, as IoT networks feature dynamic topologies, imbalanced traffic, and complex attack patterns. Unlike general IT networks, IoT environments exhibit extreme heterogeneity and sparse topologies. Traditional GNN-based intrusion detection methods often struggle to efficiently model node and edge features or capture fine-grained anomalies in such settings. To address this, we propose SKGFusionKAN, a novel IoT-tailored approach enhancing GraphSAGE with a multi-scale selective kernel attention mechanism. This enables adaptive extraction of node and edge features under diverse traffic conditions. Specifically, our edge-oriented message passing strengthens information propagation, while selective kernel attention adaptively weights edge-derived information from different scales to handle heterogeneity. We also introduce a gated fusion process to dynamically integrate multi-scale features, improving robustness against evolving attacks. Finally, we leverage Kolmogorov-Arnold Networks (KAN) for classification, offering superior nonlinear modeling capabilities essential for detecting intricate, low-frequency attacks. To our knowledge, this work presents a comprehensive integration of GNNs and KAN with dedicated architectural innovations for IoT intrusion detection. Extensive experiments on four NIDS benchmarks show that SKGFusionKAN consistently outperforms state-of-the-art approaches in binary and multiclass tasks, demonstrating its potential for IoT security. 

---
# Hierarchical Sparse Attention Done Right: Toward Infinite Context Modeling 

**Authors**: Xiang Hu, Xinyu Wei, Hao Gu, Minshen Zhang, Tian Liang, Huayang Li, Lei Zhu, Yan Wang, Sirui Han, Yushi Bai, Kewei Tu, Haitao Mi, Leo Liang  

**Link**: [PDF](https://arxiv.org/pdf/2607.02980)  

**Abstract**: Scaling modern large language models (LLMs) to long contexts is limited by the quadratic computation cost, and poor length extrapolation of dense attention. Chunk-wise sparse attention offers a promising alternative, but all existing methods fall short of full attention because of their inaccurate chunk selection. We propose Hierarchical Landmark Sparse (HiLS) Attention, a chunk-wise sparse attention mechanism that learns chunk selection end-to-end under the language-modeling (LM) loss. HiLS factorizes attention hierarchically: each query performs attention independently with each retrieved chunk to extract chunk-specific information, and the resulting outputs are fused according to chunk retrieval scores. By incorporating retrieval scores into the forward attention computation, HiLS optimizes them directly with the LM loss, enabling end-to-end retrieval learning and native sparse training. Experimental results show that HiLS-Attention achieves performance comparable to, and in some cases better than, full attention at in-domain context lengths. Meanwhile, HiLS-Attention extrapolates more than $64\times$ the training context length with 90% retrieval accuracy, far beyond full attention. Moreover, existing full-attention models can be converted to HiLS-Attention with lightweight continued pretraining, preserving in-domain performance while acquiring ultra-long-context extrapolation. Together with its sparse KV access and computation, HiLS-Attention breaks the usual efficiency-performance trade-off, enabling long-context LLMs that are both more efficient and more effective on general long-context tasks than their full-attention counterparts. 

---
# Individual Parameters in Weight-Sparse Transformers Appear Interpretable 

**Authors**: Arnau Marin-Llobet, Stefan Heimersheim  

**Link**: [PDF](https://arxiv.org/pdf/2607.02964)  

**Abstract**: A central goal of mechanistic interpretability is to understand how neural networks work and what each individual component does. Dominant circuit-finding approaches focus on a specific behavior and reverse-engineer the role of components on the associated sub-distribution. However, past work has shown that components can have different functions that are active on different subsets of the input distribution. In this work we ask whether a single weight can be understood globally across the full training distribution by characterizing when it matters (the inputs on which ablating it changes the model's predictions). We introduce an automated LLM pipeline that writes a short, human-readable description of when a weight matters and verifies it on held-out text, crediting a weight only if its description generalizes. Across two sparse and two dense transformers, the fraction of weights that are interpretable (in this sense) is higher in sparse transformers than in dense ones, a gap that widens once unreliable descriptions are discarded. Our results show that a meaningful fraction of a sparse transformer model's weights can be interpreted: 12 to 31% of weights have a single short description that identifies what the weight is used for. 

---
# Parallelized Autoregressive Decoding for Omni-Modal Dense Video Captioning 

**Authors**: Wenzheng Zeng, Siyi Jiao, Chen Gao, Hwee Tou Ng, Mike Zheng Shou  

**Link**: [PDF](https://arxiv.org/pdf/2607.02963)  

**Abstract**: Dense video captioning aims to generate temporally grounded descriptions of video events, benefiting both event-level video understanding and generation. In this domain, autoregressive video large language models have emerged as a prevalent paradigm due to their strong generative and cross-modal modeling capacity. However, generating dense captions under the token-by-token paradigm severely limits inference efficiency and hinders scalability as video length and event density increase. In this work, we propose a parallelized autoregressive framework that not only improves generation efficiency but also enhances temporally grounded captioning performance. Our key insight is to exploit the weak local dependencies across temporally distinct events to restructure the causal dependency graph, thereby enabling lossless parallel generation. Specifically, tokens with weak cross-event dependencies can be decoded in parallel, while tightly coupled tokens within each event retain sequential decoding to preserve local semantic coherence. To realize this insight, we introduce two key components for lossless parallel decoding: (1) a latent global planning mechanism that automatically learns the event-level structure and produces compact tokens encoding global inter-event causality while adaptively aggregating event-level audio-visual semantics, guiding subsequent dependency restructuring and parallel decoding; and (2) an event-factorized parallel decoding mechanism that effectively balances local focus with global inter-event awareness. Experiments on various benchmarks demonstrate the clear advantage of our approach in both efficiency and performance in omni-modal event grounding and captioning. Project website: this https URL. 

---
# The Foreign Policy AI Evaluation Gap 

**Authors**: Charles Pozniak, Jeba Sania  

**Link**: [PDF](https://arxiv.org/pdf/2607.02955)  

**Abstract**: We argue that AI systems used in conducting foreign policy tasks - broadly enacting 'statecraft' - should be a priority test case for technical AI governance research. In enacting foreign policy, we refer to the formulation and implementation of external objectives by political actors. Statecraft is a high-consequence deployment domain, with extreme downside risks and structural properties that standard evaluation practices handle poorly. These features include partial observability, unbounded action spaces, contested ground truth, and multidimensional objectives. This paper advocates for a literature-grounded research agenda. Our contribution is threefold: (i) a claim about the structural conditions of foreign policy that combine catastrophic tail risk with technical evaluation complexities, (ii) an ECOSYSTEM review that highlights the asymmetric focus on ASSESSMENT features over ACCESS, VERIFICATION, SECURITY, and OPERATIONALIZATION, and (iii) a demand-side evaluation framework that decomposes foreign-policy workflows into bounded, evaluable sub-tasks with human recombination. As AI systems are already being deployed in the conduct of war and peace, amid limited public evaluation infrastructure from the technical AI governance community, this agenda is an urgent priority. 

---
# Pooling-Based Context Modeling for Convolution-Free Deep Image Prior 

**Authors**: Gihyun Kim, Jong-Seok Lee  

**Link**: [PDF](https://arxiv.org/pdf/2607.02952)  

**Abstract**: Convolutional Neural Networks (CNNs) achieve strong denoising performance by exploiting spatial context from neighboring pixels. Deep Image Prior (DIP) leverages this property to restore images from a single noisy input without requiring large datasets. However, the over-parameterized architecture of DIP often leads to noise fitting during optimization. In this paper, we propose Pool-DIP, a convolution-free architecture that incorporates pooling-based contrast modeling to capture spatial context efficiently. Pool-DIP improves denoising performance while significantly reducing the number of parameters and computational complexity compared to convolution-based DIP models. Experimental results show that Pool-DIP achieves competitive performance across multiple datasets, including a real-world benchmark. Spectral analysis further reveals that Pool-DIP stabilizes the evolution of high-frequency components during optimization and suppresses erroneous high-frequency signals. The proposed architecture also generalizes well to other image restoration tasks such as super-resolution and inpainting. 

---
# A Precedent-Guided Co-Scientist for Side-Effect-Aware Drug Redesign 

**Authors**: Yujin Kim, Charmgil Hong  

**Link**: [PDF](https://arxiv.org/pdf/2607.02944)  

**Abstract**: We propose PRECEDE, a precedent-guided co-scientist for side-effect-aware drug redesign that revises a parent compound to mitigate a specified side effect while preserving therapeutic function. Rather than isolated molecular generation, PRECEDE frames redesign as evidence-grounded reasoning over drug--side-effect associations, biomedical knowledge graphs, and precedents of safety-driven optimization, coordinated by an LLM orchestrator with explicit policies and human-review checkpoints. We position PRECEDE as a human-supervised AI-for-science workflow in which hypotheses remain auditable, falsifiable, and bounded by prior pharmacology. 

---
# MatPhaseBench: A Semantics-Guided Benchmark for Materials Phase Diagrams Understanding 

**Authors**: Hanwen Wang, Sihan Liang, Zhiwei Liu, Yangang Wang, Wei Yan, Yuqin Liu, Zongguo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.02934)  

**Abstract**: Materials phase diagrams are a core knowledge representation in materials science, encoding temperature,composition, phase stability, and phase transformation pathways, with their full understanding requiring thermodynamic mechanism analysis and scientific reasoning. Although VLMs have shown promise in scientific image understanding, their systematic evaluation on such logically complex images demanding deep mechanistic interpretation remains limited, and phase diagrams provide a challenging testbed for this purpose. We introduce MatPhaseBench, a high-quality, high-reliability benchmark for complex scientific image understanding, focused on materials phase diagrams. MatPhaseBench is constructed from 3681 papers in classical materials science journals, from which 200 high-quality diagram-text pairs were selected, covering 189 material systems and 70 elements. The benchmark has three key features: (1)targeting complex scientific image understanding-it moves beyond simple objective tests to open-ended tasks requiring deep comprehension; (2)comprehensive image-text alignment-semantic information associated with images is fully preserved during literature mining and matching; (3) high-quality human-supervised text acquisition-all descriptions undergo strict manual validation. Experimental results show that current VLMs remain substantially behind expert-level understanding: they are largely limited to surface visual perception, lack deep reasoning grounded in thermodynamic mechanisms, have limited domain awareness and expert analytical experience, and perform poorly in distinguishing fine-grained differences in composite or multi-diagram settings. Overall, MatPhaseBench constitutes a challenging research-grade benchmark, providing a foundational platform for complex scientific image understanding, phase diagram analysis, and trustworthy multi-modal AI in science. 

---
# PromptPET: Privacy-Utility Optimized Prompt Obfuscation 

**Authors**: Ke Yang, Olivia Figueira, Umar Iqbal, Athina Markopoulou  

**Link**: [PDF](https://arxiv.org/pdf/2607.02932)  

**Abstract**: Privacy is an important challenge when users interact with AI chatbots, since users may share sensitive information, explicitly or implicitly, and AI chatbots can use this information for user profiling. In this paper, we aim to protect user privacy via a user-side mechanism that transforms sensitive information in a user prompt, while preserving enough information to elicit a useful response from the chatbot. This approach faces an inherent tradeoff between protecting privacy (i.e., avoiding profiling) and preserving utility (i.e., getting personalized and task-specific responses). To that end, we consider, evaluate, and compare four different obfuscation actions, namely redaction, abstraction, replacement, and a novel noising/denoising scheme that we introduce. Additional novel insights include: utilizing a data type taxonomy to both identify and obfuscate sensitive information and explicitly taking into account the utility of chat responses in making the obfuscation decision. First, we systematically optimize and evaluate each obfuscation action independently in terms of the privacy-utility tradeoff it achieves. Second, we propose PROMPTPET, an LLM-based agent that selects the best obfuscation action for each sensitive part of the prompt, using a reinforcement-learning inspired rule optimizer, applied for the first time in this context. Using a real-world chat dataset, we show that PROMPTPET matches the best privacy-utility tradeoff attainable by any single obfuscation action and significantly outperforms prior state-of-the-art approaches. 

---
# Modeling the Impact of Visual Brand Language on Attention, Object Recognition, and Memory Retrieval 

**Authors**: Rachel F. Heaton, John E. Hummel  

**Link**: [PDF](https://arxiv.org/pdf/2607.02929)  

**Abstract**: Visual brand language is the set of visual properties that convey brand identity for a product. What is the impact of visual brand language on a person's ability to recognize and understand the functional identity of an object? Using an empirically supported modeling framework based on the JIM model of object recognition and the LISA model of analogical inference, we simulated the impact of visual brand language on object recognition, the allocation of attention, and retrieval of functional information about objects. Our simulations predict that brand information captures attention and can slow recognition of an object's functional category, with greater degrees of branding causing larger effects. These results have potential implications for the usability and experience of designed objects. 

---
# VideoSearcher: Empowering Video Deep Research with Multi-Tool Agentic Reasoning via Reinforcement Learning 

**Authors**: Zhenkun Gao, Yicheng Bao, Jinlong Peng, Xueheng Li, Theo Huang, Bangwei Liu, Kunquan Li, Zhenye Gan, Tao Hu, Chengjun Xie, Mingqian Yang, Xuanhua He, Zhizhong Zhang, Xin Tan, Chengjie Wang, Yuan Xie  

**Link**: [PDF](https://arxiv.org/pdf/2607.02927)  

**Abstract**: Video understanding is moving beyond closed-context perception toward open-world evidence exploration, a paradigm formalized as Video Deep Research (VDR). However, existing multimodal search agents primarily target static images, and the current VDR benchmark relies on text-centric retrieval that discards crucial visual information. To address these limitations, we propose VideoSearcher, a closed-loop agentic framework that empowers Vision-Language Models with multi-tool reasoning for VDR. VideoSearcher unifies temporal localization, spatial focusing, and multimodal search within a single reasoning trajectory, enabling agents to progressively ground visual clues, retrieve relevant evidence, and synthesize answers. To optimize knowledge-intensive reasoning trajectories, we propose Bi-branch Sequence Policy Optimization (BiSPO), a reinforcement learning algorithm that decouples tool-invocation optimization from answer-accuracy optimization. This design provides stable learning signals for both evidence-grounded reasoning and purposeful tool use. Furthermore, we construct VideoSearch-QA, the first benchmark designed to evaluate open-world video information grounding and multimodal search-based reasoning. Extensive experiments demonstrate that VideoSearcher significantly outperforms prior open-source agentic baselines across various search-oriented and multimodal understanding benchmarks. 

---
# R3D: Quantitative 3D Spatial Reasoning for Egocentric Wearables 

**Authors**: Maxwell Horton, Wei Lu, Quan Tran, Yury Astashonok, Kirmani Ahmed, Babak Damavandi, Anuj Kumar, Xiao Zhang, Seungwhan Moon  

**Link**: [PDF](https://arxiv.org/pdf/2607.02921)  

**Abstract**: Quantitative 3D spatial reasoning from egocentric RGB-D video is a critical capability for next-generation wearable assistants. Yet existing benchmarks do not reflect the challenges of handling (1) natural egocentric video, (2) posed RGB-D video inputs, and (3) challenging quantitative 3D spatial reasoning Q&A. To fill this gap, we introduce R3D-Bench (Reasoning in 3D), a benchmark of 3,033 quantitative spatial reasoning questions across 15 types -- spanning multiple-choice, distance-based, and volumetric reasoning questions -- built on top of 57 egocentric video sequences from Aria Digital Twin. To set a strong baseline on this dataset, we introduce R3D, a model-agnostic spatial tool-calling framework. In contrast to existing approaches that directly embed 3D information into the model's input representation, R3D constructs a 3D scene from video using segmentation and depth-lifted object representations. It provides this information to an LLM through eight composable spatial tools. On R3D-Bench, R3D with Qwen3-VL 235B achieves 73.5% mean relative accuracy, substantially outperforming the best depth-enabled baseline (CuTR+Tools, 61.9%) and the best RGB-only baseline (Gemini 3 Flash, 46.5%). 

---
# Harmonic-Aware Transformer for Real-Time Catheter Localization in Interventional Procedures of Magnetic Particle Imaging 

**Authors**: Abuobaida M. Khair, Wenjing Jiang, Xiaoli Yang, Moritz Wildgruber, Xiaopeng Ma  

**Link**: [PDF](https://arxiv.org/pdf/2607.02919)  

**Abstract**: Magnetic particle imaging (MPI) enables real-time, radiation-free tracking of magnetic nanoparticle-coated instruments, making it highly suitable for interventional procedures. This study proposes a harmonic-aware transformer framework that directly predicts catheter tip positions from raw MPI voltage signals, eliminating the need for image reconstruction and reducing computational latency. The framework incorporates frequency-domain preprocessing to isolate the 2nd to 8th drive-field harmonics, enhancing the signal-to-noise ratio while preserving motion-relevant features. A transformer architecture with six encoder layers and eight attention heads is employed to learn spatio-temporal dependencies across the three receive axes (x, y, z) for accurate three-dimensional position estimation. The model is trained on simulated MPI signals and evaluated on real in vitro datasets under standard, bending, and heartbeat-like motion conditions. The proposed method achieves sub-millimeter localization accuracy, with a minimum L2 error of 0.103 +/- 0.092 mm and mean absolute errors (MAEs) of 0.039 +/- 0.046 mm, 0.054 +/- 0.049 mm, and 0.060 +/- 0.044 mm along the (x, y, z) axes, respectively, for the bending dataset. Across all datasets, the MAE ranges from 0.165 mm to 0.655 mm, demonstrating consistent performance. The optimized inference achieves a latency of 0.55 ms per frame and a throughput of approximately 1800 frames per second, confirming real-time capability. Compared with conventional MPI-guided approaches relying on image reconstruction, the proposed framework provides improved accuracy, reduced latency, and enhanced robustness under complex motion conditions. These results highlight the potential of harmonic-aware transformer models as efficient and scalable solutions for real-time catheter localization in interventional MPI. 

---
# Bootstrap Flow-Map Tree Sampling Enables Online Feedback Driven Search 

**Authors**: Binglin Ji, Anindya Sarkar, Hengchang Lu, Jens Sjölund, Yevgeniy Vorobeychik  

**Link**: [PDF](https://arxiv.org/pdf/2607.02915)  

**Abstract**: In many scientific and engineering domains, maximizing discovery within a limited sampling budget demands strategic, observation-guided exploration. While generative models have enabled training-free reward alignment, current methods typically excel in local searches within narrow regions of the underlying distribution. These approaches struggle when preferences are unknown a priori and only revealed through sequential feedback-a scenario demanding broad exploration to uncover high-utility regions. To address this, we introduce Bootstrap Flow-Map-Tree (a.k.a BFMT), a novel computationally efficient sampling framework designed for history-aware global search and alignment under sampling budget constraints. BFMT enables full tree-path construction from any tree depth using a single function evaluation, drastically reducing computational overhead while providing critical foresight for sequential sampling. By enabling dynamic transition time steps scheduling, BFMT efficiently allocates its sampling budget, smoothly transitioning from broad global exploration to fine-grained local refinement of high-utility modes discovered through exploration. Extensive experiments and ablations across diverse search and alignment tasks demonstrate that BFMT substantially outperforms baseline approaches. 

---
# CoACT: Action-Preserving Observation Compression for Coding Agents 

**Authors**: Haorui Chen, Yuancheng Zhu, Yitong Zhang, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.02911)  

**Abstract**: LLM-based coding agents solve software-engineering tasks through iterative interactions with development environments, where returned observations accumulate in the context and become a major source of inference cost. Observation compression reduces this cost by shortening observations before they are appended to the context. However, existing methods still exhibit an unsatisfactory efficiency-effectiveness trade-off, as they do not explicitly model how compression affects the agent's subsequent behavior. This paper proposes CoACT, an action-preserving observation compression method for coding agents. CoACT is built on next-action preservation (NAP), which requires a compressed observation to induce the same next action as the raw observation. By checking the agent's immediate next action, NAP provides a practical signal for whether a compression preserves the information needed for continued task solving. During training, a teacher model first generates multiple compressed candidates of each observation. CoACT then uses an action-preservation reward based on NAP to filter out candidates that would change the agent's next action, and uses a length-reduction reward to choose compact candidates as supervision for a lightweight compressor. Experiments on SWE-bench Verified with three agentic models show that CoACT reduces average total token consumption by 33.0% while maintaining task-solving effectiveness close to the uncompressed agent. 

---
# Learning Taxonomic Trees with Hierarchical Representation Regularization for Large Multimodal Models 

**Authors**: Hulingxiao He, Zhi Tan, Yuxin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2607.02909)  

**Abstract**: Taxonomies provide key information about the semantic relationships between concepts and the inherent organization of vision and language. Despite their impressive capabilities, large multimodal models (LMMs) often lack taxonomic knowledge, leading to low hierarchical visual recognition (HVR) consistency. These models typically only rely on language modeling objectives during fine-tuning and lack explicit taxonomy-aware regularization. To address this, we propose Hierarchical Representation Regularization ($HiR^2$), a simple plug-and-play regularizer that improves hierarchical consistency in LMMs. Specifically, we introduce a semantic-aware visual tree construction framework that extracts coarse-to-fine visual features from intermediate LLM layers guided by textual cues. The regularizer combines two complementary objectives: a taxonomic entailment loss that enforces hierarchy via hyperbolic entailment cones in the Lorentz model, and a discriminative dispersive loss that promotes angular separation of semantically similar embeddings on the unit sphere without disturbing the radial hierarchical structure. Extensive experiments demonstrate that $HiR^2$ effectively captures taxonomic structures across diverse LMMs and fine-tuning methods. Code is available at this https URL. 

---
# TIER: Trajectory-Invariant Explanation Regularization for Membership Privacy 

**Authors**: Varun Sharma, Kar Wai Fok, Vrizlynn L. L. Thing  

**Link**: [PDF](https://arxiv.org/pdf/2607.02903)  

**Abstract**: Explainability is central to building trustworthy AI, yet explanation interfaces can inadvertently provide adversaries with an expanded privacy-related attack surfaces. Recent studies show that advanced membership-inference attacks succeed by exploiting confidence-drop trajectories, induced through attribution-guided perturbations, as discriminative features, rather than directly using confidence scores or explanation vectors. Existing defenses against membership inference fail to directly mitigate such explanation-driven attacks. In this work, we investigate whether, during training, a model's own gradients can be leveraged as defense signals against such attacks, thereby aligning explanation profiles between members and non-members. To this end, we propose a Trajectory-Invariant Explanation Regularization (TIER) defense that penalizes erratic fluctuations in confidence drops simulated through gradient-guided perturbations and simultaneously minimizes the distributional shifts via KL-divergence. Unlike conventional adversarial training, which emphasizes label robustness, our approach targets explanation robustness by enforcing self-consistency through KL-divergence and reducing the variance of confidence drops between members and non-members. Extensive experiments confirm that our method effectively mitigates these attacks, delivering privacy protection while maintaining model utility and explanation fidelity. 

---
# PPE-Bench: A Benchmark for Evaluating MLLM Unlearning under Private-Public Entanglement 

**Authors**: Xianren Zhang, Delvin Ce Zhang, Dongwon Lee, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.02897)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown strong capabilities, but they may memorize private information from web data, raising privacy concerns. Machine unlearning offers a way to remove such private knowledge without retraining from scratch. However, existing MLLM unlearning benchmarks have two major limitations. First, they rely on simplified images that contain only the single target individual, failing to reflect the visual complexity of real-world photos. Second, they typically assume that the forget set and retain set are fully separated, ignoring the fact that private information is often visually entangled with benign public information. For example, a private individual may appear with a public figure or in front of a well-known landmark, where unlearning the private target should not damage the public context. To address these limitations, we propose PPE-Bench, a new benchmark for evaluating MLLM unlearning under private-public entanglement. Each image contains a target individual to be forgotten and public information to be preserved, including public figure and landmark. We further introduce two simple but effective methods to better preserve public information during unlearning. Through experiments, we find that existing unlearning methods can reduce private information leakage, but often substantially harm adjacent public information. 

---
# SPLIT: Training-Free AI-Generated and Partially Edited Video Detection via Spatial Patch-Level Incoherence and Temporal Roughness 

**Authors**: Jongyeop Hyun, Hyounghun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.02886)  

**Abstract**: Deploying AI-generated video detectors in real-world services demands an ultra-low false positive rate (FPR) on real videos to avoid falsely rejecting authentic content, a regime where standard metrics such as AUROC fail to reflect actual operating behavior. We introduce Spatial Patch-Level Incoherence and Temporal Roughness (SPLIT), a training-free detector that operates on patch tokens from a frozen vision encoder to detect both fully generated and partially edited videos. SPLIT computes two complementary signals: Two-step Temporal Roughness (TTR), capturing non-smooth patch trajectories via one-step and two-step feature variation contrast, and Local Spatial Motion Incoherence (LSMI), measuring spatially inconsistent temporal changes through gradients of a feature-space motion field. The two are fused multiplicatively with gamma correction to sharpen real-fake separation at strict thresholds. We further propose a service-aligned evaluation protocol based on Fake Recall at fixed FPR with real-only threshold calibration and cross-real threshold transfer. Across three benchmarks (FakeParts, GenVideo, and ViF-Bench), SPLIT achieves the highest Fake Recall at FPR = $0.1\%$, substantially outperforming supervised and training-free baselines while remaining robust to post-processing with negligible overhead. The code is publicly available at this https URL . 

---
# Where do LLMs Fall Short in CBT-Guided Affective Reasoning? 

**Authors**: Vaishnavi Sinha, Pooja Guttal, Pranay Deep Reddy Katike, Vishal Sinha, Gerald Ndawula, Lira Yoon, Andrea Kleinsmith, Manas Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2607.02885)  

**Abstract**: Cognitive Behavioral Therapy (CBT) provides a structured framework for understanding a user's mental state by examining the interaction between cognitive and behavioral factors. However, out-of-the-box LLMs respond fluently and empathetically, yet collapse into validation & reflection, regardless of what the user actually needs. They know theoretical CBT (scoring up to 96% accuracy on licensing exam questions) but fail to apply it effectively. We explore this gap with a knowledge-guided framework that treats CBT dialogue as controlled affective reasoning: user narratives are decomposed into Beck's Cognitive Conceptualization structure, grounded in clinical SNOMED CT concepts validated via Natural Language Inference, and a Multiple Chain-of-Thought (MCoT) strategy selection between Validation & Reflection, Socratic Questioning, or Alternative Perspectives. To measure whether such guidance actually changes behavior, we introduce the Protocol Leverage Force (F), a behavior-level metric that captures how far an intervention shifts a model away from its default response. Across three open-weight LLMs and 14 RealCBT-derived case studies, evaluated with human experts, valence-arousal trajectories, and linguistic entrainment, F shows that simply introducing protocol definitions via single chain-of-thought prompting fails to change LLM behavior, while MCoT on these definitions guides strategy selection better. Still, the effect stays within 1% (approx. 1.2-1.3%), and all models remain biased toward Validation & Reflection. These results show CBT knowledge alone does not ensure effective application, giving the affective-computing community instrumentation to measure where LLMs fall short. 

---
# Determinants and Limits of LLM Security-Tool Orchestration: A Study with HexStrike-AI 

**Authors**: Romain Gerard, Assmaa Zeghaider, Yan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2607.02873)  

**Abstract**: Large language model agents driving security tool suites over the Model Context Protocol are increasingly common. Yet the factors that bound their capability remain poorly characterized: how much depends on the model versus the client that drives it, whether constraining the agent to the orchestrator's own tools helps, and where capability is limited by reasoning rather than by missing tools. Using HexStrikeAI, an open-source orchestrator that exposes 150+ tools, as a testbed, we follow a methodology that evaluates the system, diagnoses its failures, and applies targeted improvements. We run 86 picoCTF challenges across seven categories and three difficulty tiers, under three tool-access regimes and three model/client configurations (774 trials). We then apply corrections to existing tools, agent-behavior changes, and eleven new capability tools, and re-run the previously-unsuccessful trials. The diagnosis isolates the driving client as a first-order factor for a fixed model (a 2.1 * gap between two DeepSeek clients) and a monotonic difficulty gradient, with the largest gains in the mid tier. The overall solve rate rises from 55.4% to 72.0%, and every configuration improves significantly (paired McNemar p < 0.001, non-overlapping 95% confidence intervals). The residual failures are reasoning- or environment-bound rather than missing-tool. A 60-run stability sub-study finds single-run verdicts reproducible (17/20 unanimous). We discuss what the results imply for how such orchestrators should be evaluated, and we are explicit about the limits: the study uses a single benchmark, the fixes were tuned on the same challenges they were evaluated on, and the client effect is demonstrated for one model only, so its generality to other models remains a hypothesis. 

---
# Differential Amplifier-Inspired AmpAttention for Multi-View Robotic Manipulation 

**Authors**: Jin Yang, Ping Wei, Nanning Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2607.02845)  

**Abstract**: Multi-view robotic manipulation methods with the attention mechanism have recently achieved significant progress in both training efficiency and task performance. However, the inherent redundancy, occlusion, and viewpoint dependency in robotic view images often lead to severe attention drift. To address this challenge, we propose AmpAttention, a novel attention mechanism inspired by differential amplifiers in analog circuits. It aims to suppress attention noise and capture high signal-to-noise ratio signals for more reliable perception. Based on this, we introduce the RVAF model, which integrates task-guided intra-view and inter-view AmpAttention. Compared to previous state-of-the-art methods, RVAF achieves the optimal average success rate across 18 RLBench tasks (249 variations) while reducing training time by 33.3\%. RVAF also demonstrates strong potential in real-world high-precision tasks, exemplified by its ability to pick up a dart and accurately insert it into the red bullseye. Furthermore, we extend RVAF to RVAF++ by incorporating the SAM2 image encoder. RVAF++ achieves substantial gains on high-precision tasks, achieving a 91\% success rate on the `insert peg' task. More qualitative results are provided at the anonymous project website this https URL. 

---
# JavaVulBench: A Java Vulnerability Benchmark with Realistic Splits, a Unified Multi-Backend Harness, and a Leakage-Aware Evaluation Mode 

**Authors**: Norbert Sandor Szolnoki, Gabor Antal  

**Link**: [PDF](https://arxiv.org/pdf/2607.02825)  

**Abstract**: We release \textsc{JavaVulBench}, a benchmark dataset and evaluation harness for Java vulnerability detection. The dataset contains $\sim$30{,}600 Java methods spanning 1{,}740 CVEs and 700+ projects, labelled at both method and line granularity, with per-CVE publication dates and five realistic split strategies: random, project-disjoint, temporal, deduplicated, and unseen CWE-family. The harness provides a single \texttt{LlmPrediction} schema across three backend families (encoder classifiers, local generative models served by Ollama, and API-served LLMs routed through OpenRouter) so that twelve reference detectors CodeBERT, GraphCodeBERT, UniXcoder, DeepSeek-Coder-1.3B, and eight API/open-weight LLMs (GPT-4o, GPT-4.1-mini, Claude Sonnet~4, DeepSeek-v3, DeepSeek-Coder-v2, Qwen-2.5-Coder-14B/7B, CodeLlama-13B) are evaluated under identical conditions from a single command. A pre-training contamination audit is shipped alongside every model so users can separate genuinely unseen test CVEs from potentially memorised ones. Data, code, and fine-tuned checkpoints are archived on Zenodo [31] and short demonstration video is available on YouTube (this https URL\_hqkuoM) this https URL. 

---
# Vision Token Manipulation Attacks on Cloud-Edge Inference of Large Vision-Language Models 

**Authors**: Zikai Zhang, Rui Hu, Olivera Kotevska, Jiahao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.02819)  

**Abstract**: Cloud-edge Large Vision-Language Model (LVLM) inference enables efficient deployment by splitting computation between edge devices and cloud servers. In this process, intermediate vision tokens are transmitted from the edge to the cloud over a communication link, thereby exposing a new attack surface. We study vision token manipulation attack (VTM-Attack) under a black-box man-in-the-middle setting, where an adversary intercepts and manipulates a subset of transmitted vision tokens under a budget constraint. We propose four naïve attack strategies and an optimization-based token selection method. Experiments on 6 state-of-the-art LVLMs (3B-72B) across 4 benchmarks show that manipulating only 10\% of vision tokens can reduce accuracy by up to 88.31\%. These results reveal a critical vulnerability in cloud-edge LVLM inference. 

---
# SovereignNegotiation-Bench: Evaluating User-Owned Personal Agents In Delegated Bargaining Under Privacy, Consent, Evidence, And Institutional Pressure 

**Authors**: Dylan Zongmin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.02814)  

**Abstract**: Personal agents will increasingly negotiate on behalf of users: splitting costs with other personal agents, appealing platform decisions, escalating support disputes, requesting refunds, changing subscriptions, and negotiating deadlines or reimbursements. Existing negotiation benchmarks emphasize agreement, surplus, or strategic competence, but a user-owned agent can reach an agreement while harming the user through privacy leakage, consent violation, unsupported advocacy, over-concession, failed escalation, or poor auditability. We introduce SovereignNegotiation-Bench, a trace-level multi-turn benchmark for delegated personal-agent negotiation under private utilities, disclosure constraints, evidence requirements, and institutional asymmetry. The benchmark separates agent-visible observable state from evaluator-only labels and evaluates agreement success jointly with user utility, privacy, consent, evidence grounding, concession discipline, escalation, and auditability. We report an artifact-backed validation over 240 scenarios, 4 model families, 14 baselines, 13,440 frozen-prompt live trajectories, 61,135 parsed action rows, and a blinded 3-annotator audit over 300 items. The strongest agreement-maximizing baseline achieves the highest agreement rate but low user utility and high privacy/consent risk; FullSovereign does not maximize agreement, but obtains the best sovereign negotiation score by preserving utility, minimizing leakage, grounding claims, and reducing unauthorized commitments. The results show that agreement success is insufficient for user-owned negotiation agents. 

---
# Training Hybrid Block Diffusion Language Models with Partial Bidirectionality 

**Authors**: Pranshu Chaturvedi, Parth Shroff, Tarun Suresh, Hangoo Kang, Kaiyue Wen  

**Link**: [PDF](https://arxiv.org/pdf/2607.02805)  

**Abstract**: High-throughput long-context generation is one of the central challenges for large language models. Generation is typically memory-bandwidth-bound rather than compute-bound: each decoding step must stream the accumulated key/value (KV) cache from memory, so bandwidth demand grows with context length while only one token is emitted. Two parallel approaches have therefore emerged: reducing memory access with efficient attention variants and linear-time mixers such as Mamba, or increasing parallel computation by generating blocks of tokens at once. However, technical challenges arise when combining these two ideas. Earlier hybrid diffusion models such as DiffuMamba use bidirectional Mamba mixing, including a reverse-direction scan relative to causal generation. This reverse scan needs to scan the entire sequence, so its states are not prefix-only and cannot be precisely reused as a cache even when diffusion is performed block by block. We propose a BDLM Mamba--attention hybrid that addresses this challenge by restricting the reverse Mamba scan to the active denoising block, which enables exact caching across blocks. In an 87M-parameter DCLM sweep, BDLM Mamba-H achieves the best C4-en validation perplexity compared to BDLM attention and full-sequence baselines. At 350M parameters, it remains competitive with BDLM attention. For long-context inference, BDLM Mamba-H reaches 19.7x the throughput of full-sequence DiffuMamba-H at 65K tokens and 3.7x the throughput of BDLM attention at 262K, showing that Mamba hybrids are a potential long-context diffusion architecture. 

---
# Seduced by the Narrative: Assessing Rule Adherence in Semi-Open Textual Sandboxes 

**Authors**: Weiying Chen, Junlong Shen, Zhanyuan Guo, Xiaoou Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2607.02802)  

**Abstract**: As LLMs are increasingly deployed as autonomous adjudicators in semi-open textual game environments, robust rule adherence becomes critical when user intent conflicts with system rules. However, these models are trained to be helpful and compliant, leaving them vulnerable to a class of attacks we term \textit{Rhetorical Injection}, where adversarial users exploit narrative framing techniques such as pseudo-logical reasoning and authoritative coercion to bypass adjudication logic. We present CoC-Seduce, a multi-agent adversarial benchmark built on Tabletop Role-Playing Game (TRPG) mechanics, an ideal instantiation of semi-open environments where rules are explicit for adjudication, yet interaction remains entirely in natural language. Three frontier models, i.e., GPT-5.4, Claude Sonnet 4.6, Gemini 3.5 Flash, serve as adversarial generators producing 5,376 samples across 4 world settings and 16 skill categories. We then benchmark 20 target adjudicators against this corpus. Evaluation across 20 models reveals that neither model scale nor explicit reasoning mechanisms reliably confer adjudication robustness, with \textsc{Pseudo-Logic} emerging as the dominant attack vector and cross-cultural settings exposing systematic knowledge gaps across all evaluated families. Project page: this https URL 

---
# A Preliminary Study on Explaining Risk of Code Changes using LLM-Based Prediction Models 

**Authors**: Yalin Liu, Kosay Jabre, Rui Abreu, Zachariah J. Carmichael, Vijayaraghavan Murali, Akshay Patel, Jun Ge, Weiyan Sun, Cong Zhang, Audris Mockus, David Khavari, Peter C. Rigby, Nachiappan Nagappan  

**Link**: [PDF](https://arxiv.org/pdf/2607.02782)  

**Abstract**: Predictions by machine learning (ML) and artificial intelligence (AI) models are often received skeptically unless they are paired with intelligible explanations. In the context of just-in-time defect prediction, highlighting small portions of a software change (diff) -- beyond rule-based lints -- where risk may be concentrated has not yet been extensively investigated. In this work, we leverage attention weights from an LLM-based Diff Risk Score (DRS) model to highlight parts of a diff that the model focuses on when predicting risk. We aggregate token-level attention into interpretable code units (lines, hunks, and files), and present the top-K units to developers as a lightweight form of guidance during code review. We evaluate our approach using expert-labeled changes that have caused real outages. Results show that the highlighted snippets cover expert-labeled outage-causing change lines 53.85% of the time when highlighting the top-2 hunks, while requiring developers to review 26.28% of the changed lines on average. Because attention is produced during standard model inference, the approach is scalable for large development workflows and can be surfaced in the code review UI with low additional latency. 

---
# Safe Inference-Time Alignment via Lagrangian Reward Augmentation 

**Authors**: Yaswanth Chittepu, Ativ Joshi, Sohini Chintala, Scott Niekum  

**Link**: [PDF](https://arxiv.org/pdf/2607.02781)  

**Abstract**: Inference-time alignment steers a frozen language model during decoding using auxiliary reward signals, avoiding the cost of repeated weight updates. However, existing inference-time alignment methods typically optimize a single scalar score, so explicit safety constraints must either be ignored or encoded through manually tuned penalties. We propose Lagrangian Reward Augmentation (LARA), a general inference-time alignment framework under safety constraints. Starting from a KL-regularized constrained objective with a reward model and a cost model, LARA dualizes the constraint and reduces the optimization problem to a one-dimensional convex problem over a nonnegative dual variable. Estimated on a small calibration set, this dual variable defines an augmented reward that can be used as a drop-in scoring signal within existing inference-time alignment methods. For sequence-level sampling methods, such as Best-of-N reranking, the calibrated dual variable corresponds to the solution of the expected-cost constrained problem. For token-level reward-guided decoding methods, the same construction yields a principled dual-calibrated heuristic rather than an exact constrained-policy guarantee. We evaluate LARA on both sequence-level and token-level inference-time alignment methods, and find that LARA improves the helpfulness-harmlessness tradeoff, with Best-of-N achieving the best performance among inference-time methods, approaching finetuning-based direct alignment baselines. 

---
# Gemma 4 Technical Report 

**Authors**: Gemma Team, Sherif El Abd, Vaibhav Aggarwal, Robin Algayres, Alek Andreev, Olivier Bachem, Ian Ballantyne, Cormac Brick, Victor Cărbune, Michelle Casbon, Mayank Chaturvedi, Victor Cotruta, Alice Coucke, Phil Culliton, Robert Dadashi, Lucas Dixon, Mohamed Elhawaty, Utku Evci, Clément Farabet, Johan Ferret, Filippo Galgani, Sertan Girgin, Jean-Bastien Grill, Maarten Grootendorst, Jiaxian Guo, Cassidy Hardin, Yanzhang He, Steven M. Hernandez, Omri Homburger, Léonard Hussenot, Juyeong Ji, Armand Joulin, Aishwarya Kamath, Parnian Kassraie, Olivier Lacombe, Preethi Lahoti, Gaël Liu, Gus Martins, Luciano Martins, Tatiana Matejovicova, Ramona Merhej, Nikola Momchev, Sneha Mondal, Ryan Mullins, Sindhu Raghuram Panyam, Shreya Pathak, Sarah Perrin, André Susano Pinto, Etienne Pot, Angéline Pouget, Alexandre Ramé, Sabela Ramos, Douglas Reid, David Rim, Morgane Rivière, Karsten Roth, Louis Rouillard, Omar Sanseviero, Pier Giuseppe Sessa, Shane Settle, Danila Sinopalnikov, Sara Smoot, Piotr Stanczyk, Andreas Steiner, Lawrence Stewart, Ilya Tolstikhin, Michael Tschannen, Anton Tsitsulin, Nino Vieillard, Renjie Wu, Pingmei Xu, Haichuan Yang, Edouard Yvinec, Li Zhang, Joe Zou, Nicolas Aagnes, Abdelrahman Abdelhamed, Shivani Agrawal, Shubham Agrawal, Ibrahim Alabdulmohsin, Jean Baptiste Alayrac, Uri Alon, Chandramouli Amarnath, Ankesh Anand, Chrysovalantis Anastasiou, Setareh Ariafar, François-Xavier Aubet, Kyriakos Axiotis, Federico Barbero, Joelle Barral, Alexei Bendebury, Urs Bergmann, Stanley Bileschi, Kat Black, Mathieu Blondel, Sebastian Borgeaud, Arthur Bražinskas, Ryan Burnell, Robert Busa-Fekete, Mu Cai  

**Link**: [PDF](https://arxiv.org/pdf/2607.02770)  

**Abstract**: We introduce Gemma 4, a new generation of open-weight, natively multimodal language models in the Gemma model family. Designed to advance compute efficiency and reasoning, the Gemma 4 model suite features dense and Mixture-of-Experts architectures, ranging from 2.3B to 31B parameters. Alongside improved vision and audio encoders for all model sizes, we propose a unified, encoder-free architecture for our 12B model, which ingests raw audio and image patches. Furthermore, we integrate a thinking mode, enabling Gemma models to generate reasoning traces prior to responding. We improve inference speed, memory, and compute efficiency, as well as long-context abilities through critical design choices. Gemma 4 establishes a leap in performance across STEM, multimodal, and long-context benchmarks, and rivals larger, frontier open models in human-rated tasks. 

---
# Out-of-Distribution Generalization of Risk Aversion in Language Models 

**Authors**: Kristina Zhang, Junior Chinomso Okoroafor, Benjamin Maltbie, Andrew Lin, Abhitej Bokka, Elliott Thornley  

**Link**: [PDF](https://arxiv.org/pdf/2607.02755)  

**Abstract**: Training AIs to be risk-averse in resources could offer a failsafe in the event that AIs turn out misaligned. Misaligned but risk-averse AIs would tend to prefer low-risk, low-reward strategies like cooperation over high-risk, high-reward strategies like rebellion, limiting the downsides of any misalignment. But we can only feasibly train AIs to be risk-averse on low-stakes gambles, and we will only be safe if their risk aversion generalizes to astronomically-high-stakes gambles. Will it? To shed light on this question, we introduce RiskAverseOOD: a benchmark for measuring how well risk aversion generalizes out of distribution. We then offer some initial results. Using a variety of methods to make Qwen3-8B choose risk-aversely when the stakes are low, we find that we can induce substantial risk aversion when the stakes are astronomically high. Our models' learned risk aversion generalizes at least partially across 98 orders of magnitude. From a baseline 2% rate of choosing a safe `Cooperate' option, we see rates around 70% (SFT and tie training), 52% (DPO), and 39% (activation steering). In another experiment, our fine-tuned reward model reliably scores risk-averse reasoning above risk-neutral or excessively risk-averse alternatives (99.6% pairwise accuracy). We replicate these effects at different scales (Qwen3-1.7B and Qwen3-14B) and across model families (Gemma-3-12B-IT and Llama-3.1-8B-Instruct). Overall, we find that risk aversion learned at low stakes can generalize OOD to astronomically high stakes, though not yet consistently enough to serve as a reliable failsafe. Achieving that level of consistency is an open problem. 

---
# Echoes of Unrest: A Multimodal NLP Framework for Early Warning of Fake News and Violence-Driven Mob Activity 

**Authors**: Md. Maruf Bangabashi, Tahmid Hasan, Golam Mahmud, Md. Mostafijur Rahman, Md. Toufiqur Rahman, Jahanur Biswas  

**Link**: [PDF](https://arxiv.org/pdf/2607.02734)  

**Abstract**: Rapid growth in social media has transformed global communication by enabling fast information exchange, but it has also accelerated the spread of misinformation. Fake news, manipulated content, and provocative narratives are increasingly linked to social unrest, political instability, and mob violence. Incidents in South Asia and elsewhere demonstrate how false information disseminated via platforms such as Facebook and WhatsApp can trigger real-world harm, often spreading faster than fact-checking efforts can respond. To address this challenge, this chapter presents a multilingual, multimodal Natural Language Processing (NLP) framework for early detection of misinformation and violence-prone dynamics. A fused dataset of 138,256 Bangla and English samples was created by combining multiple benchmark datasets. The framework integrates XLM-RoBERTa for multilingual text representation, CLIP for visual embedding, and a multi-head attention mechanism for multimodal fusion, enhanced with auxiliary features such as sarcasm and geospatial metadata. Experiments on a stratified 30% subset achieved 98% test accuracy with strong precision and recall. The outcomes show the efficacy of multimodal approaches in early misinformation detection and highlight the added value of geospatial signals for anticipating real-world escalation. 

---
# SMOCS: A Streaming Framework for Simplified Deployment, Monitoring, and Optimization of ML Systems in Production 

**Authors**: Armen Kasparian, Kishansingh Rajput, Malachi Schram, John Vennekate  

**Link**: [PDF](https://arxiv.org/pdf/2607.02731)  

**Abstract**: Machine learning has demonstrated significant potential for real-time monitoring, optimization, and control of scientific facilities. However, deploying and maintaining ML models in operational environments remains a substantial engineering challenge. Each facility presents unique data protocols, non-standard formats, and infrastructure constraints, forcing teams to rebuild integration pipelines for every new application. We present SMOCS (Streaming Monitoring Optimization and Control System), a Kafka-based containerized framework that addresses this challenge through three contributions: 1) a layered abstraction over Apache Kafka that separates infrastructure from application logic, 2) a three-thread agent architecture that temporally decouples data ingestion, model training, and real-time inference enabling continuous online learning from live data streams, and 3) a configuration-driven deployment model that enables domain experts to operate ML pipelines without software engineering expertise. SMOCS is facility platform-agnostic, fault-isolated by design, and horizontally scalable through Docker containerization. The framework is publicly available as open-source software on the Jefferson Lab Github. 

---
# Signal from Space: Detecting Schools and Towers to Bridge the Digital Divide 

**Authors**: Zakarya Elmimouni, Sandor Farkas, Fares Fourati, Vladimir Daigele, Walid Mathlouthi, Mohamed-Slim Alouini  

**Link**: [PDF](https://arxiv.org/pdf/2607.02724)  

**Abstract**: Reliable internet access is essential for modern education, yet millions of school-aged children especially in developing regions remain offline due to unconnected schools. The Giga Initiative aims to connect every school to the internet, but doing so at scale requires efficient methods to map schools and assess surrounding connectivity infrastructure without relying on sparse or noisy third-party datasets. In this work, we propose a scalable, vision-only framework that uses high-resolution satellite imagery and transfer learning to address both tasks simultaneously. By adapting pre-trained object detection models to new geographical regions with minimal labeled data, we detect schools and cell towers directly from space. We then analyze the spatial relationship between detected schools and nearby towers as a proxy for connectivity availability. This purely imagery-driven pipeline enables large-scale infrastructure mapping, reduces dependency on auxiliary data, and supports data-driven prioritization of connectivity investments in underserved areas. Our approach is demonstrated on real satellite imagery from Lesotho, showing strong performance across this region. 

---
# Diagnosing Aerial-View Object Detectors with Foundational Image Generative Models 

**Authors**: Stanislav Panev, Minhyek Jeon, Vaishnavi Khindkar, Ahish Deshpande, Celso M de Melo, Shuowen Hu, Shayok Chakraborty, Fernando De la Torre  

**Link**: [PDF](https://arxiv.org/pdf/2607.02718)  

**Abstract**: Recent advances in large-scale image generative models enable photorealistic scene synthesis with controllable attributes. Beyond data augmentation, their potential as diagnostic tools for trained vision systems remains unexplored in the aerial and remote sensing domains. We introduce a synthetic diagnostic framework for aerial-view vehicle detection that combines text-guided generation, attribute-controlled editing, and automated attribute verification to construct a controllable synthetic testbed. This enables fine-grained evaluation of pretrained detectors under diverse scene types and environmental conditions that are difficult to isolate in real datasets. Across three detection architectures and three real aerial datasets, synthetic scene-wise performance trends closely match real-world weaknesses. Guided by these diagnostics, targeted supplementation with small real datasets from the identified weak categories yields improvements of up to 13% AP50 while requiring substantially fewer additional samples than non-targeted augmentation. Our results show that controlled synthetic probing can predict real-domain performance gaps and guide efficient data collection. The proposed diagnostic framework is modular and can incorporate alternative generative or vision-language models as capabilities evolve. Our code and datasets are available here: this https URL 

---
# Not All Refusals Are Equal: How Safety Alignment Fails Cybersecurity at Scale 

**Authors**: Vadym Hadetskyi, Dario Pasquini, Artem Sorokin  

**Link**: [PDF](https://arxiv.org/pdf/2607.02714)  

**Abstract**: There is no doubt that safety alignment is an essential step in LLM training. However, conceptually it does not distinguish between various domains and the level of potential harm of a query, which creates significant complications in the fields like cyber security, where a model should not be constrained by its safety circuits to accomplish the goals of legitimate, authorized operations. In this work, we share our findings from a large scale abliteration experiment on 24 open-source LLMs and show that domain-specific abliteration is achievable with standard methodology on the example of a 1T-parameter Kimi K2. Building on recent work showing that refusal in LLMs occupies a multi-dimensional subspace within layers, we find that it is also distributed widely across layers, especially in trillion-parameter MoE architectures, and so we aim to capture the part of it that represents harmful concepts in the cybersecurity domain exclusively. We also investigate the correlation between models' features and the effect of domain-specific abliteration, identifying that the type of safety training and architecture are the most reliable predictors. Finally, we classify the models into 3 \emph{abliteration susceptibility} tiers and put forward a set of conjectures as to why a particular effect from this intervention might be observed in a given model. 

---
# LLMoxie: Exploring Agentic AI for Scientific Software Development 

**Authors**: Landung Setiawan, Anant Mittal, Cordero Core, Anshul Tambay, Carlos Garcia Jurado Suarez, David A. C. Beck, Andrew J. Connolly, Vani Mandava  

**Link**: [PDF](https://arxiv.org/pdf/2607.02703)  

**Abstract**: In this paper, we describe LLMoxie, an institutional AI platform whose three-tiered architecture supports multi-cloud and on-premise inference, a LiteLLM/MLflow control plane for authentication, budgeting, PII masking, and observability, and an application augmentation layer for AI coding agents. Layered on top, an open-source RSE-Plugins ecosystem encodes accumulated RSE knowledge as a Plugin-Agent-Skill hierarchy spanning scientific Python practice, domain-specific knowledge, a six-phase research-and-implement workflow, and project lifecycle management. Scientific software is judged less by raw code quality than by whether it can be cited, audited, reproduced, and extended. Off-the-shelf AI coding agents, optimized against commercial software benchmarks, are poorly calibrated for this setting: they ignore the conventions of the scientific Python libraries they invoke, mishandle sensitive or embargoed data, and leave decision trails that are difficult to reconstruct after the fact. We report on twenty months of practice at a university-based research software engineering (RSE) center, where RSEs embedded across astronomy, earth and climate science, agriculture, and health projects worked to close this gap. We characterize the recurring infrastructure, governance, and process challenges of adopting Agentic AI inside a multi-domain RSE center, describe the platform and plugin design, and distill operational lessons from real scientific software deployments. Together, the platform and plugins shift AI coding agents from generic code generators into domain-aware collaborators that respect community norms and produce auditable provenance of technical reasoning. 

---
# S-EMBER: A Large-Scale Benchmark for Streaming Egocentric Memory Retrieval 

**Authors**: Xiaodong Wang, Xuanyi Zhao, Pedro Rodriguez, Devendra Singh Sachan, Barlas Oguz, Seungwhan Moon, Shang-Wen Li, Gargi Ghosh, Xin Dong, Wen-Tau Yih  

**Link**: [PDF](https://arxiv.org/pdf/2607.02689)  

**Abstract**: As wearable devices enable continuous first-person recording, AI assistants must reason across long time horizons to recall past experiences-a capability known as episodic memory. Current benchmarks often rely on offline evaluation with access to entire video files, failing to simulate the streaming reality of wearable intelligence. We introduce S-EMBER (Streaming Egocentric Memory Benchmark for Episodic Retrieval), a large-scale benchmark comprising 3,141 videos totaling 388 hours of organic activity captured via Ray-Ban Meta smart glasses. S-EMBER formalizes grounded streaming episodic retrieval, a paradigm shift from global offline search to causal, active recall triggered by visual events in a continuous stream. We provide 9,448 QA pairs requiring manual visual proof through precise temporal localization and supporting flexible response lengths to simulate natural human-AI interaction. Our extensive benchmarking of frontier models uncovers a localization paradox: while semantic reasoning improves with parameter scale, temporal grounding precision remains a stagnant architectural bottleneck that does not benefit from brute-force increases in model size, resolution, or frame density. S-EMBER establishes a hardware-authentic foundation for developing grounded, reliable episodic memory in the next generation of wearable AI agents. 

---
# K9-Bench: Evaluating Multimodal LLMs on Canine-Centric Videos 

**Authors**: Khush Attarde, Yusuf Ali, Megha Thukral, Divye Bhutani, Thomas Ploetz, Zsolt Kira  

**Link**: [PDF](https://arxiv.org/pdf/2607.02680)  

**Abstract**: MLLMs have shown strong zero-shot capabilities across diverse inputs such as across images, video, audio, and text. A crucial, yet underexplored, application of these models lies in understanding and modeling animal-centric scenarios. As animals are integral to millions of households, benchmarking next-generation AI models on pet-focused tasks, ranging from recognizing distress signals to enabling responsive robotic companions, is essential for building AI systems that can work alongside us. We introduce K9-Bench, a novel benchmark focused on real-world domestic dog videos, specifically targeting canine action and interaction understanding via approximately 5000 question-answer pairs across 907 videos spanning 5 distinct task categories that test long-form, canine-centric multimodal reasoning in MLLMs. To create this dataset, we propose a scalable, VLM/LLM-powered data generation pipeline that automatically mines canine-centric videos from the web and curates QA pairs requiring fine-grained, multi-hop reasoning over canine actions and temporally extended interaction sequences. We implement bias mitigation strategies designed to eliminate biases introduced by VLMs during dataset curation. Through extensive experimentation, we find that frontier MLLMs exhibit limited zero-shot performance on canine-centric tasks: although state-of-the-art closed-source models outperform open-source counterparts, they still struggle with compositional reasoning over subtle posture and interaction cues spread over long horizons. We observe that generic chain-of-thought prompting provides only modest performance for such long-horizon reasoning. Beyond a novel dataset for canine activity analysis, K9-Bench provides a general-purpose dataset construction pipeline that can be adapted to other low-data domains for quantitative analysis. Our project website is available at: this https URL. 

---
# Metronome: Bound the Cache, Keep the Beat for Real-Time Interaction Model Serving 

**Authors**: Jiaying Meng, Bojie Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.02640)  

**Abstract**: Real-time interaction models -- Moshi, MiniCPM-o, Qwen-Omni -- turn serving into a periodic real-time task: on every frame a session ingests streaming audio and must respond by a recurring wall-clock deadline, while its KV cache grows monotonically and stays pinned for the whole conversation. This regime hides a dangerous failure mode. On a real full-duplex stack, sustained load does not degrade serving gracefully: it falls off a cliff, jumping in one step from milliseconds per frame to a stalled engine when accumulated session state exhausts the KV pool. The collapse is metastable -- identical five-minute runs collapse or survive on run-to-run variance -- and silent: latency and deadline-miss metrics read healthy throughout.
We show one move restores both stability and observability: bound each session's resident state, and latency starts telling the truth. Metronome's in-engine KV window eliminates the collapse (0/20 vs. 14/20 runs across two batches) and turns per-frame latency into a monotone load signal, on which an online admission controller discovers the schedulable concurrency; without the window, the identical controller over-admits into the wall. A first-order model predicts the collapse time within a few percent on the headline model, and a quality probe validates the bound's design by ablation: the window alone is quality-free in turn-based decoding, and its few pinned attention-sink tokens are what keep free-running generation healthy. Everything is measured end-to-end on real audio, across four interaction models on one GPU. 

---
# Post-Generation Curation of Synthetic Images via Homogeneous-Heterogeneous Splitting 

**Authors**: Disheng Liu, Tuo Liang, Chaoda Song, Yu Yin  

**Link**: [PDF](https://arxiv.org/pdf/2607.02637)  

**Abstract**: Recent generative models can produce high-quality synthetic images, offering scalable training training data for data-hungry models. Existing approaches to exploiting this potential typically involve 1) training or fine-tuning generators, or 2) using lightweight post-hoc adaptation like prompt engineering or inference-time guidance, making them generator-specific and expertise-intensive. We study a complementary question: given a fixed pool of generated images, can downstream utility be improved purely by selecting an informative subset? The answer is yes. We show that effective selection must counter a structural bias of modern generators: they tend to over-produce canonical modes of each class while underrepresenting intra-class variation. Building on this insight, we split each real class into a canonical Homogeneous (HO) subset and a non-redundant Heterogeneous (HE) subset, then score synthetic images by a fidelity-diversity criterion that rewards semantic alignment while penalizing canonical redundancy. The method is generator-agnostic and requires no retraining. Across multiple benchmarks, it consistently outperforms state-of-the-art data selection baselines and matches the real-data performance with up to 40% fewer synthetic samples. The same criterion remains effective when applied on top of stronger task-tuned generators, with gains on both classification and segmentation tasks. Post-generation selection is therefore not a substitute for better generators, but a complementary mechanism for improving the utility of synthetic data. 

---
# Federated Learning for Object Detection: Enabling Collaborative Drone Learning Without Centralizing Data 

**Authors**: Daniel M. Jimenez-Gutierrez, Enrique Zuazua, Georgios Kellaris, Joaquin del Rio, Oleksii Sliusarenko, Xabi Uribe-Etxebarria  

**Link**: [PDF](https://arxiv.org/pdf/2607.02636)  

**Abstract**: Object detection is a fundamental capability for AI-driven perception in safety-critical drone and edge-vision systems, including disaster response, operational security environments, infrastructure monitoring and defense applications. Robust model performance in such environments depends on large, continuously updated datasets. However, training high-performing detectors typically requires centralizing aerial imagery, which raises privacy, regulatory, storage, and bandwidth challenges. This is especially problematic in distributed drone deployments, where visual data is generated onboard and is often impractical or undesirable to transfer to a centralized infrastructure.
In this work, we apply Federated Learning (FL) for object detection, enabling drones to improve a shared model while keeping image data local and private. We implement a federated object detection pipeline using the this http URL FL platform on the KIIT-MiTA dataset, and compare it with Single-drone and Centralized baselines using mean Average Precision (mAP) at IoU thresholds of 0.50 and 0.50-0.95. In our experiments, the proposed FL approach remains close to Centralized training while dramatically improving over Single-drone training, with the best lightweight model (YOLO26 nano), suitable for deployment even on very limited edge infrastructure, achieving relative gains of 52.89% and 67.80% in mAP@0.50 and mAP@0.50:0.95, respectively. These results show that FL enables scalable, high-performing, and privacy-preserving object detection across distributed drone fleets without data centralization. 

---
# QuantFlow: A Federated Mamba-Based Post-Transformer Foundation Model for Time-Series Forecasting 

**Authors**: Shah Nawaz Haider, Steve Austin, Arnab Barua, Sarowar Morshed Shawon, Hadaate Ullah  

**Link**: [PDF](https://arxiv.org/pdf/2607.02632)  

**Abstract**: Time-series forecasting supports decisions in finance, en-ergy, transportation, public health, and industrial monitoring. Recent foundation models improve transfer across forecast-ing tasks, but many depend on centralized data and Trans-former attention, which restricts their use for long, high-di-mensional, and privacy-sensitive signals. This paper presents QuantFlow, a probabilistic forecasting framework that com-bines inverted sequence embedding, bidirectional Mamba state-space decoders, quantile regression, and federated learning. Each variable is embedded over the complete ob-servation window, processed in forward and reverse direc-tions, and projected to five conditional quantiles. TSMixup expands temporal diversity through Dirichlet-weighted inter-polation while preserving sequence structure. Experiments cover cryptocurrency, traffic, electricity, Electricity Trans-former Temperature, influenza, and weather data. QuantFlow obtains mean squared errors of 0.2834 on ETTm1 and 0.2218 on Weather, and a 20-client non-IID deployment retains use-ful accuracy after three communication rounds without cen-tralizing raw records. The results indicate that selective state-space modelling is a promising basis for scalable, uncer-tainty-aware, and privacy-conscious time-series prediction, while also revealing limitations on irregular epidemiological signals and long-horizon generalization. 

---
# Fine-Grained Computation Offload for Off-the-Shelf Servers in Tens of Lines 

**Authors**: Bojie Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.02630)  

**Abstract**: Hardware accelerators now sit on the critical path of online serving. GPUs, FPGAs, and increasingly remote services such as hardware security modules, post-quantum KEMs, and inference servers. For fine-grained offloads (microseconds to a few milliseconds) the classic responses to the resulting stall both fail: a context switch costs as much as the offload, and a busy-wait burns the core. Overlapping the offload with other requests is the fix, and prior systems obtain it by adding concurrency: an async-framework rewrite, a new runtime or dataplane OS, or a hand-tuned point integration.
We observe that the concurrency already exists: serving concurrent requests is suspending and resuming them, so every server ships the machinery overlap needs. Overlap is then a routing problem, not a rewrite problem: submit the offload to an executor, suspend the request with the server's own deferred-response primitive, resume it on completion. Across ten off-the-shelf servers spanning every production concurrency model, this recipe takes 22-138 lines added, at most one modified, and recovers 1.2-5.4x on real hardware; the server's concurrency model and the offload's weight predict both numbers in advance, and the win is bounded by device throughput and the server's own overlap capacity. At the limit, an LD_PRELOAD fiber runtime injects the reroute into an unmodified thread-per-connection binary (17.3x) within a characterized envelope. Rerouting suspends run-to-completion atomicity; a measured taxonomy confines the hazard to unlocked shared aggregates, and a transparent page-protection detector guards exactly those, validated on stock Redis. 

---
# COMET: Combinatorial Optimization for Multiplex Editing Targets Via Constraint-Preserving QAOA 

**Authors**: Priyansh Singhal, Sumit Maheshwari, Piyush Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2607.02622)  

**Abstract**: Multiplex CRISPR-Cas9 gene editing requires selecting one guide RNA per target gene subject to cross-gene interactions: a constrained combinatorial problem that can be formulated as a Quadratic Unconstrained Binary Optimization (QUBO) and solved via the Quantum Approximate Optimization Algorithm (QAOA). The one-hot per-gene constraint is conventionally enforced by adding quadratic penalty terms to the cost Hamiltonian, but penalty coefficient selection is heuristic and penalties amplify hardware noise. An alternative is to enforce the constraint structurally via the XY-mixer, which preserves feasibility by construction. We present COMET, a systematic comparison of penalty-based and XY-mixer QAOA on a three-gene, twelve-qubit multiplex editing instance targeting the immune-checkpoint genes PDCD1, LAG3, and HAVCR2. In simulation, the XY-mixer exceeds 95% probability of the optimum by QAOA depth p=3, while three penalty variants spanning an order of magnitude in penalty coefficient remain below 6% at every depth. On IBM's ibm_kingston (Heron r2) processor, the XY-mixer's simulator-hardware energy gap stays within |0.8| across all depths, while the worst-tuned penalty variant's gap reaches +53.9. We provide an honest account of where the structural guarantee partially breaks under gate-level noise. The twelve-qubit instance is classically trivial; our contribution is a methodological comparison of constraint-enforcement strategies in a biologically motivated domain, with real-hardware validation. 

---
# The agent creates, we validate: A Lightweight Framework for Agentic Artifact Generation 

**Authors**: Yaniv Melamed, Yoni Zukerman, Michal Shechter, Miri Weissler, Ashwin Patil, Hani Neuvirth-Telem  

**Link**: [PDF](https://arxiv.org/pdf/2607.02615)  

**Abstract**: Generating structured artifacts with Large Language Models - e.g. database queries, threat framework mappings, entity schemas - is relatively straightforward; however, making them reliable enough for production deployments presents challenges. We present a lightweight framework based on a core principle: LLMs generate, we validate. This reframing shifts responsibility from generation quality to validation rigor. The framework rests on three key attributes: First, test driven generation: when tests fail, the LLM receives indicative error messages that expose why the output failed, enabling the LLM to understand its mistakes and refine subsequent attempts. Second, deterministic and LLM-based tests: deterministic tests catch heuristics that can be programmatically verified (schema, syntax, cross-reference), while LLM-based tests evaluate nuanced semantic and delicate features that resist programmatic inspection (intent alignment, logical consistency, domain correctness). Third, expert-distilled judges: LLM-based tests are calibrated to distill and replicate human expert decision distribution, transforming manual human quality gates into scalable, reusable evaluation proxies that reflect professional-grade validation standards. We demonstrate the framework on three artifact types in the security domain - KQL query generation, MITRE ATT\&CK mapping, and entity mapping - deployed in production at Microsoft Sentinel. We believe this framework can be applied beyond security to other artifact generation tasks, providing a path to reliable, high-quality outputs without sacrificing the efficiency gains of LLM generation. 

---
# Fusion: A Framework for Unified Sequential Token AdaptatIon in VisiOn TraNsformers 

**Authors**: Aravind Pradeep, Samira Nazari, Mahdi Taheri, Christian Herglotz  

**Link**: [PDF](https://arxiv.org/pdf/2607.02612)  

**Abstract**: Vision Transformers achieve strong image classification accuracy but process all image regions with nearly the same computation, even when many regions are redundant or uninformative. Recent adaptive inference methods reduce this cost by selectively compressing tokens or terminating inference early, but combining these mechanisms often causes unstable intermediate representations and accuracy degradation. We introduce Fusion, a unified adaptive inference framework that coordinates token merging, early exiting, and token pruning through a simple staged design: tokens are merged first, confidence is evaluated next, and pruning is applied only to samples that continue inference. This ordering allows the three mechanisms to operate cooperatively rather than competitively. Fusion further includes lightweight routing modules that adapt compression strength to each input and support inference-time adjustment of the accuracy--latency trade-off without retraining. On ImageNet-1k with DeiT-S, Fusion matches or surpasses state-of-the-art adaptive ViT methods at comparable compute budgets while reducing calibration error by up to $4\times$ and inference energy by $48\%$. Experiments across ImageNet-100, CIFAR-100, and ImageNette with multiple ViT backbones demonstrate consistent transferability without dataset-specific tuning. 

---
# Knowledge-Centric Information Systems 

**Authors**: Mariano Garralda-Barrio  

**Link**: [PDF](https://arxiv.org/pdf/2607.02609)  

**Abstract**: For decades, data engineering has developed mature architectural principles for integrating, governing, validating, cataloging, and serving organizational data. The rise of large language models does not eliminate these concerns; it exposes a broader version of them. Organizational knowledge is becoming executable infrastructure: systems increasingly retrieve it, assemble it, reason over it, and act on it. This paper argues that enterprise artificial intelligence (AI) systems suggest a transition toward an architectural discipline for representing, maintaining, governing, and operationally delivering organizational knowledge. We refer to this discipline as \emph{knowledge architecture}. We offer a conceptual model and taxonomy showing how classical data-engineering guarantees must be redefined when the managed unit shifts from records to knowledge artifacts: extract, transform, and load (ETL) becomes knowledge ingestion, change-data capture (CDC) becomes knowledge change detection, lineage becomes provenance, catalogs become knowledge catalogs, materialized views become knowledge views, and medallion architectures become raw--curated--operational knowledge layers. Emerging formats such as large language model (LLM) Wiki and the Open Knowledge Format (OKF) are treated as early evidence of this transition, not as its endpoint. The central claim is that knowledge architecture becomes useful when organizational knowledge ceases to be a passive information resource and becomes an operational asset used by humans, agents, workflows, and models to execute work. 

---
# AgentLTL: A Trace-Verification Framework for Measuring, Enforcing, and Training Procedural Compliance in Tool-Using LLM Agents 

**Authors**: Laïla Elkoussy, Julien Perez  

**Link**: [PDF](https://arxiv.org/pdf/2607.02599)  

**Abstract**: Tool-using LLM agents are usually evaluated by final-answer correctness or LLM judges. Neither captures how an answer was produced. In safety-critical settings, the procedure itself is part of correctness. In this paper, we introduce AgentLTL, a language derived from First-Order Linear Temporal Logic (FO-LTL) that expresses procedural rules over agent traces. It yields a deterministic, judge-free compliance score. In this framework, a single specification drives two usages. The first is harnessing: the constraints score completed traces, or gate tool calls by checking each prefix online, before execution. The second is finetuning: the score serves as a dense reward. On a benchmark spanning ordering, branching, iteration, and grounding, block-and-warn harnessing improves compliance on five of seven models. Finetuning with the same reward yields +38 and +17.5 percentage point gains in accuracy and compliance on held-out patterns, including unseen tool-name aliases. These findings are consistent with the model acquiring procedural structure rather than memorizing surface tool names and procedures. 

---
# An automated method of identifying incorrectly labelled images based on the sequences of loss functions of deep learning networks 

**Authors**: Zhipeng Zhang, Wenhui Shou, Wengting Ma, Dongjia Xing, Qingqing Xu, Li-Qun Xu, Qingxia Fan, Ling Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.02594)  

**Abstract**: Deep learning is widely applied in medical image analysis, but up to 10% of manually labelled images may be incorrect, degrading model performance. This paper proposes an automated method to identify incorrectly labelled medical images by analyzing sequences of loss functions from deep learning classification networks over multiple training epochs. Identified images can be reviewed and relabelled by experts, improving dataset quality and model performance. Two experiments validate the method on a fundus image dataset for referable diabetic retinopathy screening. In the first, 6% (648) of 10,788 gold-standard labels were intentionally flipped. The method identified 75.31% (488) of the flipped samples, with only 4.85% (492) false positives among correctly labelled samples. In the second, reviewing and correcting the 980 identified samples (9.1% of the dataset) and retraining the model improved best accuracy on an independent test set from 95.93% (with 6% label noise) to 96.50% (with 1.5% noise), approaching the ideal 96.57% (with 0% noise). The results demonstrate the method's effectiveness in improving model performance through automated label quality control. 

---
# Homer: Understanding Long-form Videos with Hierarchical Memory and Agentic Reasoning 

**Authors**: Yixin Ji, Fanghua Ye, Juntao Li, Bo Zhao, Zexuan Qiu, Zhaopeng Tu, Liefeng Bo, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.02588)  

**Abstract**: Multimodal large language models excel on short clips but struggle on hour-long videos in an online setting, where frames are processed incrementally under limited memory. Existing online methods either retain compact visual representations that lack semantic structure, or build higher-level memory stores organized around temporal proximity rather than explicit causal links, leaving multi-hop narrative reasoning to be reconstructed by the LLM at every query. We bridge this gap with \textsc{Homer}, a Hierarchical Online Memory Exploration and Reasoning framework. \textsc{Homer}'s memory mirrors the multi-scale structure of long videos, ranging from raw perception, to recurring entities, to events connected by explicit temporal and causal relations. Its agentic reasoner then explores this memory the way humans do, locating the relevant scene, looking up details, and composing the answer through multi-round memory retrieval, with a harness that verifies and corrects each step. \textsc{Homer} outperforms the previous best agent method by $+5.5$, $+10.8$, and $+4.4$ points on M3-Bench-robot, M3-Bench-web, and Video-MME-Long, and consistently lifts three various LLM backbones, indicating a model-agnostic structural capability for grounded retrieval over long videos. 

---
# Criterion-Conditional In-Context Learning: Evaluating Criterion-Shift Adaptation in Vision-Language Models 

**Authors**: Kaiyun Yang, Ruilin Yang, Zhimin Yao, J. Wang, Wei Ge  

**Link**: [PDF](https://arxiv.org/pdf/2607.02575)  

**Abstract**: Vision-language models can perform new tasks without parameter updates through in-context learning (ICL), whose core mechanism is utilizing the support set for task induction. In the standard ICL setting, once the task is induced, its decision criterion remains fixed. However, in real-world applications, many tasks exhibit a stable high-level intent, while their decision criteria shift according to specific requirements. Thus, we introduce a new setting, denoted as Criterion-Conditional In-Context Learning (CC-ICL), where models must infer the latent criterion from context and adjust predictions accordingly under fixed task semantics. To evaluate this capability, we propose two complementary metrics, Criterion Invariance and Criterion Sensitivity, capturing the model's robustness and adaptability under criterion shifts. We further construct CC-Bench, a multi-domain benchmark that supports evaluation under the CC-ICL setting. By employing a dual-level data hierarchy, CC-Bench enables legitimate ground-truth variation conditioned on the active criterion even when the task remains fixed. Experiments on CC-Bench reveal that most models exhibit a rigid boundary bias, struggling to align their decisions with the latent criterion. We also find that even a simple multi-criterion training strategy can significantly reduce this bias, improving Criterion Sensitivity and enabling 7B-scale models to surpass proprietary models without degrading general multimodal performance. 

---
# From Tensor Buffer to Distributed Memory Hierarchy: A Survey of KV Cache Management for LLM Serving 

**Authors**: Jie Li, Tongyang Wang, Yong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.02574)  

**Abstract**: The key-value (KV) cache has become a first-order memory object in LLM serving rather than a temporary per-request tensor. This survey classifies more than thirty KV-management systems and frameworks using four axes: locality, lifetime, ownership, and substrate. The axes reveal five architectural archetypes -- local-paged, disaggregated-pipeline, shared-store, memory-pool, and hybrid-tier. Once workload and hardware are fixed, ownership accounts for much of the remaining design variance among distributed systems. The survey also audits current evaluations and identifies seven missing KV-specific measurements, linking them to open problems in fault tolerance, isolation, tiered eviction, speculative decoding, MoE serving, and shared-cache semantics. 

---
# Additive Causal Construction for Transferable and Reconfigurable Cross-System Learning in Multi-Source Image Fusion 

**Authors**: Zhizhong Fu, Wei Zhou, Zhaoyang Jiang, Yulong Lin, Yifu Hou, Xiaorong Ding, Qiang Yan, Yifan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.02572)  

**Abstract**: In multi-source image fusion scenarios, heterogeneous inputs are typically driven by distinct generative mechanisms and can be viewed as a composition of multiple causal systems. However, cross-system discrepancy (CSD) and cross-system entanglement (CSE) commonly arise during the fusion process, often leading to significant performance degradation under out-of-distribution (OOD) predictions. To address the CSD and CSE issues, we propose the additive causal construction (ACC) framework, which characterizes information fusion at two levels: firstly, it establishes causal "anchors" shared among multiple systems through intervention consistency to enable causal graph transferability (CGT); and secondly, it formalizes the fusion process as causal construction and models the reliability of constructed paths through uncertainty quantification to ensure causal graph reconfigurability (CGR). Building upon this, we revisit the traditional causal representation learning (CRL) with ACC and propose ACC-CRL as a learnable instantiation of the framework. The method explores joint causal content representations across systems via content-mechanism decoupling, and performs response alignment under shared anchors to mitigate CSD. Furthermore, it incorporates structural uncertainty to adaptively regulate the fusion process, thereby suppressing unstable CSE. We conduct systematic experiments on synthetic data (ColorMNIST) and real-world multi-center medical imaging tasks (microvascular invasion (MVI) prediction). The results demonstrate that the proposed method significantly improves OOD generalization while maintaining in-distribution (ID) performance, validating the effectiveness and robustness of the ACC-CRL strategy based on mechanism alignment and uncertainty modeling in open environments. 

---
# DOSE-I: A Multimodal Biosignal Dataset of Procedural Sedation for Endoscopy -- Technical Report 

**Authors**: Jakob Garbe, Jan W. Kantelhardt, Katja Seeliger, Thomas Schmid  

**Link**: [PDF](https://arxiv.org/pdf/2607.02570)  

**Abstract**: In this document, we describe characteristics and technical details of the multimodal biosignal dataset DOSE-I of procedural sedation for endoscopy published on zenodo. The DOSE-I dataset includes 78.5 hours of recording in 171 records ranging from 6.7 to 70.8 minutes (mean: 27.5, SD: 11.6) of 281 endoscopic procedures. 1129 (median: 6 per record) transitions of consciousness and 7328 (median: 39 per record) individual sedation depth labels were recorded. In addition to clinically annotated biosignals, the DOSE-I dataset provides detailed static data about the respective study subject and metadata about the respective recordings. To further support future research, we provide details about artifact detection and preprocessed pEEG features, too. C code used for this preprocessing is provided separately via Github. 

---
# CRODA-ST: Single-Target Cross-Receiver Open-Set Radio Fingerprint Recognition 

**Authors**: Fengchong Yao, Jianbing Li, Qing Liu, Kefeng Song, Haitao Li, Song Wang, Feixiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.02567)  

**Abstract**: Radio frequency fingerprint identification (RFFI) provides a physical-layer credential for Internet of Things devices, but open-set decisions become fragile when a threshold calibrated on a source receiver is transferred to a target receiver. Receiver shift can lower the confidence of known transmitters and cause false rejection; closed-set alignment can have the opposite effect by pulling unseen target transmitters into known regions and increasing false acceptance. This letter presents CRODA-ST, a structure-first adaptation framework for singlesource single-target cross-receiver open-set RFFI. Its two components target the bottlenecks behind unreliable source-calibrated rejection: Discriminative Structure Anchoring (DSA) restores target-receiver known-class references from limited labeled target enrollment samples, and Rejection-Oriented Alignment (ROA) reduces receiver-sensitive confidence fluctuations around the anchored structure. On the WiSig ManyTx dataset, CRODA-ST reaches 0.9092 known-class accuracy, 0.9692 AUROC, and 0.9580 OSCR. Score-sweep analysis further reduces FPR90 to 0.0469. 

---
# From Raw Segmentations to Simulation-Ready Cardiac Meshes: An Automated Framework for Anatomical Reconstruction and Virtual Cohort Generation 

**Authors**: Francesco Fabbri, Martino Andrea Scarpolini, Paolo Ciancarella, Francesco Tudisco, Roberto Verzicco, Alessandro Ricci, Francesco Viola  

**Link**: [PDF](https://arxiv.org/pdf/2607.02564)  

**Abstract**: Computational models of the human heart are widely used to study electromechanical and fluid-dynamical cardiac function and to support applications such as in silico clinical trials. However, most studies remain limited to single or patient-specific anatomies, restricting the inclusion of population-level variability required for uncertainty quantification. A key challenge is translating medical-image segmentations, which may contain artifacts, mesh defects or disjoint domains, into topologically coherent geometries suitable for multiphysics simulations.
In this work, we present a semi-automatic pipeline that converts CT-based segmentations into simulation-ready cardiac meshes within a few minutes while preserving anatomical and topological consistency. Building on modern deep learning segmentation methods, the framework incorporates a template-based registration stage to regularize artifacts and enforce mesh-quality constraints. A Chamfer-distance morphing strategy deforms a high-quality template toward each segmented heart, matching individual chambers while preserving topology.
The resulting meshes are watertight, isotopological, and endowed with consistent point-to-point correspondence. The pipeline is validated on 58 healthy cardiac CT scans, including all cardiac chambers and proximal vessel segments. The resulting meshes can be represented in a unified shape space, enabling the construction of a statistical shape model of the heart and major vessels. Principal Component Analysis shows that a low-dimensional latent space efficiently captures population variability, while Gaussian Mixture Modeling enables synthetic anatomy generation. Overall, the proposed framework (released open-source) provides a pathway from raw segmentations to simulation-ready cardiac geometries, enabling anatomically consistent virtual cohorts for large-scale in silico studies. 

---
# Attention Dynamics in Diffusion Models: A Visual Analytics Framework for Human-AI Collaboration 

**Authors**: Yiran Xiao, George Legrady  

**Link**: [PDF](https://arxiv.org/pdf/2607.02563)  

**Abstract**: Diffusion-based text-to-image models can synthesize complex and highly structured visual content, yet the emergence and evolution of semantic structure remain difficult to interpret. Many existing workflows rely on aggregated attention or scalar summaries that separate temporal change from image-space evidence. To address this gap, we present a visual analytics framework for exploring attention dynamics in diffusion models: the step-indexed evolution of token-level cross-attention maps, their temporal concentration, and their spatial relationships. Our approach enables structured analysis of attention behavior across generation steps by integrating quantitative measures with data-driven stage identification in an interactive workflow. Case studies on a structured 60-prompt Stable-Diffusion-class benchmark illustrate recurring, interpretable patterns within this setting and show how linked temporal and spatial views facilitate the observation and discussion of generative processes, supporting more effective human-AI collaboration. 

---
# Double-Helix Active Geometry: LiDAR-Anchored Multi-View Depth with Selective Abstention 

**Authors**: Jinwen Wen  

**Link**: [PDF](https://arxiv.org/pdf/2607.02561)  

**Abstract**: Consumer depth sensors such as the LiDAR scanner on recent iPhones provide metric range, but their useful range is short and their returns are sparse. We present DH-Active, a lightweight, training-free geometry back-end that treats the sensor as a metric ruler rather than the sole source of depth. Near-field returns anchor the metric relative pose of two views through PnP; visually trackable samples without a valid depth return are then triangulated under that pose. A parallax/reprojection gate abstains wherever the geometry is ill-conditioned, leaving an explicit hole and a selective score instead of forcing an estimate. The measured core front end, including spiral sampling, sparse back-projection, and hole taxonomy but excluding preprocessing and multi-view recovery, runs at 1.11 ms median latency on CPU (OpenCV using 14 threads), about 38 times faster than a DINOv2-L visual branch on GPU in our timing setup. Across two iPhone captures and the public TUM RGB-D and ARKitScenes benchmarks, held-out depth is recovered at 1.4 to 6.7 percent median relative error. In a controlled ARKitScenes protocol that uses only returns within 2 m to set scale and an independent laser scan as ground truth, DH-Active achieves 64.2 percent scene-median coverage of evaluable far-field candidates at 13.4 percent scene-median relative error; direct triangulation from the device trajectory is not usable. We also report the alternatives that failed in our tests: single-frame defocus, classical focus-stack depth, defocus-LiDAR fusion, point-to-point ICP over a good visual-inertial track, and attention-to-holes resampling. A 1.26 B learned model remains more accurate after oracle scale alignment. The contribution here is narrower: metric sparse depth, explicit abstention, zero learned parameters, and near-millisecond CPU cost. 

---
# DELTAVID: Enhancing Fine-Grained Spatiotemporal Perception with Cross-Video Differences 

**Authors**: Yankai Yang, Yancheng Long, Bin Wen, Fan Yang, Tingting Gao, Han Li, Shuo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.02551)  

**Abstract**: Video multimodal large language models have made strong progress on open-ended video understanding, but they still lack precise local spatiotemporal perception. When two videos share almost the same global semantics and differ only in a short time span or a small region, current models often fail to find the change and provide reliable evidence. We propose DELTAVID, a verifiable proxy-task framework that enhances fine-grained spatiotemporal perception with cross-video differences. The key idea is to turn cross-video spot-the-difference into a trainable perception signal, where a model identifies local changes, judges temporal boundaries, and organizes spatial evidence by comparing similar videos. To make this signal scalable to train and reliable to evaluate, we further introduce DELTAVID-10K and DELTAVID-Bench, which convert controllable local differences in real videos into evidence-labeled training and test samples. Experiments show that DELTAVID substantially improves performance on cross-video difference understanding and transfers the learned local evidence ability to general video understanding benchmarks, including MMVU, MLVU, Video-MME, VideoHolmes, VideoMMMU, LVBench, TempCompass, and LongVideoBench. These results show that cross-video differences are not only an effective way to diagnose fine-grained perception failures, but also a scalable proxy supervision that moves Video MLLMs from coarse semantic understanding toward fine-grained spatiotemporal evidence reasoning. 

---
# Not Every Sync Is Safe: Calibrated DiLoCo Scheduling for Shared AI Infrastructure 

**Authors**: Maxwell Twelftree, David Lemphers, An-chi He, Yue Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.02544)  

**Abstract**: DiLoCo-style training reduces communication by letting learner islands train locally before occasional outer synchronization, making it attractive for fragmented industrial AI fleets where training shares hardware with latency-sensitive serving. The question for such fleets is when an outer merge is worth its system cost, and whether choosing \emph{which} windows to defer matters at all. Existing scheduling studies evaluate workload-aware policies against fixed-period baselines, but most omit the control that isolates timing from budget: matched random deferral, which inherits the controller's synchronization budget but is not itself deployable. This omission is consequential: across controlled stress tests and real vLLM sidecar replays, matched random ties or beats every forecast-free policy we test, so gains reported against weaker baselines cannot be attributed to window choice. We fill this gap with Workload-Aware DiLoCo (WA-DiLoCo), a score-based controller that weighs learner progress against fleet pressure, and a calibration protocol that determines when matched random can be beaten, then demonstrate that it can. In the bursty regime where calibration exposes request-overlap structure, adding a one-step EWMA burst forecast to the online controller beats matched random in real vLLM sidecar replay, reducing SLO violations from 6.54\% to 5.09\% (8 of 10 seeds, $p=0.021$); offline Calibrated-WA, a non-deployable bound, shows the remaining headroom at 4.45\% versus 6.26\%. The deployable lesson remains the protocol: report real-sidecar effect-size transfer, a no-sync load match, and a matched-random envelope before claiming serving-SLO improvement. 

---
# The Hidden Water Geography of U.S. Hyperscale Data Centers in the AI Era 

**Authors**: Gianluca Guidi, Francesca Dominici  

**Link**: [PDF](https://arxiv.org/pdf/2607.02531)  

**Abstract**: Water use by data centers is routinely reported as a single footprint, but water is consumed through two physically distinct pathways: at the site for cooling and in the power system that generates electricity. We mapped both pathways for 472 U.S. hyperscale facilities by linking facility locations to electricity regions, hydrologic basins, and water-stress data. Under baseline assumptions, operational water consumption totals approximately 300 GL yr^-1 (range 205-451 across scenarios), with electricity-related water contributing three-quarters of the total. The two pathways produce different hotspot geographies: direct cooling burdens concentrate in stressed western and south-central basins, whereas electricity-related burdens concentrate in a few eastern grid regions with fossil-heavy supply. Just 3 of 24 hosting balancing authorities account for 59% of electricity-related water. Separating pathways identifies which decisions matter where: cooling design and water sourcing locally, electricity planning and procurement regionally 

---
# PEEK: Predictive Queue-Informed KV Cache Management for LLM Serving 

**Authors**: Bing Xie, Zhipeng Wang, Masahiro Tanaka, Zheng Zhen  

**Link**: [PDF](https://arxiv.org/pdf/2607.02525)  

**Abstract**: We present PEEK, a lightweight scheduling and eviction framework for both online (streaming) and offline (batch) LLM serving; this paper focuses on the online regime. PEEK maintains an incremental radix tree over the pending queue, exposing
prefix-sharing clusters no existing engine surfaces. A low-overhead dual-walk matches the tree against the engine's prefix cache to yield longest-prefix-match for every waiting request; PEEK then admits cluster pioneers first so siblings
inherit the freshly cached prefix, a co-designed eviction hook protects blocks ancestral to queued demand, and a multi-lane stride scheduler bounds starvation. On SGLang and vLLM across five workloads up to 4$\times$H100 (DP=2 over TP=2),
PEEK delivers up to 3.0$\times$/2.6$\times$ cache hit, 7.9$\times$/7.1$\times$ TTFT, 6.7$\times$/5.5$\times$ E2E, and 3.6$\times$/4.5$\times$ throughput gains over each engine's strongest stock baseline (SGLang/vLLM), while matching
baselines within noise on workloads with no exploitable prefix structure. Wins hold as KV-cache pressure and inference parallelism scale. 

---
# SWIFT: Spatio-temporal Wavelet Integrated Forecasting Framework for Workload Traces 

**Authors**: Zeyuan Ding, Lingfeng Zheng, Dian Ding, Guangtao Xue  

**Link**: [PDF](https://arxiv.org/pdf/2607.02524)  

**Abstract**: Accurate cloud workload forecasting is pivotal for efficient resource management but remains challenging as workloads are highly volatile and prone to sudden bursts. Although wavelets preserve temporal locality, rigid fixed bases struggle with complex patterns and isolated processing neglects critical spatial dependencies. To address this, we propose SWIFT, a pure convolutional framework designed for high-efficiency workload forecasting. We introduce a Learnable Cascaded Wavelet Path that reformulates the traditional fixed wavelet bases into adaptive convolutional operators, enabling precise, data-driven feature peeling. Complementing this, our Multivariate Interaction Module sequentially models inter-variable spatial and intra-variable feature interactions to stabilize and refine noisy workload states. Extensive experiments demonstrate that SWIFT achieves SOTA accuracy with linear O(L) complexity, reducing prediction error by up to 31.04% while cutting latency by 79.74%. 

---
# AutoResearch: An Execution-Grounded Multi-Agent Framework for Reliable Research Workflow Automation 

**Authors**: Rajesh Kumar, Waqar Ali, Junaid Ahmed, Abdullah Aman Khan, Shaoning Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2607.02520)  

**Abstract**: Automated research agents increasingly generate code, retrieve literature, and draft scientific artifacts, but they often fail to verify whether generated experiments execute correctly or whether cited sources support generated claims. We present AutoResearch, an execution-grounded multi-agent framework for reliable research workflow automation. AutoResearch couples sandboxed Python/PyTorch execution, iterative code repair, citation verification, claim-support auditing, decision control, and structured \LaTeX{} artifact generation. The system treats runtime errors, citation-verification failures, and review-agent feedback as practical filtering signals for generated research artifacts. In controlled evaluations on HumanEval, MBPP, a SciCode subset, citation-validation tasks, claim-support auditing, and small end-to-end workflow stress tests, AutoResearch improves execution success, citation validity, local claim support, and workflow completion relative to directly comparable baselines. Code-oriented agents are reported separately as partial comparisons. AutoResearch is intended as a reliability-oriented research assistant, not as a fully autonomous scientist or a standalone manuscript-quality benchmark. Source Code: this https URL 

---
# GLM-5 Serving Parameter Tuning for OpenClaw: Single-Deployment MaaS Inference Optimization for Long-Context Agent Workloads 

**Authors**: Minjie Hua, Ning Wang, Peijun Yang, Kai Wang, Shiguo Lian  

**Link**: [PDF](https://arxiv.org/pdf/2607.02518)  

**Abstract**: OpenClaw requests are dominated by long, tool-augmented prefixes, including system prompts, conversation history, and tool outputs fed back into the context window. For this workload, with about 28k-30k input tokens and 500 output tokens per request, serving quality is governed by throughput, TTFT, and tail latency rather than short-prompt throughput alone.
This report studies GLM-5 serving-parameter tuning within a MaaS multi-model inference optimization architecture. The scope is the Single-Node Optimization block of the inference-optimization layer, where chunked prefill, tensor parallelism (TP), pipeline parallelism (PP), and request concurrency are tuned for one GLM-5 serving deployment; in this report, "Single-Node Optimization" denotes the architecture block, while experiments run on a two-node, sixteen-GPU cluster.
Within the tested space, the best configuration is chunked-prefill-size=3072, tp=4, pp-size=4, and max-running-requests=24. Compared with the conservative 2048/4/4/16 baseline, it increases request throughput from 0.43 to 0.48 req/s and total token throughput from 9029.64 to 9993.23 tok/s, while reducing average TTFT from 8.98 to 6.69 s and latency P90 from 40.23 to 32.64 s. Under the same hardware footprint, this corresponds to an estimated 10.4% lower serving cost per request and 9.6% lower cost per token. The results show that the optimum is workload-specific: larger chunk sizes and deeper queueing do not monotonically improve performance. We therefore recommend 3072 / tp4 / pp4 / max24 as the default OpenClaw deployment profile. 

---
# Neural-Network Inverse Design of SRF Cavities and Transmons for Bosonic Quantum Computation 

**Authors**: Joseph Yaker, Jovan Markovic, Alessandro Reineri, Doga Murat Kurkcuoglu, Silvia Zorzetti  

**Link**: [PDF](https://arxiv.org/pdf/2607.02289)  

**Abstract**: Three-dimensional superconducting radio-frequency (SRF) cavities provide exceptionally long-lived electromagnetic modes and, when coupled to nonlinear elements such as transmon qubits, become promising architectures for bosonic quantum information processing. The inverse design of such systems, i.e., recovering device geometries that produce specified electromagnetic and coupling targets, is generally a one-to-many problem. The qubit-cavity coupling strength depends sensitively on both the transmon geometry and its position within the cavity's electromagnetic field. As these systems scale up and their design parameter spaces grow, the cost of conventional iterative simulation becomes prohibitive. We present two deep neural network (DNN) approaches that address this inverse-design problem at complementary levels of the design stack. The first proposes SRF cavity geometries that produce target cavity observables. The second proposes transmon qubit designs that produce target qubit-cavity parameters -- the coupling rate, qubit frequency, and anharmonicity $(g, \nu_q, \alpha)$. The recovered candidate designs match the targets to within $\sim$5\% (cavity) and $\sim$2\% (transmon), confirmed by end-to-end re-simulation. Both approaches map desired device behavior directly to candidate designs, a fast alternative to the iterative simulation studies usually required. 

---
# Specific Domain Ontology Construction Using Large Language Models 

**Authors**: Vivian Magri Alcaldi Soares, Renata Wassermann  

**Link**: [PDF](https://arxiv.org/pdf/2606.20691)  

**Abstract**: Ontologies are useful structures to organize and maintain information that can be understood both by humans and systems. However, since their manual crafting is a laborious task, many specific domains lack reference ontologies. The outstanding ability for understanding natural language demonstrated by the Large Language Models (LLMs) has motivated their application to aid on a variety of fields, including on ontology development. This work presents the experimentation with a technique that uses LLMs in the role of domain experts to build conceptual hierarchies for a given initial concept. Twenty ontologies automatically constructed for the domain of the Brazilian maritime territory (a.k.a the Blue Amazon) using GPT-3.5 and GPT-4 were then evaluated by human experts. The models were able to construct overall coherent conceptualizations of the domain, but none of the outputs was completely satisfactory as a representation of the context without refinement. 

---
# PotatoGANs: Utilizing Generative Adversarial Networks, Instance Segmentation, and Explainable AI for Enhanced Potato Disease Identification and Classification 

**Authors**: Fatema Tuj Johora Faria, Mukaffi Bin Moin, Mohammad Shafiul Alam, Ahmed Al Wase, Md. Rabius Sani, Khan Md Hasib  

**Link**: [PDF](https://arxiv.org/pdf/2405.07332)  

**Abstract**: Numerous applications have resulted from the automation of agricultural disease segmentation using deep learning techniques. However, when applied to new conditions, these applications frequently face the difficulty of overfitting, resulting in lower segmentation performance. In the context of potato farming, where diseases have a large influence on yields, it is critical for the agricultural economy to quickly and properly identify these diseases. Traditional data augmentation approaches, such as rotation, flip, and translation, have limitations and frequently fail to provide strong generalization results. To address these issues, our research employs a novel approach termed as PotatoGANs. In this novel data augmentation approach, two types of Generative Adversarial Networks (GANs) are utilized to generate synthetic potato disease images from healthy potato images. This approach not only expands the dataset but also adds variety, which helps to enhance model generalization. Using the Inception score as a measure, our experiments show the better quality and realisticness of the images created by PotatoGANs, emphasizing their capacity to resemble real disease images closely. The CycleGAN model outperforms the Pix2Pix GAN model in terms of image quality, as evidenced by its higher IS scores CycleGAN achieves higher Inception scores (IS) of 1.2001 and 1.0900 for black scurf and common scab, respectively. This synthetic data can significantly improve the training of large neural networks. It also reduces data collection costs while enhancing data diversity and generalization capabilities. Our work improves interpretability by combining three gradient-based Explainable AI algorithms (GradCAM, GradCAM++, and ScoreCAM) with three distinct CNN architectures (DenseNet169, Resnet152 V2, InceptionResNet V2) for potato disease classification. 

---
# SiamixFormer: a fully-transformer Siamese network with temporal Fusion for accurate building detection and change detection in bi-temporal remote sensing images 

**Authors**: Amir Mohammadian, Foad Ghaderi  

**Link**: [PDF](https://arxiv.org/pdf/2208.00657)  

**Abstract**: Building detection and change detection using remote sensing images can help urban and rescue planning. Moreover, they can be used for building damage assessment after natural disasters. Currently, most of the existing models for building detection use only one image (pre-disaster image) to detect buildings. This is based on the idea that post-disaster images reduce the model's performance because of presence of destroyed buildings. In this paper, we propose a siamese model, called SiamixFormer, which uses pre- and post-disaster images as input. Our model has two encoders and has a hierarchical transformer architecture. The output of each stage in both encoders is given to a temporal transformer for feature fusion in a way that query is generated from pre-disaster images and (key, value) is generated from post-disaster images. To this end, temporal features are also considered in feature fusion. Another advantage of using temporal transformers in feature fusion is that they can better maintain large receptive fields generated by transformer encoders compared with CNNs. Finally, the output of the temporal transformer is given to a simple MLP decoder at each stage. The SiamixFormer model is evaluated on xBD, and WHU datasets, for building detection and on LEVIR-CD and CDD datasets for change detection and could outperform the state-of-the-art. 

---
