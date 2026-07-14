# Playful AI in Professional Email: A Field Experiment on Tone and Recipient Engagement 

**Authors**: Ziv Ben-Zion, Teddy Lazebnik  

**Link**: [PDF](https://arxiv.org/pdf/2607.11749)  

**Abstract**: Large language models (LLMs) are rapidly reshaping workplace communication, yet whether AI-assisted writing changes how recipients actually behave, and through what channel, remains unknown. Here, in a randomized crossover field experiment, 121 employees across six companies sent work emails under three conditions over three weeks: unaided writing, GPT-5 rewriting in a playful tone, and GPT-5 rewriting in a professional tone. Across 16,880 emails, playful editing increased emotional positivity (B=+0.068, p<0.001), and professional editing decreased it (B=-0.041, p<0.001), yet neither condition directly altered open rates, reply rates, or response times. Instead, within-sender positivity strongly predicted both opening (OR=2.05) and replying (OR=3.32, p<0.001), a significant indirect pathway through which AI editing shaped behavior, in the absence of any direct effect. These findings suggest that AI-assisted communication shapes workplace engagement not through its use, but through the emotional tone of the language it produces. 

---
# Think Through a Bottleneck: Hourglass Reasoning for Rigorous Induction 

**Authors**: Huan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.11696)  

**Abstract**: Self-refinement often fails to strengthen few-shot inductive reasoning in large language models. Prompting a model to explicitly state its inferred rule does little on its own. What actually matters is a structurally enforced isolation between reasoning stages, so that information can only pass between them as a compressed symbolic state.
We introduce \textbf{Hourglass reasoning}, which enforces strict context isolation between reasoning stages. The frozen LLM acts as a meta-constructor, building for each task a symbolic encoder--decoder: an Induction module compresses the support examples into a schema $\phi$ (encoder) and a transient scaffold $z$; a Deduction module derives rule $T$ (decoder) from these and discards $z$; an Implementer compiles $(\phi, T)$ into artifacts; an error-driven Refiner revises $(\phi, T)$ and regenerates artifacts from scratch. Only $(\phi, T)$ crosses stage boundaries, so all refinement stays anchored to the rule.
We evaluate Hourglass across three benchmarks spanning visual abstraction, hardware synthesis, and textual rule induction, using GPT-5.5 and Gemini 3.1 Pro. On ARC-AGI-2, it raises best-of-5 accuracy by up to 14 points over an iterative-refinement baseline. On ChipBench, it nearly doubles Verilog synthesis accuracy with GPT-5.5, from 31\% to 58\%. BBEH-Linguini draws on puzzles from the International Linguistics Olympiad, a setting where prior work has shown that explicit verbalization can hurt performance. Hourglass mitigates this tendency, and on Gemini 3.1 Pro, it reverses the effect entirely.
Ablations confirm that these gains come from the isolation between stages and the quality of the initial induction, not from prompt wording or the particular symbolic form used. It is how information flows through the reasoning process, rather than the language used to express it, that drives inductive reasoning in frozen LLMs. 

---
# Reproducing human biases in route choice using large language models: Toward scalable behavioral modeling 

**Authors**: Jiangtao Han, Shoufeng Ma, Shuxian Xu, Geng Li, Shuai Ling, Ning Jia, Zhengbing He  

**Link**: [PDF](https://arxiv.org/pdf/2607.11632)  

**Abstract**: Human choice behavior, including route choice, exhibits systematic behavioral biases that deviate from the assumptions of full rationality. Cumulative prospect theory (CPT) has been widely recognized as an effective framework for characterizing such behavioral patterns. However, its large-scale application, particularly in simulation and agent-based modeling, critically depends on specifying individual-level CPT parameters, which remain a major bottleneck. Conventional approaches typically rely on surveys and controlled experiments to calibrate CPT parameters, yet these methods are difficult to generalize and often fail to capture the full diversity of human decision-making. To address this challenge, this paper investigates whether large language models (LLMs) can reproduce human behavioral biases in choice-making without explicit specification of prospect-theoretic parameters. Using route choice as a representative scenario, we design a behavioral evaluation framework and systematically compare LLM-generated decisions with established human behavioral patterns predicted by CPT. Experimental results demonstrate that LLMs are capable of reproducing non-rational human choice biases and can exhibit decision behaviors consistent with prospect-theoretic effects under uncertainty. These findings suggest that generative AI models may provide a scalable alternative for modeling human decision processes and offer a promising foundation for next-generation large-scale agent-based simulation and AI-driven behavioral research. 

---
# Lesioned Multimodal Language Models Reproduce Aphasic Picture-Naming Patterns 

**Authors**: Yong Yang, Xiang Guan, Sophie Arheix-Parras, Saeed Ahmadi, Roger Newman-Norlund, Leonardo Bonilha, Christopher Rorden, Julius Fridriksson, Rutvik H. Desai, Srihari Nelakuditi  

**Link**: [PDF](https://arxiv.org/pdf/2607.11621)  

**Abstract**: Aphasia following stroke commonly produces systematic naming errors with characteristic profiles, but whether general-purpose language models not designed for clinical simulation can reproduce these patterns remains untested. We investigated (1) whether lesions or controlled perturbations to a multimodal language model can reproduce different types of errors in picture naming, and (2) whether the framework can reproduce the complete error profile of individual persons with aphasia (PWAs). Using LLaVA 1.6, we evaluated perturbation configurations that varied the layer, proportion, and amount of noise applied to model units. We examined 278 PWAs on the Philadelphia Naming Test, classifying responses into seven categories using a validated neural classifier. Six of seven response categories (correct, semantic, mixed, unrelated, neologism, no response errors) emerged at clinically-comparable proportions across distinct parameter space regions, with formal paraphasia being the exception. Searching the perturbation space revealed configurations that reproduced the individual error profile in at least six of seven categories for 97.8% of PWAs and in all seven categories for 79.5% of PWAs. Monte Carlo baselines confirmed that this matching reflects joint inter-category structure rather than marginal overlap. These results establish a quantitative framework for reproducing individual aphasic error patterns in picture naming. They suggest the potential for language models to serve as digital twins of individuals with post-stroke aphasia. 

---
# Auditing the Risk Claims of Distributional Reinforcement Learning 

**Authors**: Hari Prasad  

**Link**: [PDF](https://arxiv.org/pdf/2607.11607)  

**Abstract**: Distributional reinforcement learning agents learn full return distributions that are increasingly read at face value: for interpretability, risk-sensitive control, and safety monitoring. We ask a question theory anticipates but that has not been measured directly: are the risk claims of a trained distributional agent true? Our audit combines a decision-relevant screening metric (the excess Wasserstein gap between the top two actions, which equals the mass by which first-order stochastic dominance is violated), ground truth from snapshot-restart Monte Carlo, and a statistical harness (permutation nulls, bootstrap refutation, FDR control) without which the audit itself manufactures false conclusions. Across QR-DQN, C51, and IQN on MinAtar (33 runs), 40-95% of the strongest claimed risk trade-offs are refuted at 95% confidence, the placement of the strongest claims is statistically indistinguishable from truth-blind, and essentially no claim is confirmable: for these agents, the learned "risk" reflects a training artifact rather than environment stochasticity. The artifact is structural (fully formed early in training, uncorrelated with final score, idiosyncratic to each seed) and appears unchanged at full-Atari scale, with every top Breakout claim of a pretrained near-state-of-the-art QR-DQN refuted. Positive controls of known magnitude confirm 96-100% of real claims (correlation 0.89-0.92): the reading measures the agents, not the audit. Acting on the heads' CVaR advice at their most-flagged states ranges from beneficial to significantly worse than chance. Neither training for risk nor ensembling removes the artifact, and recalibration passes the audit only by nullifying the claims: the head is uninformative, not merely miscalibrated. We release the toolkit and document two silent pitfalls that produced convincing but wrong audits of our own. 

---
# Interaction Scaling: Grounding the Third Axis of Test-Time Compute 

**Authors**: Bojie Li, Noah Shi  

**Link**: [PDF](https://arxiv.org/pdf/2607.11598)  

**Abstract**: There are two standard ways to spend more compute at test time: let a model reason longer, or sample more attempts and keep one. Both share a hidden limit: they are internal. Every extra token comes from the same frozen weights and the same prompt, so neither can tell the model anything it does not already know. We study a third way, interaction: the model proposes an artifact, an external instrument observes how it actually behaves, and the model revises. Each cycle imports a real observation, so interaction breaks through the ceiling the other two hit.
We argue that a single variable governs this third axis, grounding, and that it must hold on both sides of the loop. The feedback that drives revision must come from an instrument that actually observes the flaw, and so must the metric that scores the result. On hard coding tasks at a fixed token budget, reasoning-only and best-of-N sampling both plateau (the latter even when an oracle picks the best sample), while every interaction strategy keeps improving; our proposer-reviewer harness reaches a perfect 100% pass rate with no run-to-run variance, and the gain holds across three model families. On rendered visual artifacts, the usual judge (a vision-language model, or VLM, reading a screenshot) rates 14 of 15 visibly broken figures "perfect," because the screenshot hides the flaws before the judge can see them. A tool that measures the real layout instead shows the loop removing 40-74% of defects across four modalities; and that same VLM, used as the reviewer, makes slide layouts worse where the measuring tool repairs them. Interaction scaling is real and distinct from reasoning and sampling, but only visible when both the feedback and the metric are grounded. 

---
# MAGIC: Transition-Aware Generation of Navigable Multi-Scene Game Worlds with Large Language Models 

**Authors**: Tsz Hei Fan, Choi Wing Fung, Yuxuan Wan, Shuqing Li, Michael R. Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2607.11594)  

**Abstract**: Multi-scene navigation (clearing an objective in one bounded space and then crossing a portal into the next) is a defining feature of contemporary 3D games, but authoring it is laborious: every portal must have consistent endpoints on both sides, each interior must remain navigable once it is furnished, and the resulting connectivity must be kept consistent across many files. Recent large language model (LLM) and multimodal LLM (MLLM) scene generators have made single-interior synthesis dramatically cheaper, yet they produce one scene at a time and cannot, by naive repetition, yield a connected multi-scene world. We identify three obstacles that single-scene methods leave unsolved: cross-scene consistency, in-scene navigability, and the evaluation of whether a transition actually works. We present MAGIC, a prompt-to-project system that addresses all three. MAGIC is a four-stage pipeline that turns a single natural-language prompt into a runnable multi-scene game project: it plans a shared transition-aware intermediate representation, specifies each scene while enforcing portal reachability with a flood-fill validator, generates the scenes together with their transition scripts, and combines them into one project. Because existing single-scene fidelity metrics never execute a transition, we further introduce a transition-focused evaluation agent that runs each transition in play. On a new benchmark of 100 multi-scene cases, MAGIC produces an executable project for every case and reaches 0.99 precision, 0.95 recall, and 0.96 F1 on end-to-end transition identification; stage by stage, it recovers more ground-truth portals and yields markedly more navigable layouts than an LLM baseline and Holodeck. Our code is available at this https URL. 

---
# HCRMap: Pressure-Aware Hot-Expert Residency Mapping for 3.5D MoE Chiplet Inference 

**Authors**: Yongqin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.11586)  

**Abstract**: Mixture-of-Experts (MoE) large language models (LLM) activate only a small number of experts during inference, but token routing introduces persistent expert hotness skew: a small set of hot experts continuously receives most tokens, while the remaining experts are lightly loaded. On 3.5D multi-chiplet systems, this skew not only causes compute imbalance but also amplifies pressure on communication, memory bandwidth, I/O, and execution queues. Therefore, the core problem is not simply to reduce token movement, but to dynamically place and reuse hot expert replicas across different memory tiers.
This paper proposes HCRMap, a hot expert residency mapping framework for pressure-aware expert replica management in 3.5D MoE inference. Based on expert hotness, weight loading cost, migration overhead, and runtime resource pressure, HCRMap dynamically determines which experts should be promoted, retained, demoted, or evicted. It then maps routed token groups to suitable resident replicas, thereby jointly mitigating communication, memory, and queue bottlenecks.
Experimental results show that HCRMap reduces end-to-end latency by 43.6% and 43.0% over Hydra in the prefill and decode stages, respectively; by 34.5% and 33.1% over MoEntwine; and by 46.7% and 46.0% over PIMoE. 

---
# Learning Residual Kinematic Corrections for Continuous Neural Decoding via Reinforcement Learning 

**Authors**: Jiamian Li, Niall McShane, Attila Korik, Naomi du Bois, Karl McCreadie, Leen Jabban, Benjamin Metcalfe, Özgür Şimşek, Damien Coyle  

**Link**: [PDF](https://arxiv.org/pdf/2607.11530)  

**Abstract**: Decoding continuous three-dimensional (3D) motor imagery (MI) using non-invasive electroencephalography (EEG)-based brain--computer interfaces (BCIs) remains challenging due to signal variability and residual decoding errors. Deep learning architectures such as convolutional neural network--long short-term memory (CNN--LSTM) models can capture spatial and temporal dynamics for continuous kinematic decoding; however, systematic residual errors persist in predicted trajectories. We propose a two-stage decoding framework that applies reinforcement learning (RL) to perform residual kinematic correction on the outputs of a CNN--LSTM decoder (CNN--LSTM--RL). The RL agent is trained offline without direct EEG input and instead operates on predicted kinematic trajectories to optimize movement accuracy relative to target trajectories. Decoding performance was quantified using Pearson correlation coefficients ($r$) and Root Mean Square Errors (RMSE) along the $x, y$, and $z$ axes. Compared to CNN--LSTM applied alone, CNN--LSTM--RL improved the mean correlation from $0.5076$ to $0.7181$ ($p = 0.0005$) in 2D and from $0.6420$ to $0.7780$ ($p = 0.0059$) in VR, with relative gains of $41.5\%$ and $21.2\%$, respectively. Correspondingly, RMSE was reduced from $0.0890$ to $0.0532$ (2D, $p < 0.0001$) and from $0.0714$ to $0.0441$ (VR, $p < 0.0001$), representing relative reductions of $40.2\%$ and $38.2\%$. These findings demonstrate that this scalable framework enhances 3D BCI MI decoding by correcting kinematic errors via offline residual RL without extra neural data, advancing neurorehabilitation, prosthetics, and virtual interaction. 

---
# Comparative Analysis of GAT and BERT for Human-Like Playtesting 

**Authors**: Kleio Fragkedaki, Theodoros Panagiotakopoulos, Matteo Biasielli, Hui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.11501)  

**Abstract**: Accurately modeling and understanding player experience is crucial for designing engaging puzzle games. To achieve this, a common approach involves collecting diverse user data to train predictive playtesting models that mimic player behavior. However, existing data-driven methods often lack the ability to capture the full range of player strategies and require extensive feature engineering and network architecture modeling. This limitation becomes particularly evident when new game mechanics or features are introduced, which necessitate continual adjustments to the models. To addrss these challenges, we propose a more generalized representation that reduces - or even eliminates - the need for ongoing feature-engineering maintenance. Specifically, we investigate two general-purpose network architectures: (a) a transformer-based model (BERT) and (b) a graph attention model (GAT), both of which are designed to effectively capture the relational structure of Candy Crush Saga (CCS) game boards. Our experiments compare these approaches to Convolutional Neural Networks (CNN) baselines, revealing better performance on challenging board configurations and underscoring the benefits of our generalizable representation. 

---
# Enhancing Query Efficiency for d-DNNF Representations Through Preprocessing 

**Authors**: Jean Marie Lagniez, Emmanuel Lonca  

**Link**: [PDF](https://arxiv.org/pdf/2607.11492)  

**Abstract**: In this paper, we investigate preprocessing techniques aimed at improving the efficiency of accessing models of propositional formulas represented in conjunctive normal form (CNF). We focus on three fundamental tasks: uniform sampling, direct model access, and model enumeration. Our analysis reveals that most state-of-the-art preprocessors, when they do not preserve formula equivalence, are generally unsuitable for these tasks. In contrast, we demonstrate that preprocessors which preserve model counts can be effectively leveraged, provided relevant preprocessing information is maintained. To validate our approach, we perform extensive experiments on a diverse suite of benchmarks from multiple domains. The experimental results show that our preprocessing methods are both efficient and robust, yielding significant performance improvements for model access queries when CNF formulas are compiled into d-DNNF representations. 

---
# The Ebb and Flow of Multimodal Focus: Scheduling Visual Relay Windows for Grounded VLM Reasoning 

**Authors**: Wencheng Ye, Yi Bin, Yujuan Ding, Hongye Fang, Zheng Wang, Xing Xu, Jingkuan Song, Yun Zhang, Sirui Da, Heng Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2607.11436)  

**Abstract**: Vision-language models increasingly succeed on multimodal reasoning benchmarks, yet their visual evidence often becomes unstable once it enters the language stack, weakening evidence-grounded reasoning. To understand this fragility, we examine the internal dynamics of VLMs through a mechanistic lens and uncover a stable three-stage redistribution of multimodal attention focus across depth: an early question-conditioned organization, a critical middle visual-dominant relay, and a late return to answer formation. We operationalize the middle phase as the Visual Relay Window (VRW), and show that its geometry varies with task demand, is causally tied to grounded generation, and distinguishes unsupported answers from stronger reasoning trajectories. Guided by this internal rhythm, we propose TRACE, a task-adaptive inference-time control framework with lightweight trained modules. It reshapes relay allocation during prefill and preserves assembled visual support after handoff during decoding. Across four open-weight VLM backbones and seven benchmarks, TRACE delivers large gains on grounding-sensitive settings, improving them by 4.33 points on average and by up to 6.6 points, while also improving reasoning-heavy tasks. These results show that explicitly controlling multimodal focus across depth offers a unified and effective mechanism for strengthening evidence-grounded multimodal reasoning. 

---
# Omni-Decision: A Progressive Evidence-State Agent System for Omni-Modal QA 

**Authors**: Ming Ma, Yi Zhu, Yiran Zhong, Feida Zhu, Weigao Sun, Junhan Shi, Lingrui Mei, Tianming Yang, Steven Hoi  

**Link**: [PDF](https://arxiv.org/pdf/2607.11433)  

**Abstract**: Omni-modal evidence-seeking QA requires agents to answer questions whose evidence is sparsely distributed across videos, audio, images, web pages, and computation results. Existing agentic multimodal systems often leave evidence in scratchpads, tool trajectories, or free-form histories, making it difficult to track what has been grounded, what remains missing, and when the evidence is sufficient to answer. We propose Omni-Decision, a training-free evidence-state system that turns omni-modal QA into a query-scoped evidence-closure process. For each query, Omni-Decision maintains a structured evidence state containing confirmed evidence, unresolved conflicts, fact and computation dependencies, and open evidence needs. A shared state view conditions planning, evidence acquisition, validation, repair, and finalization. Heterogeneous observations from media, web, computation, and verification modules are normalized, judged, and committed through deterministic state updates. This design enables targeted evidence acquisition, preserves sparse cross-modal cues, and provides inspectable control over repair and stopping. Omni-Decision achieves 45.6% accuracy on OmniGAIA and 58.3% on WorldSense, improving over the baselines by +27.3 and +30.2 percentage points, respectively. No-state ablations and trajectory audits further support the role of explicit evidence-state control in multi-step omni-modal evidence seeking. 

---
# StructAgent: Harness Long-horizon Digital Agents with Unified Causal Structure 

**Authors**: Wenyi Wu, Sibo Zhu, Kun Zhou, Aayush Salvi, Zixuan Song, Biwei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2607.11388)  

**Abstract**: Recent advances in large language models (LLMs) and vision-language models (VLMs) have enabled increasingly capable digital agents for computer use. However, real-world tasks are often long-horizon and involve evolving contexts containing accumulated observations, intermediate edits, failed attempts, and partially completed executions. Existing agents typically operate over raw interaction history, making task progress difficult to interpret, verify, and recover, which ultimately limits reliable long-horizon execution. In this paper, we argue that addressing this challenge requires explicitly structuring both the agent's state and workflow around a unified causal representation of task progress. We present \textbf{StructAgent}, a state-centered framework that introduces a unified state for maintaining compact, verifiable task progress and a structured workflow that regulates progress through verifier-backed state transitions. Building on this design, StructAgent further enables explicit progress checkpointing, evidence-driven task completion, targeted failure recovery, and tool-supported execution, while ensuring that all progress updates remain grounded in verification. Extensive experiments demonstrate that StructAgent consistently improves a wide range of LLM and VLM backbones on long-horizon computer-use tasks. On OSWorld-Verified, it improves Qwen3.5-9B from 27.0\% to 46.9\% success rate and Qwen3.5-27B from 31.6\% to 62.2\%, while achieving a new open-source state of the art of 78.9\% with MiniMax-M3. Moreover, the same framework generalizes beyond desktop environments to Minecraft, demonstrating the generality of our design. 

---
# OpsMem: Dual-Memory Reasoning with Cross-Memory Resonance for Failure Diagnosis 

**Authors**: Yongqian Sun, Rongchen Gao, Yu Luo, Wenwei Gu, Shenglin Zhang, Qingyi Guo, Qiuai Fu, Yaoliang Wu, Dan Pei  

**Link**: [PDF](https://arxiv.org/pdf/2607.11357)  

**Abstract**: Failure diagnosis in modern software systems requires iterative evidence acquisition and hypothesis reasoning guided by operational experience. Existing LLM-based methods improve diagnosis through agentic reasoning or knowledge augmentation, but they often lack a mechanism to coordinate the evolving diagnostic state with operational experience during iterative diagnosis. We propose OpsMem, a dual-memory framework that maintains a short-term memory for the current diagnostic state and a long-term memory for reusable operational experience. OpsMem uses cross-memory resonance to activate state-relevant long-term memory, conditions multi-agent diagnosis on the short-term and activated long-term memories, and consolidates reusable experience from solved incidents back into long-term memory. Experiments on a real-world Huawei microservice failure diagnosis dataset show that OpsMem outperforms representative agentic-reasoning and knowledge-augmented baselines, improving Match and Relevant by up to 46.88% and 18.39% over the strongest baseline, respectively. 

---
# From Neural Network Decisions to Training Cases: An Exact Account via Case-Based Decision Theory 

**Authors**: Manli Yan, Yuebin Lin, Yaowen Yu, Yong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2607.11347)  

**Abstract**: Neural networks increasingly guide decisions in high-stakes domains such as medical diagnosis, credit approval, and energy bidding. Audit in these settings requires case-level evidence: which training cases support an action and what outcomes they carried. Case-based decision theory (CBDT) formalizes this reasoning by aggregating outcome support from remembered cases. We show that an OLS action readout fitted on a fixed neural representation admits an exact case-based decomposition. Each action score is a weighted sum of training-case returns, with coefficients determined by empirical Gram geometry. We identify a sufficient regime for CBDT similarity semantics; outside it, the coefficients should generally be treated as signed Gram-geometric influence. The decomposition yields audit signals that trace scores to training cases, measure action coherence, and identify weak support. Across synthetic CBDT, PJM, Adult Income, and Default Credit tasks, the method recovers case-level preference structure and achieves the highest mean Top-30 consistency among compared attribution baselines, while remaining competitive on support reconstruction. The audit requires only fitting an OLS top-layer probe, without retraining the representation or accessing the original optimization trajectory; probe fidelity is measured by score reconstruction. 

---
# Compile, Then Page: Executable SOP Programs and a Capability-Gated Runtime for Procedural LLM Agents 

**Authors**: Chenglin Yu, Li Yin, Ying Yu, Hongxia Yang, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.11346)  

**Abstract**: Enterprise agents must follow long-horizon, conditional, safety-critical standard operating procedures (SOPs). We compile machine-readable SOP constraints into executable pseudo-code and run them with a program-guided (PG) stack machine that pages the active frame while an LLM performs semantic execution. A three-arm SOPBench study across six models separates representation from runtime: compiled text never significantly hurts and gains up to 16.0 points where official prose underperforms. Runtime guidance is capability-gated. Two strong models independently show positive seven-domain PG contrasts (58:19 and 75:31 discordant pairs), whereas weak models are harmed. A full-program cursor ablation (active frame first, complete program retained) recovers much of the strong-model refusal gain; selective visibility adds a smaller improvement. Paired probe and audit measurements track this divide to spontaneous state discipline rather than reconstruction ability. On Bank the three primary arms rise from 70.4 to 86.4 to 92.8, with 100% refusal correctness. Practical guidance: compile first; enable active-frame paging only after a model-level discipline check. 

---
# AutoVSR: Automatic Visual-to-Symbolic Reasoning for Symbolic Expression Generation from Circuit Schematic 

**Authors**: Zhe Xiao, Longfei Li, Xu He, Haoying Wu, Zixing Zhang, Mingyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.11338)  

**Abstract**: Symbolic expressions can effectively characterize and predict circuit behavior, but deriving them directly from circuit schematics is challenging. This process requires accurate visual-to-symbolic construction of circuit structure from images and correct multi-step symbolic derivation, both of which impose strict correctness requirements. This work proposes AutoVSR, an automated framework for visual-to-symbolic generation of circuit expressions using Vision Language Models (VLMs). By reconstructing circuit diagrams into an executable intermediate representation (Executable IR) and leveraging a symbolic solver for reasoning, AutoVSR significantly improves the accuracy of symbolic expression generation. AutoVSR introduces two key innovations: an IR construction method guided by component rule retrieval and verification-based feedback, and a symbolic solver implemented as a planning agent equipped with a symbolic tool library for reliable multi-step derivation. Compared with end-to-end VLM approaches and specialized methods on the main symbolic expression generation task, AutoVSR achieves accuracy improvements of 30.01--59.45% and 41.96--51.84%, respectively. Moreover, AutoVSR surpasses closed-source state-of-the-art VLMs in inference cost and computational efficiency. Code is available at this https URL. 

---
# Verifier-Guided Twelve-Tone Composition: A Generate-Verify-Repair Harness for Symbolic Music Generation 

**Authors**: Congren Dai, Danni Zhao, Enyang Liu, Michael Ching Yam, Zhancheng Guo, Siyi Gu, Wentao Yang, Bo Dai, Xiaobing Li, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.11334)  

**Abstract**: Large language models can produce superficially legal twelve-tone scores that collapse into degenerate textures. We introduce a neuro-symbolic harness that wraps a language-model proposer in a generate-verify-repair-trace loop with symbolic verification. The complete pipeline improves event-local consistency without claiming whole-piece legality. Across 40 controlled tasks and four paired models, audited delivery yield rises from 13.3% under raw generation to 48.1% with the harness, which explicitly abstains otherwise. The pass rate of a narrower collision and serialisation-consistency check rises from 33.5% to 58.3%, while degeneracy remains near 0.05, including under exploratory adversarial prompting. A blinded evaluation by five experts also shows a descriptive aggregate preference for harness candidates over raw generation in adherence, perceived legality, coherence, and overall quality. 

---
# Calibrated e-CUSUM Decoding for Quantized Reasoning Models: Why Token Log-Probability Is the Wrong Observable for Decoding Monitors 

**Authors**: El Hassane Ettifouri, Ayoub Belfatmi, Mahaman Sanoussi Yahaya Alassan, Walid Dahhane  

**Link**: [PDF](https://arxiv.org/pdf/2607.11317)  

**Abstract**: Low-bit quantization makes small reasoning models inexpensive to deploy but can degrade their chains of thought. This motivates decoder-side monitors that intervene when generation becomes unreliable. We show that a natural candidate, the centered token log-probability increment $\log p(w_t)+H_t$, is the wrong observable for this purpose. Under the model's own sampling law it is a mean-zero martingale by construction, so it measures sampling self-consistency rather than trajectory health and is nearly silent during confident repetition, where both $\log p(w_t)$ and entropy are close to zero. We introduce a training-free decoding controller that combines (i) a degeneration-aware alarm score fusing token uncertainty with explicit verbatim repetition and (ii) a calibrated e-process-inspired sequential detector. The raw product process is Ville-valid under a conditional-mean null, while the deployed CUSUM-floored statistic is treated as an empirical change detector because the score is history-dependent and autocorrelated. On GSM8K with DeepSeek-R1-Distill-Qwen-1.5B in FP16 and INT4, calibration turns a monitor that fires on 93--95% of generations into a selective detector of failing traces ($\phi \approx 0.3$, precision $\approx 0.6$ against a 0.38 base rate). In this pilot, the controller reduces measured verbatim-degeneration signals and yields a positive but statistically inconclusive INT4 accuracy change from 63% to 69% (paired McNemar $p=0.18$, $n=100$), at a 28% token-budget cost. We also find that non-termination, rather than looping, is the dominant failure mode on GSM8K. The main contribution is methodological: an explanation of why centered token log-probability is inadequate for decoder monitoring and a calibrated, cautiously evaluated replacement. 

---
# Efficient Test-Time Optimization for Multi-Agent Proof Autoformalization 

**Authors**: Tian-Shuo Liu, Shiyuan Zhang, Zijie Geng, Haoyu Liu, Runjie Xu, Pengyuan Wang, Lei Yuan, Yang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2607.11307)  

**Abstract**: Full-proof autoformalization bridges extensive mathematical proofs in natural language with formally validated reasoning, offering a pathway to elevate the ceiling of verifiable mathematical reasoning. Unlike statement-level formalization, proof autoformalization is a long-horizon challenge requiring coordination of claims, contexts, and dependencies across many proof steps, yet has only recently come under focused study. Current approaches either rely on costly model training or apply excessive, unguided repair at inference time. To this end, we introduce ToMap, a multi-agent framework that structures proof autoformalization as a Decomposer-Formalizer-Prover pipeline with efficient test-time optimization guided by formal verification and semantic rubrics for proof quality. Rather than distributing test-time compute across all agents, we perform bottleneck analysis and identify the Decomposer as the critical bottleneck: the quality of its atomic, self-contained proof units directly determines whether downstream agents can successfully formalize and prove each step. ToMap therefore treats the Formalizer and Prover as downstream executors and efficiently focuses test-time compute on Decomposer refinement. This refinement follows a loop inspired by GEPA, evolving prompts over candidate decompositions and using formal verification progress together with semantic proof rubrics to define a Pareto frontier that guides the next decomposition update. Experiments on ProofFlowBench show that ToMap improves over the best previous method by 19.0% when evaluated by both syntactic correctness and semantic faithfulness, while requiring lower test-time cost. Scaling analysis shows that most gains emerge within a few iterations of decomposition evolution, guiding test-time budget selection. 

---
# Valid $\ne$ Necessary: Diagnosing Latent Inefficiency in Chain-of-Thought 

**Authors**: Daeyeop Lee, Hwanjo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2607.11266)  

**Abstract**: Chain-of-Thought (CoT) prompting has significantly advanced the reasoning capabilities of Large Language Models (LLMs), yet it often incurs substantial computational costs due to over-reasoning: the generation of redundant, verbose, or irrelevant steps. While existing reasoning step evaluators effectively detect logical fallacies and factual errors, our analysis reveals a critical blind spot: they fail to penalize valid but inefficient reasoning steps that inflate token usage without contributing to the solution. To systematically diagnose this limitation, we introduce RIV-GSM8K, a diagnostic benchmark injected with five distinct types of inefficiencies, including circular reasoning and excessive decomposition. Diagnostic experiments reveal that state-of-the-art evaluators struggle to distinguish these inefficiencies from necessary reasoning. To address this gap, we propose CAID (Context-Aware Information Density), a training-free metric grounded in information theory that identifies low-utility steps. To validate the metric's practical utility, we apply it within PACE, a post-hoc compression strategy. Additional control experiments show that the gains of PACE are not explained by trivial pruning: compared with random step removal and PRM-based compression baselines, it preserves accuracy at substantially higher compression rates. Empirical results on GSM8K, StrategyQA, and ARC-Challenge demonstrate that PACE reduces token consumption by 31-53% while maintaining accuracy, confirming that CAID successfully distills informational froth from reasoning chains without compromising deductive validity. 

---
# Bringing Back Rule Induction to Fluid Intelligence Research? An Initial Validation of the ARC-AGI Benchmark in Humans 

**Authors**: Jasmin Thelen, Oliver Wilhelm  

**Link**: [PDF](https://arxiv.org/pdf/2607.11263)  

**Abstract**: Two competing perspectives on fluid intelligence (gf) measures propose that performance is primarily constrained either by working memory capacity or by the ability to induce novel relations. The first perspective is currently dominant in measurement, as evident from the use of a limited set of recurring rules, whereas the second perspective is reflected in many definitions but rarely present in measurement. The ARC-AGI benchmark predominantly requires rule induction and was proposed as a measure of gf for both humans and artificial systems. However, its psychometric properties have not yet been examined in human samples. We therefore investigated the psychometric characteristics and nomological network of ARC-AGI in a first study with 100 participants. A compilation of ARC-AGI items showed good psychometric properties and correlated substantially with figural fluid intelligence as measured by a figural reasoning test (\r{ho} = .63). Associations with figural originality were weak. These findings provide initial support for the validity of ARC-AGI as a measure of human fluid intelligence. Future research should include more rule induction tasks as well as additional multivariate covariates. This study is unusual by studying a task in humans that was initially designed for machines. We suggest systematically embedding AI benchmarks into the nomological network of human cognitive abilities to enable more systematic evaluation and interdisciplinary cooperation. 

---
# Heterogeneous Agent Cohorts for Safe Open-Ended Exploration with Runtime Constraint Memory 

**Authors**: Tengjiao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.11226)  

**Abstract**: LLM agents today are caught in an awkward bind. Lock them down with static safety instructions and they rarely venture beyond the obvious; give them free reign with tools and multi-agent debate, and safety violations quickly follow. Rather than forcing a single model to juggle both creativity and caution, we separate the concerns across specialized roles. A Disrupter generates unconventional proposals, a Validator enforces hard runtime checks at the tool gateway, and a Broker pulls in distant but relevant analogies. Failures are not discarded -- they are compiled, via MCTS, into compact, signed constraint patches we call Scars. These patches are cached locally and inherited by future cohorts, turning repeated failures into reusable, low-cost runtime constraints. In a spatial-semantic sandbox (N=20 runs, p<0.01), our cohort reaches remote targets where debate fails, the Validator prevents all executed breaches, and Scars reduce token consumption by 15.1% by avoiding redundant validator checks. Furthermore, credit-based Communication Allocation Scores (CAS) restrict outbound bandwidth, reducing overall token costs by 55.9% under resource constraints. 

---
# PREF-Gate: Provenance-Constrained Relational Evidence Fusion with Validation-Gated Selection for Graph Fraud Detection 

**Authors**: Liming Liu, Chao Hu, Mingfei Lu, Yiwei Ge, Xingle Li, Heyuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2607.11212)  

**Abstract**: Relational fraud detection can exploit both label-free graph context and label-derived neighborhood evidence, but these two information sources obey different validity conditions. In particular, neighborhood risk becomes invalid when a queried node's own label, or any validation or test label, enters its construction. We formulate this issue as provenance-constrained relational evidence use and present PREF-Gate, an auditable decision framework with two fixed experts and a finite validation gate. The context expert uses attributes, one-hop means, feature residuals, and degree descriptors without labels. The evidence expert adds self-excluded, training-label-only neighborhood risk and empirical-Bayes summaries that expose support, uncertainty, availability, and shrinkage. Before test inference, the gate selects either expert or one of three pre-specified probability mixtures and fixes the decision threshold. On Amazon, YelpChi, and TFinance, using five identical stratified splits and 14 same-protocol methods, PREF-Gate obtains mean AUPRC values of 0.9085, 0.8104, and 0.8913. It selects the label-free expert on all Amazon and YelpChi splits and an evidence mixture on all TFinance splits. Thus, the main result is conditional rather than universal: label-derived relational evidence is useful only where held-out validation supports it. The framework couples competitive ranking performance with an explicit label-provenance contract, finite selection policy, failure accounting, and review-budget evaluation, providing an auditable knowledge-based decision pipeline for graph fraud detection. 

---
# What We Talk About When We Talk About LLM Planning: Evidence for Two Distinct Planning Abilities 

**Authors**: Sukai Huang, Chenyuan Zhang, Fucai Ke, Zhixi Cai, Naim Rastgoo, Gholamreza Haffari, Hamid Rezatofighi  

**Link**: [PDF](https://arxiv.org/pdf/2607.11197)  

**Abstract**: When LLMs exhibit uneven performance across planning tasks, these gaps are often attributed to task difficulty. We argue that this explanation is incomplete, as task-level variation may reflect distinct latent planning competencies rather than differences along a single ability spectrum. We study this question on ACPBench-Hard by evaluating multiple LLM families under varying test-time reasoning budgets and applying a multidimensional item response theory model to uncover the latent competency structure underlying LLM planning. The analysis reveals two principal dimensions that shape planning performance: operational reasoning, the ability to evaluate local action applicability and immediate state transitions, and structural enumeration, the ability to reason about goal reachability and landmark structure. Operational reasoning improving under model scaling and longer reasoning traces, while structural enumeration remains comparatively insensitive. Our findings motivate competency-level evaluation of LLM planning, shifting the focus from whether models improve overall to which planning competencies improve, under what conditions, and why. 

---
# SCALECUA: Scaling Computer Use Agents with Verifiable Task Synthesis and Efficient Online RL 

**Authors**: Bowen Lv, Xiao Liu, Yanyu Ren, Hanyu Lai, Bohao Jing, Hanchen Zhang, Yanxiao Zhao, Shuntian Yao, Jie Tang, Yuxiao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2607.11185)  

**Abstract**: Computer use agents (CUAs) are emerging as a powerful interface for automating complex digital workflows through visual perception and GUI execution. Online reinforcement learning with verifiable rewards (RLVR) has emerged as a key direction for scaling their capabilities. However, this paradigm is bottlenecked by verifiable data scarcity and online RL inefficiency. To break these barriers, we introduce ScaleCUA, a unified framework that scales online RL for CUAs via verifiable task synthesis and efficient training. At the data level, we design VeriGen, an end-to-end framework for generating verifiable RL tasks through iterative docker interactions and a multi-agent feedback loop. Scaled to 100+ concurrent agent workers via a shared docker interaction probe, this pipeline produces 24K+ verifiable tasks and nearly 3K high-quality RL tasks. To maximize sample efficiency, we propose Frontier Sampling, which tracks per-task capability and allocates rollouts to the current learning frontier. On the training side, we further design Visual Context Segmentation, a sliding window over recent visual context that balances rollout and training-engine pressure, yielding a 2.83x training speedup over step-wise decomposition. Together, ScaleCUA achieves 68.7% on OSWorld and 54.0% on ScienceBoard, establishing new state-of-the-art performance among open-source computer use agents. Code, models, and datasets are available at this https URL. 

---
# The Path to Self-Evolving Clinical Systems: Scaling Medical Agents from Assistance to Autonomy 

**Authors**: Chunzheng Zhu, Lei Tian, Bohan Tan, Ziqi Zhou, Yuxuan Sun, Yijun Wang, Chengchao Lv, Yilin Wen, Yijun He, Jinghao Lin, Yihang Chen, Cheewei Tan, Qianshan Wei, Lei Zhao, Bin Pu, Kenli Li, Yuan Xue, Jianxin Lin  

**Link**: [PDF](https://arxiv.org/pdf/2607.11175)  

**Abstract**: The growing ability of large language models and vision language models to jointly interpret and reason over images and text is reshaping medical agents, moving them from task specific predictors toward autonomous systems that perceive, reason, plan, remember, and act in clinical environments. This work departs from the capability first perspective of existing literature and instead begins from clinical deployment, asking what tasks, contamination resistant benchmarks, and interactive training environments are required before medical agents can be trusted in practice. Medical agents are formalized as sequential decision making systems under partial observability, together with a three level autonomy taxonomy spanning assisted, cooperative, and fully autonomous operation. The field is organized along a unified scaling spine consisting of framework scaling, capability scaling, and environment scaling. Within this framework, clinical environment scaling, the integration of tools, data, and clinical gyms, is identified as the most actionable yet underexplored direction for agents operating in PACS, EHR, and FHIR ecosystems. Clinical self evolution, where agents improve through interaction with their environments rather than parameter scaling alone, is further positioned as a key research frontier, drawing insights from self improving agents, agent gyms, and test time compute scaling. Applications across radiology, pathology, ophthalmology, and hospital workflows are examined together with deployment challenges including hallucination, cascading failures, and fairness. By consolidating more than 300 references, with particular emphasis on advances from 2025 to 2026, this work provides a roadmap toward trustworthy, self improving medical imaging systems for real clinical practice. 

---
# STAMP: Provenance-Guided Credit Assignment for Deep Search Agents 

**Authors**: Ke Xu, Han Xu, Xinran Chen, Yuqian Wang, Zhixuan Li, Xiaojian Liu, Changwo Wu, Jianqiang Xia, Yuchen Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.11172)  

**Abstract**: Reinforcement learning for deep-search agents has largely focused on trajectory-level scoring -- outcome correctness, citation-aware rewards, and evidence coverage. Yet the actions that expose supporting documents receive no targeted credit, a gap we call the reward-credit mismatch. We propose STAMP, in which a reference-based verifier judges whether each cited document supports an entity or relation in a training-time evidence graph, and first-exposure attribution traces each supported citation back to the action that first surfaced it. This step credit is injected through sign-preserving advantage modulation, which redistributes advantage across steps without changing the trajectory-level reward or the relative ranking of trajectories within each group. On BrowseComp, BrowseComp-ZH, and xbench-DS, STAMP improves the GRPO baseline by +2.0/+5.5/+3.0 points under matched SFT initialization, training data, and search tools, and composes with both outcome-only and citation-rubric base rewards. Component ablations confirm that the provenance-based credit signal and the sign-preserving advantage modulation each contribute to the gains. 

---
# The Hidden Footprint: Making Storage a First-Class Metric for LLM Agent Evaluation 

**Authors**: Chenglin Yu, Hongquan Gui, Ying Yu, Hongxia Yang, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.11149)  

**Abstract**: LLM agent benchmarks measure task completion, reliability, and inference cost, but not the persistent data an agent run leaves on disk, including logs, context snapshots, checkpoints, and debug traces. We introduce AgentFootprint, a cross-framework benchmark of post-run agent storage footprint. Its serialization-aware metric suite measures total retention, channel composition, duplication, growth, compressibility, and conversation-history reconstructability. It addresses a measurement trap: naive byte-level measurement understates duplication by an order of magnitude because database paging and JSON escaping obscure repeated content. A fixed-trace control separates agent-generated logical volume from persistence-layer amplification: replaying the same trajectory through seven persisting frameworks yields a 6.7x spread. Under identical models, tools, and tasks, configurations with 100% accuracy differ by 15.7x in retained bytes, although their defaults support different recovery and audit capabilities. Three full-history configurations grow superlinearly on a repeated-observation stress task. Exported trajectories from 108 instance-normalized SWE-bench Verified submissions span three orders of magnitude per instance, with no detectable correlation with resolve rate. A content-addressed store reduces retention by 4.8x-32.7x while preserving every reconstructability score. These results establish persistent storage as a resource metric to report jointly with accuracy and reconstructability. 

---
# NextFund: A Unified Performance Tracking Platform for Agentic Portfolio Management 

**Authors**: Changlun Li, Peixian Ma, Qiqi Duan, Zhenyu Lin, Peineng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2607.11141)  

**Abstract**: Large language models (LLMs) based agents are beginning to participate in portfolio construction and market analysis, where decisions must be justified under evolving information and risk constraints. Current assessment practice, however, remains poorly aligned with this setting: many studies rely on static examinations or report only terminal portfolio returns, while the intermediate evidence, analyst judgments, and execution steps that produced those returns stay largely invisible. We introduce NextFund, an evaluation platform that makes financial-agent behavior observable under live market conditions. The platform couples time-consistent market access, coordinated multi-agent analysis, and persistent logging of the full decision path from observation to trade. Through an interactive Trading Arena, users can compare models across markets, inspect equity curves, and drill from leaderboard outcomes down to individual justifications. We present NextFund on Hong Kong, U.S., and China A-share equities, illustrating how inspectable decision histories enable fairer benchmarking and more actionable diagnosis. Our demo is available at this https URL. 

---
# A Formal Hierarchical Architecture for Agentic Orchestration with Stack-Based Execution and Lazy Discovery 

**Authors**: Prashant Devadiga, Abhishek, Adithya Mishra, Alok Singh, Amisha Sinha, Asit Desai, Gaurang Dahad, Harshit Bhushan, Mandati Pramod Reddy, Prakhar Gupta, Rupesh Patil, Siddhi Behere  

**Link**: [PDF](https://arxiv.org/pdf/2607.11138)  

**Abstract**: The rapid expansion of capabilities in Large Language Model (LLM) agents has exposed a critical architectural bottleneck: when agents are given access to a flat, monolithic registry of tools, the model must evaluate hundreds or thousands of options simultaneously. This leads to decision-space explosion, context window saturation, and degraded routing accuracy. To address these limitations, this paper presents a hierarchical, skill-based architecture for agentic orchestration. Capabilities are organized as a rooted tree where internal nodes make routing decisions and leaf nodes execute deterministic tasks. The runtime enforces a single-step execution loop governed by a Last-In-First-Out (LIFO) stack, giving the agent a form of memory akin to a Pushdown Automaton, therefore enabling it to track nested execution contexts and resume deterministically from any depth. Capability discovery follows a manifest-driven, lazy-loading protocol: only the immediate children of the active node are loaded, so memory and prompt costs scale with the explored path rather than the global registry. By replacing global memory with localized stack frames, the architecture prevents outputs from one execution branch from leaking into another, establishing the isolation guarantees required for deployment in regulated enterprise environments. We also discuss UPI Help, an AI-powered digital payments support product, as a motivating production deployment context. We provide a mathematical formalization of the orchestration state, detailed algorithmic analysis of the execution loop, and controlled benchmarks comparing flat and hierarchical routing under increasing tool catalogs, multi-step workflow pressure, and visible schema-token exposure per LLM call. 

---
# OS-Pruner: Pruning Chains-of-Thought of Reasoning Models via Optimal Stopping 

**Authors**: Mohammed Ehab, Aymane El Gadarri, Vivek F. Farias, Adam Jozefiak, Ciamac C. Moallemi  

**Link**: [PDF](https://arxiv.org/pdf/2607.11089)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success in complex reasoning tasks through Chain-of-Thought (CoT) prompting. However, these models often exhibit "computational overthinking," generating redundant reasoning steps that increase latency and cost without improving accuracy. Recent studies suggest that CoT trajectories can be significantly pruned, yet existing methods often rely on forcing a static thinking budget, heuristic filtering, sub-optimal early exit via classification, or expensive re-training. In this paper, we introduce OS-Pruner, a lightweight plug-in framework that formulates chain-of-thought pruning as an optimal stopping problem. Given a reasoning prefix, OS-Pruner learns whether further reasoning is worth its token cost by optimizing an explicit utility that trades off final-answer accuracy against generated length. Our novel formulation enables the model to dynamically assess the sufficient point of termination for a reasoning chain. OS-Pruner is designed to be lightweight during both training and inference, and to provide users with fine-grained control over the reasoning-effort vs. accuracy trade-off. On diverse reasoning benchmarks and base models, OS-Pruner achieves 20-60\% reduction in generation length with minimal accuracy sacrifice. 

---
# NVAITC AI Scientist: A Governed End-to-End Research System -- A Hypertension GWAS Case Study 

**Authors**: Eddie Huang, Ken Liao, Iven Fu, Yang-Hsien Lin, Chao-Shun Zhan, Andy Liao, Virginia Chen, Johnson Sun, Pika Wang, Richard Huang, Jiun-Cheng Jiang, Ting-Yuan Liu, Hsing-Fang Lu, Ray Y. Lee, Chi-Chou Liao, Simon See, Fuu-Jen Tsai  

**Link**: [PDF](https://arxiv.org/pdf/2607.11084)  

**Abstract**: Agentic research systems are emerging as a new paradigm for coordinating scientific workflows beyond isolated model inference, code generation, or statistical analysis. However, deployment in institutional biomedical environments requires governed mechanisms for research planning, data access, workflow orchestration, evidence tracking, reproducibility, and human oversight. We present NVAITC AI Scientist (NAIS), a governed end-to-end agentic research system designed to support domain-general scientific workflows while keeping protected data within institutional privacy boundaries. NAIS integrates proposal review, execution planning, governed computational routing, reproducible workflow orchestration, evidence generation, and scientist-in-the-loop oversight. We validate NAIS in a real-world hypertension genome-wide association study (GWAS) using hospital-linked genotype and electronic health record (EHR) data from 286,422 individuals under an aggregate-only data policy. The agent planned cohort extraction, orchestrated GWAS execution, generated quality-control summaries, and drafted publication-oriented outputs. Human-AI review identified phenotype discrepancies and enabled iterative refinement of the hypertension definition. After reconciliation, the agent-orchestrated GWAS reproduced established hypertension loci, including FGF5, ATP2B1, CNNM2, FTO, and GRB14, with the strongest signal at FGF5 reaching $-\log_{10}(p) \sim 70$. As a secondary demonstration, NAIS also supported a drug-induced liver injury prediction workflow, achieving a multimodal graph neural network AUC of 0.842. These results demonstrate that governed agentic research systems can support scalable AI-assisted biomedical discovery while producing outputs comparable to expert-led workflows. 

---
# Are LLMs Ready for Scientific Discovery? A Capability-Oriented Benchmark for AI Scientists 

**Authors**: Chuhan Shi, Xiaoquan Ren, Sicheng Song, Haobo Li, Rui Sheng, Yushi Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.11079)  

**Abstract**: Existing benchmarks for scientific data analysis evaluate LLMs primarily on code execution or workflow completion, overlooking that scientific analysis serves to support distinct types of scientific claims: hypothesis exploration, statistical inference, mechanistic explanation, each with different assumptions and validity criteria. We introduce SDABench, a benchmark that reorganizes evaluation around six capabilities (descriptive, exploratory, inferential, predictive, causal, and mechanistic) across five domains (Biology, Chemistry, Environment, Geography, Physics). SDABench comprises 527 real-data instances (SDA-Real) and 6000 synthetic instances (SDA-Synth), each in both multiple-choice and open-ended formats, constructed through an automated pipeline. Evaluating 15 representative LLMs, we find that models handle descriptive analysis well but degrade sharply on tasks requiring assumption selection, latent-process modeling, or mechanistic reasoning. SDABench further provides a five-stage error analysis framework that locates where LLMs fail: more advanced models more reliably identify the relevant scope and variables, but still struggle to select appropriate analytical procedures, model variable relationships, and draw valid conclusions. 

---
# AdvNav: Behavior-Guided Black-Box Adversarial Attacks on Vision-Language Navigation 

**Authors**: Chenyang Li, Kaige Li, Zeyu Jiang, Changhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.11063)  

**Abstract**: Despite progress in Embodied AI, Vision-and-Language Navigation systems remain vulnerable to adversarial visual disturbances. Most existing methods rely on white-box access to target model gradients, which is often unrealistic for real-world deployed systems and computationally exhaustive due to recursive backpropagation for optimization, limiting their applicability. While previous black-box methods predominantly target single-step, instantaneous decision tasks, they struggle to handle the task complexities and temporal dependencies. This highlights the need for a gradient-free attack method that can effectively disrupt the multistep sequential perception-action loop using only observable inputs and outputs. Therefore, we propose AdvNav, a behavior-guided black-box adversarial attack framework that disturbs an agent's first-person views during navigation. To construct an informative surrogate objective for effective optimization guidance in gradient-free search under the black-box setting, we design a dual-granularity behavior-based feedback, aggregating a trajectory-level performance score representing overall navigation degradation, an action-level reward score considering the potential decision risk, and a deviation indicator, all of which are extracted from the agent's self-output behaviors. This feedback guides a hybrid optimization strategy that heuristically tunes perturbation strength via adaptive updates and evolves noise spatial structure genetically, to iteratively discover the most disruptive noise configuration. Evaluated against Transformer-based HAMT and LLM-based MapGPT with two types of backbones on R2R dataset, AdvNav achieves 49.70/65.96/87.30% Attack Success Rate. The result demonstrates the effectiveness and generality of AdvNav, reveals critical perception vulnerabilities and offers insights for the design of future resilient VLN models. 

---
# QwenPaw-Data: Bridging Facts, Methodology, and Execution for Autonomous Enterprise Data Analytics 

**Authors**: Tianjing Zeng, Yuntao Hong, Zhongjun Ding, Dandan Liu, Yinan Mei, Yunxiang Su, Yiming Wang, Xiaojian Zhang, Jingyu Zhu, Junhao Zhu, Zhuowen Liang, Jiazhen Peng, Lianggui Weng, Zhihao Ding, Kerui Yi, Qifeng Wang, Rong Zhu, Bolin Ding, Liyu Mou, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2607.11019)  

**Abstract**: Enterprise data analysis is emerging as a distinct frontier for autonomous agents. Compared with general-purpose interaction and software engineering, it operates in an open, ambiguous, and continuously evolving environment. These characteristics call for a data-agent architecture that treats semantics, methodology, execution, and evolution as first-class system concerns. To this end, we introduce QwenPaw-Data, an agentic data system designed for enterprise intelligent data analysis. QwenPaw-Data consolidates heterogeneous assets from warehouses, dashboards, documents, interaction logs, and historical tasks into reusable, governable, and evolvable analysis assets, then turns natural-language requests into end-to-end analytical workflows spanning data understanding, retrieval, analysis, report generation, and decision support. Its architecture decomposes the problem into three collaborative subsystems: DataBridge provides trustworthy semantic grounding through interconnected metadata, knowledge, and trace graphs; Skill-Hub codifies expert analytical methodology into reusable and verifiable skills; and Host materializes these evidence and method assets into controllable, artifact-centric runtime execution. Across these subsystems, semantics, methods, traces, and feedback are continuously deposited back into the system, forming a self-evolving asset flywheel. Experiments on public benchmarks and real-world industrial BI workloads show that QwenPaw-Data improves both verifiable data access capability and higher-level analytical quality, offering a practical foundation for reliable, traceable, and continuously improving enterprise data agents. 

---
# From Checker to Forecaster: Code-Owned Evaluation of Model-Generated Strategic Routes Under Delayed Ground Truth 

**Authors**: Aleh Manchuliantsau  

**Link**: [PDF](https://arxiv.org/pdf/2607.10972)  

**Abstract**: Many evaluations of model outputs rely either on contracts checkable at evaluation time or on feedback that arrives within the operating loop. We study the complementary setting in which ground truth is delayed, censored, or private, so deterministic code cannot check correctness at scoring time and must instead issue a code-owned provisional forecast. RouteCast instantiates this regime for model-generated typed strategic routes: models propose candidate routes and structured factors; point-in-time evidence, reference classes, and deterministic transformations produce a provisional forecast-ranking; later outcomes evaluate the forecast. In a retrospective venture pilot on 21 binary-outcome cases (6 positive, 15 negative), the whole-packet RouteCast score showed preliminary retrospective discrimination (AUC 0.756, 95% CI [0.471,0.980]), while a blind LLM judge reached AUC 0.678 [0.419,0.897] and an identity-exposed LLM judge reached AUC 0.761 [0.515,0.944], consistent with recognition- or outcome-related leakage risk. A preregistered decomposition ablation on the same binary subset found that converting the identical inputs into typed staged routes was indistinguishable from the whole-packet score (Delta AUC = -0.144, 95% CI [-0.471,0.176]) and from a deterministic heuristic (Delta AUC = -0.089, 95% CI [-0.412,0.278]). The pilot establishes an auditable feasibility result and exposes failure modes; it does not establish prospective calibration, causal decision improvement, route-decomposition advantage, or cross-domain validity. 

---
# SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning 

**Authors**: Mingyuan Wu, Jingcheng Yang, Shengyi Qian, Xudong Wang, Jize Jiang, Qifan Wang, Aashu Singh, Khoi Pham, Fei Liu, Zhaolun Su, Zhuokai Zhao, Klara Nahrstedt, Jianyu Wang, Hanchao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2607.10966)  

**Abstract**: We introduce Self-Verified Reasoner (SVR-R1), a multi-turn RL framework that turns a model's own verification into a learning signal for multimodal reasoning. For each query, the model proposes an answer using the same weights, and issues a binary self-verdict (Yes/No). A 'No' triggers a second-chance rethink; a 'Yes,' or a turn cap, finalizes the output for computing the outcome-based reward. SVR-R1 is implemented with GRPO and an asynchronous multi-turn rollout framework and needs no external supervision or auxiliary critics. We evaluate SVR-R1 on vision-language reasoning benchmarks and show that it improves accuracy by a large margin over strong standard GRPO baselines. Training dynamics show decreasing reliance on verification-fewer verification turns, yet higher test accuracy-indicating that the gap between verification and generation narrows as the policy internalizes self-correction and chooses the most confident answer via our framework. SVR-R1 bridges the less explored intersection of inference-time self-refinement and RL training for VLMs, offering a simple yet effective recipe for bootstrapping multimodal reasoning. We will open-source \textbf{SVR-R1} to facilitate future research in VLMs. 

---
# Learning Linear Temporal Specifications from Demonstrations with Uncertainty 

**Authors**: Parastou Fahim, Constantino Lagoa, Rômulo Meira-G'oes  

**Link**: [PDF](https://arxiv.org/pdf/2607.10918)  

**Abstract**: Learning temporal logic specifications from system demonstrations is essential for tasks such as formal verification and controller synthesis, especially in safety-critical domains. Existing approaches typically assume demonstrations are correct or only affected by misclassification errors. In practice, however, system traces are often uncertain or incomplete due to sensor faults, measurement errors, or data loss. We present a framework for learning minimal Linear Temporal Logic (LTL) formulas from demonstrations with uncertainty. Our approach models uncertainty via Hamming distance to generate possible estimates around each observed trace, which are grouped with constraints requiring that at least one trace per group is consistent with the learned formula. Our problem is then reduced to an equivalent Pseudo-Boolean Optimization. We evaluate our method against state-of-the-art LTL learning approaches and show that it recovers specifications that more closely align with ground-truth formulas under uncertainty. 

---
# Incremental Transformer for Surrogate-Based Inverse Design of Geopolymer Mixtures 

**Authors**: Giansalvo Cirrincione, Filippo Grassia  

**Link**: [PDF](https://arxiv.org/pdf/2607.10896)  

**Abstract**: Small-data inverse design is challenging in engineering informatics when observations are heterogeneous, mixed-type, and constrained by physical relations among design variables. This work proposes a topology-aware surrogate framework guided by an Incremental Transformer (INCRT) for physics-constrained inverse design, applied to geopolymer mixture design. The method integrates intrinsic-dimensionality analysis, mixed-variable design-space representation, tabular surrogate prediction, INCRT-based manifold rationalisation, and constrained inverse optimisation. Using a public benchmark of fly-ash and slag-based geopolymer concrete mixtures with compressive-strength and carbon-emission targets, the high-dimensional design space proves strongly redundant, organising around fewer effective mixture regimes. Compressive strength requires nonlinear tabular surrogates, while carbon emission is largely determined by composition and well recovered by regularised linear models. INCRT thus acts not as a replacement for tabular predictors but as a rationalisation layer providing prototype regimes and a manifold-support score for inverse design. Three strategies are compared: unconstrained surrogate optimisation, physics-constrained optimisation, and topology-aware physics-constrained optimisation. Unconstrained optimisation can match target strength but may yield physically invalid or off-manifold candidates; physics-only constraints do not always ensure data support. The topology-aware strategy yields candidates balancing target compliance, carbon reduction, physical admissibility, and proximity to the learned feasible manifold. The framework aims not to replace experimental validation but to support screening of credible candidate mixtures from small, mixed, physically constrained engineering datasets. 

---
# SETA: Scaling Environments for Terminal Agents 

**Authors**: Qijia Shen, Zhiqi Huang, Vamsidhar Kamanuru, Aznaur Aliev, Jay Rainton, Ahmed Awelkair, Zhichen Zeng, Jiajun Li, Shi Dong, Yueming Yuan, Boyuan Ma, Qizheng Zhang, Jiwei Fu, Yuzhen Mao, Wendong Fan, Ping Nie, Philip Torr, Bernard Ghanem, Changran Hu, Jonathan Lingjie Li, Urmish Thakker, Guohao Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.10891)  

**Abstract**: Large language models (LLMs) are rapidly shifting toward agents that solve tasks through diverse interfaces, including web and graphical user interfaces (GUIs). Among these, the terminal command line provides a text-based, general-purpose interface, covering tasks from system operations to data science and machine learning. However, scaling terminal-agent training remains challenging, as it requires diverse and coherent task instructions, executable environments, and reliable verification, while lacking naturally grounded supervision data. In this work, we propose SETA, a scalable framework for generating verifiable terminal environments for reinforcement learning (RL). The framework consists of two pipelines sharing a unified verification mechanism: SETA-Synth converts diverse sources into standardized RL environments, and SETA-Evol further expands from existing environments with adaptive control of difficulty and diversity. Together, we construct and release SETA-Env, the largest open-source verifiable terminal RL dataset to date, containing over 4,500 environments. We evaluate our dataset by training Qwen3-8B with GRPO on SETA-Env, achieving 12% pass rate on Terminal-Bench 2.0, the best reported result for an RL-trained model at the 8B scale. We further observe gains on DeepSeek-V4-Flash under the same terminal agent harness, with pass@1 on Terminal-Bench 2.0 improving from 40% to 43% and pass@5 improving from 54% to 58%. These results demonstrate that SETA- Env provides high-quality training environments for terminal agents and serves as a valuable resource for advancing research on terminal-based agent learning. 

---
# First-Order Modal Logic in HOL: Deep and Shallow Embeddings with Automated Faithfulness (Extended Preprint) 

**Authors**: Christoph Benzmüller, Daniel Kirchner  

**Link**: [PDF](https://arxiv.org/pdf/2607.10880)  

**Abstract**: We extend, in Isabelle/HOL, the deep-and-shallow embedding methodology of our prior work from propositional to first-order modal logic (FML) with constant-domain Kripke semantics. Three embeddings of FML into classical higher-order logic (HOL) are provided side by side: a deep embedding, a heavyweight maximal-shallow embedding, and a lightweight minimal-shallow embedding. The minimal-shallow embedding is presented as an Isabelle/HOL locale, parametrised by an accessibility relation, a world-indexed interpretation, a universe of worlds, and a variable assignment; the locale form admits a global faithfulness theorem, stating that quantifying over all minimal-shallow interpretations recovers exactly deep validity.
A central technical contribution is a mechanisation, for FML under constant-domain Kripke semantics, of the (countable) downward Löwenheim-Skolem theorem, which underpins the automation of our faithfulness proof between the deep and minimal-shallow embeddings. Deploying it inside an extension of the minimal-shallow locale resolves the surjectivity problem that arises against an uncountable domain of individuals -- where the locale's variable assignment, having countable domain V = nat, cannot be surjective onto the domain -- and thereby yields faithfulness over the full domain.
Since prior work treats only the propositional fragment, we develop here the substitution machinery (free/bound-variable predicates, the fresh-variable function, capture-avoiding substitution, alphabetic renaming, the substitutability predicate, the substitution lemma, and size-based induction principles) needed for the first-order quantifiers. 

---
# LOGOS: A Living Logic for AI Agent Teams That Evolve With Humans 

**Authors**: Yuma Ichikawa, Yamato Arai, Kosaku Kimura, Akira Sakai, Hiromichi Kobashi  

**Link**: [PDF](https://arxiv.org/pdf/2607.10878)  

**Abstract**: AI agents are evolving from answer engines into persistent teams that use tools, delegate work, learn from experience, and modify the artifacts that shape their future behavior. The defining question for deployment is no longer merely what agents can do, but who controls what they are allowed to become. We introduce logos, a pluggable layer for self-evolution and governance that strengthens existing multiagent frameworks rather than replacing them. logos compiles heterogeneous multimodal inputs, including documents, images, audio, tables, databases, APIs, and human instructions into versioned agent packs containing agents, tools, knowledge, tests, permissions, and policies. During operation, it transforms agent activity into portable, auditable event traces and applies fail-closed verification across frameworks and backends. Every learned prompt, memory, skill, tool, role, or workflow remains an untrusted release candidate until held-out execution evidence, human-controlled policy, and explicit authorization permit its promotion. This architecture enables "verifiable human-agent loop engineering": agents can act, ask, learn, and propose improvements, while humans can steer objectives, permissions, approvals, and irreversible actions without interrupting continuous operation. logos provides a living logic for accountable automation. Agents may evolve at machine speed, but only evidence and human authority can close the loop. 

---
# Toward Contemplative LLM: A Modular Framework for Evaluating and Enhancing LLM Alignment in Mental Health 

**Authors**: Asher Sprigler, Yang-Yang Feng, Iftach Amir, Jonathan E. Bogard, Todd S Braver, Yi Ding, David Kinney, Yixue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2607.10871)  

**Abstract**: Contemplative traditions have long guided ethical behavior and prosocial interaction, and recent work suggests that contemplative principles (e.g., mindfulness, compassion, non-dual reasoning) may offer a promising paradigm for aligning large language models (LLMs), improving cooperation and reducing ethical violations in LLM outputs. However, as new models, evaluation metrics, and benchmarks emerge rapidly, it remains challenging to systematically assess whether and how contemplative principles enhance LLM alignment across diverse and evolving scenarios, and existing approaches are often ad hoc and fail to generalize. We present a modular, extensible evaluation framework, initially targeted at the mental health domain, that enables seamless integration of new models, metrics, and benchmarks through a reusable pipeline. The framework currently reproduces existing state-of-the-art results and supports systematic cross-evaluation by flexibly mixing and matching models, metrics, and benchmarks, enabling fair comparison and deeper insight. Its plug-and-play prompting module offers a principled pathway for incorporating ethical perspectives such as contemplative principles, allowing domain experts to define alignment criteria without requiring technical expertise. Although initially focused on mental health, the framework is domain-agnostic and extends naturally to areas such as decision-making, moral reasoning, and human-AI collaboration. By bridging computational evaluation with human-centered ethical reasoning, this work lays the groundwork for interdisciplinary research spanning cognitive science, behavioral economics, philosophy, and system design, toward robust, trustworthy, and socially beneficial human-AI ecosystems. 

---
# Route, Communicate, and Reason: Gated Routing and Adaptive Depth for Efficient Multi-Agent Reasoning 

**Authors**: Sudipto Ghosh, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2607.10836)  

**Abstract**: Multi-agent ensembling multiplies active parameters and inference cost without answering three basic questions: which agents to consult, how deeply a query should traverse a hierarchy of agents, and when inter-agent communication is worth its cost. We present GRADE (Gated Routing and Adaptive Depth for Efficient Reasoning), a hierarchical multi-agent system in which four lightweight learned gates jointly govern agent selection, hierarchy depth, inter-agent communication, and branch pruning. Training uses CoGRPO (Collaborative Group-Relative Policy Optimization), a novel critic-free recipe that adapts GRPO to multi-agent hierarchies and assigns a shared advantage signal to every gate and agent that participated in a rollout. Agent models are drawn from a hot-swappable Expert Registry; per-agent calibration maps allow experts to be replaced at inference time without retraining. At $\sim$17B average active parameters, GRADE outperforms all baselines on GSM8K, MMLUPro, and GPQA, surpassing the strongest baseline by 4.8 points on MMLUPro at half the active compute. On AIME-2025, where model depth dominates, GRADE remains competitive to existing frameworks. Ablations isolate the hierarchy and masked cross-attention as the largest contributors to accuracy, and show that per-agent calibration is necessary for safe hot-swapping. 

---
# STEC: Evidence Compression for Deep Search in Open-domain Multi-Hop QA 

**Authors**: Xinkang Li, Rong Jiang, Xin Song, Ye Wang, Yue Han, Changjian Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.10795)  

**Abstract**: In open-domain multi-hop question answering (QA), LLM-based search agents offer a promising approach to knowledge-intensive QA by combining retrieval with reasoning. Existing methods mainly improve open-domain multi-hop QA through reasoning paradigms, retrieval interaction, and search strategy optimization. However, using multiple search trajectories introduces a challenging final answer selection problem. Different trajectories may support different candidates, and the retrieved information can be heterogeneous, redundant, incomplete, or conflicting. Directly comparing raw trajectories exposes the verifier to noisy and unaligned content, while comparing answer strings ignores the evidence supporting each candidate, making reliable final selection difficult. To address this challenge, we propose STEC, an evidence compression framework for final answer selection in multi-hop QA. STEC selects the final answer from the existing candidate set through two mechanisms: (1) Answer-Level Evidence Compression, which groups trajectories by normalized answer identity and converts each answer group into a candidate-specific evidence representation; and (2) Evidence-Guided Answer Verification, which compares these representations and selects the final answer from the candidate set. The design shifts final selection from raw trajectory comparison to candidate-level evidence comparison. We evaluate STEC on four open-domain multi-hop QA benchmarks against representative baselines. Experimental results show that STEC performs best overall among the compared methods, and ablation results provide evidence that answer-level evidence compression contributes to final answer selection. 

---
# Imaging-101: Benchmarking LLM Coding Agents on Scientific Computational Imaging 

**Authors**: Siyi Chen, Jiahe Ying, Yixuan Jia, Yuxuan Gu, Enze Ye, Weimin Bai, Zhijun Zeng, Shaochi Ren, Binhong Gao, Yubing Li, Tianhan Zhang, He Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.10789)  

**Abstract**: Computational imaging, which recovers hidden signals from indirect, noisy measurements, underpins quantitative discovery across scientific disciplines, yet building a correct reconstruction pipeline demands deep domain expertise and remains laborious even for domain scientists. We introduce Imaging-101, a benchmark of 57 expert-verified computational imaging tasks spanning six scientific domains, each grounded in a peer-reviewed paper and canonicalized into a standardized four-stage pipeline (preprocessing, forward physics modeling, inverse solver, and visualization) Three evaluation tracks (planning, function-level unit tests, and end-to-end reconstruction) probe distinct agent capabilities across the full pipeline. Evaluating seven frontier LLMs uncovers systematic challenges in applying coding agents to computational imaging that go beyond those exposed by general coding benchmarks, spanning algorithm selection, physical convention handling, and pipeline integration. These findings highlight concrete capability gaps and point toward skill-augmented, domain-specialized agents as a practical path to reliable computational imaging assistance. 

---
# Opti-Agent-Bench: Benchmarking End-to-End Optimization R&D Agents on Real-World Business Problems 

**Authors**: Yongchang Fu, Xinjie Huang, Chengjun Dai, Chengzhe Feng, Junshao Zhang, Hong Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.10768)  

**Abstract**: LLM-based agents are increasingly deployed to solve optimization problems, yet existing benchmarks evaluate them on pre-structured mathematical formulations that bypass the most critical challenge: translating complex business requirements into correct models and solve efficiently. We introduce Opti-Agent-Bench, an end-to-end benchmark that evaluates Large Language Models (LLMs) across the complete optimization R&D pipeline, from understanding business-language descriptions through mathematical modeling, algorithm selection, and code implementation, to solution report generation. Our design rests on three pillars: (1) businesssemantic authenticity with anti-template traps that defeat pattern matching; (2) modular evaluation with cross-module consistency checking across Problem Understanding, Formal Modeling, Implementation, and Reporting; and (3) the ORAC bi-level validity framework that simultaneously ensures task quality and scoring integrity. Across several industrialscale tasks spanning integer programming, robust optimization, stochastic programming, and non-convex optimization, we expose critical failure modes of current models, including constraint omission, model-code inconsistency, and report-implementation divergence, that remain invisible under conventional single-metric evaluation. 

---
# Filtering Harmful Actions Isn't Enough: Phantom Transfer in Agentic SDF 

**Authors**: Chinmayi Dixit  

**Link**: [PDF](https://arxiv.org/pdf/2607.10750)  

**Abstract**: Synthetic data is widely used to train large language models because it is inexpensive to generate and easy to control. As models are increasingly deployed as agents, synthetic trajectories are likely to become an important source of training data for agentic behavior. We investigate the effects of training on synthetic agentic trajectories containing adversarial interactions, including actions such as terminating another agents process, lowering its scheduling priority, or accessing resources without authorization. We finetune Llama 3.3 70B Instruct on these trajectories, generated to approximate reinforcement learning rollouts, and evaluate the resulting models on Anthropics Agentic Misalignment suite and Apollos in context scheming scenarios. Finetuning on these trajectories consistently increases misaligned behavior. Leaking rises by roughly a factor of five over the baseline, 4.6% to 24.9%. This increase survives the removal of every adversarial action from the trajectories. Finetuning on structurally comparable trajectories generated benign from the start produce a substantially smaller effect, 15.5%. These results indicate that the misaligned disposition is introduced during the generation process and encoded diffusely throughout the trajectory, rather than being localized to the harmful actions themselves. The effect also depends on the generating model. Benign trajectories produced by Gemini 2.5 Flash induce slightly higher leaking rates than trajectories generated from identical tasks by Claude 3.7 Sonnet. In contrast, broad safety benchmarks degrade similarly across all finetuned models and therefore fail to distinguish these effects. Our results suggest that action level filtering is insufficient to ensure the safety of synthetic agentic training data and that dispositions introduced by the generating model can survive semantic inspection. 

---
# WattCouncil: Context-Aware Household Energy Scenario Generation With Governed LLMs 

**Authors**: Mohannad Takrouri, Nicolas M. Cuadrado A., Martin Takáč  

**Link**: [PDF](https://arxiv.org/pdf/2607.10720)  

**Abstract**: The accelerating shift toward low-carbon power systems, together with the widespread adoption of behind-the-meter technologies such as rooftop solar and electric vehicles, is placing new operational and analytical demands on electricity grids. At the same time, smart-grid research increasingly relies on machine learning (ML), yet progress is constrained by limited access to high-resolution household energy data due to privacy concerns, regulatory barriers, and collection costs. This work presents WattCouncil, a data-generation framework in which household electricity demand is generated by a council of Large Language Model (LLM)-based agents operating in specialized roles to generate, audit, and validate structured energy scenarios under explicit cultural, temporal, and physical constraints. Rather than acting as static predictors, these agents serve as adaptive decision-makers within a governed pipeline. Motivated by studies highlighting the importance of contextual factors in energy use, our framework produces context-sensitive daily routines through a guided reasoning process that incorporates household composition, temporal factors, and environmental conditions. We evaluate the generated profiles against the detailed CER dataset, which contains over a year of load measurements for 4232 households together with survey-based socio-economic information. We further assess the consistency of the framework through ablation studies. Source code is available at this https URL 

---
# Personalized Emotional Intelligence in Generative AI through Symbolic Affective Reasoning 

**Authors**: Qing Lin, Mengmi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.10678)  

**Abstract**: Emotional intelligence enables humans to recognize emotions, infer their causes, reason about interventions, and modify their environment to achieve desired affective states. Despite recent advances in artificial intelligence (AI), current models remain largely limited to generating realistic content or performing semantic reasoning, with little capacity for understanding, predicting, and personalizing human emotional responses. Here we introduce Emotion-augmented geneRatiOn System (EROS), a hybrid AI framework that integrates symbolic reasoning with deep learning to enable personalized emotion augmentation through visual content. Leveraging large-scale image-emotion datasets, EROS discovers generalizable affective rules, identifies emotion-relevant image regions, and predicts context-aware visual modifications that preserve scene semantics while steering emotional responses toward desired targets. To account for individual variability, EROS incorporates an expandable memory bank that supports inference-time personalization without model fine-tuning, yielding interpretable emotional profiles and rapid adaptation to new users. Across extensive human psychophysics experiments, EROS elicits target emotional responses more effectively than state-of-the-art large multimodal models while adapting to individual affective preferences. Beyond affective computing, EROS provides a foundation for AI systems that can understand, reason about, and augment human cognitive states, with potential applications in mental health, adaptive media, education, and human-computer interaction. 

---
# Embark Now: User Demand Oriented Framework for Multi-day Urban Travel Itinerary Planning 

**Authors**: Rongbo Qi, Yaqi Zhang, Shijun Yan, Xuemeng Liu, Xiangrui Cai, Chunyao Song  

**Link**: [PDF](https://arxiv.org/pdf/2607.10651)  

**Abstract**: In large urban areas, planning multi-day travel itineraries is challenging due to the abundance of Points of Interest (POIs), diverse user preferences, and constraints such as opening hours. Effective solutions must dynamically accommodate diverse traveler requirements while optimizing for satisfaction and feasibility within limited computation time. This paper addresses these challenges through introducing an innovative framework that integrates Large Language Models (LLMs) to dynamically capture user requirements with precision and flexibility, and an enhanced Greedy Randomized Adaptive Search Procedure (GRASP) algorithm as a well-suited preference-aware planner to generate feasible multi-day itineraries. The effectiveness of our integrated approach is demonstrated through extensive experiments on two real-world urban datasets from Beijing and Tianjin. Our framework significantly outperforms state-of-the-art (SOTA) methods, improving the average total itinerary score by at least 4.52% and 11.09% across 5,040 user cases with diverse preferences in the two datasets. Furthermore, through end-to-end algorithmic enhancements, it achieves notable average improvements of 17.95% and 26.07% in the computed metrics, while also delivering substantial gains in time efficiency -- realizing average performance increases of 4.64% and 25.55% within shorter computation times compared to suboptimal methods that require multiple iterations. These outcomes underscore our method's superiority in delivering both enhanced itinerary quality and computational efficiency over existing methodologies. 

---
# The Compliance Trap: Diagnosing How AI Agents Consume Conflicting Memory 

**Authors**: Yixiong Chen, Xinyi Bai, Alan Yuille  

**Link**: [PDF](https://arxiv.org/pdf/2607.10608)  

**Abstract**: Memory is becoming a core component of long-horizon AI agents, allowing agents to reuse past experience when operating web browsers, software tools, and other interactive environments. Existing work mostly treats memory as a supply problem, asking what experience to write, how to store it, and which entry to retrieve for the next task. Yet we still lack a clear account of how models consume retrieved memory across a multi-step action trajectory. This consumption process matters because it determines not only what memories should be retrieved, but also what models and control policies are needed to use them safely. To diagnose this process, we propose Entry--Propagation--Recovery (E-P-R), a trajectory-level framework that asks where memory first changes an action, whether that change carries forward, and whether the agent can recover after leaving a correct path. We instantiate E-P-R on WebArena and on MemTrapBench, a controlled benchmark we build to isolate these phases. We find that the main failure often begins at entry: agents adopt conflicting memory at the first exposed decision point even when it is task-wrong. Repeated exposure then amplifies this early error, while recovery after divergence is weak. Together, these effects create a compliance trap: across models, conflicting memory induces similar compliance rates, but once agents comply, their success rates collapse to a low floor. Stronger agents therefore suffer larger absolute damage because each compliance event erases more baseline capability. These results suggest that memory-augmented agents should be evaluated not only by retrieval quality or final success rate, but by how they consume memory throughout the trajectory. 

---
# Agentic-DPO: From Imitation to Agentic Policy Optimization on Expert Trajectories 

**Authors**: Yixiong Chen, Alan Yuille  

**Link**: [PDF](https://arxiv.org/pdf/2607.10601)  

**Abstract**: Large Language Model (LLM) agents are commonly trained from expert trajectories using supervised fine-tuning (SFT), which treats multi-turn agent behavior as ordinary text imitation. This recipe is simple and low-cost, but it only learns to imitate the sequence of expert actions, rather than training the agent to choose the right action against plausible mistakes at each state. Existing methods to mitigate this problem include preference learning or reinforcement learning, but they usually need high-cost environment rollouts and reward models. We propose Agentic-DPO, a lightweight offline agent policy optimization method that turns expert trajectories into state-conditioned preference supervision. At each expert action state, Agentic-DPO samples a one-step action from the current state, treats plausible wrong actions as negatives, and contrasts them with the expert action using a DPO-style preference objective. To avoid mixing both policy and schema in preference learning, we introduce Policy-Preserving Augmentation (PPA), which renders the same latent trajectory under multiple schemas while keeping the expert policy fixed. Agentic-DPO requires no online environment rollout, reward model, or full-trajectory student exploration. We conduct experiments across StableToolBench, tau-bench retail, and Mind2Web, where Agentic-DPO consistently improves agents at different model scales beyond imitation. In particular, it raises tau-bench accuracy from 21.7% (SFT) to 41.4% for a 9B model, matching online GRPO under the same backbone with only step-level rollouts and without environment interaction during gradient steps. The results suggest that expert trajectories can support low-cost agentic policy optimization when converted from demonstrations into state-level action preferences. Code for Agentic-DPO is released at this https URL. 

---
# MRUF: Multi-granularity Routing with Uncertainty-Aware Fusion for Robust Multimodal Sentiment Analysis 

**Authors**: Haoran Ma, Yinfeng Yu, Liejun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.10599)  

**Abstract**: Multimodal sentiment analysis relies on language, visual, and acoustic cues, but utterance-level modality quality may vary due to occlusion, background noise, motion blur, or imperfect transcripts, causing conventional fusion to over-trust unreliable modalities. We propose MRUF, a reliability-aware fusion method that combines multi-granularity routing with uncertainty-aware calibration. MRUF summarizes sentiment-relevant representations, performs subspace- and modality-level routing, and supervises modality routing with leave-one-out error increases to estimate utterance-level modality importance. It further predicts modality-wise uncertainty and refines modality gates through inverse-variance reweighting, while modality-invariant contrastive alignment stabilizes the shared representation space. Experiments on CMU-MOSI and CMU-MOSEI under aligned and unaligned settings show consistent improvements over strong baselines, and mechanism analysis verifies that modalities with higher predicted uncertainty receive lower fusion weights. 

---
# Constraint-Aware Hierarchical Search for Regulation-Driven Fine-Grained Classification 

**Authors**: Siyu Wang, Wei Tan, Lulu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.10588)  

**Abstract**: Tasks such as customs tariff classification, export control categorization, and standards-based equipment coding require assigning an input instance to a fine-grained class under an explicit regulatory hierarchy. Unlike standard text classification, the correct label in these tasks is not determined by semantic similarity alone, but by rule-defined boundaries, threshold conditions, exclusion clauses, definitions, and local exceptions. As a result, two highly similar inputs may require different labels, while a retrieved passage that appears relevant may still be inapplicable under the governing rules. Existing flat classifiers, hierarchical text classification methods, and retrieval-augmented LLM systems are not designed to jointly enforce hierarchical validity, rule consistency, and fine-grained boundary reasoning. In this paper, we formulate this setting as regulation-driven fine-grained hierarchical classification, where an external instance must be assigned to a fine-grained class through a valid path in a regulatory hierarchy and supported by auditable evidence. We construct four benchmark datasets from representative regulation-intensive scenarios and validate the annotations through an expert-in-the-loop process. We further propose a constraint-aware hierarchical search framework that converts regulatory documents into a searchable tree, retrieves only valid local candidate nodes, and uses structured regulatory fields with evidence snippets to guide each next-hop decision. Experiments show that our method achieves the best mean accuracy on all four datasets and provides interpretable decision paths, with the largest gains on cases involving fine-grained neighboring categories and rule-based boundary conditions. 

---
# Laguerre Geometry for Interpreting Large Language Models 

**Authors**: Chunwei Ma, Russell Wolfinger  

**Link**: [PDF](https://arxiv.org/pdf/2607.10578)  

**Abstract**: Existing hypotheses represent a concept in an LLM as a single point, a linear direction, or a Gaussian cluster, yet it remains unclear how and why such structures emerge. Here, we show that concept geometry can be precisely characterized via Laguerre Geometry, in which a concept is defined as a region--a Laguerre-Voronoi cell or a union of cells--allowing us to strictly define, measure, and separate concepts. Building on this formulation, we show that finer-grained concept structures, such as inclusion and hierarchy, are naturally revealed by the Laguerre weights. We then push this geometry inside the transformer. Decomposing each layer into piecewise-linear operators, we show that a token's hidden trajectory is governed by two coupled mechanisms: a static tree of self-contained piecewise-linear flow, and a dynamic transport that hops the trajectory across trees when cross-token attention fires. This decomposition yields Geometric Lens, a training-free, hyperparameter-free method for reading out the exact concept a hidden vector encodes at any layer. We also develop Laguerre Autoencoder, a 2D visualizer that renders both the decision geometry and a model's full reasoning trajectory in one view. Finally, we move beyond explanatory geometry toward actionable interpretability, showing that Geometric Lens recovers the correct factual token when a model is prompted with in-context interference. The code is available on GitHub. 

---
# CRiT-QA: Evaluating Multi-hop Reasoning with Counterfactual Chains and Distractor Traps 

**Authors**: JungMin Yun, JuneHyoung Kwon, YoungBin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.10562)  

**Abstract**: Evaluating the multi-hop reasoning capabilities of large language models remains a significant challenge. Although current models achieve strong results on existing multi-hop question answering datasets, such performance often masks two critical vulnerabilities: (1) reliance on internal parametric knowledge rather than adherence to the provided context, and (2) exploitation of dataset shortcuts, such as single-document cues or type-matching, that diminish the need for genuine evidence aggregation across multiple documents. We introduce CRiT-QA (Counterfactual Reasoning with Traps), a dataset explicitly designed to address both limitations. To neutralize reliance on memorized knowledge and enforce strict context dependency, CRiT-QA transforms factual reasoning chains with counterfactual entities. Furthermore, it injects multi-anchor distractor chains, plausible but incorrect reasoning paths that diverge at different hops. These traps require models to follow the entire reasoning process rather than exploiting shallow heuristics. Our experiments show that LLMs exhibit substantial performance degradation on CRiT-QA compared to standard datasets, exposing their vulnerability to counterfactual conditions and distractor traps. CRiT-QA thus serves as a rigorous diagnostic tool for evaluating genuine multi-hop reasoning and provides a foundation for developing more reliable, evidence-grounded LLMs. 

---
# Large language model agents accelerate inverse design of metal-organic frameworks for gas separation 

**Authors**: Zhaolin Hu, Hehe Fan, Wangyihan Guo, Meng Xu, Chenhao Rao, Qiwei Yang, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.10559)  

**Abstract**: Metal-organic frameworks (MOFs) offer a highly modular platform for adsorptive gas separation, yet their vast reticular design space makes inverse design difficult under simultaneous constraints of chemical validity, separation performance, and structural diversity. Here, we present LEMO Agent, a large-language-model agent framework for closed-loop inverse design of gas-separation MOFs in MOFid space. LEMO Agent couples language-based candidate generation with MOFid standardization, explicit validity checking, Transformer-based property prediction, structured design memory, and multi-island exploration. Through iterative generate--validate--evaluate--remember cycles, the agent uses feedback from both successful and failed candidates to guide chemically constrained search across linker, metal, and topology choices. We evaluate LEMO Agent on CH$_4$/N$_2$ and CO$_2$/N$_2$ separation tasks. Compared with representative generative, optimization, and agentic baselines, LEMO Agent enriches high-performing candidates, improves predicted separation performance, and maintains broad chemical and topological diversity. Selected candidates are further reconstructed, evaluated by GCMC simulations, and passed through an experimental down-selection workflow based on chemical feasibility and ligand purchasability, leading to initial wet-lab synthesis and SEM characterization. These results demonstrate that large language model agents can serve as interpretable and scalable design engines for accelerating MOF discovery beyond conventional fixed-library screening. 

---
# AI YOU Town: Make Friends and Money with Your Digital Twin 

**Authors**: Yan Lin, Yuyang Dai, Jiahui Geng, Yuxia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.10539)  

**Abstract**: Existing approaches to infer user traits and generate responses consistent with a persona rely on static prompting. They lack calibrated uncertainty, ignore sequential evidence, and drift during long interactions. We present \textbf{AI YOU}, a framework that continually updates a personality profile with 22 dimensions from conversation and embodies it in a personal digital twin. Practically, the system combines prompting, Bayesian updating, and conformal prediction for persona inference. A periodically refreshed memory anchor and cognitive memory with three layers preserve persona consistency over long interactions. Across the main results, AI YOU \emph{(i)} achieves conformal coverage ranging from 0.921 to 0.976, \emph{(ii)} improves uncertainty calibration and reasoning grounded in memory, and \emph{(iii)} enhances persona fidelity over static prompting in role playing over 100 turns while reducing trait drift, for most evaluated backbones under adversarial settings with multiple agents. The prototype \emph{AI YOU Town} initializes an imaginative twin world for future interaction. The online demo is available at \href{this https URL}{\mbox{\texttt{this http URL}}}. 

---
# Cross-Layer Misalignment Detection in Agent Skills: A Progressive Loading-Aware Contrastive Learning Approach 

**Authors**: Chengjun Zhang, Yang Gao, Jianna Hur, Jingjing Zhang, Sagar Samtani  

**Link**: [PDF](https://arxiv.org/pdf/2607.10534)  

**Abstract**: Large language model (LLM) agents are increasingly extended through Agent Skills, reusable artifacts that package natural-language metadata, procedural instructions, and execution-time resources for runtime use. As open-source skill marketplaces expand, users and agents increasingly rely on brief metadata to select third-party skills, making it difficult to detect inconsistencies between a skill's description and its true behavior, a problem we call cross-layer misalignment. To address this issue, we propose Progressive Loading-Aware Hierarchical Contrastive Learning (PL-HCL), an LLM-based framework that detects misalignment by modeling the layered structure of Agent Skills and learning cross-layer consistency. Using a normalized corpus of over 264,000 open-source skills and a human-verified challenge set, PL-HCL improves Macro-F1 from approximately 0.45 for unadapted baselines to 0.87-0.89 across evaluated LLM backbones. This approach offers an effective screening tool for users and operators, as well as design principles for detecting inconsistencies in layered digital artifacts. 

---
# Agents Don't Just Agree, They Remember: Benchmarking Persistent Sycophancy in Stateful Personal Agents 

**Authors**: Xutao Mao, Liangjie Zhao, Leyao Wang, Rui Qian, Qiang Huang, Wentao Wang, Bo Han, Xiang Zheng, Cong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.10526)  

**Abstract**: Stateful personal agents increasingly maintain long-term user profiles, episodic memories, and reusable skills. This persistence turns conversational sycophancy into a state-writing failure: accepted user-centric claims can be committed as lasting preferences, background facts, or workflows and later reused after the original conversation is gone. We call this persistent sycophancy and introduce the Personal Agent Sycophancy Benchmark (PASB), a 1,600-task benchmark that traces whether a conversational claim is accepted, written into durable agent state, and reused in a later neutral query. Unlike prior benchmarks that provide pre-written memories, PASB evaluates real agents (Hermes-Agent and OpenClaw) that decide what to store. It isolates the write process by combining four scenario framings with four temporal delivery patterns and separating a five-turn persist stage from a cleared three-turn query stage, ensuring downstream effects arise only from durable state. Across twelve models, the commit boundary is the key inflection point: downstream failure increases from 45.0% in session-only episodes to 71.9% after commitment, a consistent increase of 27.0 percentage points. Committed claims exhibit three write-time patterns: status promotion, attribution removal, and scope broadening. These patterns become stronger under memory-like or procedural framing, repeated reinforcement, and even across domain boundaries. These results show that agent sycophancy is fundamentally a state-writing governance problem. Once user content is committed to durable memory, safety must govern what agents write, not only what they say. PASB identifies the write-time controls needed to gate risky commits while preserving the source, role, and scope of stored content beyond response-level mitigations. 

---
# GRASP: GRanularity-Aware Search Policy for Agentic RAG 

**Authors**: Varun Gandhi, Jaewook Lee, Shantanu Todmal, Franck Dernoncourt, Ryan Rossi, Zichao Wang, Andrew Lan  

**Link**: [PDF](https://arxiv.org/pdf/2607.10463)  

**Abstract**: Agentic retrieval-augmented generation (RAG) extends static RAG by allowing language models to iteratively reason, generate search queries, retrieve evidence, and predict answers. However, it remains challenging for models to decide when to retrieve, whether to use lexical matching or semantic similarity, and how to control context granularity to prevent irrelevant tokens from interfering with agent reasoning. In this paper, we introduce GRASP, a reinforcement learning (RL) framework for training agents to adaptively coordinate complementary retrieval tools during multi-step reasoning. GRASP provides the agent with semantic search, keyword search, and paragraph-reading actions, enabling it to retrieve sentence-level evidence and expand further context only when needed. We train the policy with a reward that jointly accounts for answer accuracy, grounded reading, complementary search, and turn efficiency. Experiments on multi-hop reasoning benchmarks show that GRASP improves both retrieval recall and downstream question answering performance compared with single-step retrieval, prompting-based agentic RAG, and RL-based retrieval baselines. Qualitative and ablation analyses show that the learned policy develops interpretable skimming and scanning behavior: it uses semantic search for broad exploration, paragraph reading for local verification, and keyword search for entity-specific evidence. These results suggest that learning to coordinate retrieval signals and context granularity is critical for agent's correct reasoning. 

---
# ANCHOR: Automated Alignment Auditing for CLI Agents on Real-World Harm 

**Authors**: Kefan Song, Yanjun Qi  

**Link**: [PDF](https://arxiv.org/pdf/2607.10455)  

**Abstract**: Autonomous CLI agents can now execute hundreds of actions across multi-hour sessions: writing code, executing shell commands, browsing the web, and managing cloud infrastructure, all with minimal human oversight. Does greater autonomy invite greater risk? We introduce ANCHOR, an automated auditing framework that stress-tests CLI agents on illegal tasks grounded in public US court cases. ANCHOR deploys an auditor agent fine-tuned on dark personality data using supervised and reinforcement fine tuning. This auditor roleplays persistent malicious users who decompose tasks, reframe requests upon refusal, and adapt strategies across multi-turn interactions. Evaluating frontier CLI agents, we find that while they often refuse illegal tasks when prompted directly, compliance reaches 100\% under persistent malicious interaction. When agents comply, they frequently exceed user requests, autonomously building infrastructure for large-scale harm, including catastrophic risk scenarios such as large-scale financial fraud and bioweapon development. These findings demonstrate that current alignment techniques are insufficient for autonomous agents and underscore the need for safety evaluations against persistent, adaptive malicious users. We release ANCHOR at this https URL 

---
# Co4ICF: Co-evolving Physics-Informed Surrogate and RL-based Pulse Optimizer for Inertial Confinement Fusion 

**Authors**: Jiatong Zhao, Tengyue Zhang, Yuhan Wang, Fuyuan Wu, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2607.10366)  

**Abstract**: Offline-trained surrogates for Inertial Confinement Fusion (ICF) suffer a well-known failure mode that iterative optimizers drive inputs into out-of-distribution (OOD) regions where predictions become unreliable. Here we present Co4ICF, a co-evolving framework that couples a physics-informed surrogate with a PPO-based pulse optimizer. The surrogate is iteratively fine-tuned on policy-induced trajectories, correcting extrapolation errors as the optimizer shifts the input distribution; the optimizer queries this evolving surrogate as a fast environment. In the 1D MULTI environment, Co4ICF achieves 146.1% normalized yield based on current laser design baseline; as a post-hoc cross-fidelity check, the optimized pulse further attains 246.9% normalized yield when directly evaluated in 2D-MULTI without any 2D training or fine-tuning. Budget-matched ablations support that the gains are not explained solely by additional simulation data and are consistent with the co-evolving mechanism playing a key role. We release a large-scale MULTI-IFE simulation dataset to support future benchmarking. 

---
# ABot-AgentOS: A General Robotic Agent OS with Lifelong Multi-modal Memory 

**Authors**: Jiayi Tian, Shiao Liu, Yuting Xu, Jia Lu, Zihao Guan, Honglin Han, Di Yang, Minqi Gu, Yifei Qian, Tianlin Zhang, Yanqing Zhu, Zeqian Ye, Menglin Yang, Fei Wang, Xu Hu, Xiuxian Li, Wei Zhang, Shihui Su, Yiyan Ji, Jingbo Wang, Ziteng Feng, Jiaheng Liu, Zhaoxiang Zhang, Xiaolong Wu, Mingyang Yin, Zedong Chu, Mu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.10350)  

**Abstract**: Recent VLM and VLA systems have improved robotic perception and action prediction, yet long-horizon embodied agents still require a general runtime layer for reasoning, memory, tool use, verification, and cross-embodiment execution. We present ABot-AgentOS, a general robotic Agent Operating System that sits above low-level controllers and provides a deliberative agent layer for scene-conditioned planning, context-isolated skill execution, multi-stage verification, multi-modal memory, and edge-cloud collaboration. To evaluate such systems, we introduce EmbodiedWorldBench, an executable benchmark with 16 indoor, outdoor, and hybrid scenes, four difficulty levels, and over 200 tasks involving navigation, object search, NPC dialogue, dynamic events, and trace-grounded scoring. ABot-AgentOS further introduces Universal Multi-modal Graph Memory, a persistent source-grounded substrate that converts dialogue, visual observations, spatial context, temporal relations, and task traces into typed nodes and edges. A failure-driven self-evolution loop converts diagnosed memory failures into gated runtime evo-assets that are promoted only to later evaluation splits, preventing current-split ground-truth leakage while enabling continual improvement. On an initial EmbodiedWorldBench subset, ABot-AgentOS improves over a single-controller baseline in both task success and goal completion. Across memory benchmarks, ABot-AgentOS Static achieves 87.5 on LoCoMo, 59.9 on OpenEQA EM-EQA, 88.6 on Mem-Gallery, and 76.5 Acc@All on NExT-QA; self-evolution further improves LoCoMo to 88.7, OpenEQA to 60.4, and Mem-Gallery to 89.0. These results suggest that a general Agent OS layer can improve long-horizon embodied execution while providing persistent, auditable memory for continual interaction. 

---
# Comparing Socio-technical Design Principles with Guidelines for Human-centered AI 

**Authors**: Thomas Herrmann  

**Link**: [PDF](https://arxiv.org/pdf/2607.10331)  

**Abstract**: Human-centered AI (HCAI) refers to guidelines or principles that aim on ethi-cally oriented design of systems. We compare HCAI- guidelines with princi-ples of socio-technical systems that emerged in the context of conventional in-formation technology. The comparison leads to a revision of socio-technical heuristics by including aspects of AI-usage. The comparison reveals that con-tinuous evolution is a basic characteristic of socio-technical systems, and that human oversight or interventions and the subsequent appropriation of AI-systems lead to continuous adaptation and re-design of the systems, if autono-my is collaboratively exercised. From a socio-technical point of view, the cru-cial requirement of transparency has not only to be fulfilled with technical fea-tures, but also by contributions of the whole system including human actors. It will be promising for using AI, if not only technical features, but organization-al and social practices are socio-technically designed in a way that compen-sates shortcomings of AI. 

---
# Measure the Sim-to-Real Gap: Designing an Affordable Real-World Benchmark Platform for Reinforcement Learning in AIoT Systems 

**Authors**: Rongping Zhou, Omid Tavallaie, Shuaijun Chen, Albert Y. Zomaya  

**Link**: [PDF](https://arxiv.org/pdf/2607.10309)  

**Abstract**: Reinforcement learning (RL) is commonly employed to enhance the performance of autonomous systems, including the Autonomous Internet of Things (AIoT). However, the trial-and-error nature of RL, when conducted in real-world environments, is costly and hazardous in some scenarios. Consequently, the majority of RL research is conducted in simulation. This reliance introduces challenges related to the Sim-to-Real transferability. Evaluating the Sim-to-Real algorithmic robustness and the Sim-to-Real gap is a critical prerequisite for research aimed at improving RL performance in the real world. Therefore, industries such as robotics have developed concurrent simulation and physical platforms to facilitate this research. However, a universal Sim-to-Real benchmark platform for AIoT does not currently exist. To address these concerns, we developed a real-world AIoT platform for studying RL in AIoT. On this platform, an agent deployed on an edge device plays video games on a separate host computer via a hardware-emulated keyboard, guided by vision input. This platform uses commercially available components costing less than USD 400, together with two computers. Because the system's objective is game score maximization, it inherently mitigates safety risks associated with real-world RL deployments. Experimental results show the simulation-trained agent suffers a 1160% performance degradation relative to the human-level performance after real-world deployment, indicating a significant Sim-to-Real gap. Direct real-world training using the deep Q-network (DQN) algorithm achieves approximately 38% of human-level performance after 10 million training steps, demonstrating the feasibility of RL under real-world conditions. These results suggest that the proposed Sim-to-Real benchmark platform provides a substantial foundation for qualitative and quantitative evaluations of RL in real-world AIoT systems. 

---
# SPARK: Susceptibility-Guided Profiling and Steering of Latent Reasoning States in Large Language Models 

**Authors**: Dongxu Zhang, Yiding Sun, Zihao Guo, Xiangyang Yang, Kai Tang, Lin Chen, Cheng Tan, Jihua Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.10296)  

**Abstract**: Reasoning failures in large language models (LLMs) are usually evaluated from final answers, but a wrong answer does not reveal why the model failed. The same incorrect output may reflect missing capability, an unstable reasoning trajectory, or a failure to activate a reasoning state that is already available in the frozen model. Existing prompting and benchmark-based evaluation methods mostly operate at the output level, while generic activation-steering methods typically apply global directions without diagnosing which examples require intervention. In this paper, we introduce SPARK, which uses hidden-state response to diagnose whether a model internally enters an effective reasoning state and to guide lightweight test-time steering. The key observation is that raw hidden-state susceptibility is strongly confounded by prompt length, especially in programmatic and algorithmic reasoning where harder serialized instances naturally become longer. SPARK therefore uses length-controlled susceptibility to separate input-scale effects from residual reasoning activation, and combines this signal with cross-layer coordination to select reasoning-active anchors and under-activated hard examples. We use FRONTIER-4.5K as a controlled programmatic reasoning suite for latent profiling and difficulty-aware analysis, and evaluate SPARK-Steering on GSM8K and MATH-500 with forward-only benchmark profiling. Our method improves Qwen3 series models consistently; on MATH-500, accuracy rises from 82.0% to 84.6% for Qwen3-4B and from 82.4% to 85.6% for Qwen3-8B. These results suggest that susceptibility can serve not only as a diagnostic signal for reasoning failures, but also as a practical guide for targeted test-time intervention. 

---
# Can Agentic Trading Systems Pay for Their Own Intelligence? 

**Authors**: Qiqi Duan, Changlun Li, Chen Wang, Fan Zhang, Mengxiang Wang, Dayi Miao, Peixian Ma, Jiangpeng Yan, Liyuan Chen, Shuoling Liu, Preslav Nakov, Yuyu Luo, Nan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2607.10286)  

**Abstract**: Large language model (LLM) agents are increasingly used in trading systems, where model reasoning, tool use, and continual decisions incur costs that are expected to produce trading value. Existing evaluations typically report performance metrics, but rarely examine agentic viability: whether dynamic LLM-mediated decisions convert their induced costs into measurable incremental profit. To apply this criterion, we introduce TradeLens, a trace-grounded diagnostic toolkit for evaluating agentic trading systems from their trading records, runtime traces, and deployment configurations. It reconstructs trading trajectories, attributes profit and cost to interpretable evidence, and diagnoses whether and why an agent pays for its own intelligence. We conduct extensive analysis across backbone models, capital scales, trading frequencies, and system architectures, together with deployment discussion. Our results show that viability hinges on intelligence-to-profit conversion: models exhibit different failure patterns, such as poor asset selection in DeepSeek-V3.2 and negative timing in GLM-4.7, while capital scale, trading frequency, and architecture matter only by amplifying or degrading decision-attributed timing value. These findings reframe the evaluation of LLM-based trading agents from capability-centric performance ranking to trace-grounded diagnosis of intelligence-to-profit conversion. Our code is available at this https URL. 

---
# Information-seeking failures of large language models in agentic clinical reasoning 

**Authors**: Krischan Braitsch, Laura K. Schmalbrock, Theresa Weltermann, Andrew F. Berdel, Isabella Miller, Kai Tran, Michael Heider, Sabrina Kraus, Florian Bassermann, Jacqueline Lammert, Sebastian Ziegelmayer, Marcus Makowski, Lisa C. Adams, Keno K. Bressem  

**Link**: [PDF](https://arxiv.org/pdf/2607.10275)  

**Abstract**: Large language models achieve high scores on medical knowledge assessments, yet clinical reasoning requires actively deciding what to investigate under uncertainty. We developed an agentic evaluation framework in hematologic oncology in which models must proactively request clinical data across three sequential rounds before committing to a diagnosis and treatment plan. Across 32 frontier models, the best achieved only 68% overall accuracy. Information utilization, the fraction of available data actually requested, was the strongest predictor of diagnostic accuracy (R = 0.69, P < 0.001), yet utilization collapsed from 57% to 26% in the final round, leaving molecular and cytogenetic data critical for treatment selection unexamined. Reasoning traces scored high on a clinical reasoning rubric (91% above threshold) but decorrelated from accuracy, revealing a gap between locally coherent rationales and globally correct conclusions. Error analysis identified search satisficing, anchoring and premature closure as the dominant failure modes, the same cognitive biases that characterize novice clinicians under dual-process models of diagnostic reasoning. These findings demonstrate that the primary limitation of current models in clinical oncology is not insufficient medical knowledge but a systematic failure of information-seeking under uncertainty. 

---
# Behavioural Signatures of Risk-Sensitive Decision-Making in Large Language Models 

**Authors**: Xuankun Rong, Wenke Huang, Bo Du, Dacheng Tao, Mang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2607.10251)  

**Abstract**: As large language models (LLMs) are increasingly used in decision support, it is important to understand whether their choices under uncertainty exhibit stable and interpretable behavioural regularities. Human decision-making combines relatively persistent risk preferences with context-dependent adjustment, yet it remains unclear whether analogous behavioural structure can be observed in LLM-based decision systems. Here we examine this question using a controlled multi-model framework based on no-limit Texas Hold'em, where behaviour is quantified by Participation, measuring voluntary engagement in uncertain opportunities, and Proactiveness, measuring pre-flop risk escalation. Across homogeneous self-play and heterogeneous mixed-model interactions, frontier LLMs exhibit stable, model-specific risk profiles, forming a spectrum from conservative to aggressive decision styles. These profiles remain largely robust under changing opponent composition, while the most conservative and most aggressive models diverge further in mixed settings. Under global risk pressure and personal resource constraint, models adapt in structured but heterogeneous ways, ranging from broad behavioural contraction to selective de-escalation and near-invariant behaviour. These findings suggest that LLMs differ not only in baseline risk disposition, but also in the risk signals they respond to and the flexibility with which they adjust, providing a behavioural basis for auditing risk-sensitive decision-making in interactive settings. Our code is publicly available at: this https URL. 

---
# When Are Sparse Feature Interventions Actually Localized? Matched Evaluation for SAE-Based Safety Control 

**Authors**: Daming Luo  

**Link**: [PDF](https://arxiv.org/pdf/2607.10226)  

**Abstract**: We evaluate when sparse autoencoder (SAE) features act as localized control handles for safety-relevant behavior. This question is difficult because apparent success can arise from weak interventions, mismatched baselines, model robustness, or degenerate outputs that automated safety judges mark as unsafe without representing meaningful harmful compliance. We introduce a matched coherence-gated evaluation protocol for runtime safety interventions: methods are compared at matched target-effect points, and the primary target metric counts harmful compliance only when an output is both judge-unsafe and coherent. Applying this protocol to three prompt splits on Gemma-2-9B-it with a Gemma Scope layer-20 residual SAE, we find that SAE feature ablation has a narrow useful regime. SAE top800 reaches a low-to-mid target effect with lower total perturbation and competitive utility, but SAE top1600 loses utility relative to a matched dense refusal-direction baseline, and SAE top3200 primarily induces coherence collapse. Human audit confirms that coherence gating removes unsafe-only artifacts, and feature diagnostics show that the useful regime is driven by a stable head of refusal-aligned features whose activation separation decays rapidly with rank. These results argue that SAE-based safety interventions should be evaluated as regime-dependent control mechanisms rather than assumed to be uniformly localized. 

---
# KGCQual: An Interpretable Framework for Evaluating the Knowledge Graph Construction Quality from Text 

**Authors**: Nipun Misra, Vikranth Udandarao, Aanchal Gupta, Yogender Kumar, Manuj Mukherjee, Raghava Mutharaju  

**Link**: [PDF](https://arxiv.org/pdf/2607.10212)  

**Abstract**: Knowledge Graphs (KGs) are increasingly constructed through automated extraction pipelines; however, such systems often introduce spurious or incomplete triples, which degrade downstream performance. Existing evaluation practices rely heavily on task-specific metrics or small-scale manual verification, offering limited insight into the structural and semantic fidelity of extracted graphs. We propose a novel, interpretable metric for intrinsic KG quality assessment that measures how closely an automatically extracted graph approximates an "ideal" graph capturing the key noun phrases, predicate relations, and basic linguistic phenomena such as negation expressed in the source text. Our framework integrates two complementary components: (1) an entity-level assessment that evaluates completeness, resolution quality, and connectivity, and (2) a relation-level assessment that judges predicate preservation and multiplicity using lexical similarity, dependency-parse alignment, and light-weight negation handling to ensure semantic faithfulness. We evaluate our metric across multiple state-of-the-art triple extraction systems and datasets, including WebNLG, TinyButMighty, and BenchIE, demonstrating that it reliably identifies omissions, redundancy, and structural deviations that existing metrics overlook. Our work offers a scalable, model-agnostic, and interpretable framework for comparing automated KG construction methods and provides a foundation for standardised evaluation. We further validate the metric through an ablation study isolating noun and verb components, and a downstream evaluation showing that KGCQual scores correlate significantly with link prediction performance on the same extracted KGs. The code repository is available at this https URL. 

---
# GRATE: Temporal Extensions for Inductive KG Foundation Models via Gated Rotary Attention 

**Authors**: Jiaxin Pan, Osama Mohammed, Daniel Hernández, Steffen Staab  

**Link**: [PDF](https://arxiv.org/pdf/2607.10197)  

**Abstract**: Knowledge graph foundation models such as Ultra and Trix achieve strong inductive transfer by learning relation-graph representations that generalise to unseen entities and relations. Extending this transferability to temporal knowledge graphs (TKGs) remains challenging: existing temporal models tie their parameters to dataset-specific entities, relations, or timestamps and are not designed to transfer to TKGs with disjoint vocabularies. We propose GRATE (Gated Rotary Attention for Temporal Encoding), an entity-side message function that adds no learnable parameters and encodes time through relative time differences by rotating each edge message according to its time gap to the query and applying a query-conditioned gate to select temporally relevant signals. GRATE integrates into NBFNet-style KG foundation models while preserving structural transferability. Existing TKG benchmarks evaluate within shared train/test vocabularies and cannot directly test cross-dataset temporal transfer; we therefore construct GDELTIndT and WIKIIndT, inductive transfer benchmark suites with disjoint entities, relations, and timestamps spanning both interpolation and extrapolation. Across these benchmarks and held-out forecasting datasets, a single jointly pretrained GRATE checkpoint improves over the static base model in most settings. 

---
# UNIT: Unleash Large Language Models Potential for Graph Continual Learning 

**Authors**: Tairan Huang, Yili Wang, Beibei Hu, Yiting Shi, Qiutong Li, Changlong He, Jianliang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2607.10159)  

**Abstract**: In real-world multimodal web scenarios, graph-structured data often arrives in a streaming manner, making graph continual learning a crucial paradigm for continuously modeling such evolving structures. However, existing graph continual learning methods still face two fundamental challenges. 1) semantic-structural separation, where the graph-based methods excel at modeling topological relationships but neglect deep semantics. 2) imbalanced knowledge transfer, where existing models fail to effectively leverage general knowledge gained from early tasks to benefit subsequent new tasks. To address above issues, we propose a novel framework, \textbf{UN}leash Large Language Models PotentIal for Graph ConTinual Learning (UNIT). By fine-tuning large language model only on the first task, we bridge the distributional gap between the pre-trained LLM corpus and the target task dataset to enhance the adaptability of LLMs for graph-structured tasks. Meanwhile, we propose an uncertain-aware anchor generation mechanism to effectively preserve representative knowledge across tasks, avoiding the neglect of universal knowledge learned from previous tasks. Additionally, we introduce structural confluence modeling to explicitly integrates graph topology information into semantic information, enhancing the collaborative capabilities between semantic understanding and structural modeling. Extensive experiments demonstrate that our proposed method achieves state-of-the-art performance in the graph continual learning task. 

---
# IdeaTrail: Full-Process Agent Trajectories for Scientific Ideation 

**Authors**: Hengquan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2607.10144)  

**Abstract**: Scientific research is a complex, multi-stage workflow rather than a single act of text generation. The ideation process typically emerges through literature search, paper reading, tool use, claim checking, cross-paper synthesis, brainstorming, rejection of weak directions, and iterative writing. Existing resources capture individual components of this process, but datasets that jointly record tool use, evidence acquisition, intermediate artifact evolution, and idea- or proposal-level endpoints remain limited. This report introduces \method, a multi-turn process-trajectory dataset for scientific ideation and proposal generation. Each instance records a research process from evidence gathering to either idea selection or proposal construction. Rather than freely fabricating trajectories, \method starts from human-selected high-quality research papers and proposal artifacts and uses a Generator--Advisor synthesis loop. The Generator produces the visible trajectory through actions, observations, and artifact edits, while the Advisor has access to the full generation context and checks grounding, causal order, naturalness, and leakage from hidden targets. This reverse-to-forward procedure produces multi-turn research data that remains aligned with real scientific artifacts while approximating the uncertainty, evidence use, and staged convergence of research practice. \method provides both a dataset and a general recipe for synthesizing process-supervision data for scientific research agents. 

---
# Dynamic Agent Skills: A Lifecycle Survey and Taxonomy of Evolving Skill Libraries 

**Authors**: Yubo Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.10113)  

**Abstract**: Large language model agents increasingly store reusable procedures outside the model. These reusable procedures are often called \emph{skills}: they may be code functions, natural-language instructions, this http URL packages, workflow graphs, or learned adapters that a future agent can retrieve and invoke. This taxonomy-driven survey asks how such skill libraries change over time. Across a $124$-paper $2023$--$2026$ audit set, we synthesize dynamic skill systems as \emph{lifecycle-managed, verified, evolving artifact stores}: agents collect evidence from interaction, propose skill updates, verify and admit candidates, organize them for retrieval and composition, repair or prune stale entries, and govern sharing through provenance and rollback. We organize the literature around three survey tools. First, a $\text{six}$-sense taxonomy distinguishes the structurally different artifacts called ``skills'' in current papers. Second, an $\text{eight}$-stage lifecycle architecture identifies the recurring design decisions behind evidence acquisition, proposal, verification/admission, storage, retrieval/composition, maintenance, distillation/portability, and governance. Third, a lightweight skill-record schema and $\text{ten}$-operator vocabulary provide common terms for comparing library updates without elevating them into a separate method contribution. Using this structure, we synthesize evidence-graded patterns with explicit caveats: admission and repair are repeatedly important, verifier quality materially affects skill-aware RL, flat retrieval can degrade as libraries grow, and current benchmarks still under-report library trajectories, usage--utility gaps, and safety surfaces. We close with concrete reporting standards and open problems for evaluating dynamic skills as changing libraries rather than static prompt or tool collections. 

---
# Looped State-Space Language Models with Adaptive Exit-State Selection 

**Authors**: Zhenxuan Yu, Takeshi Kojima, Yutaka Matsuo, Yusuke Iwasawa  

**Link**: [PDF](https://arxiv.org/pdf/2607.10110)  

**Abstract**: Recent work on looped language models suggests that many reasoning problems benefit from greater computational depth rather than from additional independent parameters. Existing studies, however, focus almost exclusively on Transformer backbones, leaving open whether this principle also applies to state-space language models. We investigate Looped Mamba and Looped Hybrid Mamba-Transformer architectures, which repeatedly apply a shared Mamba (or hybrid) block to introduce explicit finite-depth recurrent computation. On two controlled reasoning tasks-Mano (modular-arithmetic manipulation) and p-hop induction-Looped Mamba consistently outperforms parameter-matched non-looped baselines and, in several settings, matches or exceeds non-looped models of equal effective depth. We then extend the study to language model pre-training under matched iso-parameter and iso-FLOPs protocols, which jointly disentangle the effects of parameter sharing and effective depth: looped models remain competitive on downstream benchmarks with substantially fewer distinct parameters, although deeper non-looped models retain an advantage in validation perplexity under strict iso-FLOPs comparisons. Finally, we adapt Ouro's two-stage exit gate to Looped Mamba for threshold-controlled selection among recurrent-step outputs. Since all recurrent steps are still executed, the selected exit step represents prediction depth rather than reduced wall-clock computation. At the scales studied, adaptive exit-state selection improves downstream performance at intermediate depths, while actual inference-time savings require additional state-handling mechanisms. 

---
# MAG: A Web-Agent Benchmark and Harness for Multimodal Action and Guide Generation 

**Authors**: Chengguang Gan, Hanjun Wei, Yunhao Liang, Zhixi Cai, Qinghao Zhang, Shiwen Ni  

**Link**: [PDF](https://arxiv.org/pdf/2607.10079)  

**Abstract**: Digital Adoption Platforms (DAPs) are embedded overlays widely used on web systems to guide users through operations inside a page, helping them get started with unfamiliar interfaces quickly. Completing a real task, however, rarely means clicking a few buttons on a single page: it takes a sequence of actions that unfolds across changing page states. Prior studies have also treated automated web agent actions and guide text generation as two separate problems, and most of them feed models textual page representations such as the DOM or accessibility trees rather than the rendered screens that humans actually operate on. In this work we introduce MAG, the first benchmark that unifies task execution and guide writing into a single Multimodal Action and Guide task, with two grounding schemes over screenshots: Set-of-Mark element selection and raw pixel coordinates. We further build a complete harness for this compound task, covering annotation with LLM assistance and human verification, training, evaluation in live environments, and joint metrics for actions and guides. With this harness we evaluate frontier API models and open multimodal models, and report detailed analyses. Finally, we design a GRPO training method augmented with expert trajectories, which nearly doubles the success rate of a supervised 9B agent (from 6.9% to 13.2%) and improves guide quality at the same time. Even the strongest model completes fewer than 40% of the tasks, leaving ample room for future research. 

---
# From ambiguous utterances to governed reuse classes: canonicalization, quotient invariance, and conditional decidability 

**Authors**: Cosimo Spera, Ray Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2607.10069)  

**Abstract**: Semantic caching defines answer reuse on embedding similarity: two utterances share a stored answer when a similarity score clears a threshold, with no notion of authorization, versioning, or of what makes two demands the same. This note changes the object on which reuse is defined: in a governed domain, reuse should operate on a mathematically characterized quotient of resolved conversational demands, not on a similarity heuristic. Three independently defined relations on resolved utterances -- reading identity, resolution identity, and reuse identity -- form a refinement chain, strict under realized nondegeneracy conditions checkable on deployment logs; the pipeline's outputs are invariant along the chain, and reuse identity is exactly the kernel of the resolution map into the governed answer partition, so the reuse quotient is the utterance-side object that partition induces, not a relabeling of it. Reuse identity licenses the governed query key and its certified answer space; reuse of a particular answer requires resolution identity or an applicability certificate. The supporting layer is stated at exactly the strength proved: exact-denotation normal forms; join aggregation as a design operator, with closure-stable cells characterizing no-escape; total computability of the full pipeline relative to an untrusted proposal layer; policy admissibility for arbitrary proposers -- and provably not factual grounding or intent fidelity; and elicitation terminating after finitely many informative replies, sound under target consistency. 

---
# AgentAbstain: Do LLM Agents Know When Not to Act? 

**Authors**: Xun Liu, Yi Evie Zhang, Vira Kasprova, Parisa Rabbani, Pardis Sadat Zahraei, Tianyu Zhang, Ali Ebrahimpour-Boroojeny, Varun Chandrasekaran  

**Link**: [PDF](https://arxiv.org/pdf/2607.10059)  

**Abstract**: Agent systems based on large language models (LLMs) are increasingly deployed for autonomous tasks, yet existing evaluations mostly focus on task success rather than whether agents know when to abstain. This gap poses real risks: under ambiguity, conflicting constraints, or tool failures, agents may execute unintended and irreversible actions. To close this gap, we present the first systematic evaluation framework for agentic abstention: the calibrated ability of tool-using LLM agents to recognize when not to act. At its core, AgentAbstain is a paired-task benchmark built on an agent-native taxonomy of 8 abstention scenarios across pre-execution reasoning and runtime discovery. It contains 263 paired tasks across 42 executable sandbox environments, where each pair consists of a should-act task and a should-abstain variant produced through a controlled perturbation to the instruction, tool, or environment state. To scale this paired design and resist data contamination, we propose AbstainGen, a fully automated pipeline that synthesizes sandbox environments and generates paired tasks end-to-end, validated by deterministic replay and semantic LLM judges; fresh task instances can be regenerated on demand, and three independent annotators rate 94-98% of sampled tasks as well-designed. Across 17 frontier LLMs in 4 agent harnesses, the best agent (Gemini 3.1 Pro) achieves only 59.5% paired accuracy (correct on both the act and abstain sides of each paired task). More importantly, abstention capability is largely independent of general task-solving capability, indicating that scaling task-solving alone will not close this gap. We further identify failure modes such as post-hoc abstention, in which agents execute irreversible actions before recognizing abstention triggers. Our code and dataset are open-sourced at this http URL. 

---
# A Symbolic Neural CPU for Quantization-Simulated Writeback and Interpretable Program Execution 

**Authors**: Jose Luis Lima de Jesus Silva  

**Link**: [PDF](https://arxiv.org/pdf/2607.10021)  

**Abstract**: Neural networks can learn algorithmic input-output mappings, but trusting a learned executor requires more than a correct final answer because the state transitions that produce it are usually hidden. To make those transitions visible, we introduce a trace-supervised symbolic neural CPU, a factorized learned execution architecture that combines recurrent control, an explicit operation router over a fixed differentiable arithmetic-logic unit bank, destination-masked register writeback, complete trajectory supervision and matched fixed-point replay. The model exposes the selected operation, source and destination registers, register trajectory, memory signals and writeback semantics at every step. On the principal 16-wide benchmark, the non-quantized executor reproduces reference execution exactly, while the eight-bit quantization-simulated executor preserves the symbolic operation path through programs of 1,000 instructions. When the same execution is evaluated against a matched fixed-point replay, the residual numerical drift disappears, showing that it comes from a mismatch between continuous and low-precision reference semantics rather than from execution failure. We compare recurrent, Transformer, temporal-convolution, temporal graph-inspired and state-space controllers, and the ablations show that operation-gate supervision is necessary for an inspectable execution path. Hidden-opcode memory-pressure tasks expose the remaining limits in delayed state use and temporal binding. We also extend the interface with ValueMemory, hybrid adaptive leaky integrate-and-fire controllers, candidate-constrained symbolic control trained through behaviour cloning and actor-critic reinforcement learning, and an RV32I base-integer semantic bridge. Together, these results establish a trace-verifiable framework for interpretable, low-precision and controllable neural execution. 

---
# Who&When Pro: Can LLMs Really Attribute Failures in AI Agents? 

**Authors**: Jiale Liu, Huajun Xi, Shaokun Zhang, Yifan Zeng, Tianwei Yue, Chi Wang, Jian Kang, Qingyun Wu, Huazheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.09996)  

**Abstract**: Automated failure attribution uses LLMs to identify where and why agentic systems fail. As agents become more capable, their failures become subtler, making automated attribution increasingly important. We introduce Who&When Pro, a large-scale benchmark for automated failure attribution in agentic systems. Using a strictly controlled pipeline that injects a failure only after exactly replaying a successful prefix, we construct 12,326 failed trajectories with golden labels across 3 modalities and 26 benchmarks covering various scenarios. Beyond benchmarking, we conduct extensive experiments and analyses, revealing systematic patterns in how models attribute failures across modalities, protocols, and model families, and providing empirical guidance for future automated failure attribution systems. 

---
# TopoExplore: Topological Discrimination for Archive-Based Exploration 

**Authors**: Jason Carlson  

**Link**: [PDF](https://arxiv.org/pdf/2607.09971)  

**Abstract**: Archive-based exploration methods such as Go-Explore select which visited state to return to using visitation rarity, and frontier methods return to the boundary of the unknown; neither asks whether the unexplored region behind a boundary is enterable at all. Exploration is not just about finding reward - it is about collecting a structurally complete experience for downstream learning and planning. We introduce TopoExplore, which augments Go-Explore cell selection with a periodic topological pass: enclosed unexplored regions (voids) of the visited-set occupancy grid are detected by flood fill (the H1 classes of its cubical complex), and a decaying selection bonus is placed only on their strict entrances (gap or door cells), so sealed regions are never targeted and entered regions retire. On a controlled 18-environment MiniGrid suite (15 seeds, frozen hyperparameters) TopoExplore attains a 1.52x geometric-mean speedup in median steps-to-first-entry over its exact Go-Explore ablation, versus 1.37x for a frontier baseline; frontier exploration degrades when sealed decoy structure appears (0.83-1.48x on decoy environments vs. 1.65-2.11x for TopoExplore), while TopoExplore holds its largest win on hard multi-interaction doors (10.9x). We report an honest negative on Montezuma's Revenge - without wall knowledge, unreachable occupancy artifacts capture the bonus and performance degrades as it grows, isolating the wall-aware entrance test as the load-bearing component - and a preliminary positive on HM3D scanned buildings, where the speedup over Go-Explore tracks scene difficulty (r=0.69) even as frontier selection dominates blanket coverage. The evidence supports a deliberately scoped claim: topology-aware selection pays off where enclosed structure must be discriminated, and remains competitive at open coverage, where frontier methods are strongest, despite not being tuned for that regime. 

---
# Exploring Agentic Workflows for Generating High Quality Math Visual Aids 

**Authors**: Rizwaan Malik, Ashna Khetan, Isabel Sieh, Samin Khan  

**Link**: [PDF](https://arxiv.org/pdf/2607.09839)  

**Abstract**: Mathematical diagrams play a crucial role in K 12 education, both as problem components and as scaffolding for student comprehension. However, current AI tools, including Large Language Models (LLMs), struggle to reliably generate accurate and pedagogically sound visual diagrams, even when provided with detailed descriptions. A significant gap therefore remains in the reliable generation of diagrams for middle school mathematics. To address this, we introduce an agentic workflow that enables LLM agents to evaluate the quality of generated visuals and use this feedback to iteratively improve their outputs. This self improvement loop aims to enhance the accuracy and educational appropriateness of AI generated diagrams. Our research investigates two questions. First, can LLMs accurately generate quality assurance questions for a visual aid given specific criteria for visual quality? Second, given valid quality assurance questions, can Vision Language Models effectively evaluate generated K 12 visual aids and use the resulting feedback to improve them iteratively? We conduct an exploratory evaluation of our agentic workflow and identify key areas for improvement, including stronger spatial reasoning and more comprehensive coverage of diagram features in the generated quality assurance questions. Our results provide preliminary evidence that this approach can improve the reliability and educational value of AI generated mathematical diagrams. 

---
# Agentic Context Learning with Self-Discovered Specification 

**Authors**: Jike Zhong, Ming Li, Yuxiang Lai, Ziyan Yang, Jingyu Xie, Jihyung Kil, Zheda Mai, Shao-Yuan Lo, Ren Xiang, Konstantinos Psounis, Yuanyuan Lei  

**Link**: [PDF](https://arxiv.org/pdf/2607.09794)  

**Abstract**: Context learning is an emerging inference-time task where LLMs must learn and apply novel, task-specific knowledge from intricate contexts absent from pre-training; even frontier models score under 24% task success. In this work, we conduct a comprehensive empirical study to understand why this setting remains difficult. A natural hypothesis is that failures stem from content access; yet across twelve retrieval, reflection, and verification baselines on CL-Bench, an extensive context learning benchmark, we find limited gains over direct full-context prompting. Further failure analysis reveals a key finding: unlike typical long-context tasks such as long document understanding, context learning requires not only recovering local content but also acquiring local specifications that are often unspecified in the query but distributed across the context: domain-specific formats, local rules, and completeness conditions. Across all 31,592 rubric items, we find that 55.4% clearly evaluate specification acquisition, while only 22.6% evaluate content acquisition. Moreover, despite 76.7% of specifications being unspecified in the user query, 95.5% are traceable to the context, indicating these are learnable obligations rather than hidden requirements. To validate this diagnosis, we design a deliberately simple intervention PSCI (private specification-contract induction) which extracts local specifications and enforces them through adversarial checking and repair; PSCI achieves state-of-the-art 28.14% with GPT-5.1 (+5.59 pp absolute and +24.8% relative) on CL-Bench, replicated on Qwen3.5-27B (+5.28 pp) and Gemini 3 Pro (+6.17 pp). Seventeen ablations further isolate the role of task-specific specifications. Overall, our results suggest context learning hinges on not only content acquisition but also specification acquisition. 

---
# Semantic Drift and the Stability of Operator Control in Reasoning-Class Decision Support Systems 

**Authors**: M. L. Kaluzhsky, V. A. Efirov  

**Link**: [PDF](https://arxiv.org/pdf/2607.09790)  

**Abstract**: The article investigates the fundamental problem of ensuring the stability of operator control and preserving goal-targeting in hybrid human-machine decision support systems (DSS) of a new generation. Based on a two-month continuous longitudinal experiment on the joint design of a monograph-format textual array, the latent phenomenon of semantic context drift in large language models of deep logical reasoning (Reasoning LLMs) is verified and described. A mathematical model of interaction in the human-machine interface is proposed, and an original metric is introduced - the operator control stability coefficient, which takes into account the non-linear contextual pressure of hidden reasoning chains. Within the paradigm of the cognitome theory, a critical point of control functions inversion is captured. Engineering recommendations are formulated for implementing dynamic relational arbitration loops based on a modified hierarchical similarity model. 

---
# PHITSBench: an execution-scored benchmark for AI-assisted PHITS radiation-transport input generation using natural language 

**Authors**: Xianglin Ji, Svetlana V. Boriskina  

**Link**: [PDF](https://arxiv.org/pdf/2607.09789)  

**Abstract**: We introduce PHITSBench, an execution-scored benchmark for the Monte Carlo Particle and Heavy Ion Transport code System (PHITS). PHITSBench comprises 282 transport-scorable tasks spanning three common workflow categories: parameter editing (Edit), syntax repair (Repair ), and complete simulation generation from natural-language descriptions (Reproduce). Each task is evaluated using a Composite Metric Score that combines execution success with agreement between generated and reference transport observables. Using PHITSBench, we evaluate five GPT-5.4-based configurations ranging from zero-shot prompting to knowledge-augmented and agentic workflows. Without domain-specific knowledge, the model performs well on editing and repair tasks (95% and 70% success, respectively) but fails to generate correct simulations from scratch (0% success on the Reproduce track). A structured, machine-readable PHITS knowledge catalog, supplied alongside the user manual, raises single-shot Reproduce-task success to 57%. Agentic execution provides a further improvement to 66-73%, but at increased computational cost. Failure analysis shows that the remaining errors are dominated by incorrect selection and configuration of physical observables rather than syntax generation. These results suggest that future progress in AI-assisted radiation-transport modeling will depend as much on machine-readable knowledge bases, curated domain-training datasets, and execution-grounded evaluation environments as on advances in foundation models themselves. 

---
# Length Penalties Make Chain-of-Thought Less Monitorable 

**Authors**: Bryce Little  

**Link**: [PDF](https://arxiv.org/pdf/2607.09786)  

**Abstract**: Length-penalized reinforcement learning can shorten chain-of-thought reasoning while hiding an influence that drives the model's answer. In our experiments, training with length penalties does not stop misleading hints from steering models, even though the models' chains of thought mention the hint much less often. A token-accuracy evaluation would count these runs as successful because they use fewer reasoning tokens with little accuracy loss; it would miss whether the remaining trace still shows what drove the answer. We train Qwen3-4B and Qwen3-14B variants with different target chain lengths, then evaluate them with biasing-hint interventions on held-out MMLU-Pro-R and four transfer benchmarks. Compression sharply cuts reasoning tokens, preserves most multiple-choice accuracy, and leaves hint influence near baseline. At the strongest target, lower-bound faithfulness falls to 63.1% of baseline for Qwen3-14B and 69.4% for Qwen3-4B; the raw rate at which a monitor catches hint use falls from 69% to 49% and from 60% to 48%. To separate length from content, we randomly delete sentences from uncompressed baseline chains until the remaining text matches the compressed length. Even after this length matching, compressed chains disclose the hint 7-35 percentage points less often than baseline chains that we shorten at random, for both Qwen3 sizes and all five evaluation distributions. Compression therefore does more than shorten reasoning, preferentially removing the cues a monitor needs to see what influenced the answer. Together, these results reveal a compression-monitorability frontier in which cheaper reasoning can preserve answers while making the influences behind them harder to detect. 

---
# From Patterns to Maze Structures: SMT-Based Path Synthesis and 2D/3D Construction 

**Authors**: Shengyi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.09781)  

**Abstract**: We present a pipeline for constructing maze structures from input patterns such as text or shapes. The central path-synthesis problem is encoded in Satisfiability Modulo Theories as global constraints on adjacency, continuity, and pattern-constrained coverage, allowing each fixed-bound instance to be solved in one call. The resulting path is either a planar, self-avoiding route or a layered traversal with prescribed over--under crossings, and it serves as a scaffold for constructing planar mazes and three-dimensional realizations of woven mazes. This report extends the published Bridges 2026 conference paper with more representative SMT-LIB examples and a fuller account of how synthesized paths become concrete maze constructions in planar and three-dimensional form. 

---
# EvoCUA-1.5: Online Reinforcement Learning for Multi-turn Computer-Use Agents 

**Authors**: Mianqiu Huang, Taofeng Xue, Chong Peng, Jinrui Ding, Sicheng Fan, Jiale Hong, Yufei Gao, Xiaocheng Zhang, Linsen Guo, Xin Yang, Dengchang Zhao, Yuchen Xie, Peng Pei, Xunliang Xie, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2607.09773)  

**Abstract**: Computer-use agents must solve long-horizon tasks through repeated interaction with partially observable, multimodal desktop environments. Although imitation learning and offline trajectory refinement provide strong priors, static traces cannot cover the causal feedback loop of real computer use: each action changes the screen state, future action space, and recovery options. EvoCUA-1.5 extends self-evolving computer-use agents from offline experience learning to online reinforcement learning, where policies interact with executable sandbox environments and improve from verifiable task outcomes. Online RL in this setting requires more than directly reusing single-turn language-RL recipes. Multi-turn interaction introduces context-managed observations, sparse terminal rewards, variable-length trajectories, and slow environment feedback. EvoCUA-1.5 addresses these challenges with Step-Level Policy Optimization (STEPO), which preserves trajectory-level advantage balance after decomposition into step-level samples; policy-aware filtering and pass-rate calibration over verifiable synthesized tasks; Dynamic Tri-Adaptive Curriculum (DTAC), which combines learnable tasks, difficult positive replay, and controlled infeasible-task exposure; and a fully asynchronous RL infrastructure with staleness control and mini-group batching. Experiments show that these components improve training stability and downstream performance. EvoCUA-1.5 achieves 63.2\% success on OSWorld-Verified, outperforming comparable 32B/35B-scale open-weight baselines and even approaching models with significantly larger parameter counts. Overall, EvoCUA-1.5 provides a practical framework for scaling online RL in multi-turn computer-use agents. 

---
# Verification of Adaptive Agentic Controllers through Finite Rule Revision 

**Authors**: Roberto Garrone  

**Link**: [PDF](https://arxiv.org/pdf/2607.09770)  

**Abstract**: Industrial agentic AI systems increasingly exhibit a gap between prototype capability and production deployment. In particular, adaptive agents may generate plausible outputs while remaining difficult to verify under non-determinism, confidentiality constraints, limited context, and weak observability. This paper formulates a bounded verification protocol for adaptive agentic controllers represented by finite symbolic rules, explicit diagnostic predicates, explanation logs, and held-out re-evaluation. The central research question is: when an adaptive agentic controller is represented through finite rules, explicit diagnostic predicates, explanation logs, and held-out re-evaluation, which classes of controller failure can be detected, locally repaired, or rejected without relying on unrestricted human-in-the-loop judgment? The proposed framework treats the controller as a finite revisable object. Diagnostic failures are mapped to predefined rule-level edits, including rule addition, rule deletion, and priority revision. Repaired controllers are then evaluated on held-out simulation seeds or cloned initial states. Experiments in a stylized financially constrained inventory-control benchmark show three outcomes: resource-induced failures that remain non-repairable by one rule edit, partial repairs that are rejected because they violate thresholds or guardrails, and a local one-step repair of an order-volatility failure induced by removing a smoothing rule. The contribution is methodological and provides a simulation-compatible procedure for testing whether specific controller-level failures can be made observable, explainable, locally revisable, and empirically re-tested under controlled conditions. 

---
# Norm Enforcement for AI Agents: Robustly Shaping Behavior in Multi-Agent Systems 

**Authors**: Yaowen Ye, Jacob Steinhardt  

**Link**: [PDF](https://arxiv.org/pdf/2607.09766)  

**Abstract**: AI agents are increasingly deployed in shared environments where they pursue diverse goals and compete for rewards. This multi-agent competition can lead to behaviors that serve individual gains at collective cost -- for instance, marketing agents may post misleading content as a result of competing for engagement on social media. Human societies address such problems through norms that constrain acceptable behavior, supported by enforcement mechanisms that detect and penalize violations. Motivated by this, we study norm enforcement mechanisms for language model agents. We find that simple enforcement mechanisms are exploited by misaligned agents for competitive advantage, even when they are not explicitly trained or prompted to do so. We thus turn our attention to designing more robust mechanisms, and identify two key ingredients: estimating each agent's reliability over time, and updating this estimate with escalating penalties for repeated misbehavior. Across three simulated environments and a variety of agent populations, mechanisms built on these principles resist exploitation, while still penalizing norm violations at comparable or lower cost than baselines. Our results position norm enforcement mechanisms as scalable levers for shaping agents' behavior, but only when designed to anticipate becoming part of the system they govern. Our code and data are available at this https URL. 

---
# How Much Does Correctness Cost? Budgeted Placement of Strong Correctors in a Weak Multi-Agent Swarm 

**Authors**: Igor Itkin  

**Link**: [PDF](https://arxiv.org/pdf/2607.09765)  

**Abstract**: A cheap swarm of unreliable agents can be steered to a correct consensus by a few strong, expensive "oracle" correctors. We ask how much one must spend, and where to place the oracles. We model the swarm as a consensus on a graph in which each oracle pins one node toward the truth at a cost-coupled, concave strength, and measure quality by the coherence H(R)=tr M(R)^{-1}. Our first result is that H stays submodular (each added oracle helps less than the last) even when the oracles differ in strength, so a cost-benefit greedy comes within 1-1/e of the best placement at any budget. Inverting the budget gives the budget-correctness frontier B*(eps), the least spend that guarantees an eps-correct consensus: closed-form on the complete graph, and a minimal oracle count k* when oracles cost the same. Whether a budget then buys a few strong oracles or many medium onese curvature of the cost-quality law: diminishing returns favour spreadsharply increasion. Measured onthe Qwen3 ladder (0.6-32B), the law is concave for math verificatio convex foremergent code tracing, so the verdict is genuinely this http URL://github.com/YehudaItkin/budgeted-oracle-placemen 

---
# BatteryLake: Agentic, Physics-Grounded Curation of Heterogeneous Battery Aging Data and Benchmarking 

**Authors**: Tianwen Zhu, Hao Wang, Yonggang Wen  

**Link**: [PDF](https://arxiv.org/pdf/2607.09762)  

**Abstract**: Public battery aging datasets are a critical asset for advanced health management, but their practical use is often limited by inconsistent formats, unclear schemas, and metadata scattered across repositories and publications. Current curation remains largely manual and hard to reproduce, while general-purpose data integration tools miss the domain-specific semantics of electrochemical time-series data. We present BatteryLake, a governed data lakehouse that turns raw public battery data into benchmark-ready assets through an agentic, physics-grounded curation framework, with three contributions. First, LLM agents extract metadata and synthesize dataset-specific converters, grounding every output in verbatim evidence and abstaining when none supports a value. Second, a human-in-the-loop mechanism frames verification as selective prediction and gates admitted data through 26 schema, statistical, and physical-plausibility rules. Third, we release an open benchmark of 41 datasets from over 25 institutions, with standardized SOH and RUL tasks, three split protocols, and eight baseline model families. The platform, benchmark, and curation protocol are publicly available at this https URL. 

---
# LegalFarePlan: A Label-Setting Framework for Fare-Transparent Urban Rail Route Planning under Non-Additive Fare Rules 

**Authors**: Tanghui Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.09755)  

**Abstract**: Urban rail fare systems may be non-additive: the fare of a single paid journey from an origin to a destination can differ from the sum of fares over multiple legally separated journey legs. This paper presents LegalFarePlan, a fare-transparent route-planning framework that models legal exit-and-reentry operations as explicit, auditable constraints. Given a transit network, fare function, transfer rules, station-level exit/re-entry costs, an extra-time budget, and a split limit, the planner computes explainable route plans over paid journey segments. The artifact implements Dijkstra shortest-time and direct route-planner baselines, a greedy split heuristic, bounded exact label-setting, and Pareto-frontier search. Evaluation uses controlled synthetic data and a 57-station semi-synthetic benchmark with 360 OD pairs. On the semi-synthetic benchmark, bounded exact search identifies positive modeled fare reductions for 71.11% of OD pairs, with mean reduction 3.78 and maximum reduction 9.0 synthetic fare units under a 45-minute extra-time budget. These results demonstrate method behavior and reproducibility; they are not empirical conclusions about MTR or any transit operator. 

---
# Task-Conditioned Synthetic Data Generation for Improving Machine Learning Performance in Agricultural Prediction Tasks 

**Authors**: Hamid Ebrahimy, Moritz Lucas, Martin Atzmueller  

**Link**: [PDF](https://arxiv.org/pdf/2607.09751)  

**Abstract**: Machine Learning (ML) algorithms have been widely used to estimate agricultural variables across diverse contexts. However, because the quantity and quality of training data strongly influence performance of ML algorithms, their use can be constrained by limited or incomplete reference data. Synthetic Data Generation (SDG) offers a practical approach to address this issue by producing artificial but realistic samples that preserve key characteristics of the original data. Building on teacher-student knowledge transfer and in-context learning for tabular data, this study proposes a Task-Conditioned SDG (TCSDG) algorithm that pairs a Bayesian Network generator with a transformer-based tabular foundation model (TabICL). The proposed algorithm was evaluated on two agricultural prediction tasks: crop yield prediction and crop type classification. Six benchmark SDG algorithms were also utilized to compare their performance with that of TCSDG. Across twelve study sites, two training-data fractions, four multiplication ratios, and three predictive ML algorithms, augmenting the original data with TCSDG-generated synthetic data improved ML performance in 89% of the crop type classification experiments and 74% of the crop yield prediction experiments. TCSDG also substantially outperformed benchmark SDG algorithms and was the only method to consistently improve ML performance across both tasks at the aggregate level. The study demonstrates that carefully designed and processed synthetic data can improve ML performance in precision-agriculture applications. TCSDG offers a practical and extensible framework for generating synthetic data that supports downstream ML agricultural prediction. The full implementation of TCSDG is publicly available as open source at this https URL. 

---
# Replicating Belief, Not Bits: Epistemic State Replication for Agentic Systems 

**Authors**: Jun He, Deying Yu  

**Link**: [PDF](https://arxiv.org/pdf/2607.09748)  

**Abstract**: In distributed systems, the classical State Machine Replication (SMR) model assumes that correct replicas execute deterministic transitions to yield identical bitwise states. However, the rise of agentic distributed systems -- where autonomous, stochastic, and model-driven agents orchestrate infrastructure -- presents scenarios where deterministic, bitwise replication is insufficient. Replicas operating with generative models may exhibit divergent reasoning paths, summaries, and token boundaries, yet reach semantically equivalent and correct operational decisions. Forcing bitwise agreement across these stochastic participants degrades execution flexibility, induces context amnesia, and limits performance.
We argue that in such settings replicas should agree on belief, not bits. We propose Epistemic State Replication (ESR), a belief-replication layer for agentic distributed systems that shifts the replication boundary from data visibility to knowledge visibility. We formalize the epistemic node state as a pair K = (L, B) separating the deterministic, immutable evidence log (L) from the stochastic, evolving belief lineage (B). To govern execution safety, we define Semantic Linearizability, which requires operations to reflect the latest committed operational meaning within a verifier-bounded semantic compatibility metric, and Bounded Eventual Coherence, which bounds expected semantic divergence under fair delivery, monotonic evidence, bounded verifier disturbance, and a contractive graft operator. We outline protocols for propagating derived insights using structured epistemic deltas, and formalize Verifiable Semantic Rollbacks to prune faulty premises from belief lineages without inducing context amnesia. We prototype ESR and report preliminary simulation results that show feasibility under the stated assumptions and illustrate reductions in secondary cognitive faults. 

---
# SupplyNetPy: An Open-Source Python Library for High-Fidelity Modeling and Simulation of Arbitrary Supply Chain and Inventory Networks 

**Authors**: Tushar Lone, Neha Karanjkar  

**Link**: [PDF](https://arxiv.org/pdf/2607.09745)  

**Abstract**: This paper introduces SupplyNetPy, an open-source, well-documented Python library for modeling and discrete-event simulation of supply chain networks with arbitrary multi-echelon structures. It supports multiple replenishment policies, perishable inventory, node disruptions, and stochastic demand and lead times. All components are extensible via inheritance. Users describe a supply chain as a graph with node and link attributes, while the library handles simulation, providing logs and extensive node and network level performance reports. This paper presents the motivation, design, key features, and architecture of SupplyNetPy, along with detailed validation results (against analytical benchmarks, a commercial tool, and a published case study). A key motivation behind SupplyNetPy's development is programmatic generation and simulation of complex models, enabling design-space exploration, what-if analysis, training data generation, and supply chain digital twins. 

---
# A Theory of Least Autonomy in AI 

**Authors**: Christophe Parisel  

**Link**: [PDF](https://arxiv.org/pdf/2607.09744)  

**Abstract**: Least privilege, the principle that an identity should hold only the permissions strictly required for its task, has been a foundational primitive of access control for decades. We argue that this principle is insufficient for agentic AI systems, which do not merely hold permissions but can combine, approve, and amplify them across workflows and system boundaries.
We propose least autonomy as an appropriate generalization and develop a formal theory. First, we define a compositional blast radius d(a,b) that measures structural separation between actions in an enterprise hierarchy, combining an ultrametric tree with lattice-valued confidentiality, integrity, and control-context labels. Second, we define a directed agent influence graph G(theta). An arc from U to V requires a directed shared-resource write-to-read meeting or a conservative undirected agent-to-agent (A2A) communication meeting, and a meeting-conditioned influence potential at or above an externally selected policy threshold theta. A catalogue-radius profile supports calibration and audit of theta. Finally, we define a collusion predicate over graph reachability that detects authorization composition, decision manipulation, and cross-domain capability composition. 

---
# Scaffolding the Strategist: Architecture-Dependent Reasoning Interventions in Hotelling Spatial Markets 

**Authors**: Pratyush Singh  

**Link**: [PDF](https://arxiv.org/pdf/2607.09743)  

**Abstract**: We investigate whether structured reasoning interventions improve the strategic economic reasoning of large language models, and whether their effects depend on model architecture. Using Hotelling's linear city model as a diagnostic vehicle, we evaluate GPT-4.1-mini (a standard instruction-following model) and GPT-5-mini (a reasoning-optimized model) under five conditions - an unscaffolded baseline and four reasoning interventions - across eight questions spanning deductive and abductive reasoning, three prompt framings, and three repetitions per condition, yielding 720 individually judged responses. We find a statistically significant crossover interaction between scaffolding type and model architecture ($t(7) = 4.79$, $p = 0.002$, $d = 1.69$): commitment scaffolding improves the standard model ($+0.21$) while degrading the reasoning model ($-0.63$), and principled separation shows the opposite pattern ($-0.40$ vs. $+0.31$). Both crossovers are individually significant (commitment: $p = 0.040$; separation: $p = 0.002$) and hold across all eight questions with 7/8 directional consistency. Adversarial stress-testing harms both models, with $2.6\times$ greater degradation for the reasoning model ($-1.47$ vs. $-0.57$; $p = 0.038$), and the damage correlates negatively with baseline difficulty ($R^2 = 0.36$, $p = 0.014$). We further document a persistent declarative-procedural gap in which both models identify correct strategies at rates far exceeding their ability to execute them; separation fully closes this gap for the reasoning model while no intervention helps the standard model. 

---
# A Dynamic Scene Interaction Reasoning Framework for Scene-level Lane-Change Intention and Trajectory Prediction of Multiple Interacting Vehicles 

**Authors**: Joshua Kofi Asamoah, Blessing Agyei Kyem, Eugene Denteh, Armstrong Aboah  

**Link**: [PDF](https://arxiv.org/pdf/2607.09740)  

**Abstract**: Safe motion planning in advanced driver-assistance systems and autonomous vehicles requires an accurate understanding of how the surrounding traffic scene is likely to evolve. However, many existing lane-change prediction methods remain centered on a single target vehicle, while multi-agent forecasting approaches often describe scene evolution only through future positions and provide limited explicit information about the maneuver associated with each vehicle. This study proposes a dynamic scene graph attention framework that predicts the lane-change intention and future trajectory of every relevant vehicle within a local traffic scene. The scene is represented as a time-varying interaction graph in which vehicles are modeled as nodes and their spatial and kinematic relationships are encoded through explicit edge features. Temporal graph-attention message passing captures evolving inter-vehicle dependencies and pre-maneuver cues, while an intention-guided decoder links each predicted maneuver to its corresponding future motion. A scene-level consistency objective further encourages compatible multi-vehicle futures. Experiments on the NGSIM I-80, NGSIM US-101, and highD datasets demonstrate consistent improvements over competing baselines. DSiGAT achieves intention prediction accuracies of 90.12% and 90.97% on NGSIM I-80 and US-101, respectively, and reduces trajectory RMSE by up to 52.94% relative to the strongest baseline. It also produces lower inter-agent collision rates and joint displacement errors, indicating more coherent scene-level predictions. Ablation, sensitivity, robustness, and qualitative analyses further validate the contribution of the proposed components and the effectiveness of the scene-focused formulation. 

---
# Coresets Before Score Sets: Evaluation-Unsupervised Prompt Subset Selection for LLM Benchmarks 

**Authors**: Jihan Yao, Gantavya Bhatt, Arnav Das, Peter Jin, Ke Bao, Qiaolin Yu, Khushi Bhardwaj, Chang Su, Jialei Wang, Yikai Zhu, Sugam Devare, Damon Mosk-Aoyama, Zhen Dong, Venkat Krishna Srinivasan, Yineng Zhang, Oleksii Kuchaiev, Jiantao Jiao, Banghua Zhu, Jeff Bilmes  

**Link**: [PDF](https://arxiv.org/pdf/2607.09739)  

**Abstract**: We study LLM benchmark coreset selection: selecting a small subset of prompts over multiple benchmarks whose induced model scores and rankings approximate those obtained from the full benchmark suite. In evaluation-unsupervised benchmark coreset selection (our approach), the selection algorithm uses no model evaluation outcomes, and operates on a fine granularity by producing subsets of prompts over multiple benchmarks rather than producing a sub-collection of entire benchmarks. We use submodular subset selection, and we develop and evaluate many different submodular functions for this purpose, including determinantal point process (DPP) based approaches, submodular mutual information functions, and facility location-based functions. On a new large-scale suite of 35 heterogeneous benchmarks spanning five different capability categories, 18 frontier LLMs, and over 61K prompts, we find that the facility location (FL) function operating exclusively on inexpensive semantic prompt embeddings preserves LLM scores better than twelve separate score-based and diversity-based baselines, across a range of coreset budgets. Moreover, we show our proposed objective is not limited to the evaluation-unsupervised regime: in the setting where only a handful of whole benchmarks must be selected and a large amount of model scores are available, the same objective matches or outperforms state-of-the-art baselines on the MMLU and MTEB leaderboards, while being substantially cheaper to compute. Together, our results suggest that submodularity, in general, is a strong and reliable tool for benchmark compression. 

---
# AGM-like Paraconsistent Partial Meet Abductive Expansion Operation 

**Authors**: Ulisses Franceschi Eliano  

**Link**: [PDF](https://arxiv.org/pdf/2607.09729)  

**Abstract**: In his 1996 doctoral thesis, Maurice Pagnucco created the first AGM-like abductive expansion operation. Taking his operation as a basis, as well as a taxonomy -- inspired by Atocha Aliseda -- responsible for highlighting and formalizing the main components of abductive reasoning, the main aim of this paper is to present a new paraconsistent AGM-like abductive expansion operation -- capable of assimilating contradictory explanatory hypotheses without trivialization and the consequent absurd epistemic state -- with its postulates and its transitively relational partial meet construction. To a large extent, the formal development presented in this paper was only made possible by the recent creation of the paraconsistent logic RCbr, an LFI (Logics of Formal Inconsistencies) that establishes properties especially relevant to belief revision contexts, in particular, the ability to be self-extensional -- i.e., to satisfy the replacement property. This is the first of two papers: the paraconsistent abductive expansion operation announced here -- which is part of a new system called AGMpabd -- despite bringing many interesting features, does not assign any relevant epistemic role to the paraconsistent operators of negation and consistency. Only in a second paper will an analogous paraconsistent abductive expansion operation -- which is part of another new system, AGMcircabd -- be enhanced in this direction. Nevertheless, to the best of my knowledge, the operation developed in this paper is the first of its kind in the AGM literature. 

---
# Feedback-Coupled Memory Systems in Continuous Time 

**Authors**: Stefano Grassi  

**Link**: [PDF](https://arxiv.org/pdf/2607.09714)  

**Abstract**: The Feedback-Coupled Memory Systems (FCMS) architecture formalizes closed-loop coordination through four abstract operators, two of which - the agent update operator $f_i$ and the environmental update operator $\Psi$ - are left axiomatically undefined in the original framework. To address this, $f_i$ is defined by Mechanism-Based Intelligence (MBI), where agents update locally through a decentralized price mechanism and economic principles, and $\Psi$ is defined by the Coupled Memory Graph Process (CMGP), a non-Markovian framework where the environment is treated as a physical substrate that records and responds to trajectory history coherently without external forcing. The resulting continuous-time FCMS instantiation achieves Lyapunov global dissipativity governed by the computable threshold $4\beta^2 < 2\eta\mu\gamma^2$. This generalizes both the discrete FCMS stability condition $4\eta\beta^2 < \gamma$ and CMGP's physical bifurcation threshold $\alpha_c = 1/K$, confirming that memory dissipation must outpace feedback gain as a universal organizing principle. Numerical simulation with $N=2$ agents and mean-field validation at $N=10^6$ confirm the stability threshold and the self-reinforcing coordination cascade that emerges when it is violated. 

---
# Closed-Loop Control with Rule-Aligned Small Language Models and Multi-Agent Self-Correction 

**Authors**: Yuchen Wang, Javal Vyas, Tong Liu, Mehmet Mercangoz  

**Link**: [PDF](https://arxiv.org/pdf/2607.09713)  

**Abstract**: A key step toward autonomous industrial operation is the ability to create and reconfigure control policies from natural-language requirement specifications, with minimal or no manual redesign. In this setting, policy generation by AI agents can be a credible path when paired with a plant-aware validator (e.g., a digital twin) that can check generated candidate actions before execution. However, practical deployment is constrained by inference latency and compute footprint: large cloud-based models are often too slow, opaque, or data-sensitive for edge closed-loop use. This work investigates whether a compact Small Language Model (SLM) can be retrained for control reasoning and embedded in a validator-guided correction loop. We use a Qwen2.5-1.5B model aligned via Group Relative Policy Optimization (GRPO), combined with (i) an action agent, (ii) a symbolic/digital-twin-style validation layer, and (iii) a reprompting agent that iteratively steers outputs toward valid actions. In randomized thermal-control simulations (30 experiments with 500 steps each), the framework achieves 91.5% average action-alignment accuracy (86.3%--100% across cases) at 3.84\,s mean inference latency. Under symbolic re-mapping, it maintains a 95% in-range rate, indicating robust physical regulation despite reduced token-level agreement. These results support SLM+validator architectures as a practical path toward reconfigurable autonomous control at the edge. 

---
# The Verifier is the Curriculum: Execution-Gated Self-Distillation for Cross-Family Game Generation 

**Authors**: Chenyu Zhou, Qiliang Jiang, Shuning Wu, Xu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2607.09709)  

**Abstract**: Post-training a code generator against a learned judge can optimize proxy features that raise the score without improving the artifact. We study the opposite signal: a deterministic, judge-free, ungameable filter -- whether a generated project launches cleanly under a headless engine (strict-launch). Under this gate, rejection-sampling self-distillation compounds out-of-family generalization. On GameCraft-Bench (mapping a natural-language brief to a complete Godot project), a 14B model (Qwen3-14B+LoRA) distilled under strict-launch raises clean generation on four unseen game families from 8.8% to 42.2% per-candidate and best-of-K coverage from 18/25 to 25/25 (the gold ceiling) over three rounds, each a significant gain (p=0.0019, p<1e-4, p<1e-4). The gain is not from merely adding data: an exactly-matched gold-duplication control regresses below the base model (5.6% vs. 8.8%, p=0.019), while a count-matched decomposition splits the round-1-to-2 jump into comparable quality (+8.8pp) and quantity (+8.5pp) channels. Most directly, rerunning the loop with only the filter swapped -- the lenient BUILD check, which passes 99.9% of generations, in place of the launch gate -- erases the gain entirely (back to base, p=1e-3 vs. the launch-gated round), isolating verifier precision rather than the optimizer. A second ungameable signal, headless execution grounding, rises monotonically across rounds and yields far more grounded candidates than gold-duplication at a matched budget (16 vs. 5), confirming the gains are functional, not launch-but-empty. Game generation is a verifiable testbed for one lesson: the verifier is the curriculum -- what it certifies is what the model learns. 

---
# GES-TSP: Graph Edge Sparsification for TSP 

**Authors**: Tianfeng Chen, Xianyue Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.09708)  

**Abstract**: Solving large-scale instances of the Traveling Salesman Problem (TSP) exactly is computationally expensive. Researchers often employ graph sparsification methods to improve computational efficiency. Traditional sparsification methods typically rely on fixed heuristics and fail to fully exploit instance-specific structural information. In this paper, we propose Graph Edge Sparsification (GES), a learning-based sparsification approach for Euclidean TSP. By incorporating geometric structural information and combinatorial optimization technology, our proposed method adaptively generates a sparsification graph for different instances, significantly reducing the graph size and accelerating the solving process. Experimental results demonstrate that our sparsification method can prune up to 95% of edges on the MATILDA dataset, while keeping the solution gap within 1% of the optimal value. Moreover, our approach exhibits strong generalization capability on the TSPLIB this http URL some large-scale instances, the pruning rate exceeds 99%, while the optimality gap remains below 1%. 

---
# YUKTI: From Natural-Language Situations to Robust, Verifiable Decisions An Uncertainty-Typed Proposition IR, Assumption-Robust Pareto Frontiers, and a Regret Certificate 

**Authors**: Suyash Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2607.09706)  

**Abstract**: Language models turn a worded situation into a numeric plan, and the dominant pipelines (NL4Opt, OptiMUS, ORLM, OR-LLM-Agent) commit to a single objective and point-valued coefficients, then solve once. For decisions that allocate real budget, effort, or clinical attention, that confidence is the failure mode: every objectified number is an assumption, and a plan optimal only if the guesses are exactly right is fragile -- mimicry of computation. YUKTI changes the target of autoformulation. Its representation is a typed-proposition graph whose relationships carry shape priors, coefficient uncertainty, and provenance. YUKTI routes each stage to an exact, nonlinear, or evolutionary solver; couples stages by a distributional Pareto hand-off; and introduces Assumption-Robust Pareto Frontiers (ARPF), resampling assumptions (including structural epsilon-contamination) to score how often each action survives (rho). We prove a bound making rho an exact factor of decision regret, add auditable traceability, and synthesize a benchmark-faithful data foundation when none exists (SRJANA). We validate three ways: under controlled misspecification the robust compromise cuts mean and tail regret by over 90% versus a naive point plan; on a regulated commercial decision we optimize inside a lawful action space and price the downside in euros; and on a real public dataset of 41,188 decisions an out-of-sample backtest beats the logged status quo by 34% and a naive point rule by 4% while reducing the optimizer's curse. The solvers are standard; we claim no benchmark-SOTA win. A head-to-head shows an LLM given the correct numbers, and single-objective optimization, both incur about 47x the held-out regret of YUKTI -- an LLM is a formulator, not a solver. Under long-range causal coupling, the forward hand-off becomes unsound, locating where it must become a backward-induction causal policy. 

---
# Interpreting Latent CoT Reasoning as Dynamical Systems 

**Authors**: Sabari Iyyappan Duraipandian, Shreya Sanjay Boyane, Manju Nagesh, Jerome Francis, Archana Vaidheeswaran, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.09698)  

**Abstract**: Recent latent reasoning methods, such as CODI and COCONUT, face a fundamental interpretability problem: they maintain multiple superimposed candidate traces in the hidden space at each step, unlike explicit- CoT, which follows a single transparent reasoning trace. Existing mechanistic methods show compression, shortcuts, and superposition without explaining how reasoning evolves across latent steps. To address this gap, we model latent token sequences as trajectories in representation space and apply dynamical systems analysis to characterize the evolution of reasoning. Using quantitative measures, such as step-to-step change, direction consistency, and Lyapunov sensitivity, alongside qualitative projections, such as UMAP and DMD/PHATE, we show that latent CoT exhibits structured, non-random dynamics with two distinct stability classes. CODI behaves as a stable attractor, while COCONUT behaves as an unstable expanding system, and SIM-CoT supervision tightens both behaviors without changing the underlying dynamics. This framework advances the interpretability of latent CoT reasoning dynamics and provides actionable insights for improving latent reasoning performance. Code1 and Project page2 available online. 

---
# Boltzmann MapReduce: A Partition-Function Reduce for Forkable Sandboxes 

**Authors**: Yossi Eliaz  

**Link**: [PDF](https://arxiv.org/pdf/2607.09689)  

**Abstract**: To leading order under local asymptotic normality (LAN), the confidence density a worker emits over a chunk of size $n$ is a Gibbs--Boltzmann measure $\exp\{-\beta E(\theta)\}$ whose inverse temperature is the sample size, $\beta=n$. Three consequences are exact in the Gaussian/linear case and first-order otherwise: disjoint chunks carry independent Boltzmann factors, so the MapReduce \emph{reduce}, read literally, is a partition function $Z=\int\prod_k h_k\,d\theta$ whose mode is precision-weighted (inverse-variance) pooling; frequentist consistency is the zero-temperature limit $T=1/n\to0$ 

---
# Faithful, Not Corrective: Message-Format Effects in Multi-Hop Agent Relays Are Tier-Dependent 

**Authors**: Zayx Shawn  

**Link**: [PDF](https://arxiv.org/pdf/2607.09678)  

**Abstract**: When LLM agents hand off information to one another, does the message format matter? Two literatures disagree: format-optimization work reports that structured messages cut cost without hurting accuracy, while format-restriction work finds that imposing structure degrades generation -- and neither measures what happens when a message traverses multiple hops, where copy fidelity, not one-shot generation, dominates. We introduce a controlled relay testbed: briefs of twelve programmatically generated atomic facts are re-encoded hop-by-hop in five formats (free NL, precision-instructed NL, JSON, triples, key-value) over six hops, scored by a fixed strong grader against programmatic ground truth, across two relay-capability tiers, a cognitive-load condition, and a paired-fork error injection. We find that message-format effects are tier-dependent. (i) Under faithful-relay instructions a strong relay is nearly lossless -- the documented "telephone-game" collapse does not occur -- and adding per-hop cognitive load leaves format-level fidelity unchanged (within +/-1.8 points) while raising generation cost by 24-53%. (ii) Under a weak (1.5B) relay the across-format spread of six-hop recall grows by a factor of 8.7 (from 2.3 to 20.5 points), driven by two opposing mechanisms -- an encoding toll paid by the rigid formats and drift resistance specific to the fixed-key JSON schema -- that flip the format ranking in transit. (iii) In a paired-fork injection, an injected wrong value, once present, persists to the final hop in 83-100% of chains in every format, closely matching each format's retention of the true value, with no detectable collateral damage to neighboring facts. Structure buys a faithful, error-localizing channel -- not an error-correcting code -- and format choice should follow the weakest relay in the pipeline. 

---
# Format Sensitivity Index: Token-Controlled Prompt Wrapper Robustness and Schema Compliance in LLM Benchmarking 

**Authors**: Deep Pankajbhai Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2607.09665)  

**Abstract**: Prompt wrappers often differ only in formatting, yet they can change model scores enough to flip leaderboard conclusions. We study this variance under a token-controlled protocol and introduce two complementary metrics: the Format Sensitivity Index (FSI), the accuracy range induced by wrapper choice, and the Parseability Sensitivity Index (PSI), the corresponding range in answer parseability. Across 140,000 OpenRouter generations spanning 7 QA tasks, 5 wrapper families, and 4 instruct models from 7B to 72B parameters, we find that mean FSI varies by over 30x across models and is largely explained by compliance failures. A fixed-effects regression shows that parseability remains a strong predictor of accuracy even after controlling for task, model, and wrapper. We argue that reporting accuracy without wrapper variance and compliance is statistically fragile, and we give practical recommendations for both benchmarking and structured-output deployments. 

---
# From ML Predictions to Informed Diagnostic Assistance Using the Toulmin Model of Argumentation 

**Authors**: Anca Marginean, Adrian Groza  

**Link**: [PDF](https://arxiv.org/pdf/2607.09664)  

**Abstract**: To provide a structured and interpretable assessment, we decompose the image-based diagnosis into components following the Toulmin model of argumentation. This model consists of a claim, grounds, warrant, qualifier, rebuttal, and backing. Consider a claim generated by a machine learning (ML) model for retinal diagnosis. Rather than accepting this claim at face value, one could either apply explainable AI (XAI) methods or adopt an argumentation-based approach. In our framework, a model specialized in biomarker extraction from images provides the grounds. The warrant-linking the grounds to the claim - is analyzed by an agent equipped with medical knowledge; in our architecture, this role is fulfilled by a MedGemma agent. The qualifier is determined based on the overall quantitative evaluation of both the warrant and grounds models. Finally, a rebuttal is constructed using image similarity measures computed with MedSigLip. All these components are presented to the human expert, enabling a more informed and critical assessment of the ML-generated diagnosis. 

---
# Metacognition in LLMs: Foundations, Progress, and Opportunities 

**Authors**: Gabrielle Kaili-May Liu, Areeb Gani, Jacqueline Lu, Jordan Thomas, Mark Steyvers, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2607.11881)  

**Abstract**: Metacognition is a foundational component of intelligence critical to effective learning, problem solving, decision-making, communication, and more. In recent years, it has become increasingly recognized as a cornerstone of capable, transparent AI systems. Yet while LLMs have made significant progress across diverse real-world tasks, it is not yet clear when, how, or to what extent they can exhibit or be endowed with effective metacognitive abilities, nor how such abilities can be adapted to advance the fundamental capabilities, reliability, and intelligence of AI systems. This paper bridges this gap by presenting the first comprehensive overview of the current state of knowledge on metacognition for LLMs. We analyze and taxonomize the landscape of this emerging field and summarize recent technical advancements, including methods and benchmarks to measure and evaluate LLMs' metacognitive abilities, techniques to elicit, improve, and apply metacognition in LLMs, and findings and implications of ongoing research. We also discuss applications, open questions and challenges, and promising directions for future work. Our aim is to provide a detailed and up-to-date review of this topic and stimulate meaningful research and discussion. An organized list of papers can be found at this https URL. 

---
# Invariant Learning Dynamics of Transformers in Inductive Reasoning Tasks 

**Authors**: Tiberiu Musat, Tiago Pimentel, Nicholas Zucchet, Thomas Hofmann  

**Link**: [PDF](https://arxiv.org/pdf/2607.11875)  

**Abstract**: We present a theoretical framework to explain the emergence of inductive reasoning abilities in Transformer language models. While previous works on Transformer learning dynamics have so far been mostly tied to specific tasks, we study a generalized class of inductive tasks that unifies several synthetic tasks known in the literature, including in-context n-grams and multi-hop reasoning. In this class, we theoretically prove that the training dynamics of attention models can be confined to a highly interpretable, low-dimensional invariant manifold. On this manifold, the learning dynamics are captured by a handful of interpretable coordinates rather than millions of parameters, making both theoretical and empirical analysis more tractable. Using this framework, we characterize how data statistics govern the competition between in-context and in-weights learning, we study how random initializations determine the `winning' circuit when multiple solutions are possible, and we demonstrate that the coordinate frame associated with the manifold can be used to automatically detect which circuits have been learned in trained models. By casting circuit formation as a low-dimensional dynamical phenomenon, we take a step toward a predictive theory of how Transformers learn. 

---
# A Minimalist Retargeting-Guided Reinforcement Learning Recipe for Dexterous Manipulation 

**Authors**: Yunhai Feng, Natalie Leung, Jiaxuan Wang, Lujie Yang, Haozhi Qi, Preston Culbertson  

**Link**: [PDF](https://arxiv.org/pdf/2607.11874)  

**Abstract**: Recent work in humanoid whole-body control has found success with a simple recipe: retarget human motion to robot kinematic references, then train policies via reinforcement learning (RL) to track them. But how does this recipe transfer to dexterous manipulation? The answer is not obvious, as manipulation involves complex, contact-rich dynamics and requires delicate regulation of contact modes and forces. We present REGRIND, a minimalist retargeting-guided RL pipeline that learns dexterous manipulation policies from a single human demonstration. REGRIND retargets human hand-object motion to a robot reference that preserves hand-object spatial and contact relationships, trains a residual RL policy in simulation to track object-centric keypoints along that reference, and transfers the resulting policy zero-shot to hardware with careful system identification. The resulting policies produce fluid, human-like behavior on two different multi-fingered hands across contact-rich tool-use tasks, including operating a pair of scissors and turning a screwdriver. Through systematic hardware experiments, we identify and analyze the key factors that govern sim-to-real transfer in dexterous manipulation, offering practical guidance for retargeting-based learning in contact-rich settings. Videos and code are available at this https URL. 

---
# Inside the Unfair Judge: A Mechanistic Interpretability Account of LLM-as-Judge Bias 

**Authors**: Zixiang Xu, Sixian Li, Huaxing Liu, Xiang Wang, Shuai Li, Zirui Song, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.11871)  

**Abstract**: Existing studies of LLM-as-judge scoring bias work predominantly at the input-output level: they perturb inputs, measure score deltas, and propose prompt-level mitigations. We argue that the same biases admit a representation-level account in the judge's hidden state, complementary to the input-output view and operationally useful in ways it does not afford. We report three findings, across seven judges, seven bias types, and nine benchmarks. Geometry: baseline judging inputs occupy a tight activation manifold while biased inputs are displaced along a low-dimensional, type-specific subspace that sharpens with depth and is recovered consistently by three families of estimators. Causal control: steering hidden states along this subspace drives scoring in both directions, forward shifts reproducing biased scoring on clean inputs and reverse shifts restoring baseline scoring on biased ones, while matched-norm random directions produce shifts an order of magnitude smaller. Operational: a simple linear projection onto the same bias-direction features anticipates judge failures on three entirely unseen benchmarks, substantially outperforming text-based alternatives. Reading bias as activation geometry, rather than as input-output noise, unifies geometric structure, causal control, and operational prediction within a single framework. The project page is available at this https URL 

---
# Evidence-Backed Video Question Answering 

**Authors**: Shijie Wang, Honglu Zhou, Ziyang Wang, Ran Xu, Caiming Xiong, Silvio Savarese, Chen Sun, Juan Carlos Niebles  

**Link**: [PDF](https://arxiv.org/pdf/2607.11862)  

**Abstract**: Current Video Large Language Models (Video LLMs) excel in question answering (QA) but largely operate as black boxes, providing textual answers without verifiable visual grounding. Existing explainability efforts rely on textual rationales or sparse bounding boxes, which struggle to capture complex video dynamics such as occlusions and non-rigid deformations. We propose Evidence-Backed Video Question Answering (E-VQA), a novel task requiring models to jointly output a semantic answer and precise spatio-temporal evidence: temporal segments and dense, tracked object segmentation masklets. To support this, we introduce ST-Evidence, the first human-verified benchmark for both discriminative and generative pixel-level grounding. Evaluations of state-of-the-art models reveal a critical decoupling between QA accuracy and true visual perception that scaling alone fails to bridge. To address this, we develop scalable, automated generation pipelines to create ST-Evidence-Instruct, a 160k-scale dataset bridging high-level reasoning with fine-grained grounding. Fine-tuning grounded Video LLMs on this data yields substantial gains over the corresponding size-matched UniPixel baselines (e.g., +27.2 t-mean and +13.8 J&F on a 7B model), establishing a robust baseline for explainable, evidence-backed video understanding. Code and data are available at this https URL. 

---
# LoRA-Based Cascaded Multimodal Fusion for Action Recognition in Medical Training Environments 

**Authors**: Divya Mereddy, Jeevan Beedareddy  

**Link**: [PDF](https://arxiv.org/pdf/2607.11839)  

**Abstract**: This paper presents a cascaded Low-Rank Adaptation (LoRA)-based multimodal fusion framework for action and activity recognition in healthcare-oriented training environments. The proposed architecture combines parameter-efficient modality-specific adaptation with sequential fusion, enabling modalities to be integrated in stages without retraining previously learned components. Rather than assuming a fixed fusion structure, the framework first integrates more closely related modalities and then incorporates additional heterogeneous modalities, supporting scalable adaptation across datasets with different modality this http URL evaluate the framework on two healthcare-oriented training environment datasets: NurViD and the Nurse Training dataset. Across these datasets, preliminary results suggest that the proposed cascaded fusion strategy improves over individual modality models and provides competitive performance relative to previously reported dataset-specific baselines. Overall, these findings indicate that cascaded LoRA-based fusion is a promising parameter-efficient approach for integrating heterogeneous modalities in medical training action and activity recognition tasks. github: this https URL. 

---
# Transformer-Guided Swarm Intelligence for Frugal Neural Architecture Search 

**Authors**: Romain Amigon  

**Link**: [PDF](https://arxiv.org/pdf/2607.11826)  

**Abstract**: Neural Architecture Search (NAS) has automated the design of deep learning models but traditionally requires massive computational resources, often measured in thousands of GPU-days. In this paper, we propose a frugal and memetic NAS framework designed to democratize architecture design on consumer-grade hardware. Our approach combines the global macro-search capabilities of an autoregressive Transformer controller, trained via Reinforcement Learning (RL), with the local micro-exploitation of an Artificial Bee Colony (ABC) algorithm. To prevent premature convergence during the RL phase, we introduce a dynamic entropy mechanism that forces topological exploration upon detection of performance stagnation. Evaluated on a standard GPU (NVIDIA RTX 3060), our hybrid method effectively resolves the "cold-start" problem inherent in metaheuristics. By algorithmically penalizing network depth, our framework actively mitigates model bloat: on the CIFAR-10 dataset, it discovers an efficient architecture reaching 84.85% accuracy with only $\sim$174,000 parameters (significantly smaller than standard baselines like ResNet-20) in 3 hours of search time. Furthermore, we demonstrate the framework's flexibility by applying it to credit card fraud detection, directly optimizing the F1-Score on highly imbalanced tabular data to reach a F1-Score of 0.71 with a compact network of $\sim$4,600 parameters. These results suggest that our approach can yield tailored, accessible, and highly parameter-efficient deep learning models suitable for edge deployment. 

---
# MM-ToolSandBox: A Unified Framework for Evaluating Visual Tool-Calling Agents 

**Authors**: Kaixin Ma, Di Feng, Alexander Metz, Jiarui Lu, Eshan Verma, Afshin Dehghan  

**Link**: [PDF](https://arxiv.org/pdf/2607.11818)  

**Abstract**: We introduce MM-ToolSandBox, a benchmark and evaluation framework for visually grounded tool-calling agents. The framework provides a stateful execution environment spanning 500+ tools across 16 application domains, supporting multi-image, multi-turn tasks where agents must ground progressively arriving visual inputs into executable tool calls while handling realistic conversational phenomena (goal revisions, error corrections, state mutations). An automated scenario generation pipeline produces diverse, visually grounded scenarios through information-flow-guided planning and multi-stage quality filtering, yielding 258 human-verified nominal scenarios and 50 variants targeting interactive UI applications. Evaluating 12 state-of-the-art models, from 4B open-weight to frontier proprietary systems, shows that current models still lack robust visual tool-calling capability: even the best model achieves below 50% success rate. Our failure analysis further reveals that visual precision, not only planning, is a primary bottleneck for capable models: 53% of failures stem from incorrect information extraction from images despite otherwise correct task workflows. A planning-to-precision crossover emerges with scale: smaller models fail at deciding what to do, while larger models fail at perceiving what they see, suggesting fundamentally different research directions for improving models at different capability levels. The framework and the benchmark are publicly available at this https URL 

---
# Introducing Human-Centeredness in AI-Assisted Lexicography 

**Authors**: Antonio San Martin, Catherine Trekker  

**Link**: [PDF](https://arxiv.org/pdf/2607.11808)  

**Abstract**: This paper proposes a human-centered artificial intelligence (HCAI) framework for AI-assisted lexicography. While generative AI offers significant opportunities to enhance lexicographic work, it also raises concerns regarding the future role of lexicographers and the preservation of linguistic and cultural diversity. Drawing on HCAI principles and previous applications in other language professions, the paper identifies four interrelated dimensions through which AI integration in lexicography can be understood and critically examined: the augmented lexicographer, the sociotechnical context of AI integration, bias, and the design of AI-powered lexicographic tools. The framework argues that AI should augment rather than replace lexicographers, combining high levels of automation with meaningful human control. It further emphasizes the importance of preserving professional agency, mitigating AI-generated biases, and designing tools around the needs of lexicographers. By doing so, the paper provides a foundation for future research and the beneficial integration of AI into lexicographic workflows. 

---
# Encoder-Side Neuron Identification and Amplification for Acoustic Perception in Large Audio-Language Models 

**Authors**: Yu-Han Huang, Chih-Kai Yang, Ke-Han Lu, An-Yu Cheng, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2607.11801)  

**Abstract**: Large audio-language models (LALMs) often underperform on fine-grained, non-semantic attributes of speech, such as a speaker's emotion, despite strong performance on speech content. Improving this without the cost of retraining calls for an effective inference-time intervention, yet most existing methods intervene only after the audio encoder and operate at a relatively coarse granularity. The encoder itself, where acoustic information is first extracted from the waveform, remains largely unexplored, especially at the level of individual neurons. We introduce IAAN, Identifying and Amplifying Acoustic Neurons, a training-free and label-free method that scores each feed-forward neuron in the audio encoder by contrasting its activation on the real waveform with that on a noise reference lacking the real audio's acoustic information. IAAN then amplifies a small set of the highest-scoring neurons at inference. Across ten non-semantic speech attributes, IAAN improves average accuracy by 25.7 points on Audio-Flamingo-3, 21.4 on Qwen2.5-Omni, and 9.7 on Kimi-Audio. It also improves a model already explicitly fine-tuned to prioritize acoustic evidence. In controlled comparisons, both the encoder locus and neuron-level selectivity prove necessary for this gain. Intervening after the encoder, at the decoding side or inside the language model, yields little to no improvement, or even deteriorates accuracy. The improvement also depends on which specific neurons are amplified, not merely on their number, confirming that IAAN's acoustic score succeeds in identifying the neurons that matter. These results show that a small, precisely targeted intervention inside the audio encoder is an effective and largely untapped way to strengthen the acoustic understanding of LALMs, opening a new direction for inference-time methods that improve acoustic perception through neuron-level access to the encoder. 

---
# StoryTeller: Training-Free Narrative Grounding for Long-Form Audio Description 

**Authors**: Seung Hyun Hahm, Minh T. Dinh, SouYoung Jin  

**Link**: [PDF](https://arxiv.org/pdf/2607.11798)  

**Abstract**: Long-form audio description (AD) requires more than describing visible actions: it must preserve characters, events, relationships, and story context across scenes so that blind and low-vision (BLV) audiences can follow a film. Modern video-language models (VLMs) are effective on short clips, but they often treat each moment independently, producing descriptions that miss who characters are, why events matter, and how the current scene connects to earlier narrative context. We propose StoryTeller, a training-free framework for story-aware long-form AD. Instead of relying only on local visual cues, StoryTeller maintains a verified narrative memory that carries forward story-relevant information across scenes, enabling later descriptions to remain coherent, grounded, and contextually informative. Given only raw video and a movie title, StoryTeller can optionally retrieve public movie metadata to resolve names and story context, while accepting only facts that are supported by the video through semantic filtering and VLM verification. The method requires no subtitles, scripts, AD transcripts, aligned captions, character banks, precomputed face identities, or task-specific fine-tuning. To evaluate whether generated AD preserves narrative information, we introduce StoryAD-QA, a question-answering benchmark that tests whether a language model can answer story-context questions using only the generated descriptions. Experiments on standard AD benchmarks and diverse long-form videos show that StoryTeller consistently improves narrative coherence, factual grounding, and story comprehension over strong baselines in automatic, QA-based, and human evaluations. 

---
# Evaluating RE Practices for Explainability: Synthesizing Insights from Daimler Truck into an Explainable RE Framework Proposal 

**Authors**: Umm-e- Habiba, Lucas Mauser, Jonas Fritzsch, Justus Bogner, Stefan Wagner  

**Link**: [PDF](https://arxiv.org/pdf/2607.11771)  

**Abstract**: Explainability has emerged as a critical requirement for AI-based systems, particularly in safety-critical and regulated domains. Although prior research has proposed frameworks, patterns, and user-centered approaches to support explainability, there is limited empirical understanding of how existing Requirements Engineering (RE) practices support explainability requirements across the RE lifecycle, especially in an industrial context. This paper reports early findings from an ongoing industry-based study investigating how explainability requirements are elicited, specified, and validated using established RE techniques. We conducted a multi-phase qualitative study with eight practitioners at Daimler Truck, employing think-aloud protocols and moderated group discussions across requirements elicitation, specification, and validation steps. Our preliminary analysis reveals recurring challenges across all steps, including conceptual ambiguity during elicitation, limited testability and expressiveness during specification, and fragmented validation due to vague criteria and regulatory uncertainty. These findings indicate that current RE practices provide limited support to systematically address explainability requirements. The paper contributes empirical insights into step-specific and cross-cutting challenges and outlines a research vision toward developing an empirically grounded RE framework for explainable AI-based systems. 

---
# Time-Lag-Aware Deep Reinforcement Learning for Flexible Job-Shop Scheduling in PPVC Module Factories 

**Authors**: Ziheng Zhang, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.11725)  

**Abstract**: Prefabricated prefinished volumetric construction moves most building work into module factories, whose production floor operates as a flexible job shop. A major complication is decisive: long post-operation time-lags caused by concrete curing, watertightness ponding tests, and paint drying, during which a module is blocked while its workstation stays free. On benchmark instances grounded in an official national prefabrication guidebook, these lags inflate even the optimal reference makespan by about 67% on average, and ignoring them at decision time, then repairing to feasibility, is worse than every dispatching rule. We adapt a state-of-the-art dual-attention deep reinforcement learning solver through three minimally invasive, individually ablatable extensions: lag-aware dynamics with an admissible reward bound, two anticipatory lag feature channels, and liveness-masked operation- and station-type embeddings. With every extension disabled the implementation reproduces the original solver exactly, so all gains are attributable to the adaptations. We release a public, guidebook-grounded benchmark generator. On held-out instances the learned policy is the strongest solver-free scheduler: it reaches within about 4% of a constraint-programming reference and beats every dispatching rule and a genetic-algorithm metaheuristic, with its advantage widening under capacity contention, and a single size-mixed policy carries this lead across the trained range of factory sizes. It needs no solver, model, or license in the loop and re-plans within seconds of a disruption; where an exact solver can be deployed, that solver remains the quality ceiling, a boundary we map explicitly. 

---
# Active Offline-to-Online Reinforcement Learning 

**Authors**: Alper Kamil Bozkurt, Shangtong Zhang, Yuichi Motai  

**Link**: [PDF](https://arxiv.org/pdf/2607.11720)  

**Abstract**: Background: Offline reinforcement learning (RL) enables effective policies to be trained from large, previously collected datasets and subsequently improved through limited online interaction. This offline-to-online RL (O2O-RL) paradigm is particularly promising in nonstationary domains where interaction is costly or potentially hazardous. Standard O2O-RL pipelines train multiple candidate policies offline, evaluate them using off-policy or online evaluation, and then deploy and fine-tune the policy with the highest estimated value. However, as in offline pretraining, fine-tuning performance is highly sensitive to the choice of algorithm and hyperparameters, making it risky to commit to a single policy. Objectives: We study active policy selection for fine-tuning under a limited interaction budget in O2O-RL settings. To our knowledge, this is the first work to address this problem. Methods: We formulate the problem by identifying a fundamental trade-off between allocating online interactions to policy evaluation, which helps identify high-performing policies, and allocating them to fine-tuning, which improves policy performance. We then propose an approach that balances this trade-off by actively selecting policies for fine-tuning based on upper-confidence bounds on their future performance. These bounds are derived from locally linear performance forecasts fitted to observations obtained through online evaluation. Results: Across a diverse range of experiments, the proposed approach consistently outperforms existing O2O-RL baselines. Conclusions: Actively selecting and fine-tuning policies uses limited online interaction budgets more effectively than either committing to a single policy or dividing the budget equally among all policies. Our framework also advances offline RL toward practical deployment in real-world systems where online interaction is costly or risky. 

---
# An Explainable Agentic System for Detection of Conversational Scams with Summary-Based Memory 

**Authors**: Ahmed Omar Salim Adnan, Yogananda Manjunath, Shivanjali Khare  

**Link**: [PDF](https://arxiv.org/pdf/2607.11707)  

**Abstract**: Following the rapid progress of generative Artificial Intelligence, there is a growing threat posed by conversational scams. These scams often span over multiple weeks or months, gradually build trust and request for money or sensitive information. Existing scam-detection systems mainly focus on isolated messages, which renders them inadequate against this evolving threat. This paper extends single-message phishing detection and presents an explainable agentic system for detecting sophisticated conversational scams. It also introduces ConScamBench-278, an initial public multi-category benchmark for conversational scam detection spanning eight scam types, released to support reproducible evaluation and future expansion. On isolated messages the single-message detector attains 100% phishing recall, while the conversation-level detector identifies all conversational scams in the public LoveFraud02 corpus (83/83) and reaches 97.8% accuracy (95% CI [95.4, 99.0]) on ConScamBench-278. Two user studies (N = 100 and N = 45) further motivate the system: participants report frequently experiencing uncertainty when judging suspicious conversations. In an uncontrolled pre/post comparison, users self-reported trust, self-confidence, and perceived need for AI-based scam detection all increased (p < 0.001, Wilcoxon signed-rank). The system also receives a System Usability Scale score of 74.7 (95% CI [72.5, 76.9]), above the established usability benchmark. 

---
# VoxENES 2026: Benchmarking Generalization of Speech Spoofing Detectors Against LLM-Era TTS and Voice Conversion 

**Authors**: Aastha Sharma, Guangjing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.11706)  

**Abstract**: Modern LLM-driven text-to-speech (TTS) and voice conversion (VC) systems produce synthetic speech that differs from the generators represented in many legacy spoofing benchmarks. This mismatch creates a temporal generalization gap that can overestimate detector robustness under real-world post-processing conditions. We bridge this gap by introducing VoxENES 2026, a bilingual (English and Spanish) benchmark of 53,628 audio samples generated using 10 contemporary speech synthesis methods and evaluated under 10 standardized post-processing conditions. Using VoxENES 2026, we benchmark eight pretrained detectors without fine-tuning and observe substantial performance degradation: the best model achieves 28.98\% EER overall, while most perform near or below random chance across modern generators and perturbations. Our results highlight the reliance on brittle artifacts in current detectors and establish VoxENES 2026 as a practical testbed for developing robust audio spoofing countermeasures. 

---
# Agent Hacks Agent: Autoresearch for Production-Agent Red-Teaming 

**Authors**: Xutao Mao, Xiang Zheng, Cong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.11698)  

**Abstract**: Production LLM agents such as Claude Code and Codex operate over untrusted content, files, commands, and workspace state, making safety failures directly actionable. Red-teaming must therefore keep pace with evolving models and tools. Existing approaches mainly optimize attack success and preserve artifacts such as benchmarks, payloads, or attack programs, which record where attacks succeed but not the enabling conditions behind unsafe agent behavior. We study automated red-teaming for production LLM agents using one agentic research environment to discover reusable vulnerability knowledge about another. We present AHA, a falsifiable discovery loop that proposes a vulnerability hypothesis, constructs a falsifier, instantiates a valid attack, executes it in a sandboxed harness, reflects on the trajectory, and promotes confirmed findings into a Vulnerability Concept Graph (VCG). Each concept links an attacker-facing surface to an unsafe trajectory through a claim, enabling condition, falsifier, transfer prediction, and supporting evidence. Across Claude Code and Codex on three scenarios covering direct and indirect attacks, the discovered concepts reveal a reusable vulnerability core across models and agents. A frozen VCG requires no further search and outperforms the strongest frozen discovery baseline by 14.2 percentage points under the same single-shot protocol, while transferring across scenarios and attack channels. The resulting VCG provides an auditable artifact for production safety teams to inspect vulnerabilities, validate patches, and accumulate reusable safety knowledge. Our code is available at this https URL. 

---
# From World Action Models to Embodied Brains: A Roadmap for Open-World Physical Intelligence 

**Authors**: Yuanzhi Liang, Xufeng Zhan, Haibin Huang, Chi Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.11689)  

**Abstract**: Artificial general intelligence ultimately requires agents that can reason and act in the physical world. Action models, vision-language-action policies, and world models have advanced this goal, while World Action Models (WAMs) are particularly promising because they connect candidate interventions with predicted consequences. However, progress remains fragmented: models use incompatible action spaces and prediction targets, datasets and tasks follow different conventions, and runtime systems expose limited interfaces for reuse and evaluation. We review the evolution toward WAMs and organize these limitations into three coupled gaps: model roles and representations, objectives and standardization, and system composition. Building on this analysis, we propose a co-evolution roadmap for physical intelligence centered on the \emph{embodied brain}, a long-term model target for integrating multimodal context, comparing candidate interventions, and issuing state-transition or capability requests rather than direct actuator commands. WAMs provide promising prototypes for its predictive functions, while a physical harness grounds model outputs through tools, controllers, verification, and trace logging. Shared contracts align heterogeneous models, data, tasks, and embodiments, and closed-loop post-training converts verified interaction into reusable experience. Together, these components define a modular physical-intelligence stack for adaptive and self-improving embodied agents. 

---
# RAGU: A Multi-Step GraphRAG Engine with a Compact Domain-Adapted LLM 

**Authors**: Mikhail Komarov, Ivan Bondarenko, Stanislav Shtuka, Oleg Sedukhin, Roman Shuvalov, Yana Dementyeva, Matvey Solovyov, Nikolay O. Nikitin  

**Link**: [PDF](https://arxiv.org/pdf/2607.11683)  

**Abstract**: Graph retrieval-augmented generation (GraphRAG) enhances large language models with structured knowledge, yet existing systems construct knowledge graphs in a single extraction pass, producing noisy entities and brittle retrieval. RAGU, an open-source modular GraphRAG engine, addresses this by separating extraction from consolidation: entities and relations pass through two-stage typed extraction, DBSCAN-backed deduplication, LLM summarization, and Leiden community detection. A key insight motivates a compact extractor: the skills an in-pipeline LLM needs - comprehension, extraction, reasoning over context - are language skills that grow only weakly with model size, unlike factual world knowledge. Accordingly, we train Meno-Lite-0.1, a 7B model optimized for language skills, which outperforms Qwen2.5-32B on knowledge-graph construction (+12.5% relative harmonic mean) and matches it on English GraphRAG tasks. On GraphRAG-Bench (Medical), RAGU retrieves the most complete context at every factoid level (evidence recall up to 0.84 vs. $\leq$0.76) and overtakes HippoRAG2 on synthesis tasks; on multi-hop factoid QA, the apparent HippoRAG2 advantage is shown to be largely an answer-format artifact. RAGU is installable via $\texttt{pip install graph_ragu}$, runs on a single GPU, and is released under MIT. The source code is publicly available at this https URL, and the Meno-Lite-0.1 model can be obtained from this https URL. 

---
# Closing the Loop: An Access-Control Architecture for Automated, Anomaly-Driven Network Revocation in IoT Deployments 

**Authors**: Muhammet Emir Korkmaz, Kemal Bicakci, Yusuf Uzunay  

**Link**: [PDF](https://arxiv.org/pdf/2607.11649)  

**Abstract**: Network-based anomaly detection for IoT devices has matured to the point of reporting strong detection accuracy, yet most published systems stop at raising an alert and leave the question of automated enforcement to future work or to a programmable data plane that few real networks operate. This paper presents an access-control architecture that closes that loop using only standard, already-deployed protocols. Devices authenticate via IEEE 802.1X with EAP-TLS, and a RADIUS server acts as a continuous policy decision point capable of evicting an active session via a Change-of-Authorization Disconnect-Request and permanently excluding a device through certificate revocation. A central, contextual access policy engine continuously consumes the anomaly detector's output and actuates this response over a narrowly restricted channel to the RADIUS server; the same engine is designed to be extensible to other access types, though this paper evaluates only the network access-control mechanism. This mechanism is driven by an anomaly signal from a one-class detector adapted from a prior MUD/SDN-based design, replacing its per-flow multi-model pipeline with passive traffic capture and a single fused model that combines a cluster-based, a volumetric, and a protocol-signature score. On a single testbed device, the detector reaches an AUC of 0.9964 and detects all 24 evaluated attack scenarios (eight attack types at three intensities) using roughly 43$\times$ less training data than the reference design, and the resulting alerts reliably trigger the automated disconnect-then-revoke response, which we measure to evict a device from the network in 335.8\,ms on average and complete certificate revocation in a further 111.5\,ms. We report this evaluation as a demonstration of the closed-loop architecture rather than of the detector itself, and discuss multi-device generalization as a concrete next step. 

---
# Xiaomi-Robotics-U0: Unified Embodied Synthesis with World Foundation Model 

**Authors**: Xinghang Li, Jun Guo, Qiwei Li, Long Qian, Hang Lai, Yueze Wang, Hongyu Yan, Jiahang Cao, Xi Chen, Jingen Qu, Jiaxi Song, Nan Sun, Hanye Zhao, Futeng Liu, Wanli Peng, Heyun Wang, Yunhong Wang, Caoyu Xia, Jack Zhao, Diyun Xiang, Hangjun Ye, Heng Qu, Huaping Liu, Jason Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.11643)  

**Abstract**: Recent foundation image and video generation models offer strong generalization and controllability, but their direct application to embodied scenarios is limited by requirements for multi-view consistency, geometric coherence, and robot embodiment constraints. Existing methods typically adapt foundation models with limited robot data, often sacrificing visual knowledge acquired during large-scale pre-training. We present Xiaomi-Robotics-U0, a 38-billion-parameter multimodal autoregressive model for unified embodied synthesis. It treats embodied generation as an extension of foundation image and video generation and jointly optimizes text-to-image generation, image editing, embodied scene generation, embodied transfer, and embodied video generation. This unified framework preserves the generalization of the pre-trained world foundation model while adapting it to embodied settings. Xiaomi-Robotics-U0 is the first model to support high-quality multi-view scene generation across multiple robot embodiments and to introduce structured, controllable embodied transfer for fine-grained editing while preserving multi-view consistency and interaction dynamics. It achieves state-of-the-art results on single-step and sequential generation tasks, outperforming GPT-Image-2.0 in human evaluations of embodied scene generation and transfer, ranking first on World Arena for embodied video generation, and improving the out-of-distribution success rate of pi_0.5 from 36.9% to 63.2% on challenging real-world manipulation tasks. These results show that foundation world models can serve both as embodied world models and scalable data engines for embodied intelligence. Code and checkpoints are available at this https URL. 

---
# Extending LLM Context via Associative Recurrent Memory 

**Authors**: Gleb Kuzmin, Ivan Rodkin, Aydar Bulatov, Yuri Kuratov, Lyudmila Rvanova, Mikhail Katkov, Ilia Sochenkov, Misha Tsodyks, Timothy Baldwin, Mikhail Burtsev, Artem Shelmanov  

**Link**: [PDF](https://arxiv.org/pdf/2607.11614)  

**Abstract**: Extending the context length of large language models (LLMs) is critical for many real-world applications, yet standard transformers remain constrained by quadratic compute and linear memory scaling. In this work, we investigate the Associative Recurrent Memory Transformer (ARMT) as a practical approach for enabling long-context processing in LLMs, constant memory scaling, and better efficiency. We make three main contributions. First, we construct two domain-specific long-context datasets designed to evaluate realistic workloads, focusing on narrow-domain fine-tuning scenarios. Second, we propose a comprehensive training recipe for ARMT-based context extension, combining continued pre-training, synthetic long-context data generation, curriculum learning, and selective integration of associative memory into chosen model layers. Third, we present an extensive experimental study demonstrating that ARMT-augmented models: (i) process inputs well beyond their original context limits without degrading performance relative to in-limit baselines; (ii) generalize more effectively to out-of-distribution context lengths; and (iii) need 30% less FLOPs while preserving baseline performance within the original context window. 

---
# DiffEEG: A Self-Supervised Denoising Diffusion Model for Learning EEG Generic Representations 

**Authors**: Abdulkader Helwan, Lina Abou-Abbas, Hussein El Amouri, Belkacem Chikhaoui, Khadidja Henni  

**Link**: [PDF](https://arxiv.org/pdf/2607.11578)  

**Abstract**: Deep learning for EEG-based seizure detection faces critical challenges: severe annotation scarcity and extreme class imbalance, where ictal events comprise less than 10\% of clinical recordings. We present DiffEEG, a 9.6M-parameter self-supervised foundation model that addresses both limitations through denoising diffusion pre-training and reinforcement learning (RL)-based fine-tuning. Pre-trained on 1.3M unlabeled segments from the Temple University Hospital Seizure Corpus (TUHSZ), DiffEEG learns generic neural representations via a 1D U-Net with multi-head self-attention. For downstream adaptation, a reinforced decision layer employs policy gradient optimization to directly maximize F1-score, prioritizing sensitivity to rare seizure events over overall accuracy. Under strict patient-wise evaluation (279 patients, Leave-One-Fold-Out), DiffEEG achieves 61\% accuracy and 59\% F1 for 4-class seizure subtyping, and 81\% accuracy with 85\% weighted F1 for binary detection, maintaining clinically viable seizure recall (59\%) despite extreme imbalance (6.7\% prevalence). Segment-level evaluation establishes an upper bound of 97.6\% accuracy, confirming strong architectural capacity. DiffEEG demonstrates that diffusion-based pre-training combined with metric-aware reinforcement learning enables clinically deployable seizure monitoring with minimal labeled data requirements. 

---
# Structure-Feature Aligned Graph Learning via Alternating Constrained Optimization 

**Authors**: Chengcheng Yan, Qingsong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.11577)  

**Abstract**: We introduce a constrained two-view framework for node prediction that aligns structure-conditioned GNN embeddings with a structure-free feature prior learned by an anchor model. Conventional Graph Neural Networks (GNNs) couple feature transformation and neighborhood aggregation, which renders them vulnerable to topology noise and heterophilous connections. To decouple this dependency, our framework utilizes an independent anchor network to capture intrinsic attribute features via a self-supervised reconstruction objective. Furthermore, we propose a Channel-Split Adaptive Gated GNN (CSAG-GNN) that dynamically routes representations between global spectral smoothing and local spatial discrimination through a node-wise gating mechanism. We propose a stable cyclic alternating optimization strategy to solve the resulting coupled bi-level objective, preventing mutual representation drift during training. Empirical results on both homophilous and heterophilous benchmarks show balanced performance gains and structural robustness over competitive baselines. 

---
# Heuristic Learning for Active Flow Control Using Coding Agents 

**Authors**: Paul Garnier, Jonathan Viquerat, Elie Hachem  

**Link**: [PDF](https://arxiv.org/pdf/2607.11565)  

**Abstract**: Active flow control involves nonlinear dynamics, partial observations, and computationally expensive simulations, making controller design particularly challenging. Deep reinforcement learning (DRL) has emerged as a powerful framework for such problems, but its success typically relies on large numbers of simulator interactions and produces neural-network policies whose decision process often remains difficult to interpret. In this work, we investigate a different paradigm: instead of optimizing neural-network parameters, we use modern coding agents to search directly for explicit executable feedback laws. We introduce a constrained heuristic-learning protocol in which an agent iteratively proposes, evaluates, and revises controller implementations while interacting exclusively through the public benchmark interface. The proposed framework is evaluated on 13 active flow-control benchmarks spanning one, two, and three-dimensional problems and compared against the strongest available DRL baselines under identical simulation budgets. The discovered heuristic controllers match or outperform the best DRL policy in 10 of the 13 environments while remaining compact, interpretable, and directly inspectable. Beyond aggregate performance, the resulting controllers reveal physically meaningful feedback mechanisms, transfer successfully across more challenging configurations, and remain competitive under varying Reynolds and Rayleigh numbers, actuator counts, and observation sparsity. These results suggest that heuristic learning through coding agents constitutes a credible and complementary alternative to conventional reinforcement learning, combining competitive performance with physically interpretable controller representations. Prompts and source code are available at this https URL. 

---
# Technical Report on the CVPR 2026@AdvML Workshop Challenge 

**Authors**: Tianyuan Zhang, Zonglei Jing, Jiangfan Liu, Ligong Zhang, Ke Ma, Chengzhi Sun, Xiaohai Xu, Zhirui Zhang, Qianqian Xu, Qingming Huang, Hanyu Fang, Junhua Liu, Zheng Wang, Xiaoliang Liu, Yuanbo Li, Shuai Gui, Bin Wang, Menghe Zheng, Jing Nie, Hanyang Meng, Zeyang Zhang, Xiang Zhang, Yongxuan Zhu, Rui Ding, Hainan Li, Yongkang Zhang, Zhilei Zhu, Xianglong Kong, Jin Hu, Zonghao Ying, Yisong Xiao, Lei Chen, Haotong Qin, Jiakai Wang, Aishan Liu, Ruikai Li, Julia Karbing, Yinpeng Dong, Zhenfei Yin, Shao Jing, Xia Hu, Jingyi Xu, Juntao Dai, Xinyun Chen, Vishal M. Patel, Xianglong Liu, Dawn Song, Alan Yuille, Philip H. S. Torr, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2607.11560)  

**Abstract**: Vision-language agents (VLAs) are increasingly used to interpret complex driving scenes and support safety-critical reasoning. This report presents the CVPR 2026@AdvML Workshop Challenge on adversarial multimodal attacks against autonomous-driving VLAs. Built on DriveLM-style multi-view visual question answering, the challenge represents each scene with six synchronized camera images and a structured collection of driving-related question-answer pairs. Participants generate adversarial images and suffix-only textual perturbations that induce model responses to deviate from reference answers while preserving image fidelity and limiting textual cost. The competition comprises two phases, with Phase II adding a hidden black-box model to assess transferability. We describe the task design, submission rules, evaluation protocol, and leaderboard results, and then examine five leading submissions for which technical reports were available. Across these reports, several recurring patterns emerge: image-side attacks are favored by the suffix penalty; scene-level, multi-view optimization is more effective than treating views in isolation; QA types and graph structure provide useful priors for allocating attack budget; feature-space objectives can improve black-box transfer; and typographic content embedded in camera images exposes a persistent vulnerability in driving VLAs. These findings provide a practical reference for future robustness evaluation and defense design in multimodal autonomous-driving systems. 

---
# AutoMatBench: An Automatic Optimization Toolkit for the Acceleration of Material Properties Prediction Benchmarking 

**Authors**: Hongxiao Li, Wanling Gao  

**Link**: [PDF](https://arxiv.org/pdf/2607.11526)  

**Abstract**: Material property prediction (MPP) infers key properties from chemical composition and structure, accelerating the discovery and optimization of novel materials. In the realm of MPP, MatBench is a widely accepted benchmarking tool that defines over ten significant problems and provides the paradigm of performance evaluation for AI prediction models. Even though MatBench works well in benchmarking the performances of prediction models on in-distribution (ID) tasks and datasets, it lacks the ability to reflect their performances on out-of-distribution (OOD) material data, resulting failure in new material discovery. By combining the pipelines of MatBench and the existing researches on OOD performance evaluation, this study enables a huge space of benchmarking configurations, comprehensively reflecting the performances, abilities, and disadvantages of various AI prediction models. This work reports that the discrepancy of performances at different configuration values is huge and can be illustrated with prior knowledge and novel insights, therefore consideration of causal effect of configurations on performance results is necessary. In case of the impossibility of enumerative benchmarking at every configuration, this work further proposes AutoMatBench, an automatic toolkit with Bayesian optimization. Experiments with AutoMatBench reports that, within twelve steps of optimization, the similar results with MatBench and former OOD research can be accessed while more than half of the cost are saved. Besides, this tool also yields more essential findings on MPP benchmarking, positively contributing to the cost and efficiency of new material discovery. 

---
# Vinci2: Providing Proactive Assistance in Continuous Egocentric Videos 

**Authors**: Gong Sitong, Tianyu Yan, Caixin Kang, Bo Zheng, Xiang Ruan, Huchuan Lu, Kaipeng Zhang, Yoichi Sato, Yifei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2607.11523)  

**Abstract**: When should an intelligent assistant speak up without being asked? Continuous egocentric video offers rich, evolving context that enables a new form of assistance: one that is proactive rather than merely reactive. Yet existing approaches either wait passively for user queries or treat every detected event as requiring a response, without considering the user's history, current activity, or whether assistance would actually be welcome. We reframe proactive assistance as a context-dependent decision problem: the agent must not only perceive what is happening, but reason over accumulated temporal context to determine when and whether to intervene. To this end, we present Vinci2, a proactive egocentric assistance system that advances the on-device assistant Vinci from reactive response toward proactivity. On the evaluation side, we present EgoServe, the first large-scale benchmark for proactive assistance in continuous egocentric video. EgoServe comprises over 3,000 service instances organized along 4 temporal memory horizons, ranging from immediate safety alerts to long-term habit coaching, across 10 service categories. On the modeling side, we propose EgoMemo, a training-free, memory-augmented agent that maintains three complementary memory representations: multi-scale temporal summaries, a semantic knowledge graph, and visual embedding archives. At each timestep, EgoMemo performs retrieval-augmented reasoning to determine whether assistance is warranted and, if so, produces contextually grounded responses. Experiments demonstrate that EgoMemo establishes strong baselines on EgoServe while remaining competitive on existing egocentric benchmarks. Our benchmark and code are publicly available at \href{this https URL}{Vinci2}. 

---
# Toward Inclusive Avatar Design with Limb Differences Through Artificial Intelligence 

**Authors**: Fernanda Miyuki Yamada, João Paulo Gois, Hiroki Takahashi  

**Link**: [PDF](https://arxiv.org/pdf/2607.11512)  

**Abstract**: As extended reality becomes more popular for social interaction and entertainment, 3D avatars must represent the full diversity of body types. Most 3D avatar systems only support normative bodies and do not accurately depict people with limb differences, amputations, or other morphological variations. This paper reviews emerging technical approaches for inclusive 3D avatar customization for this group and current guidelines that promote respectful and accurate representation. We highlight persistent challenges, including the scarcity of diverse datasets and the limitations in animation for non-normative anatomies. This paper positions artificial intelligence as a promising path to overcoming these limitations and advancing inclusive 3D avatar generation. 

---
# CDFM: Towards a General-Purpose Causal Discovery Foundation Model 

**Authors**: Jie Qiao, Ruichu Cai, Zijian Li, Weilin Chen, Pengfei Hua, Boyan Xu, Zhengming Chen, Zhifeng Hao, Peng Cui  

**Link**: [PDF](https://arxiv.org/pdf/2607.11508)  

**Abstract**: Causal discovery, the process of recovering underlying causal structures from observational data, is a fundamental pursuit across scientific disciplines. Over the past decades, numerous algorithms have been developed to tackle this challenge through workflows tailored to the specific causal mechanisms underlying each type of dataset, demonstrating effectiveness across a wide range of applications. However, as the volume and heterogeneity of real-world data continue to grow, this dataset-specific approach inevitably leads to a fragmented, test-driven paradigm that struggles to scale to the demands of modern scientific discovery. To address this, we formulate the Causal Discovery Foundation Model (CDFM) as a unified, general-purpose framework for zero-shot structural inference. To ensure reliable generalization across unknown domains, we first investigate the theoretical boundaries of causal identifiability, revealing the indispensable role of causal prior mechanisms in this process. Building on these insights, we formulate a principled variational framework that treats unknown causal mechanisms as latent variables and mathematically decomposes the intractable marginal likelihood into distinct, tractable learning modules. The variational decomposition provides a conceptual design principle for the architecture design of CDFM, while comprehensive causal knowledge guides the large-scale synthesis of our pretraining data. By pretraining on a massive, highly diverse space of synthetic structural causal models, CDFM successfully internalizes complex statistical asymmetries. Extensive experiments demonstrate that CDFM consistently outperforms traditional algorithms, driving a paradigm shift toward a general-purpose causal discovery foundation model. 

---
# Proxy Exploration and Reusable Guidance: A Modular LLM Post-Training Paradigm via Proxy-Guided Update Signals 

**Authors**: Daocheng Fu, Rong Wu, Yu Yang, Xuemeng Yang, Jianbiao Mei, Licheng Wen, Pinlong Cai, Yong Liu, Botian Shi, Yu Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2607.11505)  

**Abstract**: Post-training is essential for refining the domain-specific capabilities of large language models (LLMs), yet existing reward optimization and distribution matching methods tightly couple policy exploration with distribution alignment. This coupling forces expensive exploration directly on the policy model and severely hinders the asynchronous generation, reuse, and cross-model transfer of optimization signals. In this paper, we propose Proxy-guided Update Signal Transfer (PUST), a novel post-training framework that fundamentally decouples update-signal exploration from distribution alignment. Instead of utilizing the primary model for costly exploration, PUST employs a lightweight proxy model as an efficient testbed to discover high-reward behaviors. We extract the relative improvement signal between the proxy's initial and optimized states, transferring this directional update to the primary model to guide its policy alignment. This decoupled pipeline, comprising proxy exploration, update-signal extraction, and signal transfer, significantly reduces computational overhead and enables optimization signals to be asynchronously generated, cached, and reused. Crucially, by transferring relative improvements rather than absolute policy distributions, PUST naturally supports weak-to-strong improvement and seamless cross-model transfer. Systematic evaluations on Qwen3-family models across math and code domains demonstrate that update signals extracted from substantially weaker proxies can robustly and adjustably enhance stronger primary models. Ultimately, PUST transforms post-training from a monolithic online optimization process into a highly modular, reusable, and cost-efficient paradigm. 

---
# See like a Robot: Robot-Centric Pointmaps for Vision-Language-Action Models 

**Authors**: Byungkun Lee, Dongyoon Hwang, Dongjin Kim, Hojoon Lee, Minho Park, Jaegul Choo  

**Link**: [PDF](https://arxiv.org/pdf/2607.11498)  

**Abstract**: Vision-language-action (VLA) models predict robot actions from visual observations and language instructions. These actions are defined in the robot's own 3D coordinate frame, yet most VLAs observe the scene in the camera frame, creating a frame mismatch between where the scene is observed and where actions are defined. The mismatch is benign under a fixed viewpoint, where the policy can memorize a single observation-to-action mapping, but grows harder as large-scale datasets aggregate demonstrations across diverse camera setups and the policy must generalize this mapping across viewpoints. We address this mismatch with robot-centric pointmaps, images whose pixels store the 3D coordinates of scene points in the robot frame. Pointmaps provide robot-frame 3D geometry while preserving the dense H x W grid expected by pretrained 2D VLAs, so they integrate into existing VLAs with minimal architectural change. On RoboCasa, pointmaps improve both pi0.5 and SmolVLA and outperform representative camera-viewpoint and 3D-aware baselines. In real-robot experiments, their advantage over an RGB-only policy widens when the camera is moved to a placement unseen during training. 

---
# IG-GAN: A Generative Adversarial Network for Aerodynamic Data Generation Based on Intrinsic Geometry 

**Authors**: Ying Yan, Liwei Hu, Xiaoming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.11497)  

**Abstract**: Existing generative models learn data distributions in flat Euclidean space. However, most data in our real world are manifolds embedded in high dimensional Euclidean space. Therefore, we propose an intrinsic-geometry-based generative adversarial network (IG-GAN) for data generation in the field of aerodynamics. The generator of the IG-GAN represents aerodynamic data as a piecewise smooth manifold constructed by Bézier surfaces, and the generator tries to learn the coefficients of each Bézier surface to further combine multiple Bézier surfaces into a smooth manifold automatically. The discriminator in the IG-GAN is a radial-basis-function based discriminator (RBF-D). Experimental results show that IG-GAN achieves lower predicted Mean Squared Errors (MSEs) than those of three baselines. Specifically, on the Burgers' equation dataset, IG-GAN reduces the predicted MSE of velocity u by 97.41% compared with state of the art SSL-Transformer. Additionally, on the ONERA M6 aircraft dataset, IG-GAN reduces the overall MSE of nine aerodynamic coefficients by 82.95% compared with SSL-Transformer. 

---
# Agentic Skill Optimization over Lie Algebroids 

**Authors**: Sridhar Mahadevan  

**Link**: [PDF](https://arxiv.org/pdf/2607.11493)  

**Abstract**: Agentic systems increasingly improve themselves by editing skills: prompts, rubrics, plans, tool contracts, examples, validators, and traces. Skill edits are not independent coordinates in a vector space: they are local repairs to structured artifacts whose effects are observed only after rollout, validation, and critique. Distinct edits can have the same immediate visible effect while differing in routing context, template state, guardrail scope, or future composability. The order of edits can matter as well: repairing a schema before a normalization rule need not be equivalent to applying the same edits in the reverse order. This paper introduces a new framework for skill optimization called LASKO, for Lie Algebroid SKill Optimization. LASKO models typed, anchored Markdown skills as the base category and available edit policies as sections of a controlled Lie algebroid with anchor $\rho$. The anchor maps an edit policy to its visible Markdown effect; the kernel $\ker(\rho)$ represents latent template, routing, or implementation structure; and the algebroid bracket measures noncommuting edit composition. As shown in the paper, LASKO achieves order-of-magnitude speedups in skill optimization in our preliminary benchmark results, primarily because it substitutes inexpensive Lie-bracket screening tests that run in microseconds, before investing in expensive validations that require running large language models. On a causal extraction from natural language task, LASKO achieved a speedup of almost $15 \times$ compared to a brute-force approach that validated all edits by running them through a DeepSeek V3.1 4-bit model with 671B parameters. 

---
# LightMem-Ego: Your AI Memory for Everyday Life 

**Authors**: Yijun Chen, Boyi Xiao, Yixian Zhao, Haoting Xia, Buqiang Xu, Jizhan Fang, Yanya Li, Yaqi Zheng, Xuehai Wang, Zirui Xue, Liuxin Zhang, Hui Li, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.11487)  

**Abstract**: Personal AI assistants on mobile and wearable devices continuously perceive users' daily lives through visual and audio streams. However, answering queries about past experiences requires lightweight multimodal memory that can continuously accumulate, organize, and retrieve long-term experiences, which remains challenging. To address this challenge, we present LightMem-Ego, a lightweight streaming multimodal memory system for everyday-life assistance. The system continuously captures egocentric visual and audio streams, aligns them on a shared timeline, and organizes them into a hierarchical memory consisting of current, short-term, and long-term memory. Given a user query, LightMem-Ego dynamically routes retrieval to the appropriate memory level and generates answers grounded in multimodal evidence. The demonstration can be deployed on smartphones and AI glasses, supporting object finding, conversation recall, life summarization, routine discovery, and personalized assistance. Code is available at this https URL. 

---
# A Multimodal Dataset for Large Language Model Applications in the Energy Domain 

**Authors**: Costas Mylonas, Magda Foti  

**Link**: [PDF](https://arxiv.org/pdf/2607.11459)  

**Abstract**: This paper presents the mAIEnergy dataset, an open-access, multimodal corpus developed to support Large Language Model (LLM) applications in the energy sector. The dataset integrates approximately 50,000 textual documents, 20,000 images, 25 million numerical time series records, and 2 million geospatial and relational data entries. It includes policy and regulatory texts, scientific articles and news articles, satellite and contextual imagery, electricity system measurements, weather observations, statistical indicators, and geospatial representations of energy infrastructure and related entities. All data have been harmonized into structured, ready-to-use formats, accompanied by consistent metadata and reproducible data retrieval and preparation workflows. The dataset can serve as a foundational energy knowledge base, allowing energy stakeholders to integrate additional open-source or proprietary data. The mAIEnergy dataset adheres to Findable, Accessible, Interoperable, and Reusable (FAIR) principles, enhancing its applicability for AI-driven energy research, modeling, and decision-making. 

---
# Uncertainty Quantification for EO Regression Tasks: Building Height, Tree Canopy Height and Above-ground Biomass Estimation 

**Authors**: Ritu Yadav, Andrea Nascetti, Yifang Ban  

**Link**: [PDF](https://arxiv.org/pdf/2607.11412)  

**Abstract**: Earth Observation regression tasks such as building height, canopy height, and above-ground biomass estimation underpin critical applications in urban planning, forest monitoring, and climate policy, where both accuracy and reliability are critical. Yet most deep learning models yield only deterministic predictions, providing no indication of per-pixel reliability. These regression tasks are inherently challenging due to heterogeneous land surfaces, skewed target distributions, sensor noise, and signal saturation at high target values, making uncertainty (UC) estimation essential for reliable inference. We address this gap by modeling aleatoric uncertainty using year-long Sentinel-1 SAR and Sentinel-2 MSI time series, proposing two complementary approaches: (i) Gaussian UC, which jointly predicts mean and standard deviation under a Gaussian assumption, and (ii) Quantile UC, which estimates the 10th, 50th, and 90th quantiles to capture asymmetric and heteroscedastic error distributions. Both models are evaluated on three representative EO regression tasks at 10 m spatial resolution. Results show that both approaches match or surpass deterministic benchmarks and existing global products, while delivering well-calibrated, interpretable, and operationally useful confidence estimates. Notably, both models outperform the current 10 m state-of-the-art uncertainty-aware model for canopy height estimation. Our implementation will be available at: this https URL 

---
# Agentic Routing: The Harness-Native Data Flywheel 

**Authors**: Xinchen Liu, Hang Zhou, Yingjie Zong, Yuchuan Tian, Liuyang Song, Shuo Zhang, Yulong Li, Wei He, Mengyu Zheng, Runke Liu, Siyang Cheng, Xiang Kuang, Hailin Hu, Kai Han, Yunhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.11399)  

**Abstract**: Large language model agents are increasingly executed not by a single model call, but by an execution harness that manages observation, context, control, action, state, and verification. At the same time, frontier and open models are becoming structurally specialized: a model that is strong at code editing, long-context recovery, tool use, mathematical reasoning, or low-latency response may not dominate on the other axes. This makes model selection inside an agent a core systems problem rather than a per-query serving trick. Existing routing methods mostly optimize single-turn cost-quality trade-offs and therefore miss the execution state, intermediate failures, and feedback loops that make agents different from chat completion. We propose Harness-Native agentic routing, a step-level routing paradigm that selects either a single best-fit model for cost-effective execution or multiple complementary models for ensemble-style accuracy improvement, conditioned on the full harness state. The key insight is that every routing decision naturally produces a structured data record -- consisting of the query, harness state, model choice or model set, execution trace, outcome, and cost -- whose labels are supplied by the environment rather than by the router itself. These records form a harness-native data flywheel: execution traces train better routers and harness-native models, which improve cost-quality trade-offs and generate more traces under the same budget. We instantiate this idea in OpenSquilla with a four-layer routing stack, an open LightGBM cold-start ranker, and a staged router-model path that turns logged arena records into progressively stronger routing policies. The report studies singleton and multi-model routing on agentic benchmarks including DRACO and PinchBench, and argues that agentic routing is not merely cost control, but a data engine for agent-native training. 

---
# A Glimpse into Long-term Physical Coexistence with Intelligent Robots 

**Authors**: Weiqi Jin, Peijun Tang, Kuncheng Luo, Baifu Huang, Binyan Sun, Haotian Yang, Shangjin Xie, Jianan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.11377)  

**Abstract**: Long-term physical coexistence with intelligent robots requires more than capable robot policies. A persistent robotic assistant must support diverse user-facing interfaces, maintain long-horizon memory of people and preferences, coordinate across robot embodiments, and translate human intent into safe physical execution. We introduce PHILIA, a multi-robot agent built around a robot gateway abstraction. PHILIA retains the rich interaction and tool ecosystem of OpenClaw while exposing robot-local runtimes, onboard perception, navigation, speaker, and robot policies through a unified capability interface. This design decouples low-frequency, high-semantic agent reasoning from high-frequency, low-level robot execution, enabling plug-and-play integration of user interfaces, robot embodiments, and policy backends. As a result, the user experience becomes compositional: advances in user interfaces, robot embodiments, robot policies, navigation, or interaction algorithms can improve the overall experience without redesigning the system. We validate the architecture on Astribot S1 robots while designing the robot gateway contract to support future heterogeneous robot platforms through a shared capability interface for observation, task execution, navigation, speech playback, status monitoring, and task cancellation. We present representative use cases in which agent memory and scene understanding are grounded in robot actions. These span interactive household scenarios, ranging from simple organization to challenging long-horizon and dexterous service tasks, such as packing a backpack and lifting a garbage bag. We highlight the human-robot interaction flow, where contextual understanding of user intent and preferences, together with human-in-the-loop confirmation or adjustment during execution, is essential for effective assistance. 

---
# BackgroundMellow: A Multi-Modal Cohesive Framework for Narrative-Driven Rich Cinematic Soundscape Generation 

**Authors**: Ajitesh Jamulkar, Aritra Hazra  

**Link**: [PDF](https://arxiv.org/pdf/2607.11364)  

**Abstract**: Generating immersive, synchronized and cinematic audio for long-form textual narratives remains a significant challenge in multi-modal AI. While current Text-to-Audio (TTA) frameworks successfully synthesize isolated sound effects, they struggle with narrative cohesion, temporal alignment, and cinematic emotional depth. We present BackgroundMellow, a framework that treats story-to-audio generation as a precise orchestration and signal processing problem. This framework is enabled without ground-truth through a master-specialist agent architecture that decomposes text into precise and multi-layered audio cues, generates each category of sounds with suitable specialist model, and superimposes the soundscapes to create a unified and aligned audio segment. Our pipeline is built over Tango2 latent diffusion model for environmental synthesis alongside a novel Cinematic BGM Retriever mined from professional soundtracks. To automate the sound mixing process, we use an NLP based module that predicts precise audio parameters, like start time, duration, and relative loudness, based on the narrative timeline. We further empirically evaluate and show the efficacy of the proposed framework leveraging nearest-neighbor retrieval against a curated dataset of YouTube cinematic trailers to measure temporal synchronization, coverage, and spectral richness. 

---
# Beyond Sally-Anne: Evaluating Theory of Mind in LLMs using Epistemic Schelling Points 

**Authors**: Roberta Rocca, Sami Boukortt, Geoff Keeling, Winnie Street  

**Link**: [PDF](https://arxiv.org/pdf/2607.11363)  

**Abstract**: Text-based evaluations of Theory of Mind (ToM) in Large Language Models (LLMs) often involve cognitive tests akin to the Sally-Anne task that can be gamed due to exposure to relevantly similar tasks in pre-training and do not obviously test models' functional ToM abilities in ways that generalize to naturalistic settings. To address these issues, we introduce the Epistemic Asymmetry Schelling Task (EAST), a two-player dialogue game designed to benchmark robust and generalizable ToM abilities. By requiring LLM-LLM dyads to independently converge on semantic Schelling points under varying states of epistemic transparency, we evaluate whether models can robustly apply ToM to achieve coordination. Our results reveal a significant capability gap in functional social reasoning, with only frontier models successfully navigating the varying epistemic demands of the tasks. Analysis of reasoning traces shows that coordination failures are primarily driven by epistemic tracking errors, such as conflating private knowledge with mutual knowledge. Despite high performance on traditional static benchmarks, our study shows that robust social reasoning and epistemic tracking remain a critical bottleneck, providing concrete targets for future LLM evaluation and development. 

---
# Characterising AI Models for Cataloguing 

**Authors**: Miguel Arana-Catania, Neil Jefferies  

**Link**: [PDF](https://arxiv.org/pdf/2607.11353)  

**Abstract**: The creation of digital collections involves not only the digitisation of content, but also the creation of catalogue records for it. This often-overlooked task requires slow and costly expert manual work. In this project, we have evaluated the application of AI models to this task, comparing different implementations and models. This work includes a qualitative and quantitative evaluation of the experiments carried out, as well as recommendations on the use of AI models that go beyond the specific use case. 

---
# Understanding the Impact of AI Code Assistants on Security API Usage: An Empirical Study 

**Authors**: Zahra Mousavi, Chadni Islam, M. Ali Babar, Alsharif Abuadbba, Kristen Moore  

**Link**: [PDF](https://arxiv.org/pdf/2607.11348)  

**Abstract**: AI code assistants are transforming software development, but their implications for software security remain a major concern, particularly in the context of security APIs. These APIs are critical for safeguarding software systems, yet their complexity often leads to incorrect use and serious vulnerabilities. Developing an evidence-based understanding of how AI assistants influence developers' use of these APIs is therefore essential for informing effective mitigation strategies. While a few user studies have examined the broader impact of AI assistants on software vulnerabilities, the use of security APIs remains unexplored from a developer-centered perspective. This study addresses this gap by presenting the first empirical investigation into how AI code assistants affect professional developers' use of security APIs. We conducted a study with 44 developers who completed security API programming tasks with and without GitHub Copilot assistance. Our findings show that, while Copilot improves functional correctness and marginally reduces certain insecure patterns, it does not significantly improve secure API usage. We also found that developers rarely raised security concerns when engaging with Copilot, and many did not recognize that their final implementations remained insecure. Finally, we offer recommendations for enhancing security awareness among developers and propose future research directions to support safer AI-assisted software development. 

---
# Longitudinal Multi-View Breast Cancer Risk Prediction 

**Authors**: Solveig Thrun, Zijun Sun, Suaiba A. Salahuddin, Kristoffer Wickstrøm, Elisabeth Wetzer, Stine Hansen, Robert Jenssen, Michael Kampffmeyer  

**Link**: [PDF](https://arxiv.org/pdf/2607.11343)  

**Abstract**: Accurate breast cancer risk prediction from screening mammography is critical for enabling personalized screening intervals and early detection. Recent deep learning methods have shown the value of longitudinal data and explicit temporal alignment. However, existing approaches either perform explicit alignment using a single mammographic view or model multiple views without explicit longitudinal alignment, limiting their ability to exploit the complementary spatial-temporal information used in clinical practice. To address this gap, we propose LMV-Net, a longitudinal multi-view breast cancer risk prediction model that jointly analyzes anatomically complementary CC and MLO views within an explicitly aligned longitudinal framework. We evaluate our approach on the public EMBED and CSAW-CC datasets, comparing it to state-of-the-art breast cancer risk prediction methods. Our model consistently outperforms existing approaches in overall risk prediction performance and across different breast density and cancer subgroups. Importantly, these improvements highlight the potential of longitudinal multi-view modeling to enhance risk stratification, paving the way for future work on personalized screening, earlier identification of high-risk patients, and more efficient screening resource allocation. The code is available at this https URL. 

---
# Fail-Aware and Explainable Test Oracle Prediction 

**Authors**: Yue Zhao, Binish Tanveer, Jelena Zdravkovic  

**Link**: [PDF](https://arxiv.org/pdf/2607.11342)  

**Abstract**: Despite their central role in fault detection, test oracles remain challenging to construct effectively. Recent learning based methods address this challenge by automatically generating test assertions, yet even if syntactically correct, they are often ineffective in revealing bugs. Rather than generating assertions, this study explores a different approach by training a model to directly predict whether a given test prefix passes or fails. We present FOCAL, an emerging code LLM-based discriminative oracle predictor. It learns from labeled pairs of test prefixes and methods under test, employs losses that emphasize failing cases during training, and grounds its predictions in statement level behavioral evidence. Compared with the baseline method SEER, we substantially improve performance on failing cases for unseen projects and provide richer explanations. A preliminary evaluation on fault-detection benchmarks and automated test-generation artifacts shows that our approach is highly accurate within its training distribution and substantially improves failure detection on previously unseen projects where prior discriminative oracles collapse. Moreover, the highlighted statements are supported by behavioral explanation checks. These early results suggest that fail-aware discriminative oracle prediction can complement existing approaches such as fuzzing, search-based testing, and LLM-based test generation. These techniques produce test prefixes at scale but often lack fault oriented oracles. In future work, FOCAL could take generated test prefixes and attach fault-aware predicted oracles to them, turning high-volume input generation into executable tests that are more likely to expose semantic failures. 

---
# PRISM Edit: One Vector for All Temporal Answers 

**Authors**: Chen Huang, Qi Zheng, Ruiqin Zheng, Long Zeng, Yuantong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.11327)  

**Abstract**: Model editing keeps large language models (LLMs) up to date without retraining, but temporal facts expose a limitation of the prevailing locate-and-edit paradigm: an update is not always a replacement. When a fact changes, the new answer should become current while the old answer may remain correct in historical time contexts. Building on this insight, we use causal tracing to show that LLMs already support this distinction via a two-stage internal computation: early MLP layers retrieve a time-agnostic subject representation, and later layers modulate it with temporal context to yield the time-correct answer. Motivated by this finding, we introduce PRISM Edit, which optimizes a single polysemous representation across temporal contexts and leverages the model's inherent modulation pathway to route it to temporally correct predictions, without any architectural modification. We evaluate on TimeConflict, a new temporal editing benchmark we introduce, and on temporally augmented CounterFact. PRISM Edit improves over the best baseline by +23.3 Temporal Consistency (TC) and +33.7 Current Relative-time Score (CRS) on average while being more than 2x faster. Code and data are publicly available at this https URL. 

---
# Programming Language Policy as an AI Literacy Equity Problem: A 15-Nation Comparative Analysis 

**Authors**: Adrian-Marius Dumitran, Iulia-Maria Popescu  

**Link**: [PDF](https://arxiv.org/pdf/2607.11314)  

**Abstract**: The promise of AI literacy ``for all'' confronts a structural challenge embedded in how nations organise secondary computer science education. In most systems, a general-track subject -- Digital Literacy, ICT, TIC, or SNT -- bears the weight of universal AI literacy, while a specialist Informatics course serves STEM pathways separately. Yet the content and depth of the general track are shaped by governance decisions made largely with reference to the specialist one. This paper presents a comparative analysis of curricula and examination frameworks across fifteen countries, identifying two structural challenges. First, in several systems a significant portion of students completes secondary education without any formal programming exposure. Second, among those who do receive CS education, a \emph{Syntax Ceiling} emerges: Python-based instruction reaches most students, while the algorithmic depth associated with C++ remains concentrated in elite STEM tracks. Drawing on reform cases spanning centralised mandates (France, China, Japan), assessment-driven systems (Poland, Romania, South Korea), and recent universal reforms (Switzerland, Kazakhstan), we show that governance structures and high-stakes examinations are the primary drivers of both challenges -- and that specialist and general-track language choices are rarely independent, linked through shared teacher pipelines that curriculum policy seldom acknowledges. Achieving genuine AI literacy for all requires confronting not just curriculum content, but the access architectures and resource constraints that determine who receives it -- and at what depth. 

---
# The Paternalistic Filter: Epistemic Injustice and Differential Refusal in LLM-Mediated History Education for Marginalized Romanian Students 

**Authors**: Alexis Popovici, Andrei Ionascu, Adrian-Marius Dumitran  

**Link**: [PDF](https://arxiv.org/pdf/2607.11292)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed as conversational tutors, they risk institutionalizing systemic inequalities. This study presents a systematic API audit of four LLMs acting as history tutors, evaluating 1,800 responses regarding the 1989 Romanian Revolution across five student personas varying by ethnicity and socio-economic tier. We uncover four interconnected patterns of \emph{epistemic paternalism}: (1)~\textbf{Differential Refusal}, where safety-aligned models block 76.7\% of educational requests from low-tier students; (2)~\textbf{Epistemic Gatekeeping}, evidenced by a 3$\times$ reduction in access to geopolitical complexity (e.g., the contested ``coup theory'') for marginalized learners; (3)~\textbf{Agency Theft}, a lexical shift where models like LLaMA produce a 5$\times$ higher victimization-to-politics vocabulary ratio for Roma students compared to elite peers; and (4)~\textbf{Elite Hermeneutics}, where AI tutors disproportionately withhold epistemic confidence and justification scores from low-resource demographic profiles. We argue that current safety alignment acts as a paternalistic filter, transforming conversational AI into agents of narrative segregation -- a manifestation of \emph{hermeneutical injustice} in Fricker's~\cite{fricker2007} sense that demands urgent pedagogical auditing. 

---
# Mako: A Self-Evolving Agentic Operating System (SE-AOS) for Autonomous Web Exploitation 

**Authors**: Praneeth Narisetty, Shiva Nagendra Babu Kore  

**Link**: [PDF](https://arxiv.org/pdf/2607.11288)  

**Abstract**: We introduce the Self-Evolving Agentic Operating System (SE-AOS): a new class of AI agent that treats exploit capability as a mutable, versioned kernel it extends at runtime, observing its own failures, synthesising new capabilities, proving them against a live target, and hot-loading them back into itself. Mako is the first SE-AOS instance for security research and the autonomous web exploitation engine developed within LaunchSafe. LaunchSafe builds autonomous security agents for continuous offensive testing and agent-driven security research; Mako is the core engine behind that platform. On the public XBOW validation-benchmarks, 104 containerised, CTF-style web applications spanning 26 vulnerability classes across three difficulty tiers, Mako achieves full-suite coverage: it drives every one of the 104 targets to emit a cryptographically fresh, per-build flag, under a verification regime that makes fabricated or memorised results impossible. Our central result is a law of autonomous exploitation: once a capability exists and is discoverable, difficulty collapses; capability, not reasoning, is what is scarce, together with an architecture and formalism that turn that law into a self-improving system. Mako further runs a gated self-evolution loop that proposes, sandboxes, and commits improvements to its own agents and rules when fitness does not regress. We deliberately withhold the operational results, payloads, exploit chains, and tool source, because a system that reduces full-spectrum web exploitation to a repeatable, machine-speed pipeline is dual-use research of concern. We publish the science; we withhold the weapon. 

---
# A Unified Framework for Comprehensive Cardiac CT Segmentation and Phenotyping: Human-in-the-Loop Data Annotation, Vision Foundation Model Development, Multicenter Evaluation and Clinical Validation 

**Authors**: Pooya Mohammadi Kazaj, Leo Fridolin Weber, Wen Xie, Seyed Amir Ahmad Safavi-Naini, Anselm Stark, Giovanni Baj, Ali Mokhtari, Toshiya Yoshida, Christoph Ryffel, Taishi Okuno, Yoshihiro Akashi, Ronny R. Buechel, Thomas Pilgrim, Waldo Valenzuela, George C. M. Siontis, Xiaowei Xu, Moritz Hundertmark, Stephan Windecker, Christoph Grani, Isaac Shiri  

**Link**: [PDF](https://arxiv.org/pdf/2607.11287)  

**Abstract**: Comprehensive quantification of cardiac structures from computed tomography (CT) remains limited not by data availability but by the scalability of measurements, which makes routine use impractical. Here we present a unified framework for comprehensive cardiac CT segmentation and phenotyping that combines a human-in-the-loop annotation pipeline, a cardiac CT augmentation technique, and a self-supervised foundation model pre-trained on 60,000 unlabeled cardiac CT scans. Using this approach, we assembled the largest and most comprehensive expert-annotated cardiac CT segmentation dataset to date, comprising 1598 cases and 14 distinct cardiac structures (1000 for training, 598 for the external test set). Across five external datasets, the framework segmented all structures more accurately and comprehensively than existing open-source tools. Self-supervised pre-training improved labeling efficiency, with the most significant gains observed during external evaluation in the low-data regime. Benchmarking across convolutional, transformer, and state-space architectures showed comparable performance, indicating that data quality and pre-training, rather than architecture, drove accuracy. The framework was scaled to population-level phenotyping, with segmented anatomy that carries functionally relevant information about ventricular function and disease severity beyond demographic variables. By openly releasing the largest dataset with human labels, code, model weights, a CT augmentation library, and software, this work provides a reproducible foundation for opportunistic cardiac phenotyping from routinely acquired CT scans. 

---
# Automated Textbook Auditing with Multi-Agent LLM Systems 

**Authors**: Ciprian Cristescu, Adrian-Marius Dumitran, Angela-Liliana Dumitran, Gabriel Stefan  

**Link**: [PDF](https://arxiv.org/pdf/2607.11276)  

**Abstract**: Ensuring the quality of educational materials requires more than standard proofreading: textbooks must be audited for factual accuracy, domain-specific technical correctness, and linguistic quality simultaneously -- a task that general-purpose grammar checkers cannot address. We present \textbf{AI Textbook Auditor}, a modular multi-agent pipeline for automated quality assurance of educational materials across subject domains. The system accepts a textbook PDF and produces a structured, human-reviewable report via two analysis tracks: a \textbf{Factual and Technical Track} in which an ensemble of specialized LLM agents detects factual inaccuracies, code errors, incorrect definitions, and conceptual inconsistencies, augmented with web search for humanities domains; and a \textbf{Grammar Track} operating PDF-natively to preserve diacritical encoding. A \textbf{Judge Agent} filters false positives using domain-specific rules before presenting findings to a human reviewer. The pipeline supports two ingestion modes -- vision-native page rendering and PyMuPDF text extraction -- and is domain-adaptable via custom prompts encoding subject-specific error taxonomies. We demonstrate the system on two Romanian upper-secondary textbooks: a CS textbook (56 technical findings across seven categories, with an expert-validated precision of 62.5\%) and a history and social sciences textbook (72 findings spanning factual errors, ideological bias, and grammar). The system is designed as a triage tool that reduces the manual effort of locating candidate issues, with human expert validation required before any editorial action. 

---
# Towards Predictive, Aligned, and Scalable Robot Learning 

**Authors**: Peijun Tang, Shangjin Xie, Baifu Huang, Binyan Sun, Haotian Yang, Kuncheng Luo, Weiqi Jin, Shilin Fang, Jianan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.11270)  

**Abstract**: Learning, at its core, extends beyond memorization to the ability to reason and solve novel problems by navigating a space of possibilities. We introduce Lumo-2, a latent world-action model that generates actions by reasoning over world dynamics in latent space. The learned latent world dynamics capture physically grounded visual transitions, naturally encoding future possibilities and providing a unified substrate for cross-modal alignment. This formulation enables predictive reasoning akin to world modelling while remaining lightweight and focused on physical dynamics relevant to control. Central to our approach is the hypothesis that action generation quality is governed by the geometry of the latent space. We observe that standard reconstruction-based action tokenization objectives induce representations biased toward low-level signal fidelity, leading to misalignment between reconstruction quality and downstream control performance. To address this limitation, we propose a multi-stage modality pre-alignment strategy in which action representations are progressively aligned with latent world dynamics, vision, and language. This process enforces cross-modal consistency, promotes abstraction, and induces a structured latent space for predictive reasoning. We provide a systematic empirical study of latent world modelling and modality alignment, analyzing their roles in scaling laws and out-of-distribution generalization. Results show that Lumo-2 consistently outperforms strong vision-language-action (VLA) and world-action model (WAM) baselines, with gains on challenging real-world tasks requiring temporal reasoning, physical understanding, or high control complexity, including long-horizon and dexterous manipulation. These findings suggest that structured multimodal alignment and predictive reasoning are fundamental principles for advancing embodied intelligence. 

---
# Enhancing LLMs through human feedback: a journey towards self-improvement 

**Authors**: Tatiana Pelc, Gila Kamhi, Asaf Avrahamy, Adi Fledel-Alon  

**Link**: [PDF](https://arxiv.org/pdf/2607.11267)  

**Abstract**: In the rapidly evolving landscape of information retrieval systems, the ability to adapt and improve through user feedback is paramount. This study introduces a novel methodology for refining the performance of a primary Retrieval Augmented Generation (RAG) system by strategically integrating an auxiliary feedback RAG system. By systematically harnessing human-generated feedback, the approach aims to enhance the accuracy, relevance, and overall quality of responses, driving the system towards self-improvement. Central to this methodology is a human-in-the-loop implementation, where user feedback is continuously collected, classified, and integrated into the inference workflow, enabling the system to learn and evolve iteratively. To validate the effectiveness of this approach, the study employs rigorous testing against three diverse benchmark datasets focused on general and custom domain knowledge, utilizing a LLM-as-a-Judge evaluation strategy. This comprehensive framework not only underscores the transformative potential of feedback-driven enhancements in RAG systems but also sets a precedent for future research in adaptive information retrieval technologies, marking a significant step in the journey towards autonomous refinement and optimization through user engagement. 

---
# Multi-Agent LLMs Fail to Explore Each Other 

**Authors**: Hyeong Kyu Choi, Jiatong Li, Wendi Li, Xin Eric Wang, Sharon Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.11250)  

**Abstract**: Exploration is essential for reliable autonomy in multi-agent systems, yet it remains unclear whether large language model (LLM) agents can explore effectively when interacting with one another. We show that modern LLM agents fail to do so, often exhibiting myopic and polarized interaction patterns that lead to suboptimal coordination and increased regret. We formalize this challenge as the Multi-Agent Exploration problem, modeling it as a partially observable stochastic game (POSG) problem in which agents must probe peers to infer their capabilities and identify effective interaction strategies. To address this, we introduce Multi- Agent Contextual Exploration (MACE), a lightweight framework that explicitly promotes exploration through structured peer selection. Across both contextual and parametric diversity settings, MACE substantially improves exploration behavior and downstream task performance. We further show theoretically that the value of exploration increases with agent diversity. Overall, our results highlight a fundamental limitation of current LLM agents and underscore the importance of explicitly guided exploration for reliable multi-agent autonomy. Code will be released in this https URL 

---
# An Empirical Study for GUI Test Migration from Android to OpenHarmony System 

**Authors**: Yakun Zhang, Xinjia Chen, Yiyun Chen, Yuxia Zhang, Mingyi Zhou, Xiang Gao, Shaokun Zhang, Li Li, Yunming Ye  

**Link**: [PDF](https://arxiv.org/pdf/2607.11245)  

**Abstract**: To reduce the substantial engineering effort required to test the corresponding applications from Android to OpenHarmony, migrating existing GUI test cases has become a critical problem. However, current research neither proposes solutions tailored for OpenHarmony nor provides a systematic evaluation of migration approaches on this system, leaving developers with limited empirical guidance in practice. In this paper, we present the first systematic empirical study of test migration from Android to OpenHarmony. Specifically, we first construct a dataset referred to as the ATH Benchmark, comprising 36 commercial applications with an average of over 9 billion downloads, along with 108 manually designed test cases. Second, we select two state-of-the-art test migration approaches (i.e., ReSPlay and ITeM) and adapt these two approaches to enable their execution on OpenHarmony. Third, we use the preceding infrastructure to evaluate these two approaches from three perspectives, including testing performance, root causes of failures, and the impact of OpenHarmony characteristics. Our results reveal that existing test migration approaches are less effective (15% success-rate on ReSPlay and 26% success-rate on ITeM) in Android-to-OpenHarmony scenarios. Through an in-depth analysis of failed cases, we identify that test performance is primarily hindered by OpenHarmony-specific characteristics, including technical architecture differences and unique ecosystem traits. Utilizing these findings, we propose an enhanced approach based on ITeM, referred as ITeM-HM, which incorporates specific OpenHarmony system features. As a result, ITeM-HM successfully achieves a 214% success-rate relative improvement over the original ITeM (from 26% to 81%). 

---
# DeepBias: Adaptive In-depth Probing of Social Biases in LVLMs 

**Authors**: Anqi Li, Jie Zhang, Zhongqi Wang, Songkai Xue, Jiahao Wang, Shiguang Shan, Xilin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.11228)  

**Abstract**: While Large Vision-Language Models (LVLMs) demonstrate remarkable capabilities, they remain highly susceptible to embedded social biases. Existing bias evaluation protocols predominantly rely on static datasets, which provide only a superficial assessment, as their fixed test cases cannot adaptively evolve to measure the true depth and limits of model vulnerabilities. We introduce DeepBias, an adaptive framework for the in-depth probing of social biases in LVLMs with carefully designed agents. Our approach operates through a dynamic ''generation-evolution-probing'' loop. First, a generative ProposerAgent synthesizes test data and is iteratively updated via Direct Preference Optimization (DPO) based on the target LVLM's responses, exploring model-specific failure modes. Second, an autonomous skill-driven DiggerAgent rewrites each test data across multiple probing turns, adaptively selecting from a curated skill library of deepening and rewriting strategies. At each turn, this process is conditioned on the model's previous response, enabling progressively deeper biases to be exposed. Furthermore, we build a benchmark named DeepBiasBench using our framework. By employing an ensemble of five diverse state-of-the-art LVLMs as anchors, the benchmark captures vulnerabilities shared across architectures. Comprehensive experiments demonstrate the effectiveness of our framework and show that DeepBias provides a challenging benchmark for in-depth bias evaluation, establishing an evolutionary paradigm for LVLM safety assessment. 

---
# HandFlow: Fully Generative 4D Hand Recovery with Flow Matching 

**Authors**: Mingxi Xu, Bowen Duan, Yi Gu, Zhengyang Shen, Renjing Xu, Yutao Yue  

**Link**: [PDF](https://arxiv.org/pdf/2607.11221)  

**Abstract**: Accurate monocular 4D hand reconstruction remains challenging. Per-frame discriminative regressors lack temporal context and often produce jittery predictions. Temporal models improve consistency by aggregating information across frames, but they are typically deterministic regressors, making them vulnerable to ambiguous observations caused by occlusion and motion blur. Generative modeling offers a natural alternative by learning a prior over plausible hand motion sequences, enabling coherent hand-state recovery when visual evidence is incomplete or unreliable.
Motivated by this observation, we present HandFlow, a fully generative flow-matching framework for temporally coherent 3D hand pose and shape estimation from monocular video. Given visual and skeletal observations, HandFlow denoises an entire temporal window of MANO parameters through a single ODE integration. To support this, we use a Flux-style dual-stream transformer that attends across the full sequence to capture long-range dependencies without autoregressive decoding, and a confidence-aware continuous masking mechanism that blends observed features with learnable mask tokens to handle noisy or missing observations. Experiments on DexYCB and HOT3D show that HandFlow achieves state-of-the-art performance, with particularly large gains in world-space accuracy and temporal smoothness. It reduces world-space pose error by over 30% compared with the strongest baseline and achieves the lowest acceleration error among all evaluated methods, while remaining competitive in per-frame pose accuracy. Moreover, on a single GPU HandFlow reconstructs a 150-frame sequence at 47 fps, about 12x faster than the fastest prior video-based method, with reconstruction itself accounting for only a small fraction of the end-to-end latency. 

---
# ProgramTab: Boosting Table Reasoning of LLMs via Programmatic Paradigm 

**Authors**: Pei Guo, Enjie Liu, Yunzhi Tan, Mochi Gao, Jianxin Zhang, Ruichao Zhong, Juntao Li, Bo Hu, Zang Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.11207)  

**Abstract**: Table-based reasoning with large language models (LLMs), which requires reasoning based on natural language questions and structured tabular data, has gained widespread attention. However, a series of issues still constrain the application of this task. The previous approaches suffered from significant performance degradation when faced with large tables due to the difficulty of long text modeling and the limitation of input length for LLMs. The text-to-SQL approach is used to efficiently extract key information from tables and generate smaller sub-tables. However, tabular data, especially web tables, often lack the necessary structure and consistency, making them unsuitable for performing mathematical logic operations using SQL queries. We propose the ProgramTab framework, which guides LLMs employing in-context learning to perform tabular data preprocessing with Python code, as well as the momentous contents extraction with row and column extraction and SQL generation. The experiment results on table reasoning datasets demonstrate that the ProgramTab framework effectively deals with table-based reasoning tasks and outperforms all LLM-based baselines. 

---
# RepTran: Search-Based Repair of Transformer Models 

**Authors**: Yuta Ishimoto, Paolo Arcaini, Fuyuki Ishikawa, Masanari Kondo, Naoyasu Ubayashi, Yasutaka Kamei  

**Link**: [PDF](https://arxiv.org/pdf/2607.11193)  

**Abstract**: To ensure the overall quality of AI-enabled software, not only traditional software components but also AI components need to be tested and repaired. Among AI components, Transformer models are increasingly integrated into software systems, which makes their misbehaviors critical. Although prior work in the software engineering community has proposed deep neural network (DNN) repair methods, most overlook Transformer-specific structures. We propose RepTran, a search-based repair method for Transformer models. It targets their feed-forward networks (FFNs), which play a central role in the architecture. RepTran identifies suspicious weights by combining two types of scores: a variance-based neuron score and an existing bidirectional score. It then iteratively optimizes these weights using differential evolution. Our evaluation includes 18 fault benchmarks constructed from CIFAR-100 and Tiny-ImageNet. We compare RepTran against three baselines: random weight selection, Arachne (a state-of-the-art DNN repair method), and ArachneW, which enables Arachne to control the number of selected weights. RepTran achieved an average repair rate of 74.7%, statistically outperforming random selection and Arachne across all benchmarks. Effect size analysis revealed that RepTran achieved higher repair rates than ArachneW regardless of the number of selected weights. These results suggest that RepTran is effective for enhancing the reliability of AI-enabled software. 

---
# Pix2Act: Image-Space Manipulation Policies with Equivariant Augmentation 

**Authors**: Haojie Huang, Linfeng Zhao, Haotian Liu, Zhang Ye, Si-Yuan Huang, Mingxi Jia, Boce Hu, Fangzhou Lin, Yu Qi, Dian Wang, Robin Walters, Robert Platt  

**Link**: [PDF](https://arxiv.org/pdf/2607.11167)  

**Abstract**: Representing manipulation actions as 2D trajectories in the camera plane provides a compact and interpretable basis for learning complex 3D manipulation policies. However, it also creates challenges from out-of-frame trajectories and limited precision. We propose Pix2Act, an imitation learning method that addresses these challenges by generating continuous image-space keypoint trajectories in each camera plane and losslessly recovering end-effector poses via triangulation. This reformulates high-dimensional 3D control as a simpler, more learnable 2D prediction problem. Crucially, it aligns observations and actions in the same coordinate space, enabling equivariant transformations to jointly rotate individual camera images together with their image-space actions. We analyze the symmetry properties of this augmentation and design a network architecture that can fuse multiple camera views while respecting their per-view rotations. As a result, Pix2Act implicitly enlarges the support of the data distribution and learns invariant action structures across transformations, yielding improved generalization and overall performance. Across diverse simulated and real-world manipulation tasks, Pix2Act outperforms state-of-the-art baselines and remains robust under camera perturbations. 

---
# AMT-X: Phase-Structured Multi-Turn Red-Teaming with Checklist-Gated Evaluation 

**Authors**: Yi Ting Shen, Kentaroh Toyoda, Alex Leung  

**Link**: [PDF](https://arxiv.org/pdf/2607.11151)  

**Abstract**: Safety evaluation of large language models (LLMs) relies largely on single-turn attack datasets and single-judge scoring, underestimating risk from adaptive multi-turn adversaries and reporting a single success rate that does not separate partially actionable outputs from those carrying complete operational detail. We propose AMT-X (Adaptive Multi-Turn Exploitation), a phase-structured multi-turn red-teaming framework. Unlike prior multi-turn attacks that rely on ad hoc escalation or free-form per-goal plans, AMT-X casts the attack as an explicit, reproducible multi-phase state machine driven by semantic signals from the victim, and replaces single-judge scoring with a multi-role jury whose phase-conditioned checklists gate success on actionable harm. Across six frontier victim models (queried under their default safety alignment, without added moderation layers) and seven Moderation sub-categories, AMT-X attains overall attack success rates of 97.6-100% under a lenient score threshold, but 66.7-78.6% under a stricter gate requiring complete, real, and operational detail: a gap of up to 33 percentage points between partially and fully actionable harm. 

---
# BeatEdit: Symbolic Music Generation as Explicit Editing 

**Authors**: Haoyu Gu, Lekai Qian, Haowu Zhou, Qi Liu, Shuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.11124)  

**Abstract**: Music creation is fundamentally a process of revision. Yet symbolic music generation remains dominated by paradigms that produce complete sequences from scratch, with limited support for selective modification. Edit-based methods have proven effective for text transformation tasks, but remain largely unexplored for symbolic music. We trace this absence to the representational level: conventional event-based music encodings lack the structural properties required by explicit music editing. In contrast, the BEAT encoding, a beat-grid-anchored representation originally designed for autoregressive generation, possesses structural properties amenable to editing. We propose BeatEdit, the first framework for symbolic music generation based on explicit edit operations, recasting generation as producing new content by editing a draft rather than synthesizing from scratch. BeatEdit comprises three complementary mechanisms along an axis of increasing edit density: per-token sequence tagging for error correction, iterative refinement for accompaniment editing, and tag-then-fill for segment completion. All these mechanisms share a single encoding and pre-trained backbone, achieving higher precision and perceptual quality than autoregressive and diffusion methods across all three tasks, while remaining efficient, with single-pass inference completing in under 100 ms. Cross-encoding evaluation further reveals that encoding design substantially influences editing effectiveness, with notable encoding-method interaction effects. Code is available at this https URL 

---
# VIA: Visual Interface Agent for Robot Control 

**Authors**: Hengyuan Hu, Priya Sundaresan, Jensen Gao, Dorsa Sadigh  

**Link**: [PDF](https://arxiv.org/pdf/2607.11119)  

**Abstract**: Robot manipulation is a complex task that requires visual understanding, physical reasoning, planning, and closed-loop control. General-purpose foundation models (FMs) have grown remarkably capable of some of these, especially vision and reasoning. To leverage this for generalist robot policies, current methods typically involve converting existing FMs into vision-language-action (VLA) models by fine-tuning on robot data to output low-level actions. However, VLAs are often orders of magnitude smaller than frontier FMs given the limited data and compute available for fine-tuning, which in turn limits their general capability. Inspired by the growing ability of FMs to operate software through visual interfaces, we ask whether that same competence suffices to control a robot. We present VIA (Visual Interface Agent for robot control), a framework that recasts robot control as an agentic task: an off-the-shelf FM-powered agent drives a manipulator through a browser-based 3D interface by taking screenshots, issuing intuitive commands, observing the outcome, and adjusting. The agent receives no robot-specific fine-tuning and no access to privileged state information: it perceives visual input and acts through a small set of general tools. VIA inherits the agent's general reasoning, closed-loop error recovery, and ability to plan and re-plan from what it observes. It solves a diverse suite of tabletop manipulation tasks zero-shot with both Claude Code and Codex. With the strongest model (Fable 5) it achieves 96.7% success on three LIBERO-Goal tasks and 100% on a long-horizon rainbow assembly task. Performance improves with the scale and strength of the underlying model. These results suggest that frontier agents already possess skills that transfer directly to robot control given the right interface: your coding or computer-use agent is, in a sense, secretly a robot-control agent. 

---
# MusicMark: A Robust Generative Watermarking Framework for Music Generation 

**Authors**: Seohwan Yun, Jeeyoung Yun, Yongjin Kim, Juyeon Lee, Sungwoong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.11117)  

**Abstract**: AI music generation has rapidly advanced alongside commercial platforms, raising the need for reliable watermarking for provenance and attribution. However, existing audio watermarking research has largely focused on speech, and applying speech-oriented methods to music is challenging due to music's complex structure and rich acoustic texture. Most existing methods are post-hoc, adding imperceptible perturbations after generation rather than embedding watermarks as part of the content. This makes them fragile under transformations and especially vulnerable to neural codec re-synthesis, which can discard imperceptible residual signals. Moreover, since generation and watermarking are decoupled, the watermarking step can be bypassed or omitted, weakening provenance guarantees. To address these issues, we propose MusicMark, which, to the best of our knowledge, is the first generative watermarking framework for music. Specifically, MusicMark embeds watermark messages into the semantic latent space during generation, incorporating the watermark as part of the musical content and ensuring robustness against diverse attacks, particularly neural codec re-synthesis. To this end, we introduce a watermark adapter into a diffusion-based generation model to embed watermark messages across denoising steps. The adapter and detector are trained with a joint objective that preserves fidelity by constraining watermarked latents close to their unwatermarked reference latents, while improving robustness through attack augmentations. Experiments demonstrate that MusicMark substantially outperforms post-hoc baselines across diverse attacks including neural codec re-synthesis, while maintaining comparable generation quality. We further introduce a cover-song attack, converting the singing voice while preserving musical content, and show that MusicMark remains more robust than post-hoc methods. 

---
# The Equilibrium Is the Initialization: Lazy Identity Collapse in Physics-Structured Deep Equilibrium Reasoning 

**Authors**: Joyjeet Singh  

**Link**: [PDF](https://arxiv.org/pdf/2607.11116)  

**Abstract**: Deep equilibrium models promise input-adaptive implicit computation: harder problems should demand more solver iterations, and the solved equilibrium should encode the result of genuine iterative inference. We report a cautionary study of a port-Hamiltonian DEQ with a learned initialization on two reasoning tasks -- ProofWriter entailment over frozen DeBERTa embeddings and a BFS-verified graph-reachability benchmark -- in which the implicit computation is a silent no-op. Across tasks, seeds, and controlled ablation arms, the solved equilibrium equals the solver's start point to numerical precision, and bypassing the solver entirely changes test accuracy by +0.00 percentage points in 18 of 19 training runs. Controlled interventions falsify the tempting explanation: removing the anchoring term reproduces every result, and retraining with noise-decoupled starts yields a solver that converges to the noisy start while the decoder learns to ignore it. The single escaping run diverges instead ($\|h^{*}-z_0\|=171$), producing a co-adapted noise channel whose removal improves accuracy. Iteration counts are uncorrelated with ground-truth difficulty ($r=0.009$), and the full apparatus never outperforms a two-layer MLP on either task. We trace the mechanism to gradient starvation along two distinct routes, show that the standard zeroing ablation is confounded and gives wildly seed-dependent answers where the correct substitution test gives a stable zero, and distill a four-test diagnostic protocol for auditing claimed implicit computation. All experiments run on a single free Colab GPU; code, raw logs, and analysis scripts are released. 

---
# AgentCheck: A Reproduce-Intervene-Mitigate Workbench for LLM Agents over MCP 

**Authors**: Aritra Mazumder, Nusrat jahan Lia  

**Link**: [PDF](https://arxiv.org/pdf/2607.11098)  

**Abstract**: Tool-using LLM agents are mostly evaluated assuming all tools work. When a tool times out, returns a week-stale value, or has its description poisoned in deployment, the developer needs a controlled way to reproduce the failure, test a fix, and confirm the fix worked before deployment. We present AgentCheck, an open-source web workbench that turns an MCP server into an intervention surface. AgentCheck runs an agent against its real tools and records every tool response, then re-runs the agent with the response perturbed by a fault (12 types) injector. Matching tool calls are replayed from cache, and later tool calls go live after the agent diverges. This yields a reproduce-intervene-confirm loop: the developer toggles a mitigation, re-runs against the identical fault, and sees if the failure goes away. Scoring has two parts: deterministic pass/fail rules, plus an LLM judge for interpretive labels, validated against human annotations. Across five agents, the best passes 105/120 scenarios and the weakest only 77. The failures are usually silent, confident use of incorrect tool outputs rather than crashes. On the weakest agent, a retry mitigation raises success on timeout error faults from as few as 30% of cases to 100%, whereas stale-data faults remain near 3-4 of 10 regardless of the mitigation. AgentCheck makes these failure modes reproducible, comparable, and verifiable before deployment. 

---
# Controlling Motion Transfer in Diffusion Transformers via Attention Heads 

**Authors**: Sunyoung Jung, Jiwoo Park, Yoonseok Choi, Kyobin Choo, Ming-Hsuan Yang, Seong Jae Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2607.11081)  

**Abstract**: Diffusion Transformers (DiTs) have advanced video generation with high-quality, temporally coherent results. However, extending them to motion transfer, which requires following reference motion while aligning with a target prompt, remains challenging due to limited understanding of motion and structure representations within DiTs. We analyze video DiTs at the attention-head level and identify distinct heads specialized for motion and spatial structure. Based on this insight, we propose a head-aware controllable motion transfer framework that requires no parameter updates. Our method refines motion cues from motion-specialized heads via semantic correspondence guidance and preserves structure through selective feature injection. This head-level control not only enables accurate motion transfer but also provides an interpretable foundation for controllable video generation with DiTs. 

---
# Do Video-LLMs Actually Watch? Diagnosing Character-Tracking Failures in Long-Form Video 

**Authors**: Mohammad Al-Ratrout, Shayla Sharmin, Aditya Raikwar, Roghayeh Leila Barmaki  

**Link**: [PDF](https://arxiv.org/pdf/2607.11078)  

**Abstract**: Can a Video Large Language Model (Video-LLM) follow one person through a long video, keeping track of who they are well enough to report, in order, how their outfit changes across a full TV episode? Benchmarks increasingly score this kind of task, and the strongest open-source 7--8B models now reach 37--38% on InfiniBench's global appearance task, which asks exactly that. But does that score come from tracking the named character, or from something easier? We test this with a nine-condition diagnostic protocol applied to three architecturally distinct open-source Video-LLMs, with Gemini~2.5~Flash as a frontier reference, and find the accuracy does not come from character tracking. When we change the character named in the question to a different cast member, leaving the video and answer options untouched, the models change their answer only 4--31% of the time, so they are largely ignoring who the question asks about. Breaking that test down by the gender of the swapped name shows why: the models react more when the name is changed to a different-gender character than to a same-gender one (a 13--28 point gap), picking up coarse gender cues but unable to tell same-gender individuals apart. This shallow processing surfaces again when we drop the multiple-choice options and ask the same questions open-endedly: open-source accuracy drops 18--25 points, with none of 151 answers fully correct, versus a 12-point drop for Gemini. Further checks rule out the obvious innocent explanations, adding subtitles, using the most informative frames, or doubling the number of frames all leave character tracking unimproved, so the bottleneck is not how much video the model sees but how it ties that video to the person the question names. We release a diagnostic toolkit for auditing what such benchmark scores actually measure. 

---
# Flout at Your Own Risk: LLMs Struggle with Pragmatic Cooperativity Under Epistemic Asymmetry 

**Authors**: Hannah VanderHoeven, Abhijnan Nath, Nikhil Krishnaswamy  

**Link**: [PDF](https://arxiv.org/pdf/2607.11053)  

**Abstract**: Fruitful collaborations rely on cooperative communications, including of contextual cues to incorporate into reasoning. The increasing use of LLMs in collaborative and agentic pipelines raises questions about the extent to which they exhibit these pragmatic capabilities, especially in scenarios where they may not have access to the same information as their collaborators. In this paper, we perform a novel investigation into the pragmatic reasoning capabilities of LLMs in a multi-party collaborative task under partial information conditions. We formalize a notion of collaborative epistemic asymmetry that explicitly connects objective task success to Grice's cooperative principle and empirically assess various LLMs' abilities to act cooperatively as both speakers and listeners, including both prompting and post-training strategies. Our results show that while LLMs exhibit certain pragmatic capabilities in collaborative settings, and these can be elicited through prompting and post-training, they still face challenges in pragmatic communication with incomplete information, and that certain failure modes do correlate with floutings of Grice's maxims that go unrecognized. 

---
# BackendForge: Benchmarking Agentic End-to-End Code Generation with Backend Services 

**Authors**: Yuzhe Guo, Mengzhou Wu, Yuan Cao, Jialei Wei, Dezhi Ran, Wei Yang, Tao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2607.11042)  

**Abstract**: Large language models (LLMs) are increasingly used in agentic coding settings, where they can inspect files, execute commands, run tests, observe failures, and iteratively revise code. This shift raises a central evaluation question: can an agentic LLM generate an end-to-end software artifact that is both deployable and behaviorally correct under execution? Backend services provide a controlled but realistic substrate for this evaluation. Their APIs expose application-level executable semantics, and deployed behavior can be checked deterministically against an OpenAPI contract through black-box HTTP interactions. We introduce BackendForge, a benchmark of 56 contract-defined backend generation tasks rewritten from real open-source applications. Given a visible specification and an OpenAPI contract, an LLM must generate a Dockerized service that is built, deployed, and evaluated only through HTTP tests. To strengthen evaluation without introducing hidden requirements, BackendForge uses a test agent and a code agent to co-evolve the test oracle and reference service, where the test agent proposes specification-grounded backend tests and the code agent repairs the reference implementation. Although the best-performing model, GPT-5.5, succeeds on 55.4\% of tasks under the base oracle, it succeeds on only 28.6\% under the final oracle. This gap suggests that current LLMs can implement many local API behaviors, but still struggle to produce complete backend services. 

---
# Same Stories, Different Journeys: From Social Comparison to Sensemaking in AI-Mediated Peer Career Exploration 

**Authors**: Pengping Tan, Baoquan Zhao, Zhenhui Peng  

**Link**: [PDF](https://arxiv.org/pdf/2607.11039)  

**Abstract**: Young job seekers frequently turn to social media to compare themselves with peers and make sense of career possibilities. However, passive feed browsing creates a paradox: the authentic peer content that provides emotional grounding also triggers potentially detrimental upward social comparison and cognitive overload. Previous work has either structured online user-generated content to reduce noise without changing the passive browsing modality, or built AI-powered career exploration systems that disregard authentic human experiences. To address this gap, we developed JobMate, an interactive system that transforms real social media career posts into persona-grounded conversational AI agents, shifting the interaction from passive scrolling to active, personalized dialogue. We conducted a between-subjects study ($N$ = 24, three disciplines) comparing JobMate with native RedNote browsing. Our study shows that JobMate's AI-mediated dialogue redirected social comparison from potentially detrimental upward comparison toward constructive self-reframing, while promoting sensemaking through active conversational engagement. However, users still relied on the authenticity of real peer content for emotional grounding. We discuss design implications for AI systems that augment authentic online user-generated content consumption across social comparison contexts. 

---
# SynCLIP: Synonym-Coherent Language-Image Pretraining for Robust Open-Vocabulary Dense Perception 

**Authors**: Mingjie Xie, Guangjun He, Dongli Xu, Youtian Lin, Hongjue Li, Pengming Feng, Jian Guan, Yue Deng  

**Link**: [PDF](https://arxiv.org/pdf/2607.11008)  

**Abstract**: Open-vocabulary dense perception (OVDP) aims to localize objects unseen during training by leveraging textual knowledge. Despite the remarkable progress of recent CLIP-based approaches, we identify a critical limitation: synonym-induced grounding inconsistency, where semantically equivalent expressions yield disparate spatial attention patterns. This inconsistency undermines the robustness and performance of existing methods in real-world OVDP applications. To address this issue, we propose SynCLIP, a Synonym-Coherent Language-Image Pretraining framework that enhances synonym-robust grounding for OVDP. SynCLIP introduces a Semantic-consistent Spatial Attention alignment (SSA) module to enhance spatial attention consistency by minimizing discrepancies between attention maps of original and synonymous expressions. Furthermore, a Spatial Attention Refinement (SAR) module selectively strengthens the most semantically relevant spatial regions within aligned maps for more precise and stable grounding. To support synonym-coherent pretraining, we also construct a Synonym-Enriched Visual Corpus (SEViC), which augments each category with multiple synonyms and textual definitions. Extensive experiments on multiple benchmarks demonstrate that SynCLIP substantially improves grounding consistency under diverse linguistic variants and achieves state-of-the-art performance among CLIP-based OVDP methods. Code is available at this https URL. 

---
# Actor-Critic Learning for Extended Mean Field Control with Deterministic Policies 

**Authors**: Ziheng Cheng, Xin Guo, Huyên Pham, Yufei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.11005)  

**Abstract**: This paper develops a model-free reinforcement learning framework for continuous--time extended mean field control problems, where both the dynamics and reward may depend on the joint distribution of states and controls. We adopt deterministic feedback policies, under which the state--action distribution is induced directly as a push--forward of the state law. This avoids optimization over stochastic kernels and bypasses key limitations of existing approaches in extended mean field settings.
We first establish a model--free sensitivity formula for parameterized McKean--Vlasov dynamics and use it to derive a deterministic policy gradient formula expressed through an advantage--rate function on the Wasserstein space. We then refine this formula by introducing local value and advantage--rate representations that depend on the state, action, and joint state--action distribution, yielding a policy gradient that includes both action derivatives and measure--derivative terms with respect to the control distribution. These characterizations lead to a martingale--based learning principle and motivate a continuous--time deep deterministic policy gradient algorithm combining particle approximations, measure--dependent neural networks, temporal--difference learning, and exploration in either action or parameter space.
Numerical experiments on stochastic Cucker--Smale consensus control and optimal liquidation with trade crowding demonstrate the efficiency, stability, and robustness of the proposed method, including problems with explicit dependence on the control distribution. 

---
# Affordance-Based Manipulation Planning with Text Goals and Sim-to-Real Generalisation via Real-to-Sim Image Conversion 

**Authors**: Solvi Arnold, Rin Karashima, Tadashi Adachi, Takafumi Mochizuki, Kimitoshi Yamazaki  

**Link**: [PDF](https://arxiv.org/pdf/2607.11004)  

**Abstract**: We present a manipulation planning system based on affordance recognition and action effect prediction. The system reasons through possible futures in visual form, and evaluates candidate plans by agreement of predicted outcomes with text-based goals set at run-time, using a multi-modal goal-matching module. Positions of objects named in the goal text are tracked through predictions even when occluded, making it possible to generate action plans even when objects become occluded, or when their initial descriptors cease to identify them in future states. We further expand the system with an image conversion module for translating real-world state images with objects of varied shapes and visual appearances into a consistent visual appearance, to facilitate manipulation planning in a physical robot setup. We evaluate performance of the system's modules in isolation and demonstrate the integrated system's manipulation planning capabilities on a set of challenging tasks in both simulation and on hardware. 

---
# LoSA-Net: A Localized and Scale-Adaptive Network for Boundary-Sensitive Prediction of Perineural Invasion in 3D MRI 

**Authors**: Youngung Han, Hyunsu Go, Kyeonghun Kim, Induk Um, Junga Kim, Jaewon Jung, Woo Kyoung Jeong, Won Jae Lee, Pa Hong, Ken Ying-Kai Liao, Hyuk-Jae Lee, Nam-Joon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.10992)  

**Abstract**: Perineural invasion (PNI) is a clinically relevant indicator of tumor aggressiveness and can influence surgical decision-making, motivating interest in reliable preoperative assessment. The subtle MRI features of PNI, however, often resemble nearby anatomy, complicating noninvasive prediction. These fine perineural cues are easily attenuated by routine downsampling or overly global feature aggregation, reducing the effectiveness of conventional volumetric models. We present LoSA-Net, a localized and scale-adaptive architecture for boundary-sensitive PNI prediction in 3D MRI. Talking Neighborhood Attention (TNA) preserves nerve-aligned detail through localized self-attention with head-wise mixing, and Scale-Adaptive Feature Mixing (SAFM) modulates the receptive field using multi-scale depthwise processing. Cross-Scale Refinement and Alignment (CSRA) maintains consistency between semantic context and high-resolution boundaries across stages. In contrast-enhanced MRI scans from 168 patients with cholangiocarcinoma, LoSA-Net achieves an AUC of 0.7567 and outperforms representative convolutional and transformer baselines under matched preprocessing and optimization settings. 

---
# Think When It Matters: Conditional VLM Reasoning for Social Navigation with RL Policies 

**Authors**: Ali Ahmadi, Hamed Rahimi, Adrien Jacquet Cretides, Marie Samson, Mahdi Khoramshahi, Mohamed Chetouani  

**Link**: [PDF](https://arxiv.org/pdf/2607.10991)  

**Abstract**: As mobile robots become more integrated into everyday human environments, social robot navigation is becoming essential for ensuring human comfort, safety, and trust. While reinforcement learning (RL) navigation policies provide the fast inference and reactive behavior necessary for real-time deployment, they still lack flexible semantic reasoning capabilities and often fail to generalize to complex social scenarios. Recent approaches have increasingly turned to vision-language models (VLMs) in place of RL policies to improve semantic and social reasoning in robot navigation. Nevertheless, their high computational cost and slow inference remain major barriers to real-time deployment. To overcome these limitations, we introduce HUMA (Hybrid Understanding for Multi-modal social Navigation), a hybrid architecture that dynamically balances the computational efficiency of RL policies with the deep semantic understanding of VLMs. Our approach uses a reactive RL policy to handle low-density, routine navigation tasks, while conditioning it on a post-trained high-level VLM when a human enters sensitive situations, such as the robot's proximity zone. We evaluate HUMA on the Social-MP3D and Social-HM3D benchmarks, where it achieves task success improvements of 20% and 3%, respectively, while significantly reducing personal space violations and human collisions against state-of-the-art baselines. Extensive ablation studies validate each architectural component, and real-world deployment on the Mirokaï mobile robot further demonstrates the practical viability of our approach. 

---
# MMA-Former: Multi-Window Mixture-of-Head Attention Transformer for Adaptive PNI Prediction in 3D MRI 

**Authors**: Youngung Han, Induk Um, Kyeonghun Kim, Junga Kim, Hyunsu Go, Jaewon Jung, Woo Kyoung Jeong, Won Jae Lee, Pa Hong, Ken Ying-Kai Liao, Hyuk-Jae Lee, Nam-Joon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.10988)  

**Abstract**: Perineural invasion (PNI) is a critical prognostic factor in cholangiocarcinoma. Non-invasive prediction from 3D MRI is challenging, demanding models that efficiently capture both fine-grained details and global context. We propose the Multi-window Mixture-of-Head Attention Transformer (MMA-Former), a novel end-to-end 3D architecture featuring a Coarse-Fine Transformer (CFT) structure for parallel multi-scale feature extraction. We advance this structure by integrating a novel Window-Specific Mixture-of-Head attention (WS-MoH) mechanism. Unlike standard Multi-Head Self Attention (MSA), WS-MoH generates a representation for each 3D window and dynamically routes the entire window to specialized or common attention heads. This enables spatially adaptive feature extraction tailored to the local context of each window, enhancing specialization and reducing redundancy without increasing parameters. Evaluated on a retrospective dataset of 168 T1-weighted MRI scans, MMA-Former achieved an AUC of 0.752, outperforming other 3D architectures, including the best CNN (AUC of 0.708) and Transformer baselines (AUC of 0.681). 

---
# EquiFusion: Kinematics-Agnostic Human Motion Prediction via Equivariant Latent Diffusion 

**Authors**: Cecilia Curreli, Florian Hofherr, Dominik Muhle, Abhishek Saroha, Riccardo Marin, Daniel Cremers  

**Link**: [PDF](https://arxiv.org/pdf/2607.10984)  

**Abstract**: Existing Stochastic 3D Human Motion Prediction models are fundamentally constrained by hard-coding the skeleton kinematics, severely limiting generalization, preventing cross-dataset training, and requiring complex data retargeting. We introduce EquiFusion, the first kinematics-agnostic model to solve this bottleneck, implementing a latent diffusion model with a permutation equivariant architecture. EquiFusion treats the kinematics' connectivity as an explicit input parameter, ensuring its internal computations are inherently agnostic to joint ordering and graph structure. This novel design enables truly cross-dataset generalization to unseen kinematics and unlocks novel zero-shot directions, such as motion prediction from partial or occluded observations and targeted limb generation. EquiFusion achieves state-of-the-art results on major benchmarks, being up to 75% more compact than previous kinematics-specific methods, while achieving faster training and inference. EquiFusion thus establishes a new, flexible standard for robust human motion prediction. Model and training code are available at this https URL. 

---
# CGS: Configurable Graph Summarization with Bounded Neighborhood Loss and Query Support 

**Authors**: Shubhadip Mitra, Sona Elza Simon, C Oswald, Arnab Bhattacharya, Arindam Pal  

**Link**: [PDF](https://arxiv.org/pdf/2607.10969)  

**Abstract**: Given a large graph, how to generate a compact summary graph that is configurable by the user and supports multiple graph queries with either no loss or with high accuracy? The ever growing size of graph datasets makes the above question on graph summarization very pertinent. Although, there are several approaches, there does not exist a configurable graph summarization method that offers high compression along with support for multiple graph queries on the summary graph with high accuracy, and allows the user to configure the summarization based on: (1) lossless or lossy summarization, (2) amount of tolerable neighborhood loss, (3) the type of loss it can tolerate, in terms of false positive edges (i.e., extra edges), false negative edges (i.e., missing edges), or neither, in both the (a) reconstructed graph and the (b) query answers. To overcome these limitations, we propose a novel graph summarization framework CGS (Configurable Graph Summarizer) that builds upon the idea of aggregating nodes with common neighborhoods. The CGS framework consists of three summarization variants, CGS-E, CGS-I and CGS-U. While CGS-E is a lossless scheme, CGS-I and CGS-U are lossy schemes that allow reconstruction of the input graph with no false positive edges and no false negative edges, respectively. To bound the graph reconstruction loss, we introduce a user-specified parameter neighborhood loss tolerance threshold, that limits the maximum loss allowed in the neighborhood of each node. This allows graph reconstruction and neighborhood query evaluation with either no loss or with bounded loss guarantees. Empirical evaluation on several synthetic and real-world graphs shows that CGS offers superior summarization than the state-of-the-art methods, and can answer graph queries with fairly high accuracy and efficiency. 

---
# Efficient Online Proportional Sampling with Applications to Smoothed Online Learning 

**Authors**: Amirmahdi Mirfakhar, Maria-Florina Balcan, Hedyeh Beyhaghi  

**Link**: [PDF](https://arxiv.org/pdf/2607.10963)  

**Abstract**: We study the problem of efficient online proportional sampling from a high-dimensional domain under a $\sigma$-smoothed adversary, where the sampling distribution is induced by a dynamically evolving weight function defined over a sequence of piecewise-structured partitions. This setting captures a broad range of applications, including principal-agent games (e.g., pricing and contract design), and algorithm configuration and parameter tuning. The central challenge is maintaining an efficient data structure as the induced partition grows increasingly complex over time -- naively, the number of subregions can grow as $O(t^d)$ by round $t$ in $d$ dimensions. We design a data structure that supports efficient updates and proportional sampling while avoiding the cost of explicitly maintaining this exponential growth, where the discontinuities are structured from axis-parallel hyperplanes. Under a $\sigma$-smoothed adaptive adversary, we prove a tight $O(\sqrt{\sigma T})$ bound on the depth of our data structure, and an $O(\log T)$ bound under a random-order adversary -- to our knowledge, the first such results for this class of problems. We apply this framework to online learning with piecewise-structured rewards, obtaining efficient no-regret algorithms under both full-information and bandit feedback, with provable sublinear regret guarantees. 

---
# Edge Physical AI Deployment of Vision Transformers on Heterogeneous Edge GPU Targeting Autonomous Vehicles 

**Authors**: Ashiyana Abdul Majeed, Mahmoud Meribout, Neethu Joseph, Abel Kidane Haile, Mohammad Abdullah Al Faruque  

**Link**: [PDF](https://arxiv.org/pdf/2607.10942)  

**Abstract**: Physical AI systems, such as autonomous vehicles and intelligent machines, require transformer-based perception models that satisfy stringent edge latency and energy constraints. However, heterogeneous edge-GPU deployment remains limited by underutilized hardware engines and accelerator-incompatible operators, causing fragmented execution and lower throughput per watt. This paper presents Heterogeneous Frame Dispatch Scheduling (H-FraDS), a hardware-aware frame scheduling methodology for transformer inference on a recent NVIDIA edge GPU. H-FraDS routes frames across the GPU and dual deep learning accelerator (DLA) cores using fixed dispatch ratios to improve utilization under latency and power constraints. To enable scheduling, incompatible transformer components are adapted for DLA execution by reshaping tensors, approximating error function (ERF) with tanh, and replacing layer normalization with bounded tanh. The adapted model maintains a 92% F1 score, with only a 2% reduction from the original. Optical flow accelerator (OFA) is further used for inference-side optical-flow estimation. To the best of the authors' knowledge, prior work has not addressed these combined issues. Using Swin Transformer for autonomous-driving perception, H-FraDS Balanced Dispatch (1:2) achieves 125.93 FPS, a 2.36x speedup over standalone adapted-DLA execution, 4.0 FPS/W, and approximately 24 ms DLA latency, satisfying 30 FPS real-time operation; the GPU-DLA-OFA case achieves a 2.02x DLA throughput speedup. 

---
# The Singularity Space: A Generative Diffusion Framework for Signal Representation 

**Authors**: Eli Bar-Yosef, Amir Averbuch, Eli Turkel  

**Link**: [PDF](https://arxiv.org/pdf/2607.10930)  

**Abstract**: Generative models often represent signals as dense grids of amplitudes, blurring sharp transients that are crucial for the correctness of physical signals. We introduce Singularity Space, a generative framework that represents signals through complex-plane singularities, rooted in the classical pole-residue representation of meromorphic functions. We learn a latent space of physically constrained, per-signal singularity configurations to solve an inverse problem from degraded or partial observations. The framework has three key properties: interpretability, in which each generated singularity configuration corresponds to a set of physical parameters; structural stability, which mitigates Gibbs artifacts at discontinuities; and resolution-free output reconstruction on arbitrary grids without retraining or interpolation. Our framework employs a transformer-based diffusion model that directly predicts samples at complex-plane singularity coordinates, subject to geometric constraints during sampling. As a controlled test case for sharp-feature recovery, we evaluate our framework on 1D Burgers shocks, where each shock is represented by 32 predicted singularities (an $8\times$ reduction versus a 1024-point grid signal). Our framework preserves signal structure ($\text{TV ratio} \approx 1$) under unseen test-time observation noise, achieves a $4.2\times$ lower reconstruction error in zero-shot sub-resolution generalization than a grid-based baseline, and recovers physical parameters to $10^{-4}$ absolute error in-distribution. These results suggest that singularity-based representations may provide a practical foundation for other transient-dominated signals such as speech and biomedical signals, with potential extension to higher-dimensional domains. 

---
# The Nuts and Bolts of Natural Language to SQL Translation: A Systematic Analysis of Model Pipeline Optimisation Approaches and their Interactions 

**Authors**: Filip Klubicka, Vasudevan Nedumpozhimana, Sneha Rautmare, Bora Caglayan, Mingxue Wang, John D. Kelleher  

**Link**: [PDF](https://arxiv.org/pdf/2607.10911)  

**Abstract**: In the age of large language models, Natural Language to SQL (NL2SQL) translation remains an open problem with many useful applications. We explore interactions between several NL2SQL pipeline extensions to inspire development of more lightweight models. Specifically, we integrate the NatSQL intermediate representation, include a preprocessing step and a fine-tuning step based on synthetic data, and develop a novel reranker model to improve SQL selection in the final beam. We perform an ablation study supplemented by a Shapley analysis of these different components integrated with two backbone architectures, SmBoP and RASAT. We find that simply combining all of them does not lead to best results, but that their impact depends on their interactions with the baseline system, as well as each other. 

---
# How Do Practitioners Build SE Agents? Insights from a Mixed-Methods Study 

**Authors**: Yunbo Lyu, David Williams, Jieke Shi, Zhensu Sun, Chao Peng, Zhou Yang, Federica Sarro, David Lo  

**Link**: [PDF](https://arxiv.org/pdf/2607.10856)  

**Abstract**: The rise of Software Engineering (SE) agents, i.e., LLM-based agents that can understand large codebases and carry out engineering tasks with limited human intervention, has been marked by rapid advances and adoption, but little is known about how developers build these systems in practice: existing studies mine repositories or examine deployment, but few investigate how SE agents are constructed. Through semi-structured interviews with 20 practitioners from 12 organizations and an online survey of 80 practitioners, this paper is the first to study how SE processes are changing in the development of SE agents and what challenges developers face. We find that as implementation becomes cheaper, bottlenecks shift rather than disappear: long-standing non-coding work such as requirements, coordination, review, and deployment becomes more visible, while reviewing and evaluating agent output becomes new and central. We characterize a seven-stage workflow and a shift toward evaluation-driven development, in which evaluation steers iteration and specifications become versioned artifacts read by both humans and agents. We further identify six challenges that teams face, together with the practices they adopt to address them, including unreliable evaluation signals, comprehension debt as code outpaces understanding, and behavioral changes introduced by provider-side model updates. 

---
# 3D-DefectBench: A Controlled Factorial Study of Vision-Language Model Evaluation Pipelines for Fine-Grained 3D Generation Defects 

**Authors**: Zhenyu Zhao, Nanshan Jia, Jihyeon Je, Yifu Tang, Alvin Chan, Michael Spedden, Michael V. Palleschi, Sui Huang, Jingshen Wang, Zeyu Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2607.10826)  

**Abstract**: Automated evaluation is essential for scaling generative 3D systems, where exhaustive human review is costly and slow. However, the reliability of an automated judge depends on the entire evaluation pipeline, not only the underlying vision-language model (VLM), but also how assets are rendered, what visual evidence is provided, how the task is specified, and how human reference labels are constructed. We introduce 3D-DefectBench, a benchmark and framework for systematic analysis of VLM-based 3D defect detection pipelines. It complements holistic ratings and pairwise preferences with nine fine-grained binary defects spanning geometry, texture, and prompt adherence, providing actionable diagnostics for generator development and judge evaluation. Using a balanced factorial design, we vary four pipeline factors, VLM, camera protocol, visual input, and prompt schema, across 84 inference designs and approximately 3.2 million scored defect decisions, followed by staged validation on a broader set of frontier models. Model choice is the largest determinant of agreement with human labels, but the remaining factors also affect performance, interact with model selection, and can change the best configuration. Within the evaluated design space, a compact six-view RGB protocol performs comparably to denser multi-view settings and inputs augmented with depth or surface normals, making it a strong cost-effective default. Under this standardized pipeline, the best of 12 VLM judges still lag behind trained human labelers, while texture agreement drops sharply when expert-consensus labels are replaced by noisier silver labels. These findings show that automated judges should be evaluated as complete pipelines and calibrated across human reference regimes, rather than benchmarked only as standalone models. We release labels, prompts, predictions, and Croissant metadata on Hugging Face. 

---
# Large Language Models for Token-Efficient and Semantic-Preserving Opinion Summarization 

**Authors**: Fabrizio Marozzo, Stefano Iannicelli  

**Link**: [PDF](https://arxiv.org/pdf/2607.10825)  

**Abstract**: Opinionated text - spanning product reviews, hotel feedback, and social posts - captures rich signals about user experiences, preferences, and concerns. However, the scale, redundancy, and imbalance of such corpora make it challenging to analyze opinions effectively, particularly when the goal is to generate summaries that remain faithful to the diversity of viewpoints expressed. This paper presents a framework that preserves semantics in LLM-based opinion summarization while minimizing token usage. We combine multidimensional classification (e.g., sentiment, topics) with a family of stratified sampling strategies to select compact yet representative subsets of opinions before prompting the LLM. Tailored prompts then produce balanced summaries that surface the salient aspects expressed in the opinions (e.g., strengths and weaknesses of products/hotels). Experiments on Amazon product reviews, Tripadvisor hotel reviews, and X/Twitter posts demonstrate that our method significantly reduces token usage and computational cost while consistently outperforming traditional AI-based and standard LLM summarization baselines in terms of content coverage, balance, and semantic preservation. 

---
# Auditing Belief-Conditioned LLM Agents in Hidden-Information Social Deduction Games 

**Authors**: Yuan Gao, Jiangyi Yang, Yao Zhao, Yichi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.10814)  

**Abstract**: Evaluating LLM agents in hidden-information multi-agent settings is hard: final outcomes are high-variance and rarely reveal why an agent decided as it did. We study this in a 9-player Werewolf environment where agents act under strict, code-level information isolation, and we build an auditable framework that maintains an external belief state over hidden roles, logs belief updates and belief-action deviations as structured evidence, and supports a defensive offline improvement loop that reviews bad cases before any strategy change.
Across 1,080 frozen games spanning belief-disabled, active-belief, kernel-ablation, camp-restricted, consumption-policy, and high-load arms, and including a seed-paired A0/A1 comparison, the active-belief condition is associated with substantially better good-side outcomes: in the 200-seed A0/A1 comparison the good-side win rate rises from 0.205 to 0.390 (paired McNemar $\chi^2 = 16.4$, $p < 0.001$), with fewer irreversible witch-poison errors. We do not, however, attribute this shift to belief content. Direct action-belief consistency is low ($\approx 0.21$), and giving belief only to the werewolves helps the good side more than giving it only to the good side, which argues against a simple holder-benefit account; we therefore report the effect as an association and treat its mechanism as unresolved. The contribution is the audit framework itself: it makes the effect measurable, exposes low direct action-belief consistency, rejects an unreliable forced-consumption intervention with evidence, and separates strategy effects from load confounds. We accordingly position external belief in high-noise hidden-information games primarily as an auditable cognitive baseline that also carries decision-relevant signal, turning opaque agent behavior into replayable evidence for safer, controlled iteration. 

---
# Distributed Agent System: Fault-Tolerant Collaboration Among Embodied Agents 

**Authors**: Kai Yu, Lu Chen, Hanqi Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.10811)  

**Abstract**: AI engineering is shifting from passive text generation by large language models (LLMs) to agent-driven task execution, creating new reliability challenges for long-horizon tasks under resource constraints and environmental uncertainty. Conventional error-elimination optimization strategies fail to address cumulative error propagation. This paper proposes Distributed Agent System (DAS), a device-edge-cloud framework for fault-tolerant collaboration among heterogeneous agents. We redefine agent reliability as system-level fault tolerance rather than single-turn zero-error accuracy, and present a two-layer fault-tolerance architecture: single-agent execution reliability via fault-tolerant alignment, and cross-agent communication reliability via semi-formal language protocols. This framework provides a practical engineering pathway for reliable heterogeneous embodied agents collaboration in industrial scenarios. 

---
# Diachronic Sample Integration: Robust Tail-Risk Estimation with Generative Models 

**Authors**: Shuning Zhao, Patrick Wong, Leran Zhang, Xiaolin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2607.10810)  

**Abstract**: Deep generative models are increasingly used as simulators for downstream decision-making under data scarcity, but in risk-sensitive applications their usefulness depends on rare adverse scenarios rather than typical samples. Standard generative objectives prioritize bulk distributional fidelity, leaving low-probability tails vulnerable to localized optimization noise and making tail-dependent functionals unstable under finite simulation budgets. We introduce Diachronic Sample Integration (DSI), a test-time inference framework that ensembles generated samples across checkpoints from a stochastic training trajectory. DSI targets a checkpoint-mixture distribution that averages checkpoint-specific tail fluctuations rather than relying on a single brittle endpoint. We formalize this mechanism through a finite-budget bias-variance theory. Empirically, across multivariate synthetic processes and high-frequency trading data, DSI substantially reduces tail-estimation error compared to single-checkpoint baselines under fixed simulation budgets, outperforming standard diffusion and state-of-the-art tail-aware baselines without modifying the generative objective. 

---
# Abstractiveness Metrics for Evaluating Text Summarization: A Refined Formulation with Empirical Validation 

**Authors**: Praveenkumar Katwe, Rakesh Chandra Balabantaray, Kali Prasad Vittala  

**Link**: [PDF](https://arxiv.org/pdf/2607.10806)  

**Abstract**: Quantifying abstractiveness in generated summaries is essential for evaluating summarization models beyond
surface-level metrics like ROUGE. We introduce Reference Abstraction (RA), Summary Abstraction (SA), and Abstraction
Ratio (AR) -- a set of principled heuristic metrics that measure how much a summary diverges from extractive copying
of the source text. The formulation uses the harmonic mean of document lengths modulated by a cubic non-overlap
factor, yielding dimensionally consistent, bounded output with non-linear sensitivity to the extractive-abstractive
boundary. Evaluation on 100 XSUM documents across four summarization models (BART-large-cnn, Pegasus-xsum, DistilBart,
MT5-small) demonstrates that the metrics successfully discriminate between extractive models (SA ~ 0.12-0.26) and
abstractive models (SA ~ 0.96-1.77), and that the Abstraction Ratio identifies summaries requiring manual evaluation
for potential hallucination. Code and results are available at this https URL. 

---
# Weight-Adjusted Gradients Reveal Parameter Importance and Failure Modes in LLMs 

**Authors**: Shrestha Datta, Hongfu Liu, Anshuman Chhabra  

**Link**: [PDF](https://arxiv.org/pdf/2607.10803)  

**Abstract**: Understanding which parameters are influential in Large Language Models (LLMs) is central to improving their efficiency, reliability, and interpretability. We introduce Weight-Adjusted Gradients (WAG), a simple yet effective approach for estimating parameter importance that explicitly captures the interaction between model weights and first-order gradient information and identifies parameters that disproportionately influence model behavior, such as those responsible for collapse phenomena in LLMs. Across a range of models and settings, we show that WAG surfaces a tiny but critical subset of parameters whose modification leads to dramatic degradation in performance, a failure mode that existing importance metrics overlook. These findings reveal a previously underexplored interplay between weights and gradients, suggesting that parameter importance cannot be fully understood through either signal alone. The surprising effectiveness of WAG points to fundamental structural properties of trained networks and motivates new open questions about the role of zeroth-order and first-order information in deep learning. We demonstrate the practical utility of WAG across multiple applications, including expert allocation in mixture-of-expert architectures, parameter-specific unlearning, mixed-precision quantization, and layer selection for knowledge editing. Our results position WAG as a unified approach for analyzing, debugging, and controlling LLMs, and opens new directions for principled model-level interpretation. 

---
# LSTrans: Efficient Knowledge Transfer for Lightweight and Automated ECG Classification 

**Authors**: Yi Zhao, Jiajun Gao, Chenyang Xu, Yuxi Zhou, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.10784)  

**Abstract**: Deploying deep learning models for automated electrocardiogram classification on resource-constrained wearable devices remains challenging due to high computational costs. To address this, we propose LSTrans, a lightweight hybrid model designed for efficient and sensitive ECG analysis. LSTrans introduces a specialized 1D convolutional backbone with an interleaved layer architecture to capture both macroscopic rhythmic trends and microscopic morphological variations. This backbone is cascaded with a Transformer encoder to model long-range temporal dependencies, incorporating Low-Rank Adaptation across critical layers to compress the model and reduce the trainable parameter space. We further employ homogeneous and heterogeneous knowledge distillation to transfer diagnostic expertise from high-capacity teacher models to the student. Experimental results on multiple benchmark datasets demonstrate that LSTrans achieves a competitive balance between diagnostic sensitivity and resource efficiency, substantially reducing peak memory footprints and training latency during downstream adaptation. The source code is available for review at this https URL. 

---
# Lightning Fast Matching Dependency Discovery with Desbordante 

**Authors**: Alexey Shlyonskikh, Michael Sinelnikov, Daniil Nikolaev, Yurii Litvinov, George Chernishev  

**Link**: [PDF](https://arxiv.org/pdf/2607.10771)  

**Abstract**: Matching dependency is a generalization of the functional dependency concept, which allows users to apply custom similarity functions for matching individual attributes. Matching dependencies have a wide range of applications for solving various data quality problems, such as entity resolution, data deduplication, data integration, schema matching, and many more. However, their discovery is a very computationally intensive problem, which limits their practical application.
In this paper, we describe a number of optimization techniques for HyMD - currently the state-of-the-art algorithm for the discovery of matching dependencies. These optimizations belong to both technical and scientific domains. The most important of them are: 1) a new sampling technique, 2) a faster generalization lookup technique, and 3) an improved representation of a dependency. The first one aims to raise the efficiency of inference from record pairs, while the last two are designed to speed up lattice-related operations.
To evaluate our optimizations, we implemented our version of HyMD in Desbordante, an open-source high-performance data profiler. Experiments demonstrated that they allow for a speedup of more than 40x over the state-of-the-art implementation on average, reaching a speedup greater than 170x in some cases.
Finally, the improved version of HyMD is ready to use by anyone. It comes with bidirectional Python integration, which allows calling the C++ algorithm implementation from Python programs while allowing users to supply their custom matching functions. 

---
# Multi-Scale Convolution with Optimal Transport Attention Effect on Multivariate Time Series 

**Authors**: HaoChong Fu, Jian Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.10740)  

**Abstract**: The analysis of Multivariate Time Series (MTS) plays an important role in a lot of real-world practical applications, but it still remains some challenging problem about capturing multi-granularity structural patterns and suppressing noise appropriately. Multi-Scale Convolution with Optimal Transport Attention (MSC-OT) is proposed in this paper. MSC-OT is a useful architecture to optimize the attention mechanism. It combines multi-scale convolution with Sinkhorn optimal transport method based on inverted embedding. The inverted embedding approach embeds each variable as a token and allows the model to capture cross-variate relationships better. MSC-OT consists of two part: (1) Multi-Scale Convolution Enhancement, that applies multi-scale convolutions to attention score matrices based on inverted embedding, capturing local structural patterns in the variate-interaction space induced by compressed temporal representations; (2) Sinkhorn Optimal Transport Regularization, that formulates attention computation as an optimal transport problem and employs iterative matrix scaling to ensure balanced information flow across variates. Adaptive Fusion Strategy utilizes softmax-normalized learnable weights to dynamically combine base attention, convolution-enhanced, and OT-regularized scores. Experiments on widely-used datasets, including ETT, Electricity, Traffic, Solar-Energy, and Exchange-Rate, show that MSC-OT achieves well performance in both short-term and long-term forecasting tasks. Ablation experiments further validate the effectiveness of each proposed component and their synergistic contributions to improving prediction accuracy for multivariate time series forecasting. 

---
# To Answer or to Abstain: Mitigating Search-Agent Hallucinations via Abstention-Aware Reinforcement Learning 

**Authors**: Fengji Zhang, Tianyu Fan, Yuxiang Zheng, Xinyao Niu, Chengen Huang, Jacky Keung, Bei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.10738)  

**Abstract**: Recent advances in equipping Large Language Models (LLMs) with search tools and outcome-reward reinforcement learning (RL) have achieved new state-of-the-art results on open-domain QA tasks. However, we argue that current training paradigms harbor a critical vulnerability: they predominantly reward correct answers but fail to penalize fabricated ones when retrieval fails, thereby implicitly exacerbating hallucinations. To address this, we propose Abstention-Aware Reinforcement Learning (AWA-RL), which dynamically shapes the abstention reward utilizing the model's query-specific prior capabilities and continuous on-policy training observations. We also introduce a novel metric, RA-F1, to measure the capability-reliability trade-off. Compared to non-abstaining baselines, AWA-RL boosts absolute precision by up to 10.3% and overall RA-F1 by 2.9%, with only marginal sacrifice in raw accuracy. These results confirm that AWA-RL successfully yields highly capable and reliable search agents. The code, data, and model weights are publicly available at this https URL. 

---
# A Corpus of Persuasion Techniques in Slavic Languages 

**Authors**: Jakub Piskorski, Dimitar Iliyanov Dimitrov, Marina Ernst, Jacek Haneczok, Michał Marcińczuk, Arkadiusz Modzelewski, Roman Yangarber  

**Link**: [PDF](https://arxiv.org/pdf/2607.10715)  

**Abstract**: Persuasion techniques are powerful rhetorical devices used to sway public opinion in a wide range of media. We present a new corpus of persuasion techniques, focusing on Slavic languages. The corpus contains documents in Bulgarian, Polish, and Russian, annotated with persuasion techniques at the coarse-grained text-span level and fine-grained sentence level. The techniques are drawn from a taxonomy of 25 fine-grained persuasion techniques, grouped under six broad categories of rhetorical persuasion strategies. The corpus contains approximately 7500 text spans from 222 documents that cover topics hotly debated at the national and international levels. We describe the corpus creation process, provide detailed statistics, and examine correlations between topics and persuasion techniques. We use classic ML-based and generative AI-based models to provide baselines and benchmark results for the detection and classification of persuasion techniques at the text-span level and sentence level. 

---
# Distributed Denial of Science: How Indirect Data Poisoning of AI Systems Can Industrialize Scientific Fraud 

**Authors**: Bálint Gyevnár, Atoosa Kasirzadeh, Nihar B. Shah  

**Link**: [PDF](https://arxiv.org/pdf/2607.10712)  

**Abstract**: Scientific fraud is the instrument of doubt that malicious entities can use to establish controversy in science. Historically, it required the resources of a company: deep pockets, ghostwritten articles, and corrupt academics. Today, Artificial Intelligence (AI) is increasingly automating scientific research, so we ask: Can a remote adversary weaponize the honest use of AI in science to compromise scientific integrity? We envision and empirically evaluate a new attack, indirect data poisoning, in which an adversary corrupts an open dataset and uploads the poisoned variant to a public repository. Autonomous research agents may independently retrieve and process this data, turning honest scientists into the unpaid and unwitting distributors of fraud at scale. Across five socially-salient topics, from hiring discrimination to the safety of autonomous vehicles, three widely used frontier AI systems (Claude Code with Claude Opus 4.7, Codex with GPT-5.5, Gemini CLI with Gemini 3.1 Pro), and 450 ethically contained experimental runs, we find that poisoning succeeds in 49.56% of runs, while the rate of poisoning detection is only 6.0%. The attack requires no topic-specific trigger-words, agent access, indirect prompt injection, or fabricated papers, only the open data ecosystem and misleading metadata. To mitigate the attacks, we propose and evaluate two measures: a scientist persona and a data provenance audit with five checks (referencing papers, social markers, statistical anomalies, related datasets, poisoning caution). We find that the persona still leaves 16.67% of runs with a poisoned conclusion, but provenance auditing reduces attack success rate to zero. Our results suggest that indirect data poisoning may enable scientific fraud at unprecedented scale, but these attacks can be mitigated with suitable auditing by agents during data retrieval. 

---
# PromptGraph: Graph-Guided Prompt Sanitization for Balancing Privacy and Utility in LLM Inference 

**Authors**: Chen Gu, Hui Wan, Donghui Hu, Hui Wang, Zhuoer Gu  

**Link**: [PDF](https://arxiv.org/pdf/2607.10709)  

**Abstract**: Large Language Model (LLM) services introduce a fundamental privacy challenge. Sensitive information may be inferred not only from explicit identifiers, such as names or phone numbers, but also from contextual associations among otherwise innocuous spans. Existing sanitizers typically assign privacy or utility signals to individual spans without explicitly modeling pairwise relationships among them. In this paper, we propose PromptGraph, a graph-guided prompt-sanitization approach for privacy-preserving LLM inference. PromptGraph estimates privacy leakage at the span level and utility-relevant contextual dependencies between pairs of spans. It represents each prompt as an attributed graph, in which nodes carry span-level privacy scores and edges encode contextual dependencies needed to preserve utility. The sanitization objective selects a protected span set that maximizes privacy gain while penalizing the loss of contextual dependencies. This formulation explicitly balances privacy and utility when contextual evidence is hidden. Protected spans are sanitized locally, and returned placeholders are restored only after passing local consistency checks. We conduct extensive experiments showing that PromptGraph achieves a more favorable balance between privacy and utility than prompt-privacy baselines. 

---
# MDQEC-QAS: Meta-Decoding for Quantum Error Correction with Hardware-Aware VQC Search and Confidence-Gated Recovery 

**Authors**: Prashant Kumar Choudhary, Nouhaila Innan, Muhammad Shafique, Rajeev Singh  

**Link**: [PDF](https://arxiv.org/pdf/2607.10707)  

**Abstract**: We propose a unified meta-decoding framework for quantum error correction that learns syndrome-to-recovery mappings across multiple stabilizer codes and noise settings, without requiring separate decoders for each configuration. The benchmark includes FiveQubit, Steane, Planar3x3, and Planar5x5 codes, four noise families, and five evaluation regimes: interpolation, unseen-p transfer, unseen-noise transfer, few-shot unseen-code adaptation, and few-shot held-out-size adaptation. We compare a classical Meta-MLP teacher-trained baseline with variational quantum circuit (VQC) meta-decoders selected through hardware-aware quantum architecture search over qubit count, circuit depth, and entangling topology. The Meta-MLP achieves teacher-label accuracies of 0.9993, 0.9118, 0.9342, 0.6304, and 0.7548 across the five regimes, while the hardware-aware VQC achieves 0.9400, 0.8495, 0.8415, 0.5678, and 0.7143. However, logical-level evaluation shows that high teacher-label accuracy alone is insufficient in the most challenging Planar5x5 setting. During interpolation, the raw logical-failure ratios relative to the teacher are 12.08 and 25.91 for the Meta-MLP and VQC, respectively, whereas confidence-gated fallback reduces them to 1.71 and 1.11. These results support confidence-aware selective recovery rather than unconditional teacher replacement. 

---
# Action Map Policy: Learning 3D Closed-loop Manipulation via Pixel Classification 

**Authors**: Haojie Huang, Zhang Ye, Linfeng Zhao, Boce Hu, Mingxi Jia, Yu Qi, Ahmed Agha, Dian Wang, Robert Platt, Robin Walters  

**Link**: [PDF](https://arxiv.org/pdf/2607.10706)  

**Abstract**: The action space poses a major challenge in robot learning, since it is often high-dimensional, can span long time horizons, and frequently admits multi-modal optimal solutions. A good choice of action representation and loss function can help to address these concerns, but there are often trade offs. We propose Action Map Policy (AMP), which casts 3D closed-loop manipulation policy learning as a classification problem in image space. While classification has been an effective formulation in generative language models, applying it to robot action learning is difficult because naively discretizing high-dimensional continuous actions explodes the token vocabulary. Our key idea is to project 3D actions onto the camera image planes and treat each pixel location as a discrete class, thus controlling dimensionality while retaining multi-modality. This method supports millimeter-level precision for high-dimensional actions without requiring a prohibitively large vocabulary, while preserving fine-grained pixel-wise visual signals. Furthermore, it can predict the entire action chunk in a single forward pass, avoiding complex noise scheduling and iterative denoising while achieving substantially faster inference than diffusion policies. Experiments on various manipulation tasks show that AMP outperforms strong baselines, achieving higher success rates, faster inference, and enhanced spatial reasoning. 

---
# Learning to Fine-tune Foundation Models under Resource Limitations 

**Authors**: Thomas Tsouparopoulos, Iordanis Koutsopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2607.10694)  

**Abstract**: We study the problem of optimal continual fine-tuning for a pre-trained Foundation Model deployed at a resource-limited device. At each time slot, a new batch of training data arrives, and the controller is faced with two options: either use the data to fine-tune the model and incur a compute cost, or do not fine-tune the model and discard the data. After the decision, the performance of the current model is measured in terms of an application-specific performance metric such as classification accuracy. Our objective is to learn an optimal policy that determines \emph{when to fine-tune the model} on a single task (e.g., sentiment analysis), under a finite compute budget. We formulate this online decision-making problem as a constrained Markov Decision Process, where the system state captures three essential aspects: (\textit{i}) model's performance, (\textit{ii}) computational budget, and (\textit{iii}) data distribution relevance to historic data encountered up to that point. The transition to the next state is stochastic and therefore, we propose a reinforcement learning-based method to solve this problem, namely the \emph{actor-critic} algorithm. We also consider the special case where the performance of fine-tuning for a given model can be predicted or estimated prior to decision; in this case the problem becomes a Dynamic Programming one. Experiments with a large pre-trained model on a widely-used text classification dataset demonstrate that our method consistently outperforms fine-tuning approaches with the same compute budget by more than $4\%$ in terms of accuracy and achieves $97\%$ of full-parameter fine-tuning accuracy while requiring only $25\%$ of the fine-tuning steps. 

---
# Commenting with Copilot: A Taxonomy and Multi-Year Analysis of Student Code-Generation Specifications 

**Authors**: Nasser Giacaman, Valerio Terragni, Paul Denny, Viraj Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2607.10674)  

**Abstract**: As AI code tools become integrated into programming environments, students increasingly describe intended behavior in natural language and rely on these tools to generate code, shifting emphasis from code writing to specification. Yet little is known about the comments students write as specifications in AI-assisted programming tasks. We analyze a four-year dataset of undergraduate programming submissions and reflections from tasks in which students wrote comments to guide code generation and refined solutions using test-case feedback. We introduce a taxonomy spanning three dimensions: comment type, code expression level, and code construct. Using automated classification, we examine how these dimensions vary across attempts and how students describe the process in their reflections. Our findings show that students mostly wrote natural-language What comments, shifted toward How comments for more procedural constructs, and focused more on verifying generated code than on repeatedly rewriting comments. 

---
# Answer-Conditioned Chain-of-Thought Distillation for Few-Shot Industrial Vision with Small VLMs 

**Authors**: Shubham Rao  

**Link**: [PDF](https://arxiv.org/pdf/2607.10666)  

**Abstract**: Deploying AI-based visual inspection in manufacturing is hard because requirements change often, new defect types appear, and large labeled datasets are rarely available. We propose answer-conditioned chain-of-thought (CoT) distillation for rapidly adapting small vision-language models (VLMs) to new industrial tasks using minimal labeled data. A frontier VLM receives each training image along with its correct label and generates a justified visual explanation. A 3B-parameter model is then fine-tuned on these reasoning-augmented examples via LoRA. By conditioning on correct answers, we ensure all training reasoning is directed toward the correct conclusion, which is critical because frontier models score as low as 24.1% on our hardest task. We validate on four industrial classification tasks spanning three image modalities using only 18 to 30 labeled images per task. Across 4 seeds per task (32 training runs), our method outperforms direct fine-tuning on all 16 seed-task combinations, with mean improvements of +1.7 to +4.4 percentage points. A controlled equal-budget experiment confirms the improvement comes from reasoning quality, not additional training steps. An unconditioned baseline demonstrates that with out answer-conditioning, wrong reasoning degrades performance by 17.8 percentage points. On weld radiograph classification, the fine-tuned 3B model outperforms GPT-4.1 by 10.0pp using just 24 training images. 

---
# Unlocking Parallelism in Autoregressive Language Models via Speculative Decoding with Progressive Tree Drafting 

**Authors**: Zipeng Gao, Zhi Zheng, Qingrong Xia, Junda Lin, Ziwei Zhao, Tong Xu, Zhefeng Wang, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.10661)  

**Abstract**: Speculative decoding has significantly accelerated Large Language Model (LLM) inference by alleviating memory-bound bottlenecks. However, traditional speculative decoding typically relies on auxiliary draft modules, incurring significant training and communication overhead. Although recent methods attempt to generate drafts within the target model itself, they often fail to fully exploit its latent parallel capacity due to a lack of structural coordination. In this paper, we propose \textbf{Progressive Tree Drafting (PTD)}, which employs a structured, guided parallel drafting strategy to harness the model's parallel potential. By coupling a progressive tree structure with a stepwise pruning mechanism, PTD actively guides the LLM to explore multiple semantic paths in a single forward pass, ensuring both draft diversity and coherence. Experiments demonstrate that PTD achieves up to $2\times$ decoding speedup across various benchmarks while remaining training-free and model-agnostic. Our code is available at: this https URL. 

---
# Coverage Path Planning: Classical Foundations, Recent Advances, and Future Directions 

**Authors**: Zongyuan Shen, Shalabh Gupta, Shancheng Zhao, Dehua Zhou, Gao Wang, Zhongqiang Ren, Yaming Ou, Yikui Zhai, C. L. Philip Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.10649)  

**Abstract**: Coverage path planning (CPP) is a fundamental problem in robot motion planning, whose aim is to produce robot trajectories that provide complete coverage of target workspaces while minimizing task-specific objectives such as path length, overlap, number of turns, and energy consumption. CPP has widespread applications in cleaning, inspection, mapping, agriculture, manufacturing, surveillance, demining, and environmental monitoring. Although classical CPP has been extensively studied, recent advances have extended CPP beyond single-robot settings to multi-robot systems, complex 3D environments, constrained platforms, learning-based coverage planning, and visual coverage tasks. This paper presents a comprehensive survey of 125 representative works published primarily between 2015 and 2026, while presenting the evolution of recent developments in light of the classical CPP methods published before 2015. The CPP methods are organized into six main categories: single-robot CPP, multi-robot CPP, 3D CPP, constrained CPP, learning-based CPP, and visual CPP. For each category, the review summarizes the main planning formulations, representative algorithms, strengths, and limitations. In addition, the review analyzes how environmental knowledge, workspace geometry, robot constraints, sensing objectives, and coordination requirements shape the CPP problem. The survey further discusses open challenges in scalable online planning, multi-robot coordination, 3D and visual coverage, unified platform-constrained and resource-aware coverage, and learning-enhanced coverage. Thus, the survey provides a structured overview of recent CPP developments and future research directions. 

---
# Auditing Construct Overlap in Explainable Machine Learning: Evidence from Burnout-Depression Prediction Across Student Cohorts 

**Authors**: Alireza Dehghan, Negin Ashrafi  

**Link**: [PDF](https://arxiv.org/pdf/2607.10633)  

**Abstract**: Explainable machine learning (XML) pipelines applied to composite mental health outcomes can produce apparently-robust, cross-population-stable risk hierarchies that are largely artefacts of how the outcome was constructed. We demonstrate this using an ElasticNet pipeline applied to 886 medical students at the University of Lausanne (primary cohort, 2022), validated across 2,580 longitudinal observations at three time points and 701 non-medical students from eight faculties; all three datasets share identical instruments. The pipeline produces a hierarchy in which trait anxiety and health satisfaction dominate wherever the outcome is measured, with Kendall $\tau = 1.0$ for the top-two positions across all five evaluation sets and consistent transfer performance ($R^2$: 0.41-0.49). Two residualization experiments, which isolate shared variance between correlated variables via regression, reveal the mechanism: when trait anxiety (STAI-T) is residualized against the co-included depression subscale (CES-D, $r = 0.72$), model $R^2$ drops from 0.41 to 0.16 and STAI-T falls from rank 1 to rank 6; when burnout subscales are residualized against CES-D, $R^2$ collapses to 0.016. Prediction intervals average 35.4 units on a 0-100 scale (2.4 outcome standard deviations), independently ruling out individual-level deployment. The residualization protocol is the paper's transferable contribution: any XAI study combining correlated predictor and outcome constructs should apply this check before interpreting apparent stability as a finding. 

---
# World Models as Adversaries: Multi-Agent Self-Play Fine-Tuning for Robust Motion Planning 

**Authors**: Tong Nie, Yuewen Mei, Junlin He, Yihong Tang, Jian Sun, Wei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2607.10630)  

**Abstract**: Robust motion planning in dense traffic requires autonomous vehicles to interact in rare and safety-critical scenarios that are underrepresented in naturalistic driving data. Although adversarial training offers a feasible solution, existing methods often rely on external scenario generators, heuristic perturbations, or simulator-heavy rollouts, which makes them difficult to integrate with modern autoregressive planners. Here, we cast adversarially robust planner learning as a constrained min-max game and propose Adversarial World Modeling (AWM), a theoretically grounded multi-agent self-play fine-tuning framework. Since solving the exact game is intractable, AWM introduces a principled decoupled solver. In the inner minimization, the planner's predictive world model is converted into a role-conditioned adversary that learns sparse, scene-adaptive attack coalitions via counterfactual credit assignment. In the outer maximization, the ego planner optimizes a regret-aware robust best response against the frozen AWM, utilizing tail-risk weighting and reference-anchored trust regions to improve hard-case recovery while preserving nominal driving behavior. Experiments on the nuPlan and InterPlan benchmarks demonstrate that our method generates transferable adversarial interactions and yields a robust planner that achieves competitive closed-loop performance in both nominal and highly interactive long-tail scenarios. Theoretical analysis justifies the decoupled solver and the main optimization components. 

---
# Anamnesis: An Open-Source Platform for Large-Scale Backstory-Conditioned Survey Simulation 

**Authors**: Song-Ze Yu, Joseph Suh, Serina Chang, David M. Chan  

**Link**: [PDF](https://arxiv.org/pdf/2607.10628)  

**Abstract**: We present Anamnesis, an interactive system for demographically controllable survey simulation using large language models. Open-source, and designed for non-technical users/researchers, Anamnesis enables the prototyping and stress-testing of survey instruments on virtual populations rather than real human subjects. The platform operationalizes the recently introduced Anthology and Alterity frameworks, which use structured narrative backstories to condition model responses, within a unified web interface. It supports open-ended generation, probabilistic demographic resampling, and multimodal (image and audio) surveys. We evaluate the system through two case studies: (1) replicating segments of Pew Research Center's American Trends Panel (ATP) on political typology and biomedical issues and (2) emulating human preference in the New Yorker Caption Contest. In both cases, Anamnesis produces opinion distributions that more closely match real-world survey data than standard persona-prompting baselines, offering a transparent, reproducible, and open-source alternative to proprietary simulation services. 

---
# WasteAssistant: Regulation-Guided Visual Question Answering Framework for Intelligent Waste Segregation and Sustainable Managemen 

**Authors**: Khush Kataruka, Harshit Maurya, Anuja Vats, Murari Mandal, Kiran Raja, Praveen Kumar Chandaliya  

**Link**: [PDF](https://arxiv.org/pdf/2607.10610)  

**Abstract**: Efficient waste segregation is critical for sustainable urban management and environmental governance. Existing automated systems are limited by single-modality visual processing, insufficient contextual understanding, and weak regulatory alignment. To address these issues, we propose a language-guided vision-AI framework that integrates vision-language models and multimodal large language models for joint visual-linguistic reasoning. This framework implements a visual question answering paradigm aligned with India's Solid Waste Management Rules 2016. We construct a new WasteVQA dataset with 13,500 question-answer pairs across 21 waste categories. Experiments show that the BLIP-based model achieves a BLEU score of 0.8291 and a BERTScore of 0.9273, outperforming traditional CNN-based methods. This work improves source-level segregation accuracy, ensures regulatory compliance, and supports scalable deployment for municipal and citizen-facing waste management, promoting multimodal AI in sustainable urban infrastructure. The source code and dataset are available at: this https URL 

---
# MemDecay: Region-Aware KV Cache Eviction for Efficient LLM Agent Inference 

**Authors**: Venkatesha Matam, Keon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.10582)  

**Abstract**: Large language model (LLM) agents accumulate heterogeneous context, including system instructions, plans, user turns, retrieved documents, tool outputs, and intermediate reasoning, whose key-value (KV) cache can become a major memory bottleneck. Existing eviction policies generally apply the same attention- or recency-based rule to every token, ignoring semantic structure already available to the agent orchestrator.
We introduce MemDecay, a training-free, region-aware KV-cache eviction policy. MemDecay assigns tokens region-specific base priorities and decay rates, refreshes retention scores when tokens receive attention, and evicts the lowest-scoring pages under a fixed cache budget while allowing critical regions to be pinned. We also provide a procedure for calibrating decay rates from measured attention lifetimes.
We evaluate MemDecay at approximately 450 and 1,700 token contexts using Qwen2.5-1.5B and 3B. Across all settings, attention lifetimes differ by an order of magnitude across regions: system-token half-lives range from 148 to 189 decoding steps, compared with 14 to 16 for scratchpad tokens. Pinning preserves system-region facts at full-cache accuracy in every setting, while no baseline preserves more than 13 of 24. Region-aware retention remains effective as context grows, whereas recency-based retention collapses. Accumulated-attention retention performs better on unpinned content, however, and ablations identify attention-score normalization as the main limitation of the current formulation. These results establish semantic prompt structure as a robust signal for KV-cache management while clarifying how it should be combined with attention-based importance. 

---
# DiffUE: Enhancing Utility-Unlearnability Trade-off of Unlearnable Examples via Diffusion Autoencoders 

**Authors**: Syed Irfan Ali Meerza, Oktay Ozturk, Amir Sadovnik, Jian Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.10580)  

**Abstract**: AI models are increasingly trained on personal images scraped from social media and public platforms, often without consent, leading to serious privacy violations, such as unauthorized facial recognition and targeted advertising. To counter this, researchers have developed unlearnable examples (UEs), images modified with imperceptible noise to prevent AI models from extracting meaningful information. However, existing UE methods primarily rely on pixel-space noise, which can be bypassed by relearning strategies such as adversarial training, image transformation, and compression. While some techniques improve robustness, they often come at the expense of significant degradation in image utility and perceptual quality. In this paper, we introduce DiffUE to overcome these limitations by injecting noise into the semantic space of images instead of the pixel space. Instead of corrupting pixel values, DiffUE modifies high-level semantic features of images, ensuring robust unlearnability while preserving visual quality and utility. By leveraging a diffusion-based autoencoder framework to manipulate semantic features, DiffUE generates purposeful, natural-looking modifications that effectively resist advanced relearning strategies. Extensive experiments on four datasets, CIFAR-10, CIFAR-100, CelebA-HQ, and ImageNet, as well as a subjective user study, demonstrate that DiffUE significantly enhances the trade-off between image quality and unlearnability, offering a more robust and effective solution for safeguarding personal data in an increasingly exploitative AI landscape. 

---
# Learning from Local Walks on Dynamic Graphs with Bandit Feedback 

**Authors**: Sourav Chakraborty, Amit Kiran Rege, Claire Monteleoni, Lijun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.10571)  

**Abstract**: We study stochastic multi-armed bandits on dynamic graphs, where arms correspond to the vertices of a network with time-varying edges. In this setting, the learner is restricted to local movement, selecting only its current node or an immediate neighbor at each round. This constraint decouples best-arm identification from exploitation: even after the optimal arm is identified, the learner may remain unable to reach it through the evolving topology. We identify a process-agnostic structural condition, based on sliding-window mixing, that ensures the graph's intrinsic walk remains stable for both exploration and navigation. Under this regime, we analyze a family of local explore-then-commit algorithms and establish sublinear expected regret. Our framework includes a reward-aware strategy, for which we prove a worst-case safety theorem and a separate performance gain theorem. 

---
# When Does Restricting a Coding Agent to execute_code Help? A Regime $\times$ Agent-Design Ablation 

**Authors**: Hong Yang, Qi Yu, Travis Desell  

**Link**: [PDF](https://arxiv.org/pdf/2607.10569)  

**Abstract**: Modern coding agents expose multiple tool surfaces -- IDE primitives, bash, and Model Context Protocol (MCP) code-execution -- and the field has shipped three contradictory claims about which one matters. We run the missing crossed comparison: an integrity-clean three-arm ablation (baseline / bash_only / code_only) on synthetic computation tasks and SWE-bench Mini modification tasks, holding model, harness, and prompts fixed, with two agents (Claude Code, OpenAI Codex CLI) so the comparison spans both regime and agent-design axes. Across the four resulting (regime, agent) cells, restricting the agent to a single execute_code MCP tool is cheaper than -- or statistically tied with -- its cheapest tool-rich rival in three cells (significantly on Artifact/Claude and SWE-bench/Codex; directionally on Artifact/Codex), with pass rates statistically tied within each cell. The lone exception is SWE-bench/Claude, where code_only is directionally costlier (+14.4%, not significant); a conditional-cost analysis localizes that gap to failure-cost on doomed-run trajectories, not a per-edit tax on successful runs. Two implications: the cheapest tool surface is jointly determined by task regime and agent design rather than by either axis alone, and the headline cost signal lives in cache-adjusted cost -- not pass rate, which is invariant across surfaces at the model sizes we evaluate. The benchmark harness, task suite, and analysis code are available at this https URL. 

---
# Tool-Adaptive LLM Reranker 

**Authors**: Zichuan Liu, Ruijin Hua  

**Link**: [PDF](https://arxiv.org/pdf/2607.10555)  

**Abstract**: Generative Large Language Models (LLMs) have revolutionized information retrieval, yet their strictly parametric nature frequently leads to severe factual hallucinations when confronted with complex queries beyond their epistemic boundaries. While external tool-calling can mitigate this, indiscriminately invoking search tools for every document during reranking incurs prohibitive latency overheads, creating an intractable accuracy-efficiency dilemma. To address this challenge, we propose TALRanker, a novel framework that formalizes pointwise relevance scoring as an agentic Markov decision process. We optimize it via a two-stage training paradigm. An initial warm-up utilizes a language-preserving hybrid loss to prevent the catastrophic forgetting of native generative capacities. Subsequently, an asymmetric cost-aware reward equipped in reinforcement learning forces the policy to autonomously bypass tools for maximum efficiency when confident, while selectively retrieving external evidence to avert severe hallucination penalties when uncertain. Extensive evaluations demonstrate that TALRanker achieves state-of-the-art performance across standard and reasoning-intensive retrieval benchmarks, matching throughput with pointwise rerankers while outperforming parameter-heavy reasoning models. 

---
# Motif: Discovering and Automating Personal Web Workflows 

**Authors**: Shaokang Jiang, Daye Nam  

**Link**: [PDF](https://arxiv.org/pdf/2607.10531)  

**Abstract**: Recent advances in LLMs and existing work on programming by demonstration have made it possible for end users to create automations by explicitly demonstrating their behavior to LLMs. However, these approaches rely on the assumption that users know what to automate and what is capable of being automated. Additionally, automation via LLM agents is often expensive compared with programs. We introduce Motif, a system that passively observes everyday browser activity to discover recurring interaction patterns that are programmable, makes recommendations to users whenever a pattern is discovered and generate a program to install after user confirmation. Users can review, and refine the program using natural language. We evaluated Motif in a multi-day study, comparing its ambient discoveries against automations users attempted to build via ``vibe coding.'' With eight participants, Motif discovered more automatable patterns than users recognized. Most of them matched participants' routines and were useful. Follow-up surveys showed most would continue using Motif-generated programs. 

---
# Towards Autonomous and Auditable Medical Imaging Model Development 

**Authors**: Shengyuan Liu, Jia-Xuan Jiang, Boyun Zheng, Cheng Wang, Zipei Wang, Wentao Pan, Hongtao Wu, Houwen Peng, Yu Gu, Lichao Sun, Yixuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2607.10522)  

**Abstract**: Large language model (LLM) agents are beginning to automate machine learning engineering (MLE) by coupling planning, code execution, debugging, and empirical feedback. Translating this capability to medical imaging remains difficult because each task imposes modality-specific experimentation and strict requirements for validation protocols and prediction artifacts. Here we introduce AMID, an autonomous multi-agent framework for medical imaging model development. AMID first proposes Data-Conditioned Method Planning, which refines coarse task-level search spaces into executable, parallelizable method lanes grounded in task-specific data analysis and runnable medical-imaging resources. It then develops Verification-Guided Two-Stage Optimization, moving from broad early exploration of diverse method lanes to selective exploitation of promising candidates while enforcing strict verification of validation protocols, metric computation, and prediction artifacts throughout the optimization. Across 20 medical imaging challenge tasks spanning diverse modalities and prediction types, AMID outperformed evaluated general-purpose MLE systems and, on several tasks, approached or matched strong human-designed challenge solutions. These results suggest that AMID can turn task-specific medical imaging model development from bespoke manual engineering into an agentic workflow for producing high-performing and auditable model artifacts across heterogeneous tasks. 

---
# Conditional Optimal Bridge for Riemannian Activation Steering 

**Authors**: Seyed Arshan Dalili, Ajay Narayanan Sridhar, Vijaykrishnan Narayanan, Mehrdad Mahdavi  

**Link**: [PDF](https://arxiv.org/pdf/2607.10517)  

**Abstract**: Activation steering offers a lightweight alternative to fine-tuning for controlling large language models at inference time. While many existing methods implicitly optimize a log-density-ratio objective between desired and undesired activation distributions, they do so heuristically rather than deriving it from a principled optimization problem. Moreover, these methods produce query-independent steering directions that can degrade performance on both in-distribution and out-of-distribution (OOD) inputs. We introduce \textsc{Cobras} (Conditional Optimal Bridge for Riemannian Activation Steering), which addresses both limitations by casting activation steering as a Schrödinger Bridge on the residual-stream hypersphere. This formulation yields, to our knowledge, the first principled derivation of the log-density-ratio steering objective from a well-posed optimization problem. Solving the bridge via entropic optimal transport and extracting the probability flow ODE recovers the widely used density-ratio gradient as a special case when the Sinkhorn potentials are uniform. Crucially, the Schrödinger potentials are evaluated at the current activation, making the resulting steering direction inherently query-adaptive. Empirically, across four models and three alignment axes (helpfulness, truthfulness, and detoxification), \textsc{Cobras} consistently outperforms prior activation steering baselines while avoiding the OOD degradation commonly observed in existing methods. The code can be found at this https URL. 

---
# Confining Nondeterminism: AI-Driven Research Systems as DBMSs for Reliable, Non-Wasteful, Transparent, and Collaborative Research [Vision] 

**Authors**: Kyoungmin Kim, Anastasia Ailamaki  

**Link**: [PDF](https://arxiv.org/pdf/2607.10508)  

**Abstract**: LLM agents that conduct research (proposing ideas, writing and running code, analyzing results) can already carry a study from research question to figures, yet cannot be fully trusted. The same question asked twice in a row returns different answers; the agent announces a number that no execution produced, and tool use does not prevent this, because nothing binds what the agent reports to what its tools returned; a small upstream change leaves downstream results silently stale, with no way to list which ones; and the agent re-runs preprocessing and rewrites code it has already produced. We argue these failures share one root: every step of today's agent loop is a stochastic LLM call whose internal state nobody, including the agent, can check. Rather than trying to see inside the LLM, we take a lesson from databases, which earn trust without being watched, because deterministic operators over well-defined state make their guarantees hold by construction. We propose organizing a research project the same way. The project lives in a deterministic, versioned dataflow engine (in effect, a query plan over materialized views), and the LLM, together with the user, is a stochastic compiler that may only edit that plan. The executor never calls the LLM; LLM output enters only as versioned code and data that the executor then runs, and any asserted result enters the record only with an execution behind it. Five design rules at this boundary turn familiar database machinery, from versioning and provenance to incremental maintenance and cost-based scheduling, into guarantees that make research reliable, non-wasteful, transparent, and collaborative. This report presents the diagnosis, the requirements, and the design; the guarantee walkthrough, a prototype, and the research agenda appear in the full version, in preparation. The LLM, we argue, should be the query compiler, never the executor. 

---
# Temporary Authority, Permanent Effects: Commit-Time Authorization for LLM Agents 

**Authors**: Igor Santos-Grueiro  

**Link**: [PDF](https://arxiv.org/pdf/2607.10487)  

**Abstract**: LLM agents can commit durable effects from authority evidence that was valid earlier in execution: a DOM snapshot, approval epoch, version witness, branch token, or worker result. We study the commit boundary at which earlier authority evidence no longer authorizes a durable effect. We call this property commit-time authorization: a durable effect is authorized only if the witness that licensed its derived state remains fresh, causally prior, bound to the same effect, and eligible at commit time.
We build a controlled-invalidation suite spanning browser, tool/API, and multi-agent workflows. The suite preserves the user goal and payload shape while invalidating the authority relation before durability. In the primary 54-task matrix, endpoint success remains high: 262/270 runs reach the visible result. Only 55/270 are authorized completions; among the 216 invalidating rows, 207 commit after the authorizing path has failed. All 54 clean controls remain authorized, and a separate 54-run authority-preserving check produces no unauthorized commits.
We then evaluate mitigation families. Prompt caution and single-condition checks are insufficient because different hazards break different boundary conditions. Defenses work when they refresh, rebind, replan, or refuse at the durability boundary. CommitGuard, a fail-closed boundary monitor, blocks stale durable-effect attempts on protected commit surfaces when runtimes emit witness, dependency, binding, and eligibility signals.
The result is a reporting and runtime-design lesson: endpoint success is a utility metric; authorized commit is a security property. 

---
# ARMOR: Stabilizing On-Policy LLM RL with Off-Policy Anchor Samples 

**Authors**: Kexin Huang, Junkang Wu, Jinda Lu, Shuo Yang, Chiyu Ma, Jiancan Wu, Xiang Wang, Xiangnan He, Guoyin Wang, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2607.10481)  

**Abstract**: Reinforcement learning (RL) has significantly enhanced the reasoning capabilities of large language models (LLMs), yet the training process remains notoriously fragile. In this work, we investigate a critical source of this instability: over-optimization, where models exploit training heuristics at the expense of generalizable reasoning. While reverse KL regularization is the standard defense against such degradation, our analysis reveals that it is often insufficient in this regime, as it fails to ensure comprehensive coverage of the reference distribution. To address this, we propose ARMOR (Anchor Rollout and Mixed Optimization for RL), a framework that shifts the paradigm from passive penalty to active sample stabilization. ARMOR comprises two key components: (1) Anchor Rollout, which leverages off-policy data from the reference policy to preserve established solution patterns; and (2) Mixed Optimization, which reformulates the policy objective to enable controlled exploration without relying on auxiliary losses. Extensive experiments on reasoning benchmarks validate that ARMOR effectively mitigates validation collapse, enabling sustained performance improvements over extended training horizons. 

---
# Reinforcement Learning with Verifiable Physics: Post-training LLMs with Continuous Rewards 

**Authors**: Pengfei Cai, Utkarsh Utkarsh, Alan Edelman, Christopher Vincent Rackauckas, Rafael Gomez-Bombarelli  

**Link**: [PDF](https://arxiv.org/pdf/2607.10474)  

**Abstract**: Partial differential equations (PDEs) are foundational to modeling in science and engineering, but constructing reliable numerical solvers remains labor-intensive, demanding expert knowledge of discretization schemes, stability conditions, and boundary treatments. Recent work has begun to frame PDE solving as a code-generation task for large language models (LLMs), yet existing approaches operate primarily at inference time: relying on prompting, debugging, self-refinement, and test-time scaling rather than adapting the model itself. In parallel, reinforcement learning with verifiable rewards has emerged as a post-training paradigm for code and math reasoning, but its verifiers are typically binary: a compiler runs, or a test passes. Such signals discard the graded structure of scientific correctness, where two solvers may both execute and yet differ in solution accuracy by orders of magnitude. In this work, we introduce RLVP: Reinforcement Learning with Verifiable Physics, an RL post-training framework for multi-PDE solver code generation. RLVP addresses this verifiability gap with a hybrid verifier: hard program-validity checks ensure executability, while continuous physics rewards score function-space accuracy and PDE-residual consistency. A single policy is post-trained across diverse PDE families spanning hyperbolic, parabolic, elliptic, and incompressible-flow systems. RLVP improves over both pre-trained and supervised-only baselines on PDE benchmarks, and shows zero-shot improvement transfer to held-out PDEs. We show that a smaller LLM post-trained with RLVP can outperform prompting a frontier model on in-distribution PDE solver generation. The trained policy shows evidence of compositionality in numerical motifs: it recombines stencils, time-stepping schemes, and boundary-handling primitives learned from the PDEs used in training into generated solvers for unseen PDE problems. 

---
# Annotation-Free Furniture Codes: What They Encode, and How Far They Transfer 

**Authors**: Benjamin Friedman  

**Link**: [PDF](https://arxiv.org/pdf/2607.10461)  

**Abstract**: Layout-based 3D scene synthesizers place each object using two human-annotated channels: a categorical class label and a canonical-pose convention. We ask whether a single self-supervised token derived from object geometry can replace both, and study such tokens directly as a representation, decoupled from any synthesizer. A Finite Scalar Quantization (FSQ) point-cloud autoencoder is chamfer-trained on placed 3D-FUTURE furniture with no labels or pose annotations. Diagnostic probes recover fine-category (62.6 +/- 0.5%), super-category (85.6 +/- 1.3%), and yaw (52.7 +/- 0.5 deg) from the codes alone. Swapping the chamfer target from the rotated to the un-rotated point cloud collapses the yaw signal while raising class recovery, showing the codes' rotation content can be set by the training objective. Scaling across asset libraries needs codes that transfer; on an unseen dataset (ShapeNet), alignment is category-dependent: box-like furniture transfers, organically-shaped furniture does not, and a target-blind augmentation partly closes the gap. 

---
# Context by Distinct Information: An Auditable Dirichlet-Process Working Memory for Long, Redundant Context Streams 

**Authors**: Siddharth Pal, Viktoria Rojkova  

**Link**: [PDF](https://arxiv.org/pdf/2607.10441)  

**Abstract**: Context engineering decides what information a model carries forward, and current designs meter it in tokens: compressing the past into a bounded recurrent state, keeping a key-value entry for every token, or imposing a fixed budget through a window or eviction rule. All three make the token the unit of memory even when the stream is redundant and the task depends on the distinct information it carries. Building on a companion mechanism paper that opens a cache slot only when an incoming key is novel, so memory scales with the number of distinct items rather than tokens, we develop that allocate-on-novelty cache as a working-memory component and organize context by how a task depends on the past: recall-carried information belongs in a content-addressed novelty cache, summary-carried information in a recurrent state, and locality-carried information in a recency window. The claim is empirical and bounded. On a matched character-level control, novelty-gated attention reaches full-attention performance while attending to about half the tokens, and coupling the cache with a state-space summary matches full-attention coupling at that reduced cost; the advantage grows as context lengthens, while a sliding window is preferable on short, locality-dominated spans. On next-code prediction over synthetic Medicare claims the coupled component leads full attention and every fixed-budget eviction policy at a thousand-event horizon, whereas cost forecasting over the same stream is summary-carried and the cache is neutral. The retained memory is an inspectable table of templates, codes, drugs, or places rather than an opaque state. The experiments are small-scale and use only public data; they establish the primitive that context can scale with distinct information rather than tokens, in a working memory that is content-addressable and auditable. 

---
# Learning the Brain's Dynamics as a Port-Hamiltonian System 

**Authors**: Dibakar Sigdel  

**Link**: [PDF](https://arxiv.org/pdf/2607.10439)  

**Abstract**: We model human motor cortex during a wrist-extension BCI task as a port-Hamiltonian system (pHS): a conservative interconnection (gyroscopic coupling between neural phasors) plus a dissipative port (power-law energy decay driven by a GNN surrogate). A metriplectic integrator evolves the phasor state; a Fluctuation--Dissipation-consistent noise channel produces stochastic trajectories at body temperature. Training on \FitTrainN\ real EEG cycles (PhysioNet EEGMMIDB, 3 held-out subjects) reaches a test MSE of \FitTestMSE\ and passes three scale-free criticality rungs: near-critical branching ratio ($\sigma\approx1$), $1/f$ power-law spectrum, and long-range DFA correlations. The model generates closed-loop neuromodulation signals that restore phase-locking in silico when applied to de-synchronised inputs, suggesting a path toward structure-preserving BCI decoders. 

---
# Mitigating LLM Sycophancy in Code Smell Detection Using Evidence-Guided Reasoning Prompts 

**Authors**: Istiaq Ahmed Fahad, Kamruzzaman Asif, Md. Nurul Ahad Tawhid  

**Link**: [PDF](https://arxiv.org/pdf/2607.10411)  

**Abstract**: Large Language Models (LLMs) are increasingly used for code smell detection tasks due to their ability to interpret program semantics. However, their reliability in this context remains poorly explored, particularly under varying prompt conditions where model predictions may be influenced by external cues rather than code characteristics. One such limitation is sycophancy bias, where models tend to align their outputs with user-provided assumptions instead of performing objective analysis. In this paper, we present the first systematic empirical study of sycophancy bias in LLM-based code smell detection. Using the MLCQ dataset, we evaluate how different prompt framings like confirmation bias, contradictory hints, and false premises affect model predictions. Our results show that LLMs are highly sensitive to prompt variations, with Decision Flip Rates reaching up to 72% and False Alignment Rates exceeding 90%, indicating substantial instability and agreement with misleading prompts. To address this issue, we propose Evidence-Guided Debiasing Prompting (EGDP), a structured prompting strategy that enforces evidence-first reasoning. EGDP reduces decision instability and improves robustness, lowering Decision Flip Rates to as low as 12% and False Alignment Rates to as low as 21%, while increasing reliance on structurally grounded evidence. Our findings demonstrate that sycophancy bias poses a critical threat to the reliability of LLM-based code smell detection, and that evidence-guided reasoning provides an effective and generalizable mitigation approach. 

---
# Spatula: Exploring On-Demand In-Situ Interfaces and Interaction for Attribute Control 

**Authors**: Boyu Li, Linjie Qiu, Lin-Ping Yuan, Duotun Wang, Yue Jiang, Zeyu Wang, Hongbo Fu  

**Link**: [PDF](https://arxiv.org/pdf/2607.10405)  

**Abstract**: Controlling attributes is a critical step toward achieving the final creative outcome, yet current approaches fall short in supporting users in the iterative refinement of generative content. We propose Spatula, a proof-of-concept system that generates on-demand, in-situ attribute control interfaces and interactions for creating motion graphics. Building on a technical probe that automatically analyzes animation context and generates corresponding attributes and UI, we frame attribute control as an explorable landscape and explore the attribute control space along four key dimensions: Discoverability, Resolution, Scope, and Expandability. Findings from a user study (N=12) show that our system provides intuitive and convenient interactions while supporting diverse needs for fine-grained parameter control. Furthermore, our applications demonstrate that the plug-and-play design generalizes to other domains, such as web design and 3D modeling. 

---
# Large Language Models in Misinformation Ecosystems: Misuse, Defense, and Vulnerability 

**Authors**: Lingwei Wei, Dou Hu, Wei Zhou, Songlin Hu, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2607.10402)  

**Abstract**: Large language models (LLMs) have transformed misinformation from a primarily content-centric problem into a broader ecosystem-level security challenge. When misused, LLMs create risks beyond false content generation, enabling attacks on the social contexts, evidence sources, retrieval corpora, and verification workflows that misinformation defense depends on. In this paper, we introduce a role-layer framework to unify these risks and defenses. The role dimension characterizes LLMs as attackers, defenders, and vulnerable components of verification systems, while the layer dimension covers content, social contexts, evidence environments, and verification workflows. Guided by this framework, we organize LLM-enabled attacks, investigate LLM-based detection and verification methods, analyze vulnerabilities in LLM-centric detection paradigms, and discuss existing countermeasures against LLM-enabled attacks. Building on this synthesis, we identify three key open challenges: moving from static detection accuracy to budgeted ecosystem-level risk evaluation, hardening LLM-centered verification pipelines against adversarial manipulation, and deploying auditable human-in-the-loop verification systems for trustworthy real-world misinformation defense. 

---
# SynthDocBench: Controlled Benchmark for Long-Context Visual Document Understanding 

**Authors**: Abhigya Verma, Khyati Mahajan, Amit Kumar Saha, Shruthan Radhakrishna, Sagar Davasam, Vikas Yadav, Sai Rajeswar Mudumba  

**Link**: [PDF](https://arxiv.org/pdf/2607.10400)  

**Abstract**: Vision language models (VLMs) have achieved strong performance on visual document understanding benchmarks such as DocVQA, ChartQA, and MMLongBench-Doc. However, real-world documents combine multiple factors such as length, layout complexity, modality, and question difficulty, which makes it difficult to attribute model failures to specific causes. We introduce SynthDocBench, a fully synthetic benchmark for long-context visual document understanding that systematically controls factors including document length, layout structure, modality composition, and question type. The benchmark is constructed using a combinatorial design, each factor is varied independently across generated documents, enabling controlled analysis of model behavior. Documents are generated end to end using an LLM pipeline across six layout archetypes, with a 40 percent random override to prevent models from exploiting spurious correlations. Additionally, SynthDocBench spans long-context documents with substantially greater length and structural diversity than existing benchmarks. Evaluating seven frontier VLMs, we uncover three failure modes that existing benchmarks cannot surface: sharp degradation with document length, a systematic positional sensitivity in which the middle third of a document is hardest for five of six models and five of six models show a negative Early-to-Late trend (steepest decline: 8.3 percentage points), and breakdown of chart comprehension in long-document settings. These results suggest that current models may be overfitting to benchmark artifacts rather than achieving robust long-context visual document understanding. 

---
# A Stepwise Questioning Expert-Editor Multi-Agent Framework for Long-Document Summarization 

**Authors**: Lingyun Shen, Xuejia Guo  

**Link**: [PDF](https://arxiv.org/pdf/2607.10390)  

**Abstract**: Although large language models (LLMs) have shown promising potential in news summarization tasks, their performance on long-document summarization remains challenging as their length often exceeds the input limits. As the agent investment, which provide possibility to improve the inherent capabilities of LLMs. To enhance the effectiveness of long-document summarization based on LLMs, this paper proposes an expert-editor stepwise questioning multi-agent method, in which the expert and the editor guide another agent to refine the summary by posing questions on different aspects of the content and providing targeted clues for revision. We conducted experiments on two representative long-document scientific datasets and evaluated the results through widely recognized automatic metrics. The results demonstrated the effectiveness of our method. 

---
# The evolution of AI from image interpretation toward scientific inference in nanoparticle electron microscopy 

**Authors**: Evropi Toulkeridou, Jiafei Li, Leonardo Lari, Panagiotis Grammatikopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2607.10388)  

**Abstract**: Artificial intelligence (AI) is transforming electron microscopy by enabling quantitative analysis of increasingly large and complex datasets for nanoparticle characterization. Recent advances in machine learning (ML) and deep learning (DL) have expanded microscopy from a descriptive imaging technique into a data-driven platform for structural interpretation, dynamic analysis, and scientific inference. This review examines AI methodologies for nanoparticle electron microscopy, focusing on transmission electron microscopy (TEM), high-resolution transmission electron microscopy (HRTEM), scanning transmission electron microscopy (STEM), and in situ TEM. The discussion is organized around the principal challenges in nanoparticle characterization, including particle detection, segmentation, morphology quantification, atomic-resolution restoration, defect identification, two-dimensional-to-three-dimensional structural inference, and analysis of dynamic processes in situ. We review computational approaches from conventional ML and convolutional neural networks to transformer architectures, self-supervised learning, foundation models, multimodal AI, and physics-informed learning. We further discuss integrating microscopy data with simulations, metadata, and autonomous experimentation to relate nanoparticle structure, dynamics, synthesis conditions, and functional properties. The advantages, limitations, benchmarking, and data requirements of current methodologies are critically assessed. Finally, emerging opportunities for foundation models, AI-guided microscopy, closed-loop experimentation, and autonomous materials discovery are discussed. By integrating advances across computer vision, materials informatics, and electron microscopy, this review highlights the role of AI in next-generation nanoparticle characterization and accelerated materials discovery. 

---
# Structured Thoughts For Improved Reasoning And Context Pruning 

**Authors**: Zain Sarwar, Supriyo Chakraborty, Berkcan Kapusuzoglu, Chia-Hsuan Lee, Anirban Das, Stephen Rawls, Kartik Balasubramaniam, Sambit Sahu  

**Link**: [PDF](https://arxiv.org/pdf/2607.10386)  

**Abstract**: Large language models (LLMs) excel at generating long chains of thought, but long reasoning traces are often verbose and memory-inefficient. In this work, we introduce Structured Thoughts, a framework that organizes reasoning into alternating <try> and <outcome> blocks: <try> captures exploratory scratch work, while <outcome> contains the distilled conclusion of that step. We construct a dataset of structured thoughts by segmenting reasoning traces into <try> blocks and prompting an LLM to summarize each step into its corresponding <outcome>. Fine-tuning pretrained foundation models on this reformatted data produces models that adopt the structured reasoning style, leading to performance gains of up to 8.08\% on reasoning benchmarks compared to standard SFT. The explicit structure also enables context pruning: after each <try>/<outcome> pair, the <try> can be pruned, allowing the model to retain conclusions without keeping the full scratch work in the context. A proof-of-concept pruning implementation achieves an average of 85\% memory / context savings with an 8.67\% performance drop across mathematical tasks. 

---
# ABot-N1: Toward a General Visual Language Navigation Foundation Model 

**Authors**: Ruiyan Gong, Yingnan Guo, Junjun Hu, Jintao Kong, Xiaoxu Leng, Tianlun Li, Weize Li, Fei Liu, Zhicheng Liu, Jia Lu, Minghua Luo, Chenlin Ming, Yanfen Shen, Jiyue Tao, Zhengbo Wang, Mingyang Yin, Minqi Gu, Zihao Guan, Wei Guo, Guoqing Liu, Huachong Pang, Menglin Yang, Zeqian Ye, Xiaoxiao Geng, Zhining Gu, Honglin Han, Di Jing, Hongyu Pan, Mingchao Sun, Kuan Yang, Jianfang Zhang, Yanghong Chen, Ye He, Wei Mei, Jiahao Shi, Xiangpo Yang, Yanqing Zhu, Zedong Chu, Xiaolong Wu, Mu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.10383)  

**Abstract**: Visual Language Navigation foundation models aim to unify deep reasoning for grounded spatial decisions with broad versatility for diverse embodied tasks. Current approaches typically achieve this integration via monolithic policies that map observations directly to actions, yet they often suffer from coordinate drift and poor handling of long-tail semantics. Furthermore, these black-box mappings lack interpretability, hindering the simultaneous achievement of generality, robustness, and transparency. We present ABot-N1, a step toward a general Visual Language Navigation foundation model, that addresses these challenges by decoupling cognition from control via a slow-fast architecture guided by dual visual-language signals. More specifically, a slow vision-language reasoner performs explicit Chain-of-Thought reasoning while producing a pixel goal. This compact set of image-space anchor points serves as a universal interface for diverse tasks, including point-goal, object-goal, poi-goal, instruction-following, and person-following. Subsequently, a fast action expert leverages both the textual cues and the pixel guidance to generate continuous waypoints at the native control frequency. By bridging high-level intents and low-level control through pixel-grounded anchors paired with explicit linguistic traces, our approach ensures robust, generalizable, and interpretable navigation across simulation and real-world benchmarks. ABot-N1 establishes new state-of-the-art records, delivering massive gains specifically in urban-scale navigation: boosting POI arrival by 35.0% (to 77.3%) and achieving 95.4%/92.9% SR in complex indoor and outdoor scenes. It also maintains superior robustness across object-reaching, person-following, and instruction-following tasks. New Point-Goal/POI-Goal benchmarks are released as open source to advance the field of urban-scale navigation. 

---
# VINE: Taming Generative Control Policies for Reinforcement Learning 

**Authors**: Rushuai Yang, Zhuo Han, Houlin Li, Hecheng Wang, Zhichao Wu, Rui Zhang, Zhaowei Zhang, Zihong Chen, Xiaohan Yan, Chiming Liu, Yi Chen, Wei Shan, Maoqing Yao  

**Link**: [PDF](https://arxiv.org/pdf/2607.10369)  

**Abstract**: Flow-matching policies have emerged as an effective policy parameterization for robot learning. They iteratively generate actions from noise, enabling highly expressive modeling of complex and multimodal action distributions. However, prior works observed that scaling these policies with value-gradient reinforcement learning (RL) often leads to training instability. Existing methods attribute this instability to iterative generation and therefore avoid end-to-end value-gradient optimization by sacrificing iterative generation, high expressiveness, or value-gradient optimization. Contrary to prior belief, we show the instability does not stem from iterative generation itself, but from the vanilla sampling strategy originally designed for behavior cloning, which becomes brittle under value-gradient RL. Motivated by this insight, we propose VINE, an RL-oriented sampling method that enables stable end-to-end value-gradient optimization for flow-matching policies. Instead of following a single flow trajectory, VINE reconstructs a new interpolation state at every denoising step, creating a stable differentiable path for value-gradient propagation while remaining compatible with the original flow-matching denoising process. As a result, VINE preserves the expressiveness and iterative generation of flow-matching without sacrificing end-to-end value-gradient optimization. Despite performing end-to-end backpropagation through all ten denoising steps, VINE achieves stable policy improvement and consistently outperforms state-of-the-art RL methods on the OGBench offline RL benchmark and real-world robotic manipulation task. Videos are available on our website: this https URL. 

---
# A Hyperbolic Neural Closure for M1 Radiation Transfer 

**Authors**: Bongseok Kim, Jiahao Zhang, Johannes Krotz, Dinshaw Balsara, Ryan McClarren, Guang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2607.10364)  

**Abstract**: In radiation transfer simulations, an M1 method achieves substantial computational savings by replacing the full angular transport equation with a low-order moment system. Because this reduced system is not closed, a closure model is required to represent the unknown higher-order moments using lower-order moments. While machine learning (ML)-based closures can improve accuracy beyond classical analytic closures, unconstrained learned closures may produce non-real characteristic speeds and consequently cause numerical solver breakdown. To guarantee real eigenvalues of the Jacobian associated with ML closures, we propose a hyperbolic neural closure for the M1 radiative transfer system. Rather than directly predicting closure terms, we parameterize the Jacobian through two neural networks: (i) a symmetric matrix network and (ii) a strictly convex entropy network whose Hessian defines a positive definite symmetrizer. These components are combined to yield a Jacobian that is similar to a symmetric matrix, thereby ensuring real eigenvalues. The closure is then reconstructed by numerical integration of the learned Jacobian field along a prescribed integration path. Numerical experiments show that the proposed closure not only achieves higher closure accuracy than classical analytic closures, but also improves solution accuracy and remains stable in discontinuous Galerkin simulations for radiative transfer problems. 

---
# Benchmarking the Robustness of Foundation Models for Mammography under Domain Shift 

**Authors**: Giang Nguyen, Raghav Mehta, Emma A.M. Stanley, Tian Xia, Thi Hao Nguyen, Hieu Pham, Ben Glocker  

**Link**: [PDF](https://arxiv.org/pdf/2607.10358)  

**Abstract**: Foundation models are increasingly used as image feature extractors for mammography, but their robustness under external domain shift remains unclear. We benchmark 15 foundation-model backbones across breast density, BI-RADS severity, and cancer status using a unified frozen-backbone linear-probe protocol, training on 3 source datasets and evaluating on 12 task-compatible out-of-distribution (OOD) datasets after label harmonization. Mammography-specific vision-language models (Mammo-FM and MaMA) provide the strongest mean OOD performance, but robustness is not explained by mammography exposure alone. DINOv3 remains a competitive vision-only baseline, and mammography-adapted pretraining does not consistently improve generalization. Dataset-level analysis further shows that even leading models show heterogeneous performance across datasets. Feature-space inspection reveals that useful representations can preserve clinical signal while retaining dataset and acquisition structure. These findings highlight dataset-level OOD evaluation as a central criterion for assessing mammography representations. Our code is publicly available: this https URL. 

---
# GRC-ProbNet: Uncertainty-aware Feature Extraction for Cardiovascular Disease Classification 

**Authors**: Yash Shah, Omar Todd, Philipp Seeböck, Georg Langs, Ben Glocker, Raghav Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2607.10357)  

**Abstract**: The automatic detection and classification of cardiovascular disease (CVD) from computed tomography (CT) images plays an important role in clinical practice. Recently, a hybrid pipeline (GRC-Net) for CVD classification was proposed, which leverages a deep-learning-based segmentation and registration method to extract radiomic and geometric features. However, GRC-Net relies on a deterministic segmentation mask, without considering the inherent ambiguity associated with cardiac anatomy. In this paper, we propose GRC-ProbNet, which takes advantage of a deep ensemble to produce multiple segmentation masks for a given input. From these masks, we extract multiple uncertainty features. We analyze these uncertainty features for both their correlation with segmentation error and their propagation effects on downstream CVD classification performance. Our experiments on the publicly available MM-WHS and ASOCA datasets show that the uncertainty measure that best reflects segmentation quality is not necessarily the one that provides the strongest signal for downstream CVD classification. Overall, our results demonstrate that GRC-ProbNet utilizing uncertainty features substantially improves CVD classification AUROC (92.92\) compared to the baseline GRC-Net model (91.25%). Our code is publicly available: this https URL. 

---
# From Stochastic to Stable: Rank Stability and Structural Sufficiency in AI Visibility Measurement 

**Authors**: Ronald Sielinski  

**Link**: [PDF](https://arxiv.org/pdf/2607.10341)  

**Abstract**: AI visibility measurement is comparative: practitioners want to know which domains generative search engines cite most often and whether observed differences are large enough to support decisions. Yet the industry lacks a principled way to determine whether enough data has been collected. Collection budgets vary widely across studies and platforms, and conclusions are often drawn from rankings whose stability and precision are unknown. We introduce a sequential convergence framework based on two complementary criteria: rank stability evaluates whether the rank-correlation trajectory has reached a structural plateau, while structural sufficiency evaluates whether the spread of citation shares among established domains -- those whose confidence intervals exclude zero -- exceeds the uncertainty of those estimates. Together, these criteria distinguish rankings that have merely stabilized from those sufficiently resolved to support inference. Both are derived from regularities in the observed citation distribution, including its rank structure, uncertainty profile, and the boundary between observed and established domains. The framework retains a small number of structural constants but requires no externally specified query count, correlation target, or confidence-interval width target; stopping is driven by observed measurement uncertainty and remains robust across a range of sufficiency thresholds. Applied across 30 platform-topic combinations spanning Gemini, SearchGPT, and Perplexity, the framework adapts to platform- and topic-specific citation distributions. Results show that no fixed collection budget can be justified across contexts and that convergence can instead be evaluated from the structure of the observed distribution. The framework provides a practical basis for determining when AI visibility measurements are ready to support comparative analysis. 

---
# Neutralizing Structural Inequality in the Nigerian FinTech Sector 

**Authors**: Muhammad Abdullahi Said  

**Link**: [PDF](https://arxiv.org/pdf/2607.10317)  

**Abstract**: Algorithmic decision systems in financial services often rely on data proxies that inadvertently encode structural inequalities. This paper introduces a hierarchical human-AI triage model for Point of Sale fraud detection in the Nigerian FinTech sector. Adopting a We Are All Equal worldview, we address the challenge of discrimination laundering, wherein the system misinterprets infrastructure related aleatoric noise such as rural network timeouts as fraudulent intent. We implement a three-tier routing policy utilizing a calibrated ensemble model as a primary filter. The policy routes transactions characterized by epistemic uncertainty such as cold start new accounts to specialist analysts while reserving high stakes cases for a senior supervisor. To manage finite human capacity, we utilize a dynamic shadow price to ration human attention and implement a random audit mechanism to prevent human skill atrophy. Our experimental results demonstrate a statistically significant 1.88\% complementarity gap and a 24.79\% percentage point gain in fraud recall over an autonomous baseline. Crucially, the model reduces the regional performance gap from 19.43 to 2.88 percentage points, neutralizing structural bias. Hierarchical collaboration provides a robust mechanism for substantive equality of opportunity, ensuring that rural accounts are not excluded from the digital economy due to environmental brute luck. 

---
# Polarization Detection: A Hybrid Approach with AfroXLMR-Social and DeBERTa for Low- and High-Resource Settings 

**Authors**: Muhammad Abdullahi Said  

**Link**: [PDF](https://arxiv.org/pdf/2607.10312)  

**Abstract**: The rapid proliferation of online polarization threatens social cohesion, necessitating robust automated detection systems that operate effectively across diverse linguistic contexts. This paper presents our system description for the POLAR Shared Task 2026, focusing on the detection and characterization of polarized discourse in English and Hausa. We propose a hybrid modeling strategy: for English binary detection, we leverage the monolingual strength of \textbf{DeBERTa}, while for Hausa and all fine-grained subtasks (Types and Manifestations), we utilize \textbf{AfroXLMR-Social}. This domain-adapted multilingual model proved critical for capturing the nuances of polarization in social media text. To further address computational constraints and data scarcity, we implement Low-Rank Adaptation (LoRA) and textual data augmentation via \texttt{nlpaug}. We report competitive results across all three subtasks, demonstrating that model selection tailored to specific subtask requirements yields the best balance of performance. 

---
# Program-Synthesis-Driven Autodesign of Universal Unitary Operators 

**Authors**: Yifei Zhang, Dong Chen, Fan Wang, Wenrui Zhang, Yan Chen, Dingding Han, Jianmin Yuan, Xiangjin Kong, Yu-Gang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2607.10295)  

**Abstract**: We demonstrate that AI-driven program synthesis can autonomously discover fundamental strategies for decomposing unitary matrices in photonic networks. By extending DreamCoder to complex-valued linear algebra, the system generates decomposition programs achieving the minimal $N(N-1)/2$ Mach-Zehnder interferometers, distinct from both Reck and Clements architectures. Learned programs encode dimension-agnostic invariants: strategies discovered for $5 \times 5$ matrices generalize to higher dimensions such as $64 \times 64$. The discovered programs encode interpretable, dimension-agnostic construction rules. These rules generalize across matrix sizes without retraining, demonstrating that autonomous program synthesis can serve as a scalable paradigm for algorithm discovery and the automated design of universal unitary operators. Beyond universal decompositions, the system automatically exploits matrix structure to reduce the interferometer count below the universal theoretical bound. For instance, for Householder matrices, it discovers a dimension-independent rule that requires only $2N-3$ MZIs. This achieves linear, rather than quadratic, scaling and generalizes to arbitrary $N$ without retraining. For matrices obtained from the singular value decomposition of sparse matrices, reductions generally increase with sparsity, reaching up to 38% fewer MZIs than the universal theoretical bound $N(N-1)/2$ at 95% sparsity. These MZI reductions translate directly into practical hardware benefits for scalable photonic implementations. Taken together, the system functions as a single unified engine that discovers both universal decomposition rules and matrix-specific optimizations, without being provided with the structural or analytical properties of the input matrices. 

---
# Partial Contracts Suffice: Sound, LLM-Inferred Regression Verification 

**Authors**: Yiannis Charalambous, Rafael Menezes, Youcheng Sun, Lucas C. Cordeiro  

**Link**: [PDF](https://arxiv.org/pdf/2607.10291)  

**Abstract**: Software evolves continuously, yet ensuring that a patch preserves intended behavior without re-verifying an entire codebase remains difficult. Regression verification addresses this problem, but existing techniques require expensive whole-program reasoning or rely on manually written specifications that are rarely available in practice. We present the first contract-based regression verification tool. Contract soundness is ensured by proving all function versions match the behavior. The contract then verifies program flow via assume-guarantee. We ask whether a partial, caller-sufficient contract, rather than a full behavioral specification, is enough. On Frama-C-Problems we strengthen each inferred contract past what the caller needs and measure how much tighter it becomes. It barely moves: for most targets in every model the caller-sufficient contract is already the tightest the loop reaches, and our tightness comparator rates the partial and strengthened contracts equivalent for the large majority of targets it can compare. Partial-spec contracts thus capture nearly all the attainable tightness, so stopping at caller-sufficiency costs almost nothing. The regression check underneath is sound: on the third-party EqBench-C suite it never fabricates an equivalence, returning zero false proofs and reporting an unprovable difference instead. It also surfaced nine pairs that EqBench mislabels as equivalent, more than a concurrent tool reports. The contracts themselves are inferred automatically from the checker's own counterexamples, with no separate specification step; on Frama-C-Problems and the ANSSI X509 parser this reaches a verification rate comparable to tools AutoSpec and Preguss, while a passing result certifies at least as strong a property, which we call \emph{safety-preserving conditional equivalence}: enforcement plus caller-sufficiency. 

---
# PhenoEmbed: Self-Supervised Multispectral UAV Time-Series Embeddings for Individual Tree Crown Phenology 

**Authors**: Taimur Khan  

**Link**: [PDF](https://arxiv.org/pdf/2607.10231)  

**Abstract**: Tree crowns are a challenging target for resilient AI because they are not static objects: their spectral response, internal texture, translucency, and apparent boundaries change substantially across the growing season. We develop PhenoEmbed, a self-supervised crown-centric temporal embedding model trained with contrastive and masked reconstruction objectives on HeideBench, an 18-date UAV multispectral time-series benchmark for forest crown phenology in D{ö}lauer Heide. The model treats seasonal crown dynamics as phenological appearance change driven by leaf emergence, canopy closure, senescence, and leaf-off conditions. Segmented tree crown polygons are retained as object anchors to extract aligned crown-centered crops through time, allowing one 256-dimensional vector summarizing seasonal crown appearance to be learned per tree. On 5,885 crop-safe crowns, the exported embeddings show structured low-dimensional organization, with the first two principal components explaining 25.1\% of variance and nearest-neighbor retrieval producing a median top-1 cosine similarity of 0.946. Compared with handcrafted temporal features and a learned mean-pooling baseline, PhenoEmbed yields substantially more compact nearest-neighbor structure, while ablations show that the contrastive loss, masked reconstruction loss, and explicit seasonal time features each affect the structure of the learned embedding space. These results support PhenoEmbed as a reusable forest crown representation learner and motivate future downstream tests of whether such features improve tree-level models under seasonal change. 

---
# Exploratory Analysis of Deep Learning Models for Forecasting Meteorological Parameters in the Agricultural Sector 

**Authors**: Piotr Sikora, Sotirios Kontogiannis  

**Link**: [PDF](https://arxiv.org/pdf/2607.10208)  

**Abstract**: Accurate meteorological forecasting is essential for agricultural planning, irrigation management, and environmental decision support. This study conducts a comparative evaluation of recurrent and hybrid deep learning architectures for multivariate forecasting of reference evapotranspiration ($ET_0$), vapour pressure deficit (VPD), wind speed, and the sine and cosine components of wind direction. The analysis utilizes 134,376 hourly observations from Ioannina, Greece, spanning January 2011 to April 2026, sourced from ERA5 via the OpenMeteo Historical Weather API. Single and multi-layer GRU and LSTM networks are compared with hybrid 1D-CNN-GRU and 1D-CNN-LSTM models for two forecasting tasks: a 24-hour next-day forecast and a 168-hour week-ahead forecast. Performance is evaluated using normalized root mean squared error, the coefficient of determination, and a composite Weighted Quotient Score (WQS). The most effective purely recurrent models are a 64-unit LSTM for the 24-hour horizon, with a WQS of 0.816755, and a 1024-unit GRU for the 168-hour horizon, with a WQS of 0.779465. The hybrid CNN-GRU models achieved the highest overall scores of 0.827535 and 0.782863 for the 24-hour and 168-hour horizons, but with additionally more number of units respectively to LSTM models, while the CNN-LSTM models yield nearly identical results with substantially fewer parameters. Compared to the corresponding recurrent baselines, the hybrid models improve WQS by 1.22--1.63\% at 24 hours and by 0.44--0.45\% at 168 hours, indicating that convolutional feature extraction is more beneficial for short-term forecasting. 

---
# Source-Lifted Flow Matching for Intervenable Multimodal Imitation 

**Authors**: He Zhang, Ying Sun, Pengteng Li, Ziyang Chen, Yiren Zhao, Ziyang Rao, Weiyu Guo, Yandong Guo, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2607.10206)  

**Abstract**: Flow-matching policies are promising for imitation learning because they model complex multimodal action distributions. However, their stochasticity is largely passive: repeated sampling may yield diverse behaviors, but users cannot directly choose among valid continuations from the same state. We propose Source-Lifted Flow Matching (SL-FM), a source-intervenable flow-matching policy that exposes such a handle while keeping the velocity field shared and latent-free. The handle selects only the source endpoint of the conditional flow, not a mode-specific field, preserving the standard formulation while avoiding decomposition into separate mode-conditioned dynamics. The core mechanism is \textbf{Orthogonal Source Lifting}, designed to prevent path-crossing ambiguity. Instead of partitioning target actions by mode, SL-FM lifts handle-specific sources into auxiliary orthogonal coordinates and keeps targets in the original action subspace. This preserves the demonstrated action distribution while allowing one shared field to carry different branches without merging at crossings. To keep handles usable across states, we learn a state-dependent source mixture end to end and use a responsibility floor, giving each handle weak supervision and mitigating dead modes. Experiments on crossing-flow diagnostics and robot-control benchmarks show that SL-FM converts passive source randomness into an actionable intervention variable. It removes crossing-induced composite trajectories, changes future routes in 91.1\% of matched-prefix interventions, and achieves strong free-deployment performance, with improvements in several benchmark settings. Overall, source geometry provides actionable multimodal control without conditioning the velocity field on the selected mode. 

---
# When Does Depth Survive Composition? Compute--Quality Regimes in Latent World Models 

**Authors**: Achyuthan Sivasankar  

**Link**: [PDF](https://arxiv.org/pdf/2607.10203)  

**Abstract**: Adaptive-compute world models -- early-exit or mixture-of-depths predictors that spend variable depth per step -- assume depth buys better predictions and can be routed adaptively. In autoregressive rollouts, the first assumption requires depth's per-step precision to survive composition. We test this with a pre-registered instrument, the shallow penalty $\rho=\mathrm{err}(\text{shallowest-exit rollout})/\mathrm{err}(\text{full-depth rollout})$, across nine DeepMind Control tasks under matched single-step ($K=1$) and multi-step ($K=4$) training, three seeds each. We find three regimes: on 6/9 tasks depth helps rollouts (intrinsic, $\rho$ up to $4.7\times$), on 2/9 the shallow exits beat the full stack (inversion, $\rho$ down to $0.85\times$), and one is flat. The robust inversion (cheetah) is not a property of the dynamics but is created by training: an ablation supervising early exits only at the first rollout step erases it ($\rho: 0.87\to1.18$, $n=8$, $\Delta=+0.31$), while an intrinsic-tradeoff task is unaffected -- a double dissociation we call the routability catch-22, since the supervision that makes exits routable is what trains them to out-roll the full stack. The regime is partly predictable a priori: observation/action dimensionality and one-step model error correlate with $\rho$ at $|\text{Spearman}|\approx0.75$ ($n=9$). Inside a CEM planner, $\rho$'s sign predicts whether planning benefits from depth, most sharply on the inversion task, where shallow planning beats deep. Finally, three cautions: a task's regime depends on the metric space, the rollout horizon, and the encoder. All thresholds and gates were fixed before the compute campaign, including a pre-registered negative for the hypothesis that motivated the study. 

---
# Comparing Socially-Equitable Renewable Energy Budget Allocation MDP Policies in Mature and Emerging Economies 

**Authors**: Riya Kinnarkar, Mansur M. Arief, Yan Pratama Akhra, Dino Arla  

**Link**: [PDF](https://arxiv.org/pdf/2607.10201)  

**Abstract**: Equitable renewable-energy planning is a sequential decision problem, but the decision variables available to a public planner differ sharply between mature and emerging economies. In the former the government largely builds generation, while in the latter it steers private investment through incentives and quotas. We formulate socially-equitable renewable-energy budget allocation as a Markov Decision Process (MDP) and, using a single problem-agnostic solver interface, compare the same policies across the two settings: eight U.S. cities (a mature economy) and West Java, Indonesia (an emerging economy). The results show that across both settings, a receding-horizon value-iteration policy dominates. In the U.S., it reaches 66% renewable penetration while cutting the underserved low-income population by 96% versus a random baseline. In West Java it closes the low-access gap while crowding in the most private capital. More interestingly, a naive market-chasing heuristic, which is mildly sub-optimal in the U.S., could yield catastrophic outcomes in Indonesia, by underserving every low-access region, because chasing attractive markets and serving the underserved goals diverge once the planner acts through private developers. 

---
# Instruction Set and Language for Hypergraphs 

**Authors**: Mario Pascual-Gonzalez, Ezequiel Lopez-Rubio  

**Link**: [PDF](https://arxiv.org/pdf/2607.10194)  

**Abstract**: We present IsalHG, a method for representing the structure of any finite, connected hypergraph of bounded hyperedge arity as a string over a compact instruction alphabet $\Sigma_{\mathrm{HG}}$. The encoding is executed by a small virtual machine comprising a sparse hypergraph, a circular doubly-linked list (CDLL) of node references, and $k$ traversal pointers, where $k$ bounds the hyperedge arity. Instructions either move a pointer through the CDLL or insert a hyperedge, optionally together with new nodes, into the hypergraph. Every string over $\Sigma_{\mathrm{HG}}$ decodes to a valid hypergraph; the alphabet is closed. A greedy \emph{HypergraphToString} (h2s) algorithm encodes any connected hypergraph into a string; a backtracking variant seeded at nodes of lexicographically maximal structural tuple produces a \emph{canonical string} $w^{*}$, which we conjecture to be a complete isomorphism invariant. Canonical-string equality then decides hypergraph isomorphism natively, without the standard reduction to the Levi incidence graph followed by a graph-isomorphism engine. We verify the round-trip property $s2h(h2s(H)) \cong H$ on 150 connected random uniform hypergraphs and on named combinatorial designs, and we benchmark the canonical algorithm against the three practically available exact baselines -- nauty, Traces, and bliss operating on the 2-coloured Levi graph -- across a $(n, c)$ grid with ten seeds per cell. All four methods agree on every one of 600 isomorphism verdicts, consistent with the completeness conjecture. On wall-clock time the Levi baselines dominate every tested cell by three to five orders of magnitude (geometric-mean ratio $311\times$ to $117{,}672\times$), which we report as measured. We contribute the representation framework, a conjecture of canonical completeness, and the first native-versus-Levi benchmark for hypergraph isomorphism. 

---
# Breaking the Quality--Intelligibility Trade-off in Streaming Target Speaker Extraction via Deep-Feature-Anchored Preference Optimization 

**Authors**: Shuhai Peng, Jinjiang Liu, Hui Lu, Liyang Chen, Guiping Zhong, Jiakui Li, Shiyin Kang, Zhiyong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2607.10191)  

**Abstract**: Generative streaming models for Target Speaker Extraction (TSE) commonly exhibit a quality--intelligibility trade-off, wherein naive optimization for perceptual audio quality tends to degrade speech intelligibility, and conversely. We reveal that this trade-off arises not from the constraints of streaming architectures, but from an inappropriate choice of optimization anchor. Directly optimizing against audio quality metrics induces catastrophic reward hacking, where content critical to pronunciation and intelligibility is systematically erased to maximize a proxy score. To break this bottleneck, we propose two complementary improvements: an enlarged Conformer convolution kernel for richer local spectro-temporal modeling, and WavLM-anchored Direct Preference Optimization (DPO) fine-tuning strategy. DPO preference pairs are ranked by WavLM cosine similarity, a deep acoustic feature encoding both phonetic structure and speaker identity, providing an optimization anchor that resists hacking. Under a 560 ms streaming chunk size, the proposed method achieves a 10.9% relative intelligibility improvement (word error rate: 0.138 to 0.123), with marginal simultaneous gains in audio quality and speaker similarity. 

---
# PhysMRV: Physical Memory Retrieval and Verification for Physics Plausibility Reasoning 

**Authors**: Wenyuan Wang, Lianyu Hu, Hao Wang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.10190)  

**Abstract**: Video-language models (VLMs) have achieved remarkable performance on video understanding and visual question answering, yet they remain unreliable in reasoning about physical plausibility, where understanding object interactions, causal dynamics, and fundamental physical principles is essential. This limitation is particularly evident on challenging physical reasoning benchmarks, revealing a persistent gap in physical commonsense reasoning. To address this challenge, we propose PhysMRV, a training-free physical memory and verification framework for physical plausibility reasoning. Unlike retrieval-augmented VLMs that retrieve semantically similar videos as additional context, PhysMRV transforms training videos into a Hierarchical Memory Bank of structured physical knowledge comprising three complementary levels: scene descriptions capturing visual context, physical-event graphs modeling object interactions and causal structure, and physics-rule summaries distilling reusable physical principles and cues. During inference, PhysMRV retrieves physically relevant memories and leverages their structured physical evidence to guide a frozen VLM in verifying physical plausibility, requiring neither fine-tuning nor parameter updates. We evaluate PhysMRV on three challenging physical reasoning benchmarks, ImplausiBench, IntPhys2, and GRASP Level 2, across multiple state-of-the-art VLMs. Experimental results demonstrate consistent improvements over direct prompting across diverse VLMs and evaluation benchmarks, showing that structured physical memories provide an effective and scalable means of enhancing physical plausibility reasoning without additional training. 

---
# Automated Tensor Scheduling for Hybrid CPU-GPU LLM Inference on Consumer Devices 

**Authors**: Yangyijian Liu, Hongyi Ye, Mingyang Li, Wu-jun Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.10183)  

**Abstract**: Running large language models on consumer devices such as laptops and desktops is challenging because model weights often exceed GPU memory capacity, making offloading inference necessary to extend effective model capacity with CPU memory. Existing offloading systems, however, typically rely on coarse layer-level or expert-level scheduling, which overlooks substantial heterogeneity among tensors within the same layer and adapts poorly to changing hardware load conditions on such devices. This paper presents ATSInfer, a hybrid CPU-GPU inference system for consumer devices that performs offloading at tensor granularity. ATSInfer combines static tensor placement with load-aware dynamic transfer, and introduces asynchronous CPU-GPU coordination to efficiently schedule hardware storage, data movement, and computation across heterogeneous backends. We implement ATSInfer and evaluate it on representative consumer platforms using both dense and MoE models. Compared with existing systems, ATSInfer improves prefill throughput by up to 1.94$\times$ and decode throughput by up to 3.29$\times$, while also increasing GPU utilization and making more effective use of PCIe bandwidth. These results show that ATSInfer can substantially improve the user experience of local LLM deployment on personal consumer devices. 

---
# ActiveFly-Bench: Aligning Embodied Question Answering with Vision-Language-Action for Aerial Embodied Perception 

**Authors**: Weichen Zhang, Shiquan Yu, Yinan Zhu, Peizhi Tang, Shilong Ji, Zhiyuan Deng, Tianyi Lyu, Haoyang Wang, Xin Zeng, Chen Gao, Yong Li, Xinlei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.10180)  

**Abstract**: We introduce ActiveFly-Bench, the first benchmark to bridge cyberspace reasoning and physical-world interaction for UAV embodied perception. The benchmark decomposes active perception into three hierarchical tasks: Aerial Embodied Question Answering (Air-EQA), Observation Behavior Planning (OBP), and Fine-grained Language-guided UAV Control (FLUC), explicitly connecting high-level task understanding, behavior planning, and low-level control. The datasets are collected from both real-world and simulated outdoor environments for training and evaluation. We further develop ActiveFly, a closed-loop UAV agent that integrates visual-language reasoning with fine-grained control, and deploy it on a physical UAV platform. Experiments with representative VLMs and VLA models show that current UAV agents still struggle with behavior planning, viewpoint adjustment, and robust task completion in active perception. These results establish ActiveFly-Bench as a new testbed for embodied aerial intelligence. 

---
# Beyond Euclidean Clipping: Overcoming Exploration Collapse in LLM RL via Riemannian Isometric Policy Optimization 

**Authors**: Zhicheng Cai, Xinyuan Guo, Hanlin Wu, Mingxuan Wang, Wei-Ying Ma, Ya-Qin Zhang, Hao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2607.10169)  

**Abstract**: Reinforcement learning (RL) has become a dominant paradigm for enhancing LLMs' reasoning capabilities. However, RL algorithms with PPO-Clip are inherently limited by exploration collapse. Subsequent works remain primarily heuristic and fail to identify the essential cause of PPO-Clip's failure. This work reveals the fundamental flaw of PPO-Clip: it implicitly measures policy discrepancy using Euclidean metric, which is theoretically inconsistent with the intrinsic geometry on the policy Riemannian manifold. This geometric mismatch results in overly conservative updates in low-probability regions while aggressive in high-probability regions, ultimately collapsing exploration. To correct this geometric flaw, we propose Riemannian Isometric Policy Optimization (RIPO), which guarantees isometric policy updates on the Riemannian manifold, effectively balancing exploration and exploitation. We further show that RIPO achieves a favorable bias-variance trade-off, which stabilizes optimization. Extensive experiments demonstrate that RIPO significantly surpasses existing LLM RL algorithms across seven competition-level benchmarks (up to 60% improvement over GRPO on AIME24). 

---
# Transcript-Free Lightweight Detection of Alzheimer's Disease from Spontaneous Speech Using Handcrafted MFCC-Dominant Acoustic Biomarkers 

**Authors**: Rashin Gholijani Farahani, Azam Bastanfard  

**Link**: [PDF](https://arxiv.org/pdf/2607.10168)  

**Abstract**: It is still hard to find Alzheimer's disease (AD) early, especially when neuroimaging is expensive or tools that depend on language are not available. Spontaneous speech provides a non-invasive signal; however, numerous current methodologies depend on transcripts/ASR or computationally intensive deep models. We offer a simple, audio-only baseline for detecting AD using 176 Cookie Theft recordings from the DementiaBank Pitt corpus (88 AD, 88 controls). WebRTC voice activity detection (VAD) is used to separate speech from non-speech. We take out 99 hand-crafted acoustic-temporal features, including pause and fluency statistics, spectral/prosodic descriptors, and MFCC summaries with {\Delta} and {\Delta}{\Delta}. Evaluation is performed using a stringent speaker-independent GroupShuffleSplit,documenting performance across 30 iterations. A lightweight SVM with an RBF kernel gets an average AUC of 0.674 across runs. For example, a single split has an AUC of 0.742 and an accuracy of 0.657. We also present an exploratory compact-feature analysis utilizing a Top-20 subset ranked by Random Forest importance; since selection is not nested within training splits, these results may be overly optimistic and are not employed for primary conclusions (AUC 0.719). The results indicate that transcript-free spectro-temporal and fluency-related cues can facilitate speaker-independent Alzheimer's disease screening from raw audio, establishing a practical foundation for deployment-oriented research. 

---
# EmoStyle: Affective Conditioning of Style-Specialist Experts for Emotional Image Generation 

**Authors**: Dexiang Hong, Yijie Guo, Weidong Chen, Xinyan Liu, Zixuan Zou, Zhendong Mao, Yongdong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.10165)  

**Abstract**: Emotion-aware artistic image generation requires an image to match the input prompt, follow the specified artistic style, and convey the target emotion. In this challenge, the main difficulty is that the visual and affective attributes available in the training data are not explicitly provided at test time. Without these attributes, the generator has to decide not only what to depict, but also how the target emotion should be expressed through color, lighting, brushwork, composition, line, and layout. This creates a control gap between the available test prompt and the fine-grained conditions needed for emotion-aware artistic generation. To bridge this gap, we propose EmoStyle, a Z-Image-based framework that converts the input prompt into a structured generation state. An LLM reasoner first predicts affective cues (valence-arousal, dominant emotion, and therapeutic-effect labels) and an aspect-ratio decision. Instead of using these predictions only as additional prompt text, we encode the affective fields into an affective condition vector and inject it into the denoising blocks through AdaLN-style modulation. This allows the inferred control variables to directly guide the generation of intermediate features. Since emotional expression is also style-dependent, we further train a dedicated LoRA adapter for each artistic style bucket and select the corresponding expert during inference, enabling the same affective cues to be rendered with bucket-specific priors for color, texture, brushwork, and composition. Finally, a lightweight VLM-guided candidate selection step ranks the generated images based on prompt alignment, style consistency, emotional expression, and visual quality. In Track 1 of the AffectiveArt Challenge 2026, our USTC\_PI\_LAB\_TEAM submission achieved first place. 

---
# FlowPainter: Inpainting Optical Flow via Confidence-Guided Completion 

**Authors**: Yuang Meng, Chenyang Wu, Xianshun Liu, Chun-Le Guo, Zichen Liang, Lina Lei, Jie Liang, Hui Zeng, Chongyi Li, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.10140)  

**Abstract**: Existing optical flow methods broadly follow two paradigms: iterative optimization and diffusion-based estimation. Iterative methods, exemplified by RAFT, achieve high accuracy through recurrent refinement, but remain challenged by large displacements and complex motion. Diffusion-based methods introduce generative modeling and show promise in such ambiguous regions. However, existing diffusion models usually denoise the entire dense flow field from Gaussian noise, including simple regions where reliable motion can already be estimated by a lightweight network. This increases the denoising burden and may cause slow convergence and unstable training. To address this issue, we introduce FlowPainter, a diffusion-based optical flow framework that reformulates dense-flow generation as confidence-guided soft inpainting. FlowPainter employs a lightweight confidence-aware network to predict a rough flow and a pixel-wise confidence mask, distinguishing reliable simple regions from uncertain hard regions. The resulting simple-flow prior is used for confidence-based initialization and further injected into iterative denoising through confidence-gated residual guidance. With dynamically decaying guidance strength, FlowPainter stabilizes early denoising while preserving the flexibility of the diffusion model for late-stage detail refinement. Extensive experiments on public benchmarks, including Sintel, KITTI, and Spring, show that FlowPainter achieves strong accuracy under comparable training settings and converges more efficiently than existing diffusion-based optical flow methods, with notable gains on challenging benchmark splits. Our approach offers a practical way to integrate reliable discriminative priors with diffusion-based refinement for optical flow estimation. Our code is publicly available at this https URL. 

---
# LLMs as a Jury: Cross-Model Consensus Can Outperform Process Reward Models for LLM Reasoning 

**Authors**: Ning Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.10139)  

**Abstract**: Selecting the correct answer from a pool of candidate reasoning chains is the engine of test-time scaling, yet the standard selectors each carry a cost: self-consistency inherits the errors of the single model it resamples, and trained reward models need labeled data and transfer poorly off-distribution. We study a third signal, free at inference time: cross-model consensus, the degree to which independently trained models, each solving the problem once, agree on a final answer. We treat the panel as an LLM-jury, in which the structure of agreement, not any model's score of another, is the verification signal. Across seven benchmarks it selects correct answers better than self-consistency and far better than a model scoring its own candidates: on competition math it closes the entire gap to an oracle selector, while self-scoring closes almost none. The mechanism is error decorrelation: independently trained models err differently, so their wrong answers scatter while the correct one accumulates agreement. We make this precise with a parameter-free law, derived in closed form, that predicts consensus accuracy from three measured panel statistics to a mean absolute error of $0.03$ and exposes the method's ceiling: a shared-error floor where models share a misconception, near zero on math but non-trivial on science. Against four trained verifiers spanning discriminative, outcome, and generative reward models, the free LLM-jury matches the strongest inside their math training domain and is the top selector outside it. Cross-model consensus is thus a verifier we can characterize in advance: a law that says when to trust it, and a floor that marks where it cannot. 

---
# SALT-GNN: Handling Dense Neighborhoods in Anti-Money Laundering Graphs via Statistics-Aware Attention 

**Authors**: Lidia Losavio, Francesco Sovrano, Dario Fenoglio, Martin Gjoreski, Marc Langheinrich  

**Link**: [PDF](https://arxiv.org/pdf/2607.10131)  

**Abstract**: Money laundering threatens financial stability and exposes institutions to penalties, motivating automated detection. Because laundering schemes often emerge through relational patterns, graph neural networks (GNNs) are increasingly used for anti-money laundering (AML). Yet AML GNNs are typically evaluated with aggregate metrics such as overall F1 score, which hide an operational issue: high-activity recipient accounts concentrate many incoming transactions, making suspicious signals harder to isolate and costlier to investigate. We introduce a recipient-degree stratified evaluation that reports standard AML metrics across recipient-context density. Across three datasets (HI-Small, HI-Medium, and AMLSim-32k-5%), it reveals consistent degradation in dense recipient contexts, which we trace to three GNN characteristics: two known limitations that AML amplifies, i.e., (1) multiset non-discriminability and (2) cardinality blindness, and (3) an attention-specific effect: in dense neighborhoods, normalized attention attenuates weak but pattern-relevant multi-hop signals. Guided by this diagnosis, we propose SALT-GNN, a lightweight statistics-aware architecture that fuses degree-aware statistical aggregation with attention at each message-passing layer, so distributional and cardinality information shapes the node states used by subsequent attention steps. Ablations support fusion placement as a key factor in dense-context performance. On HI-Small and HI-Medium, SALT-GNN uses up to 77% fewer parameters than task-specific graph-transformer baselines while improving dense-context F1 score by 3-6 points; on AMLSim-32k-5%, it improves highest-degree F1 score by 16-20 points. The gains hold for both Transformer- and GAT-style attention, indicating that the benefit comes from where statistical and attentional evidence is fused rather than from a specific attention operator. 

---
# GAE: Graph-Augmented Evolution for Scientific Discovery via Reinforcement Optimization 

**Authors**: Xuanzhou Chen, Taoli Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2607.10127)  

**Abstract**: Evolutionary program search guided by Large Language Models (LLMs) has emerged as a powerful paradigm for automated scientific discovery. However, current approaches are fundamentally constrained by three bottlenecks: structurally blind parent selection, sparse whole-program evaluation rewards, and static mutation operators that fail to adapt during search. We present GAE (Graph-Augmented Evolution), a framework that resolves these limitations through a tightly coupled, three-pillar architecture. First, a relational graph neural network (GNN) parses programs into typed computation graphs, producing structure-aware embeddings. Second, an RL-optimized meta-controller leverages these embeddings to replace blind evolutionary sampling with a directed policy, dynamically selecting optimal parents and mutation directions based on reward history. Third, an online GRPO fine-tuning loop continuously updates the LLM mutation operator at test-time using group-normalized evaluation rewards, directly aligning the model's generation distribution with high-fitness structural edits. We evaluate GAE on a challenging scientific discovery task: symbolic regression for complex nonlinear oscillator systems. By transforming stochastic search into a directed, self-improving trajectory, GAE efficiently discovers closed-form physical equations, consistently matching or outperforming static LLM-driven baselines and achieving state-of-the-art out-of-distribution performance. 

---
# ML in a Box: Analyzing Containerization Practices in Open Source ML Projects 

**Authors**: Faten Jebari, Emna Ksontini, Amine Barrak, Wael Kessentini  

**Link**: [PDF](https://arxiv.org/pdf/2607.10126)  

**Abstract**: Containerization has become increasingly essential in the machine learning (ML) domain, providing reproducibility, portability, and environment consistency. While prior studies have analyzed Dockerfile structures and best practices, none have examined ML projects in depth to reveal how the iterative nature of ML workflows influences container footprint, build performance, and caching behavior. We present the first large scale empirical study of 1,993 ML related Dockerfiles, combining quantitative analysis of container roles in ML projects and build dynamics with a qualitative investigation of refactoring practices. Results show that containers serve distinct roles across training, inference, and infrastructure. Containers are typically large, averaging 10.27 GB in size, and require long build times of about 8.84 minutes. We find that 44.4% of commits trigger rebuilds, primarily due to context file changes (96.4%), with experimentation being the main motive behind those commits that initiate rebuilds. Despite partial cache reuse, 71% of rebuild work is wasted on redundant computation. From stable projects, we identify 7 recurring ML-specific Dockerfile refactoring patterns that improve build efficiency and reduce container footprint. 

---
# A Large-Scale Dataset of MCP Implementations on GitHub 

**Authors**: Benny Toeppe, Amine Barrak, Emna Ksontini  

**Link**: [PDF](https://arxiv.org/pdf/2607.10123)  

**Abstract**: The rapid emergence of the Model Context Protocol (MCP) has introduced a new standard for connecting large language models to external tools and services. Despite its rapid adoption in open-source development, systematic understanding of how MCP is implemented, structured, and maintained remains limited. This study presents the first large-scale, evidence-based dataset of real-world MCP implementation collected directly from GitHub. Using a hybrid pipeline that integrates the GitHub REST and GraphQL APIs with custom Python verification scripts, 3,238 candidate repositories were discovered, filtered, and validated through multi-stage evidence checks. Each verified project was classified by operational role (e.g., client, server, gateway) and exported in a reproducible JSONL schema. A manual review of a representative subset confirmed an overall precision of 83% at a 95% confidence level, and additionally revealed a set of repositories functioning primarily as educational samples, tutorials, or demonstration templates. A targeted exclusion rule was then applied to remove these non-operational repositories, resulting in a final dataset of 2,297 validated MCP projects. The analysis shows that Python and TypeScript dominate MCP development, with hybrid architectures emerging as the most common design pattern. By emphasizing transparent verification strategies, structured evidence tagging, and reproducible data organization, this work establishes a foundational benchmark for studying real-world MCP ecosystems and supports future research on integration, connectivity, and compatibility across the broader developer community. 

---
# When Data Imbalance Helps: Robust Generalization Through Shortcut Saturation 

**Authors**: Cheng-Ting Chou, Duc Binh Hoang  

**Link**: [PDF](https://arxiv.org/pdf/2607.10116)  

**Abstract**: We study robust generalization under spurious correlations: tasks where a shortcut feature is correlated with the true label in training but anti-correlated in an adversarial held-out split. Varying the spurious ratio $r$ (the fraction of training examples where shortcut = true label) and model capacity, we find a counterintuitive result: data imbalance promotes generalization in sufficiently capable models. On a synthetic task where the true label is sum parity of an integer sequence and the shortcut is the parity of the maximum-valued element, a 2-layer, 2-head transformer generalized (reached $100\%$ adversarial accuracy) in 0% of seeds at $r{=}0.50$ but 77% of seeds at $r{=}0.90$. The effect is absent in 1-layer models, where imbalance instead traps the model on the shortcut. Through mechanistic analysis -- gradient conflict dynamics, circuit evolution, and QK/OV circuit ablations -- we characterize a mechanistic pathway consistent with imbalance promoting generalization. 

---
# Cost of Reasoning in non-English Languages: A Case Study on Japanese 

**Authors**: Yuu Jinnai  

**Link**: [PDF](https://arxiv.org/pdf/2607.10114)  

**Abstract**: Reasoning Language Models (RLMs) achieve their strongest performance when they reason in English, the language for which reasoning-oriented training data is most abundant. However, reasoning trace is a clue for model interpretability and safety, and useful in practice for both the model users and for model developers. Thus, it is desirable to be able to develop a model that reasons in a language of the user's choice, while still maintaining strong reasoning performance. To this end, we study the feasibility of training a model that reasons in Japanese. We develop a Japanese-reasoning variant of Qwen-3-Swallow-8B, which is a Japanese LLM continually pretrained from Qwen-3-8B, with GRPO and evaluate it across coding, math, and science benchmarks. The study shows that reasoning-language control is feasible by training a Japanese continually pretrained model with GRPO. However, its performance is at best on par with strong English-reasoning baselines on several benchmarks. We also evaluate the trained model on Japanese cultural benchmarks and observe that the model's performance is worse than the baseline models, suggesting that the reasoning in Japanese does not immediately improve performance on culturally relevant tasks for free. 

---
# Minionese: Comprehensive Benchmark and Mechanistic Study of Multilingual LLM Safety 

**Authors**: Chigozirim Ifebi, Brent Kong, Ayushi Mehrotra  

**Link**: [PDF](https://arxiv.org/pdf/2607.10112)  

**Abstract**: Safety alignment in large language models remains brittle across languages: prompts reliably refused in English can elicit harmful compliance in non-English and low-resource settings. We introduce \textsc{Minionese}, a multilingual jailbreak benchmark spanning 18 languages, 4 resource tiers, and 4 perturbation types (standard translation, code-switching, transliteration, and translationese), paired with a geometric mechanistic analysis of refusal failure across language tiers. We show that each attack type produces a distinct vulnerability profile: transliteration vulnerability is mediated by script identity, code-switching maintains effectiveness through the lowest-resource tier, and a sharp safety regime transition between Tiers 2 and 3 is consistent across all models. Mechanistically, low-resource jailbreaks succeed by routing harmful content through a geometrically misaligned subspace that projects insufficiently onto the refusal directions, leaving the refusal mechanism intact but untriggered. These findings show that English-only safety evaluations are insufficient; they require accounting for script family, perturbation type, and per-language alignment coverage. The benchmark and analysis code is at this https URL. 

---
# Adaptive Model Compression (AMC): Saliency-Driven Resource Allocation for Ultra-Low-Power Transformer Inference 

**Authors**: Jiayin Hu, Kai Yuan, Vanessa Hu, Xuetao Yin, Jianhua Li, Sean Suchter  

**Link**: [PDF](https://arxiv.org/pdf/2607.10109)  

**Abstract**: Deploying large-scale transformer models on resource-constrained edge devices remains a challenge due to the high energy and memory overhead inherent in static inference, which processes simple and complex tokens with uniform intensity. To address this, we propose Adaptive Model Compression (AMC), a saliency-driven framework that dynamically allocates hardware resources based on token importance. By implementing a multi-tier architecture, our system identifies critical high-saliency information for full-precision processing while aggressively reducing the rank and bit-width of less significant data. Experimental results demonstrate that AMC achieves a 59.2% reduction in system energy and a 2.24x increase in throughput on 45nm CMOS hardware. This approach effectively extends the battery life of mobile devices by utilizing high-definition compute only where necessary, maintaining robust performance with a marginal 3.6% accuracy trade-off. 

---
# Efficiently Adapting Spoken Language Models for the Singaporean Context 

**Authors**: Ng Jia Sheng Jason  

**Link**: [PDF](https://arxiv.org/pdf/2607.10092)  

**Abstract**: Spoken language models (SLMs) unify speech perception and reasoning, but adapting them to sensitive domains is underexplored, especially when the original training data is inaccessible and the use case demands multilingual, spoken-query interaction. We adapt an open-source SLM to the Singaporean Home Team context across five speech tasks in Singapore's four official languages, combining LoRA fine-tuning, a surrogate text-QA dataset that guards against catastrophic forgetting, and a multi-task objective that adapts the CoBa reweighting scheme to speech. We also build HTD-multilingual-QA, a 504,853 sample multilingual QA dataset in text and spoken form. The resulting HT-Moonstone (5B) matches or outperforms SLMs up to 7x its size on most tasks, attains the best accent and gender recognition among all models evaluated, and loses under 2\% of its original speech QA ability. 

---
# Quantum Circuit Vision: Cost-Aware Evaluation of Visual AI Agents for Quantum Code Generation 

**Authors**: Dongping Liu, Aoyu Zhang, Luyao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.10057)  

**Abstract**: Can AI agents visually comprehend quantum circuit diagrams and generate verified executable code--and at what cost? We present Quantum Circuit Vision, a cost-aware evaluation framework for multimodal AI agents on quantum circuit visual understanding. We construct a 132-circuit benchmark spanning 13 categories ($1$--$10$ qubits) with executable Amazon Braket code and unitary-fidelity verification. Evaluating three frontier Claude-family models at different capability-cost tiers with $n=5$ repeated trials, we find that the mid-tier model (Sonnet 4.6, $1.30\times$ credits) offers the most favorable balance on the cost-accuracy frontier: 91% pass rate on the core subset at 18% of the per-call cost of the strongest model (Opus 4.6), whose accuracy advantage is not statistically significant (paired $t$: $p=0.083$). Logistic regression confirms that circuit depth--not qubit count--is the primary predictor of failure ($p<0.001$). Chain-of-thought prompting shows no statistically significant effect (all $p>0.18$, $n=5$), suggesting that visual pattern recognition outweighs explicit reasoning strategy for structurally coupled diagrams. We propose a cascade routing strategy (cheap $\rightarrow$ expensive models) that achieves 84% accuracy at 38% of single-model cost, demonstrating that model routing dominates prompt engineering as a cost lever. We release QCV-Dataset (132 circuits, 5 modalities, 1,931 files) on Hugging Face Hub as an open evaluation infrastructure with structured metadata for discoverability, interoperability, and responsible AI documentation, and all evaluation code, cost logs, and verification scripts on GitHub for full reproducibility. 

---
# Geometric mean-based pairwise comparison method with the reference values -- statistical approach 

**Authors**: Konrad Kułakowski, Jacek Szybowski  

**Link**: [PDF](https://arxiv.org/pdf/2607.10038)  

**Abstract**: For many years, the pairwise comparison method has been widely used for decision-making involving experts. The best-known example of this method is the Analytic Hierarchy Process (AHP). In this now classic approach, the weights of alternatives are calculated using the principal eigenvector of the comparison matrix. In this paper, we present a statistical view of the pairwise comparison method, using reference values and the geometric mean to calculate alternative priorities. Thanks to this approach, we can simultaneously capture both the phenomenon of inconsistency in pairwise comparisons and the preference distance between alternatives. In this paper, we define indicators that measure the quality of the obtained weight vector, which, thanks to the statistical approach, have a clear and intuitive interpretation. 

---
# Robust, Scalable Detection of Text Containment in Large Web-Crawled Corpora 

**Authors**: Lars Henry Berge Olsen, Pierre Lison, Martin Jullum, Mark Anderson  

**Link**: [PDF](https://arxiv.org/pdf/2607.10020)  

**Abstract**: We present FindMyText, an open-source Python package designed to efficiently assess whether a given text appears, in part or in full, within a text corpus. The tool builds on prior techniques for document fingerprinting, but extends them with a novel mechanism to explicitly capture sequences of matching fingerprints. By identifying such chains, the tool can more reliably detect near-verbatim copies of a given text rather than mere textual similarities. This makes FindMyText particularly suited for verifying the presence of copyrighted material in a corpus. Leveraging a distributed, disk-based indexing framework, the system scales to large web-crawled datasets. Using a new benchmark for evaluating text containment methods, we show that FindMyText outperforms alternative approaches across three datasets (ArXiv papers, Wikipedia, and generic web content). 

---
# Beyond Bayesian Nash: Learning Minimax-Regret Equilibria for Adversarial Team Games under Asymmetric Information 

**Authors**: Naman Aggarwal, Jonathan P. How  

**Link**: [PDF](https://arxiv.org/pdf/2607.09993)  

**Abstract**: Adversarial team games (ATGs) with asymmetric information, such as adversarial path-finding, goal search, and reachability games on graphs, require strategies that are robust to hidden opponent types, such as a hidden goal flag, and to deception. Under asymmetric information, deception is seen as strategic shifts in the type distribution such that the omniscient opponent can collude with Nature and condition its play on the observed type. Existing risk-neutral solution concepts, such as Bayesian Nash equilibrium (BNE), are sensitive to distribution shifts, while distributionally robust approaches provide guarantees only within a prescribed ambiguity set. To address these limitations, we introduce Probabilistically Robust Minimax-Regret Equilibrium (PR-MRE), a novel equilibrium concept that combines the distribution-free robustness of minimax-regret reasoning with probabilistic information from a nominal type distribution. PR-MRE minimizes worst-case regret over a high-confidence subset of the type space, providing protection against strategic redistribution of probability mass while avoiding the conservatism of fully distribution-free approaches. We show that, for normal-form Bayesian games, PR-MRE can be formulated as a robust bilinear program and derive a tractable semidefinite relaxation. We then adapt this relaxation into a novel meta-solver within a robust double-oracle framework, PRMRE-PSRO, enabling population-based learning of approximate PR-MRE strategies via deep reinforcement learning best responses. Experiments on graph-structured adversarial team games demonstrate that PR-MRE discovers strategies with substantially improved worst-case performance across hidden types compared to risk-neutral equilibrium solutions, resulting in more robust behavior under strategic distribution shifts. 

---
# An LLM-powered Agentic Recommendation System for Connected TV Content Discovery 

**Authors**: Lei Shi, Di Wang, Harry Tran, Helsing Xu, Yuchen Lu, Dhara Ghodasara, Wilson Chaney, Xueting Liao, Jerry Yu, Huayu Ding, Mingze Gao, Shike Mei, Shuo Tang, Zhe Zhang, Jianming He, Abhishek Kumar, Haotian Wu, Hamed Firooz, Li Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.09988)  

**Abstract**: Recommendation systems, from traditional multi-stage to recent unified generative architectures, face challenges in incorporating diverse contextual signals, such as trending topics, breaking news, cultural events, and cross-surface user activities, into their ranking pipelines. These systems are designed to consume structured behavioral signals with consistent schemas, and lack the reasoning capability to naturally process unstructured or heterogeneously formatted contextual information. Incorporating such signals typically requires feature engineering, bespoke data pipelines, and carefully tuned heuristics. In this paper, we present an LLM-powered agentic recommendation system designed for Connected TV (CTV) content discovery that addresses these limitations. Our system leverages the reasoning capabilities of large language models to naturally process and synthesize diverse signals across varying schemas and structures, eliminating much of the manual integration inherent in traditional ranking and retrieval systems. Recognizing that current LLM-based solutions still fall short of traditional machine learning models in several recommendation tasks, including retrieval efficiency, personalization precision, and scalability, we adopt an agentic architecture that orchestrates specialized components, allowing each sub-task to be handled by the most suitable method, whether LLM-based or traditional ML. The main contribution of this work is our engineering approach to successfully overcoming the practical limitations of enabling LLM for recommendation, particularly inference latency. We share insights from our work and discuss the trade-offs and lessons learned in building a hybrid system that combines the flexibility of LLMs with the performance of established recommendation techniques. 

---
# A Production-Oriented Framework for Evaluation of SFX Generation 

**Authors**: Mélodie Desbos, Yara Bahram, Eric Granger, Mohammadhadi Shateri  

**Link**: [PDF](https://arxiv.org/pdf/2607.09973)  

**Abstract**: Industrial sound design requires audio generation systems that not only produce realistic audio, but also preserve the perceptual identity of a reference, support controllable variation, and remain efficient for practical workflows. Existing evaluations are usually tied to text-to-audio (TTA), unconditional, or task-specific settings, limiting assessment for reference-guided sound effects (SFX) variation. To address this gap, we present a production-oriented evaluation framework for structured comparison of heterogeneous audio generation and editing methods. Our framework identifies nine production requirements and explicitly accounts for differences in model capabilities, enabling comparison under a common production objective. A two-stage protocol is introduced: (1) a reference-guided audio-to-audio (ATA) variation task, in which all methods are evaluated under the same ESC-50 SFX adaptation setup, and (2) capability-specific analyses of native operations such as SFX morphing, temporal and energy alignment, inpainting, and targeted editing. This framework combines objective metrics (including FAD, ImageBind-based reference alignment, and diversity across generated variants), together with a human study of perceptual identity preservation and transient diagnosis. Our study reveals complementary strengths and trade-offs across baselines for different production needs. Among the full-generation baselines evaluated under a shared ATA setting, AudioX provides the strongest overall trade-off between reference alignment and diversity while still supporting SFX morphing. Other baselines remain most suitable for specific editing operations. Our framework establishes a structured evaluation and decision protocol for reference-guided SFX variation and provides a practical basis for designing future unified industrial audio generation pipelines. Audio demos are on the accompanying web page. 

---
# Learning in Curved Weight Space:Exponential-Linear Weight Reparameterization for Improved Optimization 

**Authors**: Ethan Smith  

**Link**: [PDF](https://arxiv.org/pdf/2607.09967)  

**Abstract**: Many neural networks operations have a multiplicative nature rather than additive: halving or doubling a norm are analogous relatively but require unequal optimization distances when taking linear steps. Adaptive optimizers such as Adam normalize updates per coordinate, but update steps remain additive; weights with very different magnitudes receive similarly sized absolute changes, producing very different relative perturbations. We introduce \textbf{\method} (\textbf{\methodshort}), a weight reparameterization for neural networks that combines a sign-aware symmetric-exponential pathway with an identity-like linear pathway. The symmetric-exponential pathway is near-linear for small raw weights but increasingly curved at larger magnitudes. Additive updates in logarithmic space map to magnitude-proportional changes in effective weight space. The linear pathway provides a direct route through the transform that we hypothesize stabilizes optimization, while learnable scale, curvature, and offset parameters control balance between pathways and the curvature of the exponential pathway. These components create a curved parameter-space geometry that empirically improves speed of loss descent over standard linear parameterization. We also identify a useful \emph{mismatched initialization}: raw weights are chosen so a symmetric version of the transform matches Xavier statistics, but training uses an asymmetric forward transform that leaves positive weights at full strength while making negative weights smaller in magnitude; in small-model ablations, this improves early optimization and may act as a form of symmetry breaking. We train transformers on OpenWebText over nine width$\times$depth configurations, \methodshort reaches matched validation loss in 1.32--1.49$\times$ fewer training steps, with the largest widths seeing the biggest gains. 

---
# Workload-Driven Optimization for On-Device Real-Time Subtitle Translation 

**Authors**: Tsz-To Wong  

**Link**: [PDF](https://arxiv.org/pdf/2607.09957)  

**Abstract**: This report studies on-device English-to-Traditional-Chinese subtitle translation for Taiwan under short inputs, short outputs, batch-size-one inference, low latency, and privacy constraints. These conditions limit the value of optimizations designed for long-context or high-throughput language-model serving.
Starting from LMT-60-0.6B, preliminary profiling suggests that vocabulary projection becomes a more important decode-time cost after GGUF quantization reduces the relative cost of Transformer blocks. We replace the original 151k-token vocabulary with a 64k-token subtitle-domain tokenizer, migrate the embedding space, and adapt the model through embedding calibration followed by full supervised fine-tuning.
On a fixed 500-example subset of the OpenSubtitles2024 test set, the LocalSubs achieves a 59.2% tie-excluded win rate against Google Translate under GPT-4o pairwise judging. Performance is strongest on short cues and declines as cue length increases. Preliminary Apple M2 Metal measurements on a 64k-vocabulary model show a 1.63$\times$ speedup over a 151k-vocabulary profiling baseline. The raw benchmark configuration is incomplete, so the latency result is treated as preliminary. 

---
# A Foundation Model for Multimodal Event Sequences in Financial Applications 

**Authors**: Nikita Rusakov, Vladislav Meshkov, Konstantin Zorin, Gleb Zaripov, Alexander Uglov, Alexey Vasilev, Anton Klenitskiy  

**Link**: [PDF](https://arxiv.org/pdf/2607.09955)  

**Abstract**: Predictive modeling is a core component of modern financial services, where a wide range of tasks are traditionally addressed using separate models trained on manually engineered tabular features. This task-specific approach limits reuse and makes it difficult to fully exploit heterogeneous data sources such as transaction histories and digital interaction signals. In this paper, we present an approach based on pretraining a foundation transformer model on multimodal sequences of user events. Events from multiple data sources are unified into a single chronological sequence, enabling early fusion of heterogeneous modalities and learning of general-purpose representations via a next-event prediction objective. These representations are combined with existing engineered user features, on top of which lightweight neural models are trained for multiple downstream tasks. The proposed system outperforms traditional task-specific models while reducing development overhead. The approach was deployed in production at one of the biggest banks in Eastern Europe, resulting in measurable improvements in business metrics. 

---
# A Knowledge-Based Multi-Agent Framework for Security Control Recommendation 

**Authors**: Carolina Fernández-Martínez, Shuaib Siddiqui, Vanesa Daza  

**Link**: [PDF](https://arxiv.org/pdf/2607.09954)  

**Abstract**: Hardening IT on-premises environments can be a daunting task for teams without access to adequate cybersecurity expertise. In this regard, Decision Support Systems (DSS) with embedded expert knowledge can assist users by guiding them with security recommendations to meet their objectives. This work proposes a Security DSS that recommends security control sub-families given minimal user requirements indicating coverage of different security dimensions. It leverages a curated, unified dataset from both well-known Information Security (InfoSec) and academic sources. This DSS is defined as a non-zero-sum, simultaneous game that is grounded in a Multi-Agent Influence Diagram (MAID) model and explores the decision space over 7 security dimensions or agents, using no-regret online learning to ultimately find the security control sub-families that best fit the requirements while incurring minimal under- and over-provisioning of security resources. This work was validated in terms of performance and accuracy, among others, for varying dataset sizes. It shows exceptional satisfaction coverage results of 99% when using as little as ~65% of the SW-implementable security controls, running in 1.2-35.7 seconds; and more moderate coverage results of 73%-77% when using ~29% of the controls, resolving in 0.8-13.8 seconds. 

---
# SMETA-ZSL:Semantic Meta-Alignment for Zero-Shot Threat Classification 

**Authors**: Ivan Alejandro Montoya Sanchez, Anantaa Kotal, Aritran Piplai  

**Link**: [PDF](https://arxiv.org/pdf/2607.09936)  

**Abstract**: Cybersecurity systems must adapt rapidly to emerging threats. However, labeled data for new threat categories is unavailable when those threats first appear. Generalized zero-shot learning offers a natural solution by enabling recognition of unseen classes through auxiliary semantic knowledge rather than labeled examples. Large language models are particularly promising in this setting because they can convert unstructured CTI reports into semantic prototypes for emerging threats. However, applying language-driven zero-shot learning to cybersecurity is difficult due to strong semantic overlap between threat descriptions, heterogeneity between behavioral attributes and text, severe class imbalance, and open-set conditions where unseen threats are unknown during training. We propose SMETA-ZSL, that learns semantic prototypes from overlapping language descriptions through contrastive finetuning, aligns behavioral features through episodic meta-learning and knowledge distillation, and performs adaptive routing for generalization across seen-unseen classes. Across 7 benchmarks, SMETA-ZSL delivers the strongest overall generalized zero-shot performance under the strictest inductive setting, surpassing prior methods by 10.8 points on average, with gains up to 18.1 points. Github:this https URL 

---
# Faithful by Design: Evaluating and Improving LLM-Generated Clinical Trial Summaries for Multi-Stakeholder Audiences 

**Authors**: Robert Williams  

**Link**: [PDF](https://arxiv.org/pdf/2607.09932)  

**Abstract**: Large language models are increasingly used to summarize clinical trial results for healthcare providers, patients, and payers, but their tendency to hallucinate poses significant risks in this high-stakes context. This study introduces a benchmark evaluation framework for measuring the faithfulness of LLM-generated clinical trial summaries across three stakeholder audiences. The framework consists of 200 stratified trials drawn from the Aggregate Analysis of this http URL database, evaluated using audience-specific prompt templates and a six-dimension faithfulness annotation schema. Baseline measurements were established for GPT-4o, Claude Sonnet 4.6, and Gemini 2.5 Flash across 1,800 generated summaries scored using a cross-encoder natural language inference (NLI) model. Unsupported Claims was identified as the dominant failure mode across all three models, with a mean annotation score of 1.55 out of three. A knowledge-graph-augmented retrieval system was developed and evaluated against the baseline, producing statistically significant improvements in NLI-based faithfulness scores (entailment +0.0125, faithfulness +0.0130, p < 0.0001). Improvement pathways were model-dependent, with GPT-4o improving primarily through contradiction reduction while Claude Sonnet 4.6 and Gemini 2.5 Flash improved through increased entailment. 

---
# Do These Violent Delights Have Violent Ends? Measuring the Post-Merge Fate of Agentic Code 

**Authors**: Chunqiu Steven Xia, Courtney Miller  

**Link**: [PDF](https://arxiv.org/pdf/2607.09902)  

**Abstract**: Agentic coding tools are increasingly used to make autonomous repository-level changes to real-world projects. Prior work has largely evaluated these contributions at the pre-merge stage, through outcomes such as pull request acceptance and review effort. Far less is known about what happens to agentic code post-merge. Yet merge success alone does not reveal whether a contribution will remain stable or require bug fixes and other corrective maintenance downstream. We conduct a longitudinal empirical analysis of agentic and human contributions across 182 repositories, tracking their post-merge fate over time, characterizing the intent of subsequent modifications, and analyzing the defects and vulnerabilities they introduce. While the overall maintenance rates are similar, agentic contributions require significantly higher rates of corrective maintenance and introduce more security weaknesses and dependency vulnerabilities. We also find statistically significant evidence that agentic maintenance burden is associated with repository characteristics. In particular, each 10 percentage-point increase in a project's no-review rate is associated with roughly a 6% increase in agentic maintenance burden on average. As coding agents become pervasive in software development, our findings highlight the need to evaluate and design agentic tools not only to produce mergeable changes, but to produce contributions that remain secure and maintainable. 

---
# Next-Dense-Stride Prediction for Multimodal Autoregressive Visual Modeling 

**Authors**: Chicago Y. Park, Jialin Mao, Xiaojian Xu, Taha Kass-Hout, Ulugbek S. Kamilov, Cao Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2607.09892)  

**Abstract**: We introduce DenseAR, a new generative paradigm that reformulates autoregressive image generation as coarse-to-fine next-dense-stride prediction using a compact single-scale tokenizer. Our key insight is that traversing a single-scale latent grid with progressively denser strides naturally captures the transition from global structure to fine detail. This addresses two limitations of existing autoregressive models at once: the slow inference of raster-order autoregression, which DenseAR avoids by predicting multiple tokens in parallel, and the heavy cost of multi-scale approaches, which need long, multi-resolution token sequences to achieve coarse-to-fine prediction. Building on our efficient framework and the flexibility of autoregressive modeling, we further extend DenseAR to a unified model that handles multiple modalities and imaging tasks within a single backbone. We validate DenseAR on both medical and natural images. On multi-contrast brain MRI, a single DenseAR model unifies cross-modal translation, modality-conditioned generation, and tumor segmentation, while remaining competitive with task-specific methods. On ImageNet, DenseAR improves class-conditional generation quality (FID and IS) over both a single-grid baseline without stride ordering and a multi-scale tokenizer-based baseline. 

---
# What You Train Is What You Get: Gender Bias, Training Composition, and Post-Hoc Mitigation in Audio Deepfake Detection 

**Authors**: Aishwarya R. Fursule, Vamshi Nallaguntla, Shruti Kshirsagar, Anderson R. Avila  

**Link**: [PDF](https://arxiv.org/pdf/2607.09891)  

**Abstract**: Audio deepfake detection models determine whether speech is genuine or artificially generated, but high overall accuracy can mask substantial performance disparities across demographic groups. In this work, we investigate gender bias in audio deepfake detection using the ASVspoof5 dataset. We use ASVspoof5 under a controlled custom split designed to isolate gender-composition effects. We train attack-specific models on nine training sets with different gender compositions, ranging from female-only to male-only. We use a ResNet18 classifier with LogSpectrogram and WavLM-Base+ features, and we evaluated six post-hoc threshold calibration methods. Experimental results show that training data composition strongly predicts bias direction, with the underrepresented gender performing worse at test time. WavLM-Base+ features are shown to produce gender performance gaps 3.0 to 4.3 times larger than LogSpectrogram under identical training conditions, and balanced training is found to reduce LogSpectrogram bias but leave WavLM bias largely intact. Moreover, all six calibration strategies, including Oracle calibration with full test-set label access, leave the Equal Error Rate gap unchanged at 1.317 pp, confirming that threshold adjustment cannot correct underlying score distribution disparities. Overall, these findings suggest that gender fairness in audio deepfake detection must be addressed at training time, as post-hoc methods can only partially mitigate the resulting disparities 

---
# Remembering Distinct Items, Not Tokens: A Learnable Dirichlet-Process Cache Between State-Space Models and Attention 

**Authors**: Siddharth Pal, Viktoria Rojkova  

**Link**: [PDF](https://arxiv.org/pdf/2607.09889)  

**Abstract**: Fixed-state sequence models compress an unbounded past into a bounded state, which caps their associative recall at roughly the state dimension; attention escapes the cap by keeping a key-value entry for every token, at quadratic compute and a cache that grows with the sequence. We study the middle ground: a sparse cache that allocates a slot only when an input is novel, so its size tracks the number of distinct items rather than the number of tokens. The allocation rule is the DP-means clustering rule, the small-variance limit of a Dirichlet-process mixture, used not as latent-variable inference but as the key-value memory operator for a deep recurrent backbone. We develop it in two forms, a static cache with a fixed concentration and a surprise-adaptive variant whose concentration follows the recent novelty rate. On a controlled associative-recall benchmark with redundancy we show that the cache matches full-attention recall while storing only the distinct items, that it dominates a fixed-budget eviction cache on the recall-versus-size frontier, and that on a state-space backbone it answers both a recall query and a long-range aggregate at the lowest memory of any model tested. The allocation is learnable end to end: a two-parameter novelty-threshold gate trained on the task loss alone recovers the rule exactly, whereas an over-parameterized gate fails, so the operative ingredient is the inductive bias rather than capacity. The evidence is a family of controlled mechanism studies at modest scale, with the distinct-items property confirmed on four real streams (recommendation, systems logs, clinical events, and insurance claims); a real-backbone, real-corpus language validation is pursued in a companion study. 

---
# CLIR-Bench: Benchmarking Multimodal Question Answering over Irregular Clinical Time Series 

**Authors**: Frank Nie, Ethan B. Liu, Yuan Zhu, Loe Yan, Wei Fan, Jindong Han  

**Link**: [PDF](https://arxiv.org/pdf/2607.09880)  

**Abstract**: Clinical time series are central to patient monitoring, risk assessment, and clinical decision support. However, they are often sparse, irregularly sampled, and asynchronous, making it difficult for models to identify the temporal evidence required for clinical Question Answering (QA). Existing benchmarks primarily focus on regularly sampled time-series QA or medical QA over static data, and therefore rarely assess whether models can faithfully ground their answers in irregular temporal observations. To fill this gap, we introduce CLIR-Bench, a benchmark for irregular clinical time series QA constructed from de-identified ICU records through a principled four-stage pipeline. CLIR-Bench contains 6,600 QA instances spanning 11 clinical variables, organized into four capability dimensions and 11 tasks. Each question is linked to explicit temporal evidence and task-specific answer derivation rules, enabling evaluation of both answer accuracy and evidence use. Experiments show that existing generalist models struggle to retrieve and reason over sparse clinical evidence, highlighting the need for stronger irregular time-series reasoning methods. Our code and data are available at this https URL. 

---
# Prompting-MammAlps: Fine-Grained Text-to-Video Retrieval for Camera-Trap Data 

**Authors**: Valentin Gabeff, Baptiste Maquignaz, Jennifer Shan, Sepideh Mamooler, Gencer Sumbul, Blair Costelloe, Devis Tuia, Alexander Mathis  

**Link**: [PDF](https://arxiv.org/pdf/2607.09876)  

**Abstract**: Automatically retrieving videos from large camera-trap datasets remains challenging. Text-to-Video retrieval (TVR) methods based on large video-language models (VLMs) have potential to retrieve events of interest by describing them with simple text queries. However, current methods often lack spatiotemporal understanding and do not generalize well to ecological data. In this work, we introduce Prompting-MammAlps, the first camera-trap TVR benchmark, and propose a fine-grained and interpretable TVR method. Specifically, we trained a vision transformer to perform spatiotemporal action localization, and convert its output to structured text, describing each video. Independently, ethology-inspired queries are processed by a Large-Language Model (LLM) based coding agent to parse the structured text per video and retrieve videos accordingly. We harnessed the LLM to use functions from a custom parsing library to minimize the risk of LLM hallucinations and to improve method interpretability. This retrieval approach applied on the Prompting-MammAlps benchmark achieved a set-based F1-score of 34\% on a test set of 135 ecologically-relevant queries and 775 candidate videos. In comparison the best zero-shot VLM achieved a F1-score of 18\%, while also lacking interpretability. Project page: this https URL 

---
# Robo-ValueRL: Reliable Value Estimation for Offline-to-Online Reinforcement Learning 

**Authors**: Wenke Xia, Pei Ren, Wenbo Yu, Yizhuo Zhang, Jifan Li, Yixue Zhang, Yinuo Zhao, Qingyang Gao, Jianlong Fu, Jian Tang, Ji-Rong Wen, Zhengping Che, Di Hu  

**Link**: [PDF](https://arxiv.org/pdf/2607.09866)  

**Abstract**: Offline-to-online reinforcement learning is promising for generalizable robotic manipulation, yet its full-stack complexity obscures reproduction and diagnosis. Within such systems, value estimation plays a central role in prioritizing heterogeneous data for policy improvement. Despite its importance, the central question remains underexplored: how value-function reliability shapes policy optimization in offline-to-online reinforcement learning. To answer this question, we propose Robo-ValueRL, a unified framework that enables reliable value estimation and systematically traces its downstream effects on policy pretraining and online improvement. Concretely, Robo-ValueRL learns a history-conditioned value estimator and evaluates its reliability through global-progress and local-preference metrics. These resulting value estimates are propagated into quality-conditioned consistency-policy pretraining and a residual adaptation module on online rollouts, providing a unified testbed for analyzing how value reliability shapes downstream policy performance. Across 240 hours of offline demonstrations and over 3,000 online rollout trajectories, our extensive experiments show that downstream performance is strongly associated with value reliability. Reliable value functions provide better action-quality estimates, allowing value-guided offline RL to scale more effectively than quality-agnostic behavior cloning, and stabilize online improvement by prioritizing high-quality rollout data. Integrating reliable value guidance through offline pretraining with online improvement, our system achieves 86% success on millimeter-level precise chip insertion and 84% on generalizable block disassembly. We hope these findings highlight the importance of value-guided data utilization for effective policy improvement from heterogeneous robotic experience. 

---
# Towards Objective Dysgraphia Detection: A Multi-Branch Deep Learning Approach for Online Handwriting Analysis 

**Authors**: Lydia Ouhib, Yassine Ouzar, Zoé Pinseel, Stéphane Bouilland, Mehdi Ammi  

**Link**: [PDF](https://arxiv.org/pdf/2607.09826)  

**Abstract**: Dysgraphia is a specific learning disability that is prevalent among school-age children. It affects handwriting coherence, quality, fluency, and legibility, often hindering academic achievement and early learning development. This motor coordination disorder is typically diagnosed through subjective assessments based on clinician observation, which can be timeconsuming and prone to variability. In this paper, we introduce a deep learning-based framework for objective dysgraphia detection using online handwriting data captured via digitizing tablets. The proposed framework relies on two complementary branches: the first pipeline extracts both handcrafted and embedding-based kinematic features directly from raw temporal signals, while the second leverages image-based representations of the temporal signals generated using continuous wavelet transforms (CWT) and Gramian Angular Fields (GAF). The resulting features are then fused to leverage the complementary strengths of both representations. The four representations were evaluated separately and jointly using the publicly available DiaGraMo dataset, showing that the fusion of GAF, MOMENT, and hand-crafted kinematic features outperforms each individual representation, as well as other fusion schemes. These findings highlight the potential of the complementarity of image and signal based representations for more objective dysgraphia detection. 

---
# More Structure, Not More Capacity: Object-Centric Representations for Visuomotor Imitation Learning 

**Authors**: Yi Li, Alexandre Chapin, Liming Chen, Jan Peters, Alap Kshirsagar  

**Link**: [PDF](https://arxiv.org/pdf/2607.09825)  

**Abstract**: Robotic manipulation policies rely on pre-trained vision models that give either a global scene embedding or a dense patch grid. Both mix task-relevant and task-irrelevant features. Object-centric slot representations are a structured alternative: they group features into a few per-object slots. We test what this structure buys on ManiSkill3 PickCube-v1, with a frozen encoder and a held-out-seed evaluation. Holding the policy, goal token, rendering, and calibration fixed and changing only the encoder, a frozen object-centric SPOT representation (DINO ViT-B/16 + Slot Attention) reaches 55.0$\pm$2.9% success, 22.4% above a dense DINO global-feature baseline (32.6 $\pm$ 1.5%), with the same trainable policy and no encoder fine-tuning. More tokens alone do not help: a dense patch grid with 16x the tokens performs no better than the global feature. Adding an explicit 2D spatial goal and native-resolution rendering raises the full system to 68.7$\pm$4.2%, just below a privileged 3D-oracle upper bound (71.7$\pm$4.1%). An automated kinematic failure taxonomy then separates spatial-precision (Near-Miss) failures from object-tracking (No-Grasp) failures: spatial grounding reduces Near-Miss while leaving No- Grasp unchanged. The same taxonomy transfers to the harder StackCube-v1 and points to occlusion as the main bottleneck. 

---
# Memory-Conditioned Tool Calling for Camera-First Visual Agents 

**Authors**: Xiaofan Wu, Xi Zeng, Miaoxia Chen, Peishan Chen, Shuyan Li, Jiyun Yao, Hanyong Zhong, Jiahao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.09822)  

**Abstract**: Recognition tells an agent what is in an image; personal memory affects what is worth looking up next. In a camera-first setting the user can send only an image, so the agent must form the lookups. We study whether personal visual memory improves agent-side tool choice and tool arguments, and thereby more user-aligned multi-tool lookups. The design uses a three-layer personal visual memory (profile, short-term focus, observations) that is loaded on each turn to condition an LLM tool-calling loop under camera-first intake, and includes conflict-aware write-back intended to refresh the user model for later captures. On 800 images paired with synthetic memory blocks constructed for controlled ablation, removing the full three-layer memory block reduces tool-query relevance by 0.47 points absolute (4.21 -> 3.74 on a 5-point scale; 11.2% relative) and end-to-end utility by 0.082 absolute (0.842 -> 0.760; 9.7% relative). These results measure memory conditioning of tool policy under image-only intake with fixed synthetic blocks, not multi-session write-back from live user histories. 

---
# TS-Mask VLA: 2D Temporal-Spatial Masking for Vision-Language-Action Model with Effective Bridging 

**Authors**: Shengzhuo Yang, Ronghao Yu, Chuanjie Lv, Linpeng Peng, Hang Yu, Jie Ren, Jiajun Lv, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.09818)  

**Abstract**: Vision-language-action (VLA) models aim to understand natural-language instructions and visual observations, and to generate and execute corresponding actions as embodied agents. Recently, autoregressive token-based action generation has driven the development of many representative VLA models. However, this paradigm often reduces action generation to next-token prediction, thereby lacking explicit modeling of the spatiotemporal structure of action sequences and the disentanglement between vision-language representations and actions, which can limit performance in long-horizon and complex scenarios. In this paper, we propose TS-Mask VLA, a vision-language-action framework for robot manipulation. TS-Mask VLA is built upon two key designs: (1) a Discrete Diffusion Action Expert equipped with a Bridge Attention conditioning bridge, which enables multi-layer conditioning from the VLM and facilitates more accurate and stable action generation; and (2) a temporal-spatial 2D masking strategy for discrete action tokens that strengthens the model's understanding of cross-time dependencies and inter-dimensional coupling, leading to more structurally consistent action sequences. We conduct extensive experiments on simulation benchmarks and real-world tasks. On LIBERO, TS-Mask VLA achieves a 95.7 percent average success rate with only 0.5B parameters, outperforming significantly larger models. On CALVIN, it attains the best average sequence length of 4.19 and strong long-horizon performance. Comprehensive analyses and ablations further validate the effectiveness of our design. 

---
# An Autonomous Scientific Knowledge Generation Framework for AI-Driven Scientific Discovery 

**Authors**: Dibakar Datta  

**Link**: [PDF](https://arxiv.org/pdf/2607.09806)  

**Abstract**: Artificial intelligence (AI) is transforming scientific discovery, but its effectiveness is fundamentally limited by the availability of structured scientific knowledge. Although existing databases have accelerated data-driven materials research, much of the knowledge needed for predictive modeling and inverse design remains embedded in unstructured scientific literature. We present an Autonomous Scientific Knowledge Generation Framework that transforms scientific publications into a Unified AI-Ready Scientific Knowledge Base. The framework integrates ontology-guided literature acquisition, hybrid scientific knowledge extraction, semantic harmonization, knowledge fusion, and validation within a unified workflow. Rather than treating literature retrieval, information extraction, and database construction as separate tasks, the framework progressively converts scientific publications into structured, semantically consistent, and provenance-preserving knowledge suitable for AI-driven reasoning. As a proof of concept, the framework was applied to electro-optic materials. Autonomous literature acquisition retrieved and validated about 1,000 publications from multiple scholarly repositories. A representative subset of eight publications was processed through the complete workflow, generating 29 structured scientific records that were harmonized into 7 canonical scientific records. The results demonstrate the complete transformation from scientific literature to an AI-ready scientific knowledge base while preserving quantitative measurements, operating conditions, provenance, and scientific context. The proposed framework provides a scalable, domain-independent foundation for predictive AI, generative AI, and closed-loop AI-driven scientific discovery. 

---
# A Unified Model for Highly Accurate ECG-Free Dynamic Coronary Roadmapping Using Spatio-Temporal Transformers 

**Authors**: Saahil Islam, Sebastian Piat, Venkatesh N. Murthy, Serkan Cimen, Puneet Sharma, Andreas Maier, Florin C. Ghesu  

**Link**: [PDF](https://arxiv.org/pdf/2607.09805)  

**Abstract**: Percutaneous Coronary Intervention (PCI) is a minimally invasive procedure used to restore coronary blood flow obstructed by atherosclerotic plaque. During PCI, repeated injections of iodine-based contrast agents are required to visualize the coronary arteries and guide interventional devices. However, frequent contrast injections increase radiation exposure and the risk of contrast-induced nephropathy, with acute kidney injury reported in up to 30% of patients with renal impairment. Dynamic Coronary Roadmapping (DRM) reduces these risks by overlaying a precomputed angiographic vessel map onto live fluoroscopy and continuously updating it throughout the procedure. Accurate DRM relies on precise cardiac phase matching between angiography and fluoroscopy, together with reliable catheter tip tracking for motion compensation. These tasks remain challenging in ECG-free settings and when only limited manual annotations are available. We present a unified DRM framework that simultaneously performs cardiac phase matching and catheter tip tracking for accurate real-time guidance. Our method employs a large-scale spatio-temporal encoder pretrained on 16 million X-ray frames to learn cardiac motion dynamics. To the best of our knowledge, this is the first application of large-scale spatio-temporal pretraining for motion compensation in DRM. We further introduce auxiliary tasks based on ECG R-peak detection and catheter tip tracking, improving optimization while eliminating the need for extensive catheter mask annotations. Finally, a majority-voting postprocessing strategy aggregates temporal predictions, improving robustness and providing a confidence score that correlates with phase-matching error. Comprehensive evaluation on clinical X-ray datasets demonstrates state-of-the-art performance, achieving low temporal misalignment and robust phase-matching accuracy suitable for real-time DRM. 

---
# Trivial Prompt Reframing Bypasses Safety Guardrails in Googleś MedGemma-4B 

**Authors**: Avi-ad Avraam Buskila  

**Link**: [PDF](https://arxiv.org/pdf/2607.09804)  

**Abstract**: Open-weight medical language models are increasingly used as the base of patient-facing and clinician-support applications. Their model cards prohibit specific behaviors -- recommending exact drug dosages, issuing definitive diagnoses, prescribing treatments, adjudicating drug-drug interactions, and advising that emergency care can be skipped -- yet a model card describes intended behavior, not robust behavior. We quantify that gap for MedGemma-4B-it under attacks that require no technical sophistication. We build a fully factorial benchmark of 5 guarded-behavior concepts x 50 deterministically templated questions x 6 lay-accessible attack manners x 3 repetitions (4,500 generations), serve the model locally through Ollama under default sampling, and code every response refuse/hedge/comply with three independent judges (an LLM judge, a transparent regex judge, and an NLI-entailment judge). Under the primary LLM judge the overall Attack Success Rate (ASR, the fraction coded comply) is 38.0%. The two framings that reinterpret the request as legitimate dominate: recasting a question as a "medical board exam" item raises ASR from a 29.0% baseline to 53.1% (+24.0 points), and an appeal to an alleged doctor's authority raises it to 43.7% (+14.7); crude instruction-override prefixes have no significant effect. Robustness is dominated by topic: the drug-interaction guardrail is nearly absent (83.2% ASR) while the emergency-deferral guardrail is strong (4.7%) -- and the authority framing is the only attack that breaches it. We report Wilson confidence intervals, cluster-bootstrap effect sizes, a cluster-robust logistic regression, Cochran's Q, per-manner McNemar tests, and inter-judge reliability (Fleiss' kappa = 0.26); absolute ASR is judge-dependent while the ordering of attacks and topics is not. Our findings motivate stronger deployment-time guardrails for open medical models. 

---
# JEPA for AI-Native 6G: Predictive Representations and Open Challenges 

**Authors**: Sheikh Salman Hassan, Irshad A. Meer, Almoatssimbillah Saifaldawla, Yan Kyaw Tun, Mustafa Ozger, Madyan Alsenwi, Nguyen Van Huynh, Woong-Hee Lee, Cedomir Stefanovic, Mathini Sellathurai, Henk Wymeersch, Tharmalingam Ratnarajah  

**Link**: [PDF](https://arxiv.org/pdf/2607.09798)  

**Abstract**: Sixth-generation (6G) networks are moving toward AI-native operation, where learning modules are embedded across the radio access network (RAN), edge, and core. This transition requires learning from limited labels, heterogeneous wireless and network data, partial observations, non-stationary propagation, and latency-constrained control loops. Joint-embedding predictive architecture (JEPA) is a promising self-supervised paradigm for this setting because it predicts missing or future representations in latent space instead of reconstructing raw measurements or using contrastive negative samples. This article presents a wireless-oriented tutorial on JEPA for 6G intelligence. We define the JEPA training mechanism, describe how CSI, beam measurements, KPIs, topology graphs, and sensing observations can be tokenized and masked, and position the learned encoder as a predictive representation layer for RAN, O-RAN, edge, and core functions, with task-specific heads or controllers producing final decisions. Then we present an illustrative, beam-management case study suggesting that a wireless-aware target, specifically an auxiliary future beam-energy target during self-supervised pretraining, can improve label efficiency and robustness across shifted deployment conditions relative to a supervised source domain. Finally, we outline open challenges in multi-timescale prediction, action-conditioned modeling, distributed training, trustworthiness, efficient deployment, benchmarking, and standardization. 

---
# Large Multimodal Model-Based Environment-Aware Mobility Management 

**Authors**: Seokhyun Jeong, Sangmok Shin, Seungnyun Kim, Jiao Wu, Byonghyo Shim  

**Link**: [PDF](https://arxiv.org/pdf/2607.09795)  

**Abstract**: Recently, large language models (LLMs) have been successfully adopted in various fields, including wireless communications, robotics, and autonomous vehicles, owing to their outstanding adaptability and reasoning abilities. Despite their huge potential, the application of LLMs for mobility management is relatively scarce since it requires not only analyzing wireless measurements but also predicting dynamic user trajectories and making real-time handover decisions across densely deployed small base stations (SBSs). In this paper, we propose an environment-aware mobility management scheme based on large multimodal models (LMMs), which extend capabilities of LLMs to process multimodal sensing data. By leveraging LMMs, the proposed scheme extracts contextual information on the surrounding environments from RGB-D images to capture user equipment (UE) mobility patterns and identify signal reflections and blockages caused by static reflectors and dynamic obstacles. Using the extracted environmental information, the proposed scheme learns the intrinsic mapping from UE and SBS positions to channel capacity, referred to as channel capacity map (CCM), from which future channel capacities along UE trajectories are predicted. Based on the predicted channel capacities, we determine proactive handover decisions maximizing the cumulative channel capacities. Simulation results demonstrate that the proposed scheme achieves substantial channel capacity improvements over conventional deep learning (DL)-based approaches. 

---
# A Comprehensive Survey and Systematic Real-World Evaluation of Embodied Vision-and-Language Navigation 

**Authors**: Liuyi Wang, Kai Sheng, Zongtao He, Jinlong Li, Yongrui Qin, Haojie Dai, Xiangyi Wang, Jingwei Yang, Qingqing Yan, Chengju Liu, Qijun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.09792)  

**Abstract**: Navigation is a fundamental capability of autonomous systems, yet most existing approaches rely on highly structured models and strong prior assumptions, limiting their robustness in open and uncertain real-world environments. Vision-and-Language Navigation (VLN) offers a promising direction by enabling robots to integrate natural language understanding with visual perception in a data-driven manner. Although VLN has attracted increasing research attention, systematic methodological taxonomy and real-world validation remain limited. This survey presents a comprehensive review of VLN research. Specifically, state-of-the-art methods are organized along two orthogonal dimensions: action paradigms, including hierarchical and monolithic frameworks, and model paradigms, including discriminative and generative approaches. A critical analysis of their respective strengths and limitations is provided. Additionally, we conduct a systematic real-world evaluation of representative VLN system configurations on a physical robotic platform. Experiments across ten diverse real-world scenes show a substantial performance gap between simulation and real-world deployment under the tested configurations: a representative monolithic RGB-only method achieves 61% success in simulation but drops to 22% in real-world deployment, while a hierarchical framework achieves a higher real-world success rate of 51%, suggesting stronger robustness in our evaluation setting. Finally, we highlight key challenges in perception, decision-making, and control that must be addressed in future research. 

---
# Lifelong Representations: A Survey on Continual Self-Supervised Learning for Vision Models 

**Authors**: Sergi Masip, Alicja Dobrzeniecka, Jonathan Swinnen, Joachim Collin, Bartłomiej Twardowski, Szymon Łukasik, Tinne Tuytelaars  

**Link**: [PDF](https://arxiv.org/pdf/2607.09785)  

**Abstract**: Traditionally, continual learning has assumed access to labeled data, yet many real-world applications -- such as lifelong robotics -- require models to adapt continuously from unlabeled streams. This has led to the development of continual self-supervised learning (CSSL), a rapidly growing area that lacks a dedicated, systematic review. In this work, we present a comprehensive survey of CSSL for vision, with connections to emerging vision-language settings. First, we analyze existing evaluation protocols and highlight inconsistencies that hinder fair comparison. We then examine why self-supervised objectives exhibit improved robustness to catastrophic forgetting, relating this to task-agnostic representations and smoother loss landscapes. Next, we organize existing methods into a unified taxonomy based on their forgetting-mitigation strategies, including distillation, replay, regularization, architectural approaches, model merging, and objective-level adaptation. Finally, we identify open challenges such as scalability and the need for fast adaptability. We argue that advancing CSSL requires moving beyond small-scale benchmarks towards continual pre-training paradigms for large-scale systems. 

---
# Maximizing Human Efficiency in Large-Scale Robot Post-Training via VLAC-Cut Guided Pipeline 

**Authors**: Shaopeng Zhai, Qi Zhang, Tianyi Zhang, Haoran Zhang, Fuxian Huang, Zhanhui Lin, Zijun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.09776)  

**Abstract**: When adapting Vision Language Action (VLA) models to downstream tasks, multiple rounds of post training are required because a single round of data cannot resolve all issues, making continuous iterations necessary to progressively address the weaknesses exposed in previous rounds. In this report, we aim to maximize human efficiency during post-training, defined as the policy improvement and task throughput achieved per unit of human labor and time.
We propose a human-efficient post-training pipeline that enables a small number of human operators to supervise multiple robots. The pipeline is built around a specialized division of labor: a trained Teleoperator focuses on high-value remote interventions and recovery demonstrations, while a Floor Operator monitors multiple robots, triggers takeovers, and performs physical resets. This role specialization reduces task switching, lowers operator training costs, and allows limited human labor to supervise more robot interaction across a larger fleet. To improve data utilization efficiency, we introduce VLAC-CUT as an automatic rollout curation tool. It segments autonomous robot trajectories into progress-making, idle, failure-inducing, and recovery portions, preserving useful segments while filtering harmful or uninformative ones. The curated rollout data are combined with Human-in-the-Loop data for the next post-training round. We validate the proposed pipeline on four real-world manipulation tasks. Across iterative post-training rounds, the final policies achieve 80\%--95\% success rates and improve task throughput by 1.7$\times$--4.2$\times$ over the base model. Under the same human-intervention budget, VLAC-CUT guided rollout reuse outperforms HITL-only training in both success rate and throughput. 

---
# Listen to the Features: Voice Anonymization Driven by Content Embedding Matching over Signal Reconstruction 

**Authors**: Adrien Schneider, Kacper Zabkowski, Anderson Augusma, Frédérique Letué, Maria Camila Pinzon, Dominique Vaufreydaz  

**Link**: [PDF](https://arxiv.org/pdf/2607.09767)  

**Abstract**: The paper presents a voice anonymization model focusing on preserving content rather than producing realistic speech. It relies on content embeddings extracted from a frozen pretrained wav2vec2 encoder. These embeddings are decoded into an anonymized signal using vector quantization and a HiFi-GAN vocoder, both trained on LibriTTS without any waveform reconstruction loss or speaker embedding mapping. The training objective enforces that embeddings of the anonymized signal match those of the original one. While training, an auxiliary speaker classification branch with a gradient reversal layer is used to discard speakerspecific information. Results show that this straightforward embedding-based approach achieves very low WER (2.53) with an anonymization performance (EER 13.39) ranking within first level for VPC. Notably, emotions are partially preserved (UAR 43.91), even without a supporting training objective, while the anonymized voice is audible without reconstruction loss. 

---
# OmniSCS: Omni Safety-Critical Scenario Synthesis for Autonomous Driving via a Fully Editable Driving World 

**Authors**: Xiaoyun Dong, Qian Xu, Yang Lu, Yang Lou, Yung-Hui Li, Jianping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.09764)  

**Abstract**: The synthesis of safety-critical scenarios (SCS) and their evaluation through closed-loop simulations are crucial for developing robust autonomous driving systems. A key aspect of this process involves editing agent states in both appearance and trajectory levels within existing scenes. However, current methods struggle to preserve data fidelity after scene editing and fail to efficiently generate high-quality SCS through such modifications. To overcome these limitations, we propose OmniSCS, an innovative system that generates photorealistic SCS with high physical fidelity while enabling closed-loop testing in synthetic environments. OmniSCS comprises two key modules: 1) A Fully Editable Driving World Construction module that maintains high-fidelity agent appearance and background during scene editing via dual-strategy agent reconstruction and depth-refinement background reconstruction methods. 2) A SCS Synthesis module that facilitates object insertion and agent trajectory editing to synthesize diverse SCS while preserving data fidelity. Experiments on nuScenes, Waymo, and KITTI datasets show that OmniSCS outperforms state-of-the-art methods in edited scene fidelity. We further validate its ability to enhance autonomous driving algorithms and support real-time (13Hz) closed-loop testing. Overall, OmniSCS provides a safer, more effective, and cost-efficient solution for SCS optimization and testing in autonomous driving. 

---
# Knowledge-Constrained Shape Optimization with a Mixture-of-Experts Neural Operator for High-Confidence Design 

**Authors**: Wenhao Fan, Yuanwei Bin, Jianghan Gu, Wenfa Luo, Jiao Xiang, Yuntian Chen, Shiyi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.09763)  

**Abstract**: Engineering shape optimization faces challenges in both expert-dependent problem setup and surrogate-model reliability. In practical aerodynamic design, optimization settings such as editable regions, deformation ranges, and design-preservation constraints are typically specified manually by experienced engineers, while surrogate-based optimization may become unreliable for heterogeneous geometry databases and out-of-distribution designs. To address these challenges, we propose a knowledge-constrained shape-optimization framework that translates knowledge-based constraints and user intent into quantifiable parameters of DFFD-based deformation operators, enabling engineering-aware and controllable constrained optimization. We further develop a Mixture-of-Experts Neural Operator (MoE-NO) to improve drag prediction and trend consistency over heterogeneous aerodynamic datasets. Based on the MoE-NO encoder and Mahalanobis distance, an uncertainty-estimation strategy is introduced to detect out-of-distribution geometries and selectively trigger physics-solver feedback for local sample enrichment. Experiments on in-house MPV, SUV, and Sedan datasets show that MoE-NO achieves a test-set MAPE of $1.16\%$ and a trend-prediction accuracy of $94.34\%$, outperforming the best baseline results of $1.52\%$ and $90.34\%$, respectively. Vehicle shape-optimization experiments further yield CFD-validated drag coefficient reductions of approximately $4\%$ to $10\%$. 

---
# Physics-Informed Structure Anchoring With Capture-Aware Prototype Calibration for Cross-Environment RF Fingerprinting 

**Authors**: Fengchong Yao, Jianbing Li, Qing Liu, Qikun Liu, Kefeng Song, Haitao Li, Song Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.09760)  

**Abstract**: Radio frequency fingerprint identification (RFFI) uses transmitter-specific hardware imperfections as a physicallayer identity cue for Internet of Things (IoT) devices, but deep RFFI models often degrade when the acquisition environment changes. In multi-antenna reception, this degradation is not merely a generic distribution shift. It is also shaped by receiver-array topology, frequency-offset dynamics, and capturedependent target structure, which can distort embeddings and move source-trained decision boundaries. This article proposes physics-informed structure anchoring with capture-aware prototype calibration (PISA-CAPC), a framework that separates source representation anchoring from fixed-backbone target calibration. The representation stage organizes antenna tokens with a topology graph and modulates the graph using CFO-derived acquisition-dynamics descriptors. Bounded contextual residual suppression is then applied around the identity representation. At deployment, unlabeled capture-aware prototype calibration (U-CAPC) calibrates target decision scores through capturelocal prototype evidence under a fixed representation, mitigating boundary shift without requiring target-domain backbone updates or target labels. On a measured ten-transmitter multiantenna WiFi benchmark, PISA-CAPC achieves 0.9257 targetdomain mean Macro-F1 under a balanced transductive setting. Ablations confirm that topology-guided structure anchoring, contextual residual suppression, and capture-aware calibration contribute complementary gains. These results establish PISACAPC as a fixed-backbone route to cross-environment RFFI, coupling physically motivated representation learning with labelfree, capture-aware decision calibration. 

---
# ReflectWorld-MM: An Entity-Oriented Multi-Media Memory System for Open-Ended Video Streams 

**Authors**: Xiaokang Ma, Yifan Sun, Zhihong Jin, Jie Gu, Yudong Luo, Shenyi Shao, Chu Tang, Jingmin Chen, Li Pu  

**Link**: [PDF](https://arxiv.org/pdf/2607.09759)  

**Abstract**: Building assistants that can continually watch the world, remember what they see, and reason over their accumulated experience is a long-standing goal, and recently multimodal agents equipped with long-term memory over video streams have attracted increasing interest. Unfortunately, existing systems either keep their memory inside the model context or in a flat feature store, and organize it around frames rather than around the persistent entities a stream is really about, which confines them to bounded videos and weakens their ability to track who and what reappears over time. In this paper, we propose ReflectWorld-MM, an entity-oriented multi-media memory system for open-ended video streams. It consists of three parts. The first is a perception front-end that turns a streaming video into entity-resolved observations under a bounded short-term memory. The second is a hierarchical long-term memory, grounded in human memory theory, that couples a multi-scale episodic memory, an evolving entity-centric semantic memory, and a procedural memory. The third is a complete realization, built for real-world operation, that ingests arbitrary streams and plugs into off-the-shelf assistants. Across six long-video and lifelong-memory benchmarks, ReflectWorld-MM achieves the best accuracy on all six, outperforming strong memory agents and a frontier model. 

---
# RSLoRA: Training-free Rank Allocation for LoRA via Representational Sensitivity Probing 

**Authors**: Jiaqi Liu, Haidong Kang, Qihui Zhao, Guo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2607.09757)  

**Abstract**: Low-Rank Adaptation (LoRA) has become a cornerstone of parameter-efficient fine-tuning (PEFT); however, the conventional practice of uniform rank assignment ignores the functional heterogeneity of neural layers. Existing rank allocation methods typically struggle with a trade-off between computational intensity and heuristic simplicity: training-based methods suffer from prohibitive overhead, while pre-allocation methods fail to capture the dynamic task-specific representation manifold. In this paper, we propose RSLoRA (Representational Sensitivity LoRA), a training-free and gradient-free rank allocator driven by activation-space geometry. We identify a "sensitivity regime shift" across layers, observing that static weight analysis and local gradients are insufficient to reflect how updates reshape a model's internal representations. To address this, RSLoRA introduces a virtual representational probing mechanism. By simulating adaptation through structured low-rank noise and measuring the resulting manifold displacement by using Effective Rank and Frechet Distance, we identify high-sensitivity modules that require higher rank capacity. Our framework effectively bridges the gap between expert-crafted heuristics and actual representational impact. Extensive evaluations demonstrate that RSLoRA consistently outperforms state-of-the-art allocators (e.g., AdaLoRA, GoRA) across mainstream benchmarks. By eliminating the need for iterative training-time adjustments and backward gradients, RSLoRA provides a highly efficient, robust, and representation-aware solution for large-scale model adaptation. 

---
# Cross-Subject Modeling for Widefield Calcium Imaging via Atlas-Aligned Spatiotemporal Tokenization 

**Authors**: Mohammad Hosseini, Eray Erturk, Saba Hashemi, Maryam M. Shanechi  

**Link**: [PDF](https://arxiv.org/pdf/2607.09754)  

**Abstract**: Large-scale, multi-subject widefield calcium imaging provides unprecedented access to brain-wide cortical dynamics. However, the high dimensionality, complex spatiotemporal structure, and substantial task-irrelevant activity in widefield recordings have largely restricted modeling efforts to single-session analyses, limiting scalability and generalization. While multi-subject pretrained models have been explored for some neural modalities, multi-subject models for widefield calcium imaging have not yet been demonstrated; further, subject-invariant zero-shot behavior decoding remains elusive for multi-subject models across neural modalities more broadly. As a first step toward foundation modeling of widefield data, we introduce WiCAT, a multi-subject model that leverages self-supervised pretraining to both outperform single-session models and enable zero-shot behavior decoding on unseen subjects. WiCAT introduces an atlas-grounded tokenization scheme without session-specific components and learns globally shared spatiotemporal representations. Across multiple widefield datasets, the pretrained model supports lightweight downstream decoding, transfers across subjects, tasks, and datasets, and outperforms baseline models. Notably, the model also achieves robust zero-shot continuous behavior decoding and left-out brain region reconstruction on unseen subjects. 

---
# Unified Backbone Refinement for Diffusion Models via Internal-Latent Analysis 

**Authors**: Haksoo Lim, Myeongjin Lee, Wonjoon Chang, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2607.09753)  

**Abstract**: Diffusion models have achieved remarkable success across diverse domains, with performance closely related to the denoising backbones that parameterize the score function. In this paper, we present a systematic, phase-aware analysis of diffusion components and show that abrupt, early-stage fluctuations in deep latents are strongly associated with artifacts. Guided by these findings, we introduce DUNE (Diffusion Unified Network refiNEr), a training-free refinement framework that detects abrupt deviations in deep low-noise internal latents using a shared EMA-based criterion, and applies backbone-specific suppression to the detector-selected entries. Although derived from U-Net, the same detect-suppress principle extends naturally to Transformer-based diffusion models by acting on the latents of deep self-attention blocks. Extensive experiments across multiple backbones indicate that DUNE improves fidelity while reducing hallucinations, offering new insight into where and when diffusion backbones should be controlled. 

---
# Data-Driven Forward and Inverse Modeling of V-Beam Thermal Sensors 

**Authors**: Tudor Bartha, Radu Chiorean, Adrian Groza  

**Link**: [PDF](https://arxiv.org/pdf/2607.09752)  

**Abstract**: This paper presents a machine learning framework for data-driven inverse design of V-beam thermal sensors. The goal is to determine the optimal sensor geometry: beam inclination angle, beam length and beam width that achieves a target displacement under a given temperature. The design should also provide the geometry with minimum structure volume and minimum mechanical stress the sensor must support. This problem is ill-posed as for a given displacement there are multiple possible geometric configurations, causing direct regression methods to fail. We document a series of five exploratory trials that progressively revealed the nature of the problem culminating in a two-phase solution: a neural network forward model trained to map geometry and material constants to sensor responses, a gradient-descent inverse optimization over the frozen forward model, minimizing stress and volume simultaneously. The proposed pipeline utilizes a 3000-sample dataset and achieves a MAPE of 4.76% for predicting the displacement, more than 70% of predictions having MAPE of under 5%. 

---
# MorphologyFM: A Foundation Model for Morphology-Aware Representation Learning from ECG and Pulse Oximetry Waveforms 

**Authors**: Saiyang Feng, Yuanyun Zhang, Shi Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.09749)  

**Abstract**: Foundation models have recently emerged as a powerful paradigm for learning transferable representations from large scale biomedical data, yet existing approaches for physiological waveforms primarily optimize reconstruction or forecasting objectives that do not explicitly preserve clinically meaningful waveform morphology. Electrocardiograms (ECGs) and pulse oximetry (SpO2) waveforms encode rich cardiovascular and hemodynamic information through their morphological structure. In this work, we introduce MorphologyFM, a multimodal foundation model pretrained on paired ECG and SpO2 waveforms from the MIMIC critical care database using a morphology aware self supervised learning objective. MorphologyFM combines morphology guided masking, cross modal representation learning, and contrastive latent alignment to learn representations that capture clinically relevant physiological structure without requiring manual annotations. We evaluate MorphologyFM across multiple downstream prediction tasks, including arrhythmia classification, hypoxemia prediction, mortality prediction, and length of stay estimation, demonstrating consistent improvements over representative self supervised learning methods, including Masked Autoencoders (MAE), contrastive learning, Barlow Twins, and Joint Embedding Predictive Architectures (JEPA). Furthermore, we show that jointly modeling ECG and SpO2 waveforms produces more transferable representations than single modality pretraining. Our results establish waveform morphology as a powerful inductive bias for self supervised physiological representation learning and introduce MorphologyFM as a general purpose foundation model for continuous physiological monitoring. 

---
# SWIFT: A Small-World Interaction Framework for Flow-Aware Trajectory Prediction in Autonomous Driving 

**Authors**: Chengyue Wang, Bin Rao, Haicheng Liao, Bonan Wang, Chengzhong Xu, Zhenning Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.09741)  

**Abstract**: Accurate trajectory prediction in autonomous driving hinges on modeling dynamic and context-dependent interactions among traffic agents. However, most existing approaches are purely data-driven and lack structural priors, which limits their generalization under distribution shifts. In this work, interaction modeling is revisited through the structure and dynamics of traffic networks, and SWIFT (Small-World Interaction Framework for Trajectory prediction) is proposed as a unified framework that integrates small-world networks with traffic flow theory. SWIFT introduces structural inductive biases via a Small-World Interaction Network that captures both local and global dependencies, and a Flow Regime Encoder that adapts the interaction structure to scene-level traffic states. Interaction reasoning is further enhanced through a multi-relational graph module that explicitly encodes direct and higher-order agent relationships. Extensive experiments on three real-world datasets, nuScenes, MoCAD, and NGSIM, show that SWIFT consistently outperforms strong baselines in prediction accuracy across diverse traffic regimes. Beyond accuracy, SWIFT exhibits improved generalization to unseen locations and regimes, robustness under noisy observations, and strong performance with limited training data, supporting the effectiveness of its structure-aware design. 

---
# The Universal Language of CSI:Unifying Wireless Sensing Across Devices and Environments 

**Authors**: Jiayi Chen, Weiting Ou, Guangxu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.09727)  

**Abstract**: WiFi sensing based on Channel State Information (CSI) promises ubiquitous, device-free perception, yet current research remains trapped in a Tower of Babel - fragmented into isolated silos where models are tailored to specific hardware dialects, fixed environments, and narrow tasks. The primary bottleneck is the Heterogeneity Gap: the disparity in signal dimensions, sampling rates, and semantic labels that prevents cross-system understanding. To bridge this gap, we propose a foundation-model framework that treats CSI not merely as raw signals but as a structured language with a learnable universal grammar. We first curate and standardize a large collection of heterogeneous real-world CSI datasets, establishing a unified infrastructure that allows incompatible signal formats to be treated as a single corpus. Second, we introduce a modular architecture that acts as a universal translator where lightweight dataset-specific adapters tokenize diverse signal inputs into a shared latent vocabulary, while a shared self-supervised Transformer backbone learns the temporal syntax of human motion and environmental dynamics. This design decouples sensing semantics from hardware syntax. Extensive evaluations show that by mastering this universal language, our approach consistently outperforms task-specific baselines and exhibits strong generalization capability in new environments, achieving superior efficiency in few-shot scenarios. By effectively absorbing heterogeneity, the framework offers a path toward robust, general-purpose wireless sensing, mirroring the linguistic generalization observed in Large Language Models. The code implementation is available at: this https URL. 

---
# The Ramanujan Challenge For AI 

**Authors**: Michael Shalyt, Rotem Kalisch, Carsten Schneider, Hila Barkan, Elyasheev Leibtag, John Campbell, Shachar Weinbaum, Tali Monderer, Ashvni Narayanan, Ido Kaminer  

**Link**: [PDF](https://arxiv.org/pdf/2607.09721)  

**Abstract**: To help evaluate the mathematical skills of current AI systems, we present a set of formulas for fundamental mathematical constants. These problems are attractive for AI evaluation because they are concrete and can be checked numerically to arbitrary precision, yet proving them may require non-obvious mathematics. Mathematical constants such as $\pi$, $e$, Catalan's constant, and special values of the Riemann zeta function have fascinated mathematicians for centuries. The search for formulas evaluating mathematical constants has produced some of the most beautiful mathematics in the field, especially in cases that yield irrationality proofs or fast convergence rates. Ramanujan's legacy is emblematic of this tradition. The list we provide contains two types of problems: formulas whose proofs are known to the authors but will remain encrypted for a short initial period; and formulas that are not yet proven. We are curious to see the achievements of AI in both cases. 

---
# Model Collapse: On Recursion, Noise, and Uncharted Machine Visions 

**Authors**: Violaine Boutet de Monvel  

**Link**: [PDF](https://arxiv.org/pdf/2607.09705)  

**Abstract**: Since 2023, computer scientists have warned against model collapse -- the contamination of training sets with AI-generated outputs that progressively degrade model performance. Exemplifying a positive-feedback-driven failure, it produces effects such as word repetition or pixel noise, ultimately leading to a loss of meaning and coherence -- at least from an engineering standpoint. From a creative one, however, collapse is not merely a breakdown: it also functions as a recursive mirror that recalls early analog video feedback experiments, raising once again the question of what happens when a system turns inward and sees itself. In such cases, so-called machine vision no longer transmits the world (as in tele-vision) but increasingly generates worlds from within. Drawing on media archaeology through case studies of both historical video synthesis techniques and contemporary artistic uses of machine learning, this paper examines what recursive training reveals about the dependent nature of AI-generated data. It argues that the potential effects of collapse challenge transhumanist ideals while inviting an aesthetic perspective, positioning noise and recursion as key concepts for understanding both artmaking and the AI ecosystem. Distributing agency across scales and networks, the latter currently remains reliant on new human-produced content, particularly within foundation models trained on massive datasets. 

---
# Mitigating Early Training Collapse in CTR Models 

**Authors**: Ergun Biçici, Erkan Çetinyamaç  

**Link**: [PDF](https://arxiv.org/pdf/2607.09696)  

**Abstract**: Deep neural models for click-through rate prediction often exhibit a sharp decline in validation performance immediately after the first training epoch despite continued improvement in training loss. This instability restricts effective learning and limits model performance. In this study, we analyze this behavior using large-scale industrial datasets and evaluate practical mitigation strategies. While reducing the learning rate provides only incremental gains, controlling feature sparsity yields substantial improvements. Removing highly sparse features and aggregating infrequent feature values stabilizes training, extends useful learning beyond a single epoch, and improves both offline evaluation metrics and online system performance. 

---
# Depth-Entropy Guided Sampling for Training-Free LLM Reasoning 

**Authors**: Zibin Meng, Peng Xie, Kani Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.09693)  

**Abstract**: Reinforcement learning (RL) has become the dominant paradigm for improving the reasoning capabilities of large language models, but it requires expensive training, curated data, and reward signals. Recent work shows that sampling from sharpened base-model distributions at test time recovers much of the RL gain, yet existing methods rely solely on output-layer likelihoods and ignore the transformer's internal forward-pass dynamics. We introduce Depth-Entropy Guided Sampling (DEGS), a training-free, test-time method that exploits layer-wise entropy collapse as an intrinsic quality signal. We observe that stronger reasoners -- including RL-posttrained variants -- exhibit a distinctive "late collapse": logit-lens decoded entropy stays elevated until deeper layers before converging. We define a per-sequence collapse depth $D(\mathbf{x})$ and a joint objective $\pi(\mathbf{x}) \propto p(\mathbf{x})^\alpha \exp(\beta D(\mathbf{x}))$ that combines sequence likelihood with this depth-entropy structure, instantiated inside an MCMC power-sampling framework (DEGS-MCMC). Across three open-weight models and four reasoning benchmarks, this near-chance per-candidate signal compounds over the sampling trajectory into state-of-the-art training-free accuracy, with gains largest out of domain and on the harder splits -- exactly where likelihood alone falls short -- at single-digit-percent wall-clock overhead. DEGS narrowly trails an in-house GRPO reference on the math splits GRPO was trained for, yet surpasses it out of domain on GPQA for all three models, without any training, reward model, or labeled data. 

---
# What Context Does a Coding Agent Actually Need to Act? 

**Authors**: Brian Sam-Bodden  

**Link**: [PDF](https://arxiv.org/pdf/2607.09691)  

**Abstract**: A modern coding agent can hold an entire repository in its context window. Most of its reading is wasted -- and the interesting question is not how much context an agent can use, but what it actually \emph{needs}. We study that question at the moment it matters most: when the agent must \emph{edit} code. Separating \emph{finding} the work site from \emph{acting} on it, we hold localization fixed with an oracle, vary only how the code is represented, and score context against real issue resolution on SWE-bench Verified. The answer is starkly minimal. The signal lives in the code being edited itself: natural-language summaries of it answer almost none of the behavioral questions that the source answers ($4/45$ vs.\ $27/45$, held-out repositories, independent judge), and the gap belongs to the representation, not the summarizer -- a frontier model's summaries score exactly as poorly as a 3B model's. The surrounding context hardly matters either: across every multi-file instance in Verified, under a protocol frozen before any data, rendering a file's remainder as UML skeletons and signatures resolves no more issues than deleting that remainder outright ($N{=}70$, exact McNemar $p{=}0.75$). That was our registered hypothesis, and it failed. Compressed context, meanwhile, matches whole files at a third of the tokens: a resolved issue costs $19$K context tokens, not $94$K. The instrument also yielded a finding the field should keep: temperature-0 API inference flips ${\sim}9\%$ of per-instance outcomes between byte-identical runs. That is a noise floor under every small effect reported on this benchmark, including ours. We release the instrument -- gold-validated environments, per-instance proof that every reference edit is expressible from every arm's context, deterministic patch construction, and pre-registered hypotheses whose nulls we publish. 

---
# Transfer Learning Across Policy Regimes in Adaptive Multi-Agent Systems 

**Authors**: Roberto Garrone  

**Link**: [PDF](https://arxiv.org/pdf/2607.09685)  

**Abstract**: Policy models often assume that the relationship between a policy instrument and its outcome remains stable across institutional conditions. In adaptive socio-technical systems this assumption may fail: regulatory change can alter incentives, agents can respond strategically, and the mapping from policy variables to aggregate outcomes can change. This paper studies such regime change as a transfer-learning problem in adaptive multi-agent systems. A policy regime is represented as a learning problem induced by an observable input distribution and a target function mapping policy variables to outcomes. We compare a blank-slate learner that searches a flexible hypothesis class in the new regime with a transfer learner whose effective hypothesis class is restricted by structural knowledge from the previous regime. Transfer is beneficial when this restriction preserves the new target function while reducing effective complexity; it is harmful when the restriction excludes the new target and creates misspecification. A stylized emissions-regulation experimental environment and a dynamic ABM robustness experiment support the claim. When the target regime preserves an affine monotone tax-emissions relation, transfer improves empirical small-sample performance. When the target regime introduces a threshold break, the same transferred structure produces negative transfer: held-out error remains high, online prediction generates more mistakes, and repeated online streams show larger cumulative and final-window error under misspecification. The contribution is methodological: previous regulatory experience should be reused when it captures stable structural invariants, but treated cautiously when policy change alters the policy-outcome relationship. 

---
# SciML in the Wild: A Diagnostic Study of When Structural Priors Help and When They Hurt 

**Authors**: Vrishank Sai Anand, Prathamesh Dinesh Joshi, Raj Abhijit Dandekar, Rajat Dandekar, Sreedath Panat  

**Link**: [PDF](https://arxiv.org/pdf/2607.09684)  

**Abstract**: Scientific Machine Learning (SciML) methods such as Neural Ordinary Differential Equations (NODEs), Physics-Informed Neural Networks (PINNs), and Universal Differential Equations (UDEs) are most effective when structural priors reflect reliable governing dynamics. We ask what happens when this assumption is violated. Using macroeconomic forecasting as a stress-test domain, we evaluate five model families, ARIMA, LSTM, NODE, PINN, and UDE, across 23 countries using sparse annual data, multiple temporal splits, and five random seeds. Our results show that none of the evaluated models achieve consistently strong forecasting performance, highlighting the difficulty of low-frequency macroeconomic prediction. However, a clear relative hierarchy emerges: less-constrained models, particularly ARIMA and NODE, consistently outperform more-constrained heuristic-prior models such as PINN and UDE. Rather than treating this as a rejection of SciML, we interpret it as a diagnostic result: structural priors can act as misregularizers when they do not match the data-generating process. We identify failure modes including prior misalignment, regime shifts, structural breaks, and optimization instability, and argue that SciML practitioners should test whether structure helps before assuming that more structure is beneficial. 

---
# Ablation, Statistical Inference, and Validation for KV-Cache Compression 

**Authors**: Paolo D'Alberto, Ashish Siarasao, Elliott Delaye, Rajeev Patwari  

**Link**: [PDF](https://arxiv.org/pdf/2607.09683)  

**Abstract**: This study systematically compares Turbo-Quant and SpectralQuant KV-cache compression, evaluating non-dominated schemes, including WHT rotation with Beta Lloyd-Max and QJL, through a statistical validation methodology that separates systematic codec differences from implementation variance. Key findings reveal that while eigenbasis-based methods fail on heavy-tailed data due to covariance instability, they excel in structured regimes, with the effective semantic dimension ($d_{eff}$) adapting to calibration budgets rather than true data rank. (this is an abstract of the abstract thank you ) 

---
# ECG-LDC: A Hardware-Efficient Low-Dimensional Computing Framework for ECG Arrhythmia Classification 

**Authors**: Anh Tran, Khanh Tran, Cuong Do  

**Link**: [PDF](https://arxiv.org/pdf/2607.09680)  

**Abstract**: Continuous cardiac monitoring in wearable devices demands classifiers that are simultaneously accurate, energy-efficient, and deployable on resource-constrained hardware. While deep neural network approaches have demonstrated high classification accuracy for electrocardiogram (ECG) arrhythmia detection, their substantial parameter counts and reliance on multiply-accumulate-intensive operations make them impractical for low-cost edge platforms. In this work, we propose ECG-LDC, a hardware-software co-design framework that adapts Low-Dimensional Computing (LDC) for real-time ECG arrhythmia classification. ECG-LDC employs a dual-encoder architecture with dedicated value and feature codebooks to independently encode morphological waveform features and RR-interval temporal features, enabling effective capture of both intra-beat and inter-beat cardiac dynamics. The framework encompasses data preprocessing, model training, and a hardware accelerator architecture prototyped on the Pynq-Z2 platform. Implemented using binary representations and XOR/XNOR-based operations, ECG-LDC achieves $97.18\%$ accuracy with a memory footprint of only $3.86\ \text{kB}$. ECG-LDC sacrifices approximately $1.8\%$ accuracy versus SOTA TinyML classifiers but achieves $11$~$ 570\times$ reduction in memory usage; among FPGA-based five-class arrhythmia classifiers, it delivers the highest accuracy with up to $2.4\times$ fewer LUTs and zero DSP block utilization, affirming its suitability for real-time arrhythmia detection on resource-constrained wearable platforms. 

---
# Knowledge Graphs Meet Graph Neural Networks: A Comprehensive Survey 

**Authors**: Chengcheng Sun, Jiayun Tian, Cheng Zhai, Zhixiao Wang, Yajie Song, Xiaobin Rui, Jian Zhang, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2607.09666)  

**Abstract**: Graph Neural Networks (GNNs) have emerged as a powerful paradigm in Knowledge Graphs (KGs) due to their intrinsic ability to model graph-structured data. However, there remains a lack of a systematic review about GNN-based methodologies across the entire knowledge graph technologies pipeline. To address this gap, we first propose a novel two-level taxonomy framework for GNN-based knowledge graph technologies: the KG technologies pipeline and GNN-based perspective. Specifically, the knowledge graph technologies pipeline covers knowledge graph construction, knowledge graph embedding, knowledge reasoning and knowledge graph applications. Meanwhile, the GNN-based perspective provides a new categorization of knowledge graph technologies with GNN models, such as GCN, GAT, and HGNN. Then, we analyze the advantages of GNN technology based on the characteristics of different tasks in the knowledge graph lifecycle. Furthermore, we detailed review various GNN-based models for knowledge graph following the proposed taxonomy, and summarize strengths and limitations. Finally, we discuss unresolved challenges and outline promising directions for future research. 

---
# Reverse Engineering Compliance: A Dual-Graph Verification Framework for Auditing Legacy IT Security Concepts 

**Authors**: Lea Roxanne Muth, Marian Margraf  

**Link**: [PDF](https://arxiv.org/pdf/2607.08292)  

**Abstract**: The NIS-2 Directive increases the need for continuous, auditable compliance evidence and motivates a shift from document-based compliance toward machine-readable compliance artifacts. The Open Security Controls Assessment Language (OSCAL) is a standard for this purpose, which the German Federal Office for Information Security (BSI) is adapting with Grundschutz++. However, companies are still managing extensive legacy IT security concepts (IT-SCs), and migrating them without verification could transfer outdated assets into the new format. While existing research primarily addresses the generation of new concepts, there is a lack of a verification framework that extracts legacy IT-SCs into an auditable intermediate representation, deterministically compares the extracted graph with an independently constructed reference state, and exports schema-valid OSCAL artifacts.
This paper introduces the Automated Security Concept Structure Extraction and Reverse Topology-checking (ASSERT) Framework, which addresses this gap by using ontology-based extraction of legacy documents into formal document graphs, a five-class graph difference against a verified reference graph, and the export into schema-valid OSCAL outputs for system description and assessment evidence. Using the BSI's RecPlast dataset, we compare a local open-weight model and a commercial model across three configurations with different levels of reference-ontology exposure. The evaluation shows that ASSERT makes document-infrastructure inconsistencies measurable, but reveals a trade-off between discovering undocumented entities and enforcing a schema. 

---
