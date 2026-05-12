# NanoResearch: Co-Evolving Skills, Memory, and Policy for Personalized Research Automation 

**Authors**: Jinhang Xu, Qiyuan Zhu, Yujun Wu, Zirui Wang, Dongxu Zhang, Jianxin Tang, Marcia Tian, Yiling Duan, Siyuan Li, Jingxuan Wei, Sirui Han, Yike Guo, Odin Zhang, Conghui He, Cheng Tan  

**Link**: [PDF](https://arxiv.org/pdf/2605.10813)  

**Abstract**: LLM-powered multi-agent systems can now automate the full research pipeline from ideation to paper writing, but a fundamental question remains: automation for whom? Researchers operate under different resource configurations, hold different methodological preferences, and target different output formats. A system that produces uniform outputs regardless of these differences will systematically under-serve every individual user, making personalization a precondition for research automation to be genuinely usable. However, achieving it requires three capabilities that current systems lack: accumulating reusable procedural knowledge across projects, retaining user-specific experience across sessions, and internalizing implicit preferences that resist explicit formalization. We propose NanoResearch, a multi-agent framework that addresses these gaps through tri-level co-evolution. A skill bank distills recurring operations into compact procedural rules reusable across projects. A memory module maintains user- and project-specific experience that grounds planning decisions in each user's research history. A label-free policy learning converts free-form feedback into persistent parameter updates of the planner, reshaping subsequent coordination. These three layers co-evolve: reliable skills produce richer memory, richer memory informs better planning, and preference internalization continuously realigns the loop to each user. Extensive experiments demonstrate that NanoResearch delivers substantial gains over state-of-the-art AI research systems, and progressively refines itself to produce better research at lower cost over successive cycles. 

---
# BenchCAD: A Comprehensive, Industry-Standard Benchmark for Programmatic CAD 

**Authors**: Haozhe Zhang, Kaichen Liu, Miaomiao Chen, Lei Li, Shaojie Yang, Cheng Peng, Hanjie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.10865)  

**Abstract**: Industrial Computer-Aided Design (CAD) code generation requires models to produce executable parametric programs from visual or textual inputs. Beyond recognizing the outer shape of a part, this task involves understanding its 3D structure, inferring engineering parameters, and choosing CAD operations that reflect how the part would be designed and manufactured. Despite the promise of Multimodal large language models (MLLMs) for this task, they are rarely evaluated on whether these capabilities jointly hold in realistic industrial CAD settings. We present BenchCAD, a unified benchmark for industrial CAD reasoning. BenchCAD contains 17,900 execution-verified CadQuery programs across 106 industrial part families, including bevel gears, compression springs, twist drills, and other reusable engineering designs. It evaluates models through visual question answering, code question answering, image-to-code generation, and instruction-guided code editing, enabling fine-grained analysis across perception, parametric abstraction, and executable program synthesis. Across 10+ frontier models, BenchCAD shows that current systems often recover coarse outer geometry but fail to produce faithful parametric CAD programs. Common failures include missing fine 3D structure, misinterpreting industrial design parameters, and replacing essential operations such as sweeps, lofts, and twist-extrudes with simpler sketch-and-extrude patterns. Fine-tuning and reinforcement learning improve in-distribution performance, but generalization to unseen part families remains limited. These results position BenchCAD as a benchmark for measuring and improving the industrial readiness of multimodal CAD automation. 

---
# New AI-Driven Tools for Enhancing Campus Well-being: A Prevention and Intervention Approach 

**Authors**: Jinwen Tang  

**Link**: [PDF](https://arxiv.org/pdf/2605.10804)  

**Abstract**: Campus well-being underpins academic success, yet many universities lack effective methods for monitoring satisfaction and detecting mental health risks. This dissertation addresses these gaps through prevention (improving feedback collection) and intervention (advancing mental health detection), unified under an integrated framework. For prevention, we developed TigerGPT, a personalized survey chatbot leveraging LLMs to engage users in context-aware conversations grounded in conversational design and engagement theory, achieving 75% usability and 81% satisfaction. To address its limitations in repetitiveness and response depth, we introduced AURA, a reinforcement-learning framework that adapts follow-up question types (validate, specify, reflect, probe) within a session using an LSDE quality signal (Length, Self-disclosure, Emotion, Specificity), initialized from 96 prior conversations. AURA achieved +0.12 mean quality gain (p=0.044, d=0.66), with 63% fewer specification prompts and 10x more validation behavior. For intervention, we examine Expressive Narrative Stories (ENS) for mental health screening, showing BERT(128) captures nuanced linguistic features without keyword cues, while conventional classifiers depend heavily on explicit mental health terms. We then developed PsychoGPT, an LLM built on DSM-5 and PHQ-8 guidelines that performs initial distress classification, symptom-level scoring, and reconciliation with external ratings for explainable assessment. To reduce hallucinations, we proposed Stacked Multi-Model Reasoning (SMMR), layering expert models where early layers handle localized subtasks and later layers reconcile findings, outperforming single-model solutions on DAIC-WOZ in accuracy, F1, and PHQ-8 scoring. Finally, a cohesive framework unifies these tools, enabling adaptive survey insights to flow directly into specialized mental health detection models. 

---
# Evolving-RL: End-to-End Optimization of Experience-Driven Self-Evolving Capability within Agents 

**Authors**: Zhiyuan Fan, Wenwei Jin, Feng Zhang, Bin Li, Yihong Dong, Yao Hu, Jiawei Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.10663)  

**Abstract**: Experience-driven self-evolving agents aim to overcome the static nature of large language models by distilling reusable experience from past interactions, thus enabling adaptation to novel tasks at deployment time. This process places substantial demands on the foundation model's capacities for abstraction, generalization, and in-context learning. However, most existing studies focus primarily on system-level design choices, such as how experience is represented and managed, neglecting the inherent capabilities of the underlying model. While some recent works have started to optimize the experience utilization stage via reinforcement learning, they still fail to treat self-evolution as a unified process to be jointly optimized. To this end, we propose Evolving-RL, an efficient algorithmic framework that jointly improves the experience extraction and utilization capabilities required for self-evolution. Specifically, we center the learning process on experience extraction and evaluation, using the two supervisory signals derived from evaluation to optimize the extractor and solver separately and thus enable their coordinated co-evolution. Experiments on ALFWorld and Mind2Web show that Evolving-RL effectively enhances LLMs' ability to extract and reuse experience, leading to strong performance gains on out-of-distribution tasks (up to 98.7% relative improvement over the GRPO baseline on ALFWorld unseen tasks and 35.8% on Mind2Web), and these gains are fully unlocked only through the coordinated co-evolution of experience extraction and utilization. Furthermore, Evolving-RL inherently functions as an experience-augmented RL algorithm. By internalizing reusable experience patterns directly into model parameters, it achieves remarkable performance gains over standard baselines on both seen and unseen tasks, even in the absence of test-time experience accumulation. 

---
# SkillEvolver: Skill Learning as a Meta-Skill 

**Authors**: Genrui Zhang, Erle Zhu, Jinfeng Zhou, Caiyan Jia, Hongning Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.10500)  

**Abstract**: Agent skills today are static artifact: authored once -- by human curation or one-shot generation from parametric knowledge -- and then consumed unchanged, with no mechanism to improve from real use. We propose \textbf{SkillEvolver}, a lightweight, plug-and-play solution for online skill learning, in which a single meta-skill iteratively authors, deploys, and refines domain-specific skills. The learning target of SkillEvolver is the skill's prose and code, not model weights, so that the resulting artifact drops into any agent without retraining; and the meta-skill itself is just another skill, loaded through the same interface by any protocol-compliant CLI-agent. Unlike trace-distillation, the meta-skill refines only after deploying the learnt skill, such that the learning signal comes from failures another agent encounters while using it -- not from exploratory traces alone. Refinement iterations are governed by a fresh-agent overfit audit that catches possible leakage as well as deployed-skill-specific failures, including the silent-bypass mode in which a skill appears valid in content but is never invoked at runtime. On $83$ SkillsBench tasks spanning $15^{+}$ domains, SkillEvolver reaches $56.8\%$ accuracy versus $43.6\%$ for curated human skills and $29.9\%$ for the no-skill baseline; on three GPU kernel optimization tasks from KernelBench, it also raises mean speedup from $1.16$ to $1.51$ on average. 

---
# TRACE: Distilling Where It Matters via Token-Routed Self On-Policy Alignment 

**Authors**: Jiaxuan Wang, Xuan Ouyang, Zhiyu Chen, Yulan Hu, Zheng Pan, Xin Li, Lan-Zhe Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.10194)  

**Abstract**: On-policy self-distillation (self-OPD) densifies reinforcement learning with verifiable rewards (RLVR) by letting a policy teach itself under privileged context. We find that when this guidance spans the full response, all-token KL spends gradients on mostly redundant positions and amplifies privileged-information leakage, causing entropy rise, shortened reasoning, and out-of-distribution degradation in long-horizon math training. We propose Token-Routed Alignment for Critical rEasoning (TRACE), which distills only on annotator-marked critical spans: forward KL on key spans of correct rollouts, optional reverse KL on localized error spans, and GRPO on all remaining tokens, with the KL channel annealed away after a short warm-up. Our analysis explains TRACE through two effects: forward KL provides non-vanishing lift to teacher-supported tokens that the student under-allocates, while span masking and decay keep cumulative privileged-gradient exposure finite. On four held-out math benchmarks plus GPQA-Diamond, TRACE improves over GRPO by 2.76 percentage points on average and preserves the Qwen3-8B base OOD score on GPQA-Diamond, where GRPO and all-token self-OPD baselines degrade. Gains persist under online self-annotation (+1.90 percentage points, about 69% of the strong-API gain), reducing the concern that TRACE merely imports external annotator capability. Across scales, the best routed action is base-dependent: on Qwen3-8B it is forward KL on key spans, while on Qwen3-1.7B it shifts to reverse KL on error spans. 

---
# TMAS: Scaling Test-Time Compute via Multi-Agent Synergy 

**Authors**: George Wu, Nan Jing, Qing Yi, Chuan Hao, Ming Yang, Feng Chang, Yuan Wei, Jian Yang, Ran Tao, Bryan Dai  

**Link**: [PDF](https://arxiv.org/pdf/2605.10344)  

**Abstract**: Test-time scaling has become an effective paradigm for improving the reasoning ability of large language models by allocating additional computation during inference. Recent structured approaches have further advanced this paradigm by organizing inference across multiple trajectories, refinement rounds, and verification-based feedback. However, existing structured test-time scaling methods either weakly coordinate parallel reasoning trajectories or rely on noisy historical information without explicitly deciding what should be retained and reused, limiting their ability to balance exploration and exploitation. In this work, we propose TMAS, a framework for scaling test-time compute via multi-agent synergy. TMAS organizes inference as a collaborative process among specialized agents, enabling structured information flow across agents, trajectories, and refinement iterations. To support effective cross-trajectory collaboration, TMAS introduces hierarchical memories: the experience bank reuses low-level reliable intermediate conclusions and local feedback, while the guideline bank records previously explored high-level strategies to steer subsequent rollouts away from redundant reasoning patterns. Furthermore, we design a hybrid reward reinforcement learning scheme tailored to TMAS, which jointly preserves basic reasoning capability, enhances experience utilization, and encourages exploration beyond previously attempted solution strategies. Extensive experiments on challenging reasoning benchmarks demonstrate that TMAS achieves stronger iterative scaling than existing test-time scaling baselines, while hybrid reward training further improves scaling effectiveness and stability across iterations. Code and data are available at this https URL. 

---
# Verifiable Process Rewards for Agentic Reasoning 

**Authors**: Huining Yuan, Zelai Xu, Huaijie Wang, Xiangmin Yi, Jiaxuan Gao, Xiao-Ping Zhang, Yu Wang, Chao Yu, Yi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.10325)  

**Abstract**: Reinforcement learning from verifiable rewards (RLVR) has improved the reasoning abilities of large language models (LLMs), but most existing approaches rely on sparse outcome-level feedback. This sparsity creates a credit assignment challenge in long-horizon agentic reasoning: a trajectory may fail despite containing many correct intermediate decisions, or succeed despite containing flawed ones. In this work, we study a class of densely-verifiable agentic reasoning problems, where intermediate actions can be objectively checked by symbolic or algorithmic oracles. We propose Verifiable Process Rewards (VPR), a framework that converts such oracles into dense turn-level supervision for reinforcement learning, and instantiate it in three representative settings: search-based verification for dynamic deduction, constraint-based verification for logical reasoning, and posterior-based verification for probabilistic inference. We further provide a theoretical analysis showing that dense verifier-grounded rewards can improve long-horizon credit assignment by providing more localized learning signals, with the benefit depending on the reliability of the verifier. Empirically, VPR outperforms outcome-level reward and rollout-based process reward baselines across controlled environments, and more importantly, transfers to both general and agentic reasoning benchmarks, suggesting that verifiable process supervision can foster general reasoning skills applicable beyond the training environments. Our results indicate that VPR is a promising approach for enhancing LLM agents whenever reliable intermediate verification is available, while also highlighting its dependence on oracle quality and the open challenge of extending VPR to less structured, open-ended environments. 

---
# FormalRewardBench: A Benchmark for Formal Theorem Proving Reward Models 

**Authors**: Zeynel A. Uluşan, Burak S. Akbudak, Can S. Erer, Gözde Gül Şahin  

**Link**: [PDF](https://arxiv.org/pdf/2605.10141)  

**Abstract**: Recent neural theorem provers use reinforcement learning with verifiable rewards (RLVR), where proof assistants provide binary correctness signals. While verifiable rewards are cheap and scalable without reward hacking issues, they suffer from sparse credit assignment: models receive no learning signal from difficult problems where partial progress goes unrewarded. This motivates learned reward models that can evaluate proof quality beyond binary verification. However, comparing reward models is challenging since it typically requires expensive RL training ablations. To address this, we introduce \textbf{FormalRewardBench}, the first benchmark for evaluating reward models in formal theorem proving with Lean 4. Our benchmark consists of 250 preference pairs where correct proofs are paired with incorrect variants generated through five expert curated error injection strategies: forced mistakes, minimal single-point variations, verbose incorrect proofs, natural language justification, and Python code injection. We evaluate frontier LLMs (e.g., Claude Opus 4.5), judge LLMs (e.g., CompassJudger-1-14B), general-purpose LLMs (e.g., Qwen2.5-72B-Instruct), and specialized theorem proving models (e.g., DeepSeek-Prover-V2-7B).
Our results reveal that frontier LLMs achieve the highest performance (59.8\%) while specialized theorem provers perform the worst (24.4\%), suggesting that theorem proving ability does not transfer to proof evaluation. We provide further insights on various error injection mechanisms, highlighting the challenging nature of most injection mechanisms. We release \textbf{FormalRewardBench} publicly to encourage more research on developing reward models in formal mathematics. 

---
# MAGE: Multi-Agent Self-Evolution with Co-Evolutionary Knowledge Graphs 

**Authors**: Ruiyi Yang, Zechen Li, Hao Xue, Imran Razzak, Flora D. Salim  

**Link**: [PDF](https://arxiv.org/pdf/2605.10064)  

**Abstract**: Self-evolving language-model agents must decide what to learn next and how to preserve what they have learned across iterations. Existing systems typically carry this cross-iteration knowledge as natural-language feedback, flat episodic memory, or implicit reinforcement signals, none of which cleanly supports a frozen weak backbone at inference time. This paper introduces MAGE (Multi-Agent Graph-guided Evolution), a framework that externalizes self-knowledge into a four-subgraph co-evolutionary knowledge graph. Its experience subgraph stores both teacher-written failure corrections and the learner's own past correct reasoning traces, which are retrieved as task-conditioned guidance for a frozen execution model. During evolution, the graph, a task-level search bandit, and a skill-level routing bandit are updated from the same reward stream, while the learner's backbone remains unchanged. We further provide structural analysis showing how append-only memory growth, bounded curriculum coverage, and task-filtered retrieval together support stable improvement of the retrieval substrate for frozen-learner evolution. Across nine benchmarks spanning mathematical reasoning, multi-hop and open-domain question answering, spatio-temporal analysis, financial numerical reasoning, medical multiple-choice, an open-world survival game, and web navigation, MAGE achieves strong performance against prompt-based frozen-backbone baselines. Ablations show that self-harvested success traces and teacher-written corrections are complementary, with success memories contributing most on reasoning-template-heavy tasks and corrective memories supporting harder composition and interaction settings. 

---
# HAGE: Harnessing Agentic Memory via RL-Driven Weighted Graph Evolution 

**Authors**: Dongming Jiang, Yi Li, Guanpeng Li, Qiannan Li, Bingzhe Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.09942)  

**Abstract**: Memory retrieval in agentic large language model (LLM) systems is often treated as a static lookup problem, relying on flat vector search or fixed binary relational graphs. However, fixed graph structures cannot capture the varying strength, confidence, and query-dependent relevance of relationships between events. In this paper, we propose HAGE, a weighted multi-relational memory framework that reconceptualizes retrieval as sequential, query-conditioned traversal over a unified relational memory graph. Memory is organized as relation-specific graph views over shared memory nodes, where each edge is associated with a trainable relation feature vector encoding multiple relational signals. Given a query, an LLM-based classifier identifies the relational intent, and a routing network dynamically modulates the corresponding dimensions of the edge embedding. Traversal scores are computed via a learned combination of semantic similarity and these query-conditioned edge representations. This allows memory traversal to prioritize high-utility relational paths while softly suppressing noisy or weakly relevant connections. Beyond adaptive traversal, HAGE further introduces a reinforcement learning-based training framework that jointly optimizes routing behavior and edge representations using downstream tasks. Finally, empirical results demonstrate improved long-horizon reasoning accuracy and a favorable accuracy-efficiency trade-off compared to state-of-the-art agentic memory systems. Our code is available at this https URL. 

---
# Separate First, Fuse Later: Mitigating Cross-Modal Interference in Audio-Visual LLMs Reasoning with Modality-Specific Chain-of-Thought 

**Authors**: Xuanchen Li, Yuheng Lu, Chenrui Cui, Tianrui Wang, Zikang Huang, Yu Jiang, Long Zhou, Longbiao Wang, Jianwu Dang  

**Link**: [PDF](https://arxiv.org/pdf/2605.09906)  

**Abstract**: Audio and vision provide complementary evidence for audio-visual question answering, yet current audio-visual large language models may suffer from cross-modal interference: information from one modality misguides the interpretation of another, thereby inducing hallucinations. We attribute this issue to uncontrolled cross-modal interactions during intermediate reasoning. To mitigate this, we propose Separate First, Fuse Later (SFFL), an audio-visual reasoning framework designed to reduce cross-modal interference. SFFL enforces modality-specific chain-of-thought reasoning, producing separate audio and visual reasoning traces and integrating evidence for answering. We construct modality-preference labels via a data pipeline under different modality input settings. We use these labels as an auxiliary reward in reinforcement learning to encourage a instance-dependent preference for modality cues when answering. We further introduce a modality-specific reasoning mechanism that preserves modality isolation during the separated reasoning stage while enabling full access to cross-modal information at the evidence fusion stage. Experiments demonstrate consistent improvements in both accuracy and robustness, yielding an average relative gain of 5.16\% on general AVQA benchmarks and 11.17\% on a cross-modal hallucination benchmark. 

---
# expo: Exploration-prioritized policy optimization via adaptive kl regulation and gaussian curriculum sampling 

**Authors**: Mingxiong Lin, Zhangquan Gong, Maowen Tang, Qian Li, Chuangchuang Wang, Jian Ma, Sutian Huang, Kai Tang, Haonan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.09923)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has become the standard paradigm for LLM mathematical reasoning, where Group Relative Policy Optimization (GRPO) serves as the mainstream algorithm. We point out two understudied inefficiencies existing in GRPO. First, the fixed KL penalty coefficient overly restricts policy exploration at stages where the model requires significant deviation from the reference policy. Second, uniform sampling of training questions ignores that moderately difficult problems provide the most informative gradient signals for optimization. We propose Exploration-Prioritized Policy Optimization (EXPO) with two lightweight plug-in modules. The Accuracy-Conditioned KL Scaling (AKL) dynamically adjusts KL regularization strength through a smooth nonlinear function of batch average accuracy, relaxing the penalty when the model underperforms and strengthening it when the model achieves good results. The Gaussian Curriculum Sampling (GCS) assigns sampling weights to questions following a Gaussian distribution centered at moderate accuracy around 0.5, focusing training on the model's learning frontier. We conduct extensive experiments on DeepSeek-R1-Distill-Qwen-1.5B and Qwen3-8B-Base over six mathematical reasoning benchmarks. The results show EXPO steadily surpasses vanilla GRPO. It obtains an absolute gain of 13.34 on AIME 2025 pass@32, rising from 63.33 percent to 76.67 percent, and achieves an average pass@32 improvement of 2.66 on the 8B model. The much larger performance gains on pass@32 compared with pass@1 demonstrate that EXPO effectively enlarges the model's exploration boundary under a fixed inference cost budget. 

---
# From Passive Reuse to Active Reasoning: Grounding Large Language Models for Neuro-Symbolic Experience Replay 

**Authors**: Yanan Xiao, Yixiang Tang, Zechen Feng, Lu Jiang, Minghao Yin, Pengyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.09419)  

**Abstract**: While experience replay is essential for data efficiency in reinforcement learning (RL), standard methods treat the replay buffer as a passive memory system, prioritizing samples based on numerical prediction errors rather than their semantic significance. This approach stands in contrast to human learning, which accelerates mastery by actively abstracting fragmented experiences into behavioral rules. To bridge this gap, we propose Neuro-Symbolic Experience Replay (NSER), a framework that transforms experience replay from a passive sample reuse mechanism into an active engine for knowledge construction. Specifically, NSER addresses the incompatibility between linguistic reasoning and numerical optimization through a novel neuro-symbolic grounding pipeline. It leverages Large Language Models (LLMs) in a zero-shot manner to induce candidate behavioral rules from accumulated trajectories, grounds these insights into differentiable first-order logic representations, and utilizes the resulting symbolic structures to dynamically reweight the replay distribution. By allowing abstract knowledge to directly shape policy optimization, NSER achieves consistent superior sample efficiency and convergence speed across reactive, rule-based, and procedural benchmarks. 

---
# PiCA: Pivot-Based Credit Assignment for Search Agentic Reinforcement Learning 

**Authors**: Dongyi Liu, Yifan Niu, Qinwen Wang, Han Xiao, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.09287)  

**Abstract**: Large Language Model (LLM)-based search agents trained with reinforcement learning (RL) have significantly improved the performance of knowledge-intensive tasks. However, existing methods encounter critical challenges in long-horizon credit assignment: (i) Reward Sparsity, where models receive only outcome feedback without step-level guidance to differentiate action quality; (ii) Isolated Credit, where credit is assigned to steps independently, failing to capture sequential dependencies; and (iii) Distributional Shift, where rewards are estimated on templates that deviate from the model's natural generative distribution. To address these issues, we propose Pivot-Based Credit Assignment (PiCA), a novel step reward mechanism that reformulates the search trajectory as a sequential process of cumulative search progress. Unlike prior isolated step rewards, PiCA defines process rewards as success probabilities dependent on the historical context based on Potential-Based Reward Shaping (PBRS). This approach identifies pivot steps, which comprise target golden sub-queries and sub-answers derived from historical trajectories, as information peaks that significantly boost the likelihood of a correct final answer. By anchoring these step rewards to the final task objective, PiCA provides dense, pivot-aware and trajectory-dependent guidance while maintaining distributional consistency. Extensive experiments show that PiCA outperforms existing strong baselines across seven knowledge-intensive QA benchmarks, achieving 15.2% and 2.2% improvements for 3B and 7B models. The consistent performance gains across various models show PiCA's robust generalization. The code is available at this https URL. 

---
# SeePhys Pro: Diagnosing Modality Transfer and Blind-Training Effects in Multimodal RLVR for Physics Reasoning 

**Authors**: Kun Xiang, Terry Jingchen Zhang, Zirong Liu, Bokai Zhou, Yueling Tang, Junjie Yu, Jiacong Lu, Shangrui Huang, Heng Li, Likui Zhang, Kunkun Liu, Changzheng Zhang, Yangle Fang, Boqiang Guo, Hui-Ling Zhen, Dandan Tu, Yinya Huang, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2605.09266)  

**Abstract**: We introduce SeePhys Pro, a fine-grained modality transfer benchmark that studies whether models preserve the same reasoning capability when critical information is progressively transferred from text to image. Unlike standard vision-essential benchmarks that evaluate a single input form, SeePhys Pro features four semantically aligned variants for each problem with progressively increasing visual elements. Our evaluation shows that current frontier models are far from representation-invariant reasoners: performance degrades on average as information moves from language to diagrams, with visual variable grounding as the most critical bottleneck. Motivated by this inference-time fragility, we further develop large training corpora for multimodal RLVR and use blind training as a diagnostic control, finding that RL with all training images masked can still improve performance on unmasked validation sets. To analyze this effect, text-deletion, image-mask-rate, and format-saturation controls suggest that such gains can arise from residual textual and distributional cues rather than valid visual evidence. Our results highlight the need to evaluate multimodal reasoning not only by final-answer accuracy, but also by robustness under modality transfer and by diagnostics that test whether improvements rely on task-critical visual evidence. 

---
# BoostAPR: Boosting Automated Program Repair via Execution-Grounded Reinforcement Learning with Dual Reward Models 

**Authors**: Yuanhao Li, Hongbo Wang, Xiaotang Shang, Xunzhu Tang, Yiming Cao, Xuhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.09134)  

**Abstract**: Reinforcement learning for program repair is hindered by sparse execution feedback and coarse sequence-level rewards that obscure which edits actually fix bugs. We present BoostAPR, a three-stage framework addressing these challenges: (1) supervised fine-tuning on execution-verified demonstrations with reasoning traces, (2) training dual reward models--a sequence-level assessor and a line-level credit allocator--from execution outcomes, and (3) PPO optimization where the line-level model redistributes rewards to critical edit regions. This line-level credit assignment operates at an intermediate granularity naturally suited to code changes. Trained on SWE-Gym and evaluated on four benchmarks, BoostAPR achieves 40.7% on SWE-bench Verified (+22.9pp over base model), 24.8% on Defects4J (Python-to-Java transfer), 84.5% on HumanEval-Java, and 95.0% on QuixBugs, achieving competitive results among open-source models with strong cross-language generalization. 

---
# Agentic AI Scientists Are Not Built For Autonomous Scientific Discovery 

**Authors**: Harshit Bisht, Vinay Kumar, Kevin Maik Jablonka, Mausam, N. M. Anoop Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2605.08956)  

**Abstract**: A growing body of work pursues AI scientists capable of end-to-end autonomous scientific discovery. This position paper argues that although they already function as co-scientists, agentic AI scientists are not built for autonomous scientific discovery. We identify the following challenges in building and deploying autonomous AI scientists: (1) Problem selection is influenced by the McNamara fallacy; (2) Agents are built on large language models (LLMs) whose training corpora omit tacit procedural and failure knowledge of laboratory practice; (3) Preference optimisation during post-training compresses output diversity toward consensus; and (4) Most scientific benchmarks measure single-turn prediction accuracy and lack feedback from physical experiments back to the computational model. These challenges are not just questions of scale and scaffolding; they require revisiting fundamental design choices. To build truly autonomous AI scientists, we recommend the use of scientific simulations as verifiers for training, the design of persistent world models that represent the shifting objectives governing real investigations, the establishment of a centralized preregistration repository for all AI-generated hypotheses, and application driven by scientific need rather than tool affordance. 

---
# Learning to Explore: Scaling Agentic Reasoning via Exploration-Aware Policy Optimization 

**Authors**: Xingyuan Hua, Sheng Yue, Ju Ren  

**Link**: [PDF](https://arxiv.org/pdf/2605.08978)  

**Abstract**: Recent advancements in agentic test-time scaling allow models to gather environmental feedback before committing to final actions. A key limitation of existing methods is that they typically employ undifferentiated exploration strategies, lacking the ability to adaptively distinguish when exploration is truly required. In this paper, we propose an exploration-aware reinforcement learning framework that enables LLM agents to adaptively explore only when uncertainty is high. Our method introduces a fine-grained reward function via variational inference that explicitly evaluates exploratory actions by estimating their potential to improve future decision-making, together with an exploration-aware grouping mechanism that separates exploratory actions from task-completion actions during optimization. By targeting informational gaps, this design allows agents to explore selectively and transition to execution as soon as the task context is clear. Empirically, we demonstrate that our approach achieves consistent improvements across a range of challenging text-based and GUI-based agent benchmarks. Code is available at \url{this https URL} and models are available at this https URL. 

---
# Forge: Quality-Aware Reinforcement Learning for NP-Hard Optimization in LLMs 

**Authors**: Xiaozhe Li, Xinyu Fang, Shengyuan Ding, Yang Li, Linyang Li, Haodong Duan, Qingwen Liu, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.08905)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success on reasoning benchmarks through Reinforcement Learning with Verifiable Rewards (RLVR), excelling at tasks such as math, coding, logic, and puzzles. However, existing benchmarks evaluate only correctness, while overlooking optimality, namely the ability to find the best solutions under constraints. We propose OPT-BENCH, the first comprehensive framework for training and evaluating LLMs on NP-hard optimization problems through quality-aware RLVR. OPT-BENCH provides three key components: a scalable training infrastructure with instance generators, quality verifiers, and optimal baselines across 10 tasks; a rigorous benchmark with 1,000 instances evaluating both feasibility, measured by Success Rate, and quality, measured by Quality Ratio; and quality-aware rewards that enable continuous improvement beyond binary correctness. Training on Qwen2.5-7B-Instruct-1M with 15K examples achieves 93.1% SR and 46.6% QR, significantly outperforming GPT-4o, which achieves 29.6% SR and 14.6% QR. Beyond optimization, training on OPT-BENCH transfers to diverse tasks, including mathematics (+2.2%), logic (+1.2%), knowledge (+4.1%), and instruction following (+6.1%). Our analysis reveals that quality-aware rewards improve solutions by 28.8% over binary rewards, and that task diversity drives generalization more than data quantity, offering insights into RLVR scaling for complex reasoning. 

---
# How You Begin is How You Reason: Driving Exploration in RLVR via Prefix-Tuned Priors 

**Authors**: Yifan Xu, Junren Chen, Yifan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.08817)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) recently thrives in large language model (LLM) reasoning tasks. However, the reward sparsity and the long reasoning horizon make effective exploration challenging. In practice, this challenge manifests as the \emph{entropy collapse} phenomenon, where RLVR improves single-rollout accuracy but fails to expand coverage on successful reasoning trajectories. Passive exploration techniques like entropy regularization tend to dismiss generation quality, resulting in noisy rollouts. In response to this issue, we propose an Information-Maximizing Augmented eXploration (IMAX) framework to train a pool of soft prefixes that reshapes the base model's prior over reasoning trajectories. Rather than relying on RL to incentivize exploration on top of the base model, each prefix acts as a trainable control knob that induces a distinct rollout distribution from the same backbone model. To encourage discovery of diverse and task-relevant reasoning behaviors, we derive an Information Maximization (InfoMax) reward to complement the verifiable rewards for RL training. IMAX is in general algorithm-agnostic and can be seamlessly integrated into existing RLVR pipelines. Experiment results have shown that across three backbone scales, IMAX consistently improves reasoning performance over standard RLVR, with gains up to 11.60\% in Pass@4 and 10.57\% in Avg@4. 

---
# Not All Turns Matter: Credit Assignment for Multi-Turn Jailbreaking 

**Authors**: Zhida He, Xiaoyu Wen, Han Qi, Ziyuan Zhou, Peng Yu, Xingcheng Xu, Dongrui Liu, Xia Hu, Chaochao Lu, Qiaosheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.08778)  

**Abstract**: Deploying LLMs in multi-turn dialogues facilitates jailbreak attacks that distribute harmful intent across seemingly benign turns. Recent training-based multi-turn jailbreak methods learn long-horizon attack strategies from interaction feedback, but often rely on coarse trajectory-level outcome signals that broadcast uniformly to every turn. However, we find that turn-level contributions in multi-turn jailbreaking are non-uniform, phase-dependent, and target-specific. Such coarse outcome supervision induces a credit assignment problem, leading to over-rewarding redundant turns in successful trajectories and under-crediting useful intermediate turns in failed ones. To address this, we propose TRACE, a turn-aware credit assignment framework for reinforcement learning (RL)-based multi-turn jailbreaking. For successful trajectories, TRACE estimates turn-level contributions via leave-one-turn-out semantic masking; for failed ones, TRACE assigns penalties based on prompt harmfulness and semantic relevance, with an additional local refusal-aware penalty. Furthermore, we reuse the attack-side credit signal for multi-turn defense alignment. Extensive experiments on open-source and closed-source targets show that TRACE achieves strong overall performance in effectiveness, transferability, and efficiency, yielding about a 25% relative improvement in attack success rate over the strongest RL baseline while also improving the safety-utility balance when reused for defense alignment. 

---
# AHD Agent: Agentic Reinforcement Learning for Automatic Heuristic Design 

**Authors**: Haoze Lv, Ning Lu, Ziang Zhou, Shengcai Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.08756)  

**Abstract**: Automatic heuristic design (AHD) has emerged as a promising paradigm for solving NP-hard combinatorial optimization problems (COPs). Recent works show that large language models (LLMs), when integrated into well-designed frameworks (i.e., LLM-AHD), can autonomously discover high-performing heuristics. However, existing LLM-AHD frameworks typically treat LLMs as passive generators within fixed workflows, where the model generates heuristics from manually designed, limited context. Such context may fail to capture state-dependent information (e.g., specific failure modes), leading to inefficient trial-and-error exploration. To overcome these limitations, we propose AHD Agent, a novel tool-integrated, multi-turn framework that empowers LLMs to proactively decide whether to generate heuristics or invoke tools to retrieve targeted evidence from the solving environment. To effectively train such a dynamic decision-making agent, we introduce an agentic reinforcement learning (RL) system, which leverages a novel environment synthesis pipeline to optimize a compact model's generalizable AHD capabilities. Experiments across eight diverse domains, including four held-out tasks, demonstrate that our 4B-parameter agent matches or surpasses state-of-the-art baselines using much larger models, while requiring significantly fewer evaluations. Model and inference scaling analysis further reveals that AHD Agent offers an effective trajectory toward truly autonomous heuristic design. 

---
# RewardHarness: Self-Evolving Agentic Post-Training 

**Authors**: Yuxuan Zhang, Penghui Du, Bo Li, Cong Wei, Junwen Miao, Huaisong Zhang, Songcheng Cai, Yubo Wang, Dongfu Jiang, Yuyu Zhang, Ping Nie, Wenhu Chen, Changqian Yu, Kelsey R. Allen  

**Link**: [PDF](https://arxiv.org/pdf/2605.08703)  

**Abstract**: Evaluating instruction-guided image edits requires rewards that reflect subtle human preferences, yet current reward models typically depend on large-scale preference annotation and additional model training. This creates a data-efficiency gap: humans can often infer the target evaluation criteria from only a few examples, while models are usually trained on hundreds of thousands of comparisons. We present RewardHarness, a self-evolving agentic reward framework that reframes reward modeling as context evolution rather than weight optimization. Instead of learning from large-scale annotations, RewardHarness aligns with human preferences by iteratively evolving a library of tools and skills from as few as 100 preference demonstrations. Given a source image, candidate edited images, and an editing instruction, an Orchestrator selects the most relevant subset of tools and skills from the maintained library, and a frozen Sub-Agent uses them to construct a reasoning chain that produces a preference judgment. By comparing predicted judgments with ground-truth preferences and analyzing successes and failures in the reasoning process, the Orchestrator automatically refines its library of tools and skills without additional human annotation. Using only 0.05% of the EditReward preference data, RewardHarness achieves 47.4% average accuracy on image-editing evaluation benchmarks, surpassing GPT-5 by 5.3 points. When used as a reward signal for GRPO fine-tuning, RL-tuned models achieve 3.52 on ImgEdit-Bench. Project page: this https URL. 

---
# SkillMaster: Toward Autonomous Skill Mastery in LLM Agents 

**Authors**: Min Yang, Jinghua Piao, Xu Xia, Xiaochong Lan, Jiaju Chen, Yongshun Gong, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.08693)  

**Abstract**: Skills provide an effective mechanism for improving LLM agents on complex tasks, yet in existing agent frameworks, their creation, refinement, and selection are typically governed by external teachers, hand-designed rules, or auxiliary modules. As a result, skills remain external resources to be invoked, rather than capabilities that agents can develop, adapt, and internalize through experience. To endow LLM agents with autonomous skill mastery, we propose SkillMaster, a training framework that teaches agents to create new skills, refine existing skills, and select accumulated skills during task solving. This capability is achieved through three key designs. First, we train agents through trajectory-informed skill review, teaching agents to propose, update, or retain skills based on evidence from completed episodes. Second, each candidate skill edit is designed to be evaluated by its counterfactual utility on related probe tasks, providing a direct learning signal for training skill-editing decisions. Third, we introduce DualAdv-GRPO, which separately estimates advantages for task-solving actions and skill-editing decisions, stabilizing joint training across task solving and skill management. Experiments on ALFWorld and WebShop show that SkillMaster improves the overall success rate over state-of-the-art baselines by 8.8% and 9.3%, respectively, achieving the best performance among all compared methods. Further analysis reveals a marked shift in agent capability: agents trained with SkillMaster can identify skill failures, refine procedural knowledge from trajectory evidence, and transfer improvements to future tasks with limited skill-bank edits. Overall, SkillMaster moves LLM agents beyond mere skill use toward self-improving agents capable of developing, adapting, and applying their own skill repertoires. 

---
# Self-ReSET: Learning to Self-Recover from Unsafe Reasoning Trajectories 

**Authors**: Dongcheng Zhang, Yi Zhang, Yuxin Chen, An Zhang, Xiang Wang, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.08936)  

**Abstract**: Large Reasoning Models possess remarkable capabilities for self-correction in general domain; however, they frequently struggle to recover from unsafe reasoning trajectories under adversarial attacks. Existing alignment methods attempt to mitigate this vulnerability by fine-tuning the model on expert data including reflection traces or adversarial prefixes. Crucially, these approaches are often hindered by static training data which inevitably deviate from model's dynamic, on-policy reasoning traces, resulting in model hardly covering its vast generation space and learning to recover from its own failures. To bridge this gap, we propose Self-ReSET, a pure reinforcement learning framework designed to equip LRMs with the intrinsic capacity to recover from their own safety error trajectories, which are subsequently reused as an initial state for reinforcement learning. Extensive experiments across various LRMs and benchmarks demonstrate that Self-ReSET significantly enhances robustness against adversarial attacks especially out-of-distribution (OOD) jailbreak prompts while maintaining general utility, along with efficient data utilization. Further analysis reveals that our method effectively fosters self-recovery patterns, enabling models to better identify and recover from unsafe intermediate error states back to benign paths. Our codes and data are available at this https URL. 

---
# Internalizing Safety Understanding in Large Reasoning Models via Verification 

**Authors**: Yi Zhang, Yuxin Chen, Leheng Sheng, Dongcheng Zhang, Chaochao Lu, Xiang Wang, An Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.08930)  

**Abstract**: While explicit Chain-of-Thought (CoT) empowers large reasoning models (LRMs), it enables the generation of riskier final answers. Current alignment paradigms primarily rely on externally enforced compliance, optimizing models to detect malicious prompts rather than evaluating the safety of their own outputs. We argue that this approach remains largely behavioral: our empirical analysis reveals that ostensibly aligned models lack intrinsic safety understanding, often failing to verify their own response safety and remaining vulnerable to adversarial jailbreaks. To address this fundamental limitation, we propose Safety Internal (SInternal), a framework that internalizes safety specifications by training LRMs exclusively on safety verification tasks to critique their own generated answers using expert reasoning trajectories. We demonstrate that learning to verify induces a strong generalization for response safety, significantly enhancing robustness against out-of-domain jailbreaks. Furthermore, when combined with reinforcement learning, SInternal serves as a superior initialization compared to standard supervised fine-tuning, suggesting that internalizing safety understanding creates a more robust foundation for alignment than merely mimicking safe behaviors. Our codes are available at this https URL 

---
# The Attacker in the Mirror: Breaking Self-Consistency in Safety via Anchored Bipolicy Self-Play 

**Authors**: Gabriele La Malfa, Emanuele La Malfa, Saar Cohen, Jie M. Zhang, Michael Luck, Michael Wooldridge, Elizabeth Black  

**Link**: [PDF](https://arxiv.org/pdf/2605.08427)  

**Abstract**: Self-play red team is an established approach to improving AI safety in which different instances of the same model play attacker and defender roles in a zero-sum game, i.e., where the attacker tries to jailbreak the defender; if self-play converges to a Nash equilibrium, the model is guaranteed to respond safely within the settings of the game. Although the parameter sharing enforced by the use of the same model for the two roles improves stability and performance, it introduces fundamental theoretical and architectural limitations. We show that the set of Nash equilibria that can be reached corresponds to a broad class of behaviours that includes trivial always refuse strategies and oracle-like defenders, thus limiting practical applicability. We then show that when attacker and defender share and update the same base model, the dynamics collapse to self-consistency, so that attacks do not enforce adversarial pressure on the defender. In response, we propose Anchored Bipolicy Self-Play, which trains distinct role-specific LoRA adapters on top of a frozen base model, thereby maintaining stable optimisation while preserving adversarial pressure through explicit role separation. In relation to standard self-play, we show up to 100x greater parameter efficiency than finetuning and consistent improvements in safety compared to self-play fine-tuned models. We evaluate on Qwen2.5-{3B, 7B,14B}-IT models across widely used safety benchmarks, showing improved robustness without loss of reasoning ability. Cross-play experiments further show that our attacker and defender models are superior to self-play in terms of adversarial defence and safety. 

---
# OracleTSC: Oracle-Informed Reward Hurdle and Uncertainty Regularization for Traffic Signal Control 

**Authors**: Darryl Jacob, Xinyu Liu, Muchao Ye, Xiaoyong Yuan, Pan He  

**Link**: [PDF](https://arxiv.org/pdf/2605.08516)  

**Abstract**: Transparent decision-making is essential for traffic signal control (TSC) systems to earn public trust. However, traditional reinforcement learning-based TSC methods function as black boxes with limited interpretability. Although large language models (LLMs) can provide natural language reasoning, reinforcement finetuning for TSC remains unstable because feedback is sparse and delayed, while most actions produce only marginal changes in congestion metrics. We introduce OracleTSC, which stabilizes LLM-based TSC through two mechanisms: (1) a reward hurdle mechanism that filters weak learning signals by subtracting a calibrated threshold from environmental rewards, and (2) uncertainty regularization that maximizes the probability of the selected response to encourage consistent decisions across sampled outputs. Experiments on the LibSignal benchmark show that OracleTSC enables a compact LLaMA3-8B model to substantially improve traffic efficiency, achieving a 75% reduction in travel time and a 67% decrease in queue length compared with the pretrained baseline while preserving interpretability through natural language explanations. OracleTSC also demonstrates strong cross-intersection generalization: a policy trained on one intersection transfers to a structurally different intersection with 17% lower travel time and 39% lower queue length without additional finetuning. These results suggest that uncertainty-aware reward shaping can improve the stability and effectiveness of reinforcement fine-tuning for TSC. 

---
# CoCoDA: Co-evolving Compositional DAG for Tool-Augmented Agents 

**Authors**: Ziyang Yu, Qiyue Li, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.08399)  

**Abstract**: Tool-augmented language models can extend small language models with external executable skills, but scaling the tool library creates a coupled challenge: the library must evolve with the planner as new reusable subroutines emerge, while retrieval from the growing library must remain within a fixed context budget. Existing tool-use and skill-library methods typically treat tools as flat or text-indexed memories, causing prompt cost to grow with library size and obscuring the typed, compositional structure of executable code. We propose CoCoDA, a framework that co-evolves the planner and tool library through a single code-native structure: a compositional code DAG. Nodes are primitive or composite tools, edges encode invocation dependencies, and each node stores a typed signature, description, pre/post-condition specification, and worked examples. At inference time, Typed DAG Retrieval prunes candidates by symbolic signature unification, ranks survivors by descriptions, filters them by behavioral specifications, and disambiguates with examples, keeping expensive context materialization on progressively smaller candidate sets. At training time, successful trajectories are folded into validated composite tools, while the planner is updated with a DAG-induced reward that credits composites by their primitive expansion size. We provide theoretical results showing retrieval cost reduction, sublinear retrieval time, compositional advantage under the shaped reward, monotone co-evolution under conservative updates, and DAG well-formedness. Across mathematical reasoning, tabular analysis, and code task benchmarks, CoCoDA enables an 8B student to match or exceed a 32B teacher on GSM8K and MATH and consistently improves over strong tool-use and library-learning baselines. 

---
# On Distinguishing Capability Elicitation from Capability Creation in Post-Training: A Free-Energy Perspective 

**Authors**: Yuhao Li, Shengchao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.08368)  

**Abstract**: Debates about large language model post-training often treat supervised fine-tuning (SFT) as imitation and reinforcement learning (RL) as discovery. But this distinction is too coarse. What matters is whether a training procedure increases the probability of behaviors the pretrained model could already produce, or whether it changes what the model can practically reach. We argue that post-training research should distinguish between capability elicitation and capability creation. We make this distinction operational by introducing the notion of accessible support: the set of behaviors that a model can practically produce under finite budgets. Post-training that reweights behaviors within this support is capability elicitation; whereas changing the support itself corresponds to capability creation. We develop this argument through a free-energy view of post-training. SFT and RL can both be seen as reweighting a pretrained reference distribution, only with different external signals. Demonstration signals define low-energy behavior for SFT, and reward signals define low-energy behavior for RL. When the update remains close to the base model, the main effect is local reweighting, not capability creation. Within this framework, the central question is no longer whether post-training is framed as SFT or RL, but whether it reweights behaviors already within reach, or instead expands the model's reachable behavioral space through search, interaction, tool use, or the incorporation of new information. 

---
# Mid-Training with Self-Generated Data Improves Reinforcement Learning in Language Models 

**Authors**: Aswin RRV, Jacob Dineen, Divij Handa, Mihir Parmar, Ben Zhou, Swaroop Mishra, Chitta Baral  

**Link**: [PDF](https://arxiv.org/pdf/2605.08472)  

**Abstract**: The effectiveness of Reinforcement Learning (RL) in Large Language Models (LLMs) depends on the nature and diversity of the data used before and during RL. In particular, reasoning problems can often be approached in multiple ways that rely on different forms of reasoning, and exposure to only a limited range of such approaches in the training data may limit the effectiveness of RL. Motivated by this, we investigate using diverse self-generated data during mid-training as an intermediate step before RL training. Specifically, we adopt a bootstrapped data-generation framework guided by George Polya's problem-solving approaches for generating multiple variants of correct answers for each question in the training data, and then perform fine-tuning. We first provide a theoretical perspective on how mid-training on such data improves RL and explain how policy-gradient updates can incentivize combining multiple approaches. We then empirically demonstrate that RL-trained models initialized with our mid-training data achieve consistent improvements across various mathematical reasoning benchmarks and other OOD tasks like code generation and narrative reasoning. Overall, our investigative study shows that a language model learning multiple problem-solving approaches, through self-generated data helps subsequent RL. 

---
# Alignment as Jurisprudence 

**Authors**: Nicholas Caputo  

**Link**: [PDF](https://arxiv.org/pdf/2605.08416)  

**Abstract**: Jurisprudence, the study of how judges should properly decide cases, and alignment, the science of getting AI models to conform to human values, share a fundamental structure. These seemingly distant fields both seek to predict and shape how decisions by powerful actors, in one case judges and in the other increasingly powerful artificial intelligences, will be made in the unknown future. And they use similar tools of the specification and interpretation of language to try to accomplish those goals. The great debates of jurisprudence, about what the law is and what it should be, can provide insight into alignment, and lessons from what does and does not work in alignment can help make progress in jurisprudence.
This essay puts the two fields directly into conversation. Drawing on leading accounts of jurisprudence, particularly Dworkin's principle-oriented interpretivism and Sunstein's positivist account of law as analogical reasoning, and on cutting-edge alignment approaches, namely Constitutional AI and case-based reasoning, it illustrates the value of a more sophisticated legally-inspired approach to the interplay of rules and cases in finetuning alignment and points to ways that AI can provide a better understanding of how the law works and how it can be improved by the introduction of AI. AI systems and the law should operate to empower people to act in the world, helping to expand their capabilities and the extent to which they are able to achieve their goals. As AI continues to improve in capacity, and as the constraints that legal theory places on human judges seem be coming undone, the conversation between these two fields will become increasingly essential and may help point to a better version of both. 

---
# MMVIAD: Multi-view Multi-task Video Understanding for Industrial Anomaly Detection 

**Authors**: Xiran Zhao, Jing Jin, Yan Bai, Zhongan Wang, Yifeng Sun, Yihang Lou, Xuanyu Zhu, Tao Feng, Yingna Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.10833)  

**Abstract**: Industrial anomaly detection is critical for manufacturing quality control, yet existing datasets mainly focus on static images or sparse views, which do not fully reflect continuous inspection processes in real industrial scenarios. We introduce MMVIAD (Multi-view Multi-task Video Industrial Anomaly Detection), to the best of our knowledge the first continuous multi-view video dataset for industrial anomaly detection and understanding, together with a benchmark for multi-task evaluation. MMVIAD contains object-centric 2-second inspection clips with approximately 120 degrees of camera motion, covering 48 object categories, 14 environments, and 6 structural anomaly types. It supports anomaly detection, defect classification, object classification, and anomaly visible-time localization. Systematic evaluations on MMVIAD show that current commercial and open-source video MLLMs remain far below human performance, especially for fine-grained defect recognition and temporal grounding. To improve transferable anomaly understanding, we further develop a two-stage post-training pipeline where PS-SFT (Perception-Structured Supervised Fine-Tuning) initializes perception-structured reasoning and VISTA-GRPO (Visibility-grounded Industrial Structured Temporal Anomaly Group Relative Policy Optimization) refines the model with semantic-gated defect reward and visibility-aware temporal reward, producing the final model VISTA. On MMVIAD-Unseen, VISTA improves the base model's average score across the four tasks from 45.0 to 57.5, surpassing GPT-5.4. Source code is available at this https URL. 

---
# MemQ: Integrating Q-Learning into Self-Evolving Memory Agents over Provenance DAGs 

**Authors**: Junwei Liao, Haoting Shi, Ruiwen Zhou, Jiaqian Wang, Shengtao Zhang, Wei Zhang, Weinan Zhang, Ying Wen, Zhiyu Li, Feiyu Xiong, Bo Tang, Muning Wen  

**Link**: [PDF](https://arxiv.org/pdf/2605.08374)  

**Abstract**: Episodic memory allows LLM agents to accumulate and retrieve experience, but current methods treat each memory independently, i.e., evaluating retrieval quality in isolation without accounting for the dependency chains through which memories enable the creation of future memories. We introduce MemQ, which applies TD($\lambda$) eligibility traces to memory Q-values, propagating credit backward through a provenance DAG that records which memories were retrieved when each new memory was created. Credit weight decays as $(\gamma\lambda)^d$ with DAG depth $d$, replacing temporal distance with structural proximity. We formalize the setting as an Exogenous-Context MDP, whose factored transition decouples the exogenous task stream from the endogenous memory store. Across six benchmarks, spanning OS interaction, function calling, code generation, multimodal reasoning, embodied reasoning, and expert-level QA, MemQ achieves the highest success rate on all six in generalization evaluation and runtime learning, with gains largest on multi-step tasks that produce deep and relevant provenance chains (up to +5.7~pp) and smallest on single-step classification (+0.77~pp) where single-step updates already suffice. We further study how $\gamma$ and $\lambda$ interact with the EC-MDP structure, providing principled guidance for parameter selection and future research. Code will be available soon. 

---
# Auto-Rubric as Reward: From Implicit Preferences to Explicit Multimodal Generative Criteria 

**Authors**: Juanxi Tian, Fengyuan Liu, Jiaming Han, Yilei Jiang, Yongliang Wu, Yesheng Liu, Haodong Li, Furong Xu, Wanhua Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.08354)  

**Abstract**: Aligning multimodal generative models with human preferences demands reward signals that respect the compositional, multi-dimensional structure of human judgment. Prevailing RLHF approaches reduce this structure to scalar or pairwise labels, collapsing nuanced preferences into opaque parametric proxies and exposing vulnerabilities to reward hacking. While recent Rubrics-as-Reward (RaR) methods attempt to recover this structure through explicit criteria, generating rubrics that are simultaneously reliable, scalable, and data-efficient remains an open problem. We introduce Auto-Rubric as Reward (ARR), a framework that reframes reward modeling from implicit weight optimization to explicit, criteria-based decomposition. Before any pairwise comparison, ARR externalizes a VLM's internalized preference knowledge as prompt-specific rubrics, translating holistic intent into independently verifiable quality dimensions. This conversion of implicit preference structure into inspectable, interpretable constraints substantially suppresses evaluation biases including positional bias, enabling both zero-shot deployment and few-shot conditioning on minimal supervision. To extend these gains into generative training, we propose Rubric Policy Optimization (RPO), which distills ARR's structured multi-dimensional evaluation into a robust binary reward, replacing opaque scalar regression with rubric-conditioned preference decisions that stabilize policy gradients. On text-to-image generation and image editing benchmarks, ARR-RPO outperforms pairwise reward models and VLM judges, demonstrating that explicitly externalizing implicit preference knowledge into structured rubrics achieves more reliable, data-efficient multimodal alignment, revealing that the bottleneck is the absence of a factorized interface, not a deficit of knowledge. 

---
# Step Rejection Fine-Tuning: A Practical Distillation Recipe 

**Authors**: Igor Slinko, Ilia Zavidnyi, Egor Bogomolov, Yaroslav Zharov  

**Link**: [PDF](https://arxiv.org/pdf/2605.10674)  

**Abstract**: Rejection Fine-Tuning (RFT) is a standard method for training LLM agents, where unsuccessful trajectories are discarded from the training set. In the context of SWE-bench tasks, this corresponds to filtering out runs where the submitted patch does not pass the tests. However, this approach discards unresolved trajectories, even though they form a large portion of all trajectories for hard tasks and even then may be partially correct. In this work, we propose Step Rejection Fine-Tuning (SRFT) - a practical way to leverage these unresolved trajectories. For this, we employ a critic LLM to assess the correctness of each step in a trajectory. Consequently, during training, we mask the loss for erroneous steps while retaining them in the context window. This way we ensure the model learns to recover from errors without reproducing them. Evaluation on SWE-bench Verified shows that while RFT improves the resolution rate by 2.4% by excluding unresolved trajectories, SRFT improves it by 3.7% by filtering them instead of discarding completely, reaching the total resolution rate of 32.2%. 

---
# Phoenix-VL 1.5 Medium Technical Report 

**Authors**: Team Phoenix, Arka Ray, Askar Ali Mohamed Jawad, Biondi Lee, Elijah Seah, Eva Lim, Fiona Teo, Grace Toh, Guang Xiang Teo, Jun En Tan, Jia Hui Bong, Jiale Wang, Jonathan Ng, Justin Tan, Kai Zhe Yew, Matthew Ong, Shun Yi Yeo, Wen Jett Lam, Wen Xiu Tan, Ze Yu Zhang, Gee Wah Ng, Chee Wee Ang, Mistral AI, Adrien Sadé, Guillaume Kunsch, Jia Sin Loh, Nicolas Schuhl, Rupert Menneer, Umar Jamil, Vincent Maladière, Yimu Pan  

**Link**: [PDF](https://arxiv.org/pdf/2605.10391)  

**Abstract**: We introduce Phoenix-VL 1.5 Medium, a 123B-parameter natively multimodal and multilingual foundation model, adapted to regional languages and the Singapore context. Developed as a sovereign AI asset, it demonstrates that deep domain adaptation can be achieved with minimal degradation to broad-spectrum intelligence and alignment. Continued pretraining was performed on Mistral Medium 3.1 using a localized 1-trillion tokens multimodal corpus, followed by a 250-billion tokens long-context extension phase. Subsequent post-training incorporated a novel human-annotated Singapore multimodal dataset and curated textual corpus on Singapore culture, knowledge, and legislation, totaling 22-billion tokens. An additional 5 billion tokens of model alignment was performed through Online Direct Preference Optimization. Phoenix-VL 1.5 Medium achieves state-of-the-art performance for its size on Singapore multimodal, legal, and government policy benchmarks while remaining globally competitive on general multimodal intelligence, multilingual, and STEM benchmarks. We also introduce a novel evaluation suite encompassing localized knowledge benchmarks and an institutionally aligned model behavior and safety framework. We report the data curation principles, training methodology, and highlight benchmark and inference performance. 

---
# EvoStreaming: Your Offline Video Model Is a Natively Streaming Assistant 

**Authors**: Zichen Wen, Boxue Yang, Junlong Ke, Jiajie Huang, Chenfei Liao, Junxi Wang, Xuyang Liu, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.10343)  

**Abstract**: Streaming video understanding demands more than watching longer videos: assistants must decide when to speak in real time, balancing responsiveness against verbosity. Yet most video-language models (VideoLLMs) are trained for offline inference, and existing streaming benchmarks externalize this timing decision to the evaluator. We address this gap with RealStreamEval, a frame-level multi-turn evaluation protocol that exposes models to sequential observations and penalizes unnecessary responses. Under this protocol, we observed that strong offline VideoLLMs retain useful visual understanding but lack an interaction policy for deciding when to respond. Motivated by this observation, we propose EvoStreaming, a self-evolved streaming adaptation framework in which the base model itself acts as data generator, relevance annotator, and roll-out policy to synthesize streaming trajectories without external supervision. With only $1{,}000$ self-generated samples ($139\times$ less than the leading streaming instruction-tuning approach) and no architectural changes, EvoStreaming consistently improves the overall RealStreamEval score by up to $10.8$ points across five open VideoLLM backbones (Qwen2/2.5/3-VL, InternVL-3.5, MiniCPM-V4.5) while largely preserving offline video performance. These results suggest that data-efficient interaction tuning is a practical path for adapting existing VideoLLMs to streaming assistants. 

---
# DeepRefine: Agent-Compiled Knowledge Refinement via Reinforcement Learning 

**Authors**: Haoyu Huang, Jiaxin Bai, Shujie Liu, Yang Wei, Hong Ting Tsang, Yisen Gao, Zhongwei Xie, Yufei Li, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.10488)  

**Abstract**: Agent-compiled knowledge bases provide persistent external knowledge for large language model (LLM) agents in open-ended, knowledge-intensive downstream tasks. Yet their quality is systematically limited by \emph{incompleteness}, \emph{incorrectness}, and \emph{redundancy}, manifested as missing evidence or cross-document links, low-confidence or imprecise claims, and ambiguous or coreference resolution issues. Such defects compound under iterative use, degrading retrieval fidelity and downstream task performance. We present \textbf{DeepRefine}, a general LLM-based reasoning model for \emph{agent-compiled knowledge refinement} that improves the quality of any pre-constructed knowledge bases with user queries to make it more suitable for the downstream tasks. DeepRefine performs multi-turn interactions with the knowledge base and conducts abductive diagnosis over interaction history, localizes likely defects, and executes targeted refinement actions for incremental knowledge base updates. To optimize refinement policies of DeepRefine without gold references, we introduce a Gain-Beyond-Draft (GBD) reward and train the reasoning process end-to-end via reinforcement learning. Extensive experiments demonstrate consistent downstream gains over strong baselines. 

---
# MemReread: Enhancing Agentic Long-Context Reasoning via Memory-Guided Rereading 

**Authors**: Baibei Ji, Xiaoyang Weng, Juntao Li, Zecheng Tang, Yihang Lou, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.10268)  

**Abstract**: To tackle long-context reasoning tasks without the quadratic complexity of standard attention mechanisms, approaches based on agent memory have emerged, which typically maintain a dynamically updated memory when linearly processing document chunks. To mitigate the potential loss of latent evidence in this memorize-while-reading paradigm, recent works have integrated retrieval modules that allow agents to recall information previously discarded during memory overwriting. However, retrieval-based recall suffers from both evidence loss during memory formation and interference induced by invalid queries. To overcome these limitations, we propose MemReread. Built upon streaming reading, MemReread circumvents intermediate retrieval. It triggers question decomposition and rereading when the final memory is insufficient, enabling the recovery of indirect facts that were prematurely discarded. This design supports non-linear reasoning while preserving the inherent logical flow of document comprehension. To further enhance practicality, we introduce a reinforcement learning framework that enhances length extrapolation capability while dynamically determining the number of rereading passes based on task complexity, thereby flexibly controlling computational overhead. Extensive experiments demonstrate that MemReread consistently outperforms baseline frameworks on long-context reasoning tasks, while maintaining linear time complexity with respect to context length. 

---
# Personalizing LLMs with Binary Feedback: A Preference-Corrected Optimization Framework 

**Authors**: Xilai Ma, Liye Zhao, Weijun Yao, Haibing Di, Wenya Wang, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.10043)  

**Abstract**: Large Language Model (LLM) personalization aims to align model behaviors with individual user preferences. Existing methods often focus on isolated user histories, neglecting the essential role of inter-user differences. We propose C-BPO, a framework that personalizes LLMs via preference-calibrated binary signals. By treating target user data as positive feedback and other users' data as an auxiliary set of implicit negative signals, C-BPO captures distinct inter-user differences. To mitigate the preference overlap issue, where shared task knowledge is erroneously penalized, we derive an objective grounded in Positive-Unlabeled (PU) learning theory. This approach purifies negative signals by subtracting ``positive bias'', ensuring alignment with unique idiosyncrasies without compromising general helpfulness. Empirical experiments across various personalization tasks and backbone LLMs show C-BPO consistently outperforms baselines, demonstrating the efficacy of preference-calibrated binary signals in modeling inter-user differences. 

---
# Continual Harness: Online Adaptation for Self-Improving Foundation Agents 

**Authors**: Seth Karten, Joel Zhang, Tersoo Upaa Jr, Ruirong Feng, Wenzhe Li, Chengshuai Shi, Chi Jin, Kiran Vodrahalli  

**Link**: [PDF](https://arxiv.org/pdf/2605.09998)  

**Abstract**: Coding harnesses such as Claude Code and OpenHands wrap foundation models with tools, memory, and planning, but no equivalent exists for embodied agents' long-horizon partial-observability decision-making. We first report our Gemini Plays Pokemon (GPP) experiments. With iterative human-in-the-loop harness refinement, GPP became the first AI system to complete Pokemon Blue, Yellow Legacy on hard mode, and Crystal without a lost battle. In the hardest stages, the agent itself began iterating on its strategy through long-context memory, surfacing emergent self-improvement signals alongside human-in-the-loop refinement. Continual Harness removes the human fully from this loop: a reset-free self-improving harness for embodied agents that formalizes and automates what we observed. Starting from only a minimal environment interface, the agent alternates between acting and refining its own prompt, sub-agents, skills, and memory, drawing on any past trajectory data. Prompt-optimization methods require episode resets; Continual Harness adapts online within a single run. On Pokemon Red and Emerald across frontier models, Continual Harness starting from scratch substantially reduces button-press cost relative to the minimalist baseline and recovers a majority of the gap to a hand-engineered expert harness, with capability-dependent gains, despite starting from the same raw interface with no curated knowledge, no hand-crafted tools, and no domain scaffolding. We then close the loop with the model itself: an online process-reward co-learning loop, in which an open-source agent's rollouts through the refining harness are relabeled by a frontier teacher and used to update the model, drives sustained in-game milestone progress on Pokemon Red without resetting the environment between training iterations. 

---
# G-Zero: Self-Play for Open-Ended Generation from Zero Data 

**Authors**: Chengsong Huang, Haolin Liu, Tong Zheng, Runpeng Dai, Langlin Huang, Jinyuan Li, Zongxia Li, Zhepei Wei, Yu Meng, Jiaxin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.09959)  

**Abstract**: Self-evolving LLMs excel in verifiable domains but struggle in open-ended tasks, where reliance on proxy LLM judges introduces capability bottlenecks and reward hacking. To overcome this, we introduce G-Zero, a verifier-free, co-evolutionary framework for autonomous self-improvement. Our core innovation is Hint-$\delta$, an intrinsic reward that quantifies the predictive shift between a Generator model's unassisted response and its response conditioned on a self-generated hint. Using this signal, a Proposer model is trained via GRPO to continuously target the Generator's blind spots by synthesizing challenging queries and informative hints. The Generator is concurrently optimized via DPO to internalize these hint-guided improvements. Theoretically, we prove a best-iterate suboptimality guarantee for an idealized standard-DPO version of G-Zero, provided that the Proposer induces sufficient exploration coverage and the data filteration keeps pseudo-label score noise low. By deriving supervision entirely from internal distributional dynamics, G-Zero bypasses the capability ceilings of external judges, providing a scalable, robust pathway for continuous LLM self-evolution across unverifiable domains. 

---
# Team-Based Self-Play With Dual Adaptive Weighting for Fine-Tuning LLMs 

**Authors**: Wu Li, Yigeng Zhou, Zesheng Shi, Yequan Wang, Min Zhang, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.09922)  

**Abstract**: While recent self-training approaches have reduced reliance on human-labeled data for aligning LLMs, they still face critical limitations: (i) sensitivity to synthetic data quality, leading to instability and bias amplification in iterative training; (ii) ineffective optimization due to a diminishing gap between positive and negative responses over successive training iterations. In this paper, we propose Team-based self-Play with dual Adaptive Weighting (TPAW), a novel self-play algorithm designed to improve alignment in a fully self-supervised setting. TPAW adopts a team-based framework in which the current policy model both collaborates with and competes against historical checkpoints, promoting more stable and efficient optimization. To further enhance learning, we design two adaptive weighting mechanisms: (i) a response reweighting scheme that adjusts the importance of target responses, and (ii) a player weighting strategy that dynamically modulates each team member's contribution during training. Initialized from a SFT model, TPAW iteratively refines alignment without requiring additional human supervision. Experimental results demonstrate that TPAW consistently outperforms existing baselines across various base models and LLM benchmarks. Our code is publicly available at this https URL. 

---
# LEAD: Length-Efficient Adaptive and Dynamic Reasoning for Large Language Models 

**Authors**: Songtao Wei, Yi Li, Zhikai Li, Xu Hu, Yuede Ji, Guanpeng Li, Feng Chen, Carl Yang, Zhichun Guo, Bingzhe Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.09806)  

**Abstract**: Large reasoning models, such as OpenAI o1 and DeepSeek-R1, tend to become increasingly verbose as their reasoning capabilities improve. These inflated Chain-of-Thought (CoT) trajectories often exceed what the underlying problems require, wasting compute, latency, and context budgets. While introducing length-based efficiency rewards during reinforcement learning offers a natural remedy, existing methods struggle with two fundamental challenges: the optimal balance between correctness and efficiency is non-stationary throughout training, and intrinsic reasoning budgets vary drastically across problems. Relying on static reward weights and global length constraints inevitably forces a compromise between degraded accuracy and unrealized compression. To overcome these limitations, we propose LEAD (Length-Efficient Adaptive and Dynamic reasoning), a method that replaces static heuristics with online, self-adaptive mechanisms. LEAD dynamically calibrates the correctness-efficiency trade-off at each step using a Potential-Scaled Instability, directing optimization capacity to the most informative learning signal. Furthermore, it estimates an adaptive per-problem target length online based on the model's own correct rollouts, applying a symmetric efficiency reward that penalizes both overthinking and over-compression. Evaluated on five mathematical reasoning benchmarks, LEAD achieves the highest accuracy and Accuracy-Efficiency Score among RL-trained efficient-reasoning methods while producing substantially shorter outputs than the base model. 

---
# Verifier-Free RL for LLMs via Intrinsic Gradient-Norm Reward 

**Authors**: Xuexiang Wen, Hang Yu, Linchao Zhu, Gaoang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.09920)  

**Abstract**: While Reinforcement Learning with Verifiable Rewards (RLVR) has recently emerged as a promising post-training paradigm for Large Language Models (LLMs), its dependency on the gold label or domain-specific verifiers limits its scalability to new tasks and domains. In this work, we propose Verifier-free Intrinsic Gradient-Norm Reward (VIGOR), a simple reward that uses only the policy model itself. Given a prompt, VIGOR samples a group of completions and assigns higher within-group rewards to outputs that induce smaller $\ell_2$ norms of the teacher-forced negative log-likelihood gradients under the current parameters. Intuitively, lower gradient norms suggest the completion aligns better with the current policy, serving as an intrinsic preference signal for policy optimization. To make this intrinsic signal practical for RL, we correct the systematic length bias of averaged token-level gradients with a $\sqrt{T}$ scaling, and apply group-wise rank shaping to stabilize reward scales across prompts. Across mathematical reasoning benchmarks, VIGOR outperforms the state-of-the-art Reinforcement Learning from Internal Feedback (RLIF) baseline, and it also exhibits cross-domain transfer to code benchmarks when trained only on math data. For instance, on Qwen2.5-7B-Base post-trained on MATH, VIGOR improves the average math accuracy by +3.31% and the average code accuracy by +1.91% over this baseline, while exhibiting more stable training dynamics. The code is available at this https URL. 

---
# EvoPref: Multi-Objective Evolutionary Optimization Discovers Diverse LLM Alignments Beyond Gradient Descent 

**Authors**: Dongxin Guo, Jikun Wu, Siu Ming Yiu  

**Link**: [PDF](https://arxiv.org/pdf/2605.09777)  

**Abstract**: Gradient-based preference optimization methods for large language model (LLM) alignment suffer from preference collapse, converging to narrow behavioral modes while neglecting preference diversity. We introduce EvoPref, a multi-objective evolutionary algorithm that maintains populations of Low-Rank Adaptation (LoRA) adapters optimized across helpfulness, harmlessness, and honesty objectives using Non-dominated Sorting Genetic Algorithm II (NSGA-II) selection with archive-based diversity preservation.
Our primary contribution is demonstrating that population-based methods discover substantially more diverse alignments than gradient descent. On standard benchmarks, EvoPref improves preference coverage by 18% (median 82.5% vs. 70.0% for ORPO, $p<0.001$, Wilcoxon, $n=30$) and reduces collapse rates by 47% (11.0% vs. 20.6%, $p<0.001$), while achieving competitive alignment quality (median 75.5% RewardBench vs. 75.0% for ORPO, $p<0.05$). We provide theoretical motivation extending recent multi-objective evolutionary algorithm (MOEA) runtime analysis (Dang et al., 2025) suggesting why archive-based methods escape collapse more effectively than single-trajectory optimization.
Comprehensive comparisons against MOEA/D, SMS-EMOA, CMA-ES, and gradient baselines (DPO, IPO, KTO, ORPO) with rigorous statistical testing (Friedman with Holm correction, Vargha-Delaney effect sizes, median with IQR) confirm that multi-objective selection with diversity preservation is essential. This work establishes evolutionary optimization as a principled paradigm for diverse LLM alignment. 

---
# CLR-voyance: Reinforcing Open-Ended Reasoning for Inpatient Clinical Decision Support with Outcome-Aware Rubrics 

**Authors**: Aishik Nagar, Arun-Kumar Kaliya-Perumal, Yu-Hsuan Han, Andrew Sheng-Han Huang, Kristen Kee, Yushi Cao, Yiming Chen, Hongchao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.09584)  

**Abstract**: Inpatient clinical reasoning is a sequential decision under partial observability: the clinician sees the admission so far and must choose the next action whose downstream consequences are not yet visible. Existing clinical-LLM evaluations and RL rewards signals collapse this into closed-form retrieval, clinical journey leakage, or unanchored LLM-as-judge scoring. We introduce CLR-voyance, a framework that reformulates inpatient reasoning as a Partially Observable Markov Decision Process (POMDP) and supervises it with rewards that are simultaneously outcome-grounded and clinician-validated. We instantiate the formulation as CLR-POMDP, which partitions successful patient journeys into a policy-visible past and an oracle-only future. Using the past information, an oracle LLM generates a case-specific query-answer pair, and the first adaptive rubric for clinical reasoning which is verifiable in the future of the patient journey. These rubrics are used for both post-training and evaluation of models for inpatient clinical reasoning. We post-train Qwen3-8B and MedGemma-4B with GRPO followed by model merging, yielding state-of-the-art inpatient clinical reasoning while retaining generalist capabilities. CLR-voyance-8B achieves 84.91% on CLR-POMDP, ahead of frontier medical reasoning models like GPT-5 (77.83%) and MedGemma-27B (66.66%) and has comparable or better performance on existing medical benchmarks. To ensure a clinically meaningful setting, we conduct a large-scale clinician alignment study, where physicians curate per-case rubrics, grade candidate responses, and provide blinded pairwise preferences of model reasoning. This study provides insights on clinical LLM-as-a-judge and clinical preference-model selection, which can inform the community at large. CLR-voyance has been deployed for 6+ months at a partner public hospital, drafting thousands of reasoning-heavy inpatient notes. 

---
# Skill-R1: Agent Skill Evolution via Reinforcement Learning 

**Authors**: Yash Vishe, Rohan Surana, Xunyi Jiang, Zihan Huang, Xintong Li, Nikki Lijing Kuang, Tong Yu, Ryan A. Rossi, Jingbo Shang, Julian McAuley, Junda Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.09359)  

**Abstract**: Agentic large language models often rely on skills, reusable natural language procedures that guide planning, action, and tool use. In practice, skills are typically improved through prompt engineering or by aligning the task LLM itself, which is costly, model-specific, and often infeasible for closed-source models. Skill optimization is not a one-step problem but a recurrent process with two coupled levels of credit assignment: a useful skill must improve rollout quality under current conditioning, while a useful revision must turn observed outcomes into a better skill for the next round. We propose Skill-R1, a reinforcement learning framework for instance-level recurrent skill optimization from verifiable rewards. Rather than updating the task LLM, Skill-R1 trains a lightweight skill generator that conditions on the task context, prior rollouts, and their verified outcomes to produce skills that steer a frozen task LLM. This preserves black-box compatibility with both open- and closed-source models while making adaptation substantially cheaper than model-level updates. Skill-R1 proceeds over multiple generations: at each step, the current skill induces rollouts whose verified outcomes are fed back to produce the next revision. To optimize this recurrent process, we introduce a bi-level group-relative policy optimization objective combining intra-generation and inter-generation advantages. The intra-generation term compares rollouts under shared skill conditioning, while the inter-generation term rewards revisions that improve behavior across successive generations. Together, these provide a principled objective for directional skill evolution rather than one-shot self-refinement. Empirically, Skill-R1 achieves consistent gains over no-skill baselines and standard GRPO across benchmarks with verifiable rewards, with particularly strong improvements on complex, multi-step tasks. 

---
# Cornerstones or Stumbling Blocks? Deciphering the Rock Tokens in On-Policy Distillation 

**Authors**: Yuxuan Jiang, Runchao Li, Shubhashis Roy Dipta, Dawei Li, Zhao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.09253)  

**Abstract**: While recent work in Reinforcement Learning with Verifiable Rewards (RLVR) has shown that a small subset of critical tokens disproportionately drives reasoning gains, an analogous token-level understanding of On-Policy Distillation (OPD) remains largely unexplored. In this work, we investigate high-loss tokens, a token type that--as the most direct signal of student-teacher mismatch under OPD's per-token KL objective--should progressively diminish as training converges according to existing studies; however, our empirical analysis shows otherwise. Even after OPD training reaches apparent saturation, a substantial subset of tokens continues to exhibit persistently high loss; these tokens, which we term Rock Tokens, can account for up to 18\% of the tokens in generated outputs. Our investigation reveals two startling paradoxes. First, despite their high occurrence frequency providing a disproportionately large share of total gradient norms, Rock Tokens themselves remain stagnant throughout training, resisting teacher-driven corrections. Second, through causal intervention, we find that these tokens provide negligible functional contribution to the model's actual reasoning performance. These findings suggest that a vast amount of optimization bandwidth is spent on structural and discourse residuals that the student model cannot or need not internalize. By deconstructing these dynamics, we demonstrate that strategically bypassing these ``stumbling blocks'' can significantly streamline the alignment process, challenging the necessity of uniform token weighting and offering a more efficient paradigm for large-scale model distillation. 

---
# DARE: Difficulty-Adaptive Reinforcement Learning with Co-Evolved Difficulty Estimation 

**Authors**: Yang Zhou, Can Jin, Zihan Dong, Zhepeng Wang, Yanting Yang, Shiyu Zhao, Lei Li, Runxue Bao, Yaochen Xie, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2605.09188)  

**Abstract**: Reinforcement learning improves the reasoning ability of large language models but remains costly and sample-inefficient, as many rollouts provide weak learning signals. Difficulty-aware data selection methods attempt to address this by prioritizing moderately difficult prompts, yet our analysis reveals three limitations: difficulty estimates become inaccurate under policy drift, data selection alone yields limited final-performance gains, and inference efficiency remains largely unchanged. These findings suggest that efficient and effective RL requires more than filtering by difficulty: the policy should learn to solve hard tasks while producing concise responses for easy ones. To this end, we propose **Dare**, a unified framework that co-evolves difficulty estimation with the policy via self-normalized importance sampling, maintains diverse difficulty coverage through a symmetric Beta sampling distribution, and applies tailored training strategies across difficulty tiers with adaptive compute allocation. Extensive experiments across multiple models and domains demonstrate that **Dare** consistently outperforms existing methods in training efficiency, final effectiveness, and inference efficiency, producing more concise responses on easy tasks while improving correctness on hard ones. Code is available at this https URL. 

---
# AgentForesight: Online Auditing for Early Failure Prediction in Multi-Agent Systems 

**Authors**: Boxuan Zhang, Jianing Zhu, Zeru Shi, Dongfang Liu, Ruixiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2605.08715)  

**Abstract**: LLM-based multi-agent systems are increasingly deployed on long-horizon tasks, but a single decisive error is often accepted by downstream agents and cascades into trajectory-level failure. Existing work frames this as \emph{post-hoc failure attribution}, diagnosing the responsible agent and step after the trajectory has ended. However, this paradigm forfeits any opportunity to intervene while trajectory is still unfolding. In this work, we introduce AgentForesight, a framework that reframes this problem as online auditing: at each step of an unfolding trajectory, an auditor observes only the current prefix and must either continue the run or alarm at the earliest decisive error, without access to future steps. To this end, we curate AFTraj-2K, a corpus of agentic trajectories across Coding, Math, and Agentic domains, in which safe trajectories are retained under a strict curation pipeline and unsafe trajectories are annotated at the step of their decisive error via consensus among multiple LLM judges. Built on that, we develop AgentForesight-7B, a compact online auditor trained with a coarse-to-fine reinforcement learning recipe that first equips it with a risk-anticipation prior at the failure boundary on adjacent safe/unsafe prefix pairs, then sharpens this prior into precise step-level localization under a three-axis reward jointly targeting the what, where, and who of an audit verdict. Across AFTraj-2K and an external Who\&When benchmark, AgentForesight-7B outperforms leading proprietary models, including GPT-4.1 and DeepSeek-V4-Pro, achieving up to +19.9% performance gain and 3$\times$ lower step localization error, opening the loop from post-hoc failures detection to enabling deployment-time intervention. Project page: this https URL 

---
# MARLaaS: Multi-Tenant Asynchronous Reinforcement Learning as a Service 

**Authors**: Timothy Tin Long Yu, Gursimran Singh, Ge Shi, Hanieh Sadri, Yong Zhang, Zhenan Fan  

**Link**: [PDF](https://arxiv.org/pdf/2605.08527)  

**Abstract**: Reinforcement Learning from Verifiable Rewards (RLVR) has significantly improved the reasoning capabilities of large language models (LLMs), particularly in multi-turn agentic settings involving environment interaction like tool use. However, fine-tuning such models remains prohibitively expensive due to high computational requirements, limiting accessibility. We propose MARLaaS (Multi-tenant Asynchronous RL as a Service), a system for concurrent RL fine-tuning across multiple users and tasks. Our approach is based on two key ideas: (1) sharing a base model across tenants using lightweight LoRA adapters, and (2) a disaggregated asynchronous architecture that decouples rollout generation, environment interaction, and policy training into independently scheduled stages. This design enables tasks to progress through the RL pipeline at their own pace in an event-driven manner, reducing cross-task interference, idle time, and end-to-end latency. In multi-task settings (we report up to 32 concurrent tasks), MARLaaS achieves single-task state-of-the-art performance while improving accelerator utilization by up to 4.3x and reducing end-to-end training time by 85%. 

---
# DUET: Optimize Token-Budget Allocation for Reinforcement Learning with Verifiable Rewards 

**Authors**: Haoyu Hu, Xuandong Zhao, Xuhai "Orson'' Xu, Nori Jacoby  

**Link**: [PDF](https://arxiv.org/pdf/2605.08441)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) generates hundreds of thousands of tokens per training step, with rollout generation dominating the computational cost. The overall token budget can be controlled along two main dimensions: (i) deciding which prompts to allocate rollouts to, and (ii) deciding how long each rollout should be. Prior work has generally controlled only one of these dimensions at a time. We show that jointly tuning both decisions under a shared compute budget improves both reasoning quality and wall-clock training time. We instantiate this view as \textbf{DU}al-controlled tok\textbf{E}n alloca\textbf{T}ion (DUET), a computationally efficient layer over GRPO that uses a lightweight pre-rollout surrogate of prompt informativeness to set how many rollouts each prompt receives, and a marker-gated abort rule with importance reweighting to set when to stop them. On Qwen3-1.7B trained on MATH, DUET outperforms full-budget GRPO and the other three budget-aware baseline methods. DUET's advantage further generalizes to other benchmarks across math and coding, and is on par with the best baseline on the scientific Q\&A domain, while also achieving a $1.62\times$ wall-clock speedup. More notably, using only 50\% of the token budget, DUET still outperforms all baseline methods at their full budget, achieving an even higher $2.51\times$ speedup over full-budget GRPO. We verify the high performance of DUET on other backbone LLMs, including Qwen3-4B and Llama-3.2-3B-Instruct. Notably, the gap between DUET and the strongest baseline \emph{widens} as the budget tightens, contrary to the usual pattern in which efficient methods trade off quality as compute decreases. More broadly, these results suggest that DUET budget-aware control strategies are valuable not only for accelerating training, but also for improving the quality of the learning signal. 

---
# PYTHALAB-MERA: Validation-Grounded Memory, Retrieval, and Acceptance Control for Frozen-LLM Coding Agents 

**Authors**: Mehmet Iscan  

**Link**: [PDF](https://arxiv.org/pdf/2605.08468)  

**Abstract**: Local LLM-based coding agents increasingly work in settings where correctness is earned through execution feedback, persistent state, and bounded repair, not through a single fluent answer. Static retrieval, long-context prompting, self-refinement, execution-feedback repair, and reinforcement learning over model weights each address part of this setting, but they do not jointly provide validation-grounded episodic memory, adaptive retrieval-action selection, delayed credit assignment, and structural skill reuse around a frozen local model. We introduce PYTHALAB-MERA, a lightweight external controller for local validation-conditioned code generation. The frozen language model proposes complete source files; the controller decides which memory records and AST-derived skills should enter the next prompt, validates each candidate through a fail-fast pipeline, converts validation outcomes into bounded shaped rewards, and propagates delayed credit through TD(lambda)-style eligibility traces. We evaluate the implementation as a local CLI artifact on reinforcement-learning coding tasks with strict validation gates. In the measured hard RL setting with three tasks, three repetitions, and a three-attempt budget, PYTHALAB-MERA passed 8/9 strict validations; the self-refinement baseline and the investigated GRACE extension each passed 0/9. These results support a deliberately bounded claim: in this recorded setting, the external memory-and-retrieval controller improved validation success. They do not establish general-purpose code synthesis, state-of-the-art performance, formal program correctness, or formal safety. 

---
# Interactive Critique-Revision Training for Reliable Structured LLM Generation 

**Authors**: Fei Xu Yu, Zuyuan Zhang, Mahdi Imani, Nathaniel D. Bastian, Tian Lan  

**Link**: [PDF](https://arxiv.org/pdf/2605.08327)  

**Abstract**: In structured decision-making workflows such as form filling, compliance checking, and maintenance reporting, LLM outputs must be locally correct, globally consistent, and auditable against task-specific rules. Existing refinement methods often rely on heuristic debate, self-play, or LLM-generated supervision, creating a second-order assurance problem. We propose DPA-GRPO (Dual Paired-Action Group-Relative Policy Optimization), a paired-action training method for a two-player generator--verifier game with structured verifier interventions. The generator proposes outputs and may revise them when challenged; the verifier either remains silent or raises a safety assurance case (SAC) containing a claim, argument, and evidence. These SAC/no-SAC and KEEP/REVISE decisions induce paired counterfactual action groups, which DPA-GRPO uses for role-specific KL-regularized GRPO updates. We analyze the unregularized game and show that positive probability on strictly lower-reward intervention or revision actions creates a profitable unilateral deviation. Under standard stochastic-approximation assumptions, DPA-GRPO tracks the corresponding game ODE, whose isolated asymptotically stable limit points are stationary and candidate local equilibria under role-wise local optimality. Experiments on TaxCalcBench TY24 show that DPA-GRPO improves structured decision accuracy over zero-shot generation and generator-only RL baselines across Qwen3-4B and Qwen3-8B. Training increases correct silent acceptance, reduces missed errors, and improves calibrated revision behavior, indicating gains for both generator and verifier. 

---
# Reinforcement Learning for Scalable and Trustworthy Intelligent Systems 

**Authors**: Guangchen Lan  

**Link**: [PDF](https://arxiv.org/pdf/2605.08378)  

**Abstract**: Reinforcement learning has become a powerful paradigm for improving the capability of intelligent systems, but its practical deployment faces two central challenges. First, reinforcement learning must scale efficiently in distributed environments where communication bandwidth is limited and computation is heterogeneous across agents. Second, as reinforcement learning is increasingly used in post-training large language models and autonomous agents, the optimized policies must also be aligned with human preferences and satisfy safety requirements such as privacy-aware information disclosure. This dissertation addresses both challenges through four complementary contributions spanning federated optimization, preference alignment, and contextual safety.
The first part of the dissertation studies scalable reinforcement learning in federated settings. The second part of the dissertation studies trustworthy reinforcement learning for large language models. Together, these contributions advance reinforcement learning along two complementary dimensions. On the one hand, they make reinforcement learning more scalable through communication-efficient and asynchronous federated optimization. On the other hand, they make reinforcement learning more trustworthy by improving alignment with human preferences and by reducing contextually inappropriate information disclosure in language-based intelligent systems. As a whole, this dissertation argues that the next generation of intelligent systems will require both efficient optimization and trustworthy behavior, and that reinforcement learning provides a unifying framework for addressing both goals. 

---
# HTPO: Towards Exploration-Exploitation Balanced Policy Optimization via Hierarchical Token-level Objective Control 

**Authors**: Xincheng Yao, Ruoqi Li, Cheng Chen, Daoxin Zhang, Yi Wu, Yao Hu, Chongyang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.08283)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a pivotal technique for enhancing the reasoning capabilities of Large Language Models (LLMs). However, the de facto practice of mainstream RL algorithms is to treat all tokens of one response equally and assign the same optimization objective to each token, failing to provide granular guidance for the reasoning process. While in Chain-of-Thought (CoT) reasoning, different tokens usually play distinct roles. Therefore, the current RL algorithms lack an effective mechanism to dynamically balance the exploration-exploitation trade-off during learning. To this end, we propose Hierarchical Token-level Objective Control Policy Optimization (HTPO), a novel RL algorithm that takes the divide-and-conquer idea to hierarchically partition the response tokens into specific functional groups from three aspects (i.e., prompt difficulty, answer correctness, and token entropy). Within each group, according to the contributions to exploration or exploitation, we design specialized optimization objectives to facilitate the effective execution of each token's expected functionality. In this way, HTPO can achieve a more balanced exploration-exploitation trade-off. Extensive experiments on challenging reasoning benchmarks validate the superiority of our HTPO algorithm, which significantly outperforms the strong DAPO baseline (e.g., +8.6% and +6.7% on AIME'24 and AIME'25, respectively). When scaling test-time compute, the HTPO-trained model maintains a consistent performance advantage over the DAPO baseline, and the gap widens as the sampling budget increases, validating that our adaptive token-level control method fosters effective exploration without sacrificing exploitation performance. Code will be at this https URL. 

---
# AIPO: : Learning to Reason from Active Interaction 

**Authors**: Junnan Liu, Linhao Luo, Thuy-Trang Vu, Gholamreza Haffari  

**Link**: [PDF](https://arxiv.org/pdf/2605.08401)  

**Abstract**: Recent advances in large language models (LLMs) have demonstrated remarkable reasoning capabilities, largely stimulated by Reinforcement Learning with Verifiable Rewards (RLVR). However, existing RL algorithms face a fundamental limitation: their exploration remains largely constrained by the inherent capability boundary of the policy model. Although recent methods introduce external expert demonstrations to extend this boundary, they typically rely on complete trajectory-level guidance, which is sample-inefficient, information-sparse, and may confine exploration to a static guidance space. Inspired by the potential of multi-agent systems, we propose $\textbf{AIPO}$, an enhanced reinforcement learning framework that improves LLM reasoning through active multi-agent interaction during exploration. Specifically, AIPO enables the policy model to proactively consult three functional collaborative agents, $\textit{Verify Agent}$, $\textit{Knowledge Agent}$, and $\textit{Reasoning Agent}$, when encountering reasoning bottlenecks, thereby receiving fine-grained and targeted guidance to actively expand its capability boundary during training. We further introduce a tailored importance sampling coefficient together with a clipping strategy to mitigate the off-policy bias and gradient vanishing issues that arise when learning from agent-provided feedback. After training, the policy model performs reasoning independently without relying on collaborative agents. Extensive experiments on diverse reasoning benchmarks, including AIME, MATH500, GPQA-Diamond, and LiveCodeBench, show that AIPO consistently improves reasoning performance, generalizes robustly across different policy models and RLVR algorithms, and effectively expands the reasoning capability boundary of the policy model. 

---
# Rethinking Entropy Minimization in Test-Time Adaptation for Autoregressive Models 

**Authors**: Wei-Ping Huang, Chee-En Yu, Guan-Ting Lin, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.08186)  

**Abstract**: Test-Time Adaptation (TTA) via entropy minimization (EM) has proven effective for classification tasks, yet its application to generative autoregressive models remains theoretically fragmented. Existing approaches typically rely on distinct heuristics, such as teacher forcing with pseudo labels or policy-gradient-based reinforcement learning, without a unified mathematical foundation. In this work, we resolve this discrepancy by deriving a rigorous formulation of EM tailored to autoregressive models. We show that the exact objective naturally decomposes into a token-level policy gradient loss and a token-level entropy loss, and we reinterpret prior methods as partial realizations of this unified formulation. Using Whisper ASR as a testbed, we demonstrate that our approach consistently improves performance across more than 20 diverse domains, including acoustic noise, accents, and multilingual settings. 

---
# A Unified Pair-GRPO Family: From Implicit to Explicit Preference Constraints for Stable and General RL Alignment 

**Authors**: Hao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.06375)  

**Abstract**: Large language model (LLM) alignment via reinforcement learning from human preferences (RLHF) suffers from unstable policy updates, ambiguous gradient directions, poor interpretability, and high gradient variance in mainstream pairwise preference learning paradigms. To systematically address these limitations, we establish a unified theoretical framework for preference-based RL optimization centered on the Pair-GRPO family, comprising two tightly coupled variants: Soft-Pair-GRPO and Hard-Pair-GRPO. Soft-Pair-GRPO is a minimal modification of Group Relative Policy Optimization (GRPO) that replaces group-normalized scalar rewards with binary pairwise preference rewards, retaining GRPO's clipped surrogate and KL-regularized structure. We prove a critical gradient equivalence theorem: under first-order Taylor expansion around the current policy, Soft-Pair-GRPO's gradient is a positive scalar multiple of standard GRPO's gradient, explaining its empirical stability despite discarding continuous reward magnitudes. Building on this foundation, we propose Hard-Pair-GRPO, an advanced variant introducing explicit local probability constraints and constrained KL-fitting optimization to further suppress gradient noise and global policy drift. We provide comprehensive theoretical guarantees for both variants--including monotonic policy improvement, deterministic gradient direction, gradient-variance reduction, and dynamic step-size convergence. Extensive experiments on standard LLM alignment benchmarks (HH-RLHF,UltraFeedback) and the MuJoCo continuous control task HalfCheetah-v4 demonstrate that our Pair-GRPO family consistently outperforms state-of-the-art baselines in alignment quality, human preference win rate, training stability, and generalization to general reinforcement learning. Ablation studies validate the critical contributions of each core component. 

---
# DGPO: Beyond Pairwise Preferences with Directional Consistent Groupwise Optimization 

**Authors**: Mengyi Deng, Zhiwei Li, Xin Li, Tingyu Zhu, Yulan Yuan, Zhijiang Guo, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.10863)  

**Abstract**: Although Large Language Models (LLMs) have made remarkable progress, current preference optimization methods still struggle to align directional consistency while preserving reasoning diversity. To address this limitation, we propose Directional-Groupwise Preference Optimization (DGPO), a lightweight framework that aggregates supervision signals at the group level and explicitly models direction-aware alignment through multi-candidate comparisons. DGPO organizes forward and reverse question-answer instances into structured sets and optimizes a margin-based likelihood objective that separates coherent reasoning paths from inconsistent alternatives. This group-wise formulation captures richer relative information than pairwise objectives and reinforces consistency across diverse reasoning pathways. Empirical results show that our constructed reverse data yields a 3.2% average improvement across five benchmarks, while DGPO further delivers consistent gains across multiple datasets and model families, achieving average accuracy improvements of up to 3.6%. 

---
# RubricEM: Meta-RL with Rubric-guided Policy Decomposition beyond Verifiable Rewards 

**Authors**: Gaotang Li, Bhavana Dalvi Mishra, Zifeng Wang, Jun Yan, Yanfei Chen, Chun-Liang Li, Long T. Le, Rujun Han, George Lee, Hanghang Tong, Chen-Yu Lee, Tomas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2605.10899)  

**Abstract**: Training deep research agents, namely systems that plan, search, evaluate evidence, and synthesize long-form reports, pushes reinforcement learning beyond the regime of verifiable rewards. Their outputs lack ground-truth answers, their trajectories span many tool-augmented decisions, and standard post-training offers little mechanism for turning past attempts into reusable experience. In this work, we argue that rubrics should serve not merely as final-answer evaluators, but as the shared interface that structures policy execution, judge feedback, and agent memory. Based on this view, we introduce RubricEM, a rubric-guided reinforcement learning framework that combines stagewise policy decomposition with reflection-based meta-policy evolution. RubricEM first makes research trajectories stage-aware by conditioning planning, evidence gathering, review, and synthesis on self-generated rubrics. It then assigns credit with Stage-Structured GRPO, which uses stagewise rubric judgments to provide denser semantic feedback for long-horizon optimization. In parallel, RubricEM trains a shared-backbone reflection meta-policy that distills judged trajectories into reusable rubric-grounded guidance for future attempts. The resulting RubricEM-8B achieves strong performance across four long-form research benchmarks, outperforming comparable open models and approaching proprietary deep-research systems. Beyond final performance, we perform thorough analyses to understand the key ingredients of RubricEM. 

---
# Towards On-Policy Data Evolution for Visual-Native Multimodal Deep Search Agents 

**Authors**: Shijue Huang, Hangyu Guo, Chenxin Li, Junting Lu, Xinyu Geng, Zhaochen Su, Zhenyu Li, Shuang Chen, Hongru Wang, Yi R. Fung  

**Link**: [PDF](https://arxiv.org/pdf/2605.10832)  

**Abstract**: Multimodal deep search requires an agent to solve open-world problems by chaining search, tool use, and visual reasoning over evolving textual and visual context. Two bottlenecks limit current systems. First, existing tool-use harnesses treat images returned by search, browsing, or transformation as transient outputs, so intermediate visual evidence cannot be re-consumed by later tools. Second, training data is usually built by fixed curation recipes that cannot track the target agent's evolving capability. To address these challenges, we first introduce a visual-native agent harness centered on an image bank reference protocol, which registers every tool-returned image as an addressable reference and makes intermediate visual evidence reusable by later tools. On top of this harness, On-policy Data Evolution (ODE) runs a closed-loop data generator that refines itself across rounds from rollouts of the policy being trained. This per-round refinement makes each round's data target what the current policy still needs to learn. The same framework supports both diverse supervised fine-tuning data and policy-aware reinforcement learning data curation, covering the full training lifecycle of the target agent. Across 8 multimodal deep search benchmarks, ODE improves the Qwen3-VL-8B agent from 24.9% to 39.0% on average, surpassing Gemini-2.5 Pro in standard agent-workflow setting (37.9%). At 30B, ODE raises the average score from 30.6% to 41.5%. Further analyses validate the effectiveness of image-bank reuse, especially on complex tasks requiring iterative visual refinement, while rollout-feedback evolution yields more grounded SFT traces and better policy-matched RL tasks than static synthesis. 

---
# Aligning LLM Uncertainty with Human Disagreement in Subjectivity Analysis 

**Authors**: Junyu Lu, Deyi Ji, Xuanyi Liu, Lanyun Zhu, Bo Xu, Liang Yang, Hongfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.10415)  

**Abstract**: Large language models for subjectivity analysis are typically trained with aggregated labels, which compress variations in human judgment into a single supervision signal. This paradigm overlooks the intrinsic uncertainty of low-agreement samples and often induces overconfident predictions, undermining reliability and generalization in complex subjective settings. In this work, we advocate uncertainty-aware subjectivity analysis, where models are expected to make predictions while expressing uncertainty that reflects human disagreement. To operationalize this perspective, we propose a two-phase Disagreement Perception and Uncertainty Alignment (DPUA) framework. Specifically, DPUA jointly models label prediction, rationale generation, and uncertainty expression under an uncertainty-aware setting. In the disagreement perception phase, adaptive decoupled learning enhances the model's sensitivity to disagreement-related cues while preserving task performance. In the uncertainty alignment phase, GRPO-based reward optimization further improves uncertainty-aware reasoning and aligns the model's confidence expression with the human disagreement distribution. Experiments on three subjectivity analysis tasks show that DPUA preserves task performance while better aligning model uncertainty with human disagreement, mitigating overconfidence on boundary samples, and improving out-of-distribution generalization. 

---
# Relative Score Policy Optimization for Diffusion Language Models 

**Authors**: Zichao Yu, Shengze Xu, Bingqing Jiang, Wenyi Zhang, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2605.10218)  

**Abstract**: Diffusion large language models (dLLMs) offer a promising route to parallel and efficient text generation, but improving their reasoning ability requires effective post-training. Reinforcement learning with verifiable rewards (RLVR) is a natural choice for this purpose, yet its application to dLLMs is hindered by the absence of tractable sequence-level log-ratios, which are central to standard policy optimization. The lack of tractable sequence-level log-ratios forces existing methods to rely on high-variance ELBO-based approximations, where high verifier rewards can amplify inaccurate score estimates and destabilize RL training. To overcome this issue, we propose \textbf{R}elative \textbf{S}core \textbf{P}olicy \textbf{O}ptimization (RSPO), a simple RLVR method that uses verifiable rewards to calibrate noisy likelihood estimates in dLLMs. The core of our algorithm relies on a key observation: a reward advantage can be interpreted not only as an update direction, but also as a target for the relative log-ratio between the current and reference policies. Accordingly, RSPO calibrates this noisy relative log-ratio estimate by comparing its reward advantage with the reward-implied target relative log-ratio, updating the policy according to the gap between the current estimate and the target rather than the raw advantage alone. Experiments on mathematical reasoning and planning benchmarks show that RSPO yields especially strong gains on planning tasks and competitive mathematical-reasoning performance. 

---
# Position: Avoid Overstretching LLMs for every Enterprise Task 

**Authors**: Kuldeep Singh, Anson Bastos, Isaiah Onando Mulang'  

**Link**: [PDF](https://arxiv.org/pdf/2605.09365)  

**Abstract**: Enterprise workloads are dominated by deterministic, structured, and knowledge-dependent tasks operating under strict cost, latency, and reliability constraints. While these are often addressed through large language model (LLM) deployment or distillation into smaller models, we argue this is inefficient, unreliable, and misaligned with enterprise task structures. Instead, AI systems should treat language models as interfaces rather than monolithic engines, externalizing knowledge and computation into dedicated components for greater reliability, scalability, and transparency. Our theoretical evidences show that finite-capacity models cannot fully capture the breadth of knowledge required for enterprise tasks, creating inherent limits to efficiency and interpretability. Building on this, we take the position that language models should primarily be used for structured extraction in deterministic enterprise workflows, while computation and storage are delegated to knowledge bases and symbolic procedures. We formally demonstrate that such modular architectures are more reliable and maintainable than monolithic frameworks, offering a sustainable foundation for enterprise tasks. 

---
# Personalized Alignment Revisited: The Necessity and Sufficiency of User Diversity 

**Authors**: Enoch Hyunwook Kang  

**Link**: [PDF](https://arxiv.org/pdf/2605.09119)  

**Abstract**: Personalized alignment aims to adapt large language models to heterogeneous user preferences, yet the precise theoretical conditions for its statistical efficiency have not been formally established. This paper characterizes the conditions under which personalized alignment achieves O(1) online regret and log(1/epsilon) offline sample complexity. We show that these optimal rates depend on a specific user-diversity condition: the population of user-specific heads must span the latent reward directions that can alter the optimal response. We prove that this condition is both necessary and sufficient. When it holds, simple greedy algorithms achieve benchmark efficiency; when it fails, every learner in a natural admissible class incurs at least logarithmic regret. Our results identify user diversity as the fundamental driver of personalized identifiability. 

---
# Reinforcing Multimodal Reasoning Against Visual Degradation 

**Authors**: Rui Liu, Dian Yu, Haolin Liu, Yucheng Shi, Tong Zheng, Runpeng Dai, Haitao Mi, Pratap Tokekar, Leoweiliang  

**Link**: [PDF](https://arxiv.org/pdf/2605.09262)  

**Abstract**: Reinforcement Learning has significantly advanced the reasoning capabilities of Multimodal Large Language Models (MLLMs), yet the resulting policies remain brittle against real-world visual degradations such as blur, compression artifacts, and low-resolution scans. Prior robustness techniques from vision and deep RL rely on static data augmentation or value-based regularization, neither of which transfers cleanly to critic-free RL fine-tuning of autoregressive MLLMs. Reinforcing reasoning against such corruptions is non-trivial: naively injecting degraded views during rollout induces reward poisoning, where perceptual occlusions trigger hallucinated trajectories and destabilize optimization. We propose ROMA, an RL fine-tuning framework that modifies the optimization dynamics to reinforce reasoning against visual degradation while preserving clean-input performance. A dual-forward-pass strategy uses teacher forcing to evaluate corrupted views against clean-image trajectories, avoiding new rollouts on degraded inputs. For distributional consistency, we apply a token-level surrogate KL penalty against the worst-case augmentation; to prevent policy collapse under regularization, an auxiliary policy gradient loss anchored to clean-image advantages preserves a reliable reward signal; and to avoid systematically incorrect invariance, correctness-conditioned regularization restricts enforcement to successful trajectories. On Qwen3-VL 4B/8B across seven multimodal reasoning benchmarks, our method improves robustness by +2.4% on seen and +2.3% on unseen corruptions over GRPO while matching clean accuracy. 

---
# The Extrapolation Cliff in On-Policy Distillation of Near-Deterministic Structured Outputs 

**Authors**: Xin Li, Hao Jiang, Annan Wang, Yichi Zhang, Chau Yuen  

**Link**: [PDF](https://arxiv.org/pdf/2605.08737)  

**Abstract**: On-policy distillation (OPD) is widely used for LLM post-training. When pushed with a reward-extrapolation coefficient lambda > 1, the student can lift past the teacher in domain, but past a threshold lambda* the same step violates the output contract on structured-output tasks. In a single-position Bernoulli reduction, we derive a closed-form base-relative clip-safety threshold lambda*(p,b,c) determined by three measurable quantities: the teacher modal probability, the warm-start mass, and the importance-sampling clip strength. Above lambda*, the extrapolated fixed point exits the clip-safe region, changing training from format-preserving to format-collapsing. We extend the rule to calibrated K-ary listwise JSON tasks where a single binding equivalence class dominates the output contract and SFT retains parse headroom. On Amazon Fashion, three pre-registered tests--a fine-grid cliff interval, a budget-extension test, and a small-clip cross-prediction--fall within their locked prediction windows, with the small-clip value matching the closed-form prediction below grid resolution. Operating just below lambda*, ListOPD brings a 1.7B Qwen3 student to in-domain parity with an 8B-SFT baseline at one-fifth the parameters. The gain is driven primarily by format adherence: NDCG@1 on parsed outputs remains flat across lambda, while parse validity sharply changes at the predicted boundary. The cliff diagnostic is rubric-independent, whereas the parity claim uses a Gemini-graded rubric and inherits that evaluator's exposure. 

---
# UserGPT Technical Report 

**Authors**: Yunyi Xuan, Hao Yi, Fengling Mao, Daye Cai, Leikun Liang, Xingsheng He, Jiangnan Xie, Guoshuai Wang, Yushan Han, Wenwen Guo, Xiaoxiao Xu, Lin Qu  

**Link**: [PDF](https://arxiv.org/pdf/2605.08766)  

**Abstract**: Personalized user understanding from large-scale digital traces remains a fundamental challenge. Traditional user profiling methods rely on discriminative models and manual feature engineering to predict discrete attributes, often producing fragmented and logically inconsistent profiles that generalize poorly to long-tail behaviors. In this work, we study a generative paradigm in which large language models (LLMs) summarize long and noisy behavioral histories into coherent narratives that capture nuanced user evolution. Our experiments show that even strong LLMs remain limited in complex and implicit personalization reasoning.
We propose UserGPT, a framework for improving LLM-based persona understanding through both attribute generation and summary generation. To address the scarcity of real-world behavioral data, we develop a User Behavior Simulation Engine that produces realistic and complex user trajectories. We further introduce a Data-Centric Semantization module that transforms heterogeneous behavioral logs into structured and semantically coherent inputs, reducing noise and sparsity. On top of this pipeline, we design a curriculum-driven post-training strategy that combines multi-stage Supervised Fine-Tuning (SFT) with Dual-Filter Group Relative Policy Optimization (DF-GRPO) to strengthen reasoning over long behavioral histories.
We also construct HPR-Bench, a benchmark for holistic persona reasoning derived from simulated data. On HPR-Bench, UserGPT achieves an Avg@10 score of 0.7325 on tag prediction and an $Acc_{Ex}$ score of 0.7528 on summary generation, while compressing behavioral records by up to 97.9% with critical information preserved. These results demonstrate the effectiveness of UserGPT for holistic persona reasoning and personalized user-agent interaction. 

---
# LASAR: Latent Adaptive Semantic Aligned Reasoning for Generative Recommendation 

**Authors**: Yiwen Chen, Fuwei Zhang, Zehao Chen, Deqing Wang, Hehan Li, Peizhi Xu, Hanmeng Liu, Shuanglong Li, Xin Pei, Fuzhen Zhuang, Zhao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.10207)  

**Abstract**: Large Language Models (LLMs) have demonstrated powerful reasoning capabilities through Chain-of-Thought (CoT) in various tasks, yet the inefficiency of token-by-token generation hinders real-world deployment in latency-sensitive recommender systems. Latent reasoning has emerged as an effective paradigm in LLMs, performing multi-step inference in a continuous hidden-state space to achieve stronger reasoning at lower cost. However, this paradigm remains underexplored in mainstream generative recommendation. Adapting it reveals three unique challenges: (1) the gap between prior-less Semantic ID (SID) symbols and continuous latent reasoning - SIDs lack pre-trained semantics, hindering joint optimization; (2) representation drift due to a lack of reasoning chain supervision; and (3) the suboptimality of applying a globally fixed reasoning depth. To address these, we propose LASAR (Latent Adaptive Semantic Aligned Reasoning), an SFT-then-RL framework. First, we bridge this gap via two-stage training: Stage 1 grounds SID semantics before Stage 2 introduces latent reasoning, ensuring efficient convergence. Second, we mitigate representation drift through explicit CoT semantic alignment. Step-wise bidirectional KL divergence constrains the latent reasoning trajectory using hidden-state anchors extracted from CoT text, while a Policy Head predicts per-sample reasoning depth. Third, during the GRPO-based RL phase, terminal-only KL alignment accommodates variable-length reasoning, and REINFORCE optimizes the Policy Head to dynamically allocate steps. This nearly halves the average latent step count while simultaneously improving recommendation quality. Experiments on three real-world datasets demonstrate that LASAR outperforms all baselines. It adds marginal inference latency and is roughly 20 times faster than generating explicit CoT text. 

---
