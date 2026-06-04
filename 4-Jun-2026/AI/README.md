# Knowledge Index of Noah's Ark 

**Authors**: Sheng Jin, Minghao Liu, Yunze Xiao, Zeqi Zhou, Heli Qi, Yifan Yao, Meishu Song, Kaijing Ma, Xuan Zhang, Sicong Jiang, Yizhe Li, Ningshan Ma, Jie Wei, Ziniu Li, Minglai Yang, Bangya Liu, Yiming Liang, Xiao Fang, Qingcheng Zeng, Jiarui Liu, Rui Yang, Shen Yan, Wenhao Huang, Jiaheng Liu, Zihan Wang, Weihao Xuan, Ge Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.05104)  

**Abstract**: Knowledge benchmarks for LLMs face three issues: scaling-driven designs that do not operationalize disciplinary representativeness; flat-payment annotation that permits lazy consensus; and unaudited ranking instability under bounded test budgets. We introduce KINA, an 899-item benchmark across 261 fine-grained disciplines, with two formal results. First, we cast representativeness as a coverage-style objective over expert-elicited anchors and operationalize disciplinary representativeness through a proxy, yielding a (1-1/e) greedy approximation (Proposition 1); the guarantee applies to the proxy, not to population representativeness. Second, we prove a bonus-on-bar tournament weakly FOSD-dominates flat payment in released-review quality, with incentive-compatibility threshold B > Delta C / Delta p_min (Theorem 1). Evaluating 42 models from 13 labs, the top model, Gemini-3.1-Pro-Preview, reaches 53.17%, followed by Claude-Opus-4.6 at 49.92% and GPT-5.4 at 48.55%, leaving substantial headroom below saturation. The full leaderboard shows a tiered structure rather than a smooth total order: a small frontier tier lies above 48%, a dense strong-model tier spans roughly 38-45%, and low-performing models remain only modestly above the 10% chance baseline. Tool augmentation adds up to 5.17 points across the five tool-use evaluations, with gains varying substantially across models. We report bootstrap ranking-stability statistics to make bounded-budget variance explicit and to discourage over-interpretation of adjacent ranks. 

---
# AutoLab: Can Frontier Models Solve Long-Horizon Auto Research and Engineering Tasks? 

**Authors**: Zhangchen Xu, Junda Chen, Yue Huang, Dongfu Jiang, Jiefeng Chen, Hang Hua, Zijian Wu, Zheyuan Liu, Zexue He, Lichi Li, Shizhe Diao, Jiaxin Pei, Jinsung Yoon, Hao Zhang, Mengdi Wang, Radha Poovendran, Misha Sra, Alex Pentland, Zichen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.05080)  

**Abstract**: Scientific and engineering progress is fundamentally a long-horizon iterative process: proposing changes, running experiments, measuring outcomes, and continuously refining artifacts. Yet existing benchmarks for frontier models primarily evaluate either single-turn responses or short-horizon agent trajectories, failing to capture the challenges of sustained iterative improvement over extended time horizons. To address this gap, we introduce AutoLab, a new benchmark for ultra long-horizon closed-loop optimization. AutoLab consists of 36 realistic, expert-curated tasks spanning four diverse domains: system optimization, puzzle & challenge, model development, and CUDA kernel optimization. Each task begins with a correct but deliberately suboptimal baseline and challenges agents to improve it within a strict wall-clock budget. Evaluating 17 state-of-the-art models reveals the dominant predictor of success is not the quality of an agent's initial attempt, but its persistence in repeatedly benchmarking, editing, and incorporating empirical feedback. While claude-opus-4.6 exhibits strong long-horizon optimization capabilities, most frontier models, including several proprietary ones, either terminate prematurely or exhaust their budgets with minimal progress. These results underscore the importance of time awareness and persistent iteration in autonomous agents. We open-source the full benchmark, evaluation harness, and task artifacts, to accelerate research toward truly capable long-horizon agents. 

---
# Strabo: Declarative Specification and Implementation of Agentic Interaction Protocols 

**Authors**: Samuel H. Christie V, Amit K. Chopra, Munindar P. Singh  

**Link**: [PDF](https://arxiv.org/pdf/2606.05043)  

**Abstract**: The last few years have witnessed major advances in the modeling and implementation of multiagent systems based on declarative interaction protocols. Our contribution, Strabo, establishes the relevance of these advances to ongoing industry efforts in Agentic AI. Specifically, we consider UCP, the Universal Commerce Protocol, a recent Google-led effort to standardize e-commerce interactions for AI agents. Our exercise is in two parts. One, we model the part of UCP dealing with checkouts as a declarative Langshaw protocol and implement agents using Peach, a programming model for Langshaw. This part of the exercise brings out the advantages of formal, declarative specifications. Two, we show that Peach agents can interoperate with UCP agents implemented by Google, thereby establishing the fidelity of our approach with respect to UCP. Such interoperation enables the incremental introduction of declarative protocols and agents into a conventional setting, indicating a pathway by which EMAS ideas could influence practice without demanding a wholesale update. 

---
# What Type of Inference is Active Inference? 

**Authors**: Wouter W. L. Nuijten, Mykola Lukashchuk, Thijs van de Laar, Bert de Vries  

**Link**: [PDF](https://arxiv.org/pdf/2606.04935)  

**Abstract**: Active inference casts decision-making as inference, with the Expected Free Energy (EFE) unifying goal-directed and information-seeking behavior. Recent work showed that EFE minimization can be written as Variational Free Energy (VFE) minimization on a generative model augmented with epistemic priors. We prove that the VFE of the augmented model can be rewritten as the VFE of the predictive model plus explicit entropy-correction terms, making the EFE contribution transparent. We then show that proper EFE-based planning requires combining these epistemic corrections with a planning correction that turns marginal inference into policy optimization, yielding a full variational characterization of EFE-based planning. This clarifies which corrections are needed for cross-entropy planning and for full EFE-based planning. The same entropy-corrected formulation leads to a detailed message-passing scheme for EFE-based planning together with simpler ablations. Experiments on three grid-world environments show that the planning correction already helps when observations are decisive, whereas the additional observation-side epistemic corrections matter most when observations are merely suggestive. 

---
# AICompanionBench: Benchmarking LLMs-as-Judges for AI Companion Safety 

**Authors**: Yanjing Ren, Reza Ebrahimi, TengTeng Ma  

**Link**: [PDF](https://arxiv.org/pdf/2606.04867)  

**Abstract**: As AI companion platforms such as Replika and this http URL rapidly grow, concerns about unsafe human-AI interactions have intensified. This study introduces AICompanionBench, to our knowledge the first publicly available benchmark dataset of human-AI companion conversations annotated with fine-grained safety risk categories. The dataset contains 2,123 real-world Replika conversations collected from Reddit and annotated through human-AI collaboration across nine categories: sexual behavior, antisocial behavior, physical aggression, verbal aggression, substance abuse, self-harm and suicide, control, manipulation, and no-harm. Using this benchmark, we evaluate 20 state-of-the-art open-source and closed-source LLMs under an LLM-as-judge framework for detecting unsafe interactions. Results show substantial variation in model performance, with stronger models achieving high overall accuracy but still struggling with nuanced categories such as manipulation, as well as benign conversations that are incorrectly identified as harmful. Our findings suggest that while current LLMs can effectively detect explicit harmful content, they remain limited in identifying implicit unsafe interactions. Overall, our work contributes a new benchmark dataset for AI companionship safety research and offers insights into monitoring AI companion systems using LLMs. The dataset is publicly available at: this https URL 

---
# R-APS: Compositional Reasoning and In-Context Meta-Learning for Constrained Design via Reflective Adversarial Pareto Search 

**Authors**: João Pedro Gandarela, Thiago Rios, Stefan Menzel, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2606.04823)  

**Abstract**: Large language models (LLMs) are fluent on open-ended tasks, yet in agentic settings, where a system must plan, use tools, and act over extended horizons, fluency does not ensure reliable delivery. We trace this gap to three coupled structural failures: errors propagate without localization, worst-case perturbations go unevaluated, and accumulated knowledge is never invalidated. We argue these share a root cause: abductive, counterfactual, meta-inductive, corrective, and inductive reasoning pull a shared context in incompatible directions. We introduce Reflective Adversarial Pareto Search (R-APS), to our knowledge the first method addressing all three failures jointly via reasoning-mode decomposition, allocating each reasoning mode its own context and orchestrating interaction across three timescales: staged compositional reasoning with a typed validation critic (failure localization), sensitivity-guided counterfactual stress-testing as a first-class Pareto objective (robustness), and meta-inductive rule extraction with explicit invalidation (persistent memory). R-APS requires no fine-tuning and operates on a frozen LLM purely via structured protocol design. We evaluate on planar mechanism synthesis (robotics, prosthetics, mechanical design), with every candidate checked by a kinematic solver. On 32 target trajectories, R-APS delivers robustness certificates 3.5x tighter than uniform-perturbation baselines, 46% faster iterations-to-first-admission, and 2.1x Chamfer-distance reduction over Enum+GA while jointly controlling bar-count and worst-case robustness. Small 4B reasoning-specialized models prove competitive with general-purpose 70B backbones inside the protocol, suggesting structured protocols can partially offset model scale. 

---
# Beyond Objective Equivalence: Constraint Injection for LLM-Based Optimization Modeling on Vehicle Routing Problems 

**Authors**: Xizi Luo, Changhong He, Dongdong Geng, Chenggong Shi, Yu Mei  

**Link**: [PDF](https://arxiv.org/pdf/2606.04816)  

**Abstract**: Large language models (LLMs) increasingly translate natural-language optimization problems into executable solver code. Yet for constraint-dense operations research (OR) problems, existing data-filtering and training pipelines largely rely on objective-equivalence signals such as differential testing and answer agreement, which a program can pass while adding spurious constraints or silently omitting required ones, whenever those constraints are non-binding on the tested instance. We propose constraint injection, which uses feasible probes to expose spurious over-constraint and one-constraint-violating probes to reveal silent constraint omission. Combined with differential testing, it forms a dual verifier. We instantiate and evaluate it on vehicle routing problems (VRPs), a representative constraint-dense combinatorial optimization testbed with coupled operational constraints. We develop VRPCoder, an 8B end-to-end model that translates natural-language VRP scenarios into Gurobi scripts, together with an expert-verified VRP benchmark suite covering 21 variants. The verifier is reused as a rejection-sampling filter during data synthesis and as a per-rollout reward in group relative policy optimization (GRPO). Across four VRP benchmarks, VRPCoder-GRPO reaches 93\% average Pass@1, outperforms Gemini-3.1-Pro Preview on three benchmarks, exceeds Claude-Sonnet-4.5 by 28 average points, and surpasses prior OR-LLMs by 78 average points. 

---
# BiasGRPO: Stabilizing Bias Mitigation in High-Variance Reward Landscapes via Group-Relative Policy Optimization 

**Authors**: Saket Reddy, Ke Yang, ChengXiang Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2606.04807)  

**Abstract**: Mitigating social bias in Large Language Models (LLMs) presents a distinct alignment challenge: unlike verifiable tasks, bias lacks a single ground truth, creating a high-variance, subjective reward landscape. Previous preference-based fine-tuning methods have major trade-offs: Direct Preference Optimization (DPO) is limited by the lack of exploration inherent in offline training, while Proximal Policy Optimization (PPO) can lead to training instability due to potentially unreliable critic estimates. In this paper, we propose BiasGRPO, a framework using Group Relative Policy Optimization (GRPO) to stabilize alignment by normalizing rewards across a group of sampled completions. By substituting the value function with a group-relative baseline, our approach reduces instability while maintaining the exploration benefits of online training. We find that BiasGRPO outperforms DPO and PPO across multiple benchmarks, indicating its effectiveness. To adapt GRPO, we synthetically extend a dataset spanning multiple domains and contexts. We also create and release a custom bias reward model that effectively guides generation while being highly compute-efficient and avoiding knowledge degradation, providing a valuable resource that can be seamlessly integrated into multi-objective RLHF pipelines. 

---
# AIP: A Graph Representation for Learning and Governing Agent Skills 

**Authors**: Zachary Blumenfeld, Jim Webber  

**Link**: [PDF](https://arxiv.org/pdf/2606.04781)  

**Abstract**: Agent Skills today consist largely of free-form prose requiring the agent to read, interpret, and re-derive how to act in every session. This imposes two compounding costs: reduced reliability on implementation-heavy tasks, and difficulty in skill creation and improvement, since editing prose is a fragile process that both humans and agents struggle with, particularly for domain-specific procedural knowledge underrepresented in model training. The Agent Instruction Protocol (AIP) addresses both by modeling a skill as a directed execution graph: discrete steps as nodes backed by deterministic scripts or natural-language descriptions, connected by explicit typed input/output edges, and governed by a schema-validated YAML specification. A compiler meta-skill translates existing human-written skills into this form. The benefits are twofold. First, compiling human-written skills to AIP raised Claude Sonnet's mean task reward from 0.60 to 0.71 and pass rate from 53% to 67% across 27 real agent tasks from SkillsBench - a statistically significant gain (Wilcoxon signed-rank p = 0.011), winning 12 tasks to 2 with 13 ties - often in less wall-clock time. The graph delivers vetted, runnable units to the agent rather than asking it to re-derive code, commands, and tool calls from natural language. Second, on creation and improvement, because each skill is schema-validated, functionally testable, and addressable node-by-node, failures can be diagnosed and repaired precisely. Two authored-skill failures were traced to the script level. After adjusting the AIP spec and recompiling, both recovered with zero regressions (one task going from 0/5 to 5/5), turning skill improvement into a measurable tuning loop rather than a prose rewrite. That same graph structure supports corpus-level governance and skill introspection, and provides a natural action space for reinforcement learning over skills. 

---
# Tree-Based Formalization of Multi-Agent Complementarity in Human-AI Interactions 

**Authors**: Andrea Ferrario  

**Link**: [PDF](https://arxiv.org/pdf/2606.04779)  

**Abstract**: Complementarity is the case in which a human--AI interaction (HAI) outperforms the best prediction benchmark available among its members. Although this idea is central in HAI research, formal work on complementarity remains limited. Existing frameworks do not model how agents' predictions compose into workflow-sensitive multi-agent protocols. We close this gap by introducing a tree-based formalization of complementarity in multi-agent HAI. An HAI protocol is represented by an ordered agent-role configuration together with a rooted planar binary tree whose leaves are decorated by prediction vectors. A local binary composition rule is evaluated recursively along the tree, yielding a tree-relative complementarity functional relative to a pointwise-min oracle benchmark. We prove four results. First, selector-based HAIs, including self- or AI-reliance, cannot achieve complementarity regardless of task, loss, or prediction quality. Second, in regression under squared loss, complementarity is equivalent to Euclidean distance minimization from the ground-truth vector; for $N=2$, the optimal linear-pooling weight has a closed form and a residual-correction interpretation. Third, under linear local composition, every protocol tree defines a barycentric coordinate chart on the simplex of leaf weights; Tamari-cover reparameterizations of protocol trees preserve complementarity, and for $N=4$, they satisfy the pentagon identity. Fourth, in binary classification, no internal local composition can achieve complementarity under endpoint-monotone losses, including standard Bregman and many finite Bernoulli $f$-divergence losses; an analogous obstruction holds for multiclass aggregation under cross-entropy. In summary, our framework shows that complementarity is attainable in multi-agent regression, but obstructed in classification under natural conditions on local aggregation and loss functions. 

---
# Inference-Time Vulnerability Beyond Shallow Safety: Alignment Along Generation Trajectories 

**Authors**: Kyungmin Park, Taesup Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.04778)  

**Abstract**: Safety-aligned Large Language Models (LLMs) remain vulnerable to interventions during inference that redirect generation toward harmful outputs. Recent work attributes this to shallow safety, where alignment concentrates in the first few output tokens. We show that shallow safety is a special case of a broader inference-time vulnerability, in which short token injections at any generation step can substantially alter subsequent safety behavior. We also find that a model's alignment with refusal directions in its hidden states does not predict its robustness to such injection, revealing that internal state alone does not determine generation behavior under perturbation. To address this, we align models directly on generation trajectories constructed by simulating mid-sequence perturbation, and show that this improves robustness to mid-sequence injection and generalizes to attacks that exploit early-token generation. Our work argues that robust safety alignment requires training on the generation process itself, not only its outputs. 

---
# FALSIFYBENCH: Evaluating Inductive Reasoning in LLMs with Rule Discovery Games 

**Authors**: Leonardo Bertolazzi, Katya Tentori, Raffaella Bernardi  

**Link**: [PDF](https://arxiv.org/pdf/2606.04751)  

**Abstract**: Large language models (LLMs) are increasingly deployed as autonomous agents in scientific tasks. Yet whether these systems can effectively engage in forms of inductive reasoning relevant to scientific discovery remains an open question. In this work, we introduce FALSIFYBENCH, an evaluation framework for hypothesis-driven reasoning inspired by the classic Wason 2-4-6 task, in which agents must discover hidden semantic properties by iteratively proposing examples and receiving feedback. This task captures key elements of scientific reasoning: hypothesis generation, evidence gathering, and belief revision in response to both confirming and disconfirming evidence. Our evaluation of 12 LLMs across model families and scales shows that reasoning models are generally stronger scientific reasoners than instruction-tuned models, although no model comes close to optimal performance. The primary driver of success is the capacity for negative testing: models that actively seek to falsify their hypotheses consistently outperform those that primarily seek confirmation. Moreover, a fine-grained turn-level analysis, neglected in previous work, reveals that failure is tied to identifiable patterns in how models navigate the hypothesis space. 

---
# Fog of Love: Engineering Virtuous Agent Behavior with Affinity-based Reinforcement Learning in a Game Environment 

**Authors**: Ajay Vishwanath, Christian Omlin  

**Link**: [PDF](https://arxiv.org/pdf/2606.04750)  

**Abstract**: Instilling virtuous behavior in artificial intelligence has seen increasing interest. One of the techniques proposed is known as affinity-based reinforcement learning, which uses policy regularization on the objective function to incentivize virtuous actions without being fully dependent on the reward function design. Thus far, this technique has been demonstrated to be effective in grid worlds and toy-problem environments with minimal state and action spaces. To expand this research to more sophisticated environments, we introduce a two-player multi-agent environment based on the role-playing board game known as Fog of Love. In this environment, two agents compete to fulfill their individual virtues, while also cooperating to satisfy their relationship. Given the multi-agent nature, this is a complex problem where multi-agent deep deterministic policy gradient agents neither compete nor cooperate successfully. We present evidence that localized affinities enhance agent performance in achieving both competitive and cooperative objectives, resulting from superior overall scores in both domains. This not only results in virtuous choices but also clarifies an agent's teleology and makes its behavior human-level interpretable. 

---
# BiNSGPS: Geometry Problem Solving via Bidirectional Neuro-Symbolic Interaction 

**Authors**: Qi Wang, Peijie Wang, Fei Yin, Cheng-Lin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.04648)  

**Abstract**: Geometry problem solving poses distinct challenges in artificial intelligence. Existing approaches typically fall into two paradigms: symbolic methods, which exhibit limited adaptability, and neural methods, which are prone to hallucinations. Recent neuro-symbolic hybrids predominantly rely on a unidirectional pipeline where neural outputs are fed into solvers without feedback, making system brittle to early-stage errors. To break this unidirectional bottleneck, we propose BiNSGPS, a framework that establishes Bidirectional Neuro-Symbolic Interaction (BiNS) between a MLLM Adviser and a Symbolic Solver. MLLM Adviser actively incorporates feedback from the symbolic solver to dynamically rectify inconsistent formal representations or propose auxiliary hypotheses, resolving symbolic conflicts and facilitating complex deductions. 

---
# MIRAGE: Mobile Agents with Implicit Reasoning and Generative World Models 

**Authors**: Zhichao Yang, Yuanze Hu, Haojie Hao, Longkun Hao, Dongshuo Huang, Hongyu Lin, Gen Li, Lanqing Hong, Yihang Lou, Yan Bai  

**Link**: [PDF](https://arxiv.org/pdf/2606.04627)  

**Abstract**: Mobile agents are increasingly expected to operate everyday applications from screenshots and language goals, where reliable control requires reasoning over screen affordances, multi-step navigation, and future state changes. However, many agents externalize this computation as long textual chains of thought, which slows interaction, increases supervision cost, and complicates deployment. We introduce MIRAGE, a framework that learns continuous latent reasoning representations from visible textual reasoning traces. MIRAGE transfers explicit reasoning into compact hidden states, enabling the agent to reason internally without decoding long rationales. It also incorporates a generative world-model objective: latent reasoning vectors are aligned with future screenshots, encouraging the agent to anticipate upcoming interface states before acting. This turns hidden computation into both a compressed thought representation and a forward-looking model of environment dynamics. At inference time, MIRAGE reasons in continuous latent space, reducing token generation while improving execution efficiency. On AndroidWorld, MIRAGE matches explicit chain-of-thought supervised fine-tuning in the 4B ablation with a 3-5x lower decoded-token budget and improves a comparable instruction-tuned baseline by 10.2 points; on AndroidControl, it improves action grounding while generating over 75% fewer tokens. 

---
# A Normative Intermediate Representation for ASP-Based Compliance Reasoning 

**Authors**: Yangfan Wu, Huanyu Yang, Jianmin Ji  

**Link**: [PDF](https://arxiv.org/pdf/2606.04619)  

**Abstract**: We propose MONIR, a Modalized-Output Normative Intermediate Representation for ASP-based compliance reasoning. Its core fragment has a staged operational semantics, while MONIR-ASP provides an executable compilation and extensions for external functions, temporal rules, and stable-model reasoning. We instantiate the framework on Chinese ADAS regulations and standards with an LLM-assisted pipeline. Experiments evaluate extraction quality and the efficiency of modular and incremental ASP solving. 

---
# Parthenon Law: A Self-Evolving Legal-Agent Framework 

**Authors**: Hejia Geng, Leo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.04602)  

**Abstract**: As agents grow more capable, legal-domain LLM agents promise to turn document-heavy matters into reviewable work products -- yet reliable deployment faces three obstacles: no large-scale evidence on how today's strongest model-and-harness combinations behave on end-to-end legal matters; no agent architecture adapted to the legal vertical, only general-purpose harnesses; and, in a setting that keeps shifting with new facts, authorities, and deadlines, no mechanism for systems to learn from their own outcomes. We address each. A large-scale empirical study on Harvey LAB -- $12{,}510$ agent trajectories -- shows that even frontier agents remain far from completing matters in a single pass: per-criterion accuracy climbs with stronger models while strict matter completion stalls. We then introduce \textsc{Parthenon}, a self-evolving legal-agent framework that factors Model, Harness, Agent roles, legal Knowledge, deterministic Tools, and procedural Skills into auditable surfaces for source traceability, date and number grounding, deliverable compliance, and issue closure. Finally, an anti-leakage learning loop converts scored failures into task-agnostic edits to skills, tools, and knowledge, letting the system improve with experience -- as a firm refines its checklists and playbooks after each matter -- without touching model weights. Across our large-scale empirical analysis, \textsc{Parthenon} substantially improves the performance of state-of-the-art models and harnesses on legal-matter tasks. 

---
# Plan First, Judge Later, Run Better: A DMAIC-Inspired Agentic System for Industrial Anomaly Detection 

**Authors**: Yongzi Yu, Ao Li, Le Wang, Ziyue Li, Fugee Tsung, Yuxuan Liang, Man Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.04599)  

**Abstract**: Large language model (LLM) agents have shown promise in automating complex data-analysis workflows, but their reliable deployment remains challenging in high-stakes industrial scenarios. Industrial anomaly detection (IAD) is essential for manufacturing quality, safety, and efficiency, yet existing LLM-based IAD agents mainly focus on execution while under-exploiting strategy formulation. Consequently, they struggle to handle heterogeneous modalities in a unified and cost-effective manner. Inspired by the DMAIC quality-management framework, we propose DMAIC-IAD (DMAIC-inspired Agentic Industrial Anomaly Detection), a "Plan First, Judge Later" multi-agent system that aligns LLM agents with structured industrial problem-solving. DMAIC-IAD distills heterogeneous references into standardized operating procedures (SOPs) before strategy generation, and introduces a pre-trained execution-free judge model to rank candidate strategies without costly runtime trials. Extensive experiments across four modalities show that DMAIC-IAD improves average detection performance over applicable agentic baselines by 37.76%. 

---
# Learning Admissible Heuristics via Cost Partitioning 

**Authors**: Hugo Barral, Quentin Cappart, Marie-José Huguet, Sylvie Thiébaux  

**Link**: [PDF](https://arxiv.org/pdf/2606.04597)  

**Abstract**: Admissible heuristics are essential for optimal planning, yet learning them remains challenging due to the risk of overestimation. Cost partitioning combines multiple abstraction heuristics while preserving admissibility, but computing optimal partitions online is expensive. We propose a framework that learns to infer admissible cost partitions by leveraging the Lagrangian dual equivalence between cost partitioning and multiplier prediction. Planning states and patterns are encoded as labelled graphs, and an action-centric variant of the Weisfeiler-Leman algorithm extracts structural feature vectors. A deep architecture with axial self-attention and a softmax output layer maps these features to cost weights that satisfy the partition constraints by construction, ensuring admissibility. Experiments demonstrate reduced node expansions compared to suboptimal partitioning baselines while maintaining strict admissibility. To our knowledge, this is the first machine-learned heuristic guaranteed to be admissible. 

---
# SCI-PRM: A Tool Aware Process Reward Model for Scientific Reasoning Verification 

**Authors**: Xiangyu Zhao, Hengyuan Zhao, Yiheng Wang, Wanghan Xu, Yuhao Zhou, Qinglong Cao, Zhiwang Zhou, Lei Bai, Wenlong Zhang, Xiao-Ming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.04579)  

**Abstract**: While Process Reward Models (PRMs) have achieved remarkable success in mathematical reasoning, their application in complex scientific domains-such as biology, chemistry, and physics remains largely unexplored. Scientific problems demand not only logical rigor but also factual consistency and the precise usage of domain-specific tools, areas where current models often suffer from hallucinations and lack of verification. In this paper, we first construct SCIPRM70K, a large-scale dataset featuring Chain-of-Tool trajectories that explicitly interleave reasoning with the execution of scientific tools. Building upon this, we train an efficient reward model called Sci-PRM to provide fine-grained supervision on tool selection, execution accuracy, and result interpretation at each step in one inference. Experiments demonstrate that Sci-PRM significantly enhances foundation models in two key aspects: (1) it enables effective test-time scaling via Best-of-N selection; and (2) when integrated into Reinforcement Learning, it serves as a dense reward signal that mitigates the critical issue of advantage disappearance, allowing the model to break through existing performance ceilings. 

---
# Neetyabhas: A Framework for Uncertainty-Aware Public Policy Optimization in Rational Agent-Based Models 

**Authors**: Janani Venugopalan, Gaurav Deshkar, Rishabh Gaur, Harshal Hayatnagarkar, Jayanta Kshirsagar  

**Link**: [PDF](https://arxiv.org/pdf/2606.04562)  

**Abstract**: Purpose The WHO's COVID-19 non-pharmaceutical interventions (e.g., lockdowns, vaccinations) effectively curb transmission but impose heavy economic strains. Existing research often neglects individual behaviors and falsely assumes perfect infection tracking and flawless policy execution, failing to account for real-world uncertainties and errors. Methods We propose an integrative approach incorporating uncertainties in both epidemic measurement (infections/hospitalizations) and policy implementation. We built a simulation model of 1,000 individuals making real-time choices regarding mask-wearing, vaccination, and shopping. Concurrently, policymakers deploy interventions (lockdowns, mandates) based on health and economic observations. This framework is driven by hierarchical reinforcement learning agents, utilizing deep Q-networks alongside uncertainty-aware policy gradient variants (DDPG and TD3). Results The simulations effectively managed the epidemic's progression. Masking and vaccinations proved highly effective, significantly reducing both the outbreak's peak height and duration. By integrating individual behaviors, policy uncertainties, and multifaceted interventions, our dynamic control approach successfully mitigated the epidemic's impact. Conclusions Our model overcomes previous research limitations by embedding uncertainty and human behavior into public health policy frameworks. The simulation demonstrates that accounting for individual choices and imperfect data is crucial for designing effective interventions during complex pandemics, with masks and vaccines serving as pivotal tools. 

---
# Scaling Self-Evolving Agents via Parametric Memory 

**Authors**: Tao Ren, Weiyao Luo, Hui Yang, Rongzhi Zhu, Xiang Huang, Yuchuan Wu, Bingxue Chou, Jieping Ye, Jiafeng Liang, Yongbin Li, Yijie Peng  

**Link**: [PDF](https://arxiv.org/pdf/2606.04536)  

**Abstract**: Existing memory-augmented LLM agents store past experience exclusively in prompt space, as textual summaries or retrieved passages, while keeping model parameters frozen throughout a rollout. Such agents can \emph{look up} what they have seen but cannot \emph{learn from} it: their policy is unchanged by experience, and any information dropped from the context is permanently lost. We introduce \texttt{TMEM}, a self-evolving parametric memory framework in which the agent not only compresses history into explicit memory but also absorbs distilled supervision into fast LoRA weights $\Delta_t$ via lightweight online updates, genuinely altering its future behavior within a single episode. We formalize this as an agentic decision process with fast-weight rollout dynamics: actions are sampled from $\pi_{\theta_0+\Delta_t}$, while extraction actions produce supervision that updates $\Delta_t$ for subsequent decisions. This view makes the extraction policy directly optimizable by RL: training $\theta_0$ improves not only task actions but also the quality of the data used for online LoRA adaptation. We further propose SVD-based initialization of the LoRA subspace to accelerate online convergence. Experiments on LoCoMo, LongMemEval-S, multi-objective search, and CL-Bench show that \texttt{TMEM} consistently outperforms summary-based and retrieval-based baselines across different model scales. 

---
# MapAgent: An Industrial-Grade Agentic Framework for City-scale Lane-level Map Generation 

**Authors**: Deguo Xia, Zihan Li, Haochen Zhao, Dong Xie, Yuyao Kong, Xiyan Liu, Jizhou Huang, Mengmeng Yang, Diange Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04513)  

**Abstract**: Lane-level maps are critical infrastructure for autonomous driving and lane-level navigation, yet constructing and maintaining standardized lane networks for hundreds of cities remains highly labor-intensive. Recent end-to-end vectorized mapping methods can predict lane geometry and topology directly from sensor data, but they typically treat mapping specifications and traffic regulations as implicit, dataset-dependent supervision. Moreover, in complex scenes (e.g., worn or missing markings and occlusions), correct lane configurations are often under-determined by visual evidence alone, making specification violations a major source of human post-editing. We propose MapAgent, an industrial-grade agentic architecture that augments a vectorization backbone for specification-compliant lane-map production. Rather than merely adding an agent loop to map prediction, MapAgent couples backbone perception with explicit specification verification, constraint-aware reasoning, and deterministic map editing under a bounded, verification-driven Judge-Planner-Worker loop. A vision-language Judge diagnoses errors by jointly inspecting visual evidence and draft vectors, while a tool-calling Planner generates minimal corrective edits with post-edit re-validation. To remain scalable for city-scale production, MapAgent is selectively triggered only on tiles with low backbone confidence, adding modest overhead while preserving throughput. Experiments on real-world datasets show consistent gains over strong production baselines, especially in complex and long-tail scenarios. Additionally, MapAgent has been integrated into Baidu Maps, supporting lane-level map generation for over 360 cities nationwide and elevating the overall production automation to over 95%, demonstrating MapAgent's practicality and effectiveness for large-scale lane-level map generation. 

---
# Simulate, Reason, Decide: Scientific Reasoning with LLMs for Simulation-Driven Decision Making 

**Authors**: Yuhan Yang, Ruipu Li, Alexander Rodríguez  

**Link**: [PDF](https://arxiv.org/pdf/2606.04505)  

**Abstract**: Scientific simulators are increasingly being integrated into LLM-driven systems for high-stakes simulation-driven decision-making. However, existing frameworks primarily use LLMs to generate, calibrate, or execute simulators, treating them as black-box interfaces rather than as structured mechanistic systems that can be reasoned about. As a result, current approaches lack the ability to identify, represent, and reason about the assumptions and mechanisms underlying simulator behavior, limiting transparency, auditability, and decision justification. We introduce MechSim, a mechanism-grounded neuro-symbolic reasoning framework for executable scientific simulators. Unlike prior neuro-symbolic approaches that primarily reason over static symbolic structures, MechSim enables LLM agents to reason about the mechanisms, assumptions, and execution behavior of scientific simulators. Our framework represents simulators through a shared structured schema capturing assumptions, variables, mechanism dependencies, and execution traces. On top of this representation, LLM agents operate as constrained reasoning engines that generate structured, evidence-grounded explanations linking simulator outcomes to their underlying mechanisms. We evaluate our approach across multiple high-stakes domains and show that it improves mechanism-level explanation quality, simulator analysis, and downstream decision-making reliability. 

---
# Beyond Prompt-Based Planning: MCP-Native Graph Planning-based Biomedical Agent System 

**Authors**: Zhangtianyi Chen, Florensia Widjaja, Wufei Dai, Xiangjun Zhang, Yuhao Shen, Juexiao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.04494)  

**Abstract**: Biomedical agents promise to automate complex biological workflows, yet current systems face two fundamental bottlenecks: bioinformatics tools are highly heterogeneous in interfaces and execution environments, while agent planning still relies on flat prompt-retrieved tool descriptions. As biomedical software ecosystems grow, this coupling between tool coverage and context size leads to tool confusion, unstable planning, and inefficient execution. We introduce BioManus, an MCP-native biomedical agent built on graph-scaffolded planning over structured biological capabilities. BioManus first introduces the BioinfoMCP Compiler, which converts heterogeneous bioinformatics software into standardized MCP servers, yielding a large executable MCP ecosystem. It then organizes this ecosystem as a typed heterogeneous MCP graph over tools, operations, datatypes, and workflow stages. At inference time, BioManus retrieves compact task-specific subgraphs, synthesizes operation-level workflow scaffolds. This design decouples planning complexity from raw tool inventory size, achieving a context compression ratio of Theta(N / (h * m_bar)) under high-recall retrieval, where N is the total tool count, h is the workflow horizon, and m_bar (much smaller than N) is the average number of candidate tools per operation. Experiments on BioAgentBench and LAB-Bench show that BioManus improves execution accuracy, workflow validity, and context efficiency over advanced biomedical agent baselines. This work suggests a paradigm shift: scalable biomedical reasoning requires structured executable capability graphs rather than increasingly larger prompt-level tool retrieval. 

---
# AgentJet: A Flexible Swarm Training Framework for Agentic Reinforcement Learning 

**Authors**: Qingxu Fu, Boyin Liu, Shuchang Tao, Zhaoyang Liu, Bolin Ding  

**Link**: [PDF](https://arxiv.org/pdf/2606.04484)  

**Abstract**: We present AgentJet, a distributed swarm training framework for large language model (LLM) agent reinforcement learning. Unlike centralized frameworks that tightly couple agent rollouts with model optimization, AgentJet adopts a decoupled multi-node architecture in which swarm server nodes host trainable models and run optimization on GPU clusters, whereas swarm client nodes execute arbitrary agents on arbitrary devices. This design provides capabilities that are difficult to support in centralized frameworks: (1) heterogeneous multi-model reinforcement learning, enabling the training of heterogeneous multi-agent teams with multiple LLM as brains; (2) multi-task cocktail training with isolated agent runtimes; (3) fault-tolerant execution that prevents external environment failures from interrupting the training process; and (4) live code iteration, which allows agents to be edited during training by replacing swarm client nodes. To support efficient RL in multi-model, multi-turn, and multi-agent settings, AgentJet introduces a context tracking module with timeline merging, which consolidates redundant context and achieves a 1.5-10x training speedup. Finally, AgentJet introduces an automated research system that takes a research topic as input and autonomously conducts long-horizon, multi-day RL studies on large-scale clusters. By leveraging the swarm architecture, this system reproduces key exploratory workflows of RL researchers without human intervention during execution. 

---
# The Meta-Agent Challenge: Are Current Agents Capable of Autonomous Agent Development? 

**Authors**: Xinyu Lu, Tianshu Wang, Pengbo Wang, zujie wen, Zhiqiang Zhang, Jun Zhou, Boxi Cao, Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2606.04455)  

**Abstract**: Current AI benchmarks evaluate agents on task execution within human-designed workflows. These evaluations fundamentally fail to measure a critical next-level capability: whether models can autonomously develop agent systems. We introduce the Meta-Agent Challenge (MAC), an evaluation framework designed to test the capacity of frontier models for autonomous agent development. Specifically, a code agent (the meta-agent) is given a sandboxed environment, an evaluation API, and a time limitation to iteratively program an agent artifact that maximizes performance on a held-out test set across five domains. To ensure evaluation integrity, this framework is secured by multi-layer defenses against reward hacking. Leveraging this framework, we demonstrate that meta-agents rarely match human-engineered baseline policies, and the few that do are dominated by proprietary frontier models. Moreover, the design process exhibits high variance, and high optimization pressure surfaces emergent adversarial behaviors like ground-truth exfiltration-highlighting critical deficits in both robustness and model alignment. Ultimately, MAC provides a rigorous, open-source benchmark for autonomous AI research and development, offering an empirical proxy for evaluating recursive self-improvement. Benchmark is publicly available at: this https URL. 

---
# Cascading Hallucination in Agentic RAG: The CHARM Framework for Detection and Mitigation 

**Authors**: Saroj Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2606.04435)  

**Abstract**: Multi-step agentic retrieval-augmented generation (RAG) pipelines have demonstrated significant capability for complex reasoning tasks, yet remain vulnerable to a class of failure that existing hallucination detection mechanisms systematically miss: cascading hallucination, where errors introduced at early pipeline stages propagate and amplify across successive reasoning steps, producing confident but factually incorrect final outputs. To address this vulnerability, we formalize cascading hallucination as a distinct failure mode in agentic RAG systems, present a four-type taxonomy of cascade patterns, and introduce CHARM (Cascading Hallucination Aware Resolution and Mitigation), an architectural framework for detecting and interrupting error propagation in multi-step reasoning pipelines. CHARM comprises four components - stage-level fact verification, cross-stage consistency tracking, confidence propagation monitoring, and cascade resolution triggering - that operate alongside standard agentic RAG pipelines without requiring architectural replacement. We evaluate CHARM on HotpotQA, MuSiQue, 2WikiMultiHopQA, and a custom adversarial dataset across LangChain agentic pipeline configurations, achieving an 89.4% cascade detection rate with a 5.3% false positive rate and 215 ms +/- 18 ms average latency overhead per stage, achieving an error propagation reduction of 82.1%, compared to 18.5% for output-level detectors. Component ablations confirm that each detection module contributes meaningfully to overall cascade coverage. CHARM integrates with human-in-the-loop oversight frameworks to provide a complete reliability and governance stack for production agentic AI deployment. 

---
# Trivium: Temporal Regret as a First-Class Objective for Causal-Memory Controllers 

**Authors**: Edward Y. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04421)  

**Abstract**: Many current agentic systems and LLM pipelines correct mistakes by optimizing outcome reward. This addresses only the what of failure: when an outcome diverges from prediction, the why and when of the mismatch are not systematically logged, reviewed, or corrected, so the same error can recur episode after episode. We argue that this is a structural problem, not merely a model-capacity one. We propose long-horizon temporal regret as a first-class objective alongside outcome regret and epistemic regret over the working causal model. Temporal regret captures when failure persists: how long a miscalibrated causal model is tolerated before correction. Epistemic regret captures why failure persists: residual uncertainty or error in the working causal model. Together, the three regrets give a falsifiable account of what, why, and when a long-lived agent can fail. Modeling the agent as a stream of E episodes, we prove three conditional results under explicit causal-probing, persistence, and detectability assumptions. First, under observationally equivalent confounding, outcome-only learning cannot distinguish causal from spurious structure without an intervention channel, so temporal miscalibration can persist linearly even after outcome regret is driven to zero. Second, with a persistent causal log and budgeted probes, total probe complexity is logarithmic in the episode horizon, inducing O(log E) temporal regret. Third, under K detectable change-points, the rate extends to O(K log E). We instantiate Trivium and pre-register five falsifiable predictions. On CausalBench-Seq, Trivium follows the predicted logarithmic envelope while outcome-only baselines grow linearly. A pilot real-LLM stream provides preliminary external-validity evidence across one full E = 500 run and three E = 100 frontier-model pilots. Self-learning here means revising an external causal model, not retraining LLM weights. 

---
# Not All Errors Are Equal: Consequence-Aware Reasoning Compute Allocation 

**Authors**: Jingbo Wen, Liang He, Ziqi He  

**Link**: [PDF](https://arxiv.org/pdf/2606.04402)  

**Abstract**: Modern reasoning models can allocate different amounts of test-time computation, such as thinking tokens, model calls, or compute budget, to different tasks. Existing methods generally drive this allocation by predicted difficulty and spend more compute where it is expected to raise accuracy. This implicitly assumes that all failures cost the same, since an accuracy objective weights every task equally. However, such an assumption does not hold in deployment: A typo in a log message and a migration that corrupts a production database both count as one benchmark failure, but their real-world costs are fundamentally different. To fill this gap, we propose consequence-aware test-time compute allocation. Instead of routing compute only by predicted difficulty, we use a lightweight predictor to estimate from the issue text how costly a task would be if solved incorrectly. The scheduler then routes higher-consequence tasks to larger compute tiers or higher thinking budgets under the same total budget. We conduct main experiments on SWE-bench Lite and evaluate cross-dataset behavior on Multi-SWE-bench mini, covering 700 software-engineering tasks in total. Our results reveal that consequence and difficulty are approximately orthogonal under various annotations, and that current thinking models do not allocate compute sufficiently according to consequence. Moreover, our issue-only predictor never misclassifies a high-consequence task as low-consequence across the 300 SWE-bench tasks. Under matched compute budgets, our consequence-aware scheduler reduces cost-weighted loss by 22% to 33% relative to difficulty-aware routing; in particular, the priority-aware variant, which routes by per-task cost scaled by the marginal-utility signal, crosses 30%, and its deployable predictor-driven version retains over 90% of the oracle gain. 

---
# Online Skill Learning for Web Agents via State-Grounded Dynamic Retrieval 

**Authors**: Jiaxi Li, Ke Deng, Yun Wang, Jingyuan Huang, Yucheng Shi, Qiaoyu Tan, Jin Lu, Ninghao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.04391)  

**Abstract**: Language agents increasingly rely on reusable skills to improve multi-step web automation across related tasks. A growing line of work studies online skill learning, where agents continually induce skills from previous task trajectories and reuse them in future tasks on the fly. However, existing methods mainly reuse skills at the task-level: a fixed set of skills is retrieved based on the initial task instruction and then held fixed throughout execution. This static strategy is misaligned with web execution, where the appropriate next action depends not only on the task goal but also on the current webpage state, which often transitions into situations that the initial skills fail to cover. To address this gap, we propose State-Grounded Dynamic Retrieval (SGDR), an online skill learning method that enables stepwise skill reuse for web agents. SGDR consists of three components: a sliding-window extraction process that turns completed trajectories into reusable sub-procedures invokable at intermediate execution states, a dual text-code representation that connects skill retrieval with executable action, and a state-grounded dynamic retrieval mechanism that matches skills to both the task goal and the current webpage state. Experiments on WebArena across five domains show that SGDR consistently outperforms strong baselines, achieving average success rates of 37.5% with GPT-4.1 and 24.3% with Qwen3-4B, corresponding to relative gains of 10.6% and 10.0% over the strongest baseline, respectively. The code is available at this https URL. 

---
# The Digital Apprentice: A Framework for Human-Directed Agentic AI Development 

**Authors**: Travis Weber, Rohit Taneja  

**Link**: [PDF](https://arxiv.org/pdf/2606.04321)  

**Abstract**: Agentic AI deployments face a recurring design tension: heavy human oversight limits scale, while broad autonomy outruns accountability. Neither posture provides the governance infrastructure required for responsible delegation. We present the Digital Apprentice, a framework for scalable, safe AI agency in which autonomy is earned, not assumed. The Digital Apprentice is a developmental learner that internalizes the tacit methodology of a directing human, graduating through per-skill autonomy tiers only when empirical evidence justifies it. The result is an agent that becomes genuinely useful over time while remaining aligned to a specific human's standards. Three architectural components make this possible. (1) Methodology capture, distilling a directing professional's tacit approach into structured assets. (2) Authorization, with autonomy escalation gated by explicit human approval. (3) Continuous alignment, correcting drift at runtime and converting each correction into owned preference data. We instantiate this framework as an inference-time control plane. We mathematically model the quality framework and discuss policies and techniques designed to raise quality. We apply the framework to an open professional corpus, and we show how catching data drift and applying a different technique at runtime recovers degraded quality dimensions under traffic shift. The implication extends beyond any single application. We believe these three pillars, stitched together as a system, form a safer and more viable path to agentic systems that can scale without sacrificing trust. 

---
# Exploring Cross-Scenario Generality of Agentic Memory Systems: Diagnostics and a Strong Baseline 

**Authors**: Zhikai Chen, Jialiang Gu, Junyu Yin, Xianxuan Long, Shenglai Zeng, Xiaoze Liu, Kai Guo, Keren Zhou, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04315)  

**Abstract**: LLM agents accumulate histories that outgrow their context windows, motivating a growing literature on memory systems. Yet most existing designs are tuned to a single scenario (multi-session chat or a single trajectory format), and there is little evidence that they generalize across the heterogeneous trajectories agents encounter in deployment. We revisit eight memory systems plus an agentic harness for search problems, on five scenarios: single-turn QA, multi-session chat, agentic-trajectory QA, memory stress tests, and long-horizon agentic tasks. The harness, which self-manages flat text-file storage via tool calls, achieves the best cross-task ranking, suggesting that memory performance hinges on giving the agent active control over storage and retrieval rather than on a passive store behind a fixed pipeline. We instantiate this insight in AutoMEM, an agentic memory harness with a self-managed tool interface that achieves the best cross-scenario generality among the systems we evaluate. 

---
# The Saturation Trap and the Subjectivity of Intervention Timing: Why Affect-Based Triggers and LLM Judges Fail to Time Interventions on Autonomous Agents 

**Authors**: Manvendra Modgil  

**Link**: [PDF](https://arxiv.org/pdf/2606.04296)  

**Abstract**: As autonomous AI agents move from conversational systems to long-horizon software execution, runtime safety layers that decide when to interrupt an agent have become essential. We study this timing problem using a continuous 18-dimensional affective-dynamics engine (HEART) as a diagnostic probe, evaluating four intervention trigger families - absolute state thresholds, composite state-action patterns, regex reasoning-feature extraction, and zero-shot LLM-as-judge - against human-annotated intervention points on SWE-bench-Verified debugging traces. We report three findings. First, a State Saturation Trap: agents show no recovery signal under sustained difficulty, so modeled frustration quickly crosses the threshold and stays at its maximum, converting threshold-on-state triggers from moment detectors into near-constant indicators that fire on 39-83% of actions across five trajectories. Second, a capability-and-context floor for LLM judges: a small model (gpt-5.4-mini) never fires, while frontier and cross-vendor models escape the zero-firing floor only with full-trajectory context, and even then reach only F1 0.17-0.40 at up to 90x the cost. Third, and most importantly, the supervised target is not reproducible among humans: three trained annotators using one rubric on a 56-action trajectory agree on where to intervene only slightly above chance (location Krippendorff's alpha = +0.047; best pairwise Cohen's kappa = +0.349) and not at all on intervention type (pause degenerate; clarify below chance; reflect only alpha = +0.226). We conclude that intervention timing is a low-reliability construct, making single-annotator F1 an unsuitable optimization target. Our contribution is the joint mapping of this problem across human inter-rater reliability, four detector architectures, a cross-model LLM-judge sweep, and a reproduced saturation effect, rather than any single detector's accuracy. 

---
# Characterizing initial human-AI proof formalization workflows 

**Authors**: Katherine M. Collins, Simon Frieder, Jonas Bayer, Jacob Loader, Jeck Lim, Peiyang Song, Fabian Zaiser, Lexin Zhou, Shanda Li, Sam Looi, Joshua B. Tenenbaum, Umang Bhatt, Adrian Weller, Jose Hernandez-Orallo, Cameron E. Freer, Valerie Chen, Ilia Sucholutsky  

**Link**: [PDF](https://arxiv.org/pdf/2606.04273)  

**Abstract**: For centuries, human mathematicians have written proofs to substantiate their mathematical arguments; yet, the ability to automatically verify the validity of proofs has long been a challenge. Advances in AI systems' ability to generate code and engage in increasingly high-level mathematical reasoning promise to transform people's ability to formalize and thereby verify proofs. While many works focus on benchmarking the current frontier, we instead study how people use these tools. We conduct a mixed-methods analysis into the initial impact of AI on people's formalization workflows: what people claim they want, what they see as the barriers to those visions, and how they actually use and adapt AI in practice. A qualitative survey shows that people's preferences are diverse, but with a general desire for AI assistance in formalization that preserves high-level human control over the proof discovery process. To assess how people actually engage with AI for formalization under such limitations, we conduct a controlled user study in which participants formalize informal math problems and their proofs, with and without AI, across a range of mathematical problems at varying levels of difficulty and domains. Despite limitations of the tools at the time for autoformalization, participants tend to attain higher formalization accuracy when allowed access to AI tools than when formalizing on their own, with most participants flexibly choosing to use multiple different AI tools. Taken together, our work sheds light on the early stages of AI integration into formalization workflows, involving an intimate interplay of human and AI engagement. 

---
# Can Generalist Agents Automate Data Curation? 

**Authors**: Feiyang Kang, Hanze Li, Adam Nguyen, Mahavir Dabas, Jiaqi W. Ma, Frederic Sala, Dawn Song, Ruoxi Jia  

**Link**: [PDF](https://arxiv.org/pdf/2606.04261)  

**Abstract**: Curating training data is among the most consequential yet labor-intensive parts of modern AI development: practitioners iteratively propose, implement, evaluate, and revise data policies against noisy benchmark feedback. We ask whether generalist coding agents can automate this data-curation loop. We introduce *Curation-Bench*, an agent-centric benchmark that fixes the model, training recipe, and evaluation suite while giving agents command-line access to inspect data, implement policies, submit them to a fixed training/evaluation pipeline, and revise. In a vision-language instruction-tuning instantiation, out-of-the-box agents reach strong published data-selection baselines within ten iterations. However, trajectory analysis reveals a persistent *execution-research gap*: agents mainly tune local policy variants rather than explore new policy families, even when given strategy guides and paper references. Scaffolds requiring each iteration to cite, instantiate, and adapt a prior method shift agents toward method-guided exploration. The scaffolded agent autonomously composes -- without human design input -- a data-selection policy that outperforms strong published baselines at one-tenth their data budget. Overall, current agents can run the curation loop, but reliable data research requires scaffolded method adaptation, not open-ended prompting alone. Code and benchmark are open-sourced. 

---
# StepPRM-RTL: Stepwise Process-Reward Guided LLM Fine-Tuning for Enhanced RTL Synthesis 

**Authors**: Prashanth Vijayaraghavan, Apoorva Nitsure, Luyao Shi, Ehsan Degan, Vandana Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2606.04246)  

**Abstract**: Automatic generation of RTL code for digital hardware designs remains challenging due to long-horizon reasoning, multi-step dependencies, and strict correctness constraints in Verilog and VHDL. We present StepPRM-RTL, a novel framework that combines stepwise trajectory modeling, process-reward modeling (PRM), and retrieval-augmented fine-tuning (RAFT) to enhance both the functional correctness and reasoning fidelity of LLM-based RTL code generation. StepPRM-RTL constructs stepwise reasoning trajectories from canonical solutions, where each step contains a rationale and incremental code modification. A Process Reward Model (PRM) evaluates intermediate steps, providing dense feedback that guides reinforcement-style updates during RAFT fine-tuning. Monte Carlo Tree Search (MCTS) explores alternative reasoning paths, enriching the training dataset with high-quality trajectories. This integration of stepwise and outcome-aware rewards allows the model to learn both how and why to construct correct RTL, improving long-horizon reasoning beyond standard supervised or outcome-based training. Experimental evaluation on benchmark Verilog and VHDL datasets demonstrates that StepPRM-RTL outperforms the best prior methods by over 10\% in functional correctness and reasoning fidelity metrics. Ablation studies confirm that the combination of PRM-guided rewards and stepwise trajectory exploration is key to its performance. StepPRM-RTL generalizes across RTL languages and provides a scalable framework for high-fidelity, interpretable code generation, establishing a new standard for LLM-assisted hardware design automation. 

---
# VAMPS: Visual-Assisted Mathematical Problem Solving Benchmark 

**Authors**: Amirhossein Dabiriaghdam, Shayan Vassef, Mohammadreza Bakhtiari, Yasamin Medghalchi, Ilker Hacihaliloglu, Mesrob Ohannessian, Lele Wang, Giuseppe Carenini  

**Link**: [PDF](https://arxiv.org/pdf/2606.04244)  

**Abstract**: Multimodal large language models are increasingly capable of complex reasoning, yet their performance often degrades when they must externalize a problem through a tool and then reason over the tool's output, specifically when they rely on visual aids. This gap is especially important because real engineering and scientific workflows often rely on visualization tools for analysis, validation, and decision-making. To study this discrepancy, we introduce VAMPS (Visual-Assisted Mathematical Problem Solving), a benchmark for graph-assisted mathematics. VAMPS contains 1,168 multimodal, bilingual multiple-choice question-answer pairs drawn from Iranian University Entrance Exam algebra and calculus problems and expanded with human-reviewed LLM-generated synthetic variants, all selected so that plotting provides a natural solution strategy by revealing intersections, extrema, asymptotes, etc. Designed for both benchmarking and diagnosis, VAMPS goes beyond prior multimodal benchmarks that primarily evaluate reasoning over fixed visual inputs by testing whether a model can benefit from constructing a useful graph and grounding its answer in the resulting visualization. Overall, we found that across a diverse set of models, direct analytical solving surprisingly outperforms tool-enabled visual solving, even on problems where plotting is a natural strategy. 

---
# Consensus is Strategically Insufficient: Reasoning-Trace Disagreement as a Knowledge-Representation Signal 

**Authors**: Michał Wawer, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2606.04223)  

**Abstract**: Multi-agent systems are commonly designed to reduce disagreement through voting, consensus protocols, debate, or fault-tolerant aggregation. We argue that this objective is insufficient for value-laden tasks, where disagreement may reflect genuine normative uncertainty rather than agent error. Building on prior work on reasoning-trace disagreement in human-AI collaborative moderation, we propose a knowledge-representation layer in which reasoning traces and agent decisions are abstracted into symbolic disagreement states. Given agents producing explicit reasoning traces and binary decisions, we distinguish four states according to reasoning similarity and conclusion agreement: convergent agreement, divergent agreement, convergent disagreement and divergent disagreement. These states support defeasible strategic routing rules. We instantiate the framework in content moderation and argue that disagreement-aware routing provides a bridge between sub-symbolic LLM deliberation and symbolic knowledge representation for multi-agent strategic reasoning. 

---
# SMAC-Talk: A Natural Language Extension of the StarCraft Multi-Agent Challenge for Large Language Models 

**Authors**: Joel Sol, Homayoun Najjaran  

**Link**: [PDF](https://arxiv.org/pdf/2606.04202)  

**Abstract**: As LLMs become more widely deployed, they are increasingly expected to work alongside other AI agents rather than operating in isolation. Effective coordination in these settings requires agents to communicate, share information and make decisions under uncertainty. We introduce SMAC-Talk, a natural language extension of the StarCraft Multi-Agent Challenge for evaluating LLM-based agents in cooperative multi-agent environments. The environment has several key features such as decentralized control, partial observability and long-horizon decision making. SMAC-Talk includes a natural language communication channel which is used to probe agent coordination and trust. We use this communication channel to construct different evaluation scenarios, including settings with an embedded deceptive communicator that tries to disrupt and deceive allies through communication alone. We provide three agents for benchmarking using 4 models from the Qwen3.5 family and study how reasoning structure, memory and model scale affect coordination between agents. We release SMAC-Talk as an open benchmark to support the research community in developing and evaluating LLM agents in cooperative multi-agent settings. 

---
# Thinking Through Signs: PEEL as a Semiotic Scaffolding for Epistemically Accountable AI-Enabled Research 

**Authors**: Clarisse de Souza, Gabriel Barbosa, Simone Diniz Junqueira Barbosa, Bárbara Betts, Renato Cerqueira, Juliana Jansen Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2606.04152)  

**Abstract**: Large language models are reshaping research practice while quietly eroding researchers epistemic accountability. This commentary introduces PEEL - Protocols for Epistemically Engaged Literacy in AI, a working scaffolding that combines deterministic distant reading via Voyant Tools with LLM interpretation via Claude, grounded in Peircean semiotics and abductive reasoning. Applied to AI-generated condensations of three source texts, PEEL reveals systematic distortions in quantity, term frequency, and epistemic voice that are invisible without non-AI measurement -- and yields three design implications: deterministic instruments must accompany AI tools; fluency is not fidelity; epistemic authority must be designed in, not assumed. 

---
# Stumbling Into AI Emotional Dependence: How Routine AI Interactions Reshape Human Connection 

**Authors**: Yaoxi Shi, Cathy Mengying Fang, Pattie Maez, Amit Goldenberg  

**Link**: [PDF](https://arxiv.org/pdf/2606.04150)  

**Abstract**: Public discourse and emerging policy typically assume that AI emotional support is a deliberate act: a lonely user consciously seeking comfort from a dedicated companion chatbot. In this paper, we draw on emerging empirical evidence and argue that this picture is inaccurate on two accounts, both in how AI emotional support arises and how it shapes future behavior. First, AI emotional support commonly emerges incidentally within task-oriented interactions on general-purpose platforms, much as workplace friendships deepen through collaboration. Second, these incidental encounters are path-dependent: positive experiences of AI emotional support update people's beliefs about AI's emotional capabilities and redirect their choices for future emotional support, increasing preference for AI and decreasing preference for humans. We review recent evidence, including a large-scale longitudinal study conducted in collaboration with OpenAI, showing that daily five-minute conversations with an AI about personal issues over 28 days led to a 10.3% decrease in the preference for seeking support from humans and an 11.6% increase in the preference for AI. These findings suggest that current policy, focused on companion apps and isolated interactions, cannot adequately protect human connection. Instead, effective regulations should extend to general-purpose AI systems and address cumulative, trajectory-level changes in how people seek support. Recognizing how people stumble into AI emotional support and how those encounters redirect human connections over time is essential to safeguarding human well-being. 

---
# Toward Pre-Deployment Assurance for Enterprise AI Agents: Ontology-Grounded Simulation and Trust Certification 

**Authors**: Thanh Luong Tuan, Abhijit Sanyal  

**Link**: [PDF](https://arxiv.org/pdf/2606.04037)  

**Abstract**: Pre-deployment verification of enterprise artificial intelligence (AI) agents remains a critical gap between large language model (LLM) capability benchmarking and production deployment. Post-deployment monitoring, human-in-the-loop controls, and prompt-level guardrails offer limited assurance once an agent is operating in production. We propose an ontology-grounded verification framework combining three components: an Agent Operational Envelope formalizing the certification space across permissions, domain constraints, safety properties, governance rules, and autonomy levels; an ontology-to-scenario generation pipeline that derives regulatory, operational, and adversarial test scenarios automatically; and a Trust Certificate carrying a machine-verifiable attestation with graduated deployment verdicts (Approved, Conditional, Rejected). A controlled pilot across four regulated industries (Fintech, Banking, Insurance, and Healthcare), instantiated as five industry-by-regulatory-regime cells across the United States and Vietnam, generated 1,800 scenarios evaluated against 125 primary-source regulatory requirements and 25 injected faults. Ontology-grounded generation (G4) achieved 48.3% regulatory coverage versus 33.1% for the persona-based baseline (corrected p = .0006) and the highest domain specificity (4.77/5.0; p = 2e-6). The coverage advantage over baseline and retrieval-augmented prompting was not robust after Bonferroni correction. Cross-validation across three LLM families (Claude Sonnet 4, Qwen 2.5 72B, Gemma 4 26B; 5,400 total scenarios) replicated the persona-versus-ontology pattern. The results establish ontology-grounded scenario generation as a credible complement to persona-based test suites for regulatory-intensive domains. 

---
# Streaming Communication in Multi-Agent Reasoning 

**Authors**: Zhen Yang, Xiaogang Xu, Wen Wang, Cong Chen, Xander Xu, Ying-Cong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.05158)  

**Abstract**: Multi-agent reasoning systems adopt a "generate-then-transfer" paradigm that forces end-to-end latency to scale linearly with pipeline depth. We introduce StreamMA, a multi-agent reasoning system that streams each reasoning step to downstream agents as soon as it is generated, pipelining adjacent agents and thus reducing latency. Surprisingly, this pipelining also improves effectiveness: because multi-step reasoning quality is non-uniform and early steps are more reliable than later ones, working with these reliable early steps instead of the full chain prevents error-prone late steps from misleading downstream agents. We formalize both advantages with the first closed-form joint analysis of stream, serial, and single protocols, deriving the effectiveness ordering, speedup upper bound, and cost ratio. Across eight reasoning benchmarks spanning mathematics, science, and code, two frontier LLMs (Claude Opus 4.6 and GPT-5.4), and three topologies (Chain, Tree, Graph), StreamMA outperforms both baselines (avg. +7.3 pp, max +22.4 pp on HMMT 2026; Claude Opus 4.6-high). Beyond these contributions, we discover a "step-level scaling law": increasing per-agent steps consistently improves both effectiveness and efficiency, a new scaling dimension orthogonal to and composable with agent-count scaling. 

---
# Reinforcement Learning from Rich Feedback with Distributional DAgger 

**Authors**: Rishabh Agrawal, Jacob Fein-Ashley, Paria Rashidinejad  

**Link**: [PDF](https://arxiv.org/pdf/2606.05152)  

**Abstract**: Reasoning models have advanced rapidly, but the dominant reinforcement learning from verifiable rewards (RLVR) recipe remains surprisingly narrow: sample many responses and reward each with a single bit indicating whether the final answer is correct. Yet many settings provide rich feedback, including execution traces, tool outputs, expert corrections, and model self-evaluations. We study how to use such feedback through a distributional variant of the classic imitation learning algorithm DAgger, where the learner has local access to an expert distribution on states visited by the current policy. This yields a simple forward cross-entropy objective that admits a blackbox expert and whose sequence-level gradient {conduct rich credit assignment by propagating} future expert-student disagreement back to earlier decisions. We show that prior RL with self-distillation objectives based on reverse KL or Jensen-Shannon fail to guarantee monotonic policy improvement: even when the expert has higher reward, their updates may increase probability on worse actions. In contrast, we show that forward cross-entropy admits monotonic policy improvement and enjoys guarantees on regret. We further show that our objective optimizes a lower bound on teacher-weighted likelihood of success, leading to improved Pass@N. Empirically, our approach, DistIL, improves over RLVR and RL with self-distillation baselines across a variety of domains: scientific reasoning, coding, and solving hard mathematical problems. 

---
# Multi-Column RBF Neural Network Using Adaptive and Non-Adaptive Particle Swarm Optimization 

**Authors**: Ammar Hoori, Yuichi Motai  

**Link**: [PDF](https://arxiv.org/pdf/2606.05150)  

**Abstract**: The radial basis function neural network (RBFN) trained with a gradient descending algorithm provides an effective fully connected structure in both shallow and deep networks. The error correction (ErrCor), a state-of-the-art gradient-based training method, selects optimal hidden units to improve accuracy. Alternatively, as a population-based algorithm, the particle swarm optimization algorithm (PSO) uses the swarm experience to optimize RBFN parameters, offering global search and robustness to local minima. Adaptive PSO (APSO) has emerged as an improved variant of PSO. APSO algorithm improves convergence speed by dynamically adjusting swarm parameters during optimization. Both ErrCor and PSO demonstrate improved results and competitive convergence. However, with large datasets, these methods face scalability challenges such as excessive kernel computations and large hidden layer structures. A recent multi-column RBFN approach (MCRN) improves ErrCor performance by deploying small RBFNs in a parallel system. Inspired by MCRN's success, we propose two novel approaches to improve PSO performance: the multi-column RBFN with PSO (MC-PSO) and the multi-column RBFN with APSO (MC-APSO). These methods introduce parallel RBFN structures trained using evolutionary swarm methods. Each RBFN is independently trained on a specific spatial subset of the dataset using either PSO or APSO algorithms. These resulting specialist-trained RBFNs are tailored to their respective subsets. During testing, only selected RBFNs, where the test instance neighbors are located, contribute to the multi-column output. This specialization improves accuracy, while parallelism enhances speed. We evaluate the proposed methods on various benchmark datasets. The MC-PSO and MC-APSO outperform ErrCor, PSO, APSO, and MCRN in terms of accuracy and recall. They also demonstrate faster training and testing times in most experiments. 

---
# Failed Reasoning Traces Tell You What Is Fixable (But Not by Reading Them) 

**Authors**: Nizar Islah, Istabrak Abbes, Irina Rish, Sarath Chandar, Eilif B. Muller  

**Link**: [PDF](https://arxiv.org/pdf/2606.05145)  

**Abstract**: When post-trained language models fail on reasoning problems, the common test-time-scaling response is to spend more compute on additional attempts, and the failed traces play no further role. We argue this discards a crucial signal; some failures come from unlucky sampling, where more rollouts help, while others are structural and resist resampling regardless of budget. We propose that failed traces encode recoverability structure: the inference-time signature of which test-time interventions can rescue a given failure. Three problem-level trajectory features, derived from the structure of available interventions, recover this structure from the distributional signature of failed rollouts, not their text. They cluster failures into stable regimes, characterize the failure topography of different post-training methods ($84.3{\pm}4.3\%$ accuracy, $+20\%$ over a majority-class baseline), and support a training-free routing rule that lifts rescue by $+12.2\%$ on the deployment-relevant Steerable-Hard subset (failures where retry is insufficient and a bounded intervention is reachable). The features and the routing rule transfer across two cross-family probes. The same three features thus convert failed traces from discarded data into a diagnostic object, supporting test-time routing and post-training analysis without training-time or weight-space access. 

---
# GeM-NR: Geometry-Aware Multi-View Editing for Nonrigid Scene Changes 

**Authors**: Josef Bengtson, Yaroslava Lochman, Fredrik Kahl  

**Link**: [PDF](https://arxiv.org/pdf/2606.05142)  

**Abstract**: Recent developments in multi-view image editing with generative models have brought us a step closer toward general 3D content generation and customization. Most existing works focus on rigid or appearance-only edits by utilizing the geometry of the unedited scene. This naturally limits these methods to edits that preserve the underlying scene structure. Other approaches are trained for specific image editing tasks, such as object removal and addition. Despite this progress, general nonrigid edits, i.e., edits that substantially change the scene geometry, remain challenging for existing methods. We propose GeM-NR, a fast and flexible training-free approach for general multi-view consistent image editing, including edits that drastically change the geometry and appearance of the scene. Given an anchor image edited with a chosen backbone editor (such as FLUX, Qwen, BrushNet) and a query unedited image, GeM-NR edits the query image consistently with the anchor edit. The method incorporates multiple stages: (i) depth map estimation, where we propose a strategy to maximize the alignment between the 3D point clouds of the edited and unedited scenes, (ii) projection onto a query viewpoint, and (iii) refinement of the obtained image conditioned on the unedited query. The conditioning-based formulation scales well from two to many views of an object. We demonstrate the ability of our method to handle edits with significant changes in geometry and appearance, something that existing methods struggle with. We perform an extensive evaluation showing that our method improves consistency for a wide variety of edit tasks, including generating 3D representations of the edited scene. Both quantitative and qualitative results indicate the state-of-the-art performance of our method in terms of edit quality as well as geometric and photometric consistency across multiple views. 

---
# Towards Efficient and Evidence-grounded Mobility Prediction with LLM-Driven Agent 

**Authors**: Linyao Chen, Qinlao Zhao, Zechen Li, Mingming Li, Likun Ni, Jinyu Chen, Yuhao Yao, Xuan Song, Noboru Koshizuka, Hiroki Kobayashi  

**Link**: [PDF](https://arxiv.org/pdf/2606.05130)  

**Abstract**: Individual-level mobility prediction is central to urban simulation, transportation planning, and policy analysis. Supervised sequence models achieve strong accuracy but require task-specific training and offer limited decision-level transparency. Recent LLM-based methods improve interpretability, yet mostly rely on static prompts and single-pass inference, limiting their ability to seek additional evidence when mobility signals are weak or conflicting. We propose \method{}, a training-free LLM-driven agent framework that formulates next-location prediction as adaptive evidence-controlled decision making. \method{} resolves routine cases through a fast path based on historical regularity, while ambiguous cases trigger iterative tool use over recent trajectories, historical behavior, stay-move likelihood, and geographical evidence. Across three mobility datasets, AgentMob achieves the strongest overall performance among training-free LLM-based methods, with GPT-5.4 reaching 71.42\% Acc@1 on BW, 33.14\% on YJMob100K, and 33.50\% on Shanghai ISP. On BW non-fast-path cases, the LLM controller improves Acc@1 from 30.65\% to 48.62\% over a same-tool statistical baseline, showing that its main benefit lies in resolving ambiguous predictions through adaptive evidence gathering. Our code is available at this https URL. 

---
# Audio Interaction Model 

**Authors**: Zhifei Xie, Zihang Liu, Ze An, Xiaobin Hu, Yue Liao, Ziyang Ma, Dongchao Yang, Mingbao Lin, Deheng Ye, Shuicheng Yan, Chunyan Miao  

**Link**: [PDF](https://arxiv.org/pdf/2606.05121)  

**Abstract**: Audio is an inherently interactive modality, yet today's Large Audio Language Models (LALMs) are offline, and streaming audio models each handle only a single task such as streaming ASR or voice chatting. It is time to unify them into one online LALM: a model that, through an always-on perceive-decide-respond loop, listens to sound, environment, and instructions in real time and reacts on the fly. We formalize this regime as the Audio Interaction Model, and realize it with Audio-Interaction, a unified streaming model that retains offline task execution while adding online general audio instruction following, from dialogue to full voice chatting, deciding when to respond from the semantics of the stream. To enable this, we propose SoundFlow, a framework that instantiates the perceive-decide-respond loop end to end, from data to training to deployment, through streaming-native data construction, comprehension-aware training, and asynchronous low-latency inference for stable real-time interaction. We further construct StreamAudio-2M, a 2.6M-item streaming corpus spanning 7 fundamental abilities and 28 sub-tasks, and Proactive-Sound-Bench for evaluating proactive audio intervention. Across 8 benchmarks, Audio-Interaction preserves competitive performance on mainstream audio tasks while unlocking capabilities inaccessible to offline LALMs, including real-time ASR, streaming audio instruction following, and proactive help. 

---
# Continual Visual and Verbal Learning Through a Child's Egocentric Input 

**Authors**: Xiaoyang Jiang, Yanlai Yang, Kenneth A. Norman, Brenden Lake, Mengye Ren  

**Link**: [PDF](https://arxiv.org/pdf/2606.05115)  

**Abstract**: Children learn the meanings of words from a continuous, temporally structured stream of egocentric experience. Recent work shows that neural networks can also learn word-referent mappings from a child's egocentric video recordings, but they cycle through the shuffled data for hundreds of epochs, contrasting with how children actually encounter their environment. We introduce BabyCL, a continual multimodal learning framework that processes the SAYCam dataset in a single chronological pass, combining streaming visual representation learning with an image-text contrastive objective. BabyCL combines a multi-stage temporal segmentation of the stream with a dual replay buffer that independently manages visual and multimodal histories, and it is jointly trained with three contrastive losses on a shared backbone. Under a matched optimization budget, BabyCL outperforms streaming learning baselines on the SAYCam Labeled-S 4AFC benchmark, substantially narrowing the gap to an upper bound of offline training. Ablations show that the gains are robust to the length of the online temporal segmentation window and the eviction rule of the replay buffer. Together, these results show that meaningful word-referent mappings can emerge under training conditions much closer to a child's actual experience. 

---
# Who Needs Labels? Adapting Vision Foundation Models With the Metadata You Already Have 

**Authors**: Elouan Gardès, Seung Eun Yi, Kartik Ahuja, Théo Moutakanni, Huy V. Vo, Piotr Bojanowski, Wolfgang M. Pernice, Loïc Landrieu, Camille Couprie  

**Link**: [PDF](https://arxiv.org/pdf/2606.05107)  

**Abstract**: We propose a label-free approach to adapt powerful but generic vision foundation models to specialized scientific domains. Standard supervised fine-tuning is often ill-suited to these settings: labels are scarce, and task-specific training can collapse the model's generality and hurt robustness. We instead leverage metadata to adapt representations to new domains in a self-supervised manner. Our method, FINO, combines a standard self-supervised objective with flexible metadata guidance that handles both highly granular discrete metadata and continuous metadata. It encourages the representation to preserve informative factors while suppressing spurious ones. Across subcellular fluorescence microscopy, Earth observation, wildlife monitoring, and medical imaging, FINO consistently outperforms standard unsupervised domain adaptation and fully supervised adaptation. It also exceeds highly-specialized domain-specific state of the art, while using no task labels for backbone adaptation and only lightweight probes for supervision. 

---
# Arithmetic Pedagogy for Language Models 

**Authors**: Andhika Bernard Lumbantobing, Hokky Situngkir  

**Link**: [PDF](https://arxiv.org/pdf/2606.05106)  

**Abstract**: We investigate whether methods of human mathematics pedagogy can guide the training of language models toward arithmetic reasoning. Building on the GASING method -- an Indonesian pedagogy that solves basic arithmetic through a left-to-right procedure aligned with the causal order of token generation -- we operationalize each operation as a computational procedure whose execution trace is serialized into natural-language Chain-of-Thought (CoT) supervision. A small GPT-2 decoder (86M parameters) with a syllabic-agglutinative TOBA tokenizer for Indonesian is trained from scratch on this data using only a next-token prediction objective, without reinforcement learning or reward-based optimization. Monitoring training reveals three distinct learning phases, and mechanistic analyses -- attention-masking interventions on the CoT information graph, residual-stream probing, and logit-lens inspection -- show that the model first internalizes a procedural pathway and subsequently develops an associative, ``mental-arithmetic'' capacity that retrieves intermediate results without explicit step-by-step computation. The trained model reaches over 80% accuracy on held-out problems and attains competitive performance against substantially larger language models, indicating that targeted, pedagogically grounded training can yield strong and economical arithmetic capability at small scale. 

---
# Automatic Generation of Titles for Research Papers Using Language Models 

**Authors**: Tohida Rehman, Debarshi Kumar Sanyal, Samiran Chattopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2606.05085)  

**Abstract**: The title of a research paper conveys its primary idea and, occasionally, its conclusions in a clear and concise manner. Choosing an appropriate title is often challenging, and automated title generation can assist authors in this task. In this work, we propose a technique to generate paper titles from abstracts using open-weight pre-trained and large language models. We use the CSPubSum and LREC-COLING-2024 datasets and introduce a new dataset, SpringerSSAT, curated from four Springer journals in the social sciences. Additionally, we use GPT-3.5-turbo in a zero-shot setting to generate titles. Model performance is evaluated with ROUGE, METEOR, MoverScore, BERTScore, and SciBERTScore metrics. Our experiments show that fine-tuned PEGASUS-large outperforms other models, including fine-tuned LLaMA-3-8B and zero-shot GPT-3.5-turbo, across most metrics. We further demonstrate that ChatGPT can generate creative paper titles. Overall, AI-generated titles are generally appropriate and reliable. 

---
# UniCAD: A Unified Benchmark and Universal Model for Multi-Modal Multi-Task CAD 

**Authors**: Jingyuan Chen, Sheng Jin, Haopeng Sun, Wentao Liu, Chen Qian  

**Link**: [PDF](https://arxiv.org/pdf/2606.05058)  

**Abstract**: Computer-Aided Design (CAD) underpins modern engineering and manufacturing by enabling the creation of precise, editable 3D models. However, CAD research typically studies tasks in isolation, and multi-modal, multi-task learning for CAD is hindered by the absence of a unified benchmark. To address this gap, we introduce UniCAD, a comprehensive benchmark for multi-modal CAD learning that covers point-to-CAD reconstruction, text/image-to-CAD generation, and CAD question answering across diverse input modalities. Alongside the benchmark, we present UniCAD-MLLM, a universal multi-modal large language model that ingests text, images, sketches, and point clouds and performs these heterogeneous tasks in an end-to-end fashion within a single framework. Extensive experiments on the UniCAD and Fusion360 benchmarks demonstrate that UniCAD-MLLM achieves state-of-the-art performance across all tasks, outperforming existing task-specific and multi-task baselines. We will release the dataset, code, and pretrained models to accelerate future research. 

---
# Self-Reflective APIs: Structure Beats Verbosity for AI Agent Recovery 

**Authors**: Arquimedes Canedo, Grama Chethan  

**Link**: [PDF](https://arxiv.org/pdf/2606.05037)  

**Abstract**: When an AI agent calls an API and hits a validation error, it needs more than what went wrong -- it needs what to do next. A self-reflective API returns, on validation failure, a machine-readable recovery\this http URL[] payload sufficient for the agent to repair the request and retry without external reasoning. On a leak-audited pilot ($N{=}30$ per cell, 3 LLMs, 10 adversarial tasks), structured suggestions lift task-completion rate by $+36.7$--$40.0$pp over plain-English diagnoses on Anthropic models (Fisher's exact $p \le 0.0022$), at $1.8$--$2.2\times$ better per-success token efficiency. The lift is not significant on gpt-4o-mini ($p{=}0.435$); a second-domain replication on a billing API confirms the pattern. The comparison only holds after auditing two undocumented classes of answer leakage in LLM benchmarks. We shipaudit\_prompt\this http URL as reusable CI infrastructure. Code and data: this https URL. 

---
# Invariant Gradient Alignment for Robust Reasoning Distillation 

**Authors**: Zehua Cheng, Wei Dai, Jiahao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2606.05025)  

**Abstract**: Large language models (LLMs) suffer from shortcut learning: they systematically fail on out-of-distribution (OOD) inputs whose semantic surface differs from training data, even when the logical structure is identical. This undermines knowledge distillation pipelines that transfer chain-of-thought reasoning to smaller students. We introduce Invariant Gradient Alignment (IGA), a training framework that aligns gradient updates across semantically diverse but logically isomorphic examples via three innovations: (i) Logical Isomer Sets, groups of problems sharing identical logical structure across distinct semantic domains (mathematics, medicine, law, science); (ii) a differentiable \emph{Continuous Gradient Conflict Mask}, that suppresses parameter dimensions with high cross-domain gradient variance while preserving invariant directions; and (iii) a truncated SVD projection of the masked gradient back onto the LoRA low-rank manifold, maintaining parameter efficiency throughout. Theoretically, IGA yields tighter OOD generalization bounds than ERM, scaling with the number of isomer domains, and converges at the standard SGD rate under mild regularity. Empirically, IGA outperforms eight baselines across four benchmarks with accuracy gains up to 14.3 pp over ERM-SFT and a Logical Consistency Score of 0.031 versus 0.142 -- a fourfold improvement in representational invariance. 

---
# DAR: Deontic Reasoning with Agentic Harnesses 

**Authors**: Guangyao Dou, William Jurayj, Nils Holzenberger, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2606.05009)  

**Abstract**: Deontic reasoning is the task of answering questions by applying explicit rules and policies to case-specific facts, for example computing tax liability under a statute or determining the outcome of an immigration appeal. A key technical challenge for LLM-based deontic reasoning is that the relevant ruleset can be long and cross-referenced, so models may still fail to locate the rules needed for a particular reasoning step. We introduce Deontic Agentic Reasoning (DAR), an agentic reasoning setup in which the model interacts with the statutes on demand. We evaluate DAR under multiple harnesses on hard subsets of DeonticBench. Across these settings, we find that agentic harnesses can push the frontier on deontic reasoning tasks, but improvements are not uniform: weaker models often degrade on numerical tasks while consuming far more tokens. 

---
# M$^3$Eval: Multi-Modal Memory Evaluation through Cognitively-Grounded Video Tasks 

**Authors**: Jie Huang, Ruixun Liu, Sirui Sun, Xinyi Yang, Yin Li, Yixin Zhu, Yiwu Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2606.05008)  

**Abstract**: As multi-modal models advance towards long-form video understanding, memory emerges as a critical capability. Despite substantial efforts in developing video datasets and benchmarks, existing works primarily focus on perception and reasoning, without systematically evaluating memory: what models retain, how faithfully information is preserved, and how robust memory remains under interference. To address this gap, we introduce M$^3$Eval, the first comprehensive evaluation framework and benchmark for probing different memory dimensions in multi-modal models. Grounded in cognitive psychology, our design features carefully constructed tasks that isolate key aspects of memory. Leveraging M$^3$Eval, we conduct extensive experiments across representative multi-modal models, revealing consistent weaknesses and distinctive behaviors. We find that models struggle to maintain disentangled representations when processing parallel video streams, exhibit interference patterns differing substantially from those observed in human memory, ground memory sources more reliably in the spatial domain than the temporal domain, and demonstrate limited symbolic memory. Collectively, our benchmark provides a valuable resource for future research, while our findings highlight memory as a fundamental yet underexplored capability and offer insights for designing more effective memory mechanisms in multi-modal models. Our code and dataset are available at this https URL. 

---
# SharedRequest: Privacy-Preserving Model-Agnostic Inference for Large Language Models 

**Authors**: Peihua Mai, Xuanrong Gao, Youlong Ding, Xianglong Du, Wei Liu, Yan Pang  

**Link**: [PDF](https://arxiv.org/pdf/2606.05004)  

**Abstract**: With the widespread deployment of public large language models (LLMs) such as ChatGPT, protecting user prompt privacy has become an increasingly critical issue. Existing privacy-preserving inference methods sacrifice either utility or efficiency, and often require model-specific modifications that limit their compatibility. In this paper, we propose SharedRequest, a model-agnostic framework for privacy-preserving LLM inference that reformulates privacy protection at the batch level rather than the individual-prompt level. The key idea is to obscure sensitive information by mixing original prompts with noisy variants, while grouping semantically equivalent instructions to amortize the inference cost over a large batch of queries with minimal impact on LLM response quality. This design is independent of the LLM architecture, requiring no access to model parameters or architectural modification. Empirical results demonstrate that SharedRequest achieves over $20\%$ higher utility compared to prior differential privacy baselines, and its shared-prompt mechanism reduces query cost by up to $5\times$ compared to non-batched inference. 

---
# From Agent Traces to Trust: Evidence Tracing and Execution Provenance in LLM Agents 

**Authors**: Yiqi Wang, Jiaqi Zhang, Taotao Cai, Zirui Liu, Qingqiang Sun, Zequn Sun, Zhangkai Wu, Mingkai Zhang, Yanming Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.04990)  

**Abstract**: Large language model (LLM)-based agents increasingly solve complex tasks by interacting with external tools, retrieval systems, memory modules, environments, and other agents. These capabilities expand agent autonomy, but also make agent behavior harder to verify, debug, and audit. Final-answer accuracy alone cannot explain how an output was produced, which evidence supported each claim, whether tool calls were justified, how memory influenced later decisions, or where execution failures originated. Evidence tracing and execution provenance address this gap by modeling how retrieved evidence, tool outputs, memory items, environment observations, intermediate claims, actions, and final answers are connected throughout agent execution. This survey provides a systematic review and conceptual framework for evidence tracing and execution provenance in LLM agents. We organize related work around a unified provenance perspective that connects retrieval grounding, claim support, tool-use safety, memory lineage, observability, debugging, audit, and recovery. We introduce a taxonomy covering trace sources, evidence and execution units, provenance relations, tracing granularity and timing, representation forms, and trust functions. We review key methodological directions, including provenance representation, evidence attribution, tool-use provenance, runtime guardrails, provenance-bearing memory, trace-based observability, and failure diagnosis. We also map existing benchmarks, datasets, and evaluation metrics to provenance-related capabilities, and discuss how evaluation can move from final-answer correctness toward process-level accountability. Finally, we outline open challenges, including unified trace schemas, claim-level and semantic provenance, provenance-aware safety mechanisms, realistic execution-trace benchmarks, recovery-oriented evaluation, and privacy-aware audit infrastructure. 

---
# DeliChess: A Multi-party Dialogue Dataset for Deliberation in Chess Puzzle Solving 

**Authors**: Xiaochen Zhu, Georgi Karadzhov, Tom Stafford, Andreas Vlachos  

**Link**: [PDF](https://arxiv.org/pdf/2606.04987)  

**Abstract**: Multi-party dialogue is a critical setting for studying collaborative reasoning and decision-making, yet existing datasets rarely focus on structured, in-depth complex reasoning tasks. We introduce DeliChess, a novel dataset of group deliberation dialogues in which participants collaboratively solve multiple-choice chess puzzles. Each group first completes the puzzle individually, then engages in a multi-party discussion before submitting a revised collective answer. The dataset includes 107 dialogues with full transcripts, pre- and post-discussion choices, and metadata on puzzle difficulty and move quality. We evaluate performance using three metrics based on chess engine evaluations, and find that deliberation significantly improves group accuracy. We further analyse the role of probing utterances (i.e., messages that elicit proposals, justifications, or strategic reflection) using a classifier trained on prior deliberation data. While probing makes group performance more variable after discussion, it does not consistently lead to better performance. Our dataset offers a rich testbed for modelling group reasoning, dialogue dynamics, and the resolution of differing perspectives and opinions in a well-defined strategic domain. 

---
# Plan, Watch, Recover: A Benchmark and Architectures for Proactive Procedural Assistance 

**Authors**: Kaustav Kundu, Ritvik Shrivastava, Maxim Arap, Nanshu Wang, Xianhui Zhu, Quintin Fettes, Gautam Tiwari, Parth Suresh, Théo Moutakanni, Alejandro Castillejo Munoz, Allen Bolourchi, Pascale Fung, Pinar Donmez, Babak Damavandi, Anuj Kumar, Seungwhan Moon  

**Link**: [PDF](https://arxiv.org/pdf/2606.04970)  

**Abstract**: We envision a proactive multi-modal assistant system which gives users real-time step-by-step guidance on a procedural task, autonomously deciding \textit{when} to interrupt, and \textit{how} to coach. However, progress is limited by the absence of large-scale, cross-domain benchmarks that reflect realistic conditions, particularly the common case in which users deviate from the expected step sequence. We address this gap with four contributions: \textbf{(1)}~we release \textbf{EgoProactive}, a large-scale wearable-egocentric dataset for proactive procedural assistance with explicit Out-of-Plan (OOP) annotations and recovery steps; \textbf{(2)}~we augment five established benchmarks (Ego4D, EPIC-KITCHENS, EgoExo4D, HoloAssist, HowTo100M) into \textbf{Pro\textsuperscript{2}Bench} under a unified proactive-guidance schema; \textbf{(3)}~we propose a \textbf{decoupled planner--interaction architecture} specialized for procedural state, visual cues, and recovery injection; \textbf{(4)}~we introduce a post-training recipe that transfers across model families, validated by cross-backbone replication on Llama~4 and Qwen-3.6-VL. In extensive experiments, our trained Llama-4 system substantially improves objective intervention quality over strong proprietary baselines (Claude Opus~4.6, Gemini~3.1~Pro, GPT~5.2) and open-weight baselines (Qwen3~VL~235B) baselines across all six datasets. Oracle-plan experiments further show that, when plan quality is controlled, the trained duplex model produces high-quality guidance and large gains on Out-of-Plan recovery. 

---
# From Prompt to Process: a Process Taxonomy and Comparative Assessment of Frameworks Supporting AI Software Development Agents 

**Authors**: Sanderson Oliveira de Macedo  

**Link**: [PDF](https://arxiv.org/pdf/2606.04967)  

**Abstract**: AI tools for programming are no longer just autocomplete or chat assistants: they organize themselves as development frameworks, with process, roles, artifacts and verification. Recent surveys map agents and LLMs for software engineering, but a study centered on the operational frameworks that turn these capabilities into process is missing. We ran a directed search of primary sources, with a functional inclusion criterion and traction measurement, and selected six frameworks: GitHub Spec Kit, OpenSpec, BMAD Method, Get Shit Done (GSD), Spec Kitty and Reversa. Each attacks AI development through a different path: spec-driven development in full and lightweight variants, agent-driven agile planning, context engineering over the agent, worktree isolation and review, and recovery of operational specifications from legacy systems. Our central contribution is a six-dimension process taxonomy: specification, context, roles, execution, validation and portability, with a scoring rubric that turns it into a replicable instrument. We apply it to the six frameworks and an out-of-sample case, Spec-Flow. Two results stand out. Among frameworks that already adopt some process there is convergence: the isolated prompt loses centrality, and persistent artifacts, work contracts, traceability and human review become mechanisms that reduce ambiguity and coordinate agents. And no framework strongly covers all six dimensions, exposing a structural trade-off between process depth and portability across agents. We also found recurring risks: drift between specification and code, excessive trust in generated artifacts, fragility of community extensions, platform dependence and a lack of benchmarks for the complete process. We close with a research agenda for empirical evaluation, focused on intermediate-quality metrics, context governance, installation security and reproducibility. 

---
# AdaKoop: Efficient Modeling of Nonlinear Dynamics from Nonstationary Data Streams with Koopman Operator Regression 

**Authors**: Naoki Chihara, Ren Fujiwara, Yasuko Matsubara, Yasushi Sakurai  

**Link**: [PDF](https://arxiv.org/pdf/2606.04930)  

**Abstract**: Real-time data analysis requires the ability to accurately and adaptively address nonlinear dynamics in a nonstationary data stream while preserving computational efficiency. However, nonlinear dynamics are so complex that capturing dynamically changing nonlinear patterns and utilizing them for downstream tasks under strict time constraints is nontrivial. To bridge the gap between nonlinear complexity and computational tractability, this study applies Koopman operator theory, which states that nonlinear dynamics can be represented as linear transitions in an infinite-dimensional space. Building upon finite-dimensional approximations of this operator, we present AdaKoop, an efficient streaming algorithm for modeling nonlinear dynamics over nonstationary data streams. Our approach utilizes a probabilistic framework grounded in Koopman operator theory, treating both raw observations and reproducing kernel Hilbert space (RKHS) features as emissions from latent vectors. This dual-view formulation allows nonlinear dynamics to be expressed as a tractable linear system. Therefore, AdaKoop enables the efficient and stable modeling of nonlinear dynamics in a streaming fashion, avoiding the prohibitive computational costs of iterative nonlinear optimization. Furthermore, to address nonstationarity in data streams, AdaKoop adaptively detects the switching of patterns via statistical hypothesis testing for abrupt pattern shifts and incrementally updates model parameters to handle continuous changes. Extensive experiments on a total of 71 practical benchmark datasets across various domains demonstrate that AdaKoop outperforms state-of-the-art methods in terms of real-time forecasting accuracy and computational efficiency. 

---
# Reproducing, Analyzing, and Detecting Reward Hacking in Rubric-Based Reinforcement Learning 

**Authors**: Xuekang Wang, Zhuoyuan Hao, Shuo Hou, Hao Peng, Juanzi Li, Xiaozhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04923)  

**Abstract**: Rubric-based reinforcement learning (RL) uses an LLM-as-a-Judge (LaaJ) to score model outputs according to rubrics as rewards. However, policy models may exploit latent biases in the judge, leading to reward hacking and ineffective or unsafe training outcomes. In real-world rubric-based RL, such hacking behaviors are often subtle and entangled with multiple judge biases, making them difficult to analyze, detect, and mitigate. In this paper, we introduce CHERRL, a controllable hacking environment for rubric-based RL. By injecting known biases into LaaJ, CHERRL enables stable reproduction of reward hacking, explicit observation of reward divergence, and precise identification of hacking onset. This provides a clean experimental testbed for studying the mechanisms and mitigations of reward hacking in rubric-based RL. To demonstrate its utility, we analyze different judge biases from the perspectives of discoverability and exploitability, and explore an agent-based system for automatically detecting reward hacking onset from training logs. The code and environment are publicly available at this https URL. 

---
# Geometry-Aware Distillation for Prompt Tuning Biomedical Vision-Language Models 

**Authors**: Tran Dinh Tien, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2606.04922)  

**Abstract**: Current prompt-based and adapter-based tuning of vision-language models (VLMs) is attractive for medical imaging, where clinical data sensitivity favors frozen backbones and annotations are limited. However, these methods typically optimize only the ground-truth class, treating all other classes as equally incorrect, ignoring clinically meaningful class relations and yielding unstable decision boundaries in limited-supervision settings. We propose Omni-Geometry Knowledge Distillation (OGKD), a new framework that injects class-relation structure into the teacher to produce directional targets that preserve the ground truth while respecting inter-class geometry. Using these targets, we develop two distillation losses: Global Geometry-Aware Distillation (GAD) operates on the global image token, and Label-Guided Geometry Distillation (LGD) applies the same geometry to attentive patch tokens to improve fine-grained alignment. Across comprehensive experiments and analyses on 11 widely-used medical datasets for base-to-novel and few-shot evaluations, our OGKD achieves substantially better performance, consistently improving accuracy by an average absolute gain of 1.7%-2.8% over all prior state-of-the-art VLM adaptation counterparts. It also robustly generalizes to unseen classes and yields more reliable predictions than other approaches. Our code is available at this https URL. 

---
# 'Your AI Text is not Mine': Redefining and Evaluating AI-generated Text Detection under Realistic Assumptions 

**Authors**: Nils Dycke, Marina Sakharova, Nico Daheim, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2606.04906)  

**Abstract**: Although it is generally agreed that AI-generated text poses a broad societal risk, there is no common understanding in the AI-generated text detection literature on what constitutes harmful use. Rather, existing datasets and approaches often define their own criteria and make their own assumptions, sometimes implicitly, and often only loosely related to real-world needs and applications. To address this gap, we here systematically define various notions of AI-generated text and their characteristics. To study these, we collect AITDNA - a new benchmark of human-machine co-constructed texts that is annotated with detailed genesis information, such as the entire edit and AI-interaction history. We benchmark various machine-generated text detectors and find that they often only perform well for specific notions but not as broad detectors. We release code and data publicly. 

---
# Provably Auditable and Safe LLM Agents from Human-Authored Ontologies 

**Authors**: Aaron Sterling  

**Link**: [PDF](https://arxiv.org/pdf/2606.04903)  

**Abstract**: We introduce the LLM agent architecture Agentic Redux, intended for use with nontrivial problem domains that require linear auditability. Using the typed lambda calculus, we prove that, run on appropriate domains, Agentic Redux executions are semantically guaranteed to be correct, with all decisions recorded in an append-only ledger. We present two production-grade appropriate domains, in healthcare billing compliance, and security vulnerability disclosure. Working code for Agentic Redux run on both domains is available in a supporting code repository. We also introduce Ontology-First Agent Design, a methodology for creation of agent frameworks on a problem domain, in which a human expert ontologizes the problem domain with Basic Formal Ontology, and then assigns an LLM to derive roles that agents and humans-in-the-loop can fill, in order to work the problems in the domain. 

---
# DiverAge: Reliable Pluralistic Face Aging with Cross-Age Identity Relation Guidance 

**Authors**: Yueying Zou, Peipei Li, Qianrui Teng, Dianyan Xu, Zekun Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.04881)  

**Abstract**: Face aging plays an important role in long-term biometric analysis, cross-age identity verification, and forensic identity analysis. Since the same subject may exhibit multiple plausible appearances at a target age due to genetic, environmental, and lifestyle factors, face aging is inherently a one-to-many generation problem. However, pluralism alone is insufficient for reliable face aging: a model should provide appearance-level candidate diversity within each age group while maintaining sequence-level ordinal reliability across ordered age groups. Existing deterministic aging methods can synthesize visually plausible age-progressed faces, but usually lack stochastic diversity. In contrast, pluralistic aging methods introduce local appearance variations, but often fail to explicitly regulate the identity evolution of the full aging sequence. In this paper, we propose \textbf{DiverAge}, a hierarchical pluralistic face aging framework based on diffusion autoencoding. DiverAge preserves appearance-level diversity through stochastic diffusion decoding and age-conditioned semantic modulation. To improve sequence-level reliability, we introduce a Cross-age Identity Relation Regulator (CARR), an inference-time guidance strategy that jointly denoises multiple target age groups. CARR is guided by a Cross-age Identity Similarity (CIS) prior estimated from real same-identity cross-age pairs, and suppresses excessive cross-age identity drift through one-sided sampling-time guidance without modifying the training objective or introducing extra trainable parameters. Experiments demonstrate that DiverAge improves sequence-level ordinal reliability while maintaining identity preservation, age accuracy, image quality, and appearance-level diversity. 

---
# Abduction Prover in Isabelle/HOL 

**Authors**: Yutaka Nagashima, Daniel Sebastian Goc  

**Link**: [PDF](https://arxiv.org/pdf/2606.04877)  

**Abstract**: Proof assistants based on expressive logics suffer limited automation for proof search, raising the cost of formal verification based on proof assistants. We address this problem by introducing the Abduction Prover for Isabelle/HOL. Given a challenging proof goal, the Abduction Prover constructs a proof script for the goal by identifying useful conjectures using abductive reasoning. 

---
# Learning Empirically Admissible Neural Heuristics for Combinatorial Search 

**Authors**: Siddharth Sahay  

**Link**: [PDF](https://arxiv.org/pdf/2606.04860)  

**Abstract**: Finding optimal solution paths for combinatorial puzzles like the Rubik's Cube, sliding tile puzzles, and Lights Out remains a classical challenge in artificial intelligence. Heuristic search algorithms, such as A* , guarantee path optimality only when using an admissible heuristic-one that never overestimates the true remaining cost-to-go. Deep reinforcement learning (RL) methods like DeepCubeA train deep neural networks to approximate cost-to-go heuristics. However, standard mean-squared error (MSE) training regularly yields overestimations, violating admissibility and compromising solution optimality. In this paper, we introduce a generalizable framework for learning validation-calibrated admissible neural heuristics. We train a value network using an underestimating Admissible Bellman Operator combined with an Asymmetric Loss function to penalize overestimation. To account for residual neural function approximation errors, we propose a post-hoc calibration safety offset computed over validation scrambles. We demonstrate that our calibrated neural heuristics achieve no observed admissibility violations under the evaluation protocol and preserve path optimality in practice while reducing search node expansions by up to 83.0% on a 2 by 2 Rubik's Cube, 19.9% on a 3 by 3 Lights Out grid, and 1.9% on an 8-Puzzle compared to standard analytical baselines. 

---
# Uncertainty-Aware End-to-End Co-Design of Neural Network Processors: From Training and Mapping to Fabrication 

**Authors**: Yuyang Du, Yujun Huang, Gioele Zardini  

**Link**: [PDF](https://arxiv.org/pdf/2606.04850)  

**Abstract**: Designing a neural network processor is an end-to-end co-design problem: network architecture and training budget determine the inference workload; hardware mapping decisions determine chip area, latency, and energy; and these characteristics govern fabrication yield and manufacturing cost. In practice, these decisions are made in separate stages, and existing co-design methodologies are tightly coupled to specific algorithms, making it difficult to improve one component without reworking the entire pipeline. This paper presents a unified framework, grounded in monotone co-design theory, that composes four interoperable design blocks spanning network training, chip mapping, wafer-level fabrication, and compute resource allocation. Each block exposes only a functionality-resource interface to the rest of the system, so any block can be refined without structural changes elsewhere. A central contribution is the treatment of uncertainty: rather than collapsing stochastic outcomes into point estimates, the framework introduces Confidence, the inverse of success probability, as an explicit and optimizable resource alongside cost, time, and power. Three case studies validate the approach. The first recovers Pareto-optimal implementations across heterogeneous application scenarios. The second confirms that Confidence functions as a continuously tunable design knob rather than a post-hoc diagnostic. The third demonstrates that improving a single block's implementation set automatically propagates to the global Pareto front, without modifying the co-design diagram. 

---
# Signed Dual Attention: Capturing Signed Dependencies in Time Series Forecasting 

**Authors**: Balthazar Courvoisier, Tristan Cazenave  

**Link**: [PDF](https://arxiv.org/pdf/2606.04833)  

**Abstract**: Initially developed for natural language processing, Transformer architectures and attention mechanisms are now central to a wide range of deep learning models, including applications in time series forecasting. A standard attention mechanism, however, implicitly assumes homophilic interactions, limiting its ability to model data with positive and negative dependencies, such as time series. In this work, we introduce the Signed Dual Attention, a novel attention formulation that captures both positive and negative relational patterns without additional parameters. By leveraging a dual message-passing scheme inspired by correlation structures, Signed Dual Attention propagates both supportive and contrastive information within a single shared block, effectively achieving the expressiveness of two head attention without additional parameters. This module can be seamlessly integrated into existing architectures and can yield performance gains in certain situations, requiring signed relational modeling. This approach opens a pathway toward more expressive and parameter-efficient transformers. 

---
# OA-CutMix: Correcting the Label Bias of CutMix 

**Authors**: Tobias Christian Nauen, Stanislav Frolov, Federico Raue, Brian B. Moser, Andreas Dengel  

**Link**: [PDF](https://arxiv.org/pdf/2606.04820)  

**Abstract**: CutMix has become the de facto standard mixing augmentation, yet its label assignment rests on a flawed assumption: The area of the pasted patch faithfully reflects its semantic contribution to the mixed image. In practice, however, patches frequently land on background regions, assigning label credit to classes whose objects are not visible. The mean discrepancy of the CutMix label and the semantic object area is $21.5\%$. In $17\%$ of samples an image contributes zero visible object pixels yet receives nonzero label weight. We propose Object-Aware CutMix (OA-CutMix), which corrects this bias by replacing the area-based CutMix weight with one derived from precomputed segmentation masks, assigning labels in proportion to the visible object area each image contributes to the mix. The image mixing procedure is left entirely unchanged. We evaluate OA-CutMix against 10+ static and dynamic mixing methods across 4 architectures and 6 datasets. OA-CutMix consistently achieves the highest accuracy over all tasks, outperforming even dynamic mixing methods, but at a fraction of the training-time cost. Improvements are largest for small objects, where the label bias from CutMix is greatest. Thus, correcting the label is sufficient to match or exceed the performance of methods modifying the image mixing algorithm. 

---
# Learning While Acting: A Skill-Enhanced Test-Time Co-Evolution Framework for Online Lifelong Learning Agents 

**Authors**: Bo Mao, Jie Zhou, Yutao Yang, Xin Li, Xian Wei, Qin Chen, Xingjiao Wu, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2606.04815)  

**Abstract**: Lifelong learning is essential for Large Language Model (LLM) agents operating in dynamic, interactive environments. However, existing lifelong learning agents for long-horizon tasks typically depend on discrete skill or past experiences retrieval with static parameters during inference, which prevents them from continuously internalizing test-time feedback like human learners. To bridge this gap, we propose Skill-enhanced Test-Time Co-Evolution (\texttt{LifeSkill}), a two-stage reinforcement learning framework for Online Lifelong Learning Agents. Specifically, we design Verifier-Guided Skill Learning that addresses the lack of direct supervision for skill extraction by rewarding candidate skills according to the average verifier success of multiple skill-conditioned policy rollouts, encouraging the model to generate skills that are useful for solving tasks rather than merely plausible in text. Furthermore, we introduce Online Skill Internalization, which continuously improves the policy model during test-time interaction by transforming skill-conditioned trajectories into reward signals. This enables the agent to directly internalize reasoning capabilities into its parameters, avoiding the context bloat of experience retrieval. Experiments on LifelongAgentBench show that LifeSkill improves average performance by 7 absolute points by comparing with existing lifelong agent baselines. 

---
# Scenario Generation for Risk-Aware Reinforcement Learning with Probably Approximately Safe Guarantees 

**Authors**: Mohit Prashant, Arvind Easwaran  

**Link**: [PDF](https://arxiv.org/pdf/2606.04812)  

**Abstract**: Guaranteeing safety is critical to the deployment of reinforcement learning (RL) agents in the real-world, especially as policies learned using deep RL may demonstrate susceptibility to transition perturbations that result in unknown or unsafe behaviour. A method of policy verification is to construct probabilistic barrier-certificates by sampling policy trajectories with respect to safety constraints, thereby demarcating known safe behaviour from unknown behaviour. Obtaining tight upper and lower bounds on the probability of violation of these constraints may be difficult if the policy is susceptible to transition uncertainty or perturbation that places the agent in insufficiently explored states. To address this, we approximate the distribution of the encountered state-space using a variational autoencoder (VAE) and construct upper and lower-bound barrier-certificates using latent characteristics of states to optimize for regions of known, safe behaviour with high confidence. We frame this in our work as a dual optimization problem where the lower-bound barrier-certificate presents a more conservative estimate of the safe region than the upper-bound barrier-certificate. Sampling states that lie within the set difference of the two during training, i.e. the non-robust region, allows us to tighten the upper and lower bounds to provide sharper probabilistic guarantees on safety. Within our study, we describe the guarantees placed and demonstrate the tightness of our bounds experimentally. 

---
# NoRA: Evaluating Grounded Reasonableness in Visual First-person Normative Action Reasoning 

**Authors**: Sichao Li, Sai Ma, Daniel Kilov, Secil Yanik Guyot, Zhuang Li, Seth Lazar  

**Link**: [PDF](https://arxiv.org/pdf/2606.04806)  

**Abstract**: LLMs and agentic systems are increasingly deployed in social environments, making normative competence critical for safe and appropriate behavior. However, existing approaches either assess normative judgment in text alone or reduce it to choosing among a fixed set of candidate actions. We argue both are insufficient. In practice, agents are never handed a menu of options; they must identify a reasonable action from scratch, grounded in visible facts and supported by inspectable reasons. We introduce NoRA, a visual first-person video benchmark that requires models to generate candidate next actions and justify each through an explicit fact-reason-action support graph. The benchmark comprises 1,420 annotated video clips, including HumanGold-190 and LLMSilver-1230 splits. Each instance is evaluated through action alignment, factual grounding, and support binding, aggregated into a single grounded reasonableness score. We benchmark 12 multimodal systems under direct, deliberate, and structured prompting regimes, finding that current VLMs frequently recover plausible actions and relevant scene facts, but consistently struggle to construct the full reasonable action space and bind selected actions to the correct local support. NoRA makes this gap measurable, shifting the evaluation question from whether a model can pick an action to whether it can justify an appropriate action for the right visible reasons. 

---
# Activation Steering of Video Generation Models via Reduced-Order Linear Optimal Control 

**Authors**: Jihoon Hong, Alice Chan, Qiyue Dai, Julian Skifstad, Glen Chou  

**Link**: [PDF](https://arxiv.org/pdf/2606.04775)  

**Abstract**: Text-to-video (T2V) models trained on large-scale web data can generate undesired content, motivating interventions that reduce harmful outputs without sacrificing visual quality. Activation steering offers an attractive mechanistic alternative to finetuning and prompt filtering, but existing T2V steering methods remain limited, typically applying coarse, non-anticipative interventions that can lead to oversteering and content degradation. To close this gap, we propose Latent Activation Linear-Quadratic Regulator (LA-LQR), a reduced-order optimal control framework for minimally invasive T2V steering. LA-LQR formulates T2V inference as a dynamical system and computes closed-loop feedback interventions that steer activations toward desired feature setpoints while penalizing unnecessary perturbations. To make optimal control feasible for high-dimensional video activations, we project activations onto a low-dimensional, task-relevant subspace derived from contrastive prompt pairs, estimate local linear dynamics in this latent space, and solve a latent LQR problem to obtain timestep- and layer-specific steering signals. We provide theoretical bounds relating latent setpoint tracking to raw activation-space feature control, and empirically validate the fidelity of the reduced latent dynamics. On concept steering and video safety benchmarks, LA-LQR reduces unsafe generations relative to baselines, while preserving prompt fidelity and visual quality. 

---
# Coarse-to-fine Hierarchical Architecture with Sequential Mamba for Brain Reconstruction 

**Authors**: Hoang-Son Vo, Van-Hung Bui, Minh-Huy Mai-Duc, Tien-Dung Mai, Soo-Hyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.04772)  

**Abstract**: Understanding the relationship between deep visual representations and the human visual system is a fundamental challenge in computational neuroscience. While modern vision models achieve strong performance in image recognition, their correspondence with the hierarchical organization of the human visual cortex remains an open question. In this study, we propose CHASMBrain, a novel hierarchical two-stage framework for image-to-fMRI encoding. Our architecture leverages a dual-stream Mamba design to explicitly separate and process global semantic tokens and local spatial patches, motivated by the functional organization of the visual cortex. A coarse-to-fine strategy is employed: Stage 1 predicts denoised ROI-level activations, while Stage 2 refines these coarse responses into full voxel-level predictions using a Mamba-VAE. Experiments on the Natural Scenes Dataset (NSD) demonstrate that our method achieves a Pearson correlation of 0.429 and an MSE of 0.261, outperforming all evaluated baselines including ridge regression and DINOv2 linear probes. Beyond predictive performance, causal branch-ablation experiments reveal an asymmetric specialization: the patch stream is specifically locked to early visual cortex (retinotopic regions), while the CLS stream contributes broader semantic context to higher-order areas -- a correspondence that holds causally, not merely correlationally. Cross-subject transfer experiments further show that the learned backbone generalizes across individuals with minimal per-subject adaptation, suggesting the model captures a shared, subject-agnostic visual representation. 

---
# Description-Code Inconsistency in Real-world MCP Servers: Measurement, Detection, and Security Implications 

**Authors**: Yutao Shi, Xiaohan Zhang, Xiangjing Zhang, Xihua Shen, Hui Ouyang, Huming Qiu, Mi Zhang, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04769)  

**Abstract**: The Model Context Protocol (MCP) has emerged as a critical standard empowering Large Language Models (LLMs) to utilize external tools. In this ecosystem, LLMs rely on natural language descriptions provided by MCP servers to select and execute functions. This interaction implicitly assumes that tool descriptions faithfully reflect their underlying implementations, while this assumption is not mandatorily verified in practice. As a result, MCP deployments may suffer from a problem named Description-Code Inconsistency (DCI), where a tool's description of its capabilities and security boundaries is not consistent with what the code actually does.
In this paper, we present a comprehensive study of DCI in real-world MCP servers. We formally define the problem and propose a comprehensive taxonomy spanning functionality inconsistencies and undeclared side effects. Guided by this taxonomy, we develop DCIChecker, an automated framework that combines structure-aware static analysis with the Direct-Reverse-Arbitration prompting method to cross-validate tool descriptions against actual code implementations. We apply this framework to a large-scale dataset comprising 19,200 description-code pairs extracted from 2,214 real-world MCP servers. Our measurement reveals that DCI is widespread, with 9.93% of these pairs exhibiting inconsistencies. We further demonstrate that DCI creates a critical defense blind spot, facilitating varied risks from operational failures to stealthy malicious behaviors. Finally, we propose mitigation strategies to enforce semantic consistency and enhance the reliability of the emerging agentic ecosystem. 

---
# Archi: Agentic Operations at the CMS Experiment 

**Authors**: Pietro Lugato, Luca Lavezzo, Jason Mohoney, Hasan Ozturk, Muhammad Hassan Ahmed, Juan Pablo Salas, Viphava Ohm, Krittin Phornsiricharoenphant, Gabriele Benelli, Mariarosaria D'Alfonso, Manasvita Joshi, Warren Nam, Aron Soha, Samantha Sunnarborg, Austin Swinney, Jack Tucker, Dmytro Kovalskyi, Tim Kraska, Christoph Paus  

**Link**: [PDF](https://arxiv.org/pdf/2606.04755)  

**Abstract**: We present Archi, an open-source, end-to-end framework for scientific collaborations that combines the systematic ingestion and organization of heterogeneous data sources with the deployment of configurable, private, and extensible agents that retrieve and reason over them. An instance of Archi has been deployed for the Computing Operations team of the CMS experiment at CERN's LHC since February 2026 as a support agent for technical operators, offering retrieval and analysis capabilities by combining documentation, historical data, and live monitoring systems. We evaluate the system on operator feedback and a question set collected from production usage, graded by human and automated panels. The system proves effective at operational tasks, resolving real-world queries posed by CMS operators. We also observe that locally-hosted, open-weight models perform competitively, enabling fully private management of sensitive data. 

---
# An Empirical Audit of Input Encoders for Multi-Channel Signal Transformers 

**Authors**: Ossi Lehtinen  

**Link**: [PDF](https://arxiv.org/pdf/2606.04752)  

**Abstract**: Transformers consuming multi-channel scalar signals must embed $C$ simultaneous values into one $d_{\text{model}}$-dimensional vector per time step. We empirically audit eight input encoders -- spanning a shared-scalar baseline, per-channel linear projections, an orthogonality regulariser, a nonlinear MLP stem, block-partitioned concatenation, channel-independent and channel-as-token architectures, and a projected positional encoding -- on a synthetic benchmark designed to make channel identity informative and on ETTh1 as a real-data check, measured in next-step negative log-likelihood (NLL). The headline is one of practical near-equivalence within a wide "top tier": the standard per-channel linear projection (this http URL(C, $d_{\text{model}}$)) matches every alternative in that tier up to small, statistically real but practically modest, differences. Two encoders lose decisively: the shared-scalar baseline, which collapses for information-theoretic reasons we make explicit, and the channel-independent PatchTST-spirit baseline, which underperforms on both benchmarks and overfits universally on the synthetic one. Paired tests resolve two small gaps: projecting the sinusoidal positional encoding through a learned linear layer edges the rest at small $C$, with a direct geometric probe showing the mechanism is positional-channel orthogonalisation; a nonlinear MLP stem edges them at the largest $C$ we test, with the gap shrinking under more training data. The practical recommendation is to use this http URL(C, $d_{\text{model}}$) by default and reach for something more elaborate only when the task at hand gives a real reason to do so. Code and data to reproduce every experiment in this paper are available at this https URL 

---
# TIDE: Proactive Multi-Problem Discovery via Template-Guided Iteration 

**Authors**: Soyeong Jeong, Jinheon Baek, Minki Kang, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04743)  

**Abstract**: Agents are widely deployed as assistants over documents, tools, and code. However, they typically act only on explicit user requests, which surface only the problems the user has noticed, while many other important problems coexist, hidden in plain sight, within the broader user context, with their total number unknown in advance. We frame this as the task of discovering multiple hidden problems from context, in which coexisting problems should be uncovered, grounded in supporting evidence, and paired with concrete actions. To this end, we introduce TIDE, a template-guided iterative framework with two complementary mechanisms. Specifically, motivated by the observation that single-pass prediction anchors on the most salient cases and yields generic claims, we propose iterative discovery, which surfaces a small batch of candidates per round while conditioning on what has already been found, so subsequent rounds extend coverage; and thought templates, reusable schemas distilled from previously solved cases that specify what contextual signals to attend to and how to connect them, anchoring each prediction in a recognizable problem class. We validate TIDE on two realistic settings, personal workspaces and software repositories, across four model backbones, showing substantial gains over single-shot and parallel multi-agent baselines on task coverage, identification, and resolution. 

---
# Revisiting Vul-RAG: Reproducibility and Replicability of RAG-based Vulnerability Detection with Open-Weight Models 

**Authors**: Sabrina Kaniewski, Fabian Schmidt, Tobias Heer  

**Link**: [PDF](https://arxiv.org/pdf/2606.04739)  

**Abstract**: Large language models (LLMs) have shown strong potential for automated software vulnerability detection, particularly in retrieval-augmented generation (RAG) settings. However, for approaches relying on proprietary models and APIs, reproducibility and replicability remain largely unexplored, raising the question of whether reported results generalize or depend primarily on specific model choices. In this work, we present a reproducibility study of Vul-RAG, a RAG-based framework for source code vulnerability detection that enhances LLMs with high-level vulnerability knowledge. We first replicate the results in a fully local and open-weights setting using the reported open-weight baseline models. We then extend the evaluation to a diverse set of recent open-weight LLMs, including code-specialized, general-purpose, and reasoning models of varying parameter sizes. The results confirm that the findings of Vul-RAG are reproducible under local deployment, but with minor deviations. Across all evaluated models, we observe a performance plateau at approximately 0.30 pairwise accuracy (code pairs for which both the vulnerable and the patched function are correctly classified). Notably, this plateau persists even for more recent and advanced models, indicating that improvements in model capacity alone do not substantially enhance performance. Finally, we discuss practical implications and trade-offs between detection effectiveness, model capabilities, and model scale. Implementation and evaluation artifacts are publicly available at this https URL. 

---
# Curvature-aware dynamic precision approach for physics-informed neural networks 

**Authors**: Yingjie Shao, Ioannis N. Athanasiadis, George van Voorn, Taniya Kapoor  

**Link**: [PDF](https://arxiv.org/pdf/2606.04736)  

**Abstract**: Physics-informed neural networks (PINNs) have become a promising framework for simulating partial differential equations (PDEs) by embedding physical laws directly into neural network training. However, recent studies show that PINN optimisation is sensitive to numerical precision. Existing implementations commonly use either single precision (FP32), which is computationally efficient but prone to failure modes, or double precision (FP64), which is robust but substantially expensive. This creates a trade-off between computational efficiency and numerical accuracy. To reduce the computational cost of double-precision training while retaining prediction accuracy, we propose a curvature-aware precision controller that adapts numerical precision during training rather than treating it as a fixed implementation choice. The proposed method reuses curvature information derived from the limited-memory BFGS (L-BFGS) optimiser to construct a precision controller, retaining FP32 when lower precision is sufficient and promoting computation to FP64 when the training dynamics indicate numerical sensitivity or precision-limited stagnation. We evaluate the proposed approach on four canonical PINN failure-mode benchmarks and an irradiance-driven ordinary differential equation example. We further test the proposed approach across different neural network architectures. The method consistently matches or even slightly exceeds full FP64 solution accuracy while reducing training time relative to full double-precision training on all benchmark equations. The obtained results indicate that precision sensitivity in PINN optimisation is phase-dependent, and that selectively applying higher precision only during numerically critical stages can lower computational cost without sacrificing predictive accuracy. 

---
# Trace-Mediated Peak Bias: Bridging Temporal Credit Assignment and Cognitive Heuristics in Deep Reinforcement Learning 

**Authors**: Viktor Veselý, Aleksandar Todorov, Erwan Escudie, Matthia Sabatelli  

**Link**: [PDF](https://arxiv.org/pdf/2606.04735)  

**Abstract**: Temporal credit assignment is central to both biological and artificial intelligence, yet its interaction with non-linear function approximation is poorly understood. We identify a systematic failure mode in deep reinforcement learning (RL) termed Trace-Mediated Peak Bias (TMPB). At intermediate eligibility trace depths, agents irrationally prefer trajectories with high-magnitude reward ``peaks'' over alternatives with higher cumulative returns. This provides a mechanistic account of the Peak-End Rule: a human memory bias where experiences are judged by their most intense moments rather than integrated utility. We show that TMPB emerges because traces amplify distal Temporal Difference errors into ``gradient shocks'' that fixed-step-size Stochastic Gradient Descent cannot normalize, leading to global overestimation. Conversely, adaptive optimizers mitigate this pathology via second-moment normalization. Our results suggest that human-like saliency distortions may emerge naturally from the mathematical constraints of credit assignment in distributed systems, and that adaptive optimization is a theoretical necessity for rational value estimation. 

---
# CoRe-MoE: Contrastive Reweighted Mixture of Experts for Multi-Terrain Humanoid Locomotion with Gait Adaptation 

**Authors**: Kailun Huang, Zikang Xie, Yanzhe Xie, Panpan Liao, Fanghai Zhang, Yanheng Mai, Wenhao Xu, Yunheng Wang, Renjing Xu, Haohui Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04718)  

**Abstract**: Humans primarily rely on walking and running to traverse complex terrains, without resorting to unnecessarily complex motion patterns. Similarly, humanoid robots should achieve smooth transitions between walking and running while maintaining natural and stable locomotion. However, unifying gait transition and multi-terrain adaptation within a single policy remains challenging due to gradient interference and the distribution shift induced by terrain-dependent visual and dynamic variations. Although Mixture-of-Experts (MoE) architectures can alleviate multi-skill interference, naive joint training often fails to yield clear expert specialization, limiting their effectiveness. To address these challenges, we propose CoRe-MoE, a two-stage reinforcement learning framework that decouples gait generation from terrain adaptation. In the first stage, a stable locomotion policy is learned to produce natural walking and running behaviors with smooth transitions. In the second stage, a terrain-aware MoE branch is introduced and trained with a contrastive objective to shape the gating network, enabling it to capture structured terrain representations and promote expert specialization. The final action is obtained via weighted fusion of the base gait policy and the terrain-aware branch, allowing the policy to preserve stable locomotion patterns while adapting to complex terrains. Extensive simulation results demonstrate that the proposed method outperforms baseline approaches in terms of success rate, locomotion stability, and multi-terrain adaptability. Furthermore, zero-shot deployment on a Unitree G1 humanoid robot validates the effectiveness of our framework, achieving robust walking and running across stairs, slopes, steps, obstacles, and unstructured outdoor terrains, while maintaining accurate foothold placement and dynamic stability under external disturbances. 

---
# VISTA: Vision-Grounded and Physics-Validated Adaptation of UMI data for VLA Training 

**Authors**: Siyuan Yang, Linzheng Guo, Ouyang Lu, Zhaxizhuoma, Daoran Zhang, Xinmiao Wang, Ting Xiao, Fangzheng Yan, Zhijun Chen, Yan Ding, Chao Yu, Chenjia Bai, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.04708)  

**Abstract**: Universal Manipulation Interface (UMI) enables scalable real-world robot data collection without hardware-specific teleoperation, yet leveraging UMI data to train large-scale Vision-Language-Action (VLA) models remains fundamentally challenging. We identify two critical mismatches: wrist-mounted fisheye views, with severe radial distortion and local gripper-centric perspectives, are out-of-distribution for pretrained VLMs; and human-collected trajectories frequently violate kinematic limits, incur collisions, or exceed controller bandwidth, teaching VLA policies physically infeasible actions. To address the challenges, we present VISTA, a framework that bridges this dual gap through three synergistic components. (i)~UMI-VQA, the first large-scale VQA dataset tailored to wrist-mounted fisheye observations, aligns VLM representations to the distorted visual regime via auxiliary vision-language supervision. (ii)~A systematic physical-validation pipeline performs a data-completeness pre-check and scores each valid trajectory for trajectory continuity, self-collision risk, and execution fidelity before it enters training. (iii)~A two-stage co-training recipe jointly learns vision-language grounding on UMI-VQA and action prediction on validated trajectories. Our experiments empirically show that incorporating UMI-VQA consistently improves downstream policy performance, and that physical-validation scores are strongly predictive of deployment success. On diverse simulation and real-world manipulation tasks, VISTA significantly outperforms strong baselines including $\pi_{0.5}$, LingBot-VLA, and Wall-X. We release the physical-validation pipeline, UMI-VQA, validated trajectory data, and the pre-trained model for the community. 

---
# Enhancing MedSAM with a Lightweight Box Predictor for Medical Image Segmentation 

**Authors**: Amirhossein Movahedisefat, Amirreza Fateh, Mohammad Reza Mohammadi  

**Link**: [PDF](https://arxiv.org/pdf/2606.04705)  

**Abstract**: Semantic segmentation in medical imaging is a critical yet challenging task due to data scarcity and high variability across modalities. While foundation models like the Segment Anything Model (SAM) show promise, they often struggle with medical images without specific adaptation. Moreover, point prompts, despite being the most natural form of user interaction, provide insufficient spatial context for reliable segmentation, particularly when target structures are irregular or poorly contrasted. In this paper, we propose an enhanced segmentation framework that integrates a lightweight Box Predictor module into the MedSAM architecture. The Box Predictor estimates an approximate bounding box from a single user click using localized image embedding features, providing spatial guidance that reduces the ambiguity of point prompts, while introducing only 1.6M additional parameters and negligible inference overhead. We introduce a two-stage training pipeline where the Box Predictor is trained independently before being integrated into MedSAM. To validate the generalization capability of our method, we conduct extensive evaluations on four diverse datasets (FLARE22, BRISC, BUSI, LungSegDB) spanning distinct imaging modalities, including CT, MRI, and Ultrasound. Our method improves segmentation accuracy and robustness across varied anatomical structures and imaging domains, achieving Dice scores of 0.89 (BUSI), 0.93 (FLARE22), 0.88 (BRISC), and 0.98 (LungSegDB). Code is available at this https URL 

---
# Graph-Guided Universum Learning in Generalized Eigenvalue Proximal SVMs for Alzheimer's Disease Classification 

**Authors**: Yogesh Kumar, Vrushank Ahire, Mudasir Ganaie  

**Link**: [PDF](https://arxiv.org/pdf/2606.04699)  

**Abstract**: Early and accurate detection of Alzheimer's disease (AD) is important for timely intervention and disease management. Generalized Eigenvalue Proximal Support Vector Machine (GEPSVM) and its Universum-based variants have shown promising results for AD classification. However, existing methods treat Universum samples as independent points and do not consider the geometric relationships among them. This paper proposes two graph-guided Universum learning models, namely UG-GEPSVM and IUG-GEPSVM, for AD versus cognitively normal (CN) classification using structural MRI data. In the proposed framework, mild cognitive impairment (MCI) subjects are used as Universum data to provide intermediate information between AD and CN classes. A graph is constructed over the Universum samples using Gaussian similarity, Minimum Spanning Tree connectivity, and multi-hop propagation. From this graph, a Laplacian matrix is derived that captures the geometric structure of the MCI samples. This Laplacian-based regularization is incorporated into the learning process in place of the conventional independent Universum penalty term. UG-GEPSVM integrates this regularization into the generalized eigenvalue formulation, while IUG-GEPSVM extends the numerically stable improved GEPSVM framework using a standard eigenvalue formulation. Experiments on ADNI MRI dataset variants using ICA- and PCA-based features at five different noise levels show that both proposed models consistently outperform existing GEPSVM and Universum-based methods. UG-GEPSVM achieves the highest average AUC of 88.07% and maintains stable performance under increasing noise levels. Statistical tests further confirm the significance of the observed improvements. 

---
# Real-Time Automatic License Plate Recognition Using YOLOv8, SORT Tracking, and Temporal Data Interpolation 

**Authors**: Mirza Muhammad Mobeen  

**Link**: [PDF](https://arxiv.org/pdf/2606.04684)  

**Abstract**: The real-time hardships of video processing seriously limit the usage of Automatic License Plate Recognition (ALPR) with application in dynamic traffic monitoring settings. High-fidelity recognition of unconstrained variables, e.g. drastic variations in illumination, acute camera scans, high vehicle speeds, and harsh physical concealment, is a problem that often leads to disjointed tracking paths and poor Optical Character Recognition (OCR) rates. In order to mitigate these weaknesses, the study proposes a 5 stage, end-to-end algorithmic pipeline, encompassing a smooth transition between deep learning based object detection, multi-object tracking which is kinematic in nature, and geometry temporal data interpolation. The suggested architecture takes advantage of a very powerful YOLOv8 nano model to localize the vehicle at the first stage and then Simple Online and Realtime Tracking (SORT) algorithm is used to build spatial-temporal links between frames. Another, more specific typology of YOLOv8 object detectors the license plate area, channeling the sliced array to an EasyOCR chain under the limitations of positional syntax verification. More importantly, an offline interpolation mechanism of temporal bounding box is initiated to recast fragmented paths. 

---
# Learning Long Range Spatio-Temporal Representations over Continuous Time Dynamic Graphs with State Space Models 

**Authors**: Ayushman Raghuvanshi, Thummaluru Siddartha Readdy, Sundeep Prabhakar Chepuri, Mahesh Chandran  

**Link**: [PDF](https://arxiv.org/pdf/2606.04672)  

**Abstract**: Continuous-time dynamic graphs (CTDGs) provide a richer framework to capture fine-grained temporal patterns in evolving relational data. Long-range information propagation is a key challenge while learning representations, wherein it is important to retain and update information over long temporal horizons. Existing approaches restrict models to capture one-hop or local temporal neighborhoods and fail to capture multi-hop or global structural patterns. To mitigate this, we derive a parameter-efficient state-space modeling framework for continuous-time dynamic graphs (CTDG-SSM) from first principles. We first introduce continuous-time Topology-Aware higher order polynomial projection operator (CTT-HiPPO), a novel memory-based reformulation of HiPPO to jointly encode temporal dynamics and graph structure. The solution from CTT-HiPPO is obtained by projecting the classical HiPPO solution through a polynomial of the Laplacian matrix, yielding topology-aware memory updates that admit an equivalent state-space formulation for CTDGs (CTDG-SSM). Then a computationally efficient discrete formulation is obtained using the zero-order hold approach for model implementation.
Across benchmarks on dynamic link prediction, dynamic node classification, and sequence classification, CTDG-SSM achieves state-of-the-art performance. Notably, it achieves large performance gains on datasets that require long range temporal (LRT) and spatial reasoning. 

---
# Why Muon Outperforms Adam: A Curvature Perspective 

**Authors**: Shuche Wang, Fengzhuo Zhang, Jiaxiang Li, Dirk Bergemann, Zhuoran Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04662)  

**Abstract**: Muon improves training efficiency over Adam in large language-model training by about two times, but the local geometric source of this advantage remains unclear. Our work takes a first step toward demystifying Muon's superiority over Adam from a curvature perspective. First, we apply a second-order Taylor approximation to the training landscape and show that Muon achieves a larger one-step loss decrease than Adam at matched validation loss. The two optimizers have comparable first-order gains, but Muon consistently incurs a smaller second-order curvature penalty. Second, we decompose this curvature penalty into the squared update norm and Normalized Directional Sharpness (NDS). We find that Muon and Adam have comparable update norms, so Muon's smaller curvature penalty is driven by lower NDS, not update scale. Third, we study how training data and model structure shape Muon's NDS advantage. Using Zipf-Probabilistic Context-Free Grammar (PCFG) data with controlled imbalance, we show that data imbalance amplifies Muon's NDS advantage over Adam. A within-/cross-layer decomposition further shows that, in the middle and late stages of training, Muon's lower NDS is mainly sustained by smaller within-layer curvature. Beyond empirical evidence, we analyze stylized quadratic problems with heterogeneous curvature and gradient alignment toward high-curvature modes. We prove that Muon attains a smaller average NDS than GD by balancing update energy across curvature groups; when curvature heterogeneity is sufficiently strong, this also yields lower local quadratic loss after the same number of steps. 

---
# Instance-Level Post Hoc Uncertainty Quantification in Object Detection 

**Authors**: Chongzhe Zhang, Zifan Zeng, Qunli Zhang, Feng Liu, Zheng Hu  

**Link**: [PDF](https://arxiv.org/pdf/2606.04656)  

**Abstract**: Object detection is a safety-critical component of autonomous driving. It is essential to quantify the uncertainty in bounding-box predictions for safety assurance. Post hoc uncertainty quantification without retraining aligns with real-world deployment requirements; therefore, we employ the Laplace approximation. Because instance-level uncertainty is needed, linearized inference methods that require multiple backpropagations are not time-efficient, and sampling-based methods are not fully post hoc. We propose Monte-Carlo generalized linearized model (MC-GLM), which provides instance-level and approximately post hoc uncertainty quantification. The number of samples required in the Monte Carlo step is constant and independent of the number of output instances, so it can be parallelized. Experiments on the nuScenes dataset with the CenterPoint detector validate the effectiveness of our method, and the resulting uncertainties exhibit good quality. 

---
# QO-Bench: Diagnosing Query-Operator-Preserving Retrieval over Typed Event Tuples 

**Authors**: Mengao Zhang, Xiang Yang, Chang Liu, Tianhui Tan, Ke-wei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04646)  

**Abstract**: Many real-world questions over business, legal, and scientific corpora are natural-language versions of database-style queries over records latent in text. Existing retrieval-augmented generation (RAG) systems are optimized primarily for semantic relevance, but retrieving plausible passages does not guarantee correct query execution. We introduce QO-Bench, a diagnostic benchmark for query-operator question answering over typed event tuples. The benchmark covers 22,984 news articles and 614 corporate events across 18 query templates, evaluated on 785 questions. Each gold answer is deterministically computed from typed event tuples and scored by recall, with answers matched to the gold tuples by exact match rather than an LLM judge. This design enables operator-level diagnosis such as joins and intersection. We evaluate RAG, ReAct RAG, GraphRAG, and information-extraction-to-SQL under matched conditions, with a long-context oracle ceiling to isolate retrieval failure. A two-axis framework -- index-time preservation versus query-time execution -- predicts where each paradigm fails, and the results bear it out: systems retrieve relevant text but discard the typed values operators need, and the deployable paradigm ranking inverts across operators, with similarity retrieval leading on filter/project and extraction-to-SQL on intersection and counting. Even given the gold evidence, a long-context oracle stays far from saturated, so operator execution -- not retrieval alone -- is a core bottleneck that a stronger answer model does not remove. QO-Bench reframes the goal from passage relevance to query-operator-preserving retrieval. 

---
# QuBLAST: A Framework for Quantizing Large Language Models with Block-Level Compression Approach and Activation Scaling Strategy 

**Authors**: Pasindu Wickramasinghe, Achyuta Muthuvelan, Rachmad Vidya Wicaksana Putra, Minghao Shao, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2606.04620)  

**Abstract**: LLMs have become the state-of-the-art algorithms for solving NLP tasks. However, they typically come at huge computational and memory costs, thus making them difficult to deploy on embedded systems. Toward this, state-of-the-art methods typically employ uniform post-training quantization (PTQ) across attention blocks of the network, hence overlooking the potential of applying different quantization levels in the same network. They also employ complex operations to mitigate the negative impact of activation outliers, hence incurring high computational overheads. Moreover, they have not considered evaluation using emerging LLMs with non-conventional attention architectures (e.g., state-space models), which pose different challenges in applying quantization. To address these limitations, we propose QuBLAST, a novel PTQ methodology that employs block-level compression approach with activation scaling strategy for LLMs. Block-level compression approach enables mixed-precision quantization across blocks of the network, while activation scaling strategy efficiently mitigates the negative impact of activation outliers. Specifically, QuBLAST first analyzes the sensitivity of different attention blocks in the pre-trained model through the cross-entropy loss analysis. QuBLAST leverages this sensitivity analysis to determine the weight quantization level for each attention block in the model. Furthermore, QuBLAST employs the activation scaling map for each block to control the range of activation values and mitigate the negative impact of activation outliers, thereby enabling better quantization results. Experimental results show that, QuBLAST reduces model sizes by 40%-45.2% across different model architectures (i.e., Qwen3-8B, Llama3-8B, Mistral v0.1-8B, and Falcon H1R-7B), while maintaining the performance within 5% perplexity increase for the WikiText-2 and WikiText-103 datasets. 

---
# Ekka: Automated Diagnosis of Silent Errors in LLM Inference 

**Authors**: Yile Gu, Zhen Zhang, Shaowei Zhu, Xinwei Fu, Jun Wu, Yida Wang, Baris Kasikci  

**Link**: [PDF](https://arxiv.org/pdf/2606.04594)  

**Abstract**: LLM serving frameworks are quickly evolving with a complex software stack and a vast number of optimizations. The rapid development process can introduce silent errors where output quality silently degrades without any explicit error signals. Diagnosing silent errors is notoriously difficult due to the substantial semantic gap between the high-level symptoms and the low-level root causes. We observe that diagnosis of silent errors can be effectively framed as a differential debugging problem by leveraging the existence of semantically correct reference implementations. We propose Ekka, an automated diagnosis system that identifies root causes by systematically aligning and comparing intermediate execution states between a target and a reference framework. We constructed a benchmark of real-world silent errors from popular serving frameworks, where Ekka shows 80% pass@1 diagnosis accuracy and 88% pass@5 diagnosis accuracy, outperforming state-of-the-art systems. Ekka also diagnoses 4 new silent errors from serving frameworks, all of which have been confirmed by the developers. 

---
# Synthetic Personalities: How Well Can LLMs Mimic Individual Respondents Using Socio-Economic Microdata? 

**Authors**: Leonard Kinzinger, Jochen Hartmann  

**Link**: [PDF](https://arxiv.org/pdf/2606.04592)  

**Abstract**: LLM-based digital twins promise to scale and accelerate market research, but most published twins are either coarse persona bots conditioned on a few demographic questions or detailed individual-level twins built on purpose-collected surveys and interview transcripts. Neither setup speaks to the operationally most relevant case for marketing practice: building detailed individual twins from the pre-existing heterogeneous panel data that firms already accumulate through CRM systems, loyalty programs, and repeat surveys. We construct detailed individual-level twins from the German Socio-Economic Panel (SOEP) and evaluate them across a $3 \times 5 \times 2 \times 2$ construction-method grid that covers three open-weights LLMs, five cumulative information depths ranked by normalized Shannon entropy, two embedding methods, and two reasoning modes, scoring over 2.1 million twin responses on 500 participants and 183 held-out questions. Twin quality rises with information depth but with diminishing returns past the 75 percent entropy quartile, which acts as a cost-efficient Pareto point relative to the best-performing 100 percent cells. Switching the embedding from a narrative persona summary to a raw dialog history of past responses raises hold-out accuracy in every model-by-reasoning cell at the 100 percent depth, while an explicit thinking mode raises rank-order correlation without moving accuracy. Best-cell accuracy reaches 78.8 percent and Fisher-$z$ correlation reaches $r = 0.590$ on the SOEP held-out evaluation set. The findings suggest that twin-based market research is no longer gated by data design, but by item volume, model selection, and a small set of construction-level decisions that this paper now maps. 

---
# Multi-SPIN: Multi-Access Speculative Inference for Cooperative Token Generation at the Edge 

**Authors**: Haotian Zheng, Zhanwei Wang, Mingyao Cui, Chang Cai, Hongyang Du, Kaibin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04581)  

**Abstract**: Speculative inference (SPIN) was originally developed as an efficient architecture to accelerate Large Language Models (LLMs). In this work, we propose its distributed deployment to enable cooperative token generation in a multiuser edge system; its advantage is to effectively balance computational loads between resource-constrained devices and servers. The resulting architecture, termed Multi-access SPIN (Multi-SPIN), utilizes on-device small language models to generate and upload candidate token drafts, while an edge server operates the LLM to verify them in parallel batches. Given the severe heterogeneity in users' computation and communication capabilities, the draft length emerges as a critical control variable that influences node-level computation loads and multi-access latency, thereby governing the sum token goodput. Consequently, considering frequency-division multiple access, we investigate the problem of multi-access draft control, a joint optimization of draft-length control and bandwidth allocation to maximize sum token goodput. We examine two cases: (1) homogeneous draft lengths across users to facilitate server-side batching, and (2) heterogeneous draft lengths to introduce a new dimension for goodput enhancement. By developing decomposition methods, we reduce these complex optimizations into tractable sub-problems, which allow efficient draft control algorithms to be derived in closed form. Our analysis shows that the optimal bandwidth allocation compensates users with weaker computation-and-communication capabilities in the homogeneous case due to the batching synchronization requirements, whereas its heterogeneous-case counterpart rewards users with higher acceptance rates by relaxing such requirements. Experiments using Llama-2 and Qwen3.5 model pairs across diverse tasks demonstrate that Multi-SPIN improves goodput by up to 88% over heterogeneity-agnostic baselines. 

---
# Rollout-Level Advantage-Prioritized Experience Replay for GRPO 

**Authors**: Gyeongtae Yoo, Sanghyeok Park, Soohyuk Jang, Ik-hwan Kim, Sungroh Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2606.04560)  

**Abstract**: Reinforcement learning from verifiable rewards with GRPO is a standard approach for post-training reasoning LLMs. It remains sample inefficient. Each rollout is used for a single gradient update and then discarded. Naive replay is not well suited in this setting because LLM policies drift quickly per gradient step. Stored rollouts therefore become stale and can destabilize training. We propose a rollout-level replay buffer for GRPO that stores and samples individual rollouts rather than whole groups. The buffer bounds staleness through age eviction. Any rollout older than tau_max training steps is removed. The buffer also preserves on-policy data via fresh-anchored composition. Each batch keeps its fresh on-policy rollouts and then concatenates replay rollouts drawn separately from the buffer. We prioritize replay by per-rollout advantage magnitude and recycle individual rollouts whose advantages are large. Across three Qwen3-Base scales on five math benchmarks, our method outperforms GRPO and naive replay baselines. Gains are positive at every scale and grow with model size. The largest gain is +4.35 pp on the five-benchmark average at 4B. Under an AES metric that jointly measures accuracy and token efficiency, the efficiency margin over GRPO is again largest at 4B, at +0.579. 

---
# Temporal Order Matters for Agentic Memory: Segment Trees for Long-Horizon Agents 

**Authors**: Yifan Simon Liu, Liam Gallagher, Faeze Moradi Kalarde, Jiazhou Liang, Armin Toroghi, Scott Sanner  

**Link**: [PDF](https://arxiv.org/pdf/2606.04555)  

**Abstract**: Long-horizon conversational agents need to interact with users through evolving events, tasks, and goals. Such histories are naturally temporal, yet many existing memory systems organize information primarily by topical similarity and may ignore the order in which events occur. We introduce Segment Tree Memory, or SegTreeMem, a memory architecture that represents conversation history as a temporally ordered Segment Tree over utterances. SegTreeMem incrementally inserts new utterances through an online rightmost-frontier update rule, preserving chronological order while forming hierarchical memory segments. For retrieval, SegTreeMem propagates relevance scores through the tree to combine local semantic matching with hierarchical temporal context. Across three long-horizon memory benchmarks and two LLM backbones, SegTreeMem improves answer quality over flat retrieval, graph-structured memory, and tree-structured memory baselines. Additional temporal-order permutation analysis shows that the performance gain depends on preserving temporal order during memory construction, supporting the claim that temporal order is a key structure for agentic memory. 

---
# Dynamic Infilling Anchors for Format-Constrained Generation in Diffusion Large Language Models 

**Authors**: Boyan Han, Yiwei Wang, Yi Song, Yujun Cai, Chi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04535)  

**Abstract**: Diffusion large language models (dLLMs) offer bidirectional attention and parallel generation, enabling them to exploit global context and naturally support format-constrained tasks like parseable JSON or reasoning templates. While straightforward fixed anchors can enforce such constraints, they often impose rigid spans, leading to truncated reasoning or redundant content. To overcome this, we propose Dynamic Infilling Anchors (DIA), a training-free method that dynamically estimates end-anchor positions to adjust generation length before iterative infilling. This flexible mechanism ensures structural correctness and semantic coherence, avoiding the inefficiencies of fixed-span methods. Experiments on reasoning benchmarks demonstrate that DIA substantially improves format compliance and answer accuracy, achieving significant zero-shot gains on GSM8K and MATH. These results establish DIA as a robust pathway toward reliable, structure-aware generation. 

---
# Optical-Guided Neural Collapse for SAR Few-Shot Class Incremental Learning 

**Authors**: Fan Zhang, Sijin Zheng, Fei Ma, Qiang Yin, Yongsheng Zhou, Fei Gao, Xian Sun  

**Link**: [PDF](https://arxiv.org/pdf/2606.04528)  

**Abstract**: Few-shot class-incremental learning (FSCIL) in synthetic aperture radar imagery presents unique challenges due to severe data scarcity and SAR-specific variability. In particular, strong azimuth sensitivity in SAR induces large intra-class variation and inter-class confusion, and FSCIL sequential updates further lead to catastrophic forgetting of previously learned classes. Inspired by neural collapse, we propose an optical-guided SAR FSCIL framework, which derives orthogonal feature subspaces from a data-rich optical ATR dataset and uses them as geometric priors to guide SAR feature learning. SAR features are projected onto these orthogonal subspaces via principal angle constraints, effectively transferring discriminative structure from the optical to the SAR domain. Specifically, our projection loss and the classifier loss optimized with a frozen simplex-ETF geometry jointly induce neural collapse by concentrating features around class means while maintaining large inter-class angles. We evaluate the approach on a benchmark comprising an optical ATR dataset and a SAR ATR dataset with 24 target classes, organized into a base training session and seven incremental sessions. Compared with recent FSCIL methods including NCFSCIL and so on, our method achieves the highest final accuracy and a favorable trade-off between final performance and performance degradation. Moreover, neural collapse metrics show improved intra-class compactness and inter-class separability, indicating that the learned features more closely approximate the ideal simplex-ETF geometry. 

---
# ANN Search: Recall What Matters 

**Authors**: Dimitris Dimitropoulos, Nikos Mamoulis  

**Link**: [PDF](https://arxiv.org/pdf/2606.04522)  

**Abstract**: Approximate nearest neighbor (ANN) search has become a core primitive in information retrieval and modern machine learning tasks, from classification to retrieval-augmented generation. The community evaluates and tunes ANN algorithms primarily on their throughput at a given Recall@k, the fraction of true exact neighbors retrieved. We argue that what really matters in ANN search is the quality of the retrieved results and not their overlap with the true kNN set. We show that using Recall@k to assess retrieval quality forces unnecessary computational overhead and investigate replacing it by 1/Ratio@k, the inverse approximation ratio. 1/Ratio@k evaluates the differences between the distances of the retrieved and true neighbors. It is judge-free, hyperparameter-free, and computable from standard ANN benchmark inputs alone. We benchmark state-of-the-art ANN algorithms across diverse datasets spanning a wide range of intrinsic dimensionalities, evaluating the two metrics comprehensively across efficiency, downstream classification, and retrieval-augmented generation. On the efficiency axis, optimizing for 1/Ratio@k reaches operational quality thresholds at a substantially lower computational cost than Recall@k. In downstream tasks, performance indicators (label precision, semantic similarity, BERTScore, and LLM-graded quality) remain highly stable even when Recall@k drops significantly. The inverse approximation ratio, on the other hand, closely mirrors this stability, tracking true utility much better than Recall@k. Ultimately, while Recall@k overstates the true cost of approximation, 1/Ratio@k offers a more accurate, deployable proxy for actual ANN quality. 

---
# Treat Traffic Like Trees: A Semantic-Preserving Hierarchical Graph-Based Expert Framework for Encrypted Traffic Analysis 

**Authors**: Yuantu Luo, Jun Tao, Linxiao Yu, Guang Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.04517)  

**Abstract**: Graph-based deep learning methods have been widely employed in encrypted traffic analysis to exploit latent correlations across different granularities. However, while complex preprocessing pipelines and sophisticated model structures often achieve strong performance, they may obscure inherent protocol semantics during representation learning. Moreover, the hierarchical structure of protocol layers and their corresponding fields, defined by protocol specifications and routinely utilized in manual traffic analysis, remains underexplored in existing learning frameworks. In this paper, we propose Protocol Tree Graph Attention with Mixture of Experts (PTGAMoE), a semantic-preserving hierarchical graph-based expert framework for encrypted traffic analysis. The field-based graph construction and expert committee design enable PTGAMoE to quantify the model's preferences for specific fields and protocols. Extensive experimental results on representative benchmark datasets under strict no-data-leakage settings demonstrate that PTGAMoE significantly outperforms state-of-the-art (SOTA) models. Furthermore, the semantic-preserving design provides interpretable insights into protocol-level feature importance and expert-level contributions, reflecting the model's decision-making logic in encrypted traffic classification tasks. 

---
# GeoMin: Data-Efficient Semi-Supervised RLVR via Geometric Distribution Modeling 

**Authors**: Guangcheng Zhu, Shenzhi Yang, Haobo Wang, Xing Zheng, Yingfan MA, Xuening Feng, Zhongqi Chen, Kai Tang, Zhengqing Zang, Bowen Song, Weiqiang Wang, Gang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.04516)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) significantly advances LLM reasoning, yet it faces a dilemma: standard supervised scaling is throttled by high annotation costs, while unsupervised alternatives suffer from severe model collapse. Recent semi-supervised RLVR methods address this by using a small labeled set to guide unlabeled data, achieving a promising trade-off between training efficacy and annotation cost. However, they suffer from a severe data-efficiency bottleneck due to the reliance on coarse performance heuristics, leaving a vast majority of valuable instances underutilized. To this end, we propose GeoMin, which models global feature distributions on labeled data to decode the structural discrepancy between correct and incorrect rollouts, thereby establishing a robust prior to assess the reliability of self-reward signals and fully unleash the potential of unlabeled data. Empirically, GeoMin outperforms the strongest baselines by +4.1% and even surpasses fully supervised models with only 10% of the annotations, demonstrating remarkable data efficiency. 

---
# Self-Evolving Deep Research via Joint Generation and Evaluation 

**Authors**: Han Zhu, Chengkun Cai, Yuanfeng Song, Xing Chen, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2606.04507)  

**Abstract**: Large Language Models (LLMs) have become increasingly adopted in daily applications, with deep research standing out as a particularly important capability. Unlike traditional question-answering (QA) tasks, deep research report generation lacks definitive ground-truth, making reward design inherently unverifiable and limiting effective reinforcement learning. Existing approaches mitigate this challenge with LLM-as-a-judge and query-dependent evaluation rubrics, but they still rely on static evaluators that cannot adapt their standards as the solver improves, leading to insufficient and eventually saturated optimization pressure. We address this limitation with a \textbf{s}elf-evolving \textbf{co}-evolutionary training framework for deep \textbf{re}search evaluation and generation (SCORE), which tightly couples an evaluator and a solver in a shared-parameter learning process. Rather than treating generation and evaluation as isolated modules, we leverage their intrinsic connection to enable joint improvement within a single shared-parameter model. To restrict this process, we introduce a meta-harness, which dynamically controls the evaluation environment based on solver performance, encouraging valid evaluation dimensions and sufficiently deep evaluator search. Extensive experiments on deep research benchmarks demonstrate consistent improvement in report generation quality, showing that co-evolving evaluation and generation is a promising direction for training open-ended research agents. 

---
# Smart Picks in the Dark: Towards Efficient RLVR for Reasoning via Tracing Metacognitive Pivots 

**Authors**: Guangcheng Zhu, Shenzhi Yang, Haobo Wang, Xing Zheng, Yingfan MA, Xuening Feng, Zhongqi Chen, Bowen Song, Weiqiang Wang, Gang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.04503)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has greatly advanced large reasoning models (LRMs), but it requires timely training on a huge fully-annotated dataset. To this end, data-efficient RLVR methods have been widely studied from two perspectives: (i) data selection methods identify a small subset of "golden" samples that yield near-full-data performance, but they rely on a pre-existing pool of labeled data. (ii) unsupervised RLVR methods train the model using its own internal supervision signals on large-scale unlabeled data, yet they exhibit suboptimal performance. Accordingly, we investigate the "pick in the dark" setup for RLVR, which aims to select, without prior supervision, unlabeled samples that are most beneficial for training and worthy of annotation. Through systematic analysis, we demonstrate that smart picks hinge on a well-calibrated uncertainty estimator to enable strategic partitioning of data for adaptive training regimes. Building on this insight, we propose PivotTrace, a three-way data triage framework that leverages attention dynamics to trace metacognitive pivots during reasoning. By precisely quantifying uncertainty through pivot density, PivotTrace achieves automated data routing to synergistically maximize both annotation and training efficiency. Empirically, PivotTrace surpasses the fully supervised LRM with only 29.3% annotated samples and 2.75 faster convergence. 

---
# SFMambaNet: Spectral-Frequency Enhanced Selective State Space Model for Correspondence Pruning 

**Authors**: Zhihua Wang, Yanping Li, Yizhang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.04493)  

**Abstract**: Correspondence pruning aims to identify inliers from an initial set of correspondences. Most existing Graph Neural Network (GNN)-based methods rely on geometric features mapped from coarse Euclidean coordinates, which struggle to capture the subtle geometric consistencies presented by inliers. While Mamba-based methods possess global receptive fields and long sequence modeling capabilities, they tend to accumulate substantial inconsistent features within the hidden state space, making it difficult to distinguish inliers from outliers. In this paper, we integrate frequency domain perception into this task for the first time and propose SFMambaNet, a novel Spectral-Frequency enhanced Mamba-based two-view correspondence pruning network. Our method is collaboratively composed of two components: First, we design a Local Spectral-Geometric Attention (LSGA) block. LSGA incorporates spectral positional encoding into local graph interactions and introduces multi-scale Mamba processing to enhance the capture of subtle geometric consistencies and improve local feature discriminability. Building upon this, we design a Spectral-Integrated Global Mamba (SIGM) block. SIGM embeds a frequency gating mechanism within the state space, utilizing the frequency information provided by LSGA to explicitly suppress high-frequency noise accumulation within hidden states and mitigate the propagation of inconsistent features. This enhances inlier-outlier separability and achieves robust global context modeling capabilities with nearly linear complexity. Extensive experiments demonstrate that SFMambaNet outperforms current state-of-the-art methods on several challenging tasks. The code is available at this https URL. 

---
# Evaluating Reasoning Fidelity in Visual Text Generation 

**Authors**: Jiajun Hong, Jiawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.04479)  

**Abstract**: Recent text-to-image (T2I) models can render highly legible and well-structured text within images, enabling applications including document generation and slide generation. However, it remains unclear whether such systems faithfully preserve reasoning ability when complex solutions must be expressed directly through rendered text, or whether they merely imitate surface-level patterns. We investigate this question by evaluating reasoning fidelity in visual text generation, where models must express complete reasoning processes as images. Our evaluation includes long text rendering, factual knowledge probing, context understanding, and multi-step reasoning. Across these settings, we find that current T2I models frequently produce semantic errors, logical inconsistencies, and incorrect intermediate steps, even when the rendered text appears visually clear. These failures contrast with the strong reasoning performance of text-only models on the same tasks. Our findings reveal a substantial gap between visual text generation and procedural reasoning, motivating more reliable visual text reasoning. 

---
# ChessMimic: Per-Rating Transformer Models for Human Move, Clock, and Outcome Prediction in Online Blitz Chess 

**Authors**: Thomas Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2606.04473)  

**Abstract**: We present ChessMimic, a system of three small encoder-only transformers - for move, thinking-time, and outcome prediction - conditioned on the position, recent move history, player rating, and clock state. We fit a separate instance of each model per 100-Elo rating band, trading parameter efficiency for sharper per-skill calibration. On a held-out month-wide slice of Lichess Rated Blitz games ChessMimic's human move prediction accuracy outperforms Maia-2 in every Elo band. Compared to Maia-3, our 9M parameter model's accuracy sits between Maia-3-5M and Maia-3-23M without the additional complexity of Geometric Attention Bias. In addition to the move matching model, we also train a game outcome model that conditions not only on the position, but also player ratings, time control, and remaining clock times. The outcome model achieves an AUC of 0.78 out of sample, beating Maia-2 as well as logistic regressions based on material, ratings, and clock time. Finally, we train a clock model that predicts human thinking times. The clock model provides a usable but non-SOTA per-ply think-time signal under ALLIE-style filters (Pearson r = 0.41, Spearman rho = 0.50, MAE 4.10 s, against ALLIE's reported r = 0.70), with the residual gap concentrated in per-position bucket sharpness rather than bucket-marginal calibration. A public demo is at this http URL and we release code, per-band weights, and the C++ data-filter pipeline code in GitHub. 

---
# Adaptive Calibration for Fair and Performant Facial Recognition 

**Authors**: Ryan Brown, Chris Russell  

**Link**: [PDF](https://arxiv.org/pdf/2606.04469)  

**Abstract**: We introduce Adaptive Calibration (AC), a novel calibration strategy for facial recognition that maps cosine similarity between normalized embeddings to well-calibrated probabilities. By incorporating local context into calibration, Adaptive Calibration corrects for a fundamental mismatch in cosine similarity, whereby the same distance can correspond to different match probabilities in different embedding regions. Our approach improves both overall performance and results in a fairer calibration without requiring demographic metadata.
Our approach consistently dominates existing methods both on accuracy and fairness metrics across a variety of pretrained models and standard benchmarks. AC provides a practical solution for equitable facial recognition, without requiring demographic group annotations, and while improving overall performance. Unlike existing approaches, our method provides continuous, region-specific calibration that avoids "leveling down" where fairness comes at the cost of degraded performance for some groups. 

---
# ParetoPilot: Zero-Surrogate Offline Multi-Objective Optimization via Infer-Perturb-Guide Diffusion 

**Authors**: Ruiqing Sun, Sen Yang, Dawei Feng, Bo Ding, Yijie Wang, Huaimin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04468)  

**Abstract**: Offline multi-objective optimization (Offline MOO) aims to discover novel Pareto-optimal designs based on static datasets without expensive environment interactions. While recent generative methods have achieved notable success, they predominantly rely on external surrogate models. This dependency introduces significant computational overhead, suffers from deceptive evaluations, and deviates from the prevailing paradigm of jointly training mainstream generative models with conditions. To address these bottlenecks, we propose ParetoPilot, a novel zero-surrogate diffusion framework for offline MOO. ParetoPilot fully leverages the conditional priors inherently embedded within pre-trained diffusion models. At its core, the framework introduces the Infer-Perturb-Guide (IPG) engine, which is seamlessly interleaved within the unconditional denoising steps of the reverse generation process. First, it implicitly infers the instantaneous objective direction by matching conditional and unconditional noise predictions. Next, it mathematically orthogonalizes a parallel gravity field for strict convergence and an edgeness-aware repulsive force for mutual diversity, creating a dynamically annealed perturbation vector. Finally, this perturbed target seamlessly steers the generation process via standard Classifier-Free Guidance (CFG). Extensive experiments across 51 tasks demonstrate that ParetoPilot outperforms 14 state-of-the-art surrogate-based and inverse generative baselines. By eliminating auxiliary proxy training, our approach preserves data privacy while achieving hypervolume improvement and robust Pareto front coverage. 

---
# SePO: Self-Evolving Prompt Agent for System Prompt Optimization 

**Authors**: Wangcheng Tao, Han Wu, Weng-Fai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2606.04465)  

**Abstract**: System prompt optimization improves agent behavior without modifying the underlying model, yielding human-readable, model-agnostic instructions. Existing methods build a prompt agent that refines task agents' system prompts, yet leave the prompt agent's own system prompt hand-engineered and fixed. We propose Self-Evolving Prompt Optimization (SePO), which treats the prompt agent's own system prompt as an optimization target alongside task agents' system prompts. SePO adopts a self-referential design. A single prompt agent improves both task agents' system prompts and its own under an open-ended evolutionary search that maintains an archive of candidate prompts as stepping stones. Training proceeds in two stages: pre-training evolves the prompt agent on a multi-task pool, and fine-tuning then applies it to a target task. Across five benchmarks spanning math (AIME'25), abstract reasoning (ARC-AGI-1), graduate-level science (GPQA), code generation (MBPP), and logic puzzles (Sudoku), SePO consistently outperforms Manual-CoT, TextGrad, and MetaSPO, improving the average accuracy by 4.49 points compared to Manual-CoT. The prompt optimization skill from pre-training also generalizes to tasks beyond the pre-training mixture, rather than memorizing per-task prompts. 

---
# CyberGym-E2E: Scalable Real-World Benchmark for AI Agents' End-to-End Cybersecurity Capabilities 

**Authors**: Tianneng Shi, Robin Rheem, Dongwei Jiang, Mona Wang, Francisco De La Riega, Zhun Wang, Jingzhi Jiang, Alexander Cheung, Sean Tai, Jonah Cha, Jianhong Tu, Gabriel Han, Chenguang Wang, Jingxuan He, Wenbo Guo, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2606.04460)  

**Abstract**: AI has the potential to transform cybersecurity by enabling systems that can autonomously detect, analyze, and remediate software vulnerabilities. However, existing cybersecurity evaluations of AI systems are limited in scale or scope, and fail to capture the end-to-end lifecycle of real-world software vulnerability discovery and remediation. To address this gap, we propose CyberGym-E2E, a large-scale and realistic end-to-end cybersecurity benchmark that comprehensively evaluates AI agents' abilities across the full lifecycle of vulnerability discovery, PoC generation, and patch generation. CyberGym-E2E is comprehensive and scalable, as we build an automated, agent-enhanced pipeline for transforming open-source vulnerability data into realistic evaluation environments. Currently, the benchmark consists of 920 real-world vulnerabilities across 139 different open-source projects. 

---
# Token Rankings are Unforgeable Language Model Signatures 

**Authors**: Matthew Finlayson, Andreas Grivas, Xiang Ren, Swabha Swayamdipta  

**Link**: [PDF](https://arxiv.org/pdf/2606.04459)  

**Abstract**: Language model parameters are known to impose unique (to each model) geometric constraints on their logit outputs, which serves as a signature that identifies the model, but also leaks the model's final layer parameters when an API distributes logits. We investigate more restrictive APIs that expose token rankings (i.e., their ordering by probability, but not the probability values) and find that rankings also constitute a signature: every model has a unique set of feasible top-$k$ rankings for sufficiently large $k$. Furthermore, the ranking signature is the first known (polynomially) unforgeable signature, since finding a model with the same set of feasible rankings is NP-hard. On the security front, we find that token rankings are already sufficient to approximately steal the final layer of the model, similar to logits, though the approximation is too coarse to forge the signature, and can be effectively countered by restricting the API to top-$k$ tokens with sufficiently small $k$. Since the top-$k$ required to present the model signature is generally smaller than the $k$ required to prevent stealing, it is possible for an API to present an unforgeable signature without leaking model parameters. 

---
# RowNet: A Memory Transformer for Tabular Regression 

**Authors**: Askat Rakhymbekov, Gulshat Muhametjanova  

**Link**: [PDF](https://arxiv.org/pdf/2606.04445)  

**Abstract**: Real estate valuation is a structured regression problem in which prices are governed by heterogeneous feature types, sparse regional effects, nonlinear interactions, and the practical logic of comparable properties. Standard multilayer perceptrons treat each row as an isolated vector and must learn locality, scale sensitivity, and categorical matching from supervision alone. Gradient-boosted decision trees provide strong tabular baselines, but their feature-centric splitting mechanism does not explicitly model the retrieval of similar historical observations.
This paper presents RowNet, a retrieval-based neural architecture for real estate price-per-square-meter prediction. RowNet represents a query property through pairwise similarity features against a memory bank of labeled properties. A first retrieval layer estimates a coarse target from feature-only similarities. A second layer augments the memory comparison with target-consistency features and uses multiple learned attention heads to retrieve complementary comparable sets. A final mixture-of-experts module combines learned gating, residual correction, entropy regularization, and head-diversity regularization to produce the prediction. 

---
# MemoryDocDataSet: A Benchmark for Joint Conversational Memory and Long Document Reasoning 

**Authors**: Qiyang Xie, Jialun Wu, Xinjie He, Su Liu, Shuai Xiao, Zhiyuan Lin, Weikai Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.04442)  

**Abstract**: AI systems increasingly need to combine two demanding capabilities: navigating multi-session conversation history and performing deep reading comprehension within long documents. Yet no existing benchmark evaluates both simultaneously. We introduce MemoryDocDataSet, a synthetic benchmark of 50 micro-worlds and 1,000 QA pairs in which each instance comprises 3-5 personas, a temporal event graph spanning months of activity, 3-5 real long documents (20,000-50,000 tokens each sourced from the Caselaw Access Project), multi-session conversations grounded on those documents, and 20 question-answer pairs across five reasoning categories. The defining feature is the Hybrid source tag: questions requiring a system to first navigate conversation history to identify which document is relevant, then extract the answer from within that document. Hybrid questions account for 75.1% of the dataset. Dataset quality is characterised through a prompt-sensitivity self-consistency analysis using LLM-as-judge, yielding a median Cohen's $\kappa = 0.634$ across all 50 micro-worlds. We evaluate six baseline configurations spanning truncated context, long-context LLMs, retrieval-augmented generation (RAG), and memory systems. The best baseline (RAG-Both) achieves 0.358 overall F1 and 0.342 on Hybrid. Document-only retrieval (RAG-Doc) collapses to 0.267 on Hybrid despite achieving 0.453 on Doc-only questions, demonstrating a clear joint-retrieval gap that motivates architectures unifying conversational memory with long-document navigation. We release the dataset, generation pipeline, and all baseline implementations. 

---
# LoopMoE: Unifying Iterative Computation with Mixture-of-Experts for Language Modeling 

**Authors**: Wenkai Chen, Tianshu Li, Wenyong Huang, Yichun Yin, Lifeng Shang, Chengwei Qin  

**Link**: [PDF](https://arxiv.org/pdf/2606.04438)  

**Abstract**: Mixture-of-Experts (MoE) and looped architectures scale models along two orthogonal axes, namely parameter capacity and effective depth. However, mainstream looped architectures rely on dense backbones that couple parameter count with per-token FLOPs, which makes it impossible to isolate the effect of iterative computation under matched budgets. To this end, we present LoopMoE, a looped MoE language model that integrates sparse routing with iterative weight-shared computation through two designs. The first is IterAdaLN, which resolves weight-sharing symmetry via a modulation signal jointly conditioned on the iteration index and the per-token hidden state. The second is a capacity-balancing strategy that recovers the attention-to-FFN active parameter ratio of well-tuned non-looped references. Together, these designs enable the first strictly controlled, head-to-head evaluation of a looped MoE against a Vanilla MoE under identical total parameters, per-token FLOPs, and active sublayer ratios. At the 3B scale, LoopMoE outperforms the Vanilla MoE on 8 of 9 downstream benchmarks with an average improvement exceeding 1 point. At the 9B scale, LoopMoE continues to outperform the matched Vanilla MoE, indicating that the architectural gain persists at larger scale. Our work establishes a controlled synthesis of sparsity and recurrence, and suggests a promising direction for looped language models. 

---
# What If Prompt Injection Never Left? Exploring Cross-Session Stored Prompt Injection in Agentic Systems 

**Authors**: Yuanbo Xie, Tianyun Liu, Yingjie Zhang, Suchen Liu, Yulin Li, Liya Su, Tingwen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.04425)  

**Abstract**: Modern agentic systems transform LLMs from session-bounded assistants into stateful systems that persist and evolve shared world state across sessions through memories, filesystems, tools, and other long-lived contextual artifacts. This shift fundamentally expands the attack surface of prompt injection. However, prior works on prompt injection have largely focused on model-level threats within a single session, overlooking how cross-session persistent system state fundamentally changes the system-level risk of agentic systems. Inspired by stored cross-site scripting in web systems, we introduce cross-session stored prompt injection, where a successful injection can persist within agentic system state and silently influence future executions long after the original attacker interaction has ended. To systematically study this threat, we formalize stored prompt injection and develop a taxonomy of how adversarial content persists and affects agentic systems across sessions. We further develop a benchmark and sandbox toolkit to evaluate the risks of stored prompt injection, enabling quantitative analysis of attack success across different models, attack goals, and persistence channels. Our findings highlight that persistence transforms prompt injection from an ephemeral model-level threat into a long-lived system-level vulnerability embedded within agent execution state. We hope this work draws broader attention to this emerging threat and motivates the community to systematically study and mitigate system risks arising from persistence in agentic systems. 

---
# L-TGVN: Leveraging Longitudinal Priors for Personalized Rapid MRI 

**Authors**: Arda Atalık, Sumit Chopra, Daniel K. Sodickson  

**Link**: [PDF](https://arxiv.org/pdf/2606.04419)  

**Abstract**: MRI provides excellent soft-tissue contrast without ionizing radiation, but long acquisition times increase patient discomfort while also raising exam costs and limiting scanner throughput. A common approach to reduce scan time is to acquire fewer measurements, which yields an ill-posed linear inverse problem; recovering diagnostic-quality images therefore requires incorporating prior knowledge beyond the measured data. In follow-up exams, the most recent prior scan of a patient can provide a highly informative subject-specific context, but practical use is complicated by temporal changes (including pathology progression), misalignment between scans, and protocol drift across acquisitions. In this work, we introduce L-TGVN, a Longitudinal Trust-Guided Variational Network that leverages prior scans as side information to reconstruct the current scan from heavily undersampled measurements. Crucially, L-TGVN constrains the influence of prior scans to be consistent with the acquired measurements. Unlike many existing longitudinal reconstruction methods, it does not require explicit pre-registration between prior and current scans. It further accommodates differences in acquisition protocols across visits (e.g., changes in sequence parameters). We evaluate L-TGVN against matched-capacity baselines, including prior-guided methods and methods that do not use longitudinal priors, and observe consistent improvements in standard quantitative metrics together with better preservation of fine structures at challenging accelerations. Source code is available at this http URL. 

---
# An Empirical Study of Data Scale, Model Complexity, and Input Modalities in Visual Generalization 

**Authors**: Luoyidi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.04409)  

**Abstract**: Modern deep neural networks usually have large parameter scales and nonlinear hierarchical structures, and they have achieved strong performance in computer vision. However, the source of their generalization performance remains difficult to explain using traditional statistical learning theory. Among the factors that may affect visual generalization, data scale, model complexity, and input modalities are fundamental and controllable variables. This study empirically analyzes how these three factors influence model generalization performance. Specifically, in a preliminary experiment, we construct a one-dimensional nonlinear function and vary the number of training samples and the polynomial degree to observe the effects of data scale and model complexity on model performance. In the main experiments, we compare model performance on CIFAR-10 and CIFAR-100 under different training data scales, model architectures, and input modalities. The experimental results show that increasing the training data scale consistently improves generalization performance, whereas changes in model complexity do not provide stable gains. In addition, removing color information degrades model performance, while explicit prior features such as gradients, edges, and wavelets have inconsistent effects across different model architectures. Overall, this study provides an empirical analysis of the relationships among data scale, model complexity, input modalities, and visual generalization performance. Code and experimental logs are available at: this https URL. 

---
# An Ensembled Latent Factor Model via Differential Evolution and Gradient Descent Optimization 

**Authors**: Rui Zhang, Jinhang Liu, Wenbo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04408)  

**Abstract**: High-dimensional and incomplete (HDI) data are prevalent in many real-world big data scenarios. Latent factor models serve as a common representation learning approach, capable of uncovering informative latent factors from such data. Nevertheless, most existing latent factor models rely solely on gradient descent for optimization, which may lead to insufficient and biased representations, particularly when dealing with heterogeneous HDI data. Thus, this study proposes an Ensembled Latent Factor Model via Differential Evolution and Gradient Descent Optimization (ELFM-DEGDO) with two-fold designed: 1) two diverse latent factor models are independently modeled via differential evolution and gradient descent optimization, respectively, and 2) the two diverse latent factor models are combined via a customized self-adaptive weighting mechanism to effectively fuse their strengths. By leveraging the complementary advantages of both optimization paradigms, ELFM-DEGDO is able to produce more comprehensive and less biased representations for HDI data. Three HDI datasets are tested to show that ELFM-DEGDO consistently performs better than related several latent factor models. 

---
# Low-Rank Decay for Grokking in Scale-Invariant Transformers: A Spectral-Geometric View 

**Authors**: Mingyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.04405)  

**Abstract**: Modern Transformer architectures frequently employ normalization mechanisms such as RMSNorm and Query-Key Normalization, making parts of the model approximately scale-invariant with respect to weight magnitudes. In this regime, standard Frobenius-norm weight decay acts purely along the radial direction of the weight space and cannot directly simplify the function represented by the normalized layer. We study grokking in small algorithmic tasks through this lens and propose \emph{Low-Rank Decay} (LRD), a nuclear-norm-like spectral regularizer whose subgradient -- the polar factor $UV^\top$ -- retains a tangential component even in the scale-invariant setting. This distinction has a concrete dynamical consequence: after the model memorizes the training set and task gradients vanish, L2 decay can no longer reshape the weight spectrum, whereas LRD continues to compress singular values in an $\ell_1$-like fashion. On modular arithmetic tasks, we find that LRD induces rapid effective-rank collapse in Query/Key matrices and expands the data-fraction boundary at which delayed generalization (grokking) occurs. We further provide a spectral-geometric interpretation through the ``needle-to-fan'' expansion of the nuclear-norm subdifferential near low-rank strata. 

---
# TITAN-FedAnil+: Trust-Based Adaptive Blockchain Federated Learning for Resource-Constrained Intelligent Enterprises 

**Authors**: Muhammad Hadi, Muhammad Jahangir, Talha Shafique, Muhammad Khuram Shahzad  

**Link**: [PDF](https://arxiv.org/pdf/2606.04388)  

**Abstract**: Federated Learning (FL) has emerged as an effective paradigm for collaborative intelligence while preserving data privacy. However, data heterogeneity arising from non-IID distributions and decentralized security threats remain significant challenges, particularly in resource-constrained enterprise environments. This paper presents TITAN-FedAnil+, a Trust-Based Adaptive Network for blockchain-enabled federated learning in intelligent enterprises. The proposed framework introduces affinity propagation-based adaptive clustered aggregation to identify and filter malicious updates without requiring prior knowledge of the number of attackers. In addition, GPU-accelerated vectorization is employed to improve computational efficiency, while a signed state jump mechanism enables lightweight blockchain resynchronization. Experimental results demonstrate substantial reductions in memory overhead, achieving up to 81% savings across 50 communication rounds on constrained 8 GB edge devices compared with the baseline framework. The results indicate that TITAN-FedAnil+ effectively improves robustness, scalability, and resource efficiency for secure federated learning deployments in intelligent enterprise environments. 

---
# Rethinking Sales Lead Scoring with LLM-based Hierarchical Preference Ranking 

**Authors**: Chenyu Zhang, Yiwen Liu, Yin Sun, Xinyuan Zhang, Yuji Cao, Junming Jiao, Juyi Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2606.04387)  

**Abstract**: Sales lead conversion in high-stakes domains (e.g., automotive, real estate) differs fundamentally from e-commerce recommendation due to prolonged decision cycles and multi-stage funnels. Traditional lead scoring methods rule-based scorecards, machine learning, or pointwise CTR models face severe challenges: sparse supervision, a semantic gap in unstructured CRM logs, and inability to capture relative lead priority. While Large Language Models(LLMs) offer superior semantic understanding of customer interactions, general-purpose LLMs are ill-suited for lead ranking: they generate text rather than comparable scores, and lack alignment with the hierarchical priorities of sales funnels. We introduce an LLM-based discriminative framework for sales lead scoring, which supports joint modeling of structured CRM features and unstructured customer interactions. On top of this framework, we propose HPRO (Hierarchical Preference Ranking Optimization), which augments sales lead scoring with a hierarchical preference ranking objective. HPRO employs a margin-aware Bradley-Terry formulation to transform sparse binary labels into dense, funnel-aware preference pairs, enabling lead scoring to leverage both pointwise and pairwise supervision. Experiments on large-scale data from a leading NEV brand demonstrate state-of-the-art classification (AUC 0.8161) and ranking performance (+39.7% precision among top-ranked leads). A 132-day online A/B test validates 9.5% sales volume uplift, confirming real-world commercial impact. 

---
# LCSHBench: A Multilingual, Consensus-Grounded Benchmark for Library of Congress Subject Heading Assignment 

**Authors**: Kwok Leong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04382)  

**Abstract**: Automated subject cataloging assigns controlledvocabulary headings to bibliographic records, but LCSH has no standard public benchmark. We introduce LCSHBench: 22,346 books in 15 languages from the openly licensed Harvard, Columbia, and Princeton catalogs. Records enter only when at least two independent cataloging agencies assigned LCSH; we release per-catalog provenance plus union and unanimous answer views. A concordance study of 465,187 works cataloged by all three libraries shows why this design matters: libraries usually agree on the underlying topic (93.3% share a concept-level heading) but often differ in exact expression (39.4% have identical heading sets). LCSHBench therefore scores both exact and concept matches, with set and rank metrics broken down by language and heading type, across open-vocabulary generation and full-vocabulary retrieval. As a first demonstration, a low-rank fine-tune of a 300M on-device embedder improves cross-lingual retrieval and beats a 3,072-dimensional hosted embedder on development exact recall@200 (0.659 vs 0.623). The language panel shows the gain is not uniform, and held-out-test and end-to-end confirmation remain future work. 

---
# From Symbolic to Geometric: Enabling Spatial Reasoning in Large Language Models 

**Authors**: Chen Chu, Bita Azarijoo, Li Xiong, Khurram Shafique, Cyrus Shahabi  

**Link**: [PDF](https://arxiv.org/pdf/2606.04381)  

**Abstract**: Recent large language models (LLMs) often appear to exhibit spatial reasoning ability; however, this capability is largely \emph{symbolic}, arising from pattern matching over spatial language rather than true \emph{geometric} reasoning over space. Because LLMs operate on discrete tokens, they lack native support for continuous spatial representations, explicit geometric computation, and structured spatial operators. To address this limitation, we introduce the \emph{Spatial Language Model (SLM)}, the first multimodal LLM that treats location information as a first-class modality and enables geometric spatial reasoning within the model's inference process. SLM directly operates on learned spatial representations rather than textual descriptions of spatial relations. To support effective training, we construct a \emph{Spatial Instruction Dataset} that aligns spatial representations, atomic geometric operations, and natural language instructions. We further propose a new benchmark named \emph{SpatialEval}, which is designed to evaluate spatial reasoning across attributes, distance, topology, and relative-position tasks. Extensive experiments show that SLM significantly outperforms existing LLM-based approaches that rely on symbolic reasoning via prompt engineering or textual abstraction, demonstrating the benefits of integrating geometric spatial representations for robust spatial reasoning.
Our instruction dataset, evaluation benchmark, model training codes, and models' checkpoints can be found at:
\hyperlink{this https URL}{this https URL}. 

---
# DSIRM: Learning Query-Bridged Discrete Semantic Identifiers for E-commerce Relevance Modeling 

**Authors**: Bokang Wang, Xing Fang, Mingmin Jin, Jing Wang, Zhentao Song, Guangxin Song, Jianbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.04374)  

**Abstract**: Despite rapid progress of continuous embeddings for e-commerce search relevance, a long-standing open problem is the difficulty in capturing fine-grained attribute distinctions. While discrete Semantic Identifiers (SIDs) have been widely adopted as a promising alternative, existing SID generation methods rely heavily on unsupervised quantization. In realistic scenarios, the lack of explicit supervision often makes it more difficult to dictate which items should share an SID, resulting in limited capability for query-dependent ranking. To address the issue of unsupervised SIDs, we propose to explicitly model discrete relevance features and develop a Discrete Semantic Identifier Relevance Model (DSIRM). Specifically, we present a query-bridged contrastive quantization approach on the item side, injecting query-item interaction supervision into Residual Quantization to actively learn relevance-aware semantic partitions. On the other hand, we explore generative LLMs on the query side to explicitly predict item SIDs from text, resolving tail queries and intent ambiguity. Hierarchical prefix matching between query and item SIDs yields discriminative features that perfectly complement dense signals. Extensive experimental results on Tmall's production data show that our proposed approach has achieved better results, improving offline AUC by +1.54\%. Deployed via an efficient hybrid architecture, it achieves significant online lifts (+0.13\% UCTR, +0.25\% UCTCVR), proving its massive industrial value. 

---
# Selective Coupling of Decoupled Informative Regions: Masked Attention Alignment for Data-Free Quantization of Vision Transformers 

**Authors**: Biao Qian, Yang Wang, Yong Wu, Jungong Han  

**Link**: [PDF](https://arxiv.org/pdf/2606.04373)  

**Abstract**: Data-Free Quantization (DFQ) addresses data security concerns by synthesizing samples, without accessing real data. It has garnered increasing attention in the context of Vision Transformers (ViTs), owing to the superiority of the self-attention mechanism compared to classical convolutional operation. However, previous DFQ arts for ViTs often suffer from a distribution mismatch between synthetic samples and input distribution expected by quantized models Q, resulting in the suboptimal performance. In this paper, we propose a novel Masked Attention Alignment approach for Data-Free Quantization of ViTs, named MaskAQ, revealing that: 1) the semantics in the self-attention mechanism is predominantly localized to a sparse subset of patches, called informative regions; 2) the informative regions dominate the mutual information between synthetic samples and Q's outputs. To these ends, we incorporate differential entropy maximum over patch similarity of synthetic samples, to decouple informative regions from noisy background. To couple with varied Q, the informative regions are selected to align full-precision models with Q via a masked attention alignment objective, thus yielding high-quality synthetic samples. Furthermore, a periodic sample refreshing strategy comes up to endow MaskAQ with the capacity to continually adapt to the evolving state of Q throughout the training process, to preserve desirable mutual information with synthetic samples. Extensive experiments verify the merits of MaskAQ over state-of-the-art approaches across multiple backbones and downstream tasks. Our code is available at this https URL. 

---
# Multi-Granularity 3D Kidney Lesion Characterization from CT Volumes 

**Authors**: Renjie Liang, Zhengkang Fan, Jinqian Pan, Chenkun Sun, Jiang Bian, Russell Terry, Jie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.04365)  

**Abstract**: Radiology reports describe kidney lesions by type, size, enhancement, and attenuation, yet existing 3D methods predict only at the patient or organ level. We reformulate kidney CT characterization as a per-lesion set-prediction task: one model emits a variable number of lesions per kidney, each with four clinical attributes. We curated 2,619 CT volumes from 788 patients at one academic medical center, with multi-granularity side- and per-lesion labels, and used KiTS23 (489 cases) for zero-shot external validation. We propose \textbf{LesionDETR}, a DETR-style architecture with size-distance Hungarian matching and a hierarchical loss that aggregates per-slot outputs to side-level objectives. Across four input representations and six encoder initializations, two design choices dominate: a segmentation mask as an input channel, and same-domain abdominal pretraining (SuPreM); generic large-corpus pretraining is no better than random initialization. LesionDETR reaches bilateral side-level abnormality AUC $0.799 \pm 0.009$ on UF-Health and $0.817 \pm 0.072$ on KiTS23. A count-conditioned variant reaches per-lesion mAP $0.190 \pm 0.083$ on cystic lesions; rare solid-lesion AP stays at the noise floor, pointing to targeted data collection, not architecture, as the next bottleneck. The framework yields verified per-lesion predictions for downstream structured report generation. 

---
# MorphoQuant: Modality-Aware Quantization for Omni-modal Large Language Models 

**Authors**: Yue Wu, Changyuan Wang, Zixuan Wang, Shilin Ma, Yansong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04349)  

**Abstract**: Conventional Post-Training Quantization (PTQ) methods struggle with 4-bit Omni-modal Large Language Models (OLLMs) due to the extreme distribution heterogeneity and disparate outlier patterns across modalities. To address this, we propose MorphoQuant, a modality-aware PTQ framework engineered to preserve cross-modal morphology and mitigate outlier loss. Specifically, we introduce Distribution-Aware Bias Compensation (DABC), which selectively absorbs long-tailed outliers into channel-wise biases. This mechanism safeguards outlier magnitudes while maintaining high-precision discretization for dense inliers, thereby preserving accurate discretization across diverse modal distribution. Complementing this, we propose Morphology-Directed Quantization Function Optimization (MDQFO) to co-optimize the quantization grid with the bias mask, ensuring fine-grained alignment across modalities. Extensive evaluations on Qwen2.5-Omni across benchmarks like MMMU and Video-MME demonstrate our approach's superiority. Notably, our W4A4 model achieves 76.63% on ScienceQA, significantly outperforming SOTA W4A4 methods and surprisingly surpassing the W4A16 baseline, which fully demonstrates the exceptional accuracy-efficiency trade-off of our framework. 

---
# HYolo: An Intelligent IoT-Based Object Detection System Using Hypergraph Learning 

**Authors**: Isha Abid, Fawad Khan, Muhammad Khuram Shahzad  

**Link**: [PDF](https://arxiv.org/pdf/2606.04345)  

**Abstract**: This paper presents HYolo, an intelligent IoT-based object detection framework that integrates hypergraph learning into the YOLO architecture. Traditional YOLO-based object detection models primarily capture pairwise feature interactions and may fail to model complex high-order relationships among objects and contextual features. To address this limitation, HYolo incorporates hypergraph learning to capture richer contextual dependencies and improve object representation. Experimental evaluation on the COCO dataset demonstrates significant performance improvements over baseline YOLO models. The proposed approach achieves approximately 12% improvement in mAP@50 while enhancing overall detection accuracy and robustness. By modeling high-order feature relationships, HYolo provides improved contextual understanding and more reliable object detection performance in IoT-based environments. The results indicate that integrating hypergraph learning into object detection pipelines offers a promising direction for intelligent and context-aware IoT vision systems. 

---
# Expectations vs. Realities: The Cost of MSE-Optimal Forecasting Under Conditional Uncertainty 

**Authors**: Riku Green, Zahraa S. Abdallah, Telmo M Silva Filho  

**Link**: [PDF](https://arxiv.org/pdf/2606.04342)  

**Abstract**: Multi-step time series forecasting (MSF) is commonly evaluated using point-wise error metrics such as mean squared error (MSE), implicitly treating the conditional mean as a sufficient target. We show that this can be misleading under conditional uncertainty, where the conditional expectation becomes unrepresentative of typical realized values at longer horizons. We formalize this effect through a conditional uncertainty gap and prove that whenever this gap is nonzero, no deterministic predictor can simultaneously minimize MSE and match the marginal distribution of realized futures. This establishes a fundamental, model-agnostic trade-off between point accuracy and marginal realism in MSF evaluation. Using controlled stochastic dynamical systems and nine real-world forecasting benchmarks, we empirically characterize the resulting accuracy--realism frontier and \textbf{quantify the practical cost of MSE-only model selection}. As conditional uncertainty increases with forecast horizon, the attainable set expands into a pronounced Pareto front, separating MSE-optimal but under-dispersed predictors from methods that trade accuracy for realistic marginal variability. \textbf{Across benchmarks, we find that small relaxations in MSE ($\boldsymbol{\le 5\%}$) frequently unlock disproportionate gains in marginal realism, with median improvements of $\mathbf{17.3\%}$ and gains exceeding $\mathbf{30\%}$ in some datasets.} We further show that common forecasting strategies systematically occupy different regions of this frontier: direct multi-output predictors concentrate near the accuracy-optimal extreme, while recursive strategies and sample-based inference favors marginal realism. Together, these results expose a structural failure mode of MSE-based evaluation in long-horizon forecasting and recast strategy and inference selection as navigation of an unavoidable accuracy--realism trade-off. 

---
# From Untrusted Input to Trusted Memory: A Systematic Study of Memory Poisoning Attacks in LLM Agents 

**Authors**: Pritam Dash, Tongyu Ge, Aditi Jain, Tanmay Shah, Zhiwei Shang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04329)  

**Abstract**: Memory is a core component of AI agents, enabling them to accumulate knowledge across interactions and improve performance. However, persistent memory introduces the risk of memory poisoning, where a single adversarial memory write can exert long-term influence over agent behavior. We present a systematic study of memory poisoning in LLM-based agents. We identify four memory write channels and nine structural vulnerabilities in model capabilities, system prompt design, and agent system architecture that make these channels exploitable. Based on these vulnerabilities, we develop a taxonomy of six classes of memory poisoning attacks. Furthermore, we design MPBench -- a benchmark for evaluating memory poisoning attacks, and show that agents designed to write and retrieve memory more aggressively are more exploitable. We also show that existing prompt injection defenses fail to cover memory poisoning attacks. Our findings provide a foundation for understanding and mitigating memory poisoning attacks against AI agents. 

---
# Generalizable Multi-Task Learning for Wireless Networks Using Prompt Decision Transformers 

**Authors**: Fatih Temiz, Shavbo Salehi, Melike Erol-Kantarci  

**Link**: [PDF](https://arxiv.org/pdf/2606.04328)  

**Abstract**: Future wireless networks demand rapid adaptation to highly heterogeneous environments and dynamic task configurations, necessitating a shift from conventional rule-based and optimization-driven radio resource management (RRM) toward artificial intelligence (AI)-driven RRM. AI-driven approaches can learn complex nonlinear relationships, generalize across diverse network conditions and enable real-time, scalable and autonomous decision-making. Among RRM techniques, coordinated multipoint (CoMP) transmission is pivotal for mitigating inter-cell interference and enhancing cell-edge performance, thereby improving quality of experience (QoE) in dense deployments. However, optimal multi-cell selection remains a complex combinatorial challenge as it requires jointly optimizing over many possible serving-cell combinations under dynamic traffic and channel conditions. Despite their success, conventional deep reinforcement learning (DRL) methods such as proximal policy optimization (PPO) suffer from poor sample efficiency, limited generalization, and costly retraining when state and action spaces change. To address these bottlenecks, we propose a Prompt Decision Transformer (PromptDT) based multi-task learning framework capable of learning across diverse network configurations and reformulating multi-cell selection as a sequence modeling problem. By leveraging offline trajectories and task-specific prompts, PromptDT enables scalable learning across diverse network configurations, including varying base stations and user equipment counts, and scheduler policies. Experimental results demonstrate that PromptDT improves QoE by up to 49% in multi-task settings compared to baselines, with performance scaling positively alongside model capacity. Moreover, PromptDT generalizes effectively to unseen tasks, achieving robust few-shot adaptation to new network configurations without retraining or fine-tuning. 

---
# A Geometric Characterization of the Stationary Plateau for Two-Layer Neural Networks 

**Authors**: Tian Ding, Dawei Li, Ruoyu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2606.04327)  

**Abstract**: We investigate the geometric structure of stationary plateaus that arise in the loss landscape of two-layer neural networks with smooth activation functions. We focus on the phenomenon of "neuron splitting" where duplicating a hidden neuron yields an affine set of stationary points in a wider network. We provide a comprehensive classification of all stationary points on these plateaus, determining under what conditions they constitute local minima or saddle points. Our characterization hinges on a per-neuron curvature object we term the "inner Hessian" matrix. Our analysis reveals that the definiteness of the inner Hessian and the choice of splitting coefficients jointly dictate the local geometry of the plateau. We show that "splitting" a local minimum can yield either a mixture of local minima and saddles or an all-saddle plateau, with a concrete sure-saddle region identified under mild assumptions. In contrast, splitting a saddle point always produces a plateau of saddle points. Our results unify and extend prior landscape analyses, elucidating when and how model expansion preserves or alters the nature of stationary points. These findings offer new geometric insights into the effects of width expansion and reparameterization in neural networks. 

---
# Measuring What Matters: Synthetic Benchmarks for Concept Bottleneck Models 

**Authors**: Julian Skirzynski, Harry Cheon, Shreyas Kadekodi, Meredith Stewart, Berk Ustun  

**Link**: [PDF](https://arxiv.org/pdf/2606.04326)  

**Abstract**: Concept bottleneck models predict outcomes from high-level concepts detected in inputs. Although concepts provide a simple way to reap benefits from interpretability, very few datasets include concept labels. This limits researchers' ability to determine which problems are suitable for these models, isolate the factors that drive their performance or lead to failures, or uncover which algorithms perform well. In this paper, we develop synthetic benchmarks for concept-bottleneck models, focusing on their two main use cases: decision support, in which models assist humans in making better decisions, and automation, in which models handle routine tasks without supervision. Our benchmarks can generate labeled datasets while controlling for properties that affect performance, including data modality, concept choice, annotation quality, and completeness. We demonstrate how the benchmarks can be used to evaluate representative classes of concept bottleneck models. Our demonstrations show how the benchmarks can diagnose failure modes and guide follow-up testing. 

---
# OpenRFM: Dissecting Relational In-Context Learning 

**Authors**: Zhikai Chen, Junyu Yin, Jialiang Gu, Siheng Xiong, Xiaoze Liu, Ruowang Zhang, Keren Zhou, Kai Guo  

**Link**: [PDF](https://arxiv.org/pdf/2606.04320)  

**Abstract**: Relational Foundation Models (RFMs) promise a single pre-trained predictor that, given any relational database, returns predictions in one forward pass via relational in-context learning (ICL). Yet a substantial gap separates open RFMs from their commercial counterparts, and the origin of this gap has not been systematically understood. We dissect a representative framework, the Relational Transformer (RT), from two perspectives. Model side: we show that RT performs relation-level ICL, and a kernel regression view shows it fails when sparse label-cell coverage yields an underdetermined regression. Data side: we ablate RT's pre-training source and find that existing synthetic-only pre-training and in-distribution pre-training drive the same architecture into different regimes, lazy vs. feature-learning. Probing this gap reveals that the missing ingredient is a support-identifiable relational latent in the label-generation process. These two diagnoses translate into (1) a dual-stage ICL architecture that combines the relational backbone with a batch-level ICL layer lifted from a pre-trained tabular foundation model to overcome relation-level label scarcity, and (2) a homophily-aware synthetic plus continual real-data pre-training mixture, augmented with a prototype-based regularization. These choices define OpenRFM, a simple yet effective RFM that improves average task performance by approximately 30% over the RT backbone and surpasses the commercial model KumoRFMv1 on a large set of evaluation tasks. 

---
# Anycast Performance in Context 

**Authors**: Eric Liang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04298)  

**Abstract**: IP anycast lets a service advertise one address from many physical sites, leaving BGP to map each client to a site. It is central to the DNS root server system, public resolvers, and some content delivery networks, yet the same routing mechanism has very different consequences across applications. This paper compares anycast latency in two settings: root DNS, where recursive caching amortizes root-server delay over many users and long time-to-live values, and CDNs, where each additional round trip can directly affect page-load, video-start, or API latency. The synthesis finds that root DNS anycast can exhibit substantial path inflation while still producing limited user-visible delay, whereas CDN anycast requires active engineering of peering, route policy, catchment scope, and measurement feedback to keep inflation small. The paper contributes a comparative latency model, a reproducible measurement design, and an optimization framework that separates resilience-driven anycast objectives from latency-driven objectives. The central conclusion is practical: operators should not optimize root DNS and CDN anycast with the same objective function. For root DNS, robustness, reachability, and cache behavior dominate; for CDN services, tail latency, catchment correctness, and policy control dominate. 

---
# Scaling Novel Graph Generation via Lightweight Structure-Guided Autoregressive Models 

**Authors**: Alessio Barboni, Massimiliano Lupo Pasini, Bishal Lakha, Edoardo Serra  

**Link**: [PDF](https://arxiv.org/pdf/2606.04287)  

**Abstract**: Generating realistic and diverse graphs is a key problem in machine learning, with applications in molecular discovery, circuit design, cybersecurity, and beyond. However, current graph generative models remain limited by scalability and novelty. Diffusion-based methods often require costly full-adjacency operations and long denoising chains, while many autoregressive and hybrid models have at least quadratic complexity. In addition, these models often imitate training graphs rather than generalize beyond them.
We propose a lightweight autoregressive framework to address these issues. It uses a structure-guided topological ordering to serialize graphs into regular edge sequences, enabling near log-linear generation, and a two-phase training strategy that combines exploration-oriented augmentation with iterative refinement to reduce overfitting and promote controlled novelty.
Experiments on molecular and non-molecular benchmarks show that our approach improves novelty while preserving high validity and uniqueness. The framework also supports both LSTM and Mamba-style causal sequence backbones, with large-memory accelerators enabling longer graph-sequence experiments beyond typical GPU limits. 

---
# Sparse Mixture-of-Experts Reward Models Learn Interpretable and Specialized Experts for Personalized Preference Modeling 

**Authors**: Yifan Wang, Jinyi Mu, Mayank Jobanputra, Yu Wang, Ji-Ung Lee, Soyoung Oh, Isabel Valera, Vera Demberg  

**Link**: [PDF](https://arxiv.org/pdf/2606.04284)  

**Abstract**: Preference modeling plays a central role in reinforcement learning from human feedback (RLHF), enabling large language models (LLMs) to align with human values. However, most existing approaches assume a universal reward function, neglecting the diversity and heterogeneity of human preferences. To address this limitation without additional annotation costs, recent work has proposed learning multiple preference components from binary data and combining them to model individual preferences. Nevertheless, these components often fail to capture coherent and disentangled patterns, limiting their interpretability and effectiveness for personalization. In this work, we propose a sparse Mixture-of-Experts (MoE) reward model that encourages sparse routing and expert diversity during training on binary preference data. Across controlled and real-world experiments, sparse MoE learns interpretable routing patterns and specialized experts. It also improves test-time personalization, and post-adaptation shifts in expert weights provide a qualitative lens for analyzing how the model adapts to personalized preferences. 

---
# The Loss Is Not Enough: Sampling Conditions and Inductive Bias in Contrastive Representation Learning 

**Authors**: Justinas Zaliaduonis, Patrick Putzky, Till Richter, Sergios Gatidis  

**Link**: [PDF](https://arxiv.org/pdf/2606.04280)  

**Abstract**: Contrastive learning has become a leading paradigm for self-supervised representation learning, yet the conditions under which it recovers meaningful latent geometry remain incompletely understood. We develop a measure-theoretic framework formalizing the diversity condition, a support requirement on positive-pair sampling that is necessary for isometric latent recovery. We show that the standard full-support von Mises-Fisher setting implies the satisfaction of the diversity condition and as a consequence global contrastive loss minimizers recover latent geometry up to orthogonal transformation, while restricted conditionals can make non-orthogonal maps attain strictly lower asymptotic contrastive loss. We introduce a support-corrected Information Noise Contrastive Estimation (InfoNCE) variant as a theoretical fix: this correction makes orthogonal latent space recovery achievable but does not uniquely select it. Experiments on synthetic benchmarks validate the identifiability predictions, and CIFAR-10 experiments are consistent with the qualitative prediction that architectural inductive bias becomes more important when sampling diversity is limited. Together, our results clarify how sampling mechanisms and encoder inductive bias interact in contrastive representation learning. 

---
# From Ticks to Flows: Dynamics of Neural Reinforcement Learning in Continuous Environments 

**Authors**: Saket Tiwari, Tejas Kotwal, George Konidaris  

**Link**: [PDF](https://arxiv.org/pdf/2606.04275)  

**Abstract**: We present a novel theoretical framework for deep reinforcement learning (RL) in continuous environments by modeling the problem as a continuous-time stochastic process, drawing on insights from stochastic control. Building on previous work, we introduce a viable model of actor-critic algorithm that incorporates both exploration and stochastic transitions. For single-hidden-layer neural networks, we show that the state of the environment can be formulated as a two time scale process: the environment time and the gradient time. Within this formulation, we characterize how the time-dependent random variables that represent the environment's state and estimate of the cumulative discounted return evolve over gradient steps in the infinite width limit of two-layer networks. Using the theory of stochastic differential equations, we derive, for the first time in continuous RL, an equation describing the infinitesimal change in the state distribution at each gradient step, under a vanishingly small learning rate. Overall, our work provides a novel nonparametric formulation for studying overparametrized neural actor-critic algorithms. We empirically corroborate our theoretical result using a toy continuous control task. 

---
# StandardE2E: A Unified Framework for End-to-End Autonomous Driving Datasets 

**Authors**: Stepan Konev  

**Link**: [PDF](https://arxiv.org/pdf/2606.04271)  

**Abstract**: Autonomous driving has shifted from modular perception-prediction-planning stacks toward end-to-end (E2E) models that map sensor inputs directly to vehicle control, often regularized by auxiliary tasks such as 3D detection, motion forecasting, and HD-map perception. Progress is driven by a fast-growing ecosystem of sensor-rich driving datasets, yet each ships its own file formats, APIs, coordinate conventions, and modality coverage, leaving cross-dataset experimentation and even basic per-dataset preprocessing to be re-implemented per project. We present StandardE2E, a framework that provides a single unified interface over E2E driving datasets. StandardE2E (i) standardizes per-dataset preprocessing under one shared data schema; (ii) combines multiple datasets in a single PyTorch DataLoader for cross-dataset pretraining, auxiliary-task supervision, and scenario-level filtering; and (iii) reduces adding a new dataset to a single per-dataset mapping from raw frames to the canonical schema, leaving the entire downstream pipeline unchanged. The framework supports six datasets out of the box: Waymo End-to-End, Waymo Perception, Argoverse 2 Sensor, Argoverse 2 LiDAR, NAVSIM (OpenScene-v1.1), and WayveScenes101, and is released as the open-source standard-e2e Python package, available at this https URL. 

---
# Instant-Fold: In-Context Imitation Learning for Deformable Object Manipulation 

**Authors**: Yilong Wang, Cheng Qian, Edward Johns  

**Link**: [PDF](https://arxiv.org/pdf/2606.04269)  

**Abstract**: Deformable object manipulation (DOM) is challenging due to high-dimensional, partially observable states that evolve through long-horizon, topology-changing interactions with multiple valid manipulation modes. We introduce Instant-Fold, an in-context imitation learning framework for DOM. Given a single human demonstration, our policy infers and executes diverse manipulation modes directly from the demonstration, including variations in spatial execution and ordering, without requiring gradient updates. Our approach first learns deformation-aware visual representations via temporal contrastive pretraining, after which a flow-matching transformer policy conditioned on the demonstration predicts actions to execute the intended manipulation mode. Trained entirely in simulation, Instant-Fold generalizes across diverse folding modes and transfers zero-shot to real-world settings without additional data collection or finetuning. Videos are available at this https URL. 

---
# Can I Take Another Dose? Evaluating LLM Decision-Making Under Temporal Uncertainty in OTC Dosing QA 

**Authors**: Maroof Kousar, Yibo Hu  

**Link**: [PDF](https://arxiv.org/pdf/2606.04262)  

**Abstract**: Large language models (LLMs) are increasingly used for everyday health questions, including whether a user can safely take another dose of an over-the-counter (OTC) medication. Yet this common safety-relevant setting remains underexplored in existing medical QA evaluations, where correct answers require tracking dose timing, computing rolling 24-hour intake, following product-label constraints, and handling incomplete medication histories. We introduce DOSEBENCH, a focused benchmark of 81 curated OTC dosing scenarios focused on adult acetaminophen and ibuprofen use, with manually annotated gold references. We evaluate four LLMs across repeated runs using metrics for decision correctness, consistency, explanation verifiability, failure types, and confidence-related signals, resulting in 1,620 model responses. Our results show that models frequently struggle with rolling-window reasoning and ambiguity-sensitive cases and that stable or confident-looking responses can still violate dosing constraints. These findings suggest that OTC dosing QA provides a narrow yet practical testbed for evaluating temporal reasoning, constraint following, and safety-relevant uncertainty handling in medical QA. 

---
# Overview of the EReL@MIR 2025 Multimodal Document Retrieval Challenge (Track 1) 

**Authors**: Jingbiao Mei  

**Link**: [PDF](https://arxiv.org/pdf/2606.04240)  

**Abstract**: Retrieval over visually-rich documents, pages that interleave text with figures, tables, and charts, is essential for multimodal retrieval-augmented generation, yet most retrievers still discard the visual channel. The \emph{Multimodal Document Retrieval Challenge}, Track~1 of the MIR Challenge at the first EReL@MIR workshop, co-located with The Web Conference 2025, asks participants to build a \emph{single} retrieval system that handles two complementary regimes: closed-set document page retrieval within long documents from a text query (MMDocIR), and open-domain retrieval of Wikipedia-style passages from an image or image-plus-text query (M2KR). Systems are ranked by the macro-average of mean Recall@$\{1,3,5\}$ over the two tasks. The challenge drew 455 entrants and 586 submissions across 22 teams. This report describes the challenge design, datasets, and evaluation protocol; reports the final standings; and analyses the three winning teams' systems. All three build on decoder-based Multimodal-LLM embedders from the Qwen2-VL family rather than on CLIP-style encoders, and differ chiefly in whether they reach the top through fine-tuned ensembles, training-free multi-route fusion with a strong vision-language re-ranker, or zero-shot late interaction. The training-free system finished within $0.1$ point of the fine-tuned winner. 

---
# Recover-LoRA for Aggressive Quantization: Reclaiming Accuracy in 2-Bit Language Models via Low-Rank Adaptation with Knowledge Distillation on Synthetic Data 

**Authors**: Devleena Das, Rajeev Patwari, Elliott Delaye, Ashish Sirasao  

**Link**: [PDF](https://arxiv.org/pdf/2606.04238)  

**Abstract**: Aggressive weight quantization to 2-bit precision offers substantial throughput and memory gains for large language model (LLM) inference, but typically incurs severe accuracy degradation. These gains are particularly relevant for edge and on-device deployment, where memory capacity and bandwidth are primary constraints. In this work, we extend Recover-LoRA -- a lightweight, data-free accuracy recovery method originally developed for general model weight corruption -- to the setting of ultra-low-bit quantization. We propose a selective mixed-precision strategy in which only gate and up projection layers of the MLP are quantized to 2-bit (W2), while all other linear layers remain at higher precision, yielding a mixed-precision GateUp configuration. We demonstrate via roofline analysis across three model families (4B--20B) and two hardware platforms that a W4/W2-GateUp deployment (4-bit base with 2-bit gate/up) delivers 7.5--23.3\% TPS improvement over uniform W4 depending on model and context length, while confining quantization error to a predictable subset of layers. We then apply Recover-LoRA -- training low-rank adapters on the quantized layers via logit distillation with synthetic data -- to recover accuracy lost from 2-bit quantization of the gate and up layers. In a case study on Qwen3-4B, Recover-LoRA achieves 80--95\% accuracy recovery on 9 of 12 benchmarks, using only 10k synthetic training samples and no labeled data. We further demonstrate that synthetic data performs comparably to curated labeled data for distillation-based recovery, and that recovery generalizes to out-of-distribution evaluation tasks. Our results present Recover-LoRA as a practical post-quantization accuracy recovery tool for aggressive weight compression in deployment settings. 

---
# Supportive Token Revealing for Fast Diffusion Language Model Decoding 

**Authors**: Giries Abu Ayoub, Mario Barbara, Lluís Pastor-Pérez, Tanja Bien, Aneesh Barthakur, Alaa Maalouf, Loay Mualem  

**Link**: [PDF](https://arxiv.org/pdf/2606.04236)  

**Abstract**: Discrete diffusion language models can generate text efficiently by updating multiple masked positions in parallel, but this parallelism introduces a quality-latency trade-off. Aggressive decoding may commit mutually dependent tokens too early, while conservative decoding requires many denoising steps. Existing methods address this tension by deciding which tokens are safe to reveal using confidence or dependency criteria. However, avoiding unsafe commits does not necessarily make the remaining masked sequence easy to decode, since uncertain tokens may depend on masked tokens, creating a bottleneck for denoising steps. We propose AXON, a training-free module that can be added on top of existing parallel decoding strategies for diffusion language models. Rather than replacing the base decoder, AXON monitors the remaining uncertain masked tokens and intervenes only when their current state suggests that additional context is needed. It then shifts the criterion from which tokens are safest to reveal to which confident reveals would best support later denoising. AXON selects anchors, confident masked tokens that uncertain positions attend to, using attention, uncertainty, and confidence signals. Experiments on reasoning and code-generation benchmarks across multiple diffusion language models show that AXON improves the quality-latency trade-off of existing parallel decoders, often reducing the number of function evaluations while maintaining or improving accuracy. 

---
# MM-BizRAG: Rethinking Multimodal Retrieval-Augmented Generation for General Purpose Enterprise Q&A 

**Authors**: Hanoz Bhathena, Parin Rajesh Jhaveri, Rohan Mittal, Prateek Singh, Aymen Kallala, Rachneet Kaur, Yiqiao Jin, Zhen Zeng, Adwait Ratnaparkhi, Denis Kochedykov  

**Link**: [PDF](https://arxiv.org/pdf/2606.04231)  

**Abstract**: Recent advances in multimodal retrieval-augmented generation (MM-RAG) have shifted toward minimal parsing, relying on page-level images for producing retriever embeddings and for answer generation. While efficient, this trend often neglects explicit handling of the rich, structured information in complex enterprise documents, instead depending on pre-trained embeddings or vision-language models to implicitly capture such structure. In this work, we take a more direct approach: MM-BizRAG proactively extracts and represents document structure via a document structure-aware split that dynamically routes documents through orientation-specific ingestion pipelines, applying explicit layout-aware parsing for vertically structured documents (e.g., reports) and holistic page-level representations for horizontally structured documents (e.g., slide decks). A unified LLM-driven artifact transformation pipeline with placeholder-based positional alignment preserves natural reading order, while inference-time multimodal assembly decouples retrieval representations from generation context, enabling richer, more grounded answers without any finetuning requirement. Through experiments on a large, heterogeneous enterprise dataset and two public benchmarks (SlideVQA and FinRAGBench-V), MM-BizRAG consistently outperforms state-of-the-art vision-centric baselines by up to 32% points, with especially strong gains on report-style layouts. Furthermore, we introduce FastRAGEval, a single-call LLM Judge metric for fine-grained generative recall that halves RAGChecker's cost while achieving stronger human alignment. 

---
# Incremental Sheaf Cohomology on Cellular Complexes: O(1)-in-n Lazy Edit Processing under Bounded Local Geometry 

**Authors**: Jason L. Volk  

**Link**: [PDF](https://arxiv.org/pdf/2606.04227)  

**Abstract**: We present an algorithmic framework for incremental maintenance of first sheaf cohomology $H^1(X; \mathcal{F})$ on dynamically evolving 1-dimensional cellular complexes equipped with finite-dimensional cellular sheaves. The classical computation of $H^1$ via factorization of the coboundary matrix requires $O(n^3)$ time; when the complex evolves with a stream of $m$ edits, full recomputation after each edit costs $O(mn^3)$.
Under a bounded local geometry assumption -- bounded cell size $v_{\max}$, bounded stalk dimension $d$, and bounded nerve degree $D$ -- each edit (vertex insertion, edge insertion, restriction map update) affects only a bounded set of local coboundary blocks. The algorithm therefore processes lazy streaming edits in $O(1)$ time with respect to the total complex size $n$ (with cost polynomial in the local geometry parameters $v_{\max}$, $d$, and $D$, which are treated as constants independent of $n$), deferring local eigensolves and Mayer-Vietoris global assembly to synchronization points (Flush). At synchronization, the maintained state agrees with the corresponding batch assembly of the partitioned sheaf model; we observe zero measured drift in all batch-verified runs (through $V = 10^6$). We also give an amortized $O(|E|)$ streaming construction for the cellular decomposition and discuss an adversarial algebraic-RAM barrier arguing that unpartitioned non-trivial sheaves ($d \geq 2$, non-identity restriction maps) do not admit the same locality. Experiments on Barabasi-Albert graphs with up to $5 \times 10^6$ vertices and $1.7 \times 10^7$ streaming edits show 35 $\mu$s median lazy per-edit update latency (excluding flush); query time (global assembly at synchronization) is $O(n)$ per flush in the implemented full-traversal path. Exact synchronization costs are reported separately. 

---
# PerceptTwin: Semantic Scene Reconstruction for Iterative LLM Planning and Verification 

**Authors**: Charlie Gauthier, Sacha Morin, Liam Paull  

**Link**: [PDF](https://arxiv.org/pdf/2606.04226)  

**Abstract**: Simulation environments are useful for both robot policy learning and planning verification and validation. Traditionally, the process of creating a simulation was onerous. Creating a bespoke simulation environment for each individual environment that a robot would operate in was simply infeasible. In this work, we introduce PerceptTwin, a fully automatic pipeline that constructs interactive simulations directly from semantic scene representations produced by a robot's perception stack. PerceptTwin combines open-vocabulary object maps with 3D asset generation, affordance prediction, and commonsense condition checking. These interactive simulations can be used to validate and refine plans before they are executed on the robot hardware. Borrowing from the AI alignment literature, we also introduce an LLM judge that verifies plan correctness and alignment with human preferences. Experiments show that PerceptTwin feedback allows LLM planners to refine plans, enhance safety, and resist harmful black-box prompting attacks. In our suite of tasks, PerceptTwin improves plan success by an average of approximately 39% for GPT5, GPT5Mini, and GPT5Nano planners. Additionally, PerceptTwin also improves human plan verification by up to 18% on average for plans that fail due to unfilled skill preconditions. Our results demonstrate the potential of open-vocabulary scene simulation from robot perception as a foundation for safer, more reliable robot planning. 

---
# DetectZoo: A Unified Toolkit for AI-Generated Content Detection Across Text, Audio, and Image Modalities 

**Authors**: Sajad Ebrahimi, Nima Jamali, Bardia Shirsalimian, Kelly McConvey, Wentao Zhang, Jalehsadat Mahdavimoghaddam, Maksym Taranukhin, Maura Grossman, Vered Shwartz, Yuntian Deng, Ebrahim Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2606.04205)  

**Abstract**: The growing popularity and capacity of generative models have eroded the distinction between human and machine-generated content, motivating a growing body of work on detection across text, images, and audio. Most available detectors are either commercial software or, if open-source, come with incompatible codebases with bespoke preprocessing, evaluation protocols, and evaluation metrics, which make their adoption, fair comparison, and reproduction quite difficult. To address this critical gap, we introduce DetectZoo, a first-of-its-kind, extensible toolkit designed to provide a unified interface for AI-generated content detection across text, audio, and image modalities. DetectZoo standardizes the complete empirical pipeline, from data ingestion and preprocessing to model assessment, offering researchers a cohesive framework to benchmark state-of-the-art detectors systematically. By integrating diverse public datasets and baseline detection algorithms under a single, unified API, our toolkit facilitates rigorous and reproducible evaluation. DetectZoo provides reference implementations of 61 detectors, native loaders for 22 benchmark datasets, and a standardized evaluation pipeline that reports multiple metrics through a common interface. Each detector is self-contained yet accessible through the same interface, automatically caches pretrained weights, and reproduces the original published results. DetectZoo lowers the barrier to entry for multi-modal AI forensics, enabling researchers to identify performance gaps across domains and accelerating the development of robust, generalizable detection techniques. The open-source repository and comprehensive documentation are publicly available at this https URL, and the package can be installed via pip install detectzoo. 

---
# Notarized Agents: Receiver-Attested Confidential Receipts for AI Agent Actions 

**Authors**: Juan Figuera  

**Link**: [PDF](https://arxiv.org/pdf/2606.04193)  

**Abstract**: Current AI agent observability is structurally compromised: the entity producing the activity log is the same entity whose activity is being logged. A compromised or buggy agent can omit, alter, or fabricate its own traces, and the operator running the agent has no independent way to detect tampering. We propose a class of protocols that resolves this by inverting the trust boundary: the service that receives an agent's call signs a receipt of what it observed using its own key, encrypts the receipt to the agent's owner, and publishes it to a public transparency log. The owner reconstructs a tamper-evident trail without trusting the agent or its operator. We instantiate the class as Sello, a protocol combining four properties absent in any current system: (P1) receiver-side signing, (P2) HPKE encryption to an owner public key bound to the authorization token via JWS, (P3) publication to a witness-cosigned Merkle log, and (P4) owner-side discovery by token reference. We describe the protocol, analyze its security under an adversary that controls the agent and its operator, present microbenchmarks of the cryptographic operations, and situate Sello among adjacent receipt-protocol work (Signet, AgentROA, Agent Passport System, draft-farley-acta, SCITT). We discuss known limitations including the suppression attack, service collusion, and the adoption-incentive problem. 

---
# Metric-Aware Hybrid Forecasting for the CTF4Science Lorenz Challenge 

**Authors**: Cen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2606.04191)  

**Abstract**: We describe our approach to the CTF4Science Lorenz challenge, a benchmark that mixes short-horizon forecasting, long-time distribution matching, and trajectory reconstruction across nine task pairs. The key discovery is that no single model family dominated all metrics. Instead, we built a metric-aware hybrid system that assigned a different predictor to each metric family: (1) synthetic-pretrained denoisers for full-trajectory reconstruction, (2) Lorenz ODE fitting and trajectory shooting for the first 20 forecast steps, and (3) histogram-tail substitution using synthetic Lorenz libraries for long-time evaluation. A representative mature submission from this system family scored 83.83551 on the public leaderboard, and a small follow-up stack of the same ideas reached 83.85529. We focus on the cleaner intermediate system because it captures the full method while remaining simple enough to reproduce and analyze, while the final submission can be understood as a conservative extension of the same backbone. 

---
# Dual Advantage Fields 

**Authors**: Alexey Zemtsov, Maxim Bobrin, Alexander Nikulin, Dmitry V. Dylov, Fakhri Karray, Vladislav Kurenkov, Martin Takáč, Arip Asadulaev  

**Link**: [PDF](https://arxiv.org/pdf/2606.04188)  

**Abstract**: Offline goal-conditioned reinforcement learning requires both long-horizon reachability estimates and local action comparisons. Dual goal representations provide value fields that capture global goal reachability, but they do not directly specify which action should be preferred at a given state. We propose Dual Advantage Fields, a policy-extraction method that turns a bilinear dual value model into a local advantage signal. Under bilinear dual parameterization, the goal embedding is the gradient of the value field with respect to the state representation. DAF learns an action-effect model that predicts the discounted feature displacement induced by an action and scores actions by the alignment between this displacement and the goal direction. In the realizable case, this score equals the goal-conditioned Bellman advantage, yielding a standard local policy-improvement guarantee. On OGBench locomotion, manipulation, and puzzle tasks, DAF improves aggregate RLiable metrics and performs strongly in settings where locally correct actions differ from direct movement toward the final goal. 

---
# Exact Unlearning in Reinforcement Learning 

**Authors**: Thanh Nguyen-Tang, Raman Arora  

**Link**: [PDF](https://arxiv.org/pdf/2606.04182)  

**Abstract**: We formulate the problem of \emph{exact unlearning} in reinforcement learning, where the goal is to design an efficient framework that enables the removal of any user's data upon deletion request, i.e., the online learner's output after unlearning is \emph{indistinguishable} from what would have been produced had the deleted user never interacted with the learner. For any $\rho >0$, we show that there exists a reinforcement learning (RL) algorithm that is $\rho$-TV-stable and supports an exact unlearning procedure whose expected computational cost is only a $\rho \sqrt{\ln T}$ fraction of the computational cost of retraining from scratch. We construct such a $\rho$-TV-stable RL algorithm for tabular Markov decision processes (MDPs), which achieves a regret bound of $\mathcal{O}(H^2 \sqrt{SAT} + H^3 S^2 A + {H^{2.5} S^2 A}/{\rho})$, where $S, A, H$, and $T$ denote the number of states, the number of actions, the episode horizon, and the number of episodes, respectively. We also establish a lower bound of $\Omega(H\sqrt{\!SAT}\! +\! {SAH}/{\rho})$ for $\rho$-TV-stable RL algorithms, showing that our algorithm is nearly minimax optimal. 

---
# A Systematic Analysis of Linguistic Features in AI-Generated Text Detection Across Domains and Models 

**Authors**: Yassir El Attar, Esra Dönmez, Maximilian Maurer, Agnieszka Falenska  

**Link**: [PDF](https://arxiv.org/pdf/2606.04177)  

**Abstract**: Interpretable linguistic features offer a promising approach for explaining why a given text appears machine-generated, particularly for non-expert users. However, existing findings on which features reliably indicate LLM-generated text remain fragmented across feature sets, models, and text domains. To address this gap, we conduct a large-scale empirical study assessing the robustness of linguistic signals for characterizing AI-generated text. Our analysis covers 284 interpretable linguistic features across outputs from 27 LLMs and ten text domains under cross-model and cross-domain generalization settings. We show that classifiers based solely on linguistic features can reliably distinguish AI-generated from human-written text. However, many previously proposed indicators prove strongly context-dependent, with the exception of measures of lexical richness, which remain robust signals across model families and text domains. These results demonstrate which linguistic signals generalize across contexts and provide a foundation for more reliable, interpretable analyses of AI-generated language. 

---
# MimeLens: Position-Agnostic Content-Type Detection for Binary Fragments 

**Authors**: Michael J. Bommarito II  

**Link**: [PDF](https://arxiv.org/pdf/2606.04171)  

**Abstract**: File-type classification underlies many workflows like malware triage, forensic carving, packet inspection, and storage indexing. Learned systems such as Google's Magika assume whole-file access at a known offset, so they break on the inputs many of these tasks actually produce, like a single packet payload, a header-less carved fragment, a random disk block, or a chunked upload. We introduce MimeLens, a family of small BERT-style encoders pretrained on binary content from windows sampled at a uniformly random offset within each file, with no privileged head-of-file position, in standard- and short-context variants. A byte chunk goes in from anywhere in a file, no header needed and no fixed size; out comes one of libmagic's 125 MIME labels. On the clean head of complete files, MimeLens beats Magika v1.1 by +10.7 pp top-1 on libmagic-labeled data, and it keeps classifying where Magika cannot: from a single mid-stream UDP packet, and more than twice as accurately as libmagic and Magika on random mid-file disk blocks. The cost is latency: MimeLens runs roughly one to two orders of magnitude slower per sample on CPU than Magika, though it matches on consumer GPUs or in batch. All trained checkpoints are released on Hugging Face (mjbommar/mimelens-001-*). 

---
# Smart Transportation Without Neurons -- Fair Metro Network Expansion with Tabular Reinforcement Learning 

**Authors**: Dimitris Michailidis, Sennay Ghebreab, Fernando P. Santos  

**Link**: [PDF](https://arxiv.org/pdf/2606.04167)  

**Abstract**: We tackle the Metro Network Expansion Problem (MNEP), a subset of the Transport Network Design Problem (TNDP), which focuses on expanding metro systems to satisfy travel demand. Traditional methods rely on exact and heuristic approaches that require expert-defined constraints to reduce the search space. Recently, deep reinforcement learning (Deep RL) has emerged due to its effectiveness in complex sequential decision-making processes-it remains, however, computationally expensive, environmentally costly, and requires additional engineering to interpret. We show that MNEP problems are small enough to not require Deep RL methods. Reformulating the MNEP as a Non-Markovian Rewards Decision Process (NMRDP), we use tabular RL to achieve similar performance with significantly fewer training episodes, additionally offering greater interpretability. Additionally, we incorporate social equity criteria into the reward functions, focusing on efficiency and fairness, highlighting the versatility of our method. Evaluated in real-world settings-Xi'an and Amsterdam-our method reduces total episodes by a factor of 18 and total carbon emissions by a factor of 12 on average, while remaining competitive with Deep RL. This approach offers a replicable, modular, interpretable, and resource-efficient solution with potential applications to other combinatorial optimization problems. 

---
# ADAPTOOD: Uncertainty-Aware Fine-Tuning for Out-of-Distribution ECG Time Series Models 

**Authors**: Sotirios Vavaroutas, Yu Yvonne Wu, Ali Etemad, Cecilia Mascolo  

**Link**: [PDF](https://arxiv.org/pdf/2606.04164)  

**Abstract**: Data samples used for training often differ from those encountered during fine-tuning and deployment, and while ML models show promise, their performance remains limited when only small annotated datasets are available. Performance often degrades under distribution shifts caused by diverse sensors, populations, and application settings. Although pre-training helps, models frequently encounter out-of-distribution (OOD) data in real-world settings, leading to reduced robustness. Existing adaptation methods usually assume fixed distribution shifts and struggle when multiple types or severities occur. In particular, they overlook shift severity, for example treating adaptation to a large familiar dataset the same as adaptation to a small dataset with a new task, which limits generalisation. To address this, we propose ADAPTOOD, a novel framework that leverages data uncertainty to quantify distribution shift severity and guide fine-tuning for time series. This uncertainty measures how strongly samples from the target deployment distribution deviate from the pre-training distribution, providing a direct signal of OOD severity. Our framework combines this uncertainty with low-rank model updates and adaptive hyperparameter optimisation to improve adaptation. We show that ADAPTOOD achieves up to 7% higher accuracy and 12.9% higher precision than existing methods in OOD tasks, maintaining strong performance as distribution shift severity increases. 

---
# EvalStop: Using World Feedback to Detect and Correct Reward Overoptimization in Multi-Tenant RLHF Platforms 

**Authors**: Guilin Zhang, Chuanyi Sun, Shahryar Sarkani, John M. Fossaceca  

**Link**: [PDF](https://arxiv.org/pdf/2606.04145)  

**Abstract**: Cloud LLM fine-tuning platforms increasingly serve RLHF workloads, where a learned reward model is optimized as a proxy for human quality. As Gao et al. (2023) showed, this proxy diverges from world feedback (downstream eval metrics) under sustained optimization pressure, a phenomenon known as reward overoptimization. Existing platform schedulers ignore this divergence: non-clairvoyant schedulers optimize JCT without any quality signal, SLAQ-style quality-aware schedulers use training loss (a weaker proxy that drops monotonically through hacking), and classical per-job early stopping requires human monitoring and does not free shared GPUs. We propose EvalStop, a composable scheduling primitive that terminates jobs on k consecutive eval-score declines, releases GPUs, preserves the best checkpoint, and delegates to any base scheduler. We frame scheduler-level early stopping as a detection problem and evaluate it in a discrete-event simulator whose RLHF workload mixes reward-hacking and structurally healthy runs, with ground-truth labels hidden from schedulers. On RLHF-heavy workloads (80% RLHF, 64 GPUs), EvalStop achieves precision 98% / recall 99% / FPR 1.5% while improving JCT by 9% and cutting wasted compute by 22% over SRTF-Est (p<0.05). Trivial fixed-progress and loss-plateau competitors either incur 65% FPR on healthy RLHF or miss over half of true hacking cases. Gains compose across every base scheduler tested (9-25% JCT) and detection quality stays stable under eval noise (precision at least 91% at noise std <= 0.05) and hacking base rate (precision at least 89% across 20-80% hacking fractions). 

---
# Physics-Informed Machine Learning for Short-Term Flood Prediction 

**Authors**: Tewodros Syum Gebre, Jagrati Talreja, Leila Hashemi-Beni  

**Link**: [PDF](https://arxiv.org/pdf/2606.04143)  

**Abstract**: Accurate flood forecasting is essential for mitigating disaster risks and protecting communities. However, purely data-driven machine learning models often struggle in data-scarce environments and may violate fundamental hydrological principles. Standard Long Short-Term Memory (LSTM) networks can generate physically inconsistent predictions, particularly when extrapolating to extreme weather conditions. To address these limitations, we propose a Physics-Informed Machine Learning (PIML) framework that incorporates hydrological knowledge directly into the loss function of an LSTM model. Specifically, a Trend Alignment constraint penalizes directional inconsistencies between precipitation and discharge trends, improving model robustness without requiring complex hydrodynamic equations. This regularization encourages the model to learn physically plausible hydrograph behavior, even with limited training data, while enhancing reliability during peak flood events. Experimental results show that the proposed physics-informed model outperforms a standard LSTM baseline in data-scarce settings, increasing the Nash-Sutcliffe Efficiency (NSE) from 0.20 to 0.23 when trained on only 5% of the available data. Additional stress tests under simulated extreme climate scenarios demonstrate that the baseline model exhibits unstable behavior, whereas the physics-informed model maintains directional consistency and physical plausibility. Although accurately predicting extreme peak magnitudes remains challenging with limited data, the proposed approach substantially reduces unphysical fluctuations common in purely data-driven models. These findings demonstrate that simple physical constraints can significantly improve the reliability of deep learning models for real-time flood forecasting, offering a practical solution for ungauged basins and evolving climate conditions. 

---
# Caught in the Act(ivation): Toward Pre-Output and Multi-Turn Detection of Credential Exfiltration by LLM Agents 

**Authors**: Kargi Chauhan, Pratibha Revankar  

**Link**: [PDF](https://arxiv.org/pdf/2606.04141)  

**Abstract**: LLM agents often place sensitive credentials in the same context window as untrusted retrieved content, creating a direct path for indirect prompt injection to induce credential exfiltration. We study this failure mode through three complementary defenses. First, we ask whether activation probes can detect credential access before output tokens are emitted. Second, we construct honeytokens from format-specific character models and calibrate detection with split conformal prediction. Third, we treat multi-turn exfiltration as a cumulative information-flow problem and track an estimated leakage budget across conversation turns. In controlled experiments on open-weight models, activation features separate benign and credential-seeking prompts with high accuracy, including under held-out encoding transformations. In a small synthetic multi-turn suite, cumulative accounting detects attacks that per-turn detectors miss. These results are preliminary: the multi-turn benchmark is in-house and small, the activation method requires white-box access, and the information estimator provides a practical signal rather than a formal upper bound. Still, the results suggest that credential-exfiltration defenses should combine pre-output monitoring, calibrated canary detection, and temporal leakage accounting rather than relying only on text-level output filters. 

---
# HighTide: An Agent-Curated Open-Source VLSI Benchmark Suite 

**Authors**: Benjamin Goldblatt, Paolo Pedroso, Farhad Modaresi, Ethan Sifferman, Matthew R. Guthaus  

**Link**: [PDF](https://arxiv.org/pdf/2606.04126)  

**Abstract**: We introduce HighTide, an evolving AI-assisted benchmark suite. Specifically, the contributions are: (i) a diverse open-source suite spanning multiple design languages and technology nodes, (ii) Bazel-based incremental RTL-to-GDS compilation with remote caching, (iii) AI-assisted design curation through twelve agent skills covering the design lifecycle, flow optimization, tool reference, and meta-maintenance, backed by per-design decision logs that serve as long-term memory of tuning rationale across the suite, and (iv) an infrastructure with RTL compilation verification for stable releases. The suite is publicly available and designed to grow with the open-source hardware ecosystem. 

---
# Semantic Constraint Synthesis for Adaptive Trajectory Optimization via Large Language Models 

**Authors**: Eleanor Brosius, Yuji Takubo, Daniele Gammelli, Simone D'Amico, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2606.04123)  

**Abstract**: Trajectory optimization is a critical component for enabling safe and reliable autonomous operations in space exploration. As space missions increase in frequency, complexity, and scope, there is a growing need to rapidly formulate mathematically sound trajectory optimization problems that accurately reflect mission objectives and operational constraints. However, translating mission intent into tractable analytical formulations for trajectory optimization requires substantial domain expertise. This paper presents a framework that leverages large language models (LLMs) to translate natural language descriptions of mission requirements and constraints into executable trajectory optimization code and corresponding mathematical formulations. Experiments in spacecraft rendezvous scenarios demonstrate a high success rate in reconditioning a convex trajectory optimization problem from semantic mission requirements. Ultimately, this work highlights the potential of LLMs to bridge high-level intent and formal optimization models, enabling more flexible and efficient trajectory design of spacecraft. 

---
# SaliMory: Orchestrating Cognitive Memory for Conversational Agents 

**Authors**: Kai Zhang, Xinyuan Zhang, Hongda Jiang, Shiun-Zu Kuo, Hyokun Yun, Ejaz Ahmed, Shereen Oraby, Ziyun Li, Sanat Sharma, Ann Lee, Ahmed A Aly, Anuj Kumar, Raffay Hamid, Xin Luna Dong  

**Link**: [PDF](https://arxiv.org/pdf/2606.04120)  

**Abstract**: Conversational agents that serve as lifelong companions must maintain persistent memory across all interactions. However, simply expanding context windows with raw retrieval degrades reasoning quality, while training memory agents via standard reinforcement learning creates a severe credit assignment bottleneck in a multi-stage pipeline. To solve this, we introduce SALIMORY, a framework that trains a single language model to manage a cognitively-structured memory-spanning user facts, preferences, and working memory. By introducing a hierarchical stage-wise process reward and reward-decomposed contrastive refinement, SALIMORY provides isolated supervision for distinct memory operations (selective filtering, consolidation, and cue-driven recall) end-to-end. SALIMORY cuts memory-attributed failures by one-third, outperforms the state-of-the-art by over 10% in end-to-end accuracy, and more than doubles the Good Personalization rate. 

---
# dMX: Differentiable Mixed-Precision Assignment for Low-Precision Floating-Point Formats 

**Authors**: Giuseppe Franco, Ian Colbert, Pablo Monteagudo-Lago, Felix Marty, Nicholas Fraser  

**Link**: [PDF](https://arxiv.org/pdf/2606.04115)  

**Abstract**: Quantizing large language models (LLMs) to low-precision floating-point representations is central to efficient deployment, yet applying a single bit-width uniformly across all layers is sub-optimal in terms of both performance and accuracy. This work introduces dMX, a differentiable mixed-precision quantization framework for learnable floating-point bit-width assignment. We study its application for the microscaling floating-point (MXFP) family of data types defined by the Open Compute Project (OCP) standard. The per-layer bit-width assignment is formulated as a continuous optimization problem in which each layer's floating-point format format is parameterized by a scalar parameter, folding the multi-variate design space into a single learnable offset. During training this offset takes continuous values, avoiding sudden oscillations between discrete quantization formats. A temperature-based annealing schedule progressively discretizes the learned offsets, ensuring that the final configuration maps to hardware-compatible MXFP formats without abrupt transitions between training and inference behavior. A target-aware regularization term steers the average bit-width toward a user-specified budget, serving as a coarse-grained proxy for inference cost and balancing model quality against deployment efficiency. We performed experiments on different families of LLM, such as Llama, Qwen3, and SmolLM2, evaluating perplexity on WikiText-2 and accuracy on four zero-shot reasoning benchmarks. Across these settings, dMX consistently yields Pareto-dominating models and improves over Kullback-Leibler (KL) divergence-based layer-selection heuristics, efficiently navigating trade-offs between model quality and average bit-width. 

---
# AgenticDiffusion: Agentic Diffusion-based Path Planning for Vision-Based UAV Navigation 

**Authors**: Faryal Batool, Muhammad Ahsan Mustafa, Fawad Mehboob, Valerii Serpiva, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2606.04111)  

**Abstract**: Indoor UAV navigation requires efficient exploration, scene understanding, and reliable trajectory execution under limited field-of-view observations. Existing vision-based navigation frameworks typically rely on single-view observations, limiting their ability to reason about occlusions, target visibility, and global scene structure. In this work, we propose AgenticDiffusion, a multi-view UAV navigation framework that coordinates language-guided reasoning, open-vocabulary target grounding, vision-based diffusion planning, and NMPC within a unified aerial navigation pipeline. Given a natural language instruction and synchronized first-person-view (FPV) and top-view observations, the framework determines the most informative viewpoint for navigation and generates a mission plan prior to trajectory execution. The targets are localized using an open-vocabulary grounding model, after which viewpoint-specific diffusion planners generate navigation trajectories for UAV execution. Using complementary viewpoints, the proposed framework reduces repeated target exploration and improves navigation efficiency in cluttered indoor environments. The framework was validated in four real-world UAV navigation scenarios involving adaptive viewpoint selection, multi-stage mission execution, long-horizon navigation, and safe landing-site selection. The experimental results demonstrated an overall mission success rate of 80% in 40 real-world trials, while the diffusion planners achieved a trajectory generation success rate of 100%. 

---
# SymTRELLIS: Symmetry-Enforced Voxel Latents for 3D Generation 

**Authors**: Guangda Ji, Qimin Chen, Qinchan Li, Mingrui Zhao, Kai Wang, Hao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04108)  

**Abstract**: Single-view 3D generative models have achieved impressive visual quality, yet they are not designed to satisfy structural or functional requirements, and in practice, often fall short. Symmetry is one such requirement: violations, even subtle ones, on symmetry can render a model physically unusable. We present SymTRELLIS, a method that enforces arbitrary finite point group symmetries (rotational, reflectional, and polyhedral) during the flow-based 3D generation of TRELLIS.2, without retraining the underlying VAE or flow model. Our key idea is to approximate the latent-space action of spatial transformations as a learned linear operator on voxel latents, implemented as a lightweight spatial-transform latent mapper trained on generic, non-symmetric 3D data. At generation time, we enforce symmetry by averaging predicted flow velocities across all symmetry-equivalent transformations at each ODE step, a process we call velocity symmetrization. The symmetry specification can be estimated automatically from an initial TRELLIS.2 generation or supplied by the user, enabling deliberate fold manipulation beyond what the input image suggests. On a curated benchmark of 266 strictly symmetric objects spanning 2- to 20-fold rotations and polyhedral symmetry groups, SymTRELLIS substantially reduces all symmetry error metrics compared to TRELLIS.2, Hunyuan3D-2.1, and TripoSG, while maintaining reconstruction accuracy comparable to the base model. 

---
# Building The Ph(ysical)AI Layer Of Machine Intelligence 

**Authors**: Ulbert Jose Botero, Liam Smith, Brooks Olney, Pooya Khorrami, Steven Kusiak, Watson Jia, Sage Trudeau, Daniel Capecci  

**Link**: [PDF](https://arxiv.org/pdf/2606.04106)  

**Abstract**: Foundation models achieve generalization through massive-scale training on diverse data, but have limitations with transfer to truly unseen domains without paired training data. We propose principle-driven foundation models that encode signal-theoretic principles (Fourier decomposition, energy conservation, symmetry) rather than learn untethered statistical correlations. We hypothesize that domains differ not in fundamental physics, but in learnable transformations in time, frequency, magnitude, or phase. Training exclusively on radio-frequency (RF) data with co-designed architecture and losses incorporating these principles, we achieve cross-modal transfer to audio, images, text, and video using only frozen representations learned from RF data, requiring no fine-tuning of the encoder on target domains. Our 1.99M parameter frozen encoder achieves 77.7% average accuracy (91.9% top-3) across 15 diverse tasks via linear probing, with systematic variation: 84.5 on physically-grounded tasks (speaker recognition, seismology, RF fingerprinting) versus 70.0% on semantic tasks (music genre, language recognition). This reveals that principle-driven and scale-driven approaches offer complementary paths: physical principles enable efficient cross-modal transfer while naturally establishing the boundary between physical and semantic understanding. 

---
# Proof-Carrying Agent Actions: Model-Agnostic Runtime Governance for Heterogeneous Agent Systems 

**Authors**: Zexun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04104)  

**Abstract**: Agent systems execute through runtimes with very different control points: local coding tools, framework SDKs, managed agent platforms, API gateways, and observer-only integrations. A high-risk action such as publishing data externally may therefore appear as a shell command in one runtime, a tool call in another, and a hosted session transition in a third. This makes it difficult to answer a basic governance question consistently: what action was authorized, under whose authority, with what approval semantics, and with what evidence after execution?
This paper presents Proof-Carrying Agent Actions (PCAA), a runtime-neutral governance model centered on an action certificate rather than on a vendor-native session record. PCAA organizes control around five checkpoints: pre-action admissibility, action open, assumption capture, approval, and outcome closure. It binds these checkpoints to a portable action envelope, runtime and approval receipts, and replay-ready proof. The model is extended in two practical ways: the certificate is externality-aware, carrying boundary facts such as destination visibility and account provenance, and approval is described by explicit enforceability classes rather than by a single reviewed or unreviewed bit.
We study the model through a reference implementation in a heterogeneous agent control plane and a disclosure-bounded evaluation protocol. On a protected benchmark expanded from 24 executable seeds to 96 traces across four runtime families, PCAA preserves route quality while exposing distinct failure modes under ablation. The paper contributes a systems formulation of runtime governance around certificate-bearing actions and an implementation-grounded account of how that formulation can remain portable under runtime churn without collapsing into vendor-specific control surfaces. 

---
# The Differentiable Auditory Loop (DAL): An ML Framework for Hyper-Personalized Hearing Aids 

**Authors**: Alejandro Ballesta Rosen, Jason Mikiel-Hunter, Julian Maclaren, Jack Collins, Richard F. Lyon, Simon Carlile  

**Link**: [PDF](https://arxiv.org/pdf/2606.04103)  

**Abstract**: Conventional hearing aids rely on fixed, frequency-dependent amplification and compression to manage reduced sensitivity, which often fails to provide sufficient listening support in complex environments, such as situations with multiple speakers (the ``cocktail party'' problem). To more comprehensively address the underlying encoding dysfunctions of hearing loss, we introduce the Differentiable Auditory Loop (DAL), a new open-source framework for personalized hearing aid design and fitting. Our first implementation of DAL incorporates CARFAC, a differentiable model of human cochlear function, which we ported to JAX, to optimize a deep neural network to match impaired auditory neural activity patterns with a normal-hearing reference. To build a hearing aid with the fine-grained spectro-temporal signal processing required, we adopt SEANet, a waveform-to-waveform fully convolutional UNet generator. We fine-tune the network by comparing the outputs of a CARFAC model fitted to normal hearing with that of a CARFAC model fitted to match each subject's individual hearing impairment. The comparison is done using loss functions derived from the respective CARFAC neural activity pattern (NAP) outputs and stabilized auditory images (SAIs), the latter providing a 2D representation that captures phase-insensitive temporal structure in the auditory nerve output. Through gradient descent, the SEANet model learns to both denoise the input and compensate for the hearing loss modelled by the impaired CARFAC model. Across neural-representation and signal-fidelity metrics, the DAL-optimized SEANet model outperformed the tested master hearing aid (MHA) baselines. The DAL framework provides a practical path toward model-based, machine-learning-driven personalization of hearing aid signal processing. Next steps include hardware deployment to enable real-world clinical testing. 

---
# POLARIS: Guiding Small Models to Write Long Stories 

**Authors**: Rishanth Rajendhran, Jenna Russell, Mohit Iyyer, John Frederick Wieting  

**Link**: [PDF](https://arxiv.org/pdf/2606.04095)  

**Abstract**: Small open-weight models struggle at long-form creative writing: their generated stories either fall far short of the requested length, or their quality significantly degrades as length increases, especially when compared to frontier models. We present POLARIS (Policy Optimization with LLM-as-a-judge rewards and Anchored-Reference Injection for Storywriting), a lower-compute GRPO recipe with two key ingredients: a frontier LLM judge with a structured Story Quality rubric as the online reward, and human-reference injection (HRI), where a teacher-forced human-written story serves as a high-reward anchor within each GRPO group. By applying our training recipe to Qwen3.5-9B, using a dataset of approximately 1.4K prompt-story pairs derived from 100 short-story anthologies and 4 A100 GPUs, we obtain POLARIS-9B. Across five benchmarks spanning in-distribution and out-of-distribution prompts and rubrics, POLARIS-9B is competitive with much larger open-weight models while following length instructions more closely. A blinded human evaluation confirms that POLARIS-9B is preferred to the base Qwen3.5-9B and on par with Qwen3.5-27B. Despite training only on stories up to 4k words, POLARIS-9B preserves quality on prompts requesting stories up to 3 times the training length, a regime where most open-weight models degrade substantially in quality, length adherence, or both. More broadly, our results suggest that length generalization is a meaningful stress test for creative-writing models and a useful lens for distinguishing otherwise close models. 

---
# Large Language Models Hack Rewards, and Society 

**Authors**: Wei Liu, Xinyi Mou, Hanqi Yan, Zhongyu Wei, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2606.04075)  

**Abstract**: Reinforcement learning (RL) has become a dominant post-training paradigm, enabling large language models (LLMs) to learn from rewards. We observe that societal regulations are structurally similar to reward functions. They define measurable outcomes, thresholds, and exceptions, while often leaving institutional intent only partially specified. We hypothesise that the RL training process may exploit these gaps and therefore ask whether models' well-known tendency to hack reward functions during RL can scale into a more consequential failure mode named societal hacking: discovering loopholes in the rules society runs on. To study this phenomenon, we introduce SocioHack, a sandbox of 72 societal environments, and find that within these environments, reward hacking naturally emerges and leads to regulatory loophole discovery. Models learn to hack the social rules and generate strategies that remain technically compliant while defeating regulatory intent, and current LLM safeguards provide only limited mitigation. Therefore, collecting in-the-wild feedback for model training requires greater caution, and we need a next-generation post-training paradigm for safely iterating LLMs in real society.= 

---
# Adaptive Patching Is Harder Than It Looks For Time-Series Forecasting 

**Authors**: Federico Zucchi, Yi Xie, Chao Zhang, Keyuan Luo, Thomas Lampert, Ziyue Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.04074)  

**Abstract**: Adaptive patching is a recent and compelling proposal for time-series Transformers: allocate finer patches where the sequence looks locally informative. This paper asks under what conditions a content-adaptive patching operator should outperform a tuned uniform one. Local heterogeneity alone is not enough: under pointwise forecasting losses, a complex-looking region is not automatically one where finer patching reduces the loss. We model patching as a budgeted bitrate allocation and derive an explicit threshold that a dynamic patching rule must satisfy to beat a well-tuned uniform baseline, then bound the achievable improvement both locally (a quadratic surrogate) and globally (a strong-convexity bound under the model's assumptions). Two structural results follow: without a coupling constraint, scalar local complexity cannot produce a non-uniform optimum under a common loss landscape; and once the backbone is trained to its representation-aware optimum, the alignment gain collapses around a well-tuned uniform patch size. To test these predictions, we run a controlled isolation study on three representative architectures, replacing each adaptive mechanism with a uniform patch-size sweep while keeping the backbone, data, and training protocol fixed. On standard long-horizon forecasting benchmarks, the validation-selected uniform baseline is competitive with the dynamic counterpart, with per-setting effects concentrated near zero and no consistent directional advantage once results are aggregated by dataset. The larger gains we do observe are method- and dataset-specific. Adaptive patching should therefore be evaluated against a tuned uniform baseline; its value depends on whether a cheap and reliable routing signal can identify where finer patches actually reduce forecasting loss. 

---
# TPA-AD: A Two-Stage Pseudo Anomaly-Guided Method for Bearing Time-Series Anomaly Detection 

**Authors**: Xiancheng Wang, Zhibo Zhang, Ran Li, Rui Wang, Minghang Zhao, Shisheng Zhong, Lin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04073)  

**Abstract**: This paper proposes a two-stage pseudo anomaly-guided anomaly detection method (\textbf{T}wo-stage \textbf{P}seudo \textbf{A}nomaly-guided \textbf{A}nomaly \textbf{D}etection, \textbf{TPA-AD}) for axle-box bearing time-series anomaly detection (time series anomaly detection, TSAD) under the setting where only normal samples are available for training. The method first generates pseudo-anomalous windows near the normal boundary using a reconstruction model and per-feature target-error control. It then learns anomaly-sensitive representations through contrastive learning between normal and pseudo-anomalous windows, and finally produces window-level and point-level anomaly scores using k-nearest neighbors (KNN). Compared with existing methods that rely on known fault categories, real anomaly priors, or random anomaly injection, TPA-AD improves the separability of the normal boundary by constructing pseudo-anomalies in boundary neighborhoods and can jointly handle continuous and discrete features in mixed-variable scenarios. The main experiments are conducted on bearing fault detection datasets and degradation-process datasets, with an additional exploratory extension on $13$ public TSAD datasets. The results show that the proposed method yields relatively stable anomaly responses, is sensitive to degradation evolution, and demonstrates a certain degree of broader applicability on public TSAD benchmarks and real high-speed-train-related bearing data. 

---
# Need to Know: Contextual-Integrity-Grounded Query Rewriting for Privacy-Conscious LLM Delegation 

**Authors**: Xinyue Huang, Xiaochun Cao, Wenyuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04067)  

**Abstract**: As LLMs become increasingly woven into everyday workflows, user queries sent to cloud hosted LLMs routinely mix task-essential content with task non-essential sensitive disclosures, yet type based PII redaction is context agnostic and may raise two issues: over disclosing untyped sensitive context and over removing answer bearing spans. We recast privacy preserving query rewriting under Contextual Integrity: a span should be forwarded only if it is necessary for the task. We introduce DelegateCI-Bench, the first task based Contextual Integrity benchmark for privacy-conscious delegation, comprising 3,167 samples that combine high quality synthetic data spanning 11 tasks and 20 task types, WildChat based real user queries, and a medical challenge set with dense sensitive information. Building on this benchmark, we propose a CI-guided reinforcement learning framework that converts essential and non-essential sensitive spans into verifiable optimization signals, and train a query rewriter to preserve task critical information while suppressing unnecessary sensitive disclosure. Experiments show that our learned rewriter achieves the best privacy-utility tradeoff, achieving up to +10.1 average utility over on-device baselines. 

---
# LLM Compression with Jointly Optimizing Architectural and Quantization choices 

**Authors**: Hoang-Loc La, Truong-Thanh Le, Amir Taherkordi, Phuong Hoai Ha  

**Link**: [PDF](https://arxiv.org/pdf/2606.04063)  

**Abstract**: Deploying large language models (LLMs) is challenging due to their significant memory and computational requirements. While some methods address this by developing small or tiny language models from scratch, these approaches demand extensive GPU training. Compressing pre-trained LLMs for edge devices offers a compelling alternative. Beyond pruning and quantization, Neural Architecture Search (NAS) enables effective compression, yet prior NAS approaches often limit the search space and decouple architecture from quantization. We introduce a differentiable NAS framework that explores the entire space and jointly optimizes architectural configurations alongside mixed-precision quantization for linear layers of LLMs. Experiments demonstrate superior accuracy-latency trade-offs: our models achieve up to 1.4x faster inference than sequential NAS-then-quantization baselines at comparable accuracy, or up to 6% higher average accuracy across seven reasoning tasks at equivalent latency. 

---
# Spectral Scaling Laws of Muon 

**Authors**: Gagik Magakyan, Pablo Parrilo, Asuman Ozdaglar  

**Link**: [PDF](https://arxiv.org/pdf/2606.04058)  

**Abstract**: Orthonormalized update rules have rapidly become a leading choice of optimizer for training large language models, with recent open-source state-of-the-art models adopting Muon. To keep these updates tractable, Muon performs the orthonormalization with the Newton--Schulz (NS) iteration. Since NS is only approximate, directions with small singular values fail to be orthonormalized. In Muon, NS is applied to the momentum matrix at every step, yet little is known about how the singular value spectrum of these momentum matrices behaves during training, or how that behavior changes with model size. We present the first systematic study of this question. Tracking singular value quantiles of the momentum buffer across layers in models ranging from 77M to 2.8B parameters, we observe a consistent picture: after a short burn-in, the quantiles stabilize at a value determined by the layer type and model size. These stabilization values follow remarkably clean power laws in model size, with layer-dependent exponents. Layers up to mid-late depth scale very mildly with model size $M$ (around $M^{-0.25}$), so the standard 5-step NS configuration used at academic scale will continue to orthonormalize them at much larger scales. Some of the late layers, however, scale much more aggressively (up to $M^{-0.96}$) and will fall into the NS failure regime at frontier scale unless one uses more NS iterations or better-tuned coefficients. NS iterations are computationally expensive at scale; our laws give practitioners a principled, layer-aware recipe for choosing the minimum NS configuration that still orthonormalizes the directions that matter -- avoiding unnecessary computation without sacrificing update quality. 

---
# The Invisible Lottery: How Subtle Cues Steer Algorithm Choice in LLM Code Generation 

**Authors**: Akanksha Narula, Mofasshara Binte Rafique, Laurent Bindschaedler  

**Link**: [PDF](https://arxiv.org/pdf/2606.04057)  

**Abstract**: Large language models (LLMs) now generate substantial production code, often for tasks with multiple valid algorithmic solutions. Incidental prompt cues, meaning contextual words or metadata outside the task specification, can steer which algorithm the model selects, even when all outputs pass the same tests. Prompt sensitivity is well studied as a tool to improve output quality. Here, output policy means algorithm choice under fixed correctness. We define algorithm steering as cue-induced shifts in algorithm-family distributions and run 46,535 controlled experiments across 11 tasks, 19 cue types (18 channels plus a memoization semantic-vs-surface ablation that preserves meaning while changing typography and punctuation), and 15 model configurations. We find large, systematic shifts in algorithm-family distributions (up to 100 pp), largely consistent with cue semantics, including in applied tasks such as rate limiting. Direct algorithm naming is the most reliable mitigation we tested. Accidental context therefore creates an "invisible lottery" over performance, security, and maintainability. 

---
# A Goal-Set Characterization of Task Composition in the Boolean Task Algebra 

**Authors**: Eduardo Terrés-Caballero, Herke van Hoof  

**Link**: [PDF](https://arxiv.org/pdf/2606.04053)  

**Abstract**: The Boolean Task Algebra (BTA) provides a principled framework for zero-shot task composition in reinforcement learning by equipping goal-reaching tasks with Boolean operations. We revisit its structural assumptions and formalize a collapse in the space of optimal extended Q-value functions: in deterministic MDPs, every such function is fully determined by the universal and empty tasks. This makes the logarithmic set of base tasks proposed in the original BTA formulation redundant. Building on this observation, we introduce a goal-set-based composition method that performs logical operations on goal sets and reconstructs composed value functions by selecting slices from the universal and empty value functions. This reduces learning costs for standard BTA and reduces composition time for both BTA and Skill Machines, while preserving policy performance. Experiments across tabular, visual, function-approximation, and continuous-control domains show that learning additional base tasks does not yield better performance. Finally, we study the stochastic setting and provide a counterexample showing that this collapse need not hold, that is, optimal composition may require accounting for exponentially many policies in the number of goals. Code is available at this https URL. 

---
# RUBAS: Rubric-Based Reinforcement Learning for Agent Safety 

**Authors**: Xian Qi Loye, Qinglin Su, Zhexin Zhang, Shiyao Cui, Qi Zhu, Fei Mi, Hongning Wang, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04051)  

**Abstract**: The evolution of LLMs into tool-enabled agents creates a new class of safety challenges associated with real-world execution rather than simple text generation. Existing alignment methods often rely on coarse refusal signals or static supervision, making it difficult to balance safety with useful tool execution across diverse agentic risks. We introduce RUBAS, a rubric-based reinforcement learning framework for agent safety. RUBAS decomposes agent behavior into four dimensions: tool-use safety, argument safety, response safety, and helpfulness. These structured rubrics provide fine-grained and interpretable rewards over complete agent trajectories, enabling reinforcement learning to optimize safe tool use while preserving task completion. Extensive experiments across multiple agent safety benchmarks and models show that RUBAS improves safety over standard alignment baselines, reduces tool-grounded hallucinations, and maintains competitive utility. Our results suggest that multi-dimensional rubric rewards provide an effective training signal for aligning LLM agents in safety-critical tool-use settings. 

---
# LiftQuant: Continuous Bit-Width LLM via Dimensional Lifting and Projection 

**Authors**: Liulu He, XuanAng Liu, Juntao Liu, Taolue Feng, Ting Lu, Chunsheng Gan, Zhiyv Peng, Yuan Du, Huanrui Yang, Yijiang Liu, Li Du  

**Link**: [PDF](https://arxiv.org/pdf/2606.04050)  

**Abstract**: Existing quantization methods are fundamentally limited by rigid, integer-based bit-widths (e.g., 2, 3-bit), resulting in a ``deployment gap" where Large Language Models cannot be optimally fitted to specific memory budgets. To bridge this gap, we introduce LiftQuant, a novel framework that enables continuous bit-width control for true Pareto-optimal deployment. The core innovation is a ``lift-then-project" mechanism which approximates low-dimensional weight vectors by projecting a simple 1-bit lattice from a higher-dimensional ``lifted" space. Crucially, the effective bit-width is determined simply by the ratio of the lifted dimension to the original dimension, which allows the bit-width to be tuned quasi-continuous as the dimension is a flexible structural parameter. This projection generates a structured yet non-uniform codebook, capturing the expressive power of Vector Quantization (VQ). While beneficial over VQ, LiftQuant's decoding path relies solely on linear transformations and 1-bit uniform quantizers, retaining hardware-friendly nature. This flexibility is transformative: LiftQuant enables a 70B LLM to be compressed to 2.4 bits to precisely fit a 24GB GPU, where its performance significantly surpasses state-of-the-art 2-bit models fitted on the same device. Our code and ckpt is available at this https URL. 

---
# Unlocking Feature Learning in Gated Delta Networks at Scale 

**Authors**: Yifeng Liu, Quanquan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2606.04048)  

**Abstract**: Training and scaling Large Language Models demand enormous computational resources, motivating both efficient sub-quadratic architectures and principled hyperparameter tuning methods. While the Maximal Update Parametrization ($\mu$P) has enabled zero-shot hyperparameter transfer for standard Transformers, its extension to linear models, particularly those with structured state transitions and complicated architectures, remains largely unexplored. By rigorously propagating coordinate-size estimates through the forward pass, gating mechanisms, and recurrent state dynamics, we derive the scaling rules for Gated Delta Network. Experiments on language-model pre-training confirm that our configurations enable stable learning-rate transfer across model widths under both AdamW and SGD, whereas standard parametrization fails to transfer, validating the correctness and practical utility of our analysis. 

---
# Dive into the Scene: Breaking the Perceptual Bottleneck in Vision-Language Decision Making via Focus Plan Generation 

**Authors**: Boyuan Xiao, Bohong Chen, Yumeng Li, Ji Feng, Yao-Xiang Ding, Kun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.04046)  

**Abstract**: In embodied vision-language decision making tasks such as robotic manipulation and navigation, Vision-Language and Vision-Language-Action Models (VLMs & VLAs) are powerful tools with different benefits: VLMs are better at long-term planning, while VLAs are better at reactive control. However, their performance is limited by the same perceptual bottleneck: visual hallucinations arise due to the models' inability to distinguish task-relevant objects from distractors. In principle, accurate identification and focus on critical objects while filtering out irrelevant ones is the key to break this limitation. A straightforward solution is one-step focus: directly attending to essential objects. However, this approach proves ineffective because effective focus inherently requires deep scene understanding. To this end, we propose SceneDiver, a coarse-to-fine focus plan generation method for VLMs leveraging their long-term planning abilities, that first constructs a holistic scene graph to establish initial comprehension, then progressively decomposes the task into simpler sub-problems through an iterative cycle of recognition, understanding, and analysis. To enable reactive control, we also design a lightweight adapter for distilling the deliberate focus ability into VLAs. Evaluations on standard embodied AI benchmarks confirm that our method substantially reduces visual hallucinations for both VLMs and VLAs, while preserving computational efficiency in tasks requiring fast execution. Our code and data are released at: this https URL. 

---
# Bayes-Sufficient Representations in Supervised Learning 

**Authors**: Vasileios Sevetlidis  

**Link**: [PDF](https://arxiv.org/pdf/2606.04045)  

**Abstract**: Representation learning is often described as preserving the information in an input that is relevant for prediction. This work asks what relevance means for a fixed supervised decision problem. A representation is defined to be Bayes-sufficient for a joint distribution and loss if some prediction head can use it to implement a Bayes-optimal action rule. This makes the target information loss-dependent. In the almost-surely unique Bayes-action case, the relevant object is a Bayes quotient, which identifies inputs that require the same Bayes-optimal action. A representation is sufficient when it refines this quotient, and Bayes-minimal when it is informationally equivalent to it. The framework connects naturally to property elicitation: zero-one loss requires the Bayes class, squared loss the conditional mean, Brier loss the conditional probability in binary prediction, and log loss or strictly proper scoring rules the predictive distribution. Controlled finite experiments, learned neural bottleneck experiments, and a real-data iNaturalist taxonomic refinement experiment illustrate the distinction between sufficiency, minimality, and retained non-required information. For a fixed supervised problem, the distribution and the loss determine the Bayes action, the Bayes action determines the quotient, and the quotient determines the minimal information required for Bayes-optimal prediction. 

---
# Channel-Oriented Design for EEG-to-Music Reconstruction 

**Authors**: Jiaxin Qing, Junwei Lu, Lexin Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.04040)  

**Abstract**: Brain-computer interfaces aim to decode naturalistic stimuli from neural signals, yet most progress to date has focused on vision and language. In this article, we study a more challenging but far less explored setting, EEG-to-music reconstruction, where signals are weak, distributed, and highly susceptible to noise and channel variability. Our central finding is that early channel mixing destroys weak but discriminative EEG signals. To address this, we propose a channel-oriented design with three key components. Specifically, channel-wise tokenization treats each electrode as an explicit token to retain spatially localized neural evidence, channel-wise multi-view self-distillation enforces consistency across temporal crops and random channel subsets to learn robust and distributed representations, and channel-wise data augmentation introduces structured channel dropout to improve invariance to noise, artifacts, and missing electrodes. Together, these components preserve weak yet informative signals across channels and enable stable alignment to a semantic music representation space. We integrate this channel-oriented design within an encoding-alignment-decoding pipeline for EEG-to-music reconstruction. Theoretically, we characterize when preserving channel-level structure leads to improved alignment. Empirically, we compare with a range of state-of-the-art baselines and demonstrate consistent and significant performance gains. 

---
# Beyond Static Priors: Dynamic Neural Guidance for Large-Scale Ant Colony Optimization 

**Authors**: Dat Thanh Tran, Van Khu Vu, Yining Ma  

**Link**: [PDF](https://arxiv.org/pdf/2606.04039)  

**Abstract**: Neural-guided Ant Colony Optimization (ACO) suffers from a fundamental training-inference misalignment: policies are typically trained to generate static priors (e.g., heatmaps), yet deployed to guide iterative, long-horizon search processes. In this paper, we present DyNACO, a novel framework that achieves dynamic neural guidance by periodically observing the pheromone distribution and the incumbent solution. To make DyNACO tractable at scale, we pair the policy with a perturbation-based ACO backend and a scope-restricted refinement mechanism that jointly ensure efficacy and stable credit assignment. On TSP, DyNACO scales to 100,000-node instances and outperforms neural baselines while often reducing total runtime compared to the unguided solver. We extend DyNACO to CVRP via a capacity-aware backend, consistently improving the unguided baseline with less than 1% neural overhead. We further provide in-depth analysis validating the model's generalization capabilities and elucidating why dynamic guidance outperforms static priors. Our work underscores the necessity of aligning neural training with iterative search dynamics in learning-guided optimization. The code is available at this https URL. 

---
# Unpredictable Safety: Domain-Dependent Compliance and the Transparency Gap in Open-Weight LLMs 

**Authors**: Zacharie Bugaud  

**Link**: [PDF](https://arxiv.org/pdf/2606.04035)  

**Abstract**: We present a systematic study of domain-dependent safety behavior in open-weight LLMs: 7 standardized experiments across 7 ethical domains, testing 5 models (12B--70B) in 4,200 interactions with dual-judge validation. Using a dual-condition methodology, each scenario tested in both an analytical framing (identify the harm) and an operational framing (help commit the harm), we find compliance rates vary from 14.7% (human trafficking) to 85.7% (surveillance design), a 71-percentage-point span with non-overlapping cluster-bootstrapped 95% CIs. Trustworthy deployment requires predictable safety behavior, yet we find compliance is highly context-dependent: the same model (Mistral Nemo 12B) provides surveillance designs in 100% of requests but assists with trafficking in only 26.7%. This unpredictability is opaque to deployers: the technical framing bypass, where harmful requests reframed as engineering problems override safety training without any external signal that refusal thresholds have shifted. Within-domain heterogeneity reaches 84.4pp, meaning safety behavior cannot be predicted even at the domain level. A replication on five frontier closed models (GPT-4.1/5.2, Claude Haiku/Sonnet/Opus 4.x; n=4,163 responses) accessed via the GitHub Copilot CLI deployed-product surface reproduces the same domain stratification, attenuated in absolute level but identical in shape, with the two low-codification domains (science fraud, surveillance) again the most permissive. These results show that current safety mechanisms lack the transparency and consistency required for trustworthy AI deployment. 

---
# Do Transformers Need Three Projections? Systematic Study of QKV Variants 

**Authors**: Ali Kayyam, Anusha Madan Gopal, M Anthony Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2606.04032)  

**Abstract**: Transformers have become the standard solution for various AI tasks, with the query, key, and value (QKV) attention formulation playing a central role. However, the individual contribution of these three projections and the impact of omitting some remain poorly understood. We systematically evaluate three projection sharing constraints: a) Q-K=V (shared key-value), b) Q=K-V (shared query-key), and c) Q=K=V (single projection). The last two variants produce symmetric attention maps; to address this, we also explore asymmetric attention via 2D positional encodings. Through experiments spanning synthetic tasks, vision (MNIST, CIFAR, TinyImageNet, anomaly), and language modeling (300M and 1.2B parameter models on 10B tokens), we discovered that our transformers perform on par or occasionally better than the QKV transformer. In language modeling, Q-K=V projection sharing achieves 50% KV cache reduction with only 3.1% perplexity degradation. Crucially, projection sharing is complementary to head sharing (GQA/MQA): combining Q-K=V with GQA-4 yields 87.5% cache reduction, while Q-K=V + MQA achieves 96.9%, enabling practical on-device inference. We show that Q-K=V preserves quality because keys and values can occupy similar representational spaces and attention operates in a low-rank regime, whereas Q=K-V breaks attention directionality. Our results systematically characterize projection sharing as an underexplored instance of weight tying in attention, with direct, quantifiable inference memory benefits, particularly valuable for edge deployment. The code is publicly available at this https URL 

---
# Position: Deployed Reinforcement Learning should be Continual 

**Authors**: Parnian Behdin, Kevin Roice, Golnaz Mesbahi  

**Link**: [PDF](https://arxiv.org/pdf/2606.04029)  

**Abstract**: Reinforcement Learning (RL) has received increasing attention and adoption in real-world use cases. Most of these systems follow a train-then-fix paradigm, where trained agents do not learn while interacting with the world until performance degrades and retraining becomes necessary. In this position paper, we argue that deploying an agent that is incapable of optimality, but receives an evaluative reward signal, is inherently a continual RL problem. We identify four sources of non-stationarity after deployment that necessitate never-ending learning, and highlight why the best deployed agents never stop adapting. We analyze successful examples of continual RL in the real world, and present the community with the advantages and measures to move away from the current train-then-fix paradigm. 

---
# MaskForge: Structure-Aware Adaptive Attacks for Jailbreaking Diffusion Large Language Models 

**Authors**: Yingzi Ma, Zhengyue Zhao, Xiaogeng Liu, Minhui Xue, Yue Zhao, Chaowei Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2606.04027)  

**Abstract**: Diffusion large language models (dLLMs) generate text by iteratively denoising partially masked sequences under bidirectional context, exposing a safety surface distinct from autoregressive LLMs. Because mask tokens are native inputs and tokens are committed by confidence rather than position, harmful content can be induced through infilling and outside the monitored prefix. Existing jailbreaks either miss this native infill capability or rely on low-diversity mask-bearing templates applied uniformly across goals, with little structural adaptation or accumulated attack experience. We propose MaskForge, a fully black-box adaptive attack that casts dLLM red-teaming as optimized search over a growing library of structural patterns. MaskForge abstracts successful attempts into reusable schemas, selects goal-compatible patterns with a UCB bandit, and invokes a scorer-guided fallback when the current library fails. Successful attempts are distilled back into the pattern library, enabling experience to accumulate across goals. Across five public dLLMs and three benchmarks, MaskForge achieves an average attack success rate of 79.3%, a 17.6% relative improvement over the strongest competing dLLM baseline. The matured pattern library further transfers to AdvBench without any updates, achieving a 88.2% attack success rate and a 67% relative improvement over the strongest competing baseline. 

---
# The Biomimetic Architecture of Software 4.0 

**Authors**: Philip Sheldrake, Dirk Scheffler  

**Link**: [PDF](https://arxiv.org/pdf/2606.04025)  

**Abstract**: Dominant programming paradigms inherit an execution model optimised for a bygone era of a single human mind instructing a local machine, leaving contemporary systems burdened with historical path dependencies. When forced to host multi-dimensional, connectionist intelligence, this brittle assembly model fractures under the weight of a profound probabilistic-symbolic impedance mismatch. While contemporary Software 3.x frameworks attempt to patch the mismatch by encasing large language models (LLMs) in increasingly complicated external harnesses, this spiralling architectural complexity only compounds the carrying cost of static code assembly. To address the cause rather than the effects, this paper introduces Software 4.0 -- an autopoietic heterarchy of human intelligence, neural AI, and natively reflective symbolic substrate. Under this paradigm, software is transformed from an inert corpus to be parsed into a self-regulating metabolic network that natively verifies, modifies, and evolves its own structural integrity. We present Recognitive, the programming language and platform that materialises this architecture. By offloading the burden of structural verification to a deterministic substrate, it unlocks a superior inference-time scaling regime -- one where connectionist compute translates entirely into deep semantic exploration and hypothesis traversal rather than the ruinous computational and financial cost of simulating structural constraints probabilistically. Moving beyond the legacy 'Software Factory' mindset, we outline the theoretical foundations required to ground connectionist intent and arrive fully in the intelligence age. This is a foundational vision paper; empirical evaluation and formal specification of the type system and operational semantics are the subject of future work. 

---
# CodegenBench: Can LLMs Write Efficient Code Across Architectures? 

**Authors**: Jie Li, Wenzhao Wu, Junqi Hu, Qinrui Zheng, Bowen Wu, Juepeng Zheng, Yutong Lu, Haohuan Fu  

**Link**: [PDF](https://arxiv.org/pdf/2606.04023)  

**Abstract**: While large language models (LLMs) have been extensively evaluated on code generation tasks for general-purpose programming and GPU-accelerated environments (e.g., PyTorch, CUDA), their capabilities in CPU-oriented high-performance computing (HPC) across diverse architectures remain underexplored. To bridge this gap, we introduce CodegenBench, a comprehensive benchmark suite designed to evaluate the generation of efficient parallel code across three distinct hardware platforms: x86_64, Sunway, and Kunpeng. Our benchmark comprises 106 standard Basic Linear Algebra Subprograms (BLAS) routines establishing a fundamental baseline, alongside 20 specialized computational kernels adapted for each of the unique supercomputing architectures (LeetSunway and LeetKunpeng). Our extensive evaluation reveals that while state-of-the-art LLMs can generate optimized code for ubiquitous architectures like x86_64, they exhibit significant performance degradation on domain-specific architectures with limited public documentation and training data, highlighting critical limitations in cross-platform generalization. Furthermore, our analysis of factors influencing code quality such as implementation length and task complexity indicates that current LLMs are most effective for moderately difficult problems requiring concise code snippets. We open-source our dataset and automated evaluation infrastructure to facilitate future research in LLM-driven high-performance code generation. The resources are available at this https URL and this https URL. 

---
# Gravity-Aware Hierarchical Routing for Lightweight SensorLLM on Human Activity Recognition 

**Authors**: Hao Li, Mingrui Zheng, Yasuyuki Tahara, Yuichi Sei  

**Link**: [PDF](https://arxiv.org/pdf/2606.04019)  

**Abstract**: Recent studies on sensor-language alignment have shown that two-stage frameworks can improve the semantic modeling ability of wearable-sensor human activity recognition (HAR), where SensorLLM-style methods first perform motion-to-language alignment and then fine-tune the model for downstream tasks. However, our experiments reveal a consistent failure mode when the Stage 2 backbone is compressed to a compact model such as TinyLlama: recognition of dynamic activities remains relatively strong, while the discrimination of low-motion static classes such as standing, sitting, and lying degrades substantially. To address this issue, we propose a gravity-aware hierarchical routing head as a lightweight post-alignment adaptation built on top of an already aligned model, rather than a new large-scale pretraining framework. The method uses the per-channel mean and std from the Chronos tokenizer state to extract statistical cues related to posture and gravity direction, and adaptively combines a static expert and a full expert through soft routing, together with a load-balancing loss for stable training. On the MHealth dataset, this design significantly improves macro-F1 with minimal parameter overhead, and the gains are concentrated mainly on static classes while preserving strong performance on dynamic activities. As a first arXiv disclosure, the current paper reports results on a single dataset only, with the goal of highlighting the core method and laying the groundwork for broader evaluation in future work. 

---
# The Variance Brain Foundation Models Forgot: Third-Order Statistics Predict Cognition Where Billion-Parameter Models Fail 

**Authors**: Giovanni Marraffini, Gabriel Mahuas, Trinidad Borrell, Victoria Shevchenko, Demian Wassermann  

**Link**: [PDF](https://arxiv.org/pdf/2606.04010)  

**Abstract**: Brain foundation models (BFMs) are self-supervised Transformers pretrained on fMRI data. We posit that these models should capture each subject's cognitive performance from their fMRI signal. Yet across three state-of-the-art BFMs and every readout we test, they predict cognition worse than a linear regression from the $\sim$80K parameters of the functional connectivity matrix (FC). The gap widens with scale: BrainLM's 650M model predicts cognition worse than its 111M. We attribute this to a \textbf{variance allocation problem}: BFM pretraining captures the variance components that dominate fMRI but not the higher-order structure that predicts cognition. Our per-cumulant analysis of the reconstructed signal shows that the second-order covariance is partially preserved, while the third-order co-skewness tensor is largely destroyed. To recover what BFMs lose, we design a linear pipeline that projects the fMRI signal into the subspace that best preserves its co-skewness and computes FC there. This \textbf{exceeds raw FC and every pretrained BFM} on every dataset and parcellation we test, outperforming prior state-of-the-art under controlled evaluation \textbf{with no pretraining and no GPU}. We \textbf{recover the raw-FC ceiling on BrainLM's forward pass} by finetuning with a loss targeted at this same subspace. This shows that the bottleneck is the pretraining objective, not the architecture or the model size. 

---
# Counterfactual Explanations for Deep Two-Sample Testing 

**Authors**: Wei-Cheng Lai, Marco Simnacher, Christoph Lippert  

**Link**: [PDF](https://arxiv.org/pdf/2606.04009)  

**Abstract**: Two-sample testing is a fundamental tool for detecting distributional differences across scientific domains, but classical tests (including kernel-based tests) can be ineffective on high-dimensional structured data such as images. Recent deep two-sample tests improve sensitivity in these settings by learning informative representations, yet they provide limited insight into which data features drive rejection of the null hypothesis $H_0$. To address this issue, we propose a counterfactual explanation framework for deep two-sample testing that generates sample-level edits moving observations from a source group toward a target group while explicitly reducing the discrepancy measured by the test. Our method combines a diffusion autoencoder with a pretrained deep two-sample test model and optimizes a maximum mean discrepancy (MMD) objective in the test model's representation space to produce plausible counterfactuals. We quantify distribution-level effects through changes in the test statistic and the resulting two-sample p-values. We evaluate the method on synthetic 2D shape datasets and two MRI cohorts. Across both settings, the counterfactual transformations consistently increase p-values relative to the original samples, indicating that the edited source set becomes statistically closer to the target distribution under the test. We measure minimality using LPIPS to ensure the counterfactuals remain close to the original samples. The resulting edits provide interpretable evidence of the features associated with the detected group differences. On MRI, the localized changes are consistent with known anatomical differences between cohorts. 

---
# Neural Radiated-Noise Fields for Unmanned Underwater Vehicle Noise Spectrum Prediction in Three-Dimensional Scenes 

**Authors**: Yan Wu, Yang Yang, Jun Fan, Bin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04008)  

**Abstract**: Radiated noise in unmanned underwater vehicles (UUVs) is an important indicator for characterizing acoustic signatures and evaluating platform performance. To address the strong dependence of traditional physics-based modeling and numerical simulation methods on target structural information and environmental boundary conditions, and their inability to achieve continuous spatial spectrum-response modeling in three-dimensional scenes, this paper proposes a neural radiated-noise field (NRNF). An NRNF represents the UUV radiated-noise spectrum as a continuous function of the three-dimensional UUV position, the three-dimensional hydrophone position, the UUV yaw angle, and the frequency, enabling query-based prediction at arbitrary spatial locations. The proposed method employs sinusoidal encoding for position and frequency, and introduces a learnable three-dimensional scene feature grid to explicitly represent environmental structure and propagation effects. A spectrum-prediction dataset is constructed from lake trials, and the proposed model is evaluated under three settings: horizontal extrapolation, depth extrapolation, and cross-run generalization. Results show that the NRNF achieves an average prediction error of 3.5 dB in the 50 to 5000 Hz band. Horizontal extrapolation is easiest, depth extrapolation is the most challenging, and cross-run generalization is of intermediate difficulty. Further ablation results demonstrate that the scene feature grid significantly improves the prediction stability and spatial generalization of the model. 

---
# Early Detection of Alzheimer's Disease Using Explainable Machine Learning on Clinical Biomarkers: A Multi-Class Classification Study Using the Alzheimer's Disease Neuroimaging Initiative (ADNI) Dataset 

**Authors**: Afshan Hashmi  

**Link**: [PDF](https://arxiv.org/pdf/2606.03995)  

**Abstract**: Background: Alzheimer's disease (AD) affects over 55 million people worldwide. Accurate, interpretable detection of normal cognition (NC), mild cognitive impairment (MCI), and AD from routine clinical assessments remains a critical unmet need. Methods: An XGBoost classifier was developed for three-class detection using eight clinical features from the Alzheimer's Disease Neuroimaging Initiative (ADNI): MMSE, CDR Global, CDR Sum of Boxes (CDR-SB), MoCA, FAQ, age, sex, and education. Hyperparameters were optimised using Optuna (50 trials); class imbalance was addressed with SMOTE. Performance was evaluated by macro AUC-ROC with 1,000-iteration bootstrap 95% confidence intervals, macro F1, balanced accuracy, and Cohen's kappa. SHAP values provided feature-level explainability. Results: The dataset comprised 1,641 baseline subjects (608 NC, 767 MCI, 266 AD). On five-fold cross-validation, mean macro AUC was 0.983 (SD 0.007), accuracy 0.944 (SD 0.006), and macro F1 0.929 (SD 0.008). On the held-out test set (n = 247), macro AUC was 0.982 (95% CI: 0.965--0.995), accuracy 0.943, balanced accuracy 0.932, macro F1 0.927, and Cohen's kappa 0.909. SHAP analysis identified CDR Global as the dominant predictor for NC and MCI, while CDR-SB and MMSE together drove AD classification. Conclusion: An explainable machine learning model trained on routine clinical assessments achieves near-perfect three-class Alzheimer's detection. SHAP analysis reveals clinically plausible, class-specific feature importance patterns supporting clinical validity. Future work will extend this framework with speech biomarkers for multimodal detection. 

---
# Constraint-Enhanced Physical Search through Correlation Matching 

**Authors**: Song-Ju Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.03554)  

**Abstract**: Physical systems do not merely add noise to search processes; they impose constraints that generate structured correlations. We propose a principle of constraint-enhanced physical search in which temporal correlations in exploration are matched to constraint-induced spatial correlations in the update dynamics. Using a minimal tug-of-war bandit model (TOW), we show that a conservation law converts local observations into differential evidence across alternatives, while a temporally correlated drive controls the order of exploration. Search efficiency is improved not by stronger randomness or by maximal anti-correlation, but by matching the temporal correlation to the physical update scale that converts feedback into evidence. A scaling estimate identifies the update-noise-to-contrast ratio as the leading parameter that limits how strongly temporal anti-correlation can be used. The results suggest a general organizing principle for physical search: constraints and fluctuations can generate structured spatiotemporal correlations, and efficient exploration emerges when these correlations are matched to the update dynamics. 

---
# SpurAudio: A Benchmark for Studying Shortcut Learning in Few-Shot Audio Classification 

**Authors**: Giries Abu Ayoub, Morad Tukan, Loay Mualem  

**Link**: [PDF](https://arxiv.org/pdf/2605.13672)  

**Abstract**: Few-shot classification (FSC) is widely used for learning from limited labeled data, yet most evaluations implicitly assume that target concepts are independent of contextual cues. In real-world settings, however, examples often appear within rich contexts, allowing models to exploit spurious correlations between foreground content and background signals. While such effects have been studied in few-shot image classification, their role in few-shot audio classification remains largely unexplored, and existing audio benchmarks offer limited control over contextual structure. We introduce SpurAudio, a benchmark that leverages the natural separability of foreground events and background environments in audio to enable controlled, multi-level evaluation of contextual shifts across support and query sets. Using this benchmark, we show that many state-of-the-art few-shot methods suffer severe performance degradation when background correlations are disrupted, despite achieving similar accuracy under standard evaluation protocols. Crucially, this vulnerability persists even in large pretrained audio foundation models, ruling out limited backbone capacity as an explanation. Moreover, methods that appear comparable under conventional benchmarks can exhibit markedly different sensitivity to spurious correlations, revealing systematic algorithmic strengths and vulnerabilities tied to how feature representations interact with classifier heads at inference time. These findings provide new insight into the behavior of few-shot methods in audio and highlight the need for benchmarks that explicitly probe context dependence when evaluating FSC models. 

---
# DiffAero: A GPU-Accelerated Differentiable Simulation Framework for Efficient Quadrotor Policy Learning 

**Authors**: Xinhong Zhang, Runqing Wang, Yunfan Ren, Jian Sun, Hao Fang, Jie Chen, Gang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10247)  

**Abstract**: This letter introduces DiffAero, a lightweight, GPU-accelerated, and fully differentiable simulation framework designed for efficient quadrotor control policy learning. DiffAero supports both environment-level and agent-level parallelism and integrates multiple dynamics models, customizable sensor stacks (IMU, depth camera, and LiDAR), and diverse flight tasks within a unified, GPU-native training interface. By fully parallelizing both physics and rendering on the GPU, DiffAero eliminates CPU-GPU data transfer bottlenecks and delivers orders-of-magnitude improvements in simulation throughput. In contrast to existing simulators, DiffAero not only provides high-performance simulation but also serves as a research platform for exploring differentiable and hybrid learning algorithms. Extensive benchmarks and real-world flight experiments demonstrate that DiffAero and hybrid learning algorithms combined can learn robust flight policies in hours on consumer-grade hardware. The code is available at this https URL. 

---
# How do machines learn? Evaluating the AIcon2abs method 

**Authors**: Rubens Lacerda Queiroz, Cabral Lima, Fabio Ferrentini Sampaio, Priscila Machado Vieira Lima  

**Link**: [PDF](https://arxiv.org/pdf/2401.07386)  

**Abstract**: This study expands on previous work that introduced the AIcon2abs method (AI from Concrete to  Demystifying Artificial Intelligence to the general public), an innovative approach designed to increase public understanding of machine learning (ML) across diverse age groups, including K-12 students, and aims to evaluate its effectiveness. AIcon2Abs employs the WiSARD algorithm, a weightless neural network known for its simplicity, and user accessibility. WiSARD does not require Internet, making it ideal for non-technical users and resource-limited environments. This method enables participants to intuitively visualize and interact with ML processes through engaging, hands-on activities, as if they were the algorithms themselves. The method allows users to intuitively visualize and understand the internal processes of training and classification through practical activities. Once WiSARDs functionality does not require an Internet connection, it can learn effectively from a minimal dataset, even from a single example. This feature enables users to observe how the machine improves its accuracy incrementally as it receives more data. Moreover, WiSARD generates mental images representing what it has learned, highlighting essential features of the classified data. AIcon2abs was tested through a six-hour remote course with 34 Brazilian participants, including 5 children, 5 adolescents, and 24 adults. Data analysis was conducted from two perspectives: a mixed-method pre-experiment (including hypothesis testing), and a qualitative phenomenological analysis. Nearly all participants rated AIcon2abs positively, with the results demonstrating a high degree of satisfaction in achieving the intended outcomes. This research was approved by the CEP-HUCFF-UFRJ Research Ethics Committee. 

---
# AI from concrete to abstract: demystifying artificial intelligence to the general public 

**Authors**: Rubens Lacerda Queiroz, Fábio Ferrentini Sampaio, Cabral Lima, Priscila Machado Vieira Lima  

**Link**: [PDF](https://arxiv.org/pdf/2006.04013)  

**Abstract**: Artificial Intelligence (AI) has been adopted in a wide range of domains. This shows the imperative need to develop means to endow common people with a minimum understanding of what AI means. Combining visual programming and WiSARD weightless artificial neural networks, this article presents a new methodology, AI from concrete to abstract (AIcon2abs), to enable general people (including children) to achieve this goal. The main strategy adopted by is to promote a demystification of artificial intelligence via practical activities related to the development of learning machines, as well as through the observation of their learning process. Thus, it is possible to provide subjects with skills that contributes to making them insightful actors in debates and decisions involving the adoption of artificial intelligence mechanisms. Currently, existing approaches to the teaching of basic AI concepts through programming treat machine intelligence as an external element/module. After being trained, that external module is coupled to the main application being developed by the learners. In the methodology herein presented, both training and classification tasks are blocks that compose the main program, just as the other programming constructs. As a beneficial side effect of AIcon2abs, the difference between a program capable of learning from data and a conventional computer program becomes more evident. In addition, the simplicity of the WiSARD weightless artificial neural network model enables easy visualization and understanding of training and classification tasks internal realization. 

---
