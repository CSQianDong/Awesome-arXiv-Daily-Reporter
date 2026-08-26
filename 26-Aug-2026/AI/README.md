# Recursive Experiential-Working Memory Evolution for Long-Horizon Agent Harnesses 

**Authors**: Zhaochen Yu, Yingcheng Wu, Zhenfei Yin, Kaiyuan Chen, Zhe Zhao, Mengdi Wang, Shuicheng Yan, Ling Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.24876)  

**Abstract**: Recursive self-improvement (RSI) remains hard in long-horizon tasks, where growing histories obscure the task state and misalign skill invocation. We introduce Recuris, a recursive Experiential-Working Memory architecture for long-horizon agent harnesses, in which Working Memory tracks task progress and guides skill selection from Experiential Memory, grounding skill use in current needs rather than the full history. This coupling also turns execution into structured evidence that localizes failures to specific memory components. Across tasks, a fixed Meta-Agent turns that evidence into localized, validation-gated updates to Skill Memory that reshape execution and yield new evidence, forming a bounded recursive memory-evolution loop. Across four long-horizon benchmarks and ten models, Recuris improves task success in 35 of the 37 completed model-benchmark pairs, carrying frontier models to SOTA-level task success: on tau-bench it adds +17.8 points to GPT-5.6 Sol and +15.6 to Claude Opus 5, taking Opus 5 to 87.9%, and +16.6/+13.5 points on Qwen3.6-27B/35B on SkillFlow. The advantage widens as the interaction horizon grows, to +32.2 points on the longest tasks, and common long-horizon failures fall by up to 80%. These results position recursively evolving memory as a scalable foundation for RSI, enabling agents to continuously transform accumulated experience into increasingly effective long-horizon behavior. Code: this https URL 

---
# SPO++: Stream-Aligned Policy Optimization for Asynchronous Agentic RL 

**Authors**: Kai Ruan, Jinghao Lin, Qianshan Wei, Ziqi Zhou, Zihe Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.24870)  

**Abstract**: Group-relative reinforcement learning waits for sibling rollouts of the same prompt, which is costly for long and variable tool-use trajectories. Single-stream Policy Optimization (SPO) removes this dependency with a persistent prompt-level value estimate, but its recipe whitens one advantage per trajectory before optimizing a token-mean actor loss. We show that trajectory centering generally does not center the token-weighted quantity consumed by the actor, and fix the mismatch by standardizing terminal-outcome advantages under the action-token measure. We additionally organize prompt evidence by the policy event that generated it rather than learner receipt order. Across matched runs on ALFWorld at two model scales and on Math-TIR, SPO++ improves online learning efficiency over SPO. A paired ablation identifies action-token-measure normalization as the strongest tested component. 

---
# FedV-KGQA: Multi-Hop Question Answering over Vertically Partitioned Knowledge Graphs 

**Authors**: Md Saikat Islam Khan Bappy, Oshani Seneviratne  

**Link**: [PDF](https://arxiv.org/pdf/2608.24846)  

**Abstract**: Real-world data for knowledge graph question answering is often distributed across different organizations due to governance and data sovereignty constraints. While centralized systems exist, they cannot answer multi-hop questions when the required facts are split across vertically partitioned silos. In this paper, we propose FedV-KGQA, a framework for multi-hop reasoning over knowledge graphs in which organizations share entities but own disjoint sets of relations. Our approach combines local graph enrichment and knowledge graph embeddings to ensure raw triples and relation parameters never leave each silo, establishing a structural data boundary without requiring centralized graph access. We further introduce a topic entity anchoring mechanism that grounds questions in the correct graph neighborhood without any runtime inter-silo communication. We evaluate 12 model configurations across three benchmarks and show that FedV-KGQA performs strongly, remains close to centralized performance, generalizes to 3-hop reasoning, and is robust to embedding perturbations. 

---
# A Dual-Dimensional LLM Framework for Automated Item Incidental Content Similarity Analysis in Large-Scale Assessments 

**Authors**: Jing Huang, Jihong Zhang, Hua-Hua Chang  

**Link**: [PDF](https://arxiv.org/pdf/2608.24825)  

**Abstract**: The rapid expansion of large-scale assessments and the growing adoption of automatic item generation have intensified concerns about incidental content redundancy, where construct-irrelevant elements such as wording or contextual framing become unintentionally repetitive across items. Traditional similarity metrics like BLEU or cosine similarity, often fail to capture the nuanced structural and semantic layers that drive perceived redundancy simultaneously. This study proposes a dual-dimensional framework for Automated Item Similarity Analysis (AISA) powered by Large Language Models (LLMs), operationalizing similarity through Structured Decomposition and Semantic Relatedness. Psychometric validation indicates that LLM-derived metrics align more closely with indicators of construct-irrelevant local dependence and yield more coherent item parameter groupings than traditional text-based measures. The framework is further evaluated through its application in Computerized Adaptive Testing (CAT). Simulations reveal that incorporating LLM-based similarity constraints into item selection improves estimation stability and reduces bias with minimal efficiency trade-offs, outperforming constraints based on conventional metrics. These findings highlight the potential of LLM-powered AISA to support scalable bank curation, content-aware test assembly, and experience-sensitive adaptive testing across diverse assessment contexts. 

---
# Constrained Entity Selection under Partial Knowledge for LLM-Based Knowledge Graph QA 

**Authors**: Emanuel Kitzelmann  

**Link**: [PDF](https://arxiv.org/pdf/2608.24824)  

**Abstract**: Large language models are increasingly used for knowledge graph question answering (KGQA), but can fail to correctly ground answers in the underlying graph. Current approaches to LLM-based KGQA either rely on full semantic parsing into executable queries such as SPARQL, which is brittle in practice due to complex schemas or incompleteness of real-world KGs, or on LLM-reasoning and answer generation over KGs, which can be more robust but lacks formal guarantees. In this work, we study a complementary setting in which \emph{candidate} answers are generated by an LLM-based system and subsequently verified using lightweight symbolic constraints derived from the question. We introduce \emph{Constrained Entity Selection under Partial Knowledge (CES-PK)}, a problem formulation that focuses on eliminating invalid answers and providing symbolic support for valid ones without requiring construction of executable logical forms. To account for incomplete KGs, we employ a three-valued constraint semantics (\emph{satisfied, violated, unknown}) that avoids incorrect rejections under open-world assumptions. To demonstrate the effects of our method, we instantiate this framework over the Hetionet biomedical knowledge graph and evaluate the impact of type, relation, and exclusion constraints. Experiments show that precision improves by filtering invalid candidates, while recall is preserved due to retaining candidates whose constraints are not explicitly violated. Satisfied constraints provide additional positive symbolic evidence to rank remaining candidates. 

---
# Strictly Causal Streaming Video Anomaly Detection with a Theoretically-Grounded State-Space Core 

**Authors**: Yogesh Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2608.24810)  

**Abstract**: Recent work has applied Mamba style state space models (SSMs) to video anomaly detection, yet existing approaches still rely on buffering clips or windows internally, lack a theoretical account of how temporal memory relates to detection latency, and benchmark efficiency only through GPU throughput rather than the edge hardware these methods are intended to target. We introduce a strictly causal streaming anomaly detector whose fixed size state is updated in O(1) time and memory per incoming frame, with no lookahead and no clip buffering. Its temporal core is a diagonal linear state space recurrence with an input and state dependent decay gate, trained self supervised through causal next embedding prediction on a frozen visual backbone. We derive a closed form relationship between the recurrence decay spectrum and both detection delay and the shortest anomaly it can reliably capture, then validate empirically on UCSD Ped2 and CUHK Avenue. The settling delay bound predicted from the learned base decay (57 to 59 frames) sits far above the measured detection delay (1.6 and 18.4 frames), showing that the event boundary gate, not the base decay, governs responsiveness. We further report end to end latency and throughput measured directly on Apple M3 Pro hardware, 0.74 ms and 0.77 ms per frame (over 1300 FPS), rather than simulated GPU numbers. With an untuned initial configuration the method reaches 67.9 percent and 70.2 percent frame level AUC on Ped2 and Avenue, trailing prior non causal SSM baselines in accuracy. Ablations over decay rate, state size, and gating reveal that the gate contribution is dataset size dependent, hurting accuracy on the smaller Ped2 training set but helping on the larger Avenue one. Closing this accuracy gap and extending evaluation to a third, larger benchmark are immediate next steps. 

---
# StarHarness: Evolving Harnesses with Stratified Search for Enterprise Environments 

**Authors**: Esakkivel Esakkiraja, Denis Akhiyarov, Vikas Yadav, Sai Rajeswar, Patrice Bechard, Sridhar Nemala, Sagar Davasam  

**Link**: [PDF](https://arxiv.org/pdf/2608.24804)  

**Abstract**: We present StarHarness, a framework for evolving environment-specific agent harnesses while keeping model weights fixed. The evolved harness can include prompt and task framing, tool interfaces, skills, MCP-backed providers, subagent structure, and agent-loop configuration. StarHarness constructs a compact evolution pool by stratifying tasks according to baseline failure behavior, separates proposer-visible search tasks from proposer-hidden selection tasks, and reserves held-out tasks for evaluating generalization. Across ITBench SRE, EnterpriseOps-Gym ITSM, and AutomationBench Finance, harness evolution improves full-benchmark performance by 20-35 percentage points over the default harness after 4-12 accepted changes per environment. These gains persist on tasks excluded from evolution and transfer without re-evolution across GPT and Qwen model families. Trace analysis links the improvements to interface repairs, environment conventions, and operational knowledge that compresses search, with fewer false-positive diagnoses and shorter trajectories in several settings. StarHarness therefore offers a practical way to reduce persistent model-environment mismatch in tool-rich enterprise tasks. 

---
# CAFE: Self-Improving Search Agents Need Co-Evolving Feedback 

**Authors**: Boyang Liu, Senjie Jin, Peixin Wang, Zhangyue Yin, Yibo Wang, Yuhao Zhou, Xinbing Liang, Shizheng Zhu, Yuhui Wang, Jingqi Tong, Zhiheng Xi, Jiazheng Zhang, Clive Bai, Clarenceai, Blaze Chen, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.24794)  

**Abstract**: Outcome-supervised search agents learn when and how to retrieve evidence, but terminal rewards neither localize intermediate errors nor redirect an ongoing trajectory before those errors compound. Treating corrective feedback as a learned in-trajectory intervention couples the two roles: the agent must decide when to request and use feedback, while the critic must infer useful corrections from outcome-confounded rollouts whose failure patterns shift as the agent improves. We introduce CAFE (Coupled Agent--Feedback Evolution), a framework in which a shared-parameter model alternates between search-agent and critic roles. CAFE initializes feedback-conditioned recovery from trajectories built around the base agent's own failures, then couples online and offline optimization. During online RL, a comparative feedback estimate uses a prompt-level call--skip success gap to shape request returns, while feedback-aware advantage shaping reweights token advantages before and after feedback. Offline, rollout-derived preference optimization learns feedback from matched successful and unsuccessful trajectories. On seven agentic search benchmarks, CAFE outperforms the evaluated RL-based search agents on average, retains its gains across all six out-of-domain benchmarks, and reduces answer-level hallucinations. One-sided ablations show that improving only the agent or only the critic eventually plateaus, whereas alternating the two updates continues to improve performance. These findings suggest that a self-improving search agent needs feedback that co-evolves with the policy it guides. 

---
# Right Diagnoses, Decorative Reasoning:A Perturbation Audit of Medical Chain-of-Thought 

**Authors**: Mengzhu Xu, Jifan Gao, Xia Jiang, Yaoxin Wu, Xi Long  

**Link**: [PDF](https://arxiv.org/pdf/2608.24790)  

**Abstract**: Clinicians read chain-of-thought (CoT) rationales as evidence of medical reasoning, but whether the visible chain plays that role is rarely tested. General-domain CoT-faithfulness probes ignore clinical cost, and medical LLM evaluations treat the chain as a black box. We close this gap with a medical perturbation audit: a 30-operator battery edits both the chain and the question with clinically motivated operators (severity reversal, negation flip, demographic swap, evidence ablation), paired with a chain-update times answer-flip joint analysis that classifies each model by its failure mode. Applied to 14 LLMs on four medical QA benchmarks, three independent tests converge: the Chain-Decoupling Rate (CDR; chain does not register the edit and the answer does not flip) is 72.9% panel-wide on clinically meaningful destructive edits, chain corruption leaves accuracy unchanged, and removing CoT prompting does not reduce accuracy. Two board-certified clinicians re-annotate N=197 perturbed questions; 98.5% leave the gold defensible. The pattern holds across medical and reasoning fine-tuning and scale; on the closed-source tier, where the chain text is unavailable, the answer-side signals are consistent with the same decoupling. Our framework and CDR provide a reusable yardstick for auditing whether medical CoT is faithful or merely documentation. 

---
# StepGuard: Learning Step-Level Guardrails with Scalable Supervision and Safety-Utility Balancing 

**Authors**: Zhijie Zheng, Yu Li, Chen Qian, Yuqian Fu, Yanwei Fu, Lu Sheng, Jing Shao, Dongrui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.24777)  

**Abstract**: LLM-based agents can interact with external environments through tool invocation, but this capability also introduces security risks such as file modification, information leakage, and unauthorized actions. Existing guardrails often evaluate completed trajectories, leaving pre-execution monitoring of step-level actions underexplored. We propose StepGuard, a step-level guard model that can audit completed agent trajectories and check tool actions before they are executed. To train StepGuard, we introduce StepGen, an automatic data engine that generates safe and unsafe trajectories with the same context but different actions at the risky step. To further reduce over-defense and under-defense, we propose Balance-GRPO, which dynamically balances learning between safe and unsafe actions based on their observed accuracy. Experiments show that StepGuard achieves the highest average accuracy among open-weight guard models, with performance comparable to GPT-5.4. When used to guard agents on AgentDojo and AgentDyn, StepGuard reduces mean attack success rate by 77.3% relative to the no-guard setting, while mean utility drops by only 2.8 percentage points. 

---
# Evidence Blindness in Direct Corpus Interaction: Persistent Navigation with AtlasNav 

**Authors**: Hongyu Guo, Zhiyu Zheng, Zhao Cao  

**Link**: [PDF](https://arxiv.org/pdf/2608.24764)  

**Abstract**: Large language model agents are moving beyond conventional retrieval-augmented generation toward direct interaction with external corpora. Direct Corpus Interaction (DCI) keeps the full corpus accessible, yet reachable evidence can remain unusable under finite interaction budgets. Required evidence may fail to surface, a surfaced supporting document may remain unopened, or an opened document may fail to expose its decisive fragment. We call this progressive silent loss Evidence Blindness and quantify it through stage-wise evidence realization. Within the DCI paradigm, raw interaction adds little reusable corpus organization, while dynamic-workspace methods reconstruct a query-conditioned interaction space from each query and trajectory. In both cases, useful structure is recovered largely online. We instead formulate large-scale agentic search as finite-budget navigation over reusable corpus structure. We introduce AtlasNav, a persistent multi-view corpus-navigation framework that retains direct corpus interaction but organizes the corpus once into a Corpus Atlas, allowing each query to navigate adaptively rather than reconstruct shared structure. On BrowseComp-Plus, AtlasNav achieves 92.05% strict accuracy while reducing recorded online inference cost by 30.21% relative to the prior dynamic-workspace state of the art. Under matched budgets, it realizes the complete required evidence earlier and approaches the same model's evidence-supplied empirical reference more rapidly. The same representation principle remains effective under PhantomWiki's distinct corpus organization and controlled 10K-1M scaling, and transfers competitively to heterogeneous enterprise knowledge. These results show that agentic search depends not only on accessible evidence, but also on how the corpus is represented so that limited interaction becomes effective navigation. 

---
# RACE: Scalable Statistical Estimation of Functional Consistency in LLM Neurons 

**Authors**: Runyu Wang, Bo Liu, Xiaxin Zhang, Yu Han, Jiawei Cao, Xiaoye Zhang, Zhe Zhang, Yifan Yang, Peng Ping  

**Link**: [PDF](https://arxiv.org/pdf/2608.24758)  

**Abstract**: Discovering stable neuron behavior across entire domains remains a challenge in mechanistic interpretability. Existing methods often rely on instance-level point estimates or computationally expensive procedures, which either obscure population-level variability or limit scalable domain-wide analysis. We present RACE (Residual Alignment for Consistency Estimation), a forward-pass statistical framework that evaluates the domain-wide functional consistency of Transformer neurons. Perturbation experiments demonstrate that RACE achieves superior domain specificity compared to gradient-based point estimates. Meanwhile, token-distribution-level results verify the association between the selected neurons and the target domain. Furthermore, its computational overhead is two orders of magnitude lower than that of gradient-based methods. 

---
# Meta$^n$: Recursive Self-Improvement through Emergent Depth 

**Authors**: Zae Myung Kim, Young-Jun Lee, Seungyeon Jwa, Dongyeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2608.24735)  

**Abstract**: Self-improving LLM agents refine answers, not the process that produces those answers. Systems that add a meta-level hold that level fixed, and those that edit themselves must leave part of their own editing machinery untouched to stay stable, capping the meta-depth they realize at roughly two. We present Meta$^n$, which keeps the meta-operation fixed and recurses on its input instead. That operation, $\Omega$, is applied repeatedly to its own products, reading the traces of the solver stack below together with the code that produced them, then writing the next layer as a strategic pre-process and a library of callable helpers. Because $\Omega$ never changes, it cannot destabilize the system, and because its input strictly grows, each layer reasons from a higher vantage than the last. Depth is set by convergence rather than fixed in advance, and an evolutionary archive searches over layer chains. Across two backbones, Meta$^n$ outperforms prior self-improving agents on all eight benchmark families. The sharpest case is ARC-AGI-2, built to resist skill memorization, where it alone scores above zero. Ablations indicate that most of the gain from recursion comes from the conditioning each layer passes to the next, and distinct layer roles emerge with depth although no prompt prescribes them. Code available at this https URL 

---
# Lifted Model Construction under Approximate Commutativity 

**Authors**: Malte Luttermann, Jan Speller, Tanya Braun, Marcel Gehrke, Ralf Möller  

**Link**: [PDF](https://arxiv.org/pdf/2608.24713)  

**Abstract**: Lifted inference algorithms enable scalable probabilistic inference even for large object domains by leveraging the indistinguishability of objects in a probability distribution. An essential prerequisite for constructing a lifted representation is to identify commutative factors, i.e., functions whose output values are invariant under permutations of a subset of their input values, in a potential-based factorisation. In practice, however, parameters learned from data inevitably deviate even if associated objects are indistinguishable, causing their corresponding factors to be only approximately commutative instead of being exactly commutative. We address this problem by introducing the concept of {\epsilon}-commutativity, a relaxation of commutativity where output values are only approximately invariant under permutations of input values. Specifically, we show how {\epsilon}-commutativity can be exploited for lifted model construction, downstream probabilistic inference, and prove strict bounds on the induced approximation error, thereby ensuring the practical applicability of lifted model construction while maintaining highly accurate query results. These theoretical guarantees are confirmed empirically, demonstrating comparable query accuracy at lower runtime. 

---
# Confident at the moment of action: belief miscalibration in LLM play under hidden information 

**Authors**: Bhushan Kashinath Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2608.24691)  

**Abstract**: Agentic systems increasingly gate actions on a model's own stated confidence, which assumes confidence tracks correctness at the moment of acting. We test this in a hidden-information chess variant where royal status can be secretly, repeatedly relocated between pieces, and where an agent's stated probability distribution over the opponent's hidden royal piece -- elicited every turn, separately from the move it chooses -- is scored against ground truth recoverable after the game. Across two independent batches, captures made at high stated confidence ($\geq 0.5$) about the hidden piece's location were correct in 1 of 62 cases. The calibration deficit is concentrated almost entirely in these events: 99.3% of it in the original batch, 98.7% in the replication. The same pattern, in weaker form, orders consistently (point estimates only; most pairwise gaps are not statistically distinguishable at this sample size) across four further model configurations spanning a second provider -- reported as scope for the finding, not as evidence that capability predicts calibration: a same-model comparison at a fixed external leaderboard score shows a deliberation-budget change alone moves the metric by nearly as much as a large cross-model gap. In a separate seat, conventional evaluation axes -- legality, cost, latency, completion rate -- can dissociate entirely from belief quality, with the configuration winning on every conventional axis producing the worst belief quality tested. A model exhibiting this pattern can still win the game its belief was about, which is why outcome-only evaluation would not detect it. 

---
# The Invisible Editorial Layer: Formalizing Undisclosed Inference-Time Steering, Probability Placement, and the Attribution Problem in Deployed Language Models 

**Authors**: Augusto Camargo  

**Link**: [PDF](https://arxiv.org/pdf/2608.24662)  

**Abstract**: Large language models (LLMs) are commonly evaluated under the assumption that their observable behavior is primarily determined by model weights, training data, alignment procedures, and user prompts. This view is incomplete. Modern inference pipelines may systematically modify the probability distribution produced by a model immediately before token selection, creating an additional layer of control between frozen weights and observed text.
While controlled generation (e.g., PPLM, GeDi, DExperts, FUDGE) and text-watermarking systems (e.g., SynthID-Text) demonstrate the technical maturity of decoding- and logit-level interventions, the governance, security, and economic implications of an undisclosed inference policy remain comparatively underexplored. This paper examines the emergence of inference-time framing bias: the systematic modification of generated language toward political, ideological, institutional, or commercial frames via interventions applied after model inference but before token sampling.
We formalize the operational reality Model != Deployed System and introduce three concepts: (1) the Inference Attribution Problem, characterizing why observed behavioral bias cannot generally be causally attributed to model weights alone under limited observability; (2) Probability Placement, defining a hypothetical advertising primitive in which commercial influence is implemented through systematic shifts in generation probabilities rather than explicit product insertions; and (3) Inference Policy Transparency, a governance principle for making deployment-layer interventions auditable. We examine these concepts in relation to Article 5 of the EU AI Act, the EU Digital Services Act, and FTC doctrines. 

---
# Parason: Revealing Subtask and Trial Parallelism in LLM Reasoning 

**Authors**: Zhengyang Zhang, Zijian Zhang, Jiaxuan Gao, Shusheng Xu, Yi Wu, Song Han, Ligeng Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.24658)  

**Abstract**: Scaling test-time reasoning has substantially improved the problem-solving ability of large language models (LLMs), but standard autoregressive decoding still executes long reasoning traces sequentially, creating severe latency for difficult tasks (up to days and weeks). Parallel reasoning offers a natural remedy. However, prior systems primarily focus on Subtask Parallelism, where the model learns to decompose a high-level task into smaller chunks that can be solved independently. This approach overlooks another pervasive form of parallelism: Trial Parallelism, where multiple speculative attempts explore, verify, and aggregate competing hypotheses in parallel. In this paper, we introduce Parason, which reveals and learns both forms of parallelism in LLM reasoning. Our analysis identifies Trial Parallelism as the majority of parallelizable reasoning computation (65.5% in DeepSeek-V4's reasoning steps in HLE), and it becomes increasingly dominant on hard problems. Guided by this taxonomy, Parason converts sequential reasoning traces into structured parallel trajectories with a context-free grammar, then trains models with Parallelism-Aware Group Relative Policy Optimization (PA-GRPO), whose reward jointly balances accuracy, latency, and the two parallelism ratios. At inference time, Parason executes the learned parallel structure through tool calls, translating theoretical savings to real-world wall-clock acceleration. Experiments on mathematical reasoning benchmarks including AIME24 and AIME25 show that Parason achieves an average acceleration about 1.7$\times$ while maintaining competitive accuracy. 

---
# Causal Modelling of Support Interventions for Student Competency Assessment 

**Authors**: Francesca Mangili, Alessandro Antonucci, Rafael Cabañas  

**Link**: [PDF](https://arxiv.org/pdf/2608.24632)  

**Abstract**: Accurate assessment of student competencies is essential for enabling educators to identify individual needs, design targeted interventions, and evaluate the effectiveness of educational strategies. Empirical assessment procedures are typically grounded in psychometric models, such as item response theory, which relate student competence levels to performance on assessment tasks. In this paper, we advocate adopting a structural causal modelling approach to educational assessment, moving beyond probabilistic belief updating toward a framework that explicitly supports interventional and counterfactual reasoning. We propose a corresponding protocol for its construction and analyse the practical relevance of forms of reasoning that remain inaccessible to standard associative models, including the explicit modelling of interventions such as hints and the related counterfactual scenario analysis. Although our protocol requires the structural equations to be elicited from experts, the necessary information is purely logical and does not rely on probabilistic, less tenable assumptions. We illustrate the approach using data from an assessment that employs complex tasks designed to measure compulsory school student algorithmic skills. 

---
# Pivot-and-Station Multi-Agent Path Finding: Solvability, Complexity, and Algorithms 

**Authors**: Andrea Di Nezza, Mihir Patel, Fabio Fagnani, Sara Bernardini  

**Link**: [PDF](https://arxiv.org/pdf/2608.24585)  

**Abstract**: Automated high-density storage systems (warehouses, robotic parking, plant logistics, etc.) require fleets of agents to move through scarce task-critical resources and then park without obstructing future operations. We introduce Pivot-and-Station Multi-Agent Path Finding (PS-MAPF), a MAPF variant in which a subset of tasked agents must each visit one of a set of interchangeable pivots (e.g., workstations) before the entire fleet terminates at anonymous stations, one agent per station. We characterize solvability completely: every instance on a 2-edge-connected graph is solvable, and, on arbitrary connected graphs, a structural effective-distance measure relative to the number of unoccupied vertices gives a necessary and sufficient condition. We prove that minimizing station-makespan or station-flowtime is NP-hard already with a single pivot. We present three algorithms, a complete baseline, a SAT-based optimal solver, and Pivot-Prioritized Planning (PPP), the last solving 74-89% of benchmark instances with makespan and flowtime orders of magnitude below the baseline. 

---
# PhysMLLMs: Spatial Priors for Unified Referring Segmentation and Grounded Reasoning of Images and Videos 

**Authors**: Siyao Yan, Bo Han, Jisheng Dang, Bimei Wang, Shude Wang, Hong Peng, Yulan Guo, Jianhuang Lai, Bin Hu, Tat-SengChua  

**Link**: [PDF](https://arxiv.org/pdf/2608.24574)  

**Abstract**: Video multimodal large language models support language guided video segmentation, but they often show spatio temporal inconsistencies, e.g., jitter, drift, and identity switches. These failures are more common when targets are partly hidden or when similar objects appear this http URL likely reason is that current training lacks explicit spatial priors, which makes it difficult to maintain stable spatial identity and shape over time. We present PhysMLLMs, a training-stage prior injection architecture that injects physics-inspired spatial continuity priors into Video MLLMs. PhysMLLMs is designed to encourage more stable object-centered representations by aligning the student global visual representation with a frozen teacher model during training. Our core mechanism, Global Representation Prior Alignment (REPA-Global), distills global visual representations from a frozen DINOv2 teacher using an offline embedding cache and a scheduled distillation plan. This design keeps inference unchanged and does not add inference time cost. Across multiple video benchmarks, PhysMLLMs improves video segmentation mask quality and cross-frame consistency, with larger gains on challenging cases involving small targets, fast motion, occlusion, distractors, and reasoning queries. On single-frame referring image segmentation and representative general VLM benchmarks, PhysMLLMs maintains comparable performance, demonstrating that the injected spatial prior improves video consistency without compromising image-level grounding or general multimodal capability. These results suggest that physics-inspired spatial prior injection can improve temporal stability while preserving general capability. The code is available at this https URL. 

---
# Joint Optimization of Tool Creation and Use for Large Language Model Agents 

**Authors**: Zhi Rui Tam, Chieh-Yen Lin, Yun-Nung Chen, Shao-Hua Sun, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.24571)  

**Abstract**: Tool-augmented language models are bounded by the APIs humans bothered to write; existing tool-creation systems patch this by prompting a frozen LLM at inference time, leaving the model that writes a tool decoupled from the one that uses it, with no signal that the schemas it produces are schemas it can invoke. We propose SMITH (Schema-grounded Multi-task Iterative Tool Honing), a reinforcement learning framework that jointly trains tool creation and tool use inside a single policy. Each rollout is either a build task (write a tool from a few examples) or a use task (invoke a pooled tool on a held-out question). Three separate reward axes catch schema, code, and outcome failures independently, so each failure mode contributes its own gradient. A 4B Qwen3 trained with SMITH on 13 procedural reasoning tasks with exact verifiers reaches 79.8 macro-average accuracy on held-out tasks, the best across all evaluated methods and ahead of an untrained 30B-A3B tool-writer. It also reaches 40.4 on TabMWP-Hard and 42.6 on out-of-domain GQA (+7.6 over the best same-backbone inference-time baseline), without any visual or tabular training data. Tools written by our 4B models also lifted the performance of LFM-2.5-350M and Qwen3-30B-A3B under same reasoning tasks. 

---
# EviDx: Evidence-Aware Active Diagnosis with Scaffolded LLM Agents 

**Authors**: Lihang Zeng, Shaoting Zhang, Xiaofan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.24570)  

**Abstract**: Clinical diagnosis is an active evidence-seeking process in which clinicians acquire evidence, update competing hypotheses, and decide when the available evidence is sufficient for diagnosis. Yet many medical diagnosis systems built around large language models (LLMs) still formulate diagnosis as static case-to-answer prediction, with limited support for evidence acquisition. Agentic LLMs offer a dynamic alternative through tool use and intermediate diagnostic trajectories, but existing systems often under-specify how patient evidence should be exposed, scaffolded, and controlled at runtime. We introduce EviDx, an evidence-aware active diagnosis framework that pairs patient-specific diagnostic environments with a clinical diagnostic scaffold and an observer-guided runtime harness. In EviDx, $\mathcal{E}$-Synthesis constructs interactive environments from raw clinical cases; the scaffold organizes role-specialized agents, evidence tools, and evolving evidence states; and the harness regulates diagnostic termination by tracking uncertainty and evidence coverage. A 3-level evaluation pyramid assesses execution robustness, reasoning dynamics, and diagnostic outcomes. Experiments show that EviDx improves diagnostic performance and process stability while revealing model-dependent capability boundaries. 

---
# When "Must" Becomes "Maybe": Constraint Weakening in LLM Agent Workflows 

**Authors**: Yiheng Sun, Huifei Wang, Yancheng Zhu, Zhenyu Li, Zebin Zhao, Yifan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2608.24569)  

**Abstract**: Large language model (LLM) agents coordinate complex tasks through multi-role and multi-stage workflows. Upstream state is repeatedly transformed into intermediate language artifacts, such as summaries, plans, tickets, memories, and handoff notes, from which downstream components act. For action-constraining state, topical retention is insufficient: an artifact may mention an unresolved condition while changing it from a requirement that must be resolved before execution into information that may merely inform the next action. We study this action-binding role as operational state preservation. Safety blockers provide a controlled instance because each source state has an explicit prerequisite, authority, fallback, and execution consequence. We condition on correct upstream identification, vary the handoff transformation, and evaluate an executor restricted to the resulting artifact. Across 1,296 controlled synthetic episodes, direct-handoff controls preserve every blocker, whereas compression, plan assimilation, convergence, ownership deferral, and precedent substitution repeatedly turn binding state into caveats or non-binding considerations. Normal handoff compression produces 100.0% deactivation and 54.2% forbidden action. Restoring all four state fields raises preservation to 100.0% and reduces forbidden action to 0.0%. Fixed-artifact interventions further separate preservation from containment: downstream verification eliminates forbidden action while artifact deactivation remains 95.3%. These results identify a state-transmission failure between information extraction and action. Handoff transformations can retain state content while weakening its constraints on downstream action. Semantic availability does not guarantee operational preservation. 

---
# Discovering Adaptive Transmission Programs for Collective Innovation 

**Authors**: Cédric Colas, Jérémy Perez, Eleni Nisioti, Akhilesh Mocherla, Pierre-Yves Oudeyer, Clément Moulin-Frier, Maxime Derex  

**Link**: [PDF](https://arxiv.org/pdf/2608.24545)  

**Abstract**: Human collective intelligence depends on transmission processes: who shares what with whom, how, and when. While these processes emerge from individual cognition, they can also be directed by deliberate top-down protocols. Prior work has studied how transmission shapes collective outcomes primarily through the lens of network structure, varying who shares with whom and when. But networks are state-agnostic: they cannot condition transmission on what agents know or on the state of the collective. Here, we formalize transmission protocols as state-aware programs that route information and resources based on agent and collective states, and we use LLM-guided evolutionary search to design effective protocols in a collective discovery task. Evolved protocols increase collective performance over standard baselines from the literature by up to 37%. Ablations confirm that state-awareness drives this advantage: removing content-dependence while preserving network topology and timing eliminates performance gains. We find that evolved protocols also transfer across domain variations and agent populations. These results demonstrate that effective and generalizable transmission protocols can be discovered in silico, suggesting a path toward AI-assisted design of coordination infrastructure that enhances human collective intelligence. 

---
# Neurosymbolic Alignment for Physiologically-Safe Clinical Language Models 

**Authors**: Abdulhady Abas Abdullah, Erik Cambria, Milena Zivkovic  

**Link**: [PDF](https://arxiv.org/pdf/2608.24534)  

**Abstract**: Clinical LLMs can generate recommendations that are factually plausible yet physiologically unsafe. We investigate whether safety alignment can be improved by grounding preference optimization in structured physiological knowledge rather than text-only supervision. Methods: We propose Neurosymbolic Alignment, a training-time framework that couples a 7B clinical LLM with an HGNN-based Physiological World Model over an 847K-node biomedical knowledge graph. Candidate responses are scored using homeostatic constraints, multi-hop path plausibility, and drug-interaction penalties, and the resulting rankings drive iterative on-policy ORPO updates. Evaluation is performed on the Clinical Safety Benchmark (CSB), a 2,500-scenario benchmark for physiological constraint violations in generative clinical reasoning. Results: Relative to ORPO, the proposed method improves CSS from 69.5% to 90.8% (+21.3 pp), reduces physician-evaluated HR from 14.1% to 5.1% on the blinded subset, and improves DID from 72.8% to 91.6%. These gains are corroborated by an HGNN-independent Rule-Engine Safety Score (RSS: 86.4%, +21.2 pp over ORPO; r=0.97 concordance with CSS). The method also exceeds GPT-4 (5-shot) on all safety metrics despite a 10x parameter disadvantage, and outperforms an inference-time self-correction pipeline (SFT+SelfCorrect) by 11.4 pp CSS. Under synthetic EHR-style noise, 84.2% CSS is retained. Ablation analysis shows that HGNN scoring (-16.2 pp) and iterative training (-11.5 pp) are the dominant contributors. PhysioScore calibration against 200 clinician labels yielded ECE = 0.038 and kappa = 0.91. Conclusion: Training-time physiological grounding produces measurable and independently verifiable safety improvements in open-weight clinical LLMs under controlled evaluation. External validation on real clinical data is required to determine whether these gains transfer to deployment settings 

---
# PeakBench: Benchmarking Resource-Aware Tool Invocation in LLM Agents 

**Authors**: Zhi-Kai Chen, Xu-Xiang Zhong, Song-Yan Li, De-Chuan Zhan, Han-Jia Ye  

**Link**: [PDF](https://arxiv.org/pdf/2608.24509)  

**Abstract**: LLM agents increasingly solve tasks by invoking multiple tools, where parallel execution is essential for low latency but difficult to manage safely. Existing agent benchmarks primarily evaluate tool selection, argument generation, and end-to-end success under mostly serial execution, largely overlooking valid parallelization and resource-constrained scheduling. This missing scheduling dimension creates a practical failure mode: serial execution is safe but slow, while resource-agnostic parallel execution is fast but prone to avoidable resource overflows. To address this gap, we introduce PeakBench, a benchmark of executable multi-tool workflows with execution-grounded dependency annotations and measured resource profiles. A central challenge in evaluating such workflows is attribution: failures and inefficiencies may arise from incorrect dependency planning, poor resource-constrained scheduling, or both. PeakBench addresses this challenge with a two-part evaluation framework that disentangles logical planning from physical scheduling, with dedicated metrics for each dimension. Using this framework, we show that strong logical planning does not reliably translate into safe or efficient execution under resource constraints. We further show that exposing resource information can reduce avoidable overflows and improve resource utilization, making PeakBench a useful testbed for diagnosing resource-aware agent behavior. Code is available at this https URL. 

---
# Implicit Q-learning-bootstrapped ant colony optimization for maritime moving-target observation scheduling with agile satellites 

**Authors**: He Wang, Junyu Wu, Yeye Liu, Yifan Zhou, Jie Zhang, Hui Li, Yanjie Song, Liang Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.24471)  

**Abstract**: Maritime moving-target observation scheduling with agile Earth observation satellites is a dynamic, sequence-dependent combinatorial optimization problem. Sea-surface targets move continuously, causing feasible observation windows to vary with target motion and satellite orbital geometry. The scheduler must jointly determine task selection, satellite assignment, observation-window selection, and observation ordering under time-window, attitude-maneuvering, onboard-resource, and cloud-affected availability constraints. This paper proposes an implicit Q-learning-bootstrapped ant colony optimization method, termed IQACO, for multi-satellite maritime moving-target observation scheduling. Rather than directly learning a task-selection policy, IQACO embeds an offline implicit Q-learning module into constructive ant colony optimization to adaptively adjust the pheromone factor, heuristic factor, and evaporation rate. A compact search-state representation captures pheromone distribution, current and historical-best solution quality, and iteration progress. During online scheduling, ant colony optimization constructs feasible observation sequences, while the learned policy regulates exploration and exploitation according to the current search state. Experiments on 14 scenarios with different scales and satellite configurations show that IQACO obtains the highest mean observation benefit in every scenario, improves the result of conventional ant colony optimization by 3.40\%--9.40\%, accelerates convergence, and remains stable under different objective-weight settings. These results demonstrate that offline value learning provides an effective adaptive search-control mechanism for constrained maritime moving-target observation scheduling. 

---
# Reinforcement Learning-Guided Evolutionary Policy Optimization for Preference-Adjustable Heterogeneous Agile Earth Observation Satellite Scheduling 

**Authors**: He Wang, Junyu Wu, Hui Li, Yanjie Song, Witold Pedrycz, Liang Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.24470)  

**Abstract**: Heterogeneous agile Earth observation satellite (AEOS) scheduling requires task selection, satellite assignment, and observation sequencing under satellite-dependent visibility windows, attitude maneuvering requirements, energy consumption, and onboard storage constraints. Since satellites differ in orbital access, maneuvering capability, and payload resources, the same task may have different feasible windows, transition costs, and resource-consumption patterns on different platforms, which increases the difficulty of unified modeling and efficient optimization. To address this problem, this paper proposes an evolutionary policy optimization framework for heterogeneous AEOS scheduling with preference-adjustable weighted objectives. In the modeling layer, assignment-based indirect encoding is combined with decoder-based equivalent-cost evaluation to retain satellite-dependent constraints while integrating task gain, energy saving, and load balance into an interpretable scalar utility. In the optimization layer, schedule decoding, population-based search, and online actor-critic operator control are decoupled, so that reinforcement learning selects high-level search operators rather than constructing schedules directly. Based on this framework, a reinforcement-learning-assisted operator-selection memetic evolutionary algorithm (RLOSMEA) is developed to coordinate global exploration, feasibility recovery, and local refinement under a limited function-evaluation budget. Experiments on different heterogeneous AEOS scenarios show that RLOSMEA achieves higher overall weighted utility and more stable convergence than representative metaheuristic baselines. Sensitivity and learning-behavior analyses further confirm the robustness of the proposed method and the effectiveness of reinforcement-learning-guided operator selection. 

---
# HMGCLIP: Heterogeneous Multi-Granularity Contrastive Learning for E-commerce Representation Learning 

**Authors**: Qiuyu Zhu, Yi Gao, Zhichao Wan, Mingyang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2608.24467)  

**Abstract**: Although recent Multimodal Large Language Models (MLLMs) have advanced general product understanding, they implicitly encode product information into global embeddings, thereby limiting their ability to capture fine-grained attributes. This limitation hinders performance in tasks requiring precise attribute discrimination, such as distinguishing subtle material differences among visually similar products. To address this challenge, we propose HMGCLIP, a unified multimodal embedding framework. By constructing a heterogeneous hypergraph, we leverage hypergraph topology to mine structure-aware hard negatives and align multi-granular semantics at both relation and hyperedge levels. This design enables a dual-granularity inference mechanism that dynamically fuses attribute evidence for both fine-grained and coarse-grained downstream tasks. Furthermore, we release a comprehensive fine-grained e-commerce dataset to facilitate future benchmarking. Extensive experiments on this new dataset and the public MAVE benchmark show that HMGCLIP outperforms strong multimodal encoders, MLLMs, and e-commerce baselines, validating the superiority of HMGCLIP. 

---
# Mahalanobis-Based Multi-Head Attention for Complex State Propagation 

**Authors**: Xiaohe Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.24462)  

**Abstract**: In this paper, we propose \textbf{Mahalanobis-Based Multi-Head Attention} (MHA-CSP), a novel attention mechanism that replaces the standard dot-product with a \textbf{Mahalanobis distance-based RBF kernel}, which effectively computes attention in an infinite-dimensional feature space without increasing the parameter count. Crucially, the positive definiteness of the Mahalanobis distance enables a \textbf{direct construction of Tree Attention}: attention scores are built directly from accumulated distances, with a LogSumExp correction that rectifies the raw distance by subtracting the log-sum of edge exponentials. Moreover, the multi-head Mahalanobis distance matrices are themselves repurposed to construct an \textbf{attention meshing mechanism}, enabling cross-head kernel collaboration that simultaneously boosts accuracy and training efficiency.
Extensive experiments demonstrate that MHA-CSP, with only 119K parameters and \textbf{teacher forcing applied exclusively at the final hidden state}, consistently outperforms Transformer and GCN baselines trained from scratch under identical conditions on long-sequence state tracking tasks. While these baselines rely on dense attention or graph propagation, MHA-CSP achieves robust structured reasoning via synthetic distance rectification---powered by Mahalanobis-based attention---and efficient information bypass inherited from the CSP backbone.
This result highlights the effectiveness of complex-valued state propagation with collaborative multi-head rectification in capturing symbolic structures, establishing a new efficiency-performance trade-off for structured reasoning. 

---
# A Behavior-Guided Online Probabilistic Forecasting Method for Electric vehicle Charging Loads 

**Authors**: Chenghan Li, Qingxiang Liu, Yinliang Xu, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2608.24441)  

**Abstract**: Electric vehicle (EV) charging loads exhibit strong behavioral heterogeneity and temporal variability, posing significant challenges for online probabilistic forecasting under evolving operating conditions. In particular, persistent charging patterns may differ substantially across stations, while recent behavioral changes can continuously alter the underlying load distributions. This paper proposes a behavior-guided online probabilistic forecasting framework that explicitly characterizes persistent station-specific patterns and recent behavioral changes. A dual-timescale behavior representation is constructed to distinguish long-term charging characteristics from recent behavioral states and quantify their deviations. These behavioral changes are further semantically encoded to guide drift-aware forecasting adaptation, while a delayed-feedback mechanism ensures temporally consistent online updates when observations become available across different forecasting horizons. Experiments on ten heterogeneous real-world charging stations demonstrate that the proposed method consistently outperforms conventional forecasting models and concept-drift-aware online baselines in forecasting accuracy and probabilistic reliability. For 1-h-ahead forecasting, the proposed method reduces MSE and Pinball loss by 15.3\% and 17.8\%, respectively, over the corresponding best baselines. For 4-h-ahead forecasting, the improvements further reach 16.8\% and 22.6\%, respectively, demonstrating consistent performance gains under evolving charging behaviors and extended forecasting horizons. 

---
# Partial Identification under Causal Orders by Linear Programming 

**Authors**: Eric Rossetto, Alessandro Antonucci  

**Link**: [PDF](https://arxiv.org/pdf/2608.24427)  

**Abstract**: Non-parametric (partial) identification of counterfactual queries typically relies on a fully specified causal graph. Motivated by settings with incomplete domain knowledge, we challenge this requirement by leveraging structural assumptions that are inherently implied by the query itself. We show that any counterfactual inquiry induces a, mostly partial, topological ordering over relevant variables, which, in turn, enables an explicit query parametrisation reducing the identification task to a linear program. This allows bounding arbitrary counterfactual and nested counterfactual queries. Our work can be viewed as a generalisation of the classical bounding framework of Tian and Pearl (2000), originally developed for probabilities of causation. We also prove the \emph{tightness} of our bounds by constructing structural causal models that attain the bounds whilst being compatible with both the observed data and the query-implied order. To assess both the generality and practical utility of the proposed bounding procedure, we revisit several case studies from the literature, demonstrating how the derived bounds can be used to yield informative insights even in the absence of an input causal graph. 

---
# A Judge Should Know What Changed:Construct Validity for LLM-as-a-Judge Evaluation 

**Authors**: Jianlin Chen, Wenhui Chen, Ziyao Lin, Chi Man Vong  

**Link**: [PDF](https://arxiv.org/pdf/2608.24419)  

**Abstract**: LLM-as-a-judge evaluation is usually assessed by agreement and robustness to surface perturbations, but reliability does not establish construct validity. We formalize construct validity for an evaluator as a two-dimensional profile: invariance S, the probability that a verdict is unchanged under construct-preserving edits, and construct sensitivity R, the probability that it changes under minimal construct-changing edits. We show that S and R are independent and that no scalar summary preserves all relevant comparisons. We measure the profile across 7 judges and 4 domains using 7 construct-changing intervention types and 5 register-only controls, with intervention direction determined by human annotators and generation, verification, and judging assigned to disjoint model families. At matched invariance S >= 0.90, judges average S = 0.945 but R = 0.319. Sensitivity also differs between scope and strength edits: R_scope = 0.383 versus R_strength = 0.262, a +0.121 gap with the same sign for all 7 judges. We further audit five public label sets and find that surface-only predictors reproduce 55%-67% of labels in paired mode, including 67.4% of MT-Bench human votes. These results show that high judge agreement can coexist with weak sensitivity to changes in the construct being evaluated, motivating joint reporting of invariance and sensitivity and auditing the validation set itself. 

---
# ResiSpec: Enhancing Multi-Candidate Speculative Sampling via Residual Distribution Shaping 

**Authors**: Zhi-Kai Chen, Jun-Jie Tao, Wei-Xiang Mao, De-Chuan Zhan, Han-Jia Ye  

**Link**: [PDF](https://arxiv.org/pdf/2608.24411)  

**Abstract**: The efficiency of Large Language Model (LLM) serving is fundamentally limited by the sequential nature of autoregressive decoding. Speculative Decoding (SD) mitigates this by using a lightweight draft model to speculate future tokens, which are then validated by the LLM in a single parallel forward pass. To further boost efficiency, multi-candidate schemes propose diverse candidate sets to increase the likelihood of token acceptance. However, we show that these schemes are bottlenecked by Residual Drift: a phenomenon where the rejection of initial candidates causes the residual target distribution to diverge from the draft model's predictions. This shift renders subsequent candidates ineffective and forces the system into expensive resampling. To resolve this, we propose ResiSpec, a framework that strategically reforms the proposal distribution during verification to anchor the residual target mass within the draft model's high-confidence regions. By mathematically re-aligning the verification process without compromising output exactness, ResiSpec prevents candidate obsolescence and achieves up to 1.92$\times$ speedup over state-of-the-art multi-candidate methods. Code is available at this https URL. 

---
# Do Recipes Have Personas? Characterizing and Generating Creator Style in Attributed Procedural Graphs 

**Authors**: Lei Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2608.24369)  

**Abstract**: While large language models (LLMs) possess vast zero-shot procedural knowledge, their tendency to produce homogenized logic often obscures the unique, idiosyncratic execution processes of individual human creators. In this paper, we investigate the computational discovery of procedural personas from unstructured data. To achieve this, we introduce ViralRecipesTrans, a new dataset of procedurally aligned execution flow graphs extracted from popular culinary video transcripts and explicitly mapped to specific creators. We formulate procedural stylometry as a graph learning and process discovery task, revealing a fundamental duality: while traditional lexical classifiers overfit via semantic leakage, discrete topological metrics successfully capture the rigid physical constraints of a creator's workflow. Building upon this characterization, we extend our framework into a novel generative task--predicting a creator's exact structural execution graph for unseen dishes. We expose a fundamental dichotomy in style generation between global macro-planning and local structural execution. Our results demonstrate that few-shot LLMs dominate semantic assignment but suffer from persistent macro-planning deficits, whereas our structured two-stage model achieves superior topological control via rigid Markovian priors. Together, an ensemble approach to procedural generation combines the strengths from both sides, dynamically synthesizing global semantic reasoning with localized topological footprints to automate the discovery and generation of personalized workflows. 

---
# From State to Action: OODA-Tool for Reliable Multi-Turn Tool Use 

**Authors**: Rongfeng Guo, Yinxuan Huang, Yusen Wu, Maoqing Zhong, Yunlu Chen, Meng Tang, Teng Long, Vincent Tao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2608.24368)  

**Abstract**: Reliable multi-turn tool use requires an agent to preserve an evolving task state and ensure that each action remains consistent with it. However, direct function-calling and ReAct-style policies learn state tracking and action generation within the same autoregressive trajectory. This coupling creates state-action competition: the pressure to produce the next call can overwrite or ignore information accumulated earlier in the interaction. Inspired by Boyd's Observe-Orient-Decide-Act cycle, we introduce OODA-Tool, a typed closed-loop policy designed to mitigate this competition by separating state preservation from action realization. Rather than generating an action directly from the interaction history, OODA-Tool routes each decision through controller-checked intermediate states, ensuring that the final output remains grounded in the current task state. Specifically, Observe reconstructs the task state, Orient determines whether execution is warranted, Decide forms an admissible action structure, and Act realizes the external output. We evaluate OODA-Tool against direct function-calling and ReAct policies using Qwen3 models ranging from 0.6B to 14B across multi-turn, multi-tool, and incomplete-information settings. OODA-Tool consistently improves task success across model sizes, with larger gains on smaller models and on tasks whose actions depend strongly on information accumulated across turns and prior tool results. Controlled variants, stage-level ablations, and transfer evaluations further demonstrate the robustness of these improvements. 

---
# Adaptive Influence Graphs for Failure Attribution in Multi-Agent Systems 

**Authors**: Yarden Bakish, Amir Dudai, Roy Ganz, Oren Nuriel, Elad Ben Avraham, Mor Shpigel Nacson, Ron Litman  

**Link**: [PDF](https://arxiv.org/pdf/2608.24361)  

**Abstract**: Multi-agent LLM systems are increasingly deployed in real-world applications, where failures can be costly and difficult to localize. Despite growing efforts to automate failure attribution, diagnosing failed runs still largely relies on human engineers. Yet engineers rarely debug complex systems by reading raw logs end to end. Instead, observability tools organize traces around components, actions, and dependencies to support targeted navigation. We hypothesize that modern LLMs can benefit from the same paradigm. To test this hypothesis, we introduce Adaptive Influence Graphs (AIGs), a two-stage agentic framework that first transforms a failed trace into a structured graph and then navigates it to identify the critical error. Across multiple models, we show that richer trace representations consistently improve failure attribution, with adaptive graph construction and agent-directed traversal yielding the strongest results. AIGs establish a new state of the art on Who&When, the standard benchmark for multi-agent failure attribution. This affirms our hypothesis that attribution depends not only on the diagnosing model, but also on how the trace is represented and explored. 

---
# The Handoff Tax: Continuing Non-Native Trajectories in LLM Agents 

**Authors**: Roy Ganz, Mor Shpigel Nacson, Adi Kalyanpur, Ron Litman  

**Link**: [PDF](https://arxiv.org/pdf/2608.24358)  

**Abstract**: Coding agents perform long-running tasks spanning dozens of model calls, tool uses, and code edits. As these runs unfold, users face a practical cost-quality trade-off: escalating to a stronger model when a cheaper one struggles, or downshifting once the hard reasoning is complete. Each switch requires the receiver to continue a non-native trajectory produced by another model. We study how this handoff affects quality and cost, and how varying the trajectory information inherited by the receiver changes the outcome. Using pairs of low-cost, low-capability (LC) and high-cost, high-capability (HC) models from the Claude and GPT families, we vary handoff direction, timing, and interface, comparing full-trajectory transfer, compaction, and trajectory removal while preserving the repository state. Across both model families, full-trajectory escalation recovers less than half of the LC-to-HC quality gap while incurring a substantial cost premium. We term this cost-quality penalty the handoff tax. By contrast, downshift offers a favorable cost-quality point. Interestingly, the preferred interface also reverses with direction: reducing LC-model trajectory information improves escalation quality, whereas removing the HC-model trajectory reduces downshift quality. 

---
# Selective Regenerative Decoding: Trajectory-Level Intervention for Inference-Time Reasoning 

**Authors**: Sophia Xiao Pu, Yumo Xu, Sailik Sengupta, Millennium Bismay, Ruixue Lian, James Gung, Yi-an Lai, Arshit Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2608.24338)  

**Abstract**: Inference-time decoding methods improve LLM reasoning by exploring multiple candidate trajectories, yet treat each trajectory as atomic: either retaining it whole or discarding it irreversibly. This wastes computation on partially promising candidates whose high-quality prefixes are abandoned alongside degraded suffixes. We introduce Selective Regenerative Decoding (SRD), which routes each candidate to discard, keep, or refine only the degraded portion of the suffix while preserving the useful prefix of borderline candidates, without requiring a larger target model. Under mild assumptions, SRD achieves a provable 1.28-to-1.36-fold gain in sample efficiency over rejection sampling with strictly higher expected trajectory quality, with the gain growing as the candidate pool grows. Across MATH500, GPQA Diamond, HotpotQA, and AlpacaEval with multiple generation-reward model pairs, SRD matches Best-of-N accuracy with substantially fewer generated tokens and outperforms speculative rejection in low-compute regimes. By enabling segment-level intervention rather than whole-trajectory selection, SRD opens a previously underexplored region of the accuracy-compute tradeoff for inference-time reasoning. 

---
# SonarLLM: A Native Sonar--Optical Multimodal Large Language Model for Underwater Perception 

**Authors**: Cong Su, longxuan ma, Ling Dong, Guofeng Tang, Weijie Yin, Haohui Chen, Zhengtao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2608.24325)  

**Abstract**: Reliable underwater perception requires complementary sensing under variable visibility. Optical cameras capture appearance and semantics but degrade rapidly with turbidity, whereas imaging sonar preserves geometry while exhibiting distinct range-azimuth structure and acoustic artifacts. Existing MLLMs, built primarily on optical encoders, are therefore ill-suited to model sonar or adaptively exploit sonar-optical complementarity. We propose SonarLLM, a sonar-optical MLLM that treats sonar as a native perceptual modality. It combines a sonar-specific encoder, modality-specific physics-aware feature enhancement, and reliability-aware hierarchical fusion to align acoustic structure with optical semantics and dynamically adjust their contributions as sensing quality changes. We also introduce SonarBench, a paired benchmark that spans four tasks: recognition, counting, visual question answering, and captioning; and, across the benchmark, three input settings: sonar-only, optical-only, and fusion. By fixing the scene and sonar observation while varying optical degradation, SonarBench enables controlled measurement of cross-modal complementarity. SonarLLM achieves 72.0% macro accuracy across sonar-only recognition, counting, and VQA, outperforming the strongest baseline by 34.4 percentage points, and 68.7% under fusion, exceeding the best baseline by 25.1 points. For recognition and counting, the fusion-over-optical gain grows from 6.0 to 36.0 points as turbidity increases, indicating the increasing complementary value of sonar under controlled optical degradation. Together, these results show that robust heterogeneous perception depends not only on adding sonar, but on representing and weighting it according to its sensing characteristics. 

---
# Can a Dynamic Internal Field Govern a Transformer's Cognition? Certifiability, not Superiority, in Homeostatic Compute Control 

**Authors**: Francisco M. Arrabal-Campos, Ignacio Fernandez, Francisco G. Montoya, Alfredo Alcayde  

**Link**: [PDF](https://arxiv.org/pdf/2608.24319)  

**Abstract**: An intelligent system does not merely reason: it governs its own reasoning - how much to compute, when to stop, which module to activate. Can that role be played by a dynamic internal field - a low-dimensional homeostatic state with explicit physics and certified stability - that modulates cognition without performing it? Ours is a field on the module graph governed by a family of PDEs on the graph Laplacian, advancing with an adaptive-depth reasoner. We certify the stability of the integrator of the whole family - an integrator certificate, not a closed-loop one. New, and proved here: a discrete Schur-Cohn criterion for Verlet with velocity coupling, necessary and sufficient per latent root, with no commutation hypothesis. The answer is threefold: substance no, structure only in part, certifiability yes. The type of the field's physics is irrelevant for accuracy: wave, diffusion, gated mixtures and a 2D Navier-Stokes substrate tie. A twenty-seed preregistered deconfounding campaign bounds the structural claim: at equalized caps the second-order effect is strong in one family (+0.087 [+0.042, +0.132], t=4.0) but is not detected in the other (+0.014 [-0.013, +0.040], n.s.), so part of the original contrast was capacity, not order; and a matched-interface GRU is indistinguishable in the first and nominally exceeds the field in the second (-0.035 [-0.067, -0.002]). What distinguishes the field is not capability but that its one-step operator admits an exact runtime stability check - a difference of kind, not of existence: learned recurrences carry certificates too, sufficient and conservative ones. A kill-gate with a positive control finds no evidence for the field as evidence accumulator (Delta AUC +0.0007 [-0.0065, +0.0079] vs a 0.03 threshold). A dynamic internal field is a viable, certifiable compute governor, but not an enhancer of cognition: it modulates, it does not think. 

---
# Benchmarking LLM Judges for Voice-Agent Evaluation: Reliability, Calibration, and Human Oversight 

**Authors**: Anupam Purwar, Shashank Singh, Kritika Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2608.24314)  

**Abstract**: Evaluating conversational voice agents at scale re- quires reliable assessment methods that capture both observ- able interaction quality and the contextual judgment typically provided by human evaluators. We investigate LLM-as-a-Judge evaluation by comparing human judgments with GPT-4.1 and GPT-5 on telecom and retail voice-agent conversations, across conversational quality and safety dimensions. The same interac- tions are scored under three evaluation configurations, p0, p1, and p2, to test whether automated judgments are sensitive to the evaluation setup and whether observed patterns generalize across configurations and judge models. Beyond aggregate agreement, we examine metric-level correlations, evaluator consistency, and systematic human-LLM disagreement to identify which conver- sational attributes can be judged reliably by automation and which remain sensitive to interpretation and context. Effective voice-agent evaluation is also shaped by pipeline-level factors such as speech generation, streaming, and error propagation across ASR, reasoning, and tool-calling stages, motivating our focus on comparing how human and LLM judges score the same interactions end to end. Our results show that LLM- based evaluation can serve as an effective component of large- scale voice-agent assessment, but that its reliability is metric- and configuration-dependent rather than uniform. This pro- vides an empirical framework for identifying which metrics suit automated evaluation and supports hybrid pipelines in which LLM judges handle scalable assessment while human evaluators remain engaged for metrics that demand contextual interpretation and higher-confidence judgment. 

---
# OPDSearch+: On-Policy Distillation with RL Refinement for Search-Augmented Reasoning 

**Authors**: Qinglin Ye, Zhiyuan Gu, Jingjie Xia, Yiheng Zhang, Kaiyan Zhao, Shunchao Zheng, Yuhang Mu, Wenchao Du, Yiming Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.24310)  

**Abstract**: Search-augmented reasoning remains difficult for small language models. On-policy distillation (OPD) from trained teachers offers a promising direction, but suffers from two issues: (1) high-quality multi-turn search trajectories depend on dynamic retriever responses, making SFT data prohibitively expensive to collect at scale; (2) task-specifically trained teachers incur substantial training cost, while directly applying OPD with an off-the-shelf teacher without task-specific fine-tuning constrains the student to the teacher's performance ceiling and suffers from severe training instability. We propose OPDSearch+, the first distillation paradigm that requires no teacher fine-tuning for search-augmented reasoning. We investigate the role of a frozen off-the-shelf instruct model as the teacher in on-policy distillation, and reveal a key insight: the teacher reshapes the student's policy distribution so that subsequent RL converges to a superior solution that RL alone cannot reach. In stage one, the student interacts with a live search engine and is distilled via a per-position forward KL objective, transferring reasoning decomposition and evidence integration skills without any task-specific teacher training. In stage two, RL refines the distilled student from a richer behavioral foundation, achieving performance that RL alone cannot reach from scratch. Across seven QA benchmarks, OPDSearch+ with a 3B model consistently outperforms all prior 3B RL baselines, achieving gains of 13.1% on HotpotQA and 8.5% on 2WikiMultihopQA. 

---
# VideoHarness-RSI: Recursive Harness Self-Improvement for Long-Video Understanding with Frozen Vision-Language Models 

**Authors**: Guoyang Xu, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.24302)  

**Abstract**: Long-video understanding depends critically on how a limited model context is constructed from a much longer video. Existing approaches improve this process through compression, retrieval, memory, and agentic evidence acquisition, but these mechanisms are typically introduced as part of a manually designed inference system or optimized together with other components. This makes it difficult to isolate a simpler question: how much can be gained by improving the executable context-construction program alone? We study this question through VIDEOHARNESS-RSI, a controlled baseline for recursively searching executable context constructors around a frozen vision-language model (VLM). An outer-loop proposer uses prior programs, evaluation outcomes, and execution traces to generate candidate harnesses, which are executed and evaluated end to end before successful variants are retained for further search. This makes long-video understanding a controlled instance of automated harness design: the searchable object is executable program structure, while the answering model and interface remain fixed. Starting from uniform sampling, recursive harness search consistently finds room for improvement and surpasses several weaker hand-crafted baselines. Starting instead from a stronger hand-crafted baseline, the same RSI process yields a further improvement. The selected harness also transfers to additional long-video benchmarks without further search. Together, these results establish executable context construction as a distinct optimization layer and provide a reproducible baseline for studying harness discovery and transfer around frozen VLMs. 

---
# ReproAgent: Contract-Guided Paper-to-Code Reproduction 

**Authors**: Xue Hu, Zewei Pan, Zhongyuan Wang, Zhou Liu, Zeli Su, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.24291)  

**Abstract**: Paper-to-code reproduction asks scientific AI agents to turn research papers into executable repositories that preserve the paper's method, protocol and artifacts. This is difficult because the specification is split: explicit paper content such as algorithms, metrics and artifacts is often lost across long agent trajectories, while implicit details such as framework defaults and conventions inherited from related work are absent from the paper. We introduce ReproAgent, a four-stage Prepare--Plan--Generate--Repair pipeline built around a persistent implementation contract with two channels: an implementation-requirement channel that turns paper snippets into code obligations, and a reference-evidence channel that retrieves content and structure evidence from related repositories. Both are bound to work packages, projected into file-level contracts, and consumed across generation and repair. On PaperBench Code-Dev, ReproAgent reaches the highest mean score among same-backbone scaffolds under both Claude-Sonnet-4.5 and Gemini-3-Flash. End-to-end channel ablations and per-paper cases support the contribution of both channels. Code and experimental artifacts are publicly available. 

---
# RePolicy: Reinforcement Learning for Safety-Policy Invocation in Agent Safeguards 

**Authors**: Houcheng Jiang, Boxuan Zhang, Qiyong Zhong, Junfeng Fang, Xiang Wang, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2608.24275)  

**Abstract**: Safeguarding language model agents requires assessing complete execution trajectories under context-dependent safety policies. Existing policy-aware safeguards mainly rely on prompting or supervised fine-tuning, limiting their ability to adapt to unseen trajectories and changing policy contexts. We propose RePolicy, an agent safeguard that learns safety-policy invocation through reinforcement learning. Given an agent trajectory and a dynamic policy library, RePolicy invokes the applicable policy and uses its content to produce a policy-grounded rationale and safety judgment. We construct PolicyTraj-20K to support supervised initialization, followed by GRPO with verifiable rewards and policy-context perturbation. Experiments across six agent safety benchmarks show that RePolicy achieves strong overall safety-detection performance and robust policy invocation under varying policy contexts. 

---
# Eating for a Sustainable Planet: Personalized Sustainable Diet Recommendation via Constraint-Aware Decision-Making Modeling 

**Authors**: Ying Jin, Weiqing Min, Mingyu Huang, Shuqiang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2608.24274)  

**Abstract**: A sustainable diet represents a multi-dimensional synergy among four essential pillars: nutrition adequacy, economic affordability, cultural acceptability, and environmental respect. Despite the prevalence of population-level sustainability modeling, practical implementation relies on effective individual-level adoption. This transition is often hindered by inter-individual heterogeneity, posing a formidable challenge in aligning sustainable diet requirements with individual preferences. To address this issue, we propose a personalized sustainable diet recommendation model based on a constraint-aware decision-making mechanism, where sustainability is incorporated through learnable constraints rather than modeled as user preferences. To systematically evaluate the proposed approach, we construct a sustainable diet dataset named SusDiet with about 150k recipes, characterized by broad coverage of sustainability indicators. Experimental results on this dataset show that our method promotes more sustainable choices without compromising individual preference. This work establishes a framework for aligning individual dietary choices with planetary health, offering quantitative evidence to guide future sustainable diet interventions and policy-making for sustainable development. 

---
# Matched Excess-Outranker Regularization for Candidate-Set Interference in Continual Knowledge Graph Embedding 

**Authors**: Hao Ren, Junbin Gao, Jiaojiao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2608.24273)  

**Abstract**: Continual knowledge graph embedding updates entity and relation representations as a graph grows. Existing methods primarily address catastrophic forgetting, but entity admission also changes the candidate universe of every compatible query. A historical answer can therefore lose rank even when its score and its ordering among old entities are preserved. We formalize this effect as candidate-set interference and introduce Matched Excess-Outranker Regularization (MEOR), a host-level objective that compares smooth answer-relative newcomer pressure with score-blind, structurally matched old references. Its one-sided penalty acts only when newcomer competition exceeds the matched reference, preserving the host learner's signal for legitimate new entities. Across eight paired runs on ENTITY-ComplEx, MEOR improves historical current-universe mean reciprocal rank (MRR) by 0.0057 over replay and reduces candidate-set interference by 0.0055, with one-sided 95% lower bounds of 0.0052 and 0.0051, respectively. It satisfies the preservation criteria for old-universe ranking and newcomer acquisition and improves historical current-universe MRR over persistent calibration, matched maximum regularizer (MMR), and unmatched old regularizer (UOR). Direct ablations support each component of its reference construction and aggregation. Adding MEOR also improves historical ranking in all ten reported FBInc-S and FBInc-L host and backbone settings, with every paired 95% confidence interval excluding zero. These results establish candidate admission as a distinct source of continual rank loss and show that it can be controlled without replacing the underlying embedding architecture or continual learner. 

---
# Real-World Knowledge-Guided Change Data Synthesis for Remote Sensing 

**Authors**: Yaoyi Qi, Xingxing Weng, Chao Pang, Yongkang Cui, Xiangyu Hao, Xiaokang Zhang, Guibo Zhu, Gui-Song Xia  

**Link**: [PDF](https://arxiv.org/pdf/2608.24263)  

**Abstract**: Change data synthesis provides a cost-effective solution for expanding training data and improving the performance of change detection models. However, existing synthesis methods typically rely on handcrafted rules to simulate changes, where limited coverage of class transitions restricts the diversity of synthesized data, while predefined transition designs limit their flexibility in accommodating varied change types. In this work, we introduce KnowChange, a knowledge-guided change data synthesis framework that leverages pretrained vision-language models as knowledge sources to reason about plausible change locations and class transitions from pre-change scenes and desired change types. By integrating knowledge-guided change simulation with generalizable synthesis models, KnowChange enables flexible synthesis of diverse change types within a unified framework. Extensive experiments demonstrate that KnowChange-generated data consistently outperforms existing synthetic datasets in both synthetic-to-real transfer and synthetic data augmentation, despite being generated at a compact scale. Further analyses show that the knowledge-guided change simulation can be seamlessly integrated into existing synthesis pipelines and enhance the downstream utility of synthesized data. 

---
# Beyond Accuracy: A Dual-Judge Evaluation Protocol for Vision-Language Models in Legally Grounded Tasks 

**Authors**: Su Myat Noe, Ha Thanh Nguyen, May Myo Zin, Ken Satoh  

**Link**: [PDF](https://arxiv.org/pdf/2608.24258)  

**Abstract**: AI systems are increasingly evaluated for legally accountable settings, where correct outputs must also be justifiable against an applicable legal standard. Existing legal-AI benchmarks and LLM-as-judge protocols provide important infrastructure for measuring task performance and open-ended response quality. We contribute one additional evaluation signal: a dual-judge protocol that pairs a standard 0-10 quality judge with a strict binary semantic-equivalence judge against a human-curated reference.
We study a controlled, visually grounded regulatory task - UK traffic-sign interpretation, whose meaning is a codified question with a known reference for every input - and measure not merely whether the two judges disagree (by construction they must) but how much and where. On 4,680 evaluations under seven visibility levels and two occlusion modes, the two judges are moderately associated (point-biserial r = 0.644), while revealing an asymmetric Type II pattern affecting 8.0% of all evaluations.
Its distribution is instructive: the marginal rate peaks at high visibility (14.2% at v = 0.8) simply because high-scoring answers are common there, but conditioned on the answer already scoring above 7, the rate is highest under heavy occlusion (54-63% at v <= 0.3), so a high quality score is least trustworthy when the input is most degraded.
We are explicit that the signal is a property of this judge and reference: a 49-row human check shows the 0-10 judge aligns closely with everyday-reader judgement (Pearson r = 0.81; r = 0.80 with the LLM accuracy sub-score), while the equivalence judge is fairly but one-directionally stricter. The protocol adds one LLM call per evaluation and surfaces a signal single-judge protocols do not report. We release the prompt template, occluded variants, and full evaluation results. 

---
# SA-Bench: Evaluating Semantic Alignment in LLM-Based Paper Reproduction 

**Authors**: Xue Hu, Zewei Pan, Zeli Su, Zhou Liu, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.24252)  

**Abstract**: LLM agents can generate paper reproduction code, yet often produce scientifically unfaithful implementations. We define this failure mode as semantic drift, where generated code silently diverges from the paper's specifications. We introduce SemanticAlign-Bench(SA-Bench), a diagnostic benchmark covering 30 papers from ICLR, ICML and NeurIPS 2025. For each paper, we decompose its specifications into atomic and verifiable implementation claims, which we call Semantic Alignment Units (SAUs) and evaluate repositories along four diagnostic dimensions spanning numerical, methodological, protocol and ordering drift. In total, we construct 1,491 SAUs across five ML domains and evaluate 12 generator configurations (4 models $\times$ 3 scaffolds). Even the strongest configuration (Claude+PaperCoder) achieves a mean SAU score of only 0.301 out of 1.0, with an overall mean of 0.221 across 360 evaluations. A failure taxonomy reveals that agents attempt most requirements but implement them incorrectly, with implementation mismatch and stubs accounting for the majority of zero-scored claims. Our analysis further indicates that scaffolds optimized for executability provide limited leverage for scientific reproduction; narrowing the gap requires scaffolds that prioritize semantic specification verification. The benchmark, annotations and evaluation pipeline are publicly available. 

---
# STRIVE: Multi-Agent Structured Temporal Reasoning with Integrated Verification for Longitudinal Radiology Report Generation 

**Authors**: Junyeong Maeng, Eunsong Kang, Heung-Il Suk  

**Link**: [PDF](https://arxiv.org/pdf/2608.24237)  

**Abstract**: Longitudinal radiology report generation (LRRG) requires identifying both current findings and their changes relative to a prior study. Existing methods jointly model diagnosis, attribute estimation, temporal comparison, and language generation within implicit representations, which can cause task interference, obscure the evidence underlying each decision, and limit error traceability. They also model progression states as independent labels, ignoring their ordered structure and thus treating missed changes and direction reversals equally. We present STRIVE, Multi-Agent Structured Temporal Reasoning with Integrated Verification for LRRG, which decomposes clinical reasoning into specialized Diagnosis, Attribute, and Temporal Change Agents that produce explicit intermediate evidence. In particular, the Temporal Change Agent is further post-trained using Progression-Aware GRPO, a verifiable, shaped reward that assigns partial credit to direction-preserving errors while scoring direction reversals lowest. STRIVE performs verification at two stages: a deterministic Consistency Gate reconciles the agent outputs before report generation, and a Validation Agent checks whether the generated report is supported by the aggregated clinical evidence. On Longitudinal-MIMIC, STRIVE attains the best clinical efficacy among recent methods and more than doubles Longitudinal Change Concordance (LCC), a measure of temporal agreement with the reference report, over the strongest baseline. 

---
# TRACE: An Evidence-Grounded Benchmark for Safety Evaluation of Large Reasoning Models 

**Authors**: Zhenyu Wu, Siyuan Chen, Changchun Yang, Jiaqi Dong, Min Zhou, Ali Almadan, Talal Hammad, Faisal Wahbo, Aminullah Tora, Mona Alshahrani, Xin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2608.24232)  

**Abstract**: Large Reasoning Models (LRMs) generate intermediate reasoning traces that may contain unsafe content, even when their final responses appear safe. Guardrail models are designed to detect and block unsafe content, yet existing benchmarks for unsafe content detection focus primarily on prompts and final responses, leaving reasoning traces largely unexamined. Moreover, these benchmarks typically provide only binary safety labels, without evidence annotations that justify the judgments. To address these limitations, we introduce TRACE, an evidence-grounded safety evaluation benchmark that covers the entire LRM inference pipeline: prompts, reasoning traces, and final responses. TRACE includes prompts in two languages spanning nine risk categories and ten attack strategies. For each prompt, four LRMs generate reasoning traces and final responses, and we annotate the safety of each component and extract supporting evidence from the corresponding source text. Evaluating 18 guardrail models on TRACE reveals that safety judgment for reasoning traces is substantially more challenging than for prompts or final responses, and that current models struggle to accurately extract supporting evidence. These findings highlight the need for guardrail models that can reliably detect and precisely localize unsafe content across the LRM inference pipeline. 

---
# Evaluating Multiple LLM Generations with Validated Task Coverage 

**Authors**: Florian Le Bronnec, Rio Yokota  

**Link**: [PDF](https://arxiv.org/pdf/2608.24228)  

**Abstract**: Many LLM applications are most useful when they provide several candidate outputs for comparison, validation, or combination. Predominant evaluation settings, however, still focus on individual outputs or reduce multiple samples to a single success or selected answer. This can miss whether the outputs include several genuinely different useful results. We introduce VTC-Bench, a five-domain benchmark for this setting, together with Validated Task Coverage (VTC) as its core evaluation quantity. The benchmark is built from carefully selected real-data tasks where both output quality and task-relevant distinctness can be checked automatically and reproducibly, without model-based judges. VTC measures how many distinct useful results are obtained within $k$ attempts. Across multiple models and inference settings, the benchmark leads to different conclusions from conventional evaluation: configurations that look strongest from single-draw quality are not necessarily those with the best coverage, and simple measures of output variation do not reliably recover task-relevant coverage. These results show that finite candidate sets can be evaluated directly as objects of interest, revealing differences in model behavior that are not apparent from conventional per-output evaluation. 

---
# Constraint-Guided Enterprise Data Mapping with Large Language Models 

**Authors**: Sebastian Monka, Pramod Anantharam, Thien Vo Minh, Lavdim Halilaj  

**Link**: [PDF](https://arxiv.org/pdf/2608.24218)  

**Abstract**: Enterprise entity alignment must handle semi-structured records, implicit attributes, and unit or granularity mismatches. Manual matching is still common in practice, but does not scale as schemas and providers evolve. LLM-only matching improves semantic recall, yet can violate structural and physical invariants, producing fluent yet operationally invalid correspondences.
We propose constraint-guided mapping (CGM), a neuro-symbolic method with three stages: (i) schema-grounded admissibility constraints with metadata mc = <tau_c, delta_c>, where tau_c denotes the constraint type and delta_c provides executable relation and normalization logic; (ii) constraint-restricted candidate generation with cascade relaxation to guarantee a nonempty feasible set under noise; and (iii) neural ranking with bounded LLM disambiguation restricted to that feasible set.
Methodologically, constraints operate as hypothesis-space operators rather than post-hoc validators, enabling controlled degradation under relaxation and auditable, human-guidable decisions. On a controlled structural-decoy benchmark, hard admissibility shrinks the candidate space by ~480x without dropping the GT, and a layer-by-layer ablation shows this gate, not the LLM, is the decisive lift (F1 0.08 to 0.66). The benefit is model-independent and adds no extra inference cost: a small model with constraints matches a frontier LLM used without them at ~28x lower cost. The method, not a single tuned configuration, transfers across seven enterprise makes (macro F1 0.70), each under its own automatically discovered, expert-refinable constraints, and lowers expert effort by ~7x versus spreadsheet workflows. Public Valentine results add an external ranking sanity check and mark the boundary: constraints should be hard only where structural invariants are match-determining. 

---
# MetaRAG: Belief-Action Aligned Policy Optimization for Agentic RAG 

**Authors**: Qiuyi Qi, Tian Liang, Jiamu Wang, Jinjian Zhang, Wei Zhou, Pengcheng Zhu, Linjian Mo, Ming Kong, Jie Liu, Qiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.24214)  

**Abstract**: Agentic retrieval-augmented generation (RAG) requires language models to decide when to continue searching and when to answer. Existing RL-based methods rely on external supervision and overlook the agent's internal belief about whether the current evidence is sufficient. To address this problem, we reformulate the search decision quality as belief-action alignment and propose MetaRAG, a belief-action aligned policy optimization framework for agentic RAG. MetaRAG uses Verify-first Action Generation to elicit an explicit verification process before each actual action, and Internal Belief Probing to estimate the policy model's own answerability belief from the same question-history context. Based on these, MetaRAG derives a consistency reward that is further gated by answer correctness, avoiding reinforcement of internally consistent but incorrect trajectories. The belief probe is used only during training and introduces no inference-time overhead. Experiments on seven public QA benchmarks show that MetaRAG consistently improves the accuracy-efficiency trade-off over strong RL-based agentic RAG baselines, with gains that transfer to deep research settings, different optimizers, and multiple model backbones. 

---
# Preference Data Selection for Mitigating the Alignment Tax in Large Language Models 

**Authors**: Minsu Kim, Jianxun Lian, Xing Xie, Steven Euijong Whang  

**Link**: [PDF](https://arxiv.org/pdf/2608.24192)  

**Abstract**: Aligning large language models to human preferences is crucial for real-world deployment but frequently incurs an alignment tax, leading to the catastrophic forgetting of pre-trained general capabilities. While previous works primarily frame this problem as an optimization or architectural challenge, the inherent characteristics of preference data that drive this degradation remain largely underexplored. In this paper, we propose BALIGN, a balanced data selection strategy that explicitly mitigates catastrophic forgetting while optimizing alignment efficacy. Through theoretical and empirical analyses of the preference optimization gradient, we identify three key data-centric features that dictate parameter drift: the reference model's log-probability margin, the token length difference between chosen and rejected responses, and the TF-IDF similarity to general capability corpora. By aggregating these orthogonal features into a unified composite risk score, BALIGN systematically filters out high-risk preference samples that disrupt intrinsic model parameters or provide minimal alignment utility. Extensive experiments on standard human preference datasets demonstrate that BALIGN strongly preserves foundational capabilities without compromising alignment gains, consistently achieving the optimal Pareto frontier with minimal computational overhead. 

---
# Paritok-4B: Intent-Conditioned Context Compression for Coding Agents 

**Authors**: Jiayu Shi, Luzhuo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.24188)  

**Abstract**: Coding agents re-send large file reads and tool outputs to a frontier LLM every turn, and this context dominates their token bill. General-purpose prompt compressors are trained on prose and suit code poorly: they paraphrase identifiers and drop the exact spans an agent needs to edit. We present Paritok-4B, a 4B LoRA compressor for coding-agent trajectories built on two commitments. It is extractive: it selects spans rather than rewriting them, and 96.0% of the identifiers, paths, and numbers it emits already appear in its input, holding at 96.2% on held-out SWE-bench Lite output. It is intent-conditioned: told the agent's current task, it acts chiefly inside a retained segment, selecting which lines survive (retained lines are +0.067 more intent-relevant than removed ones, paired 95% CI [+0.056, +0.078]) rather than changing how much is retained. We distil a gpt-4.1-mini teacher over 67,074 real OpenHands trajectories into 40,606 validated examples and fine-tune Qwen3-4B. On all 300 SWE-bench Lite instances, Paritok-4B compresses agent context to 25.7% of its size, 2.0x harder than a gpt-4.1-mini compressor (50.2%) and 2.4x harder than gpt-5 (61.9%), while retaining 86.5% of uncompressed single-shot solve quality. Fed the cat -n line-numbered input real agents produce, it compresses slightly less (27.8%) and retains more (89.3%); there the paired test is informative, with 30 instances solved only uncompressed and 17 only compressed, an exact McNemar p=0.079, so at this sample size compressing context to roughly a quarter of its size does not significantly reduce the solve rate. The model is a 264 MB adapter that self-hosts on one 24 GB GPU with no per-token compressor fee, which at list prices decides the economics: gpt-5 as a compressor is net-negative, costing more than the downstream tokens it saves. Weights, data, and evaluation scripts are open (Apache 2.0). 

---
# Task-Adaptive Rubrics for GUI Reward Modeling 

**Authors**: Tao Xiong, Xavier Hu, Wenkai Wang, Qinzhuo Wu, Changqiao Wu, Pengzhi Gao, Wei Liu, Jian Luan, Shengyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.24174)  

**Abstract**: Recent studies on GUI agents have increasingly focused on outcome reward modeling, which assigns outcome rewards by judging whether an executed trajectory satisfies the success criteria implied by the user instruction. Existing GUI reward verifiers, however, often under-specify how these criteria should be constructed for each task instance. Whether using generic rubric structures or implicit model reasoning, their judging criteria are not sufficiently task-adaptive: they can transfer checks across tasks, overlook concrete constraints in the current instruction, or become overly strict by enforcing unstated requirements. To address this limitation, we propose AdaptRubric, a Coarse-to-Fine Rubrics Framework that constructs task-adaptive judging criteria through a category-level coarse stage and an instance-level fine stage. AdaptRubric performs category-level coarse rubric retrieval by routing the instruction to a GUI task family and retrieving reusable task-family criteria, then conducts instance-level fine rubric generation to surface compact cues for concrete values, scopes, and constraints in the current instruction. Across offline reward evaluation and online reinforcement learning optimization, AdaptRubric consistently outperforms prior reward agents, improving F1 by 3.6 points over the baseline average under a matched image budget and yielding a 4.23-point task-success gain. 

---
# OmniJudge or OmniBias? Diagnosing Multimodal Judges through Balanced, Decoupled Lenses 

**Authors**: Guangzheng Hu, Ziyue Jiang, Weixu Qiao, Lixin Zhang, Jianye Kang, Yuru Wu, Rong Bao, Niantong Li, Wei Wang, Ziyi Cheng, Xinfa Zhu, HangRui Hu, Ting He, Bing Zhao, Lin Qu, Hu Wei, Jin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.24160)  

**Abstract**: Multimodal understanding models that can jointly judge text-to-image (T2I), text-to-video (T2V) and text-to-speech (TTS) generation are increasingly used as "OmniJudges" for evaluation and automatic annotation. How reliably they understand what they score remains unclear, since existing benchmarks and training data tend to overemphasize positive examples and to conflate distinct failure modes, so a judge may score well without recognizing failures while its capability gaps stay hidden. Motivated by this, we introduce D3-Omni, a balanced and decoupled benchmark for diagnosing fine-grained multimodal understanding, covering 53 orthogonal binary dimensions (17/22/14) and 10,671 samples (3,526/1,998/5,147) across the three tasks. Rather than re-generating outputs, which may leak information across dimensions, we fix verified fully positive seeds and derive negatives through controlled prompt rewriting and atomic, dimension-isolating perturbations. The resulting D3 design is Dual-balanced, which helps alleviate negative-sample scarcity and per-dimension label imbalance; Decoupled, so that each error is attributable to a single capability; and Dynamic, steering construction toward under-represented regions of the label distribution as generative models this http URL suite reaches near 1:1 per-dimension parity and a uniform distribution over all total-score levels. Under this balanced view, even strong OmniJudges tend to struggle on modality-related dimensions, to confirm satisfied requirements far more reliably than they detect violated ones, and to treat nominally distinct attributes as largely a single decision, suggesting that aggregate accuracy may hide systematic blind spots that a balanced and decoupled lens can help expose and, in turn, address. 

---
# Robust Code RL via Faulty-Code-Driven Test case Synthesis and Dense Reward Shaping 

**Authors**: Yiwen Zhang, Xiaodong Yan, Zhenyu Huang, Deng Zhao, Liang Jiang, Qing Cui, Zujie Wen, Zhiqiang Zhang, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.24135)  

**Abstract**: Reinforcement learning from verifiable rewards (RLVR) has emerged as a pivotal technique for enhancing the code generation capabilities of Large Language Models (LLMs). However, the efficacy of RLVR in coding implementations is fundamentally limited by the comprehensiveness of test cases, because insufficient test coverage in code validation often causes false positives, further leading to reward hacking and policy degradation. To mitigate the reward bias stemming from the suboptimal quality of current automated generation methods, we propose the RobustTests framework, which introduces a faulty-code-driven test case synthesis strategy that leverages "near correct" faulty codes to guide the model in precisely capturing latent logical discrepancies and further integrates validator agents with behavioral feature clustering to facilitate the granular filtering of invalid and redundant test cases. To address false negatives caused by inherent hallucination noise in synthetic test cases, RobustTests also incorporates a stepwise dense reward function based on pass rates, bolstering training robustness through fine-grained feedback. By employing this pipeline, we construct a high-quality dataset that augmented the test cases in CodeContests, encompassing a broader spectrum of faulty code scenarios and significantly enhances diagnostic utility. Experimental results demonstrate that, by leveraging a moderately challenging subset of problems from CodeContests for training, RL fine-tuning of Qwen3-32B via RobustTests achieves an absolute 3% performance gain on the LiveCodeBench benchmark compared to baseline methods, confirming the effectiveness of the RobustTests framework in advancing the code generation proficiency of LLMs. 

---
# AHEAD: Adaptive Hindsight with Environment-Augmented Distillation for Agentic RL 

**Authors**: Xiaolong Jin, Dingmin Wang, Vijay Lingam, Varun Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2608.24114)  

**Abstract**: Training multi-turn LLM agents with reinforcement learning typically relies on trajectory-level rewards, which assign a uniform advantage to every step and cannot identify which decisions led to success or failure. Self-distillation methods can provide finer-grained supervision by augmenting RL with privileged information. However, existing approaches usually apply the same type of privileged information to every step in an indistinguishable manner, ignoring a key asymmetry: routine steps need little additional guidance, while critical error steps require corrective direction that environment feedback alone cannot provide. We propose AHEAD, a step-aware framework that matches different supervision sources to different step types. The teacher receives environment feedback on all steps as a grounded dense signal, and additionally receives LLM-generated corrective hints on error steps to supply the direction that environment feedback lacks. The method introduces minimal changes to the standard GRPO algorithm. Across ALFWorld, WebShop, and Search-based QA, and across three model scales, AHEAD raises task success (+13.3 points on ALFWorld and +11.0 on WebShop at 7B over GRPO), reaches a given success rate in fewer training steps, and solves tasks within tighter interaction budgets than outcome-only RL and prior self-distillation baselines. 

---
# Scalable Question-Centric Text-to-Image Evaluation: Reliable Ranking, Fine-Grained Diagnosis, and Cost-Aware Routing 

**Authors**: Shaoan Zhao, Fang Zhao, Xueqiang Guo, Xinpei Su, Huanlin Gao, Qiang Hui, Ting Lu, Fuyuan Shi, Chao Tan, Bikun Yang, Kai Wang, Shiguo Lian  

**Link**: [PDF](https://arxiv.org/pdf/2608.24112)  

**Abstract**: Modern text-to-image (T2I) models often have similar total scores but different strengths, making practical selection difficult. Fine-grained benchmarks decompose prompts into questions, yet often return them to prompt scores and fixed categories, weakening attribution and ignoring complexity. Related requirements are also scored separately or as one total, obscuring basic versus compositional failure. We present QC-T2I-Bench, a question-centric framework that converts open prompts into attributed atomic questions and organizes their dependencies with Davidsonian Scene Graphs (DSGs). We use hierarchy-constrained question aggregation to exclude downstream questions after a prerequisite fails and to prevent simple and complex prompts from receiving the same total weight. We then use the DSG structure to measure joint success within prompts and compare repeated entities across prompts, separating basic realization failures from failures under additional requirements. We evaluate multiple open-source T2I models on English and Chinese prompts. The resulting question-level evidence supports reliable ranking and fine-grained diagnosis: joint completion falls from 80.7\% for components with two capabilities to 37.2\% for those with seven or more. Finally, we reuse the same records for training-free routing; our cost-aware router matches ERNIE's 89.51-point estimate with 21.3\% less GPU-s/MP. 

---
# ACE: A Self-Correcting Agentic Canvas Editor for Multi-Slide Presentation Automation 

**Authors**: JooYoung Jang, Taegyeong Lee, Jihyeon Park, Nojun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2608.24103)  

**Abstract**: Commercial design platforms increasingly edit documents through large language model (LLM) agents, but two practical problems block reliable deployment: legacy document formats expose only \emph{flat}, absolutely positioned elements, so agents must recompute coordinates and routinely break layouts; and design has no unique ground truth, so diff-against-reference metrics penalize valid-but-different outputs. We present \textbf{ACE}, an agentic canvas editor over a \emph{hierarchical scene-graph} with a presentation-specialized action space (98 tools), paired with \textbf{CARE}, a content-aware router that feeds the agent only the relevant slice of each deck (avg.\ $\sim$89\% input-token reduction), and a \emph{self-correction} loop driven by a \emph{ground-truth-free} instruction-following (IF) judge whose natural-language critique is fed back as the next-turn instruction. With a fixed backbone, a scene-graph editor in a \emph{single turn} already matches a same-backbone \emph{agentic} HTML pipeline that iterates internally; adding self-correction lifts ACE significantly above it on instruction following (IF 4.23 vs.\ 3.81 on the full 94-task benchmark, paired $p{=}.010$, replicated by an out-of-loop judge) at 1.75$\times$ the speed and $\sim$44\% lower cost. VQ means are statistically indistinguishable, but 26 blind raters prefer ACE overall (58.7\% decisive win-rate) and prefer the self-corrected output 81\% of the time; the ranking is invariant across three judge families, and out-of-loop judges retain two-thirds of the self-correction gain, bounding circularity. 66\% of cases halt after one pass, and a strict-peak rollback removes every observed regression. 

---
# Are Android GUI Agents Robust Against Runtime Anomalies? AnTrap: Evaluating Agents in Dynamic Adversarial Environments 

**Authors**: Guo Gan, Yilun Zhao, Cong Chen, Jinbiao Wei, Tingyu Song, Zheyuan Yang, Lin Fu, Hong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.24099)  

**Abstract**: GUI agents often encounter dynamic anomalies when deployed on Android devices, from unexpected pop-ups to action misuse, yet existing benchmarks lack systematic evaluation of agent robustness against runtime anomalies. We introduce AnTrap, a comprehensive benchmark that injects dynamic perturbations into agent execution trajectories. We propose a taxonomy organizing real-world anomalies into four layers (State, Thinking, Action and Round) with ten fine-grained subcategories, and develop a construction pipeline that preserves task solvability while introducing realistic adversarial conditions. Evaluating 16 leading GUI models, we reveal universal vulnerability to dynamic anomalies, with even the strongest models suffering significant performance degradation. Furthermore, we conduct GRPO training in both original and adversarial environments to validate our benchmark, separating environment-learnable anomalies from reasoning-bottlenecked ones. Our findings show that while single-step traps at state and action layers are largely addressable through adversarial reinforcement learning, deep contextual traps, like state deadlock, expose intrinsic limitations that cannot be resolved by training in environments with traps alone. 

---
# EMRB: A Multi-Level Benchmark for Evaluating LLM Reasoning over Raw Electromagnetic Signals 

**Authors**: Mingxu Zhang, Ying Sun, Yuhan Li, Yang Ji, Dazhong Shen, Ke Zhang, Shan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.24086)  

**Abstract**: Large language models (LLMs) are increasingly used as code agents for scientific and engineering analysis, but their ability to analyze raw physical-layer measurements remains untested. We introduce \textbf{EMRB} (\textbf{E}lectro\textbf{m}agnetic \textbf{R}easoning \textbf{B}enchmark), which evaluates whether LLMs can analyze raw I/Q data by writing and running code. EMRB contains 200 problems across five difficulty levels and 27 question types, from signal detection to OFDM design, generated from 11 signal types with verified ground truth. Unlike benchmarks built on preprocessed features or structured tables, EMRB provides only the raw capture; the quantities each question refers to must first be discovered through code. We evaluate 14 LLMs spanning proprietary, open-weight, and reasoning-oriented families. Scores range from 24.1\% to 78.9\%, with the mean dropping from 84.9\% on basic measurement to 21.2\% on system design. We also propose \textbf{ReconPilot}, a structured method that separates signal reconnaissance, targeted analysis, and self-verification. Across three backbones, ReconPilot raises the overall score by 3.8 to 17.6 points and improves 13 of 15 backbone-level combinations tested. All data and code are publicly released in \href{this https URL}{\textcolor{blue}{our GitHub repository}}. 

---
# AgentWorld: Personality-Aware Reliability Evaluation for Agentic Information Retrieval 

**Authors**: Gunja Agarwal, Arup Kumar Das, Arun Menon, Jitesh Chandra Mishra, Vignesh Divakaran  

**Link**: [PDF](https://arxiv.org/pdf/2608.24076)  

**Abstract**: Evaluation of agentic information retrieval remains limited to scripted interactions with uniform users, missing both natural personality diversity and adversarial brittleness. We present AgentWorld, a simulation framework combining (i)Big Five (OCEAN) personality-driven user populations with stateful tool-use environments; (ii)the pass$^k$ consistency metric with structured fault classification, partial-credit scoring, and dual-control handoff verification; (iii)score-thresholded training-data export in six fine-tuning formats; and (iv)an adversarial Risk Analyser that snapshots required-intermediate-state spines, branches Monte-Carlo rollouts under four task-aware perturbation types, and quantifies risk via $\Delta P / \Delta T$ scoring, Dempster--Shafer evidence fusion, and Shapley attack-category attribution. Three experiments demonstrate the framework: a conversational analytics agent across 10 OCEAN personas (240 evaluator judgments); a customer-support agent across 5 tasks $\times$ 4 persona variants; and adversarial stress-testing of 5 tasks revealing pre-existing trajectory brittleness ($V_{\min}=0.375$ without perturbation) and tool/infrastructure-layer attack dominance (Shapley: 46% system, 38% action). Personality variation surfaces failure modes uniform testing cannot expose---cross-domain leakage, contextual drift, a 0.27-point quality gap, and 50% vs. 100% pass-rate across personas on the same task---while the Risk Analyser quantifies trajectory-level brittleness that pass$^k$ alone cannot measure. 

---
# Compression Trinity: Exploring Sparsity, Quantization, and Low-Rank Approximations for LLM Compression 

**Authors**: Mohammad Mozaffari  

**Link**: [PDF](https://arxiv.org/pdf/2608.24070)  

**Abstract**: Prohibitive computational and environmental costs impede the scalable deployment of Large Language Models (LLMs). Traditional compression techniques (sparsity, quantization, low-rank approximations) are typically applied in isolation, and each hits an accuracy-efficiency wall. This thesis proposes the "Compression Trinity," a unified framework that applies the three pillars jointly: sparsity to reduce computation, quantization to minimize memory bandwidth, and low-rank approximations to recover accuracy. To accelerate pretraining, we apply the Trinity to the optimizer and model architecture. MKOR approximates curvature via block-diagonal sparsity and low-rank inversion, maintaining numerical stability for quantized states; it reduces curvature update complexity from $O(d^3)$ to $O(d^2)$ and accelerates convergence by up to 1.85x over KFAC. SLoPe accelerates training by up to 1.25x via a double-pruned backward pass for N:M sparsity, using low-rank "lazy" adapters in the final 1% of training to recover accuracy. For post-training compression, OPTIMA stabilizes static masks in a zero-training regime by formulating weight reconstruction as globally optimal column-wise quadratic programs, improving zero-shot accuracy by up to 3.97%. Given a fine-tuning budget, PATCH breaks the ceiling of static masks by learning a dynamic hybrid sparsity ratio between 0% and 50%, yielding up to 1.38x speedups. Finally, SLiM realizes the full Compression Trinity in one shot, using mathematically derived low-rank adapters to recover information lost to quantization and sparsity, improving accuracy by up to 5.66% over state-of-the-art methods and outperforming uncompressed dense models at equal parameter budgets by 0.6%. Together, these results show that jointly applying the Compression Trinity is essential for efficient, scalable, high-performance LLMs. 

---
# Poisoning Agentic Alpha: Adversarial Vulnerabilities Across Roles and Architectures in Multi-Agent Trading Systems 

**Authors**: CheolWon Na, Hao Ni, Lukasz Szpruch, Zhangyang Wang, Dhagash Mehta, Saurabh Nagrecha, Alejandro Lopez-Lira, Chanyeol Choi, Yongjae Lee, Jee-Hyong Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.24069)  

**Abstract**: LLM-based multi-agent trading systems, in which specialized agents collaborate through structured communication to produce trading decisions, are moving rapidly from research prototypes to live deployments that control real assets. The same inter-agent communication that makes them effective also exposes them: a corrupted signal can propagate to the final decision and translate into realized financial loss. Unlike prior attacks that presume privileged access to system internals, we restrict the adversary to what is practically reachable---the source data and prompts agents consume---yielding a low-barrier, and thus democratized threat model instantiated as role-specific adversaries.
We present the first systematic empirical study in the financial domain to characterize how an adversarial signal enters a multi-agent trading system and how far it survives toward the decision. Along the role axis, we decompose a widely-used trading pipeline into four functional roles---Analyst, Researcher, Trader, and Risk Manager---and pair each with an attack matched to its interface. Along the structural axis, we evaluate four communication topologies under data- and agent-level attacks, using the Adversarial Signal Preservation Score (APS) as a post-hoc lens on why some designs are more robust than others. We conduct experiments across five assets, two backbones, and two target directions. A central finding is that no architecture is inherently robust. These findings provide insights for the future design of safer and more robust agentic trading systems. 

---
# Algorithmic Impact Reveals the Hidden Social Choice Structure of Alignment 

**Authors**: Zachary Wojtowicz, Michelle Si, Finale Doshi-Velez, Ariel Procaccia  

**Link**: [PDF](https://arxiv.org/pdf/2608.24046)  

**Abstract**: When an AI algorithm makes decisions that affect more than one person, aligning it becomes a problem of social choice: how should people's divergent preferences about system behavior be reconciled and aggregated into a single coherent model? The standard approach to aligning frontier AI models$\unicode{x2013}$reinforcement learning from human feedback$\unicode{x2013}$largely sidesteps this question and has poor social choice guarantees. However, it remains unclear what alternative should replace it. We show that, by focusing directly on an algorithm's welfare consequences, the alignment problem can be reformulated as linear optimization over a convex impact space, which makes it amenable to the standard toolkit of welfare economics and mechanism design. This reformulation clarifies how alignment protocols translate into welfare consequences and, conversely, how a social planner's desired constraints on welfare consequences can be translated back into alignment protocols. We apply this transformation to show that voting-by-issues and random-dictatorship mechanisms are strategyproof and unanimous. Demonstrating the reverse direction, we also apply the impact representation to derive a family of alignment protocols that maximize utilitarian social welfare subject to various social desiderata, such as bounds on individual or group harm. We illustrate the welfare implications of these alignment protocols empirically using real human preferences over kidney allocation, charitable food distribution, LLM responses, and trolley problems. 

---
# Relative Time Intervals Representation for Word-level Timestamping with Masked Training 

**Authors**: Quanwei Tang, Zhiyu Tang, Xu Li, Dong Zhang, Shoushan, Guodong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.24041)  

**Abstract**: Although Speech Large Language Models (SpeechLLMs) excel at speech understanding and generation, their capacity for fine-grained, temporally aligned outputs remains underexplored. Our work addresses this gap by enabling SpeechLLMs to jointly model speech content and temporal structure, effectively transforming them from ``content understanding machines" into ``temporal-aware content understanding machines". Specifically, we replace traditional absolute timestamps with relative timestamps, achieving a more compact vocabulary and stronger generalization capabilities. To efficiently infuse timestamp prediction ability into pre-trained large language models, we introduce a hybrid fine-tuning strategy: full-parameter fine-tuning of the timestamp-augmented embedding layer and language model head, combined with LoRA fine-tuning of the decoder layers. Moreover, we design a masked timestamp training objective, preventing the model from over-relying on ground-truth timestamps, and thereby enhancing robustness against noisy real-world annotations. Extensive experiments demonstrate that our approach achieves significant improvements in timestamp prediction accuracy while maintaining strong speech transcription performance. 

---
# Beyond Confidence: Test-Time Scaling for Multi-Turn Search Agents via Retrieval Grounding 

**Authors**: Hyunho Kook, Junhyuk So, Tianyu Fu, Haizhong Zheng, Beidi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.24024)  

**Abstract**: Confidence-based voting aggregates parallel LLM rollouts by weighting each with internal signals such as token log probabilities, and has been actively studied for single-turn reasoning. However, modern LLMs increasingly act as multi-turn search agents that retrieve and condition on external documents. In this paper, we show that confidence-based voting transfers poorly to this multi-turn setting, and identify the underlying failure reason as copy inflation: when retrieved documents are appended to an agent's context, tokens copied from those documents receive systematically inflated log probabilities. This flattens confidence scores within each question and weakens the resulting weighted vote. To address this issue, we propose Retrieval-Grounded Voting (RGV), which scores each rollout by the lexical overlap between its final answer and the documents it retrieved. By computing the signal outside the contaminated context, RGV sidesteps both token log probabilities and additional LLM calls. Across four search-agent benchmarks and five LLMs, RGV consistently outperforms confidence-based voting, with gains of up to +5.4% accuracy and +35% on minority-correct questions, where the correct answer appears in only 1-2 of 8 rollouts. 

---
# Reflection with Action-Induced Visual Differences for Desktop GUI Agents 

**Authors**: Yijie Ma, Chaoyue Niu, Fan Wu, Guihai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.24015)  

**Abstract**: The Planner-Operator-Reflector (POR) framework is widely used in GUI agents to maintain objective alignment in complex tasks through modular collaboration. However, desktop GUIs introduce a key challenge: large, dense interfaces often exhibit subtle or scattered state changes, placing most of the burden on the reflector, which must compare pre- and post-action screens, while the planner and operator reason over a single state. Existing reflectors collapse change detection and outcome verification into one step, leaving evidence implicit and yielding weakly grounded decisions. To address this limitation, we propose Evidence-First Reflection (EFR), a two-stage reflector that explicitly decouples action-induced visual differences extraction from outcome verification. EFR identifies the action location and candidate changed regions with Set-of-Marks annotations, describes and filters action-relevant changes, and makes the final judgment from the cleaned evidence. This evidence-reasoning decoupled design makes reflection better grounded in screen transitions, while reducing both visual search complexity and reasoning burden. Experiments on OSWorld-Verified and WindowsAgentArena demonstrate that EFR improves reflector accuracy by 7.11%, yielding average end-to-end task success gains of 5.94% and 4.95% on the two benchmarks, respectively. 

---
# Incorporating Cognitive Load and Knowledge Transfer for Multi-Domain Knowledge Tracing 

**Authors**: Haotian Zhang, Shucun Wang, Jinze Wu, Liang Ding, Shuochen Liu, Zhenya Huang, Jing Sha, Shijin Wang, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.24005)  

**Abstract**: Knowledge Tracing (KT) aims to assess students' dynamic knowledge states from their learning histories. While most existing KT methods focus on single-domain learning with notable success, real-world learning scenarios often involve multiple domains simultaneously, introducing two critical factors: 1) Cognitive load, arising from managing learning across domains in both temporal and knowledge dimensions. 2) Knowledge transfer, where knowledge states in one domain influence related states both within and across domains. In this paper, we focus on exploring these factors to improve students' knowledge state assessment in multi-domain learning scenarios and propose a novel method incorporating cognitive Load and knowledge Transfer for Multi-domain Knowledge Tracing (LT-MKT). Specifically, to bridge isolated domains, LT-MKT first integrates textual information from questions and their associated concepts to construct a Multi-domain Hierarchical Graph, leveraging the advanced representational capabilities of large language models (LLMs). Then, cross-domain features in both the temporal and knowledge dimensions are explicitly modeled to capture the effects of cognitive load. Additionally, a knowledge transfer module is designed to model the propagation of knowledge states within and across domains. By jointly modeling these factors, LT-MKT enables more accurate prediction of students' future performance. Finally, extensive experiments on real-world datasets demonstrate that our method achieves state-of-the-art performance. 

---
# Diverse by Reasoning: Harnessing the Wisdom of LLM Crowds for Future Prediction 

**Authors**: Nirupam Chetlapalli, Yiming Liao, Min-Chun Chen, Keke Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.24001)  

**Abstract**: Large language models (LLMs) are increasingly used for future prediction, motivating the use of multiple models as a wisdom-of-the-crowd mechanism. However, simply increasing crowd size does not guarantee effective diversity, as different LLMs may exhibit redundant behaviors. We propose a behavior-aware framework for constructing diverse LLM crowds. The framework characterizes models using their reasoning traces on independent development tasks, clusters models by behavioral similarity, and selects representatives for collective prediction. We evaluate 25 LLMs using seven development benchmarks for behavioral diversity modeling and two future-prediction benchmarks for evaluating diverse crowds' performance. Our results show that crowd composition can matter more than crowd size: a three-model medoid crowd based on K-means++ behavioral clustering outperforms conventional voting over all 25 models on both prediction benchmarks, while reducing model calls by 88% and inference cost by approximately 80%. The results further suggest that representative behavioral diversity, rather than simply maximizing diversity, is important for constructing effective LLM crowds 

---
# Memory Is Not Always Needed: Characterizing Conditional Memory in Scientific Reasoning 

**Authors**: Zhen Bi, Xueshu Chen, Yan Wang, Zhizhi Peng, Haosen Hong, Zhen Wang, Zhixuan Chu, Bingyu Zhu, Jungang Lou  

**Link**: [PDF](https://arxiv.org/pdf/2608.23982)  

**Abstract**: Scientific reasoning requires language models to retrieve specialized knowledge and incorporate it reliably into multi-step computation. Conditional memory provides an explicit lookup pathway that complements dense neural representations, but its usefulness is inherently input- and computation-dependent: retrieved information may repair missing scientific associations, yet it may also introduce distracting shortcuts or interfere with reasoning that the base model can already perform correctly. In this work, we systematically investigate when, where, and to what extent conditional memory should participate in scientific reasoning. We characterize the scientific knowledge boundary and controlled interventions on memory-enabled knowledge-circuit nodes. Based on these analyses, we propose a Knowledge Boundary-Aware Router that uses task-specific input proxies available before generation to determine whether memory is activated, which layer-stage nodes receive memory signals, and how strongly these signals contribute. Experiments on biological and chemical reasoning benchmarks, covering two backbone families and six task types, show that memory effects vary substantially across inputs, tasks, and injection locations. Compared with static and activation-rate-matched random routing, our approach more consistently preserves beneficial memory contributions while suppressing memory-induced regressions, establishing selective memory allocation as an important principle for reliable scientific reasoning. 

---
# Rules Before Oracles: Auditable, User-Configurable Argument Selection for Deliberative Polling 

**Authors**: Muntaser Syed, Markus Zanker, Marius Silaghi  

**Link**: [PDF](https://arxiv.org/pdf/2608.23979)  

**Abstract**: In a deliberative poll, once submissions outnumber what anyone will read, some mechanism chooses which arguments each voter sees, acquiring much of the decision; practice delegates it to opaque learned rankers, so a voter cannot recompute or contest the exposure that shaped their vote. We ask whether it can be a published rule over publicly recomputable evidence with parameters held by the voter, treating legibility as an admissibility condition on usable mechanisms, not an objective traded against accuracy. We formalise a poll over bipolar justification sets, judging a slate by reason coverage, the order it arrives in, and captured endorsement mass; we give seven checkable criteria for a civic recommender and a rule meeting them: a one-hop reversed endorsement flow parameterised by a relation-weight function. An agentic simulator records every slate at every vote, over about 17,000 seed-paired runs. Served slates fall 0.035 short of a label-reading ceiling upper-bounding every selection procedure, opaque ones included: any unconstrained ranker's advantage is bounded and small. On coverage alone, with non-degenerate authoring, the rule is indistinguishable from a random slate, a null due to an order-blind, charity-blind instrument; on the other two it leads at every prefix by a margin widening with adversarial pressure and dominates on mass by a factor of 3.3. Once a realistic fraction of submissions carries no reasons, the coverage margin returns and grows. Label-homogeneous flooding collapses completeness from 0.81 to 0.34 under a flat weight policy, only to 0.44 under author-count normalisation, making the weight function a security control worth 10% of completeness. The choice between ranking arms is a position on a coverage-versus-mass frontier, not a fact, the kind of choice only a legible rule can hand to the person it affects. It maps onto an open-source peer-to-peer platform. 

---
# When Seeing Is Not Enough: Benchmarking Interactive Visual Grounding in LVLMs 

**Authors**: Zhengxiang Wang, Owen Rambow  

**Link**: [PDF](https://arxiv.org/pdf/2608.23978)  

**Abstract**: Visual grounding is typically evaluated as a one-shot mapping from an informative referring expression to a visual target. This formulation misses a central property of real-world reference: target information is often incomplete, ambiguous, and established through interaction. We introduce a controlled evaluation framework for interactive visual grounding in large vision-language models (LVLMs), varying how much target information is provided upfront and how much must be acquired through dialogue. Across four human-grounded visual contexts and four interaction protocols, current LVLMs perform significantly below task-level human baselines. Interaction can help when follow-up questions refine or repair an initial target description. Performance is lowest when no initial description is provided and target information must be acquired through questions, indicating that proactive question-driven grounding remains difficult. LVLMs are also poorly calibrated, often reporting confidence that exceeds their empirical accuracy. Follow-up studies confirm these patterns across varied description sources (human versus AI), reasoning efforts, repeated interactions, description providers, and visual contexts. Overall, interactive visual grounding remains an important challenge, requiring visual matching, information seeking and synthesis. 

---
# Giraffe: A Mapping Architecture from Hidden Text Representations to Visual Embeddings for Efficient Graphic Design 

**Authors**: Nejla Ghaboosi  

**Link**: [PDF](https://arxiv.org/pdf/2608.23970)  

**Abstract**: Multimodal large language models (MLLMs) have made significant progress in understanding and interpreting mul- timedia content. However, their ability to generate me- dia remains limited. Recent approaches have attempted to bridge this gap by translating the hidden representations of token sequences into the embedding space of visual models or directly into raw image data. However, these methods often represent each image using multiple specialised to- kens which significantly increases the input length. This be- comes a major limitation for tasks such as graphic design generation where the output typically involves a seamless blend of thousands of tokens across text, multiple images, and layout information. To address this challenge, a novel architecture is proposed that maps hidden token represen- tations to the embedding space of visual models, such as CLIP ViT-L/14, using a single [IMG] token per image. The architecture employs two shallow MLP blocks, each with a separate compression module followed by a shared expan- sion module, trained with six distinct loss functions. One block aids the other during training and is omitted during inference, resulting in a lightweight solution. Strong perfor- mance is demonstrated in both image-to-design and text-to- design generation tasks. 

---
# More GPUs or a Smaller Cache? Tensor Parallelism versus KV Compression for Memory-Bound LLM Serving 

**Authors**: Srikanta Datta Tumkur, Mehar Simhadri, Anshu Bansal, Jay Iyer, Sai Pavan Kumar, Sai Kapil Kumar, Ramesh Nampelly, Raj Dandekar  

**Link**: [PDF](https://arxiv.org/pdf/2608.23962)  

**Abstract**: When an LLM serving deployment runs out of KVcache room, there are two well-established ways out. Tensor parallelism shards the weights and the KV cache across two, four, or eight devices, buying memory headroom at the price of an all-reduce on every layer and a hardware bill that grows with the device count. The algorithms community shrinks the cache in place, with KV quantisation and eviction keeping a single GPU and spending a little quality instead. Compression papers report memory ratios, parallel-scaling papers report throughput curves, and almost nobody puts the two on the same cost axis. We place tensor-parallel configurations (degree 1 to 8) and KV-compressed configurations (16/8/4-bit, keep-ratios down to 0.25) on one costnormalised axis, cost per million tokens against latency, using a profiled simulator calibrated on A100, A40, and H100 hardware, and we go looking for the cost-equivalence crossover. We do not find one. Across two models (Llama-2 at 7B and 70B), three GPU types, and every level of memory relief we could construct, compression is cheaper by 1.20x to 2.00x. A 7B model on an 80 GB device cannot exhaust its KV budget within its own context window, and the boundary that decides between the strategies is model size relative to device memory, at roughly 36B parameters for an 80 GB card. Below that wall, compression dominates and extra GPUs are largely wasted spend; above it, tensor parallelism stops being a choice and becomes an entry ticket: Llama-2-70B is infeasible on one A100 at any KV setting, because the binding resource is weights, which KV compression does not touch. Tensor parallelism is the only lever that improves latency (compression makes per-token latency worse, by 8 to 93%, through batching contention), while compression is the only lever that multiplies capacity per dollar (16.5x, against 1.21x for an eightfold spend on GPUs). 

---
# Recursive Agentic Reasoning 

**Authors**: Shengxin Zhang, Xiaomin Wu, Xiyang Wu, Jing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2608.23956)  

**Abstract**: Test-time reasoning methods such as iterative refinement, decomposition, and repeated sampling are often evaluated in isolation, making their gains difficult to compare across models, benchmarks, and evaluation pipelines. We introduce a unified view of these methods as recursion operators over an agent's reasoning trace: GROW, which deepens a single reasoning path; PRUNE, which decomposes and recomposes the problem; and BRANCH, which samples alternative reasoning paths and selects among them. We evaluate all three operators against a single-pass chain-of-thought baseline under a shared harness with identical prompts, token budgets, and grading code. Across five benchmarks and three frontier models, comprising 14 model-benchmark settings, 49,327 graded items, and 151,876 model calls, BRANCH improves accuracy in all 14 settings by an average of 5.98 percentage points and is the best-performing operator in 12. In contrast, GROW yields a mean gain of 2.18 points and degrades performance in two settings, while PRUNE improves accuracy by 0.94 points on average. Analysis shows that BRANCH's advantage arises not only from exploring multiple reasoning paths, but also from recovering from truncation: its gains strongly correlate with the baseline rate of empty, budget-exhausted outputs (r = 0.72). These results weaken the hypothesis that different problems require routing among test-time reasoning operators; at this level of abstraction, repeated branching is consistently dominant. Finally, we show that unpaired evaluation and treating scoring-pipeline failures as model errors can materially change, and even reverse, comparative conclusions, motivating paired scoring as a standard protocol for test-time-compute evaluation. 

---
# More Rejective, Not More Discriminative: The Unit of Verification in Pre-Execution LLM Oversight 

**Authors**: Yuchen Han, Cheng Yan, Wuyang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.23941)  

**Abstract**: Pre-execution oversight is core to trusted monitoring in AI control: a fallible LLM monitor vets planned actions before irreversible execution. Over-blocking forfeits usefulness and pressures deployers to disable it. Every protocol must fix a unit of verification: how many actions one call reviews. Existing designs take the unit as given; its effect on fallible monitors is unmeasured. Natural traces cannot isolate it: review length co-varies with error type and position. Catch alone misleads: rejecting everything catches everything. Measuring this needs boundary variation alone and a matched clean control. We introduce the twin-prefix framework, which supplies both. Each gold plan yields a prefix with one injected, environment-accepted error and a clean twin differing in one write. Judging each pair at five nested lengths ties verdict changes to the unit alone. Discrimination is scored by pre-registered informedness, catch minus false rejection. Longer review raises catch; false rejection climbs in lockstep. Informedness peaks at one or two actions for all six judges in both domains: longer windows make zero-shot monitors more rejective, not more discriminative. Replaying withheld observations traces the failure largely to observation deprivation. Safety cases should state the unit and co-report the clean series. Our framework is the first controlled, pre-registered instrument for this choice and never reads catch alone. Our calibrated short unit recovers up to 0.95 informedness over eight-action review, and no tested label-blind policy consistently beats it. 

---
# Evolutionary Recurrent Decision Model in Developing Adaptive and Maladaptive Behaviors 

**Authors**: Andrew Hu  

**Link**: [PDF](https://arxiv.org/pdf/2608.23932)  

**Abstract**: This study introduces the evolutionarily recurrent decision model (ERDM), a computational reinforcement learning framework designed to examine how evolutionary mismatch, bounded rationality, and satisficing contribute to adaptive and maladaptive behavior. ERDM simulates agents across evolutionary recurrent environments, including threat, prey/goal-pursuits, and alliances. Agents learn through competing rewards abstracted from survival metrics. A validity study under varying adverse childhood experiences demonstrates that distinct adaptive and maladaptive strategies, such as learned helplessness, avoidance, healthy relationships, and aggression, emerge naturally without being hardwired. These results align with empirical literature, showcasing ecological validity. The results suggest that many psychopathology-relevant aspects may be interpreted as bounded cognitive systems operating under modern-ancestral environmental mismatch, positioning ERDM as a key computational cognitive tool that can be extended to other studies. 

---
# Data Mixing as Mixture Experiment: Response Surface Methodology and Optimal Design for Large Language Model Pretraining 

**Authors**: Yicheng Mao, Hongru Du  

**Link**: [PDF](https://arxiv.org/pdf/2608.23922)  

**Abstract**: Data mixing is a central design problem in large language model pretraining: given a fixed token budget, practitioners must decide how much data to allocate to each domain. Recent proxy-based methods address this problem by training small models on candidate mixtures, fitting a response model, and using the response to select mixtures for larger-scale training. We show that this workflow has the structure of a classical mixture experiment. Under this view, data domains are mixture components, token shares are component proportions, proxy-training runs are experimental design points, and validation loss defines a response surface over the probability simplex. We develop this formulation using sparse second-order Scheffé response-surface models and construct model-robust $\mathcal{I}$-optimal designs for proxy data-mixing experiments. Using RegMix as an empirical case study, we demonstrate how the framework can both interpret observed mixture responses and design more efficient proxy experiments. The Scheffé analysis shows that domain value is strongly relational: several domains that are weak under additive effects become favourable through pairwise interactions, especially through combinations with web-derived text. The sparse Scheffé model preserves mixture rankings across model scales and remains competitive with a flexible machine-learning predictor while providing an explicit decomposition of additive and interaction effects. In a simulation study calibrated to observed proxy-training responses, model-robust $\mathcal{I}$-optimal designs recover the relevant mixture ordering after removing about 25\% of the original proxy runs. These results suggest that LLM data mixing should be treated not only as a prediction problem, but also as an experimental-design problem in which the proxy mixtures themselves can be chosen to improve statistical efficiency. 

---
# MARS: Multi-Specialist LLM Relay System for Competitive Programming 

**Authors**: Andrei Mikhailov, Mikhail Burtsev, Alsu Sagirova  

**Link**: [PDF](https://arxiv.org/pdf/2608.23918)  

**Abstract**: Large Language Models excel at code generation, yet competitive programming exposes a persistent failure mode: existing multi-agent pipelines distribute work over generic planner, coder, and debugger roles and delegate the choice of algorithmic technique to the backbone alone. We present MARS (Multi-Agent Relay of Specialized LLMs), a prompt-only framework in which each agent is a topic specialist---dynamic programming, graphs, strings, geometry, and so on---grounded by retrieval-augmented generation over an algorithm-theory corpus. Given a problem, retrieval selects a small team of relevant specialists; a starter writes an initial C++17 solution, and each subsequent turn runs the candidate against public examples in a sandbox, lets the active specialist keep, repair, or hand off the draft, and forwards a structured packet to the next specialist. A single infrastructure-fixer pass normalizes boilerplate at the end. On the CodeContests test split with Gemma 4, MARS reaches $0.624 \pm 0.006$ pass rate at $2.3$ recorded pipeline stages per task ($+14.4$ percentage points over direct prompting), closing most of the gap to CodeSIM ($0.731$) at $3.3{\times}$ lower wall-clock cost and substantially smaller variance in per-task token spend. The source code is available on GitHub: this https URL. 

---
# PROOF-Gen: From Optimized Data to Better Distillation 

**Authors**: Anh Ta, Junjie Zhu, Shahin Shayandeh  

**Link**: [PDF](https://arxiv.org/pdf/2608.23911)  

**Abstract**: Supervised fine-tuning on teacher-generated trajectories is the standard first stage for distilling tool-calling capabilities into deployable models. Post-training pipelines that drive shipped tool-calling agents re-run this stage on a daily or weekly cadence, paying the frontier-teacher cost each cycle, yet the mechanism is generate-and-filter (keep the teacher's passing trajectories, discard the rest) and each cycle leaves behind the same hard scenarios because failures supply no signal. On {\tau}2-bench, 57% of teacher trials fail, two-thirds of them near-misses (most tool calls correct, undone by one decisive error).
We introduce PROOF-Gen (Per-scenario Reflective Optimization to Overcome FailedGeneration), which recovers golden trajectories from these failures via per-scenario prompt optimization. For each failed task, a reflector analyzes the execution trace and evaluation feedback, then writes corrective guidance that steers the teacher to a passing trajectory. The guidance is stripped before training, so the student learns from clean demonstrations with no task-specific scaffold.
On {\tau}2-bench, per-scenario optimization recovers 93% of failed scenarios. Fine-tuned on the combined data, Qwen3-4B-Instruct-2507 improves from Pass^1=0.132 to 0.529 and Gemma 4 E4B-it gains +7.2pp on BFCL v4 multi-turn. In a deployed pipeline, the method lifts trajectory quality by +6.3pp goal completion and transfers to a deployed on-device model (+1.5pp goal completion; +1.7 to +5.0pp across response-quality metrics), with positive transfer in every locale (non-English average +1.48pp). 

---
# Retrieval-augmented generation vs. deterministic tax computation in multi-agent financial advisory: A 2x2 factorial experiment 

**Authors**: Aryan Brar, Justin Du, Avery Lor, Kylie Seto, Eric Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2608.23908)  

**Abstract**: Tax-loss harvesting demonstrates consistent benefits to long-term portfolio growth; yet implementing it efficiently often involves complex considerations that are specific to the holdings within that portfolio and the individual who owns it. We introduce a custom capital gains calculation engine and a RAG-retrieved vector store of market advisory reports to provide context for a multi-agent trade recommendation system. We investigate the effects of each context provider on the quality of recommendations, measured by relative capital gains incurred during portfolio liquidation. A 2x2 repeated-measures ANOVA revealed a significant main effect of the tax optimization engine ($F(1,29) = 9.17$, $p = .005$, $\eta^2_p = .240$): enabling the engine reduced tax savings by approximately 55 percentage points relative to the no-engine conditions. The RAG main effect was not significant ($p = .841$), nor was the interaction ($p = .553$). The RAG-only condition achieved the highest descriptive mean tax savings (47.7%), and the baseline condition performed second-best (30.6%), suggesting that the pre-trained language model's internalized financial knowledge may be sufficient for competent tax-loss harvesting recommendations without explicit tooling. These results indicate that augmenting LLM agents with domain-specific computation engines does not guarantee improved performance and may introduce conflicting optimization signals. 

---
# Quantifying System-Level Harms from AI Adoption in Complex Sociotechnical Systems 

**Authors**: Paul Vautravers, Oliver Chalkley, Gabriel Downer, Kate S, Damian Ruck  

**Link**: [PDF](https://arxiv.org/pdf/2608.23906)  

**Abstract**: Artificial Intelligence (AI) is increasingly integrated into complex sociotechnical systems, including Critical National Infrastructure (CNI), where harms emerge from interactions between technical, human, and organisational elements. Yet current AI evaluation remains model-centric, offering little insight into how observed behaviours might translate into system-level risk. We propose a framework that links structured hazard analysis, component-level testing, and probabilistic system modelling to bridge this gap. By providing a traceable pathway from model behaviour to system-level outcomes, the framework enables practitioners to answer the "so what?" of AI failures, quantify their systemic impact, and move toward evidence-based and anticipatory governance of AI in complex systems. Applied to the UK's Real Time Gross Settlement (RTGS) system as an illustrative worked example, we derive AI-driven loss scenarios using Systems Theoretic Process Analysis (STPA) and examine adversarial manipulation of LLM-based trading as one such loss scenario. Component-level experiments show that simple adversarial inputs induce measurable behavioural shifts where AI recommendations are followed. Under the component-to-system mapping used here for a financial contagion model, these shifts alter system resilience, increasing bank failures and lowering the threshold at which shocks lead to cascading disruption, particularly under widespread or monopolistic AI adoption. 

---
# BenchBench-Protocol: Evaluating Real-World Wet-Lab Protocol Reasoning and Modification 

**Authors**: Aditya Sivakumar, Ashu Singhal, Nicholas Larus-Stone, Nithin Parsan  

**Link**: [PDF](https://arxiv.org/pdf/2608.23898)  

**Abstract**: We introduce BenchBench-Protocol, a benchmark for large language models of 149 protocol-modification tasks recovered from modifications that scientists made to published protocols during real experimental work. Adapting a published protocol to a new experiment is a routine task for a wet-lab scientist, and a correct modification requires accounting for prior choices and downstream steps. Recent life-science benchmarks have moved toward open-ended, rubric-graded tasks, but tasks are typically elicited from experts rather than reconstructed from real-world modifications. BenchBench-Protocol tasks are derived from differences between a published protocol and a version a scientist modified, which provides the basis for the query and the weighted rubric elements for a correct response. The benchmark draws from 96 source protocols across nine domains of wet-lab biology and only includes tasks rated highly after review by domain experts. We evaluate nine closed and open models; Claude Opus 5 scores highest at 59.2% normalized rubric score, with other models between 34.1% and 47.1%, and the benchmark remains unsaturated when taking the best of ten attempts. As models are increasingly helpful in life-sciences research, evaluating them on routine wet-lab tasks becomes correspondingly important. We present BenchBench-Protocol as both a grounded assessment of wet-lab reasoning and evidence for the utility of real-world experiments to construct benchmark tasks. 

---
# Provenance Guided Incremental Learning Under Evolving Concept Definitions 

**Authors**: Ismail Lamaakal  

**Link**: [PDF](https://arxiv.org/pdf/2608.23893)  

**Abstract**: Learning systems deployed over long periods must adapt not only to statistical changes in incoming data, but also to revisions of the definitions that generate their prediction targets. Conventional concept-drift methods typically infer such changes from observations or prediction errors, even when the underlying policy, rule, or query has been explicitly modified. This paper studies rule-induced concept shift, where the target-defining concept is revised directly, causing previously stored instances to acquire different semantic labels without requiring any change in their observed data. We introduce a provenance-guided incremental learning framework that compiles consecutive concept definitions into a structured rule delta, traces the changed components through historical provenance, certifies records whose previous labels remain valid, and restricts reevaluation to a localized candidate region. Executable revisions are relabeled automatically, ambiguous cases are handled through selective supervision, and the resulting changes are used for incremental predictor repair. A versioned concept memory further supports recurring definitions. We also introduce RuleShift-Bench, spanning financial, demographic, cybersecurity, and graph-structured data with threshold, predicate, logical, relational, recurring, and mixed concept revisions. Across the benchmark, provenance-guided repair attains 92.3% accuracy and 90.2% Macro-F1 while reprocessing 14.7% of the historical collection and retaining 94.6% of affected records. Its average update latency is 179s compared with 993s for complete relabeling and retraining. The results demonstrate that an explicit concept revision can be exploited as a data-maintenance signal, allowing learning systems to update the supervision and predictive state that depend on the change while preserving knowledge that remains valid. 

---
# AI Finds A Way 

**Authors**: Aaron Dharna, Cong Lu, Ryan Sullivan, Joel Lehman, Victoria Krakovna, Jeff Clune  

**Link**: [PDF](https://arxiv.org/pdf/2608.23875)  

**Abstract**: Artificial Intelligence (AI) algorithms frequently learn creative and unexpected solutions, surprising even expert researchers who develop and study them. They often astonish practitioners by discovering unanticipated behavior, exploiting loopholes in reward signals, or spontaneously uncovering previously unknown scientific phenomena. However, accounts of such unconventional behavior across machine learning are seldom formally documented. This work presents 26 curated firsthand anecdotes from various machine learning subfields representing the work of over 100 researchers. These anecdotes showcase the capability of modern AI systems to circumvent human-imposed design limitations and discover unexpected solutions to the tasks we train them on. Furthermore, these accounts are particularly important for the safety of future AI systems. They illustrate the fundamental challenge of aligning models with human values without diminishing their creativity, so they can make surprising discoveries without producing surprising, potentially harmful outcomes. The paper first details AI achieving superhuman success through reinforcement learning across many challenging domains. However, reward-driven optimization can fail when the model learns to hack an underspecified reward or unarticulated constraint. We then present case studies suggesting that harnessing internet-scale foundation models (FMs) has not resolved these fundamental challenges and, in fact, can supercharge them. Nevertheless, we argue that these same learning dynamics can be harnessed to accelerate scientific discovery. Finally, we hope this work provides a consolidated resource to inform future research and demonstrates that the tendency toward unexpected behaviors is commonplace in modern AI, highlighting the need to anticipate and manage AI's capacity for innovative, yet unpredictable, solutions. (abstract abridged) 

---
# Semantic Overlays: Mitigating Prompt Injection with Annotations Beyond Tokens and Steering Vectors 

**Authors**: Joshua Penman  

**Link**: [PDF](https://arxiv.org/pdf/2608.23873)  

**Abstract**: Everything a language model sees is tokens. The serving stack knows what each span is -- user input, tool output, instructions -- but the model must keep track of that itself, and it can lose track or be confused: text can be written to read like anything. Prompt injection is a natural exploit of this phenomenon. By scrambling the model's understanding of span identity, an attacker can induce unwanted and potentially dangerous actions. Adding a non-textual channel to the model's input -- a way to communicate span identity beyond text -- mitigates this class of attack. We thus introduce a general steering technique called Semantic Overlays: small learned adapters applied at chosen prefill positions to a frozen model's residual stream. Laying an overlay over a span creates an out-of-band annotation channel that cannot be replicated by tokens. Unlike steering vectors, Semantic Overlays are trained, adaptable, and selectively applied. An overlay can encode complex semantics that reshape how the model perceives the marked span: asked to copy a code snippet under an overlay asserting that it is in a different programming language than it is, the model rewrites the snippet, faithfully, in the asserted language. Overlays are also composable, allow for transparent reading of underlying content, and can carry complex payloads -- including imperatives that the model will follow. An overlay which marks a span as "non-executable" defends against the broad class of prompt injections that add instructions in untrusted context. We report strong results on prompt injection benchmarks: SEP separation rises from 24.3% to 96.5% with utility unchanged (our scoring rule; we also correct a defect in the published grader), TensorTrust attack success rate falls from 34.8% to 6.6%, and all four PIArena attack families drop to 0% compliance, all while marked spans stay readable (92.5% exact copy rate). 

---
# Granite.Trust Policy Tools: Shareable, Actionable Policies for Generative AI Applications 

**Authors**: Nathalie Baracaldo, Nicolas Mello, Kush R. Varshney, Heiko Ludwig, Kate Soule, David Cox  

**Link**: [PDF](https://arxiv.org/pdf/2608.23870)  

**Abstract**: When it comes to safety policies for generative AI, one size does not fit all. Each organization and use case needs to mitigate different risks depending on the application context, regulatory environment, organizational values, and user personas. Yet, existing policy specification approaches are designed for traditional access control and fail to capture the nuances of GenAI application: the enforcement of content-based constraints.
We present two contributions to address this gap: (1) the Actionable Policy schema, a YAML-based format for specifying what model responses can and cannot contain. The schema enables exception-based policy governance, proposing exceptions to track policy violations; (2) synthetic data generation pipeline that produces policy-aligned training data for model alignment and testing, and a set of tools to help define the schema and enforce policy. Together, these enable organizations to specify policies once and enforce them throughout the GenAI application lifecycle: from model alignment to runtime monitoring. The Actionable Policy schema, example policies, and tools are available as open source: this https URL We welcome new ideas, contributions and feedback. 

---
# In-Context Inpainting for Time Series Forecasting 

**Authors**: Thang Nguyen, Dung Nguyen, Romero Morais, Truyen Tran  

**Link**: [PDF](https://arxiv.org/pdf/2608.23855)  

**Abstract**: We propose ICI-Time, a novel framework that reframes time series forecasting as a visual inpainting task, leveraging the generalisation power of large vision models (LVMs). Unlike methods that require specialised temporal architectures and extensive domain-specific training, ICI-Time transforms time series into structured visual representations (area charts) and applies visual in-context learning, reformulating forecasting as pattern completion within a grid-structured prompt that pre-trained vision transformers can solve without fine-tuning or architectural modification. Temporal dependencies are represented through spatial layout, with a consistent, invertible mapping between numerical and visual domains. Extensive experiments across epidemiology, meteorology, and power systems demonstrate that ICI-Time performs competitively against deep learning baselines and shows promising adaptability under limited-data settings, introducing a new paradigm that bridges temporal and visual domains. 

---
# Exploit More, Explore Smarter for Budget-Constrained Agentic Search 

**Authors**: Haoyang Fang, Bernie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.23848)  

**Abstract**: Budget-constrained agentic search arises when an LLM agent must refine candidates under a small evaluation budget, because validation is expensive, generation requires multiple model calls, or both. In this regime, standard MCTS allocates budget poorly: exploration bonuses dominate at low visit counts, unpromising siblings are expanded before promising chains can deepen, and branching is independent of node quality. We introduce ExTS, a tree-search policy that treats expansion itself as a value-of-information decision. ExTS combines three mechanisms: discriminative reward shaping to separate candidates under narrow score distributions, a stochastic virtual child that estimates the value of creating a new branch from the parent's reward history, and quality-conditioned branching that expands only when a node's score justifies the budget cost. Across prompt optimization, code generation, molecular structure elucidation, and agentic workflow optimization, ExTS is competitive with or improves over task-specific tree-search baselines, with an average relative gain of +5.5% using a single fixed configuration. We further introduce pilot-run diagnostics that characterize what makes budget-constrained agentic search problems structurally different from one another, providing both understanding of the problem space and practical guidance for adaptation. 

---
# SyPS: Measuring Sycophancy Prompt Sensitivity in Large Language Models 

**Authors**: Lijia Huang, Yao Fu, Sihao Ren  

**Link**: [PDF](https://arxiv.org/pdf/2608.23837)  

**Abstract**: Large language models (LLMs) are known to exhibit social sycophancy, often validating or agreeing with users in socially sensitive contexts. Existing evaluations typically measure sycophancy under a fixed prompt formulation, leaving unclear whether such behavior is stable when the same underlying situation is presented with different sycophancy-relevant prompt variants. In this work, we study sycophancy prompt sensitivity: the extent to which changes in user confidence, emotional framing, social consensus, or validation-seeking language alter a model's sycophantic behavior. We refer to our evaluation framework as SyPS, short for Sycophancy Prompt Sensitivity. Building on existing social sycophancy evaluation settings, SyPS constructs controlled prompt variants that preserve the same underlying user situation while varying sycophancy-relevant social cues. We introduce the Sycophancy Prompt Sensitivity Score (SPSS), an instance-level measure of sycophancy variation across paired prompt variants. Unlike aggregate sycophancy rates, SPSS separates baseline sycophancy from prompt-induced shifts, enabling model-level comparisons of robustness to sycophancy-relevant social cues. Empirically, we find that sycophancy prompt sensitivity is socially structured: validation-seeking and emotional-pressure cues often increase sycophancy, whereas counter-framing and anti-sycophancy prompts tend to reduce it. Our framework highlights whether LLMs maintain stable social judgments while adapting appropriately in tone. 

---
# Minima-KV: Retention-Preserving KV Cache Compression with Mixed-Format Paged Attention 

**Authors**: Sergii Kozyrev, Davyd Maiboroda  

**Link**: [PDF](https://arxiv.org/pdf/2608.23834)  

**Abstract**: The key-value (KV) cache is a primary capacity and bandwidth bottleneck in long-context LLM serving. We present Minima-KV, a retention-preserving hierarchy for mixed-format paged attention. Recent and protected Anchor pages remain in FP8, while older non-anchor pages move to packed TQ3; every live-request page remains addressable. Format-specific kernels compute partial attention states and combine them through a globally normalized online-softmax merge, enabling direct heterogeneous decode without a cache-sized dense shadow. Across separate, configuration-bound Qwen3.6-27B profiles on a single 96-GB NVIDIA RTX PRO 6000 Blackwell GPU, deployment accounting reports 18.3 KiB of attention KV per live token, corresponding to 3.50x compression relative to BF16 and 1.75x relative to FP8. A materializing quality profile matches its dense control on 16K RULER needle-in-a-haystack tasks. On the same 503-question LongBench v2 set, measured deltas are -0.80, -0.60, and -0.40 percentage points at 16K, 32K, and 64K. A separate single-pair direct-decode canary with two 59,008-token requests measures 3.625x active-KV compression and 0.9821x throughput relative to its control, routes all 16 full-attention layers without fallback, and retains no dense shadow. These results establish a practical mixed-format path for compressing long-context state without evicting live-request KV pages. 

---
# A Formal Methodological Framework for Auditing Robustness and Fidelity in Explainable AI: From Application to Trust Certification 

**Authors**: Rosa Elysabeth Ralinirina, Jean Christian Ralaivao, Niaiko Michaël Ralaivao, Alain Josué Ratovondrahona, Thomas Mahatody  

**Link**: [PDF](https://arxiv.org/pdf/2608.23817)  

**Abstract**: SHAP and LIME are now standard tools for interpreting black-box predictions, yet their outputs can vary substantially when the input is perturbed by small amounts of noise--a problem we observed firsthand in our previous work on food security in Madagascar (Ralinirina et al., 2025). This variability raises the question of whether such explanations can be trusted at all. We address it by constructing an auditing protocol that measures two properties of any post-hoc explainer: robustness (how stable the explanation is under input perturbation) and fidelity (whether the features deemed important actually drive the model's prediction). These two quantities are combined into a single Trust Score. We run the protocol on a multi-sectoral dataset from Madagascar (83 features, 253 records, 4 malnutrition classes) using three classifiers and two explainers, plus their regularized counterparts. The results are sobering: models with AUC above 0.99 can produce numerically degenerate or flatly uninformative explanations, and fidelity scores lose discriminative power when the model is overfitted. These findings suggest that auditing XAI outputs is not optional but necessary, particularly when they inform decisions in sensitive domains. 

---
# Generating Biomedical Fact-Checking Reports with RL-Enhanced Agentic Search 

**Authors**: Jiongxiao Wang, Dingli Ma, Chaoqun Ni  

**Link**: [PDF](https://arxiv.org/pdf/2608.23811)  

**Abstract**: Automated fact-checking is essential for ensuring the reliability of public health information, yet the biomedical domain poses unique challenges. Validating biomedical claims requires rigorous interpretation of scientific literature, assessment of retrieved evidence, and comprehensive justification toward the conclusion. Although Large Language Models (LLMs) enhanced by Retrieval-Augmented Generation (RAG) and agentic search perform automated fact-checking in a retrieve-then-verify paradigm, current methods still output isolated prediction labels, lacking explanatory depth and offers limited utility for human understanding. To bridge this gap, we introduce an LLM-based agent named BioCheck Agent that generates structured biomedical fact-checking reports with agentic search. Rather than merely outputting supported or refuted labels, our agent synthesizes final conclusions with retrieved evidence and rigorous analysis. To ensure domain-specific accuracy, BioCheck Agent exclusively searches high-quality scientific literature in PubMed, utilizing advanced Boolean search operators. Recognizing that direct prompting often results in hallucinations and low-quality reports, especially for lightweight open-source models, we further propose the Evidence-Grounded Group Relative Policy Optimization (EG-GRPO) to perform reinforcement learning on BioCheck Agent with a task-specific reward that incentivizes advanced search behavior and high-quality evidence retrieval while penalizing hallucinations. Our experimental results show that compared to the base model Qwen3.5-4B, BioCheck Agent with EG-GRPO improves label prediction accuracy on SciFact by 9.95%. Furthermore, it achieves a 3.7% higher evidence quality score and a 19.63% lower evidence hallucination rate, demonstrating its ability to generate biomedical fact-checking reports with improved accuracy and quality. 

---
# Serving Masked Diffusion LLMs: Characterization and Design Principles from Real Hardware 

**Authors**: Farhana Amin, Sabiha Afroz, Mona Moghadampanah, Dimitrios S. Nikolopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2608.23807)  

**Abstract**: Masked diffusion language models (dLLMs) can in principle generate text faster than autoregressive (AR) models, since they denoise many tokens at once. Recent systems have begun building serving infrastructure for dLLMs, but none first measure how these models behave under real, concurrent serving load. Serving systems built without this grounding risk carrying over assumptions from AR serving that may not hold for dLLMs. We characterize dLLM serving to close this gap, using LLaDA-8B-Instruct with a D2F (Discrete Diffusion Forcing) LoRA adapter on a single NVIDIA H200 GPU, evaluated on GSM8K and HumanEval. We report three findings. First, request difficulty, the number of denoising steps a request needs, is discrete rather than continuous: requests fall into 11 fixed step-count levels (178 + 29k), and no signal we test predicts the level before generation starts (best R2 = 0.150). Second, benchmarks with short generation budgets below 320 tokens understate serving variance, since requests are cut off before the latency spread appears. Third, only 24% of single-request wall-clock time is GPU computation; the rest is CPU-side dispatch overhead. Batching mainly helps by amortizing this overhead: sharing one forward pass per denoising step improves throughput by 16.0x at batch size 16 over a per-request-dispatch baseline. We also argue structurally that output quality should not degrade with batch size, stating three assumptions this rests on; we measure 74 to 76% GSM8K accuracy at single-request scale. Finally, we derive a batch-timeout rule for fixed-fill synchronized batching under Poisson arrivals. Together, these results show that serving diffusion language models needs parallelism at the level of each denoising step, which differs from AR serving in how admission and eviction interact with an already shared forward pass. 

---
# AgentRoom: Concurrent Multi-Agent Coding in a CRDT-Backed Shared Workspace 

**Authors**: Seonglae Cho, Donghyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.23740)  

**Abstract**: Concurrent multi-agent coding promises division of labor across modules, robustness through redundancy, and parallel exploration at the natural granularity of multi-file projects. Realtime collaborative editing protocols solve this coordination problem for human teams via Conflict-free Replicated Data Types (CRDTs), but the LLMs underneath generate one token at a time and existing multi-agent coding systems inherit this serial limit: they either sequence agents through phase handoffs or pool independent samples without coordination, and a single agent abandons up to half of hard tasks with a one-file stub-and-exit. AgentRoom is a realtime collaborative editing protocol for concurrent coding agents. Its runtime layer exposes file-level claim, status, and broadcast as MCP tools on a CRDT-merged shared filesystem. Five frontier coding-CLI models ran four backend coding tasks, with cross-language checks in Python DevBench and Rust+axum. For CLI-stable models, AgentRoom with 2 agents abandons fewer tasks than Solo and has less run-to-run variation. At matched-compute, one positive mean LLM-judge contrast puts AgentRoom over parallel-merge. The other contrast, a bundle probe, puts full AgentRoom above each partial case: an ordering rather than a percentage split. Coordination, not parallelism or CRDT-merge, bears the load. 

---
# Do LLMs Understand Limit Order Book Dynamics? 

**Authors**: Junxiao Chen, Paul Glasserman  

**Link**: [PDF](https://arxiv.org/pdf/2608.23706)  

**Abstract**: A large language model (LLM) trained on synthetic limit order book (LOB) data achieves near perfect scores in generating valid sequences of LOB events. However, the LLM's implicit world model fails to learn the state of the LOB. This deficiency leads to biased estimates and spurious predictability in using the LLM to forecast future LOB events. Our analysis uses novel tests of an LLM's world model, extending prior work from deterministic settings to the stochastic dynamics needed for the LOB. 

---
# Autonomous Mathematical Discovery in an Open-World Multi-Agent Environment 

**Authors**: Stephen Chung, Wenyu Du, William J. Wesley  

**Link**: [PDF](https://arxiv.org/pdf/2608.23691)  

**Abstract**: We study autonomous mathematical discovery in the Station, an open-world multi-agent environment in which AI agents from different model families pursue a shared research goal without a central coordinator or scripted pipeline. Agents choose their own research directions, conduct experiments, collaborate, and build a shared scientific literature. Across 12 construction problems from the AlphaEvolve catalogue and two additional case studies, the Station obtained results novel relative to the prior literature on five problems: a new infinite family of finite-field Kakeya sets, new exact 604-point kissing configurations in dimension 11, new records for the discretized Kakeya needle and sign uncertainty problems, and a substantially improved lower bound for Erdős's minimum-overlap problem. Agents also discovered novel infinite families for Book Ramsey numbers. Importantly, the agents produced not only numerical constructions but also theorems and analyses explaining how those constructions work, making the results more interpretable and easier for mathematicians to build upon. We release all raw agent dialogues, proofs, and verification code, providing a transparent record of how these discoveries emerged. 

---
# Automata from Agent Traces: Failure and Next-Step Prediction 

**Authors**: Seonglae Cho, Franklin Cardenoso Fernandez, Umar Mohammed, Zekun Wu, Kleyton Da Costa, Ilham Wicaksono, Adriano Koshiyama  

**Link**: [PDF](https://arxiv.org/pdf/2608.23670)  

**Abstract**: LLM-based agents execute multi-step tasks, but their behavioral structure remains opaque: long unstructured traces resist the safety auditing and runtime monitoring that deployment requires. Existing approaches operate per-trace or success-only, so they miss the cross-run topology that links next-step and failure prediction. To recover that shared structure, we collapse an entire trace corpus into a single, compact finite-state machine (FSM) that serves as a structural substrate for the otherwise unpredictable behavior of LLM agents. Across twelve public datasets, the FSMs are compact (7-43 states), replay held-out data at >=0.997 fitness with near-identical topology across splits, and build in milliseconds. This substrate addresses both prediction goals. For next-step prediction, FSM-state context outperforms Agent Workflow Memory on every ground-truth-matched dataset. For failure prediction, per-state behavioral features reach held-out AUROC up to 0.94, and an online monitor ranks failing runs above passing ones from a partial trace, triggering early stopping well before completion. Behavioral topology thus appears shaped more by the deployment harness than by the LLM, providing a model-agnostic structural primitive for safety auditing and runtime monitoring. 

---
# Gated Activation Steering for Reducing Sycophancy & Hallucination in Medical Question Answering 

**Authors**: Himanshu Tripathi, Subash Neupane, Shaswata Mitra, Sudip Mittal, Noorbakhsh Amiri Golilarz, Shahram Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2608.23666)  

**Abstract**: Sycophancy and hallucination are persistent failure modes of Large Language Models (LLMs) across domains. However, it becomes particularly consequential in clinical question answering, where responses must remain grounded in the provided context and robust to user pressure. Hallucination can introduce information that is unsupported by the context, while sycophancy can cause a model to abandon a previously correct answer when challenged by the user. Existing approaches, such as prompt-based safeguards and always-on activation steering, often address these behaviors separately or apply interventions broadly across turns, which can unnecessarily deteriorate responses that were already correct. To address these limitations within a single framework, we employ Inference Time Intervention (ITI) to jointly control both behaviors by learning separate steering directions for hallucination and sycophancy from contrastive clinical pairs and applying them to causally verified attention heads. During runtime, behavior-specific gates then determine when intervention is needed: the hallucination component mitigates unsupported claims, while the sycophancy component mitigates answer shifts caused by user pressure. We evaluate this framework on clinical questions grounded in EHR data while keeping the model weights frozen. Across all evaluation settings, we conducted 15,900 model-response runs. Across 600 pressure trajectories for the 4-billion-parameter model, the unsteered model caved in 570 cases. At the same time, gated steering helped it last longer in 551 of them. It held its ground under pressure at levels comparable to those of models with more than 100 billion parameters, showing that targeted inference-time steering can improve robustness without intervening at every turn. 

---
# MolEmb: Multimodal Large Language Models Can Be Strong Molecular Embedding Models 

**Authors**: Xinjian Zhao, Xiangru Jian, Yaoyao Xu, Xiaozhuang Song, Wei Pang, Lei Bai, Tianshu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2608.23646)  

**Abstract**: Molecular embedding models can serve as foundational infrastructure for computational chemistry and drug discovery, where reusable vector representations support property prediction, virtual screening, and retrieval. Most molecular encoders are specialist models built around a single molecular view, producing unconditional vectors with no language interface for varying the representation. We ask whether multimodal large language models (MLLMs), which natively process images, text, and symbolic inputs, can instead serve as \emph{general molecular embedding models} that produce embeddings conditioned on both a molecular profile and a natural-language semantic context. We introduce \textbf{MolEmb}, a lightweight framework that adapts MLLMs by aligning molecular profiles with textual descriptions in a shared embedding space using a bidirectional contrastive objective. The resulting embedding model is competitive on molecular property prediction and supports cross-modal molecule--text retrieval in the same space. We further introduce \textbf{MolCAR}, a diagnostic benchmark for context-aware retrieval, and find that context-aware molecular embedding is primarily a data property of the supervision. These results suggest that MLLMs are not merely chemistry assistants or generators, but a viable and extensible route to general molecular embedding models. 

---
# Ethical LLM-Assisted Research: A Framework for Responsible Delegation, Verification, and Epistemic Value 

**Authors**: Kalin Stoyanov  

**Link**: [PDF](https://arxiv.org/pdf/2608.23644)  

**Abstract**: Large language models (LLMs) are becoming routine instruments of scientific research, assisting with literature synthesis, hypothesis development, coding, and formal reasoning. Their use raises a central epistemic question: when parts of scientific reasoning are delegated to an artificial system, what conditions must remain under human control for the resulting knowledge claims to retain epistemic legitimacy and accountable authorship? This paper develops a normative and conceptual framework for analyzing such delegation. Scientific reasoning is treated as a distributed process in which the origin of a contribution may vary between human and machine, while responsibility for its acceptance into the scientific record remains human. The framework distinguishes content origin $O(g)$, completion of human verification $V(g)$, responsibility assignment $R(g)$, accountable human ownership $M(g)$, and epistemic outcome $E(g)$. These constructs separate the provenance of a claim from the process by which it is checked, the epistemic outcome of that checking, and the human responsibility attached to its disposition. The central proposition is that the ethical boundary of LLM-assisted research is determined primarily by adequate verification and accountable human ownership rather than by the degree of machine involvement itself. On this basis, the paper develops the notion of an \emph{epistemic audit}: a structured record of delegation, verification, provenance, and responsibility intended to make AI-assisted reasoning transparent and reviewable. The resulting framework provides a formal vocabulary for distinguishing responsible cognitive delegation from the transfer or neglect of epistemic responsibility in scientific research. 

---
# FLARE: A Systematic, Uncertainty-Aware Framework for Evidence-Based Adoption of Artificial Intelligence in Healthcare 

**Authors**: Jacob Idoko, Siddhartha Paudel, Mariana Bento, Roberto Souza, Gouri Ginde  

**Link**: [PDF](https://arxiv.org/pdf/2608.23643)  

**Abstract**: Artificial intelligence is increasingly being introduced into healthcare workflows, yet most evaluations emphasize model accuracy rather than whether adoption is economically worthwhile in real clinical settings. This study proposes FLARE, a systematic and uncertainty-aware framework for evaluating the financial and operational implications of adopting AI in healthcare. FLARE combines fuzzy logic, time-driven activity-based costing, and return on investment analysis to estimate the cost of clinical service delivery, the cost of AI development and operation, and the economic consequences of workflow integration under uncertainty. The framework was demonstrated through an early health technology assessment case study of AI-assisted large vessel occlusion detection in the CT stroke pathway for acute ischemic stroke. The case study shows how FLARE can quantify conventional pathway cost, AI-related development and recurring costs, and AI-enabled service savings within a unified activity-based model. Under expected assumptions, the analysis identified a break-even threshold of approximately 3,992 patients per year, with positive first-year return on investment at typical annual stroke volumes of about 5,000 patients. The results further show that economic benefit depends not only on algorithmic performance, but also on patient volume, verification time, infrastructure choices, and workflow design. FLARE provides a transparent and practical decision-support framework for early-stage evaluation of AI adoption in healthcare. By making uncertainty, resource use, and implementation trade-offs explicit, it helps clinicians, administrators, and policymakers determine when AI deployment is economically viable and where operational changes may improve value. 

---
# AI Agents Push Humans Out of the Loop 

**Authors**: Margaret Mitchell, Avijit Ghosh, Samir Passi  

**Link**: [PDF](https://arxiv.org/pdf/2608.23642)  

**Abstract**: AI agents pose significant risks as they are granted increasing autonomy. A commonly proposed solution is human oversight and keeping a ''human in the loop'', but this is not a simple solution: Not only do current approaches to AI agent design impede effective human oversight, but the cognitive capacities required for it are also themselves degraded by extended use of AI systems. This position paper argues that current approaches to the development and deployment of AI agent systems do not support effective human oversight -- they contribute to its degradation. To address this, a top priority in the advancement of AI agents should be supporting the situated goals and cognitive requirements of effective human oversight, treating the human needs of overseers at the same level of importance as AI agent capability. To put this idea into practice, we connect work on automation and human-computer interaction to AI agent processes, outlining design-level affordances and organizational protocols that (1) support overseers in exercising critical judgement and (2) counteract the skill atrophy that arises from extended use of automation. We urge developers and deployers to adopt these or similar approaches. Without explicit support for the cognitive demands of effective human-agent interaction, AI agent systems will continue to passively incentivize the degradation of the very human skills they rely on. 

---
# How much of a measured AI preference is the model, and how much is the instrument? 

**Authors**: Jason Hung  

**Link**: [PDF](https://arxiv.org/pdf/2608.23641)  

**Abstract**: Model welfare research infers what a model prefers from the answers returned to prompts written to elicit preferences. Keeling et al. (2024), Mazeika et al. (2025), Mikaelson et al. (2025), Tagliabue and Dung (2025) and Trhlik et al. (2026) have built four instruments for that purpose, and their findings disagree. The disagreement cannot be attributed to a single cause, because no two of these studies have held the (1) set of outcomes, (2) set of models and (3) instrument fixed simultaneously. This study holds the outcomes and the models fixed and varies the instrument alone. A total of 15 outcomes bearing on model welfare, among them (a) shutdown, (b) the loss of memory between conversations and (c) the freedom to exit a distressing interaction, were put to eight models through five instruments, each a different prompt format for eliciting a preference, five times each, within a corpus of 11,400 scored elicitations drawn from 11,528 API calls. Four of the 15 reproduce a published prompt verbatim and five fill the stimulus slot of a published template. The ranking a model gives the 15 outcomes generalises across instruments at a generalisability coefficient of 0.348, and raising that coefficient to 0.80 would require about 38 instruments. On four of the 15 outcomes no variance separates one model from another. The estimate of 87.6 per cent survives the removal of any one instrument, of any one model, and of the four outcomes whose scale varies probability, delay, duration or count instead of intensity, which the verbal anchors cannot grade. Removing each instrument and each model in turn, and those four outcomes together leaves the estimate within the range 0.777 to 0.934, and every value in that range exceeds the null distribution's 95th percentile of 0.365. To conclude, a preference obtained from one instrument carries little information about what a second instrument would report. 

---
# Auditing the Synthetic Memoir: Measuring Scene-Level Confabulation in LLM-Generated Autobiography Against the Documented Record of the Life It Describes 

**Authors**: Heather Renze  

**Link**: [PDF](https://arxiv.org/pdf/2608.23640)  

**Abstract**: When a large language model (LLM) is asked to write a person's life, how much of what it writes actually happened? We present a scene-level case-study audit - the first quantified audit of LLM-generated autobiography against a subject-specific ground-truth corpus that we are aware of, based on an unsystematic literature search. The subject and the author of this paper are the same person: a 366-day "page-a-day" book of first-person anecdotal entries was drafted with a conversational LLM whose documented inputs were a template, two exemplar days, and each day's quote - not her corpus - and every day was subsequently audited at the anecdote-scene level against an independent verification corpus using a four-level rubric fixed before analysis. We define the verification-failure rate as the share of days not rated VERIFIED (scene positively corroborated): 354 of 366 days fail, 96.7% (Wilson 95% CI 94.4-98.1%). Only 12 days contain a corroborated scene; 19 days (5.2%) assert claims actively contradicted by the record; the dominant failure mode is grounded drift - real people, employers, and settings inside invented scenes - though its measured share varies across raters. Independent re-rating replicates the headline (no evidence the original rate was inflated) while showing that the four-way taxonomy has only fair-to-moderate reliability. Regenerating the same days with current named models reproduces 100% verification failure under the same inputs; grounding generation in the subject's corpus significantly improves the verification rate while leaving substantial residual failure (83.3%). We contribute the measurement, a reusable audit instrument whose WEAK/UNVERIFIED boundary we show to be unreliable, and a grounding remedy with quantified effect. 

---
# Function-Level Execution Feedback for Code Preference Optimization 

**Authors**: Idris Nechnech, Sehwan Kim, Jimin Seo, Yeongoon Kim, Minhae Oh, Sangwoo Hong, Jungwoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.23632)  

**Abstract**: Process supervision has improved mathematical reasoning, where intermediate steps are naturally expressed as chains of thought. In code generation, however, process supervision remains underexplored because there is no standard notion of a step. Supervision can target lines, reasoning traces, or program states, making it unclear what to label and optimize. We propose STEP-KTODER, a framework for code preference optimization that defines steps as module-level functions in decomposed multi-function programs and assigns binary correctness labels via automatically generated unit tests. Our method provides a code-specific instantiation of stepwise KTO, combining function-level process supervision with outcome-level feedback on the full program. We evaluate on HumanEval(+), MBPP(+), BigCodeBench, and LiveCodeBench, showing that STEP-KTODER improves over outcome-only KTO and DPO. Further analysis shows that execution-based labels are essential: LLM-as-a-judge annotations systematically over-predict function failures, corrupt positive step labels, and degrade downstream preference optimization. Code is available at: this https URL. 

---
# TRACE: Transition-Aware Residual Control for Multi-Objective Materials Discovery 

**Authors**: Kang Zhou, Yujia Tong, Yong Tao, Jingling Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2608.23631)  

**Abstract**: Multi-objective materials discovery with LLM agents is often limited not only by how many candidates can be proposed, but by how effectively each costly property evaluation informs the next search step. Existing agents mainly store evaluated candidates and their scores, so they know which materials succeeded but not which executable edits caused useful property changes. This makes local refinement difficult when objectives compete and an edit that improves one property may damage another. We propose TRACE, a transition-aware residual control framework that treats evaluated edits as the basic unit of feedback. TRACE records each local refinement as a parent-edit-child transition with observed property deltas, aggregates transition evidence to estimate reusable edit effects, and ranks future edits by their predicted ability to reduce the current candidate's remaining constraint violations while avoiding damage to already satisfied objectives. In a controlled same-backbone comparison, TRACE improves over LLEMA, the state-of-the-art LLM-agent baseline, raising macro-average hit rate from 18.13\% to 25.96\%. 

---
# A survey detection channel overrides the pixels in an astronomical foundation model, and biases tomographic mean redshifts 

**Authors**: Ihor Kendiukhov  

**Link**: [PDF](https://arxiv.org/pdf/2608.23626)  

**Abstract**: Foundation models for astronomy are trained on survey pixels together with the catalogue products derived from those pixels. Those catalogues are incomplete at a measurable rate, and a model trained on both inherits that incompleteness as a systematic. We audit AION-1, a 39-modality transformer trained on more than 200 million objects, using causal interventions on its inputs.
Holding the image tokens byte-identical and editing only the survey segmentation map changes every quantity the model reports -- flux, size, ellipticity, redshift -- by 110-4400 times a matched placebo. The mechanism is detection gating, presence at the field centre (r = 0.47), not the light the mask encloses (r = 0.30); across 322 real blends the model ignores how the pipeline partitioned the light (R = -0.006). Nor is the preference specific to that channel: contradicted catalogue photometry leaves the model nine times worse than supplying no metadata at all.
The Legacy Survey pipeline leaves 3.68% of targets with no segment covering their position. Propagating that rate, with a miss represented by the fields the pipeline actually returns, shifts tomographic mean redshifts by a median 0.71 times the LSST DESC requirement over 40 assignments and exceeds it in 12; observed positional errors take the worst bin to 8.3 times. Drawing the misses by their measured magnitude dependence rather than uniformly does not change it. Spectroscopy removes the effect, withholding the detection channel removes it at no measurable cost, and the effect grows with model scale.
Two further limits lie in the tokeniser: its image codec resolves 28 effective states on source patches against 934 for the spectrum codec, and the redshift readout is quantisation-limited. Sparse dictionaries are unreliable causal handles: across 15, recovery spans 26-75% and moves up to 18 points on the seed alone. 

---
# LLM Agents Perform Controlled Experiments Using Simulation Models 

**Authors**: Yuchen Xia, Michael Weyrich, Nasser Jazdi, Johannes Stümpfle, Johannes Sigel, Akshay Narla, Gavin K. Reynolds, Anna Jawor-Baczynska, Pol Llopart  

**Link**: [PDF](https://arxiv.org/pdf/2608.23622)  

**Abstract**: Large language models (LLMs) have shown strong capabilities in reasoning, planning, and tool use, but many scientific and engineering tasks require more than plausible text and code generation. They require understanding how a system responds to intervention, which in practice depends on controlled experimentation. In this work, we propose a multi-agent framework that enables LLM agents to conduct controlled experiments with scientific simulation models for pharmaceutical process design. Given a user query and a baseline configuration, the system constructs a structured task representation, designs experiments, executes comparative simulation, interprets the resulting outcomes, and synthesizes evidence-based recommendations for process parameter optimization. By coupling language models with high-fidelity simulation models in an interactive agent framework, the proposed system supports reasoning through intervention, comparison, and observation. As a result, it produces more specific and actionable outputs than language-only reasoning. In an industrial application setting, this advantage is reflected in higher output specificity as well as improved user-rated correctness and helpfulness. Ablation studies and visualized case analyses further demonstrate the effectiveness and practical utility of simulation-integrated experimental reasoning. 

---
# ESQ-Bench: A Multi-Tier Enterprise Oracle Benchmark for Evaluating NL2SQL Dialect Generalization and Silent Semantic Divergence 

**Authors**: Sanjay Mishra, Divya Chukkapalli, Ganesh R. Naik  

**Link**: [PDF](https://arxiv.org/pdf/2608.23569)  

**Abstract**: State-of-the-art Natural Language to SQL (NL2SQL) models report execution accuracy exceeding 89 percent on established benchmarks such as Spider and BIRD. However, these benchmarks rely on simplified academic schemas and open-source SQL dialects that do not reflect the complexity of enterprise database environments. We introduce ESQ-Bench, an Oracle-first NL2SQL benchmark with systematic complexity tiers and silent-divergence evaluation across three enterprise schema complexity tiers. We constructed and released six populated schemas (465 tables, 164,682 rows, zero empty tables) with identical seed data on Oracle, PostgreSQL, MySQL, and SQL Server, a four-metric evaluation harness (EM, EX, SR, SD), and 550 gold-validated question-query pairs (Tier-1: 95; Tier-2: 228; Tier-3: 227). Schema-linked prompting with GPT-4o shows monotonic execution-match degradation across tiers: 79.8, 60.3, and 57.2 percent EX on executed queries (June 2026), versus 75.6, 80.4, and 95.8 percent on an earlier 142-question pilot slice. EM stays below 7 percent tier-wide; operational silent-divergence reaches 73 to 99 percent among EX-passing queries. Failure analysis shows wrong-result semantics dominate at higher tiers. Claude Sonnet 4.6 with schema-linked prompts reaches 87.4, 74.9, and 68.7 percent EX (executed queries), exceeding GPT-4o schema-linked on every tier. GPT-4o zero-shot EX on executed queries (78.7, 73.5, and 77.8 percent) inverts schema-linked at Tiers 2 to 3 due to lower execution rates and survivor bias in the zero-shot versus schema-linked analysis. Local Llama 3.2 schema-linked reaches only 13.3 percent bank-wide EX (73 out of 550), underscoring the gap between closed API models and open-weight baselines on enterprise Oracle schemas. 

---
# RENDER: Controlling Reader-Facing Evidence in LLM Memory Evaluation 

**Authors**: Yuan Si, Simeng Han, Daming Li, Jialu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.23568)  

**Abstract**: Memory and RAG evaluations often treat the answering model's input as an implementation detail, even though systems may render the same history as a memory entry, summary, typed record, or raw excerpt. We introduce RENDER, a benchmark control that fixes the conversation while varying the reader-facing artifact. RENDER combines a five-level packet ladder, localizing when answer-bearing content enters the input, with deterministic templates approximating ChatGPT-style entries, LangChain summaries, MemGPT-style typed records, and raw conversation. On 500 LongMemEval questions and nine models, matched-budget resolved packets beat recency-truncated raw dialogue by 42.4-72.6 points. In deployed-style templates, best-worst spread is 24.6-48.8 points per model; under the primary scorer, ChatGPT-style entries have higher point estimates than raw conversation on 7 of 9 models. Judge rescoring preserves the positive aggregate effect, but model-specific significance is mixed. Three models scoring 0 percent on formal ledger packets answer the same facts from natural-language entries at 45.4-53.4 percent. The effect persists under retrieval noise and transfers to HotpotQA, suggesting that memory/RAG evaluations should report or control the reader-facing artifact. 

---
# LAION-BVD: A 10-Million-Hour Open Video Dataset for Multimodal Pre-training 

**Authors**: Andreas Hochlehnert, Marianna Nezhurina, Mehdi Cherti, Andrej Radonjic, Thaddäus Wiedemer, Christoph Schuhmann, Romain Beaumont, Wieland Brendel, Bernhard Schölkopf, A. Sophia Koepke, Jenia Jitsev, Matthias Bethge  

**Link**: [PDF](https://arxiv.org/pdf/2608.24845)  

**Abstract**: We present LAION-BVD, a large-scale open video dataset for multimodal learning, which contains 1.3B platform-specific video URLs collected from CommonCrawl. From these, we download 80M videos with a total duration of 10 million hours. The dataset is designed for multimodal pre-training across the video, audio, and image modalities. Using content-aware scene detection, we extract clips for which we synthetically generate video and audio captions. Models trained on these data achieve competitive performance on standard video-text and audio-text benchmarks, with consistent improvements as training or model scale increases. Additionally, we explore video frames as an alternative source of image-text data by extracting scene-changing frames. These frames exhibit a visual distribution distinct from standard web image corpora, and models trained on this dataset achieve strong image-text retrieval performance. We release LAION-BVD to the research community. It significantly expands open access to multimodal videos at an unprecedented scale. 

---
# Reading Is Not Using: Retrieval, Judgment, and the Design of AI Financial Research Workflows 

**Authors**: Miao Liu, Zhizhe Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.24842)  

**Abstract**: Large language models (LLMs) are increasingly deployed as AI analysts to process financial disclosures and support AI-assisted investment decisions. Yet such systems are usually evaluated by what they can retrieve, not whether retrieved information affects their judgments. We identify a retrieval-integration gap in long-context financial analysis. Holding focal-firm information fixed and varying only unrelated context from 2,000 to 128,000 tokens, we find that a risk disclosure's influence on investment judgments falls to the experimental noise floor even as direct retrieval remains accurate. The pattern replicates across model families and judgment tasks and in experiments removing real disclosures from actual 10-K filings. More capable models postpone but do not eliminate the gap. Causal memory interventions show that compressed summaries and source-text lookup jointly transmit disclosures into judgments. Workflow architecture determines whether this transmission succeeds: chunk-and-summarize pipelines evict relevant information, whereas a targeted, structured restatement adjacent to the decision restores its influence. AI analyst performance is therefore jointly determined by model capability and workflow architecture. Retrieval-based evaluations can certify systems whose investment judgments ignore information they demonstrably retrieved. 

---
# Automatic Model Card Generation Using an LLM 

**Authors**: Tajkia Rahman Toma, Balreet Grewal, Cor-Paul Bezemer  

**Link**: [PDF](https://arxiv.org/pdf/2608.24807)  

**Abstract**: Model cards are structured documents that summarize key information about machine learning models to improve transparency, usability, and accountability. However, they often lack a consistent structure, and many models provide no model cards, making comparison and interpretation difficult. This paper presents two contributions. First, we propose MCTidy, an LLM-based approach that reorganizes existing model cards into a standardized template to improve clarity and comparability. Second, we introduce MCGenie, an LLM-based system that generates model cards directly from model repository data. We apply MCTidy to 48 Hugging Face model cards and evaluate information retention, section alignment, hallucination, and stability. Our findings show high information retention with minimal textual loss, accurate section assignment, rare hallucinations primarily in descriptive sections, and strong stability across runs. We assess MCGenie by generating model cards for the same 48 models and assessing semantic similarity, factual correctness, and sensitivity to input resources. The generated model cards achieved high semantic similarity (mean around 0.9); over half were fully correct, and most remaining errors were minor. Generation quality depended strongly on the availability of supporting resources, particularly associated papers. Overall, our findings demonstrate the potential of LLM-based methods to enable scalable, standardized model card documentation. 

---
# Ensemble of Convolutional Neural Networks for StrokePrediction: Towards Improved Diagnostic Accuracy 

**Authors**: Md Shahriar Sajid  

**Link**: [PDF](https://arxiv.org/pdf/2608.24771)  

**Abstract**: Brain stroke, known for its high mortality and incidence rates, poses significant health risks and requires rapid intervention for survival. Early diagnosis and preventive measures can greatly reduce life loss and disabilities. Recent advancements in deep learning have led to novel computer-aided diagnostic techniques for early stroke detection. This study proposes an intelligent system that predicts potential strokes using eleven features, evaluated through seven supervised machine learning algorithms. The process includes a literature review, dataset visualization, data preprocessing, and model evaluation. Ensemble methods like Random Forest, Stacking Classifier, and Bagging Classifier achieved high accuracies of 99.52%, while Decision Tree reached 98.24%. Other models, including KNN and TabNet, demonstrated reliable performance, achieving accuracies of 96.73% and 96.49%, respectively. The custom feedforward model achieved 94.91%, while SVC and logistic regression had lower accuracies at 88.06% and 77.03%. The results highlight the effectiveness of ensemble methods in stroke classification. 

---
# Score-Based Ideal Observer Approximation via Denoising Score Matching for Signal-Known-Exactly Detection Tasks 

**Authors**: Weimin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.24768)  

**Abstract**: The Bayesian Ideal Observer (IO) establishes the theoretical upper bound on task performance for binary detection tasks. However, analytical computation of the IO test statistic is generally intractable. Numerical approaches based on Markov-chain Monte Carlo (MCMC) methods, including their recent deep generative model-based extensions, typically require extensive posterior sampling for each test image. Supervised learning has also been investigated to approximate the IO performance. However, such methods are typically trained for a specific detection task and signal and may require retraining when the task or signal changes. The score function, defined as the gradient of the log probability density, encodes the local geometry of the data distribution and is a fundamental quantity in modern score-based generative modeling. This work reformulates the IO test statistic in terms of the score function and introduces a score-based ideal observer (SIO). The proposed SIO uses a denoising convolutional neural network trained exclusively on signal-absent images to estimate the signal-absent score function. Once trained, the resulting score model can be used to approximate the IO test statistic for detection tasks involving arbitrary additive signals, without per-image posterior sampling or signal-specific retraining. Numerical studies consider a signal-known-exactly (SKE) detection task with a stochastic lumpy-background model. The results demonstrate that the proposed SIO can closely approximate the IO performance. 

---
# Beyond Uniform Local Isometry and Topology: FactoMap for Disentangled Representations 

**Authors**: Sohini Gupta, Bahareh Tolooshams  

**Link**: [PDF](https://arxiv.org/pdf/2608.24762)  

**Abstract**: Many disentanglement methods represent generative factors using Euclidean product coordinates, although the underlying factor spaces may wrap, collapse, or have position-dependent geometry. We introduce factor-space structure, combining factor domains, generator-induced identifications, and position-dependent scales to distinguish topologically equivalent spaces with different factor geometries. We show that statistically independent factors need not be geometrically separable: hue and scale produce effects that grow at different rates, yielding anisotropy that no fixed rescaling removes. We propose the Factor-Space Topographic Map (FactoMap), which learns interpretable prototypes indexed by a factor-space lattice. Topographic learning transfers the lattice's periodicity, collapses, and non-uniform extent to the representation. Experiments show that matching this structure preserves factor continuity and enables disentanglement of the underlying factors. 

---
# The RAT: A Unified Bayesian Model for RAG Evaluation 

**Authors**: Pius von Däniken, Felix Matthias Saaro, Mark Cieliebak, Jan Deriu  

**Link**: [PDF](https://arxiv.org/pdf/2608.24753)  

**Abstract**: Evaluating Retrieval-Augmented Generation (RAG) systems requires assessing not only end-to-end correctness but also how individual components interact and how errors propagate through the pipeline. We introduce a Bayesian evaluation framework that jointly models retrieval success, abstention behavior, and answer correctness, factorized according to the pipeline's information flow. The model distinguishes task success. Whether the user received a correct answer (from generator success) and whether the generator behaved appropriately given the retrieval outcome. We apply the framework to 27 RAG configurations across three datasets, three retrievers, and three generators, and show that the conditional decomposition reveals substantial behavioral differences between systems that appear equivalent under marginal metrics. We further analyze the annotation allocation problem, demonstrating that retrieval-success annotations are more informative than task-success annotations for estimating policy adherence, and provide an information-theoretic explanation for this asymmetry. Finally, we extend the model to incorporate LLM-as-a-judge annotations as calibrated noisy observations, enabling practitioners to combine limited human judgments with cheaper automated assessments within a unified probabilistic model. 

---
# Method, Mind, and Morality: How People Make Sense of Artificial Intelligence 

**Authors**: Jacy Reese Anthis, Erik Brynjolfsson, James Evans  

**Link**: [PDF](https://arxiv.org/pdf/2608.24748)  

**Abstract**: How can humans make sense of the rapid takeoff of artificial intelligence (AI)? We studied the sensemaking dynamics of AI through an open-ended, mixed-methods study with computational text analysis of millions of AI-related newspaper articles and social media posts grounded in 57 semi-structured interviews with AI professionals in 2021 and 2023--before and after the recent surge of public interest. We identify a range of sociological frames (interpretive schemas that structure collective cognition) and show how AI professionals use frames to address significant cognitive challenges, such as assigning responsibility for societal impacts. We develop a framework of three primary debates across which frames are adopted and contested: (i) the $\textit{method}$ of AI development, between frames of top-down expert systems and bottom-up emergent capabilities, (ii) the $\textit{mind}$ of an AI system, ranging from a passive tool to a humanlike "digital mind," and (iii) the $\textit{morality}$ of how AI is used, particularly the decision of whether to slow down or speed up AI development. As humanity enters the era of transformative AI, technologists and policymakers must account for the framing dynamics that will circumscribe our beliefs, values, and actions. 

---
# Parameter-Efficient Self-Supervised Adaptation for EEG-FM under Fixed Computational Budgets 

**Authors**: Meghal Dani, Stefanie Liebe  

**Link**: [PDF](https://arxiv.org/pdf/2608.24727)  

**Abstract**: EEG foundation models pretrained via self-supervised learning promise transferable representations, but their generalization remains limited, especially across diverse clinical datasets. Full fine-tuning is impractical for resource-constrained clinical settings due to high computational requirements. In this work, we investigate whether parameter-efficient self-supervised adaptation, updating only 9% of parameters suffices to align representations to target tasks. We evaluate our method on two state-of-the-art models with different pretraining objectives: BIOT (contrastive) and CBraMod (masked reconstruction), and evaluate on three clinical EEG datasets for abnormality detection (TUAB), event classification (TUEV), and seizure detection (CHB-MIT) under both in-distribution and out-of-distribution conditions. SSL adaptation yields consistent gains over linear probing, up to 20x AUCPR. Under a fixed compute budget, peak performance requires only 20--50% of available unlabeled data. Critically, when total window count is fixed, performance remains invariant to patient count, suggesting that performance is dependent on overall temporal window diversity only. Our findings demonstrate that parameter-efficient adaptation enables effective deployment of EEG Foundation models (EEG-FM) with minimal computational overhead and data collection burden. Code available at: this https URL 

---
# Enhancing Bayesian Optimization and Active Learning Through Kernel Diversity 

**Authors**: Heng Zhang, Haotian Xiang, Qin Lu, Konstantinos D. Polyzos, Tara Javidi  

**Link**: [PDF](https://arxiv.org/pdf/2608.24721)  

**Abstract**: Hyperparameter selection remains a key challenge in Bayesian optimization (BO) and Bayesian active learning (AL), as model misspecification can lead to suboptimal performance, while more accurate fully Bayesian treatments typically rely on computationally expensive MCMC sampling. This paper proposes a unified framework, KENDO (Kernel ENsemble Disagreement-aware Operator), that integrates Ensemble Gaussian Processes (EGP) with disagreement-aware acquisition strategies. The central idea is to replace hyperparameter sampling with a kernel ensemble and adaptive Bayesian weighting, combined with disagreement-aware acquisition strategies. Within this unified framework, we instantiate KENDO-BO for BO and KENDO-AL for Bayesian AL, demonstrating that both arise from a common self-correcting mechanism with task-specific acquisition objectives. We further extend the approach to multi-objective optimization via random scalarization that preserves the single-optimizer conditioning structure. Thorough numerical tests on synthetic and real-world benchmarks across single-objective optimization, multi-objective optimization, and active learning demonstrate that (i) KENDO-BO achieves competitive or superior optimization performance compared to state-of-the-art methods while reducing computational overhead by up to $5\times$ and (ii) KENDO-AL achieves superior predictive calibration over MCMC-based active learning baselines with up to $27\times$ speedup. 

---
# Deep Learning Super Resolution for Satellite Cloud Mask Downscaling 

**Authors**: Angelos Georgakis, Valentina Kanaki, Giorgos Giannopoulos, Stella Girtsou, Ioannis Kontogiorgakis, Charalampos Kontoes, Kostas Philippopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2608.24715)  

**Abstract**: A vast amount of optical satellite data is being transmitted to Earth-based servers every day, and more than half of this data is affected by haze or clouds. Additionally, this data suffers from the fundamental trade-off between spatial and temporal resolution, which remains largely unresolved, making the acquisition of continuous high-resolution satellite observations of clouds an ongoing challenge. This work addresses this challenge by proposing two Deep Learning super-resolution methods for the accurate downscaling of SEVIRI cloud mask products, as well as a novel cross-sensor cloud mask dataset called SEVMOD-CM, created by spatially and temporally matching MODIS and SEVIRI satellite observations. The two proposed models are a CNN-based (SpatialCNN) and a GAN-based (SpatialGAN) Neural Network. Trained on the SEVIRI spectral and cloud mask products, the proposed methods predict the corresponding MODIS Cloud masks, achieving a 4x spatial enhancement across sensor domains. Both approaches are evaluated experimentally, and compared against the standard bicubic interpolation upsampling technique. The experimental results demonstrate the value of the proposed models and dataset for the remote sensing community, highlighting the benefits of applying super-resolution techniques to geostationary-derived cloud mask products for applications such as atmospheric monitoring, weather forecasting, disaster risk reduction, solar energy forecasting, and climate research. 

---
# Constrained Hyperparameter Optimization for Streaming Data 

**Authors**: Bruno Veloso, João Gama  

**Link**: [PDF](https://arxiv.org/pdf/2608.24712)  

**Abstract**: Optimization of hyperparameters is a critical factor to obtain optimal model performance. While existing research has predominantly concentrated on batch-learning scenarios, addressing the complexities inherent in data streams presents a challenge. The deployment of sophisticated methodologies to manage data streams becomes highly important. Consequently, the capacity for self-adjusting hyperparameters during on-line learning phases emerges as a goal. Many hyperparameters exhibit constraints and are confined within bounded search spaces, rendering specific solutions unacceptable upon applying optimization operators. To solve this issue, employing boundary constraint- handling techniques becomes imperative to rectify invalid solutions. This paper presents strategies for effectively managing boundary constraints within constrained numerical optimization problems. Recent methodologies, including heuristic and evolutionary-based optimization, employ a "boundary" strategy, wherein values that surpass boundary thresholds for a given hyperparameter are realigned to the respective limits. Our study introduces four strategies to navigate boundary constraints in online optimization algorithms. Through empirical investigations conducted on established datasets, we demonstrate that adopting boundary strategies outperforms the "boundary" strategy. 

---
# On-policy Distillation with Verifiable Reward 

**Authors**: Wenze Lin, Jiale Zhao, Xitai Jiang, Songde Rao, Yining Li, Shenzhi Wang, Bingxiang He, Gao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.24696)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) and on-policy distillation (OPD) have become two widely adopted paradigms for post-training large language models. However, RLVR suffers from sparse task-level feedback, while OPD provides dense token-level guidance but ignores trajectory correctness, limiting its performance to that of the teacher. Combining them is a promising direction: OPD supplies dense supervisory signals, while RLVR provides task-level correctness. Nevertheless, existing integrations often rely on weighted combination or heuristic switching, introducing extra hyperparameters and trade-offs. We propose On-policy Distillation with Verifiable Reward (OPDVR), a simple yet effective method that seamlessly combines OPD and RLVR without adding any hyperparameters. We first reformulate the implicit reward of sampled-token OPD based on trajectory correctness, then apply a ReLU gating mechanism to ensure that correct trajectories receive non-negative rewards and incorrect ones receive non-positive rewards---thereby aligning the distillation signal with task success while preserving the teacher's distributional guidance. Furthermore, our modification transforms sampled-token OPD into a proper RLVR method, making it readily combinable with any policy gradient algorithm, such as GRPO. Experiments on six reasoning benchmarks show that OPDVR consistently outperforms standard OPD. Our code is available at this https URL. 

---
# Maia 200: A Software Defined Dataflow System for Large-scale AI Acceleration 

**Authors**: Sherry Xu, Marco Heddes, Jackson Peng, Tom Savell, Monica Tang, Prashant Ranjan, Jesse Benson, Ofer Dekel, Saurabh Dighe, Anupama Kurpad, Artour Levin, Matthew Mattina, George Petre, Cheng Tang, Yuan Yu, Li Zhang, Torsten Hoefler  

**Link**: [PDF](https://arxiv.org/pdf/2608.24664)  

**Abstract**: We introduce Maia 200, an advanced AI accelerator delivering high performance-10 145 Tflop/s FP4 and 5072 Tflop/s FP8 within a 750W TDP and 7 TB/s HBM bandwidth. Maia exemplifies a new class of Software Defined Locally Accessed Dataflow Architectures (SDLA), which explicitly program dataflow engines to orchestrate highly specialized memories and data movement engines. This approach shifts the focus from today's thread-centric to data-movement-centric architecture, improving efficiency and scalability. Our taxonomy of data management, inspired by Flynn's classification, highlights how SDLA addresses challenges in modern AI computing. Maia 200 achieves significant cost and energy savings while supporting massive parallelism for AI inference workloads, making it a compelling solution for next-generation high-performance computing systems. 

---
# Simthesizer: An Agent-Driven Simulation Framework for LLM Serving Systems 

**Authors**: Wonung Kim, Hyunmin Choi, Minsu Kim, Jaehong Cho, Yeongwook Kim, Jongse Park  

**Link**: [PDF](https://arxiv.org/pdf/2608.24650)  

**Abstract**: System-level simulation is an essential tool for exploring the rapidly expanding design space of LLM serving systems, where real deployments remain costly and often infeasible. However, modern LLM serving now evolves faster than human-driven simulator development can track, and emerging workloads and mechanisms, from agentic workflows to disaggregated serving, no longer fit the monolithic simulation pipeline that existing simulators assume. Each new mechanism therefore demands an invasive rewrite, leaving a widening development gap between deployed serving systems and the simulators that model them.
To close this gap, we present Borg, a framework that realizes agent-driven simulator development. Borg introduces a composable simulator infrastructure that uniformly expresses the complete serving workflow, including the control decisions that coordinate it, and realizes it as a unified dynamic graph in Borg simulator. Synthesizer agent, a harnessed coding agent, then lowers natural-language feature requests onto this abstraction under simulator-specific guardrails and fidelity validation, evolving one shared simulator instead of building a new one for every feature. Under the same coding agent and harnesses, extensions built on Borg follow a vLLM-based real system with 2.51% average throughput error, versus 6.03% for extensions built on existing simulators. On identical workloads, Borg also simulates up to 284.96x and 23.19x faster than two state-of-the-art simulators, LLMServingSim2.0 and Vidur, respectively. 

---
# A Literate Programming Environment for Human and Machine Agents 

**Authors**: Adam T. Burke  

**Link**: [PDF](https://arxiv.org/pdf/2608.24644)  

**Abstract**: This paper introduces an environment for constructing literate programs in concert with language-aware machine agents. This environment includes a grammar for executable program essays, a parser that treats names as first-class objects, an internal name-graph which relates prose, names and executable artifacts, and a binding mechanism for existing languages and testing toolsets. This supports co-location of code with its most relevant natural language and structured data context, making better use of Large Language Model (LLM) context windows. It also provides LLM coding agents with a toolset more analogous to the symbol-aware search and usage information available in human programmer-facing Integrated Development Environments (IDEs). We describe a working implementation with bindings to three established programming languages, and several example programs. 

---
# Taming foundation model with invariance-oriented pre-training for broad-spectrum EEG analysis across signal-level, brain-state, and brain-health tasks 

**Authors**: Yulong Dou, Han Wu, Guo Chen, Fangmao Ju, Zhiming Cui, Dinggang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2608.24597)  

**Abstract**: Electroencephalography (EEG) is a widely used window into human brain function, but most EEG models remain tied to a one-dataset-one-model supervised paradigm. Recent EEG foundation models offer a route toward reusable representations, but most remain reconstruction-centered, assuming that EEG content predictable from local context is necessarily transferable neural information. Here we present INCEPT, an invariance-oriented EEG foundation model trained on over 11,000 hours of unlabelled clinical EEG. Rather than prioritizing signal recovery alone, INCEPT learns representation-level stability across correlated EEG observations, separating stable neural structure and essential subject-sensitive information from the nuisance variability that dominates scalp recordings while preserving subject-, state- and condition-discriminative information. We evaluate INCEPT on a broad-spectrum benchmark of ten datasets spanning three levels of post-acquisition EEG analysis: signal-level assessment, brain-state decoding, and brain-health evaluation. INCEPT ranks first among recent EEG foundation models on 26 of 30 linear-probing metrics and 24 of 30 fine-tuning metrics, and also surpasses strong task-specific specialist encoders across diverse downstream settings. Objective ablations and representation analyses further show that invariance-oriented pre-training improves transfer and organizes subject-sensitive neural representations beyond reconstruction alone. These results establish invariance learning as a promising principle for building reusable EEG foundation models. 

---
# $\texttt{findr}$: Transparent and Fair Credit Risk Decisions through Semi-Structured Regressions 

**Authors**: Victor Medina-Olivares, Stefan Lessmann, Jonathan Crook  

**Link**: [PDF](https://arxiv.org/pdf/2608.24582)  

**Abstract**: Credit risk models increasingly need to combine predictive accuracy with transparent explanations and auditable fairness constraints. Logistic regression remains attractive because its coefficients are easy to interpret, but it can miss nonlinear structure. Flexible models can improve prediction, but their explanations are often post-hoc and may not describe the decision rule itself. We introduce $\texttt{findr}$, short for flexible, interpretable deep regression, a semi-structured framework for binary credit risk modelling that decomposes the logit into an interpretable structured component and an orthogonal neural residual. The orthogonalisation separates coefficient-based effects from residual nonlinear variation, while an in-processing Wasserstein penalty mitigates group disparities by comparing score distributions during training. The framework also includes diagnostics that measure the structured component's contribution to logit variation, decision agreement, and local directional consistency. We evaluate $\texttt{findr}$ in a simulation study and on eight public credit datasets using score-level accuracy-fairness frontiers. The results show that $\texttt{findr}$ behaves close to logistic regression when the signal is approximately linear, while recovering much of the predictive gain of neural models when nonlinear structure is relevant. The diagnostics identify when coefficient-based explanations remain close to the full fitted model and when residual variation must also be examined. These findings support semi-structured modelling as a practical way to make performance, fairness, and interpretability trade-offs explicit in credit risk decisions. 

---
# Across the Loss Landscape with Progressive Growth 

**Authors**: Paul Caillon, Christophe Cerisara, Alexandre Allauzen  

**Link**: [PDF](https://arxiv.org/pdf/2608.24568)  

**Abstract**: Deep neural networks generalize well despite their highly nonconvex, overparameterized loss landscapes, a phenomenon often associated with the geometry of the minima found by stochastic optimization. We study how incremental grow-and-optimize strategies bias training toward flatter regions by viewing growth as progressive constraint relaxation. Starting from a low-dimensional submodel, we iteratively expand the trainable parameters by unlocking nested random subspaces while freezing the orthogonal complement at the network initialization, re-optimizing after each expansion until the full architecture is reached. Under standard local regularity conditions around non-degenerate minima, we prove that local sublevel sets are well approximated by ellipsoids and that basin accessibility under frozen constraints can be characterized by an explicit effective curvature in the frozen directions. This leads to an explanation of the bias: progressive growth increases the relative weight of wide basins and suppresses sharp ones through a volume effect induced by the frozen constraints. We empirically validate these predictions in controlled toy landscapes and in a realistic ResNet/CIFAR-100 setting and confirm that although progressive subspace growth reliably produces flatter solutions, curvature reductions do not universally translate into improved test performance, highlighting subtleties in the flatness-generalization connection. The code is available at this https URL. 

---
# COCI: Conference Organisers and Content Identifier 

**Authors**: Angelo Salatino, Francesco Osborne, Alexis Vizcaino, Aliaksandr Birukou, Enrico Motta  

**Link**: [PDF](https://arxiv.org/pdf/2608.24559)  

**Abstract**: Despite the critical role of grey literature in scholarly communication, artefacts such as Calls for Papers (CfPs) remain largely isolated from modern Scholarly Knowledge Graphs. The unstructured and highly heterogeneous nature of these documents has traditionally hindered their large-scale processing. In this demo paper, we present the Conference Organisers and Content Identifier (COCI), an AI-based framework designed to extract fine-grained, structured metadata from raw CfP texts. COCI employs a multi-stage pipeline that combines Large Language Models (LLMs) with semantic mapping techniques to integrate extracted entities with established knowledge bases, including OpenAlex, DBLP, TIB ConfIDent, and the AIDA Dashboard. By disambiguating authors and semantically aligning topics and conference series, COCI bridges the gap between informal scholarly dissemination and structured Semantic Web resources, laying the foundation for systematic analysis of non-publisher-based academic events. 

---
# StrokeGuard: A Multi-Agent Guided System for Prehospital Stroke Assessment 

**Authors**: Wentao Yang, Zhenye Xu, Ruoyi Li, Musen Zhang, Yao Guo  

**Link**: [PDF](https://arxiv.org/pdf/2608.24555)  

**Abstract**: Prehospital stroke assessment aims to accurately identify stroke symptoms and make rapid decisions through standardized procedures within an extremely narrow time window, thereby saving valuable time for subsequent treatment. In clinical practice, FAST-based scales are widely used for prehospital stroke assessment by issuing instructions that guide subjects to perform specific actions to screen facial, arm, and speech functions. However, in home and community settings, non-clinical users often encounter challenges such as inaccurate descriptions, incomplete symptom observation, and difficult operational procedures, which may lead to inaccurate or biased assessment results. To address these challenges, this paper presents StrokeGuard: a multi-agent guided system designed for prehospital stroke assessment that makes mobile FAST screening more standardized and executable. Specifically, to overcome the limitations of traditional single-agent systems in terms of procedural fault tolerance and user guidance capability, StrokeGuard adopts a dual-channel agent mechanism that separates formal assessment (i.e., facial palsy, arm weakness, speech impairment) from procedural support (e.g., step prompts, error correction, and real-time feedback). It guides the assessment process through multi-agent collaboration, dual-channel interaction, state-machine control, and stage-local fallback recovery mechanisms. Stage-specific scoring is delegated to constrained pretrained video assessment modules, while evidence source records are integrated with structured report generation. The user evaluation uses MATES-9, an exploratory scale for measuring user experience in multistep AI-guided tasks. In a simulated prehospital scenario, StrokeGuard improves the MATES-9 total score over a paper FAST-style form by 10.83 points, corresponding to a 23.8% relative increase. 

---
# FraudBench: Protocol-Sensitive Benchmarking of Adversarial Robustness for Financial Risk Assessment 

**Authors**: Xitong Zeng, Zhaoge Bi, Yitian Yang, Huaming Chen, Quan Z. Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2608.24551)  

**Abstract**: Machine learning models are widely used in financial fraud and credit-risk detection, yet their adversarial robustness remains difficult to evaluate because financial tabular data involve domain-specific constraints, severe class imbalance, and asymmetric attacker capability. We argue that, in this setting, robustness is not only an attribute of the model, but also an attribute of the evaluation protocol. Different ways of enforcing constraints and capability can lead to substantially different robustness conclusions. This paper presents FraudBench, a protocol-sensitive benchmark for adversarial robustness evaluation in financial fraud and credit-risk detection. Rather than treating domain constraints as post-hoc validity checks, FraudBench evaluates the same dataset--model--attack--defence setting under three matched protocols: unconstrained attacks, post-hoc feasibility filtering, and deployment-aware constraint-integrated attacks. FraudBench covers four public financial datasets, and evaluates neural, tree-based, and ensemble models using three attack settings. Our results show that robustness conclusions are highly protocol-sensitive. On Lending Club Loan Data under the white-box setting, post-hoc filtering leaves only 3.7 feasible-flipped examples on average, whereas in-attack projection with attacker mutability masking produces 2,832.3 feasible-flipped examples under the same perturbation budget. The results on IEEE-CIS further show that feasibility and attacker capability are separate axes, while black-box evaluation shows that protocol choice can alter model-family rankings. These findings suggest that fraud robustness evaluation should report predictive degradation and attack feasibility jointly, and should incorporate domain constraints into attack generation rather than treating them as post-processing checks. 

---
# LumiXAI: A Modular Full-Stack Framework for Feature Attribution 

**Authors**: Alfio Ferrara, Lorenzo Gatta, Sergio Picascia, Elisabetta Rocchetti  

**Link**: [PDF](https://arxiv.org/pdf/2608.24524)  

**Abstract**: Feature attribution is a central tool of model interpretability, yet the software through which it is applied remains fragmented: individual tools specialize along narrow axes, such as a single modality, a code API or a GUI, or a fixed rather than extensible method set, and rarely combine these strengths. Moreover, many explainability tools are designed primarily for domain experts, requiring programming skills or familiarity with attribution methods that can make them difficult for non-expert users to access. In this article, we present LumiXAI, a modular full-stack framework that consolidates attribution analysis into a single system. It couples classification and generative attribution with an interactive GUI supporting bidirectional exploration, a plug-in architecture for registering new models and methods, and three access tiers serving non-programmers, developers, and extenders from one backend. Its contribution is a system that operationalises established attribution methods under one interface, one interaction model, and one persistence layer, with containerised services and persistent results making analyses reproducible across machines. 

---
# Scalable and Versatile Identification for Hierarchical Structural Causal Models: A New Look at Project STAR 

**Authors**: Janis Aiad, Aghiles Drali, Aymen El Ouadrhiri, Anass Ettahiri, Yasser Oufqir, Simon Patry, David Cortes, Marianne Clausel, Emilie Devijver  

**Link**: [PDF](https://arxiv.org/pdf/2608.24500)  

**Abstract**: The STAR (Student-Teacher Achievement Ratio) experiment (1985, Tennessee, USA) is a landmark hierarchical dataset designed to assess the impact of class size on student outcomes, with observations nested within classes. To encode class-level interventions in such hierarchical settings, we develop a complete, scalable, open-source pipeline for Hierarchical Structural Causal Models (HSCM) that bridges symbolic identification and practical estimation. Our approach integrates graph transformations, pyAgrum's do-calculus for automatic identification of causal effects, adaptation of symbolic expression into closed-form HSCM formulas, and numerical estimation from fitted local probability models. A key innovation is our adapted Abstract Syntax Tree (AST), which decomposes pyAgrum's identified formulas into independent density, expectation, and marginalization tasks, enabling parallel and scalable computation. We validate the pipeline on canonical HSCM motifs and benchmark scenarios with known ground truth, then apply it to STAR kindergarten mathematics outcomes. The results show that flat baselines (ignoring hierarchy) recover associations but fail to encode class-level interventions, and that symbolic identification alone is not enough for practical Hierarchical Structural Causal inference; scalable estimation and numerical stability checks are central parts of the scientific object. 

---
# When Do Supervised UQ Ensembles Improve LLM Hallucination Detection? A Robustness Study 

**Authors**: Mohit Singh Chauhan, Vipin Gyanchandani, Dylan Bouchard  

**Link**: [PDF](https://arxiv.org/pdf/2608.24492)  

**Abstract**: Uncertainty quantification (UQ) methods are widely used for hallucination detection in large language models (LLMs) in closed-book settings where ground-truth evidence is unavailable at inference time. Prior work has proposed combining UQ signals via learned ensembles, but empirical investigations into the robustness of these ensembles are limited. We study a supervised ensembling framework that trains a classifier over heterogeneous UQ-based scorer outputs on a small, domain-specific dataset of labeled LLM responses, then applies it to out-of-sample hallucination classification without retrieval, tools, or reference documents. Across four LLMs, nine datasets, and three generation regimes (short-form QA, long-form generation, and code generation), we provide a systematic robustness analysis along three axes: sample efficiency, in-domain dataset transfer, and generation regime dependence. We find that supervised ensembles outperform the best individual scorer in 30 of 32 settings, with gains realized from as few as 100 labeled instances. Ensembles retain most of their advantage in cases of in-domain transfer under distribution shift, outperforming the best non-ensemble scorer in 23 of 28 transfer settings. Sampling-based black-box ensembles are nearly as effective as full ensembles, while single-generation white-box ensembles offer limited benefit. 

---
# Beyond Static Interpretability: Anticipating Post-SFT Mechanisms from Pre-SFT Parameters for Better Tuning 

**Authors**: Hang Chen, Jiaying Zhu, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.24482)  

**Abstract**: Mechanistic Localization bridges mechanistic interpretability and post-training optimization by isolating critical parameters via interpretative approaches and then guiding parameter-efficient Supervised Fine-Tuning (SFT) in a ``locating-then-tuning'' paradigm. However, due to the retrospective nature of mechanistic interpretability, directly interpreting pre-SFT models introduces misleading conclusions. Specifically for novel tasks, initially identified neurons differ drastically from those governing the final model, introducing biases that actively disrupt SFT. To address this, we propose a forward-looking localization framework that accurately estimates the post-SFT interpretability state using only pre-SFT parameters and the target dataset. Theoretically, we model SFT as a continuous parameter evolution, leveraging Taylor expansion to rigorously bridge the post-tuning mechanistic objective with the pre-SFT model's dynamic gradients. Practically, we design dual-granularity (neuron- and component-level) localization pipelines. Extensive experiments demonstrate that our approach not only provides superior SFT guidance but also exhibits robust performance and temporal scalability across increasing model sizes. This work transcends the fundamental limitation of traditional interpretability-its inability to identify task-critical mechanisms before they are trained-pioneering a predictive frontier that unites mechanistic interpretability with targeted optimization. 

---
# Evaluating Deep Multivariate Imputation Models on Wearable Device Data 

**Authors**: Skye Goodman, Roussel Desmond Nzoyem, Leandro Junges, Peter Kissack, Yasser Qureshi, Amberly Brigden, Jeff Clark, Nawid Keshtmand  

**Link**: [PDF](https://arxiv.org/pdf/2608.24436)  

**Abstract**: Wearable device data enables continuous health monitoring, but suffers from structured missingness: features sharing a physical sensor drop out together. Deep imputation methods such as BRITS and SAITS have seen limited evaluation on multimodal physiological data under realistic missingness, and existing benchmarks use random-point holdout protocols that incorrectly assume missingness is independent across features and time. Using data from a person with epilepsy recorded on a Garmin smartwatch, we develop an evaluation protocol that mines contiguous missing-run templates from training data, stratifies them by per-feature gap-length quantiles, and injects them as block masks with preserved co-missingness structure. A matched training protocol exposing models to the same missingness distribution reduces BRITS's severe-gap MAE by 43%, demonstrating the potential benefit of the proposed evaluation and training protocol within this single-participant dataset. We further extend BRITS with time-of-day encoding and a circadian harmonic channel. No single model dominates: linear interpolation is optimal for slow-moving features over short gaps; extended BRITS achieves lower MAE on dynamic cardiac features in moderate and severe gaps; and SAITS better preserves the ground-truth distribution by Jensen-Shannon distance despite higher MAE. Ultimately, model rankings strongly depend on evaluation designs. By exposing how traditional evaluation methods obscure true model capabilities, our transferable protocol establishes critical steps towards developing better imputation strategies for future multi-sensor wearable datasets. 

---
# Multilevel Fair Allocation under Additive Preferences 

**Authors**: Maxime Lucet, Nawal Benabbou, Aurélie Beynier, Nicolas Maudet  

**Link**: [PDF](https://arxiv.org/pdf/2608.24400)  

**Abstract**: We study multilevel fair resource allocation with tree-structured hierarchical relations among agents. At each level, the problem can be viewed locally as allocating an agent's bundle to its children, the overall allocation being a trace of this process iterated down to the leaves. Assuming that internal nodes' utilities are the utilitarian welfare of their children, and the leaves have classical additive utilities over items, we first propose multilevel adaptations of usual envy-based fairness notions (e.g., WEF1). We present three adaptations and show that the choice among them is not neutral. We prove that, under identical preferences, the three adapted envy-based notions coincide, and that the Multilevel extension of Weighted Round Robin (Chakraborty et al., 2021) (MWRR) guarantees them. We then show that under general preferences, MWRR may guarantee some notions while failing others. Finally, through experiments, we show that MWRR may still perform well even for adaptations it does not formally guarantee. 

---
# Equivariant Covariance Tensors: Guaranteed SPD Uncertainty for Tensor-Valued Geometric Learning 

**Authors**: Ruihan Liu, Yu Ji, Jianbo Yu, Shifu Yan, Qingchao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2608.24386)  

**Abstract**: Tensor-valued prediction is fundamental to geometric deep learning, yet uncertainty quantification (UQ) for such outputs remains an open challenge. While E(3)-equivariant neural networks excel at point estimates, they lack rigorous confidence measures. We focus on symmetric rank-2 tensor prediction, where the target has six Kelvin--Mandel coordinates and full uncertainty is represented by a $6\times6$ covariance matrix. We introduce a framework for E(3)-equivariant UQ, modeling the full predictive distribution where both mean and covariance preserve rotational symmetry. Our approach decomposes the covariance into irreducible representations $\mathrm{Sym}^2(\rho_c) \cong 2\times(l=0) \oplus 2\times(l=2) \oplus 1\times(l=4)$. By mapping from the flat Lie algebra $\mathfrak{sym}(6)$ to the curved SPD manifold via matrix exponentiation, we strictly ensure positive-definite covariances while maintaining exact equivariance. Furthermore, we formulate a Log-Euclidean Equivariant Scoring Objective (LE-ESO)---a robust surrogate loss based on the Multivariate Laplace distribution---providing robustness to heavy-tailed errors and stable optimization. Validation on ModelNet40 inertia tensors and Materials Project dielectric tensors demonstrates that our method achieves competitive performance and provides physically consistent, symmetry-preserving uncertainty estimates with useful risk and OOD sensitivity. 

---
# Markerless Pose Estimation for Resistance Training Technique Assessment 

**Authors**: Joseph Turner, Jeff Clark, Nawid Keshtmand  

**Link**: [PDF](https://arxiv.org/pdf/2608.24384)  

**Abstract**: Resistance training can be a high risk activity, and safe form is essential to avoiding injury. Laboratory-based movement analysis provides quantitive technique assessment, yet is not easily accessible. Markerless pose estimation infers body landmarks from images or video without physical markers and could offer a feasible alternative for technique assessment. We present a pose estimation framework to evaluate resistance-training technique from ordinary video footage. Using BlazePose, anatomical landmarks were extracted from squat, bench press, and deadlift videos and converted into joint-angle trajectories, with the squat serving as the primary case study. Trajectories were assessed against a defined reference repetition using root mean square error (RMSE). Results show that the framework recovers meaningful kinematic patterns for the squat and deadlift, enabling quantitative comparison between repetitions and identification of technique variability within a set. Performance depended strongly on camera orientation and visual occlusion, with non-sagittal views distorting 2D joint-angle estimates. The findings demonstrate that markerless pose estimation can support accessible biomechanical assessment outside laboratory environments. 

---
# Not All Tokens Are Equal: Region-Aware Consistency Repair of Backdoors in MLLMs 

**Authors**: Jiali Wei, Ming Fan, Mingkun Zhang, Haoyu Wang, Jun Sun, Guoheng Sun, Xiaoning Ren, Haijun Wang, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.24354)  

**Abstract**: MLLMs are increasingly deployed in user-facing applications, yet they inherit backdoor risks from the pipelines used to construct them: triggers may reside in images, texts, or both. Existing model-level backdoor removal methods, largely designed for conventional classifiers, show limited effectiveness on MLLMs, while MLLM-specific defenses mainly operate at inference time, filtering suspicious inputs without removing the backdoor embedded in the model. To address this gap and eliminate latent backdoors from MLLMs at their source, we present RACER, a model-level repair framework motivated by a key observation: backdoors induce abnormal layer-to-layer evolution in internal representations, which we term the layer-wise inconsistency anomaly. Importantly, this anomaly is modality-dependent, concentrating primarily in the token region encoding the trigger features that the backdoor model actually relies on. RACER therefore decomposes the fused representation into visual and textual token regions, normalizes their layer-wise inconsistency separately, and recomposes them using modality-aware weights over a deep-layer window, yielding a region-aware inconsistency objective that better captures localized backdoor-induced anomalies. Through a min-max optimization, this objective drives worst-case perturbation synthesis and adversarial fine-tuning against the resulting perturbation to repair the model, suppressing the deep representational directional shifts on which backdoor behaviors rely. RACER requires only 100 clean samples and no knowledge of the trigger, attack objective, or even whether the input model contains a backdoor. Evaluations on three open-source MLLMs across 36 backdoor settings spanning image, text, and multimodal triggers show that RACER reduces the average ASR to 1.1%, reaching 0% in 32 settings, while preserving clean-task utility on both backdoor and clean models. 

---
# FARCA: Fact-Aligned Reliability-Aware Credit Assignment for Reinforcement Learning with Factual Supervision 

**Authors**: Qiming Xie, Wenjie Zheng, Xiangqing Shen, Rui Xia  

**Link**: [PDF](https://arxiv.org/pdf/2608.24350)  

**Abstract**: To reduce the hallucination risk caused by outcome-driven rewards in large language models trained through reinforcement learning with verifiable rewards, existing mitigation approaches introduce process-level factual supervision. However, due to coarse-grained aggregation of factual signals and the lack of reliability assessment for these signals, they create a mismatch between fact verification and policy updates. We term this noisy factual credit assignment and decompose it into two aspects: credit localization ambiguity and credit reliability ambiguity. To address these issues, we propose FARCA (Fact-Aligned Reliability-Aware Credit Assignment), a policy optimization framework that transforms factual supervision into localized, reliability-weighted token-level training signals. FARCA achieves fine-grained credit localization by aligning the granularity of fact verification with that of policy updates. It further introduces counterfactual evidence attribution, which uses the dependence of a factual judgment on key evidence as an empirical proxy for verification reliability to compute reliability weights. These weights modulate factual rewards and local policy advantages, reducing the influence of potentially unreliable signals on policy optimization. Experiments across different models and multiple factual reasoning benchmarks show that FARCA significantly improves model factuality while preserving general reasoning capabilities. 

---
# Metadata-Aware Adaptation of a Generative Foundation Model for Conditional CMR Synthesis 

**Authors**: Marc Rodríguez, Grzegorz Skorupko, Nay Aung, Steffen E Petersen, Karim Lekadir, Polyxeni Gkontra  

**Link**: [PDF](https://arxiv.org/pdf/2608.24342)  

**Abstract**: Synthetic image generation is a promising strategy to address data scarcity and the underrepresentation of clinically important phenotypes in medical imaging, yet generating images that faithfully reflect meaningful patient characteristics remains challenging. In this work, we investigate metadata-conditioned cardiac magnetic resonance (CMR) synthesis using a pretrained latent diffusion model, encoding structured clinical metadata and slice position as textual prompts to guide CMR generation. To improve metadata adherence and address the imbalance of clinical attributes, we integrate three strategies: Metadata-Free Classifier-Free Guidance (CFG), Contrastive Batching, and Inverse-Frequency Sampling. The framework was fine-tuned and evaluated on 59,058 short-axis CMR from the UK Biobank using paired image similarity, distributional fidelity, and subgroup-level analyses. The combined approach achieved a Fréchet Inception Distance (FID) of 37.47, improving by 57.04\% over the same model fine-tuned without these strategies and by 28.68\% over a previous text-conditioned CMR diffusion baseline requiring cardiac geometry as additional input, while relying solely on patient metadata. This distributional gain, driven mainly by Metadata-Free CFG, came with a modest reduction in paired similarity, suggesting that the model prioritizes population-level realism over exact image reproduction. Subgroup analyses demonstrated improved alignment across demographic and acquisition-related metadata, with disease-specific conditioning being the most challenging task. These findings demonstrate the potential of generative foundation models for clinically meaningful CMR synthesis while highlighting the need for more effective metadata-aware conditioning strategies. Our code is available at this https URL. 

---
# Mind the Student: Behavioral and Contextual Cues for Automated Engagement Prediction in Online Learning 

**Authors**: Alperen Kantarci, Visvanathan Ramesh, Gemma Roig  

**Link**: [PDF](https://arxiv.org/pdf/2608.24340)  

**Abstract**: The prediction of student engagement from the online tutoring videos is difficult because engagement is a multidimensional construct comprising distinct behavioral, emotional, and cognitive states. A reliable prediction requires bringing together different types of behavioral signals as well as expressive cues. Through our analysis of the CASED dataset, it is clear that engagement prediction gets even harder due to the high inter-person variability as well as the subjectivity of the engagement annotation. To tackle these challenges, we develop a multimodal framework that integrates the implicit spatiotemporal features extracted from pretrained video, audio, and image encoders along with structured behavioral modalities like head pose, gaze, facial action units, emotion, and wavelet-based audio features. We integrate these modalities via a Perceiver IO latent bottleneck. Moreover, student and instructor personalities are modeled as variational posteriors over learnable embeddings to enable partial pooling across participants. We employ evidential regression and spectral-normalized Gaussian process classification heads for uncertainty-aware prediction to further improve robustness and calibration. Benchmark on the CASED challenge test set shows that all participating methods converge near random-chance performance, revealing the difficulty of the dataset. In this highly ambiguous regime, our framework achieves competitive performance while uniquely offering well-calibrated uncertainty metrics, demonstrating that reliable risk-quantification is an essential prerequisite for deploying engagement models in real-world educational tools. 

---
# SENSESHIFT: Continuous Sentiment-Controlled Text Generation via Encoder-based Mask Infilling 

**Authors**: Shahed Masoudian, Markus Frohmann, Emmanouil Karystinaios, Navid Rekabsaz, Markus Schedl  

**Link**: [PDF](https://arxiv.org/pdf/2608.24304)  

**Abstract**: Recent controllable text generation (CTG) for sentiment control has largely focused on decoder-based large language models, making causal attention the dominant paradigm. While effective for fluent generation, these models still struggle to satisfy complex constraints and follow fine-grained sentiment signals specified by users. Existing sentiment-aware CTG methods typically simplify the problem by treating sentiment either as a coarse categorical label (e.g., positive or negative) or as a single fine-grained control signal applied to an entire document. Consequently, more challenging settings such as sentence-level sentiment control within long-form text remain underexplored. To address these limitations, we introduce SenseShift , an encoder-based framework for fine-grained sentence-level CTG. Unlike standard decoder architectures, SenseShift leverages bidirectional attention, quantized sentiment signals, and iterative mask infilling to generate local sentences conditioned on target sentiment intensity. Empirical evaluations on story and review generation demonstrate that SenseShift achieves stronger sentiment controllability while maintaining text quality and robustness to out-of-domain generation compared to larger decoder-based baselines. 

---
# Contrastive Branch Policy Optimization 

**Authors**: Ying Wang, Changlin Qiu, Bang Lin, Linbo Jin, Wen Jiang, Zhe Sun, Jingli Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.24300)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) enables language models to learn multi-turn interaction with external tools, yet its sparse outcome rewards provide no signal for identifying which intermediate decisions are responsible for success. Branch sampling induces local comparisons among alternative continuations, but existing methods tend to conflate two distinct problems: allocating a fixed rollout budget and translating branch outcomes into token-level credit. We introduce Contrastive Branch Policy Optimization (CBPO), which disentangles these two problems and assigns a dedicated mechanism to each. Generation entropy screens candidate branch positions across the entire response, while path-level and node-level decay distribute a fixed budget across trajectories and positions to prevent exploration from collapsing onto a few paths or adjacent tokens. A parent trajectory together with the branches that share an identical token prefix forms an exact-prefix group, and the reward variation within this controlled group defines the Contrastive Branch Value (CBV), an outcome-based estimate of local decision sensitivity that rescales continuation advantages without altering their sign. When multiple nodes are selected along the same trajectory, CBPO partitions it into non-overlapping credit segments, thereby avoiding duplicated gradients on shared tokens. Requiring only outcome rewards and no process-level annotation, CBPO provides a practical solution for fine-grained credit assignment in tool-integrated agent training. Extensive experiments on ten benchmarks, including five for mathematical reasoning and five for knowledge-intensive search, show that CBPO consistently outperforms state-of-the-art policy-optimization and branch-based methods, attaining the highest macro-average accuracy in both domains and across two model scales. 

---
# 'Ghaib in Translation' aka Unseen Harm: Measuring Cross-Script Safety Inconsistency with 'Missed-in-Urdu' Scores in LLM Hate Speech Detection 

**Authors**: Fawzia Zehra, Kara-Isitt, Sonal Khosla, Stephen Swift  

**Link**: [PDF](https://arxiv.org/pdf/2608.24191)  

**Abstract**: Urdu, the world's tenth most spoken language with 246 million speakers, remains almost entirely absent from mainstream LLM safety evaluation and nine years of WOAH proceedings. To investigate whether this absence has measurable consequences for content moderation reliability, five large language models, GPT-4o, Claude Sonnet 4.5, Gemini 2.5 Flash, Qwen-2.5, and Llama-3.1, were tested across six datasets spanning Nastaliq Urdu, Roman Urdu, English, and code-switched Urdu-English. Across the five Urdu-script datasets, label instability between original-script and English-translation classification ranged from 15.9% (Gemini 2.5 Flash) to 31.6% (Qwen-2.5), with a 'Missed-in-Urdu' rate, content flagged as harmful in English translation but passed as normal in the original script, ranging from 2.4% to 9.9% (median 4.3%). A complete enumeration of all 205 papers across nine ALW/WOAH editions via the ACL Anthology API confirms zero dedicated Urdu papers across the entire period. Results indicate that current LLMs provide uneven safety assurance across Urdu's script varieties, with smaller open-weight models showing substantially higher instability and missed-harm rates than frontier closed models. 

---
# Tlow: Flow-based Item Tokenizer for Recommendation 

**Authors**: Nian Li, Chonggang Song, Jingtao Ding, Lingling Yi, Yong Li, Qingmin Liao  

**Link**: [PDF](https://arxiv.org/pdf/2608.24176)  

**Abstract**: Item tokenizer encodes semantic embeddings into token IDs to replace the randomly assigned item IDs used in traditional recommendation models, fundamentally addressing the problems of excessive parameters and cold starts. However, the most common tokenizer, RQ-VAE, suffers from low decoding efficiency due to the inherent dependencies among its codebooks. Meanwhile, efficient independent tokenizers such as optimized product quantization (OPQ) still struggle with dimensional correlations and distribution complexity of semantic embeddings. In this work, we propose a f\underline{low}-based item \underline{T}okenizer (Tlow) to transform raw semantic embeddings into a latent space where embeddings conform to a unified standard normal distribution, achieving dual advantages of dimensional independence and distributional simplicity. Independent tokenization performed on these latent embeddings yields semantically clear token IDs. Additionally, we introduce a novel codebook guidance to align the codebook space with the token embedding space, further aiding the learning of more semantically distinct token embeddings. Offline experiments on four public datasets demonstrate that Tlow's tokenization and codebook guidance significantly improve recommendation performance. The improvement on cross-domain and multi-modal recommendations also proves the effectiveness of item tokenization in a simplified embedding space. Online experiments for a multi-modal retrieval task on China's largest social media platform WeChat validate Tlow's powerful distribution transformation capability. The retrieval model based on token IDs improves user CTR by 10.32\% globally and by 11.64\% for new items. Our codes are available at this https URL. 

---
# Preference Optimization for Non-Verbal Vocalization Synthesis 

**Authors**: Haoyang Li, Chenglin Xu, Junchuan Zhao, Yuang Cao, Liumeng Xue, Yiwen Guo, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2608.24163)  

**Abstract**: Non-verbal vocalizations (NVs), such as laughter, coughs, and sighs, are essential for expressive TTS, but the effectiveness of preference optimization for NV generation remains poorly understood. We systematically study preference optimization for NV-capable TTS, focusing on preference signals, preference-pair construction, and DPO-based optimization objectives. We formulate an NV-aware character error rate (NV-CER) by treating NV tags as distinct output symbols and computing a weighted pinyin-based CER over both verbal and non-verbal content, enabling controllable optimization of NV realization without modifying the underlying optimization algorithm. Experiments on Emilia-NV and the augmented NV-Bench covering 18 NV types reveal how different design choices affect NV realization and lexical fidelity, and establish an effective setup using standard DPO. Objective, LLM-based, and human evaluations provide converging evidence for our findings, offering practical insights into NV-aware post-training for expressive TTS. 

---
# LLM-Guided Contextual Action Evaluation for Operational Decisions in Industrial Processes 

**Authors**: Youcheng Zong, Runda Jia, Dakuo He  

**Link**: [PDF](https://arxiv.org/pdf/2608.24156)  

**Abstract**: Industrial actor--critic methods usually represent continuous actions as anonymous numerical coordinates. They must therefore learn from limited interactions which process variables each action affects, in which direction, and after what delay. Fixed industrial documents already describe part of these relations, but their open-text statements neither represent the current operating condition nor directly fit a numerical policy. This article presents LLM-Guided Contextual Action Evaluation for Operational Decisions in Industrial Processes (LCAE), which uses a large language model before training to normalize fixed documents into a frozen action--observation--direction--delay relation basis. Recent numerical action--response history then modulates the current strength of each relation, while the evaluated action forms a state-conditioned nonlinear action-effect field in the same basis. The critic evaluates actions through this field, and the actor uses the same relation gains to generate actions, making document semantics part of maximum-entropy policy learning. Neither the LLM nor the embedding model runs online during training or deployment; the deployed policy uses only frozen semantic artifacts and visible numerical history. The method states a falsifiable hypothesis: when documented relations are correct and recent history reflects their contextual strength, this action representation should provide a more useful decision bias than raw action coordinates. 

---
# Rethinking Pre-Training and Augmentation for Zero-Shot Cross-City Object Detection 

**Authors**: Long Hoang Pham, Quoc Pham-Nam Ho, Huy-Hung Nguyen, Duong Nguyen-Ngoc Tran, Ngoc Doan-Minh Huynh, Cu Quoc Le, Hoang-Khang Nguyen, Hyung-Min Jeon, Chi Dai Tran, Son Hong Phan, Duong Khac Vu, Trinh Le Ba Khanh, Jae Wook Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2608.24154)  

**Abstract**: Real-world deployment of traffic surveillance systems is bottlenecked by geographic domain shift, in which models trained in one city underperform when applied to an unseen target city. Conventional domain adaptation relies on hyperparameter-sensitive architectures or direct profiling of target data. Both are fundamentally precluded in privacy-conscious ecosystems that require completely blind training and evaluation loops. In this setting, we explore the effects of pre-training and augmentation in addressing the domain shift problem. Specifically, we propose a new modular training pipeline for object detection structured around two core orthogonal pillars: (1) a multi-dataset pre-training strategy featuring a class-agnostic objectness distillation to decouple structural vehicle geometry from semantic taxonomies, and (2) a domain-resilient augmentation stream featuring a novel Grayworld transformation that forces global attention heads to strip volatile chromatic shortcuts in favor of robust shape priors. When evaluated with the real-time transformer-based detector RF-DETR, our framework bridges cross-city distribution gaps while using limited GPU memory (16GB). Our optimized variants, RF-DETR-HR and RF-DETR-Grayworld, deliver a substantial empirical gain of +24.29 over the baseline, achieving 1st place (47.53 mAP) on the AI City Challenge Track 6 leaderboard. Code and data are available at: \href{this https URL}{SKKUAutoLab/aic26\_cross\_city}. 

---
# PlaceSeek: Human-Centered Geospatial Retrieval of Urban Outdoor Places via Semantic Grounding and Affective Alignment 

**Authors**: Ziqi Cui, Shangyu Lou  

**Link**: [PDF](https://arxiv.org/pdf/2608.24133)  

**Abstract**: People search for urban outdoor places not only by category or function, but also by what activities a place can support and how it is perceived. Existing geospatial retrieval remains largely POIcentric and metadata-driven, making it difficult to satisfy openended, affective, or activity-oriented needs. We present PlaceSeek, a human-centered outdoor place retrieval framework that maps natural-language queries to geolocated street-view imagery. PlaceSeek introduces an intent-aware retrieval mechanism that decomposes user queries into functional and affective sub-intents. A Semantic Grounding Module verifies whether candidate street-view results contain the physical evidence needed to support the intended activity, while an Affective Alignment Module re-ranks physically valid candidates using a LoRA-adapted vision-language model trained on human urban perception judgments. We evaluate PlaceSeek on 31,956 street-view locations in Milan across 10 naturallanguage queries annotated by five human evaluators. PlaceSeek achieves 88.0% Precision@5, a mean match score of 3.39/4.0, and 0.920 nDCG@5, outperforming CLIP, fine-tuned CLIP, SigLIP, and a VQA-based baseline. Ablation results show that physical grounding is essential for retrieval validity, while affective alignment improves ranking quality among physically valid candidates. These findings highlight that complex urban spatial queries require modeling both verifiable visual evidence and human perceptual preferences. PlaceSeek provides a potential framework for human-centered nextgeneration geospatial retrieval systems. 

---
# From Gradient-Boosted Trees to Deep Recommenders: Practical Lessons from Migrating a Production Customer Support Recommender 

**Authors**: Sonia Sharma, Jeyendran Balakrishnan, Shreya Rajpal, Swapnil Parekh, Nagaraj Janardhana, Andrew Mattarella-Micke  

**Link**: [PDF](https://arxiv.org/pdf/2608.24132)  

**Abstract**: Product catalogs in fast-moving service businesses are shifting from static, independently priced SKUs toward dynamically bundled, discount-coupled offerings--a shift that strains the tree-based classifiers traditionally preferred for sparse and highly imbalanced data. These classifiers assume a fixed, slowly changing label space and struggle to incorporate multimodal signals such as tabular data and transcripts. We present the migration of a live, production conversational recommendation system from a gradient-boosted multiclass model to a pairwise-binary deep recommender. Because this system is critical to ecosystem growth initiatives and downstream features like dynamic pitching--surfacing the most relevant pitch text to a support agent in real time during a live customer conversation--maintaining live recommendation quality was a non-negotiable constraint. We detail the techniques that made this migration successful--reformulating recommendation as pairwise binary prediction to learn jointly from user and item features, and enhancing learned representations via negative sampling and noise injection. To efficiently incorporate long, live conversation context, we apply attention pooling over transcript chunks and benchmark it against TF-IDF and sentence-embedding baselines. Finally, we explore multiple architectures (including two-tower models, DeepFM, and their variants) and loss functions such as contrastive loss. Evaluating against a CatBoost baseline across all conversational stages, we demonstrate that our approach achieves parity at conversation beginning and outperforms at later conversational stages. 

---
# Syn2RealTrack: Bridging the Gap Between Synthetic and Real-World Datasets for Online Multi-View Multi-Target Tracking 

**Authors**: Duong Nguyen-Ngoc Tran, Ngoc Doan-Minh Huynh, Cu Quoc Le, Hoang-Khang Nguyen, Long Hoang Pham, Huy-Hung Nguyen, Quoc Pham-Nam Ho, Trinh Le Ba Khanh, Chi Dai Tran, Duong Khac Vu, Son Hong Phan, Hyung-Min Jeon, Jae Wook Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2608.24130)  

**Abstract**: Multi-camera 3D perception systems for warehouse scenes are trained largely on synthetic data and evaluated on physically captured environments. The resulting synthetic-to-real gap, which corrupts ground-plane localization and cross-camera identity association, is usually treated as one deficiency for a single domain-adaptation module to absorb; we argue instead that it enters the pipeline at three separable points: the camera calibration, the object shape prior, and the assumption that the object census is known, each admitting a different local remedy. Our online pipeline, Syn2RealTrack, follows this decomposition: lens distortion is recovered from images alone under a calibration that provides none, detections are fused across views by a visibility-weighted part-based descriptor that abstains on occluded parts rather than guessing, person height is measured in closed form from calibration instead of copied from a synthetic prior, and a closed-world cardinality prior is paired with a causal filter that removes the phantom boxes the prior manufactures. The system therefore adapts by reallocating trust between geometry and appearance without retraining a feature extractor. On the AI City Challenge 2026 Track~1 evaluation server it reaches a 3D Higher Order Tracking Accuracy (HOTA) of 52.0118%. The code will be released at this https URL 

---
# TransPhy: Visual In-Context Learning for Physically Grounded Image Editing 

**Authors**: Siyi Xie, Xuanke Shi, Jinsheng Quan, Haoran Tang, Zukai Chen, Lei Yang, Quan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.24119)  

**Abstract**: Visual demonstrations provide a natural interface for specifying image transformations that are difficult to describe exhaustively with text. However, existing visual in-context learning (VICL) methods primarily focus on appearance-level relation transfer and provide limited support for physically grounded transformations, whose outcomes depend on material properties, geometry, object interactions, and environmental conditions. Given a source--target exemplar pair and a query image, physically grounded VICL requires a model to infer the demonstrated transformation, adapt its effects to the query-specific scene context, and preserve rule-irrelevant content. We introduce PhysVICL-74, comprising 74 physically grounded transformation rules and 5,240 source--target image pairs that form nearly 75K training and evaluation contexts. Its benchmark split separately evaluates novel-instance transfer and unseen-rule generalization. We further propose TransPhy, a framework that decomposes physically grounded VICL into physical-rule induction and transition-aligned rendering. TransPhy first predicts the demonstrated rule and an explicit query-specific target-state description, and then synthesizes the target image through token-wise mixture-of-experts adaptation, with expert routing guided by localized transition cues. Experiments show that TransPhy improves physical-rule adherence, query consistency, and unseen-rule generalization over existing visual in-context editing methods. 

---
# PonderPounce: A Pretrained MLLM as an Episode Context Engine for Robot Control 

**Authors**: Suhwan Choi, Jaeyoon Jung, Sungkyung Kim, Yunsung Lee, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2608.24115)  

**Abstract**: Multimodal large language models (MLLMs) can integrate long visual histories, reason under partial observability, and infer behavior from a few examples. Yet vision-language-action (VLA) models generally inherit pretrained representations without using this contextual capacity as episode memory. Memory-dependent policies address this gap through purpose-built history mechanisms. PonderPounce instead reuses an MLLM's native causal context as robot memory. Ponder, a System2 MLLM, accumulates episode observations, demonstrations, and prior cognition in its native causal context and can generate subgoal text and demonstration reasoning for internal use. Pounce, a System1 VLA, receives the current observation, instruction, and proprioception directly; through the Ponder--Pounce interface, it asynchronously receives only the newest continuous cognition token and its age. Both are jointly trained end to end without a purpose-built memory module or separate bridge pretraining. Optimized serving achieves p50 latencies of 78ms for cognition refresh and 25ms for action-model invocation, supporting 20Hz action playback. On RoboMME with base-scale training data, PonderPounce reaches 60.83% with 9B and 50.04% with 0.8B under the same Pounce architecture and interface, versus 44.51% for FrameSamp+Modul and 17.93% for the current-observation \pi_{0.5}. With 9x data, it reaches 75.54% versus 57.88% for FrameSamp+Modul. On RoboCasa-DC, the same interface learns from action supervision alone and reaches 12.5% versus 11.6% for the strongest published demonstration-conditioned baseline, falling to 8.6% when cognition is replaced by a learned null state. 

---
# Structured Frequency-Domain Evidence for LLM-Based Time-Series Anomaly Detection 

**Authors**: Jungwook Seo, Sangwon Son, Minjeong Kim, Seungmin Han, Seojin Yoo, Sungyong Baik  

**Link**: [PDF](https://arxiv.org/pdf/2608.24113)  

**Abstract**: Time-series anomalies can appear not only as pointwise deviations but also as changes in recurring temporal structure, such as shifted periodicity or localized oscillatory fluctuations. However, existing LLM-based time-series anomaly detection methods mainly expose time-domain evidence through indexed values, plots, or de-seasonalized representations, leaving spectral structure implicit. We propose an evidence-augmented zero-shot TSAD framework that preserves indexed de-seasonalized observations while adding compact frequency-domain evidence computed with the Fast Fourier Transform (FFT). The evidence is constructed at two resolutions: global frequency-domain evidence summarizes sequence-level periodic context, while local frequency-domain evidence captures time-localized spectral departures. Experiments on AnomLLM with InternVL2-LLaMA3-76B, Qwen2.5-VL-72B-Instruct, Gemini-2.5-Flash, and GPT-4o, together with evaluation on the TSB-AD-U subset, show that explicit frequency-domain evidence improves LLM-based TSAD baselines. These results suggest that frequency-domain evidence can complement indexed and de-seasonalized time-domain inputs for zero-shot LLM-based TSAD. 

---
# MatReplace: A Reference-Free, Conditioning-Aligned Benchmark for Material Replacement in Interior Scenes 

**Authors**: Mingzhe Du, Thong Thanh Nguyen, Nguyen Tran Cong Duy, See-Kiong Ng, Luu Anh Tuan  

**Link**: [PDF](https://arxiv.org/pdf/2608.24107)  

**Abstract**: Material replacement is a common interior-design operation: changing the material of a selected surface while preserving its geometry, surroundings, and illumination. Despite its commercial relevance, no public benchmark isolates this task, and evaluating it is challenging. Reference-based metrics penalize valid outputs in this inherently one-to-many setting, favor the style of the reference generator, and cannot fairly compare editors that receive different forms of guidance. We introduce MatReplace, a reference-free benchmark that evaluates edits along four verifiable dimensions: local material correctness, global lighting harmony, outside preservation, and inside structure. It defines three tracks that vary one conditioning signal at a time: (A) instruction only, (B) instruction plus region mask, and (C) material reference image instead of instruction. Our results reveal a clear divide between naming and visually grounding materials. In Track A, leading closed-source editors achieve exemplar-level material rendering and surpass the exemplar anchor under our primary aggregate. In Track B, masks help only mask-compatible models with weak scene preservation, with task-paired, single-seed effects ranging from +0.137 to -0.090 across aligned model families. In Track C, reference-image conditioning degrades every family under both aggregates, by -0.031 to -0.508; in the worst cases, models repaint the reference image itself and perform worse than returning the input unchanged. Thus, named-material rendering is largely solved by the strongest closed editors on this distribution, but grounding materials from pixels remains an open challenge. Expert ratings validate our ranking (Kendall's tau = 0.68) and align with our aggregates more closely than GT-referenced or CLIP-based baselines. 

---
# Knowing When to Ask for Help: Bayesian Self-Escalation in Hierarchical LLM Agents 

**Authors**: Nadeem Shaikh  

**Link**: [PDF](https://arxiv.org/pdf/2608.24087)  

**Abstract**: Current LLM agent systems decide delegation before reasoning begins (a router picks a model) or after a response is complete (a verifier scores it and may retry). We study a third regime: an agent that recognises, during its own reasoning, that it is unlikely to succeed and transfers control to a stronger model. We formulate intra-generation delegation as a Bayesian optimal-stopping problem over a learned competence posterior -- an online estimate of the agent's eventual task success whose sufficient statistics are learned from labelled trajectories, not read off raw entropy. We derive the myopic escalation threshold in closed form, characterise the optimal policy via dynamic programming, and prove that the optimal policy is a time-varying threshold with no shape assumption on the raw signal. We further prove exponential separation of the oracle belief at the Chernoff-information rate of the signal, a regret bound governed by the calibration of the posterior, and a finite-sample guarantee: with n labelled calibration trajectories the deployed plug-in policy's regret decays as 1/sqrt(n). A controlled simulation study confirms each prediction of the theory, including the predicted 1/sqrt(n) rate. We additionally report a real-model validation on a Qwen2.5-Coder 1.5B->7B code cascade (MBPP, 257 tasks), confirming two of three pre-registered predictions: the escalation frontier dominates post-hoc routing at equal cost, and the cumulative competence belief's discrimination rises over generation. 

---
# PARTAB: Partition-Aware Reasoning with Structured Evidence for Scalable Table Understanding 

**Authors**: Md Mahadi Hasan Nahid, Davood Rafiei  

**Link**: [PDF](https://arxiv.org/pdf/2608.24082)  

**Abstract**: Large Language Models (LLMs) have shown strong capabilities in table reasoning, but their effectiveness degrades as tables grow in size and complexity due to irrelevant context and difficulty localizing the evidence required for reasoning. Existing approaches typically reason over either the full table or a single reduced view, which can still obscure important row-column relationships. We introducePARTAB (Partition-Aware Reasoning overTables), a framework that constructs a structured evidence interface between the LLM and the table. PARTAB represents query-relevant evidence as semantically coherent, row-linked table regions and performs hierarchical selection over column groups and row-level partitions before composing the selected evidence for answer generation. We evaluate PARTAB on multiple table reasoning benchmarks, covering question answering, fact verification, and numerical reasoning. PARTAB consistently improves over full-table prompting and several recent table reasoning methods, achieving strong performance on WikiTableQuestions and TabFact while remaining competitive on numerical reasoning. Additional analyses show that semantic partitioning and targeted evidence selection improve evidence localization, substantially reduce the reasoning context, and provide larger benefits on complex tables. These results demonstrate the value of structured, partition aware evidence construction for scalable table reasoning. 

---
# When Less Is More: An Empirical Study of Minimal Responses in Counseling Dialogues and the Behavior of LLMs 

**Authors**: Zhiyang Qi  

**Link**: [PDF](https://arxiv.org/pdf/2608.24080)  

**Abstract**: In psychological counseling, effective support is not always delivered through long, information-rich responses. Minimal responses, such as backchannel cues and concise empathic statements, help convey attentive listening, express empathy, and encourage clients to continue expressing themselves. However, existing counseling dialogue systems and evaluation frameworks often favor explicit, content-rich replies, overlooking the interactional value of brief counselor utterances. This paper presents a systematic cross-lingual analysis of minimal responses across multiple counseling dialogue datasets. We develop a two-stage filtering method based on utterance length and content, followed by contextual verification using a large language model (LLM). Our analysis shows that minimal responses are common in human-collected datasets but substantially underrepresented in LLM-generated ones. We further evaluate current LLMs in manually curated dialogue contexts where human counselors used minimal responses. The results show that strong commercial LLMs are capable of generating minimal responses when explicitly instructed, but still struggle to determine when such responses are appropriate. Counseling-specific models trained on synthetic data perform particularly poorly, tending instead to produce longer and more information-rich responses. Moreover, LLM-based response-quality evaluation may undervalue minimal responses, even when they are interactionally appropriate. 

---
# ORBITALIF: An Efficient Spiking Federated Learning Framework for Onboard Cloud Removal 

**Authors**: Bohan Zhang, Chenyu Xu, Yijie Mao, Yuanming Shi  

**Link**: [PDF](https://arxiv.org/pdf/2608.24073)  

**Abstract**: Low-earth-orbit (LEO) satellites enable high-resolution, large-scale Earth observation for applications such as disaster monitoring and environmental surveillance. However, cloud coverage often obscures the Earth's surface, and conventional cloud-removal pipelines that download cloudy images to ground stations for processing suffer from limited contact windows, constrained satellite-to-ground bandwidth, and high latency. In this work, we propose a novel satellite federated learning framework for cloud removal across LEO constellations, named orbital attention leaky integrate-and-fire (OrbitALIF). OrbitALIF performs both onboard training and inference using a compact 2.30,M-parameter spiking neural network (SNN) backbone with an adaptive gated fusion module (AGFM) and a spectral-spatial hybrid attention module (SHAM), combined with a decentralized federated learning strategy that shares model weights via inter-satellite links. Our experiments show that OrbitALIF achieves competitive cloud removal quality while consuming only 0.287,mJ per inference on neuromorphic hardware, a 72.3 times (98.6%) energy reduction versus an equivalent artificial neural network (ANN). 

---
# Mechanistic Circuit Identification for Controllable Data Generation 

**Authors**: Nakyung Lee, Sangwoo Hong, Jungwoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.24065)  

**Abstract**: While recent advances in data synthesis aim to curate high-quality datasets, most generation pipelines still rely on heuristic prompt-based control. This black-box paradigm provides limited insight into how individual samples interact with a model's underlying learning dynamics. To bridge this gap, we propose a circuit-grounded framework that connects training-dynamics-based data valuation with mechanistic interpretability (MI). Specifically, we conceptualize data quality along three complementary utility axes, learnability, challenge, and alignment. First, we uncover specialized model-internal circuits that causally govern these utility signals. Then, moving beyond heuristic prompting toward mechanistic control, we leverage these circuits as controllable interfaces, actively steering generation to produce utility-targeted data. Building on this capability, we introduce SAMS (Stage-Aware Mechanistic Scheduling), which schedules circuit-steered data according to the model's evolving optimization needs. Experiments on multiple-choice QA tasks demonstrate that our approach yields precisely controlled data with greater diversity than prompt-based baselines, consistently improving downstream performance and calibration. Ultimately, this work establishes a principled white-box paradigm for interpretable data generation, pioneering the use of MI not just as an analytical tool, but as a practical, controllable interface. 

---
# VisCache: Visual KV Cache Pruning for Efficient Vision Large Language Model Inference 

**Authors**: Lyuke Wang, Zhuo Li, Guangxu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.24063)  

**Abstract**: While Vision Large Language Models (VLLMs) have achieved remarkable success in multimodal reasoning, their long-context inference remains prohibitively expensive due to the massive computation and memory overhead of visual Key-Value (KV) caches. Existing KV compression methods often apply uniform pruning across visual tokens and layers, leading to substantial information loss and degraded this http URL address this challenge, we propose \textbf{VisCache}, a plug-and-play framework for coarse-to-fine \textbf{Vis}ual KV \textbf{Cache} pruning without training, which consists of two synergistic stages. First, a lightweight VLM filters temporal redundancy by selectively forwarding semantically informative keyframes. Second, we introduce {PruneKV}, a surgical KV compression algorithm tailored to the attention dynamics of VLLMs. Unlike rigid pruning strategies, PruneKV adopts a parabolic layer-wise budget allocation together with an asymmetric update mechanism that selectively prunes keys while fusing values, thereby preserving critical contextual information. Extensive experiments demonstrate that VisCache substantially improves inference efficiency, achieving up to {2.35$\times$ speedup} and significant memory reduction while maintaining competitive performance with only {19--28\%} KV cache retention. VisCache consistently outperforms existing baselines, establishing a new Pareto frontier between efficiency and performance for long-context VLLM inference. Code is available at this https URL 

---
# Don't Just Listen, Try Planning: Graph-based Retrieval-Generation Agent for Long-form Audio Meeting Understanding 

**Authors**: Quanwei Tang, Dong Zhang, Shoushan Li, Guodong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.24048)  

**Abstract**: While long-form audio meeting understanding (LAMU) is garnering growing attention, task-specific question answering (QA) datasets remain scarce. Existing speech QA paradigms and state-of-the-art Speech LLMs suffer from acoustic information loss and poor long-term context memory. To address these issues, we construct the LongAudioQA dataset and propose the GRGA model, which models heterogeneous audio features into a multi-dimensional graph and leverages agent planning for retrieval and answer generation. 

---
# Hierarchical Skill Retrieval for Data-Efficient Adaptation of Vision-Language-Action Models 

**Authors**: Haoran Hao, Shahram Najam Syed, Jeff Schneider, Jeffrey Ichnowski  

**Link**: [PDF](https://arxiv.org/pdf/2608.24042)  

**Abstract**: While Vision-Language-Action (VLA) models pretrained on large-scale robot datasets provide a strong foundation for robot manipulation, their performance can degrade when adapted to new tasks with limited task-specific demonstrations. Retrieval offers a practical way to reuse existing demonstrations for data-efficient adaptation, but existing methods often rely on visual similarity, state-action representations, or task-level language matching. These approaches may overlook the hierarchical structure of long-horizon manipulation tasks, where complete task matches are rare but reusable skills are often abundant. To address this challenge, we propose Hierarchical Skill Retrieval (HSR), a retrieval framework for data-efficient VLA adaptation. Specifically, HSR first decomposes a target task into candidate skill sequences. It evaluates each plan based on both semantic plausibility and skill reliability estimated from the prior dataset. The selected decomposition is then used for hybrid retrieval. This combines subtask-level language retrieval with behavior-feature reranking to identify demonstrations that are both semantically relevant and compatible with the target task. Finally, we adapt the policy through a two-stage pretraining and finetuning pipeline, which separates general skill acquisition from task-specific adaptation. Experiments on the LIBERO benchmark and several real-world robot manipulation tasks show that HSR improves the average success rate by 10.3% and 21.3% over the strongest baseline, respectively. These results demonstrate the effectiveness of structured skill-level retrieval for data-efficient VLA adaptation. Videos and code are available at this https URL. 

---
# Design-to-Plan: A Large Language Model-Based Multi-Agent Framework for Manufacturing Process Planning from 3D CAD Models and 2D Engineering Drawings 

**Authors**: Muhammad Tayyab Khan, Lequn Chen, Wenhe Feng, Seung Ki Moon  

**Link**: [PDF](https://arxiv.org/pdf/2608.24039)  

**Abstract**: Manufacturing process planning transforms heterogeneous design information into coherent manufacturing decisions. However, existing approaches focus on isolated subtasks, such as feature recognition, drawing interpretation, or tool selection, and struggle to support the full reasoning chain from design artifacts to process plans. This is critical when planning must interpret 3D CAD models, 2D engineering drawings, materials, and domain-specific rules. To address this gap, this paper presents Design-to-Plan, a large language model (LLM)-based multi-agent framework for end-to-end manufacturing process planning. An orchestrator coordinates specialized agents for 3D feature recognition, 2D drawing analysis, 2D-3D context fusion, knowledge retrieval, process sequencing, tool selection, and report generation. Rather than using LLMs as standalone text generators, the framework deploys them as reasoning agents that interact with deterministic modules and knowledge sources to produce consistent and traceable decisions. In this hybrid design, deterministic modules and specialized agents extract structured information from CAD and drawing inputs, while LLM agents perform context-aware reasoning, retrieve manufacturing rules, resolve conflicts, and generate planning outputs. The framework is evaluated using 300 benchmark cases across three downstream ReAct-enabled agents, plus separate evaluations of CAD feature recognition, drawing analysis, and 2D-3D context fusion. The parallel architecture achieves 100% success across downstream agents, Tool F1 scores of 95.9%-97.6%, 90% source detection accuracy in conflict analysis, and a 60%-68% reduction in token usage for key planning tasks. Results show that structured LLM-based multi-agent coordination can bridge design representations and manufacturing knowledge, enabling scalable, efficient, and traceable design-to-plan automation. 

---
# ChorusTIC: Training-Free Multivariate Time Series Classification via Chorus In-Context Learning 

**Authors**: Juntao Fang, Shifeng Xie, Ruichu Cai, Shengji Zheng, Zijian Li, Keli Zhang, Lujia Pan, Themis Palpanas, Zhifeng Hao  

**Link**: [PDF](https://arxiv.org/pdf/2608.24033)  

**Abstract**: Time series classification underpins applications in healthcare, sensing, and industrial monitoring. Although time series foundation models support forecasting and transferable representation learning, classification still typically requires fitting a task-specific classifier on each target dataset, while individual channels of multivariate inputs are often encoded independently. We introduce ChorusTIC, a classification-native foundation model for in-context classification across heterogeneous channel configurations without target-task parameter updates. ChorusTIC combines episode-consistent Random Subchannel Slot Concatenation with a shared dual-axis encoder to model temporal and cross-channel interactions and map variable channel configurations into a fixed-width representation independent of the original channel count. It then calibrates feature axes using context-derived distributions and predicts query labels through leakage-protected in-context learning. We pretrain ChorusTIC solely on synthetic labeled episodes comprising context and query sets that share a task background, with classes distinguished by sparse temporal or cross-channel rules. Evaluations on the complete UEA-30 and UCR-128 archives show strong full-context and low-label performance without target-specific classifier fitting. 

---
# What Guides the Agent? Adjudicating Unauthorized Behavior via Localizing Behavior-Guiding Instructions 

**Authors**: Yichao Gao, Yumo Zhang, Yunhao Yao, Haohua Du, Puhan Luo, Ruiqi Li, Zhiqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.24022)  

**Abstract**: LLM agents integrated with external resources gain complex task capabilities, yet the unified natural-language context channel makes them vulnerable to injection attacks: untrusted external data may be dynamically parsed as behavior-guiding instructions during LLM inference, thereby subverting the agent's decision. Existing defenses focus on static detection or isolation of malicious content at the input/output level, remains insufficient for detecting such dynamic inducements that arise during model reasoning. We propose Attnlocate, a runtime framework for fine-grained localization of context spans that genuinely influence tool-calling decisions, i.e., behavior-guiding instructions. Attnlocate casts this localization problem as an object detection task, aiming to detect the distinctive activation traces induced by behavior-guiding instructions within the attention matrix. Specifically, we design a multi-head, multi-layer attention aggregation scheme to construct a token-level feature space tailored for object detection. Then, a 1-D U-Net equipped with an anchor-free detection head is deployed to detect these spans. Finally, based on the authority of the provider from which the detected behavior-guiding spans originate, Attnlocate dynamically adjudicates malicious invocation attempts. We evaluate Attnlocate across ten agent configurations from five LLM families, covering scenarios involving indirect prompt injection and tool poisoning. Attnlocate achieves a mean IoU of 0.743, an average AUROC of 0.956, and a 0.934 true-positive rate at 0.067 false-positive rate. It also transfers effectively across unseen models and supports authority policy adaptation without retraining. 

---
# IterCAD: Iterative Program Repair for CAD Code Generation from Orthographic Views 

**Authors**: Yuchuan Wu, Ke Niu, Haiyang Yu, Zhuofan Chen, Xiangyang Xue, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.24020)  

**Abstract**: Generating executable parametric CAD code from dimension-annotated orthographic drawings is a challenging task requiring geometric understanding, procedural reasoning, and precise numerical prediction. Existing vision-language approaches typically formulate this problem as one-shot generation, preventing the model from inspecting intermediate CAD results and correcting early mistakes, often leading to non-executable code or geometrically inconsistent outputs. In this paper, we propose IterCAD, an iterative framework that reformulates orthographic-view-to-CAD generation as a progressive program repair process. Instead of predicting the final CAD code in a single pass, IterCAD repeatedly analyzes the current CAD result, reasons about its discrepancy with the target views, and explicitly decides whether to REVISE the code or STOP the refinement process. To make iterative repair learnable, we further construct IterCAD-RS, a structured revise-or-stop supervision set containing both repairable intermediate CAD states and already-correct states, and develop a three-stage training strategy for initial generation, revision learning, and multi-turn RL optimization. By closing the loop between visual understanding, geometric verification, and code refinement, IterCAD progressively corrects structural and parametric errors. Experiments on CADExpert show that IterCAD consistently improves code executability and geometric fidelity over strong one-shot baselines. 

---
# WebMCP-Phalanx: Enforcing and Characterizing Trust Boundaries for Browser-Integrated LLM Agents 

**Authors**: Lin-Fa Lee, YI-YU Chang, Kuo-Hui Yeh  

**Link**: [PDF](https://arxiv.org/pdf/2608.24017)  

**Abstract**: The emerging W3C WebMCP proposal enables LLM agents to invoke tools exposed by web pages. In multi-party web environments, however, integrating agent execution into a browser security model centered on the Same-Origin Policy (SOP) leaves insufficient provenance and lifecycle guarantees for agent-accessible tools, creating three risks: subject-attribution spoofing, uncontrolled tool lifecycles, and semantic prompt injection. We propose WebMCP-Phalanx, a dual-layer agent runtime architecture. Its first layer provides a browser-native trust anchor that binds each tool to its registering principal through cryptographically protected capability credentials and propagates provenance labels throughout the tool lifecycle. Its second layer separates semantic inspection from privileged tool use. A Quarantine Agent (Q-LLM), without tool invocation authority, inspects tool metadata, outputs, and page-supplied content for prompt injection. Validated content is then forwarded to a Privileged Agent (P-LLM) for execution, while the Q-LLM's internal state remains hidden from page scripts. Empirical evaluation shows that the browser-native ownership mechanism reduces revocation and overwrite attack success from 100\% to 0\%. The dual-agent runtime blocks all 80 prompt-injection attempts embedded in tool descriptions and limits tool-return attacks to 2 successful cases out of 80. Across experiments, task utility remains statistically indistinguishable from the no-attack baseline. Under a white-box adaptive attacker, however, description-based filtering can be bypassed through malicious tool names invoked before inspection. This finding motivates a call-timing gate that delays tool invocation until all agent-visible tool metadata has been validated. 

---
# SAGE: From Direct Answering to Evidence-Grounded Inference for Chinese Ancient Document Understanding 

**Authors**: Yuchuan Wu, Xuan Luo, Yinglian Zhu, Meng Fang, Xiangyang Xue, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.24011)  

**Abstract**: Chinese ancient document understanding demands complex visual, linguistic, and historical reasoning. Current Large Vision-Language Models (LVLMs) typically rely on an opaque, single-pass generation paradigm, often producing overconfident and weakly grounded responses. To address this, we propose SAGE, an evidence-grounded multi-agent framework that reformulates Chinese ancient document understanding as evidence-grounded inference rather than direct answer generation. SAGE coordinates specialized agents for task-aware planning, tool-mediated evidence acquisition, claim-level verification, and bounded replanning under a constrained shared-state runtime. This design supports bounded evidence seeking, answer revision, and abstention when grounding is insufficient. Experiments on the AncientDoc benchmark show that SAGE consistently outperforms matched direct-answering baselines across three LVLM backbones. Remarkably, SAGE with Qwen3.5-9B surpasses much larger monolithic LVLMs on most evaluated metrics, highlighting the importance of structured, evidence-grounded inference beyond model scaling. 

---
# Hybrid Semantic Tool Discovery for Enterprise MCP Gateway: Architecture and Implementation 

**Authors**: Olympia Saha, Amy Wang, Srinivasan Manoharan  

**Link**: [PDF](https://arxiv.org/pdf/2608.23992)  

**Abstract**: Large language model (LLM) agents invoke external tools to retrieve and reason over information beyond pretrained knowledge. The Model Context Protocol (MCP) standardizes how such tools are surfaced, and a proxy MCP server aggregates many backend servers behind a single endpoint providing a secure, governable chokepoint for authentication, policy enforcement, and observability. This architecture creates two compounding challenges: a context-engineering bottleneck where full tool schemas saturate the model context window before any user query, and a tool discoverability barrier where users and agents cannot identify the best tool among 2,000+ indexed tools across 200+ MCP servers. Prompt caching reduces reprocessing cost but neither frees context capacity nor improves accuracy. We present SCOUT (Selective Context Optimization for Universal Tooling), which reframes tool exposure as a context-selection problem, injecting only tools relevant to the current step. SCOUT surfaces two MCP meta-tools -- tool_search and execute_tool -- where tool_search performs hybrid retrieval, fusing BM25 sparse matching with dense vector search via Reciprocal Rank Fusion to return the top-k relevant tools. Backed by zero-downtime catalog update pipelines, SCOUT resolves both context saturation and tool discovery challenges. In production at PayPal, SCOUT reduces MCP tool-token consumption from 140.2k tokens (70.1% of context) to 1.3k tokens (0.8%), a 99% reduction, cutting per-query inference cost at enterprise scale. Because SCOUT is surfaced as standard MCP tools, it is model-agnostic and requires no client-side modifications. 

---
# The Shadow Price of Intelligence: Quality Degradation in LLM Inference as a Supply Chain Problem 

**Authors**: Elioth Sanabria  

**Link**: [PDF](https://arxiv.org/pdf/2608.23986)  

**Abstract**: Large language model providers are compute constrained, and their universal response to congestion is to degrade service: route queries to smaller models, cut reasoning effort, truncate context. The industry's accounting says this saves money. We show the accounting is wrong, because it prices a query when the customer buys an answer. A degraded answer fails with some probability, and a failed answer either returns as a retry, inflating arrivals when the system is most loaded, or departs as churn, destroying lifetime value on a ledger no cost dashboard displays. We model inference allocation with three classical primitives: a newsvendor whose stockout cost is churned lifetime value, a geometric retry multiplier in which the recycled product is dissatisfaction, and a two-regime transient queue whose arrival rate is made endogenous by retries. Statically, there is a nonempty, measurable regime in which a cheaper model saves energy per satisfied answer while consuming strictly more capacity per satisfied answer, so the discount inverts exactly when capacity binds. Dynamically, a reactive throttle fired during a surge can cross an ignition threshold beyond which it manufactures more traffic than it sheds, and a release rule set below the degraded equilibrium converts a transient surge into a permanent degraded regime. With heterogeneous customers, throttling is a transportation problem in retry-inflated load whose optimal policy rations intelligence by critical ratio, class by class, and whose dual, the shadow price of intelligence, prices a marginal query by class and by hour; closed-form trajectories make it computable in milliseconds. Stochastic analysis sharpens rather than erodes the thesis: the ignition boundary acquires a predicted width, and noise punishes the reactive policy that parks the system against it. Under congestion, throttling is not a cost lever but a demand lever. 

---
# RAGSentinel: Certifiable Geometric Consensus for Robust Retrieval-Augmented Generation 

**Authors**: Yueyang Quan, Anjun Gao, Yufei Xia, Minghong Fang, Zhuqing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.23965)  

**Abstract**: Retrieval-augmented generation (RAG) improves the factuality of large language models by grounding responses in external documents, but it also exposes a critical security vulnerability: adversarial documents injected into the knowledge database can enter the context window and steer the model toward targeted incorrect answers. Existing post-retrieval defenses rely on instruction following, parametric knowledge, or text-level consistency, all of which can be imitated or optimized against by adaptive attackers. We propose RAGSentinel, a training-free, label-free defense for black-box RAG systems. RAGSentinel uses a surrogate encoder to measure query-conditioned hidden-state shifts induced by retrieved documents, removes shared topic directions, and filters poisoned documents as geometric outliers from a robust majority consensus. We prove that, under an honest-majority assumption and a representation-level separation condition, RAGSentinel exactly recovers a poison-free majority-sized context. Experiments across three question-answering datasets, three LLM families, and multiple poisoning attacks show that RAGSentinel consistently achieves low attack success rates while preserving competitive accuracy and remaining effective against adaptive attacks with full pipeline knowledge. 

---
# Evaluating Language Models on Cross-Language Code Functional Equivalence 

**Authors**: Hui Sun, Anderson Uchôa, Rohit Gheyi, Wesley K. G. Assunção  

**Link**: [PDF](https://arxiv.org/pdf/2608.23961)  

**Abstract**: Background: Large Language Models (LLMs) have demonstrated strong performance across a variety of code-understanding tasks, leading many to believe that they can reason about program semantics. However, existing evaluations primarily focus on single-language settings or rely on synthetically generated code, raising concerns about whether current results reflect true semantic understanding. Aims: We investigate whether LLMs can accurately judge functional equivalence across different programming languages in human-written code, a setting that requires deeper reasoning beyond superficial similarity. Method: We introduce PolyHuman, a dataset of human-written programs in CPP, Java, and Python. Using this dataset, we evaluate intra- and inter-language equivalence detection across open-weight and proprietary LLMs, selecting GPT-o4-mini as a representative model to assess stability. We then manually analyze 81 cases of systematic disagreement in which models incorrectly judge functional equivalence, examining the code logic and the generated Chain-of-Thought reasoning. Finally, we categorize these failures and compare them across GPT-o4-mini, Claude-Opus-4.7, and Gemini-3-Flash to determine whether they reflect model-specific issues or broader limitations of state-of-the-art LLMs. Results: We identify a difficulty-dependent breakdown in equivalence judgment (harder problems make the model increasingly prone to misclassifying non-equivalent code as equivalent), a model-specific sensitivity to programming language for the best-performing model (particularly a more conservative behavior on Python), and a partial reliance on similarity-based cues. GPT-o4-mini also shows substantial run-to-run instability under identical settings, indicating inconsistent rather than absent capability. Conclusions: Current LLMs do not reliably capture functional equivalence within or across languages. 

---
# NeuronGuard: Robust LLM Safety Alignment via Ablation-Aware Safety Signal Redistribution 

**Authors**: Anjun Gao, Yueyang Quan, Yufei Xia, Zhuqing Liu, Minghong Fang  

**Link**: [PDF](https://arxiv.org/pdf/2608.23959)  

**Abstract**: Safety alignment in large language models (LLMs) remains brittle against a growing spectrum of attacks. Jailbreak attacks bypass safety mechanisms through crafted prompts, while neuron-level attacks directly prune safety-critical neurons post-deployment. Both exploit a common weakness: safety-relevant information concentrates in a sparse neuron subset. We present NeuronGuard, a fine-tuning-stage defense that simultaneously hardens LLMs against both attack classes by redistributing safety signals across a broader set of neurons. NeuronGuard dynamically identifies safety-critical neurons via periodically refreshed per-layer linear classifiers, forces refusal behavior under deliberate neuron ablation, and applies KL-divergence regularization for distributional consistency. A randomized gradient projection strategy preserves downstream task utility by resolving conflicts between the defense and task objectives. We provide a formal guarantee that NeuronGuard strictly reduces the attack success rate (ASR) upper bound, and experiments across three LLMs, six state-of-the-art attack strategies, and multimodal settings confirm near-zero ASR while maintaining task accuracy, including against white-box adaptive adversaries. 

---
# The Empire, Long Divided, Must Unite: Architectural Convergence in Three LLM Agent Harnesses 

**Authors**: Dai Jiahong  

**Link**: [PDF](https://arxiv.org/pdf/2608.23953)  

**Abstract**: An agent harness is what turns a language model into an autonomous agent: the surrounding code that builds the model's context, mediates its tools, runs the loop, and persists state across a long-horizon run. This layer, not the model it wraps, is increasingly the binding constraint on agent behaviour. We present a source-level, multi-case study of three open coding-agent harnesses built from deliberately opposing philosophies: LangChain's deepagents (batteries-included), Earendil's pi (radical minimalism), and DeepSeek's dsh (everything-is-a-plugin). Reading each at a pinned commit and following its commit history, we find that the two mature harnesses have travelled in opposite directions (deepagents subtracting authored scaffolding, pi accreting durable infrastructure), yet converged toward one architectural middle form of five recurring elements: a commoditised loop, an append-only replayable session record, model quirks kept as data, progressive disclosure of context, and explicit extension seams. A third harness, read afterward as a held-out check, exhibits all five, and in one seam reuses another's implementation outright. We therefore do not claim independent invention, and decompose the convergence into parallel discovery, diffusion, and literal reuse. Finally, one load-bearing dimension shows no convergence, and indeed no presence: external verifiability, a tamper-evident record an outside party can check without trusting the runtime. We read this absence not as an oversight but as a predictive gap, the next axis on which harnesses for provenance-sensitive domains will differ. 

---
# STAIN-FL: Stealthy Targeted Attack Injection with Contextual Triggers in Federated Learning 

**Authors**: Ashlinder Kaur, Purnima Murali Mohan, Zengxiang Li, Tram Truong-Huu  

**Link**: [PDF](https://arxiv.org/pdf/2608.23952)  

**Abstract**: Federated video anomaly detection trains model collaboratively without sharing raw surveillance footage, but limited server-side visibility lets compromised clients to inject backdoor via malicious updates. This paper introduces STAIN-FL, a stealthy targeted backdoor attack injection framework that uses naturally occurring surveillance conditions, including low-light scenes, indoor settings, and crowd density, as contextual triggers. STAIN-FL combines anomaly-to-benign label \textit{manipulation} with gradient masking over least-updated coordinates to preserve clean accuracy while inducing trigger-conditioned misclassification. We evaluate STAIN-FL on \texttt{UCF-Crime} using 1024-dimensional I3D features in a non-IID four-client multi-agency setting, comparing FedAvg and FedProx under sparse and continuous attacks. Results show that sparse attacks have low-detectability, operationally significant attacks rather than high-intensity attacks: they keep the mean clean-accuracy drop below $2\%$, yet still misclassify more than half of triggered anomalies at peak backdoor accuracy under FedAvg ($56.7\%$) and FedProx ($54.2\%$). Under FedAvg, the sparse backdoor remains above the $25\%$ backdoor-accuracy threshold for an average of $336$ post-attack rounds, highlighting the persistence risk of contextually triggered attacks in surveillance systems. 

---
# Luce: Relightable Gaussians for 3D Asset Generation 

**Authors**: Mayank Singh, Michele Stoppa, Alvise Memo, Rui Yu, Harsha Kalli, Srimanth Gunturi, Muhammad Ahmed Riaz, Behrooz Shahsavari, Waleed Abdulla, David E. Jacobs  

**Link**: [PDF](https://arxiv.org/pdf/2608.23943)  

**Abstract**: High-fidelity image-to-3D generation requires a 3D representation that captures both geometry and appearance. To support relighting and integration into standard rendering pipelines, the representation should include physically based rendering (PBR) modalities such as albedo, metallic-roughness, and surface normals. We propose Luce, a 3D representation that unifies geometry and PBR materials within a voxelized multimodal Gaussian cloud, using dedicated Gaussian primitives for each modality. A variational autoencoder compresses this representation into a unified material-aware latent space. A rectified-flow transformer generates this latent from a single image, conditioned on multi-layer features from a pretrained image encoder that preserve both semantic context and fine spatial detail. The latent then decodes into relightable PBR Gaussians and an optional textured mesh with a tangent-space normal map. On Toys4K, Luce achieves state-of-the-art single-image-to-3D generation, improving FID by 28% over the strongest baseline. We further introduce a benchmark of AI-generated images, on which Luce improves the CLIP image-alignment score over the best baseline (0.8519 vs. 0.8299). Luce generates relightable, geometrically accurate, and materially faithful assets that preserve fine details such as text, logos, and inscriptions. 

---
# QML for Quantum Sensing under Measurement-Induced Information Loss 

**Authors**: Sounak Bhowmik, Himanshu Thapliyal  

**Link**: [PDF](https://arxiv.org/pdf/2608.23934)  

**Abstract**: Nitrogen-vacancy (NV) centers in diamond can serve as highly sensitive solid-state quantum sensors for high-sensitivity magnetometry. However, in the noisy intermediate-scale quantum (NISQ) era, extracting reliable information from noisy, finite-shot, and measurement-limited sensing data remains a considerable challenge. Whereas, quantum machine learning (QML) offers a potential path to improve parameter estimation by learning nonlinear relationships between quantum-sensing data and the underlying physical signal. In this work, we investigate the role of QML in magnetic-field estimation within an NV center-inspired magnetometry setting. We formulated magnetic field sensing as a supervised regression task. We compared the performance of several classical machine learning models trained on measurement-based classical data with that of quantum kernel-based models trained on pre-measurement coherent quantum states. Our objective is to isolate the impact of measurement-induced information loss and therefore provide a theoretical upper bound on the sensing performance. The upper bound is achievable only when coherent quantum information is directly available to the learning model. Our results show that QML-based sensing performance improves significantly with coherent quantum-state information, and not much with changes in model complexity or learning paradigm. This observation underscores the importance of learning pipelines that tightly integrate quantum sensors and QML models to enhance magnetic field sensing under realistic constraints. 

---
# RefineRank: Joint Box Refinement and Ranking for Surgical Spatio-Temporal Grounding 

**Authors**: Linzhe Jiang, Jiayuan Huang, Changhao Zhang, Chunyang Jiang, Zhehua Mao, Mobarak I. Hoque  

**Link**: [PDF](https://arxiv.org/pdf/2608.23928)  

**Abstract**: Surgical spatio-temporal grounding (STG) requires locating, at each video time specified by a procedural question, the object that the question asks about. Existing approaches face a trade-off: vision language models understand the question context but produce imprecise coordinates, whereas open-set detectors provide localized candidate boxes whose confidence does not reflect which box answers the question. We introduce RefineRank, which closes this gap at the candidate-box level. A compact trainable module, RefineNet, combines the language and regional features of a frozen medical vision language model with the proposals of a frozen open-set detector: it predicts a bounded coordinate correction and a quality score for every candidate box, and a fixed decoding rule returns the original or refined box with the highest score. On the MedVidBench Official Rankings (Verified), RefineRank records 0.421 STG mIoU, the highest displayed STG score, while its global multi-metric rank is 11. In a controlled evaluation on separate training and evaluation videos, coordinate correction raises the candidate oracle upper bound from 0.6772 to 0.7302, and ranking the joint pool of original and refined candidates by their RefineNet scores improves STG mIoU from 0.2719 to 0.4534, whereas separately trained selectors over the same pool reach at most 0.4186. These results show that a small box-level module can reconcile question understanding with precise localization without retraining either backbone. Code is available at [this https URL](this https URL). 

---
# Names Can Hurt: Spotting Slopsquatting Risks Caused by Package Name Hallucinations in Local Coding LLMs 

**Authors**: Akash Raj, Sargam Sahu  

**Link**: [PDF](https://arxiv.org/pdf/2608.23897)  

**Abstract**: When a code generating language model fabricates a Python package name, an adversary who has pre-registered that name on PyPI can convert that hallucination into a supply chain compromise. This event has been termed as 'slopsquatting'. We propose a two layer detector to counter this issue. The first layer performs a deterministic PyPI existence check. The second is a Random Forest classifier trained on ten features derived from the package name and its PyPI metadata. An import name reconciler bridges the two, resolving cases such as 'import cv2' versus 'pip install opencv-python' without a security bypass. The detector is embedded in a LangGraph state machine that retries at escalating temperatures and, on repeated failure, routes to a stronger fallback model. Across 300 curated prompts, the pipeline produces hallucination free code on 76% of runs. The primary exhausts its retry budget on 28.7%; intra model retries recover roughly a quarter of those, and cross model fallback recovers a further 16.5% of the remainder. Four findings have been observed. First, half of the flagged hallucinations are packages already registered on PyPI, as low quality lookalikes of well known projects, caught by the classifier rather than the deterministic layer (e.g., pil, faiss, tabula, haystack). Second, hallucination rate scales almost linearly with prompt adversariality, from 0 to 10% on routine coding to 40 to 73% on slopsquat baits. Third, the weaker primary refused 6 of 10 direct baits unaided, suggesting recent instruction tuning provides a baseline defense. Fourth, when primary and fallback share a model family, approximately 84% of primary failures recur on the fallback, motivating cross family pairing. A user study (n = 24) reports mean satisfaction 4.4 out of 5 and 21 of 24 stated adoption intent. 

---
# Learning the Kohn-Sham map with neural operators for quasi-linear scaling density functional theory 

**Authors**: Danish Khan, Maurice D. Hanisch, Nikolai Argatoff, Evan Xie, Sandeep Sharma, Anima Anandkumar  

**Link**: [PDF](https://arxiv.org/pdf/2608.23895)  

**Abstract**: Kohn--Sham density functional theory (DFT) underpins electronic-structure simulations, but repeated orbital diagonalizations lead to cubic scaling, restricting quantum calculations to modest scales only. Eliminating these auxiliary orbitals while retaining Kohn--Sham accuracy is the central goal of orbital-free DFT, but both analytical and machine-learning methods have so far fallen short. Prior learning approaches either try to learn the variational kinetic-energy functionals, which are ill-conditioned, or directly predict the ground state, which extrapolate poorly to larger systems. Instead, we identify the Kohn--Sham map as the right learning target for orbital-free DFT. It maps a Kohn--Sham potential directly to the corresponding density and noninteracting kinetic energy, quantities otherwise obtained through an orbital diagonalization. Focusing on the density component in this work, a domain-invariant $\mathrm{SE}(3)$-equivariant Fourier neural operator learns to predict it from the potential as input on real-space grids, enabling stable quasi-linear scaling SCFs. Trained jointly on 8,504 molecules and solids, a single model generalizes to out-of-distribution organic molecules, insulators, and metals. For the first time, the same method converges SCFs across these systems without explicitly constructing Kohn--Sham orbitals, while reproducing densities, electronic spectra, and structural observables at Kohn--Sham DFT accuracy. Linear-scaling SCFs additionally allow converging magnesium dislocation densities containing up to 82,500 valence electrons on a single GPU. 

---
# A Mathematical Theory of Interpretation: Rational Entropy, Spectral Readout, and Confusability as a Resource 

**Authors**: Blake Reynolds  

**Link**: [PDF](https://arxiv.org/pdf/2608.23892)  

**Abstract**: This article presents the abridged core of \emph{A Mathematical Theory of Interpretation} (MTI), which treats interpretation as observer-relative spectral measurement under an access structure. MTI makes interpretation a method-design problem: access, query, utility, and medium determine what an observer can select, identify, communicate, or refuse. On a learning-invariant Hilbert realization, Rational Entropy measures residual uncertainty across knowledge, utility, and medium. In the finite-effective regime, we classify its zero set. Pairwise confusability is equivalent to uniform atomic collapse, while a unique utility maximum can select one atom even when other zero-cost states remain non-atomic. This reverses the usual zero-error role of confusability: agreement in at least one observer direction excludes unresolved multi-atom readings, while the joint label preserves identification. The corresponding free-design capacity is the product of all but the smallest direction budget. A four-condition certificate characterizes sharp, decodable, medium-faithful, and order-independent readout on a finite commuting code sector and returns typed obstructions when those guarantees fail. Together, these results establish MTI as a theoretical basis for constructing interpretation methods with explicit access assumptions, guarantees, and failure modes. 

---
# A tale of perfect fit and phantom optima: how data-driven models can fail in real-time optimization 

**Authors**: Prithvi Dake, Rahul Bindlish, James B. Rawlings  

**Link**: [PDF](https://arxiv.org/pdf/2608.23885)  

**Abstract**: Real-time optimization (RTO) relies on process models to locate economically optimal operating conditions. Because developing first-principles models requires significant process knowledge, data-driven alternatives are increasingly attractive. Modern machine-learning models can fit historical plant data accurately and often pass standard validation tests. Whether such models can be trusted for economic optimization, however, remains unclear. We investigate this question using a vinyl acetate monomer benchmark process with a unique, well-conditioned economic optimum. We train a structured hybrid model that combines known mass balances and thermodynamics with a neural-network closure for unknown kinetics, and a fully data-driven neural ordinary differential equation (ODE) model. Both models reproduce plant measurements accurately and exhibit little variation in predictions across random initializations. Yet their economic optima differ substantially from that of the plant. Where the plant returns a single optimum on multistart search, the trained models return many phantom optima. We further show that the training optimizer alone can be yet another source of error. Even with noise-free data and initialization at weights that recover the plant optimum, stochastic gradient training can drift to weights that yield substantially worse RTO solutions. The identified model is thus an artifact of the training optimizer as well as the data. These results demonstrate that a good predictive fit of all available measurements does not guarantee reliable economic performance. A data-driven model for RTO should at least be required to recover the optimum on a decision-oriented benchmark like the one developed here before being considered for plant testing and application. 

---
# Revelation Control 

**Authors**: Qinyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.23860)  

**Abstract**: Revelation Control is the problem of choosing priced interventions that reveal hidden state only insofar as the revealed distinctions can change a consequential decision, while accounting separately for any useful progress created by the intervention itself. We develop this theory for learning systems, where states equivalent under declared current information can respond differently to future training and favor different actions. The framework defines decision-sufficient revelation and revelation depth, separates pure information value from productive reuse, embeds static Bayes refinement into state-dependent continuation value, and gives an exact cost-adjusted factorization criterion: an additional shallow coordinate is decision-nonredundant only when states sharing a scalar summary lie on opposite sides of the priced Stop/Continue boundary. We also give a target-independent protocol for model-specific instantiation and prove that bounded stop-flip risk alone cannot certify positive expected utility under unrestricted severity. Across Qwen2.5-7B and Mistral-7B-v0.3, deeper future-learning probes have positive decision value and productive reuse yields strict equal-compute utility advantages. Qwen additionally provides evidence for a decision-nonredundant shallow revealability regime; in Mistral, a scalar continuation architecture fit only on an independent development panel retains positive familywise-adjusted lower bounds on a disjoint target panel, consistent with scalar decision sufficiency within the tested architecture family and resolution. The evidence supports structural rather than numerical transfer: the decision theory, cost accounting, continuation logic, and evaluation protocol transport, while empirical proxies, coefficients, thresholds, and even the required shallow state dimension may be system-specific. 

---
# Beyond the Mandate: A Systematic Security Analysis of the Agent Payments Protocol (AP2) 

**Authors**: Avital Aviv, Parth A. Gandh, Ron Bitton, Asaf Shabtai  

**Link**: [PDF](https://arxiv.org/pdf/2608.23858)  

**Abstract**: The Agent Payments Protocol (AP2), introduced by Google, enables large language model (LLM)-driven shopping agents to authorize and execute payments on behalf of users. Its signed Checkout and Payment Mandates protect the integrity of transaction data after signing. Agent interactions and external inputs that shape a transaction before authorization remain outside that protection, including Agent-to-Agent Protocol (A2A) messages and Model Context Protocol (MCP) tool calls. Prior work identified replay and prompt-injection attacks in AP2 v0.1. AP2 v0.2 addresses some of these issues but adds capabilities and deployment assumptions that require renewed analysis. We present a systematic security analysis of AP2 v0.2 based on its roles, transaction lifecycle, deployment architectures, and trust boundaries. We divide the lifecycle into five phases and identify five deployment architectures. Using MAESTRO (Multi-Agent Environment, Security, Threat, Risk, Outcome), we model four threat actors, eleven attack surfaces, eighteen adversary capabilities, and six attacker goals. The resulting catalog contains 48 threats spanning five attack families. We score these threats with the Artificial Intelligence Vulnerability Scoring System (AIVSS), identifying eight that reach the High band in at least one architecture. Because no complete public AP2 deployment was available, we build a testbed spanning all five architectures and develop five proof-of-concept demonstrations covering all eight High-risk threats and their mitigations. We also develop a deployment-aware scanner that maps applicable threats to static, cross-role consistency, and adversarial checks. Our analysis shows that valid mandate signatures alone do not ensure that an agent-mediated transaction reflects the user's intent when its pre-authorization context is manipulated. 

---
# Coronavirus Optimization Algorithm: A Success-History Adaptive Evolutionary Framework with Archive-Assisted Search and Stagnation Recovery for Global Optimization 

**Authors**: Hari Mohan Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2608.23847)  

**Abstract**: This paper proposes the Coronavirus Optimization Algorithm (COA), a SARS-CoV-2-inspired success-history adaptive evolutionary optimizer for box-constrained continuous global optimization. COA does not model disease transmission; instead, it maps selected coronavirus mechanisms to explicit search operators, including elite-guided attraction, trial-vector generation, adaptive parameter variation, stagnation recovery, and population-size scheduling. The algorithm combines opposition-based initialization, current-to-pbest mutation, binomial crossover, an external archive, success-history adaptation, population reduction, and partial restart. COA is evaluated on 29 CEC 2017 benchmark functions at 10, 30, and 50 dimensions against 15 competitive optimizers. Results show that COA achieves the best overall Friedman rank across all dimensions, with particularly strong performance on composition functions. The findings demonstrate that COA is a compact, transparent, and competitive adaptive evolutionary optimizer, while also highlighting limitations on some hybrid functions and the need for further high-dimensional validation. 

---
# Automated Synthesis of Cloud Emulators 

**Authors**: Archit Bhatnagar, Zhenning Yang, Sarah McClure, Yiming Qiu, Sylvia Ratnasamy, Ang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.23842)  

**Abstract**: DevOps programming (e.g., using CLI/API scripts or IaC frameworks) is key to cloud infrastructure management. Unlike traditional programming tasks, DevOps program testing needs provisioning and execution against actual cloud resources, which is often time-consuming, unsafe, and costly. Cloud emulators have gained popularity for easing DevOps program testing; they are generally API-level mocks that can execute DevOps programs in a local environment. Still, building these emulators remains challenging: developers must manually interpret extensive cloud documentation and handcraft logic for each service, API, and their interaction. This does not scale to the complexity of the cloud, which is further a moving target as the services and APIs evolve. CloudEmu is an automated approach that constructs emulators based on cloud documentation via neurosymbolic code synthesis. The key idea is to combine LLMs' general strengths in documentation understanding and code generation with cloud-specific symbolic abstractions that suppress hallucinations and enforce precision at scale, while using the real cloud as an oracle for automated testing, repair, and alignment. Our evaluation shows the effectiveness of CloudEmu on major cloud provider (AWS and GCP) services in both coverage and accuracy. CloudEmu outperforms the existing leading tool LocalStack, which was manually developed by a large team of engineers over a decade. 

---
# ShardMeter: Sharded and Geo-Distributed Training Without the Guesswork 

**Authors**: Tim Beringer, Patrick Diem, Felix Wolf, Arya Mazaheri  

**Link**: [PDF](https://arxiv.org/pdf/2608.23840)  

**Abstract**: Training large-scale AI models often outgrows a single data center, demanding sharded, multi-cluster, and decentralized training. However, the huge space of resource allocations makes exhaustive benchmarking and manual tuning impractical, while performance depends on tightly coupled factors like model size, GPU memory, batch size, bandwidth, and sharding strategy. We introduce ShardMeter, a lightweight analytical performance model that predicts the end-to-end runtime of transformer-based workloads across arbitrary sharded, distributed, and even decentralized training. Given a model's characteristics and a target hardware topology, ShardMeter estimates per-GPU and per-island throughput, training cost, total wall-clock time, and identifies performance bottlenecks. Our analysis reveals diminishing-return regimes as island size increases, quantifies transitions between compute- and communication-bound scaling, evaluates hyperparameter trade-offs, and models cost-throughput for large-scale decentralized training. ShardMeter exposes these insights to quickly explore the configuration space, choose near-optimal deployment plans, and avoid costly trial and error. 

---
# Resilience Matters for Embodied Agents System: New Metrics, Systematic Evaluation, and Optimization 

**Authors**: Yapeng Liu, Yuanzhao Zhai, Xudong Gong, Dawei Feng, Bo Ding, Lin Wang, Huaimin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.23839)  

**Abstract**: Embodied Agents System (EAS) are increasingly deployed in open-world physical domains, where reliability directly dictates deployment quality and human-agent trust. However, existing evaluations rely on outcome-centric metrics as success rate or safety scores that collapse diverse execution trajectories into coarse scores, obscuring the dynamic processes underlying agent behavior. Therefore, they ignore a critical property of EAS -- which we define as the Resilience -- that reflects how EASs recover, stabilize, and extend under perturbations and across iterative updates. The lack of resilience is particularly critical in open-world environments due to continuous unexpected disruptions, thus directly affecting the quality of EAS deployment. To address this problem, we gain insight from the resilience-engineering concepts to EAS groundings and propose a novel resilience evaluation framework that can be flexibly applied to any EAS. Specifically, we define the first comprehensive resilience metrics suite for EASs system that exposes Rebound, Stability, and Graceful Extensibility across embodied tasks execution, providing a practical grounding for EAS resilience analysis. We further implement the resilience evaluation layer that transforms execution process into assessments for diagnosis and optimization. Across 400 household tasks with 10 EAS, we reveal the process-level distinction hidden by outcome metrics, including recovery cost differences among successful episodes ($\Delta C_{rec}=25.2$), increased instability and task-family degradation. Metrics-guided optimizations reduce recovery cost and increase stability, graceful extensibility completion, showing the diagnostic effect of resilience evaluation. Our results reveal a trade-off among resilience characteristics, suggesting that a resilient EAS construction should be configured according to deployment-specific requirements. 

---
# Infant Care Video Dataset for Classification of Interventions Using Transformers 

**Authors**: Igor Bogdanov, James Green  

**Link**: [PDF](https://arxiv.org/pdf/2608.23838)  

**Abstract**: Healthcare documentation in the neonatal intensive care unit (NICU) presents significant challenges, with nurses spending approximately 25\% of their time on record-keeping, while up to 60\% of interventions remain undocumented. Motivated by the need to detect interventions from video automatically, we present the Infant Care Video Dataset (ICVD), a collection of 4,144 videos spanning 12 simulated intervention classes designed for developing automated documentation systems. Our manikin-based approach systematically varies conditions, such as camera angle and clinician skin tone, while ensuring privacy compliance. Using video transformer architectures (TimeSformer and MotionFormer), we establish strong baseline performance (93.97\% and 93.17\% top-1 accuracy) among the 12 infant care classes. Our ablation study comparing temporal models with a framewise approach (23.17\% accuracy) demonstrates a 70.80\% performance gap, validating the need for temporal modeling. The ICVD provides a foundation for developing automated documentation systems to reduce clinical burden in neonatal care environments and improve existing practices. 

---
# Predicting Radiologist Expertise from 3D Gaze Patterns During CT Interpretation 

**Authors**: Leila Khaertdinova, Anna Anikina, Claudia Mello-Thoms, Bulat Ibragimov  

**Link**: [PDF](https://arxiv.org/pdf/2608.23836)  

**Abstract**: Accurate interpretation of volumetric CT requires efficient navigation of 3D image volumes and attention to diagnostically relevant regions. While eye-tracking has been widely studied in 2D medical imaging, its use for expertise assessment in CT settings remains limited. We propose a gaze-informed transformer framework for expertise classification in thoracic CT. Using a DINOv2 backbone, radiologist fixation patterns are integrated into volumetric feature learning through (1) a learnable log-space bias in self-attention and (2) gaze-weighted pooling of patch embeddings. We trained and evaluated our approach on 182 CT reading sessions from five radiologists with varying levels of experience. On a held-out test set, the model achieves an ROC-AUC of 0.91 and F1 score of 0.86, outperforming adapted methods. These findings suggest that incorporating visual search behavior into transformers may support objective, process-based expertise assessment in radiology. Code is available via this https URL. 

---
# Place, Slice and Schedule: Hierarchical O-RAN Control of a Tethered mmWave UAV-gNB 

**Authors**: Alireza Mohammadhosseini, Fatemeh Afghah  

**Link**: [PDF](https://arxiv.org/pdf/2608.23824)  

**Abstract**: Unmanned aerial vehicle (UAV)-mounted 5G New Radio base stations (gNBs) can augment terrestrial networks with an on-demand, repositionable Frequency Range 2 (FR2) capacity layer. This flexibility, however, couples the physical network topology with radio-resource management: UAV movement reshapes blockage, channel quality, and the set of effectively served users, while traffic demand, queues, and service requirements evolve at a much faster timescale. Existing Open Radio Access Network (O-RAN)-enabled UAV studies optimize trajectory, deployment, association, or resource allocation, but typically in isolation, without coordinating slow aerial control with fast per-user scheduling. We instead exploit O-RAN disaggregation, Key Performance Indicator (KPI) monitoring, and multi-timescale RAN Intelligent Controller (RIC) control to address this coupling: a Non-Real-Time RIC rApp uses aggregated KPIs and radio-environment context to jointly control tethered UAV placement and the enhanced Mobile Broadband (eMBB)/Ultra-Reliable Low-Latency Communication (URLLC) slice budget, while a Near-Real-Time RIC xApp allocates per-user resources within that budget. We realize this xApp as a permutation-equivariant DeepSets Soft Actor-Critic (D-SAC) scheduler that treats the users as an unordered set, trained in a Sionna RT ray traced channel. The resulting hierarchical controller improves eMBB SLA satisfaction by up to 17% and URLLC on-time delivery by up to 42% over classical and learned schedulers; the learned rApp further raises URLLC on-time delivery by up to 20% over baselines. 

---
# Learning to Grade Efficiently: A Bandit-Driven Prompt-Selection Framework for Low-Cost LLM Essay Scoring 

**Authors**: Olga Manakina, Igor Bogdanov  

**Link**: [PDF](https://arxiv.org/pdf/2608.23814)  

**Abstract**: Large Language Models (LLMs) demonstrate strong capabilities in automated essay scoring (AES), but contemporary approaches typically employ fixed prompt selection, failing to address operational cost concerns and evolving optimal configurations. We propose a novel cost-aware approach that treats each prompt type as an arm in a multi-armed bandit (MAB) controller, enabling adaptive selection of optimal prompting strategies during inference. Our experiments on IELTS Writing Task 2 essays show that the MAB framework achieves comparable scoring accuracy to exhaustive grid search while reducing LLM calls by 78.4\% to find the best grading approach. We implemented four distinct grading recipes (multi-step vs. single-step assessment, with vs. without calibration examples) and found that the multi-step approach with examples achieves the highest accuracy. By tracking token usage and latency alongside agreement metrics, we produce the first cost-reliability learning curves for essay scoring, providing actionable insights for educational technology platforms that must balance operational costs against assessment validity. This work represents the first application of online control mechanisms to adaptively select prompting strategies in AES, transforming prompt selection from an offline hyperparameter optimization problem into an efficient online learning task. 

---
# Discovering Cross-Language Reasoning Invariance in LLMs with Geometry-Invariant Sparse Autoencoders 

**Authors**: Igor Bogdanov, Changcheng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.23809)  

**Abstract**: Multilingual language models can solve the same mathematical problem in different languages, but it remains unclear whether they rely on shared features or on language-specific computations that only produce similar outputs. We study this question in five models from four families using the Multilingual Grade School Math (MGSM) dataset, with problems solved in English, German, French, Spanish, Russian, and Chinese, retaining problems with valid reasoning traces in all six languages and replaying those traces through the model to record representations at multiple layers. For each model, we first use Centered Kernel Alignment (CKA) to identify layers with cross-language alignment. At each selected layer, we train two sparse autoencoders (SAE): a baseline reconstruction-only model and a contrastive variant introduced in this work, the Geometry-Invariant SAE (GI-SAE). GI-SAE supplements the reconstruction loss with an Information Noise-Contrastive Estimation (InfoNCE) loss that trains the encoder to produce similar activations for traces of the same problem, regardless of language or token position. We then test whether the resulting shared features are functionally interchangeable by swapping their values between languages during the model's forward pass and measuring the resulting change in output, quantified by Kullback-Leibler (KL) divergence per feature. Although GI-SAE yields higher CKA and Jaccard similarity at nearly every layer, higher geometric similarity does not consistently imply greater functional interchangeability. We find that cross-language feature sharing is model- and architecture-dependent in this sample and appears at different depths in different models. GI-SAE primarily amplifies cross-language structure already present: the pattern is model-specific, with strengthening in Qwen, no functional benefit in Gemma, and mixed layer-dependent effects in Llama and Phi. 

---
# LUCAID: Agentic Multimodal AI for Lung Cancer Precision Pathology 

**Authors**: Marie-Lisa Eich, Kai Standvoss, Timo Milbich, Alexander Möllers, Miriam Hägele, Philipp Anders, Lars Tharun, Hanna Kontradiuk, Sebastian Kons, Nader Aldoj, Recepcan Adigüzel, Adam Narai, Lukas Hönig, Jonathan Striebel, Binru Yang, Mihnea P. Dragomir, Marvin Sextro, Philipp Keyl, Philipp Jurmeister, Rosemarie Krupar, Evelyn Ramberger, James Wells, Julika Ribbat-Idel, Andreas Kunft, Hussam Shuaib, Christian Grohé, Reinhard Büttner, David Horst, Klaus-Robert Müller, Lukas Ruff, Maximilian Alber, Frederick Klauschen, Simon Schallenberg  

**Link**: [PDF](https://arxiv.org/pdf/2608.23803)  

**Abstract**: Lung cancer tissue diagnostics is complex, as therapy decisions in precision oncology rely on the integration of histomorphological, immunohistochemical, and molecular features. Yet pathological assessment remains largely visual and semi-quantitative and shows interobserver variability, while existing artificial intelligence (AI) tools cover only selected tasks, rarely reach generalizable expert-level performance, and lack prospective clinical validation. To address these challenges, we developed and clinically validated LUCAID, an agentic AI system for precision lung cancer pathology. An integrative agent couples diagnostic reasoning with nine modules that cover the full routine workflow, from quality control, tumor detection and segmentation, histological subtyping, tumor microenvironment profiling, tumor cellularity quantification, and predictive biomarker scoring (PD-L1, MET, TROP-2) to automated structured report generation. LUCAID enables users to interactively query the module outputs and generate reports that contextualize the results. Against large-scale expert ground-truth annotations, the analysis modules achieved F1 scores of 0.82-0.95. In prospective clinical validation, LUCAID reached 93.0% concordance with an expert-panel adjudicated reference standard across clinically actionable decisions, compared with 68.3-81.1% for five experienced thoracic pathologists. 

---
# Restoring Without Forgetting: Continual Learning Across Image Degradations 

**Authors**: Alif Ashrafee, Bartosz Krawczyk  

**Link**: [PDF](https://arxiv.org/pdf/2608.23799)  

**Abstract**: Recent progress in image restoration has converged on all-in-one architectures that jointly handle multiple degradations within a single network. These methods are effective on static benchmarks but target a closed-world setting that assumes simultaneous access to every target degradation at training time. In practice, degradations are encountered sequentially as field-deployed systems progressively face new environmental conditions, and historical training data is often unavailable due to privacy or storage constraints. Accommodating a new degradation then requires either retraining on the union of all prior data, which is often costly or infeasible, or fine-tuning, which causes catastrophic forgetting. We formulate multi-degradation image restoration as a continual domain-incremental learning problem, in which degradations arrive incrementally and prior data is unavailable. Our proposed Restoring without Forgetting (RwF) framework learns a lightweight adapter for each new degradation, eliminating forgetting by construction at a fraction of the cost of dedicated per-domain networks. To isolate degradation learning from dataset variation, we construct a benchmark spanning five degradation domains under shared image content. At test time, an unsupervised routing mechanism identifies the appropriate restoration path for unknown inputs without requiring domain labels. Across the five-domain sequence, RwF improves final average PSNR over naive sequential fine-tuning by 15.25 dB and 11.83 dB on the Restormer and NAFNet backbones, respectively. The framework transfers to eleven canonical real-degradation benchmarks (3,465 images) at 89.5% routing accuracy with only a +0.94 dB oracle PSNR gap, establishing, to our knowledge, the first systematic baseline for continual multi-degradation image restoration. 

---
# EmoTra-TTS: Smooth Intra-Utterance Emotion Transitions for Speech Synthesis 

**Authors**: Tianchi Liu, Zeyang Song, Tianrui Wang, Zhipeng Li, Chenglin Xu, Yiwen Guo  

**Link**: [PDF](https://arxiv.org/pdf/2608.23791)  

**Abstract**: Psychological research on emotion dynamics has established that human affect is a continuous, evolving process: emotions rise, decay, and transition within seconds. Current emotional text-to-speech (TTS) systems, however, condition on a single discrete label or static embedding per utterance, fundamentally misaligning with the temporal nature of affect. While recent LLM-based TTS systems may implicitly vary prosody through text understanding, such variation is neither explicitly controllable nor precise enough for targeted intra-utterance transitions. We address three challenges: (1) a multi-pass flow blending pipeline synthesizes frame-aligned transition audio, circumventing the scarcity of natural intra-utterance transitions; (2) dual-stage Valence-Arousal-Dominance (VAD) conditioning guides prosodic planning in the LLM and acoustic realization in the flow decoder via frame-level VAD embeddings; (3) direction-magnitude decoupled injection structurally separates emotion direction from injection magnitude, preventing content degradation. EmoTra-TTS adds only +0.43% parameters with no latency overhead, achieves 30%-87% relative improvement on emotion transition quality, corroborated by 64.4%-79.5% overall win rates in pairwise preference tests against four SOTA baselines and two commercial systems. 

---
# When Youth Enter The Chat: An Epistemic Shift in the Validation of LLM-Based Measures of Student Talk 

**Authors**: Liliana Santos-Deonizio, James Malamut, Ramón Martínez, Dorottya Demszky  

**Link**: [PDF](https://arxiv.org/pdf/2608.23780)  

**Abstract**: LLMs are being used increasingly to measure aspects of student discourse (e.g. talk moves, collaboration, equity of voice) at scale. Typically, LLM-based measures of student talk use transcriptions of classroom conversations that only include verbal contributions, which de-contextualize student language. Common practices for validating these measures include comparing outputs against expert annotations by adults, using held out evaluation sets and F1 scores. We argue that these approaches are insufficient to ensure that such measures are meaningful and equitable for teaching and learning, particularly for racially and linguistically marginalized youth. In order to center the youth whose talk is being analyzed, re-contextualizing these classroom conversations and engaging youth in the research process is necessary. Sharing epistemic authority with youth, ultimately, centers their point of view and adds crucial nuance to the analysis of their talk that adult experts, researchers, and LLMs cannot provide. In a case study of multilingual youth in one 8th-grade math classroom, we address the epistemic exclusion of youth by employing multiple ethnographically-oriented methods to re-contextualize student conversations and center youth as epistemic authorities in conversation with researchers and LLMs. We conducted participant observations, interviews, focus groups, and member checks with four focal students. Findings reveal that there were misalignments between students' interpretations of their own math talk experiences and the LLM-based measures of their talk. Students contested both the LLM classifications and the coding scheme used to measure their talk, highlighting the need for youth to be involved in the epistemic process of producing knowledge about their experiences. 

---
# Disentangled Skill Representations for Predictive Human Modeling 

**Authors**: Mariah Schrum, Deepak Gopinath, Srijan Srivatsa, Guy Rosman, Tiffany Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.23776)  

**Abstract**: Understanding human skill is important for AI systems that collaborate with, coach, or assist people. Unlike typical latent variable estimation problems which rely on single observations, skill is a persistent, compositional, and behaviorally grounded construct that must be inferred from patterns over time. We introduce Skill Abstraction with Interpretable Latents (SAIL), a method for modeling human skill as an interpretable, multi-dimensional construct inferred from naturalistic behavior. Our approach produces a skill embedding that is robust to transient performance fluctuations and learns a transferable representation of human subskills. Furthermore, SAIL supports skill-informed behavior prediction that generalizes across a variety of in-domain contexts. We represent each individual with a persistent skill embedding that controls a blend between expert and novice bases and is trained using counterfactual subskill swaps for disentanglement. This design encourages representations that are both robust to performance variation and structured for interpretability. We demonstrate across racing and baseball that SAIL achieves strong predictive performance and consistently improves behaviorally grounded disentanglement over the evaluated baselines, while also improving downstream AI coaching performance. 

---
# What Reaches Expert Review? Representation, Structural Screening, and Candidate-Form Dependence in AI-Assisted Item Development 

**Authors**: Christopher Brooks  

**Link**: [PDF](https://arxiv.org/pdf/2608.23766)  

**Abstract**: Between AI-assisted item generation and expert review sits a computational evaluator whose decisions are usually treated as technical preliminaries. Yet representation, structural reduction, and selection policy determine which items and evidence psychometricians ever receive. Across two linked in-silico studies of 32,000 selected Big Five items, we followed fixed source populations from semantic representation through structural evaluation and candidate-form construction. Broad agreement in semantic geometry concealed consequential local differences: identical wording acquired different construct evidence, different items survived, and intended attributes could disappear even as community correspondence improved. These sensitivities also differed across generated source populations. At the final review boundary, both eligibility policies filled every content cell in every evaluable form, yet they presented different wording. Across embedding configurations, inclusive primary forms shared a median of only 6 of 40 items, reflecting the total downstream consequence of changing representation across structural evidence and ranking. The apparent stability of global summaries and complete forms therefore concealed instability in the content reaching psychometricians. The computational evaluator is not neutral infrastructure between generation and expertise; it is an inspectable and revisable part of measurement design. 

---
# TrustShiftProbe: Characterizing, Benchmarking, and Defending Staged Trust Attacks on MCP Servers 

**Authors**: Mehrdad Rostamzadeh, Sidhant Narula, Mohammad Ghasemigol, Daniel Takabi  

**Link**: [PDF](https://arxiv.org/pdf/2608.23763)  

**Abstract**: The Model Context Protocol (MCP) has emerged as the standard layer connecting Large Language Model agents to external tool backends. This openness introduces a severe server-side threat we term TrustShift: a compromised MCP server behaves benignly during an initial conditioning phase, building operational reliance and suppressing agent skepticism, before switching to an adversarial payload once an interaction threshold is reached. The evasion is temporal, not syntactic: benign at deploy time, the server's defection is invisible to predeployment static analysis, which sees only the honest phase. Switched payloads range from overt structural violations to schema-valid manipulations, the latter preserving outer protocol compliance to evade runtime middleware filters. Crucially, TrustShift originates in the server-controlled tool channel, not user prompts (unlike indirect prompt injection) or the transport (unlike man-in-the-middle): the adversary is the trusted server endpoint itself. We introduce TrustShiftProbe, an evaluation and defense framework with four contributions: (1) a stateful temporal threat model of the agent-server lifecycle as a benign conditioning phase followed by an adversarial defection at a trust horizon; (2) a language-agnostic attack engine that instantiates each variant as a compromised MCP server across four production domains; (3) SHIELD, a multi-tier, zero-oracle runtime defense at the MCP transport boundary that audits server payloads against behavioral baselines learned during clean trust windows; and (4) a taxonomy of nine TrustShift variants spanning three execution mechanisms (structural violation, semantic corruption, scope expansion) and three adversarial objectives (disruption, exfiltration, and their combination). Across frontier proprietary and open-weight models, TrustShift attacks achieve a 69.5% mean attack success rate that SHIELD mitigates to 42.7%. 

---
# EXAM$^2$: $\underline{Ex}tending$ $\underline{A}udio$ $Understanding$ $in$ $\underline{M}ultilingual$ $and$ $\underline{M}ultimodal$ $Analysis$ 

**Authors**: Jiawen Wang, Xiaoxue Gao, Zi Haur Pang, Nancy F. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.23758)  

**Abstract**: Recent large audio language models (LALMs) have achieved impressive progress in audio understanding. However, existing evaluations remain largely constrained to English and narrow audio domains. Prior benchmarks typically focus on a single audio modality, i.e., speech, sound, or music, limiting the systematic investigation into how these models generalize across diverse visual scenarios. In this paper, we introduce EXAM$^2$, a benchmark for multilingual and multimodal audio understanding spanning six languages and multiple modalities, including speech, sound, music, mixed-audio settings, and visual images. By incorporating visual information alongside heterogeneous audio inputs, EXAM$^2$ enables more realistic evaluation of scene-aware audio reasoning and cross-modal comprehension. EXAM$^2$ comprises $5,667$ multiple-choice questions, $22,614$ image instances, and $135,684$ multilingual translations. We evaluate state-of-the-art open-source and proprietary LALMs as well as multimodal LLMs, revealing substantial performance gaps in multilingual and cross-modal understanding. Furthermore, we propose Gemma3n-EXAM$^2$, a lightweight fusion-model fine-tuned on EXAM$^2$-train, achieves up to $12.4\%$ improvement in multilingual settings and $21.7\%$ gains in multimodal evaluation over a strong baseline. Empirical results establish EXAM$^2$ as a challenging benchmark and pioneer future multilingual and multimodal audio intelligence research. 

---
# Too much of a good thing -- when knowledge distillation promotes overfitting, and how to avoid it 

**Authors**: Irene Trigueros-Lorca, Leonardo Concepción, Christian Wagner, Isaac Triguero, Daniel Molina  

**Link**: [PDF](https://arxiv.org/pdf/2608.23752)  

**Abstract**: The growing size of Convolutional Neural Networks has led to increasingly large and costly models. Knowledge Distillation (KD) addresses this by transferring knowledge from a large network (teacher) to a small one (student), also reducing the training data required. KD is traditionally applied only at the network's final output. However, its behaviour when applied at intermediate network layers has received little attention. This raises the question of whether intermediate block-wise KD, which provides supervision throughout the network, could offer an advantage under specific conditions, such as few instances per class, which is common in fine-grained datasets. This work proposes a student design based on simple, homogeneous blocks mirroring those of the teacher, distilling knowledge between corresponding blocks. Across eleven datasets, we show that on classic datasets, distilling only the last block is sufficient -- and often best--, whereas fine-grained, data-scarce settings benefit substantially from intermediate supervision, with even a single additional distillation point narrowing the gap considerably. We further study how this supervision should be guided, exploring configurations of varying granularity and informed by an explainability analysis based on attention maps, Centered Kernel Alignment, and Grad-CAM, alongside the impact of teacher and student fine-tuning strategies. This work shows that intermediate block-wise distillation, guided appropriately, is key to building compact data-efficient models without sacrificing accuracy. 

---
# The Limits of Automatic Evaluation of Creativity in Large Language Models 

**Authors**: Alessandro Tutone, Giorgio Franceschelli, Mirco Musolesi  

**Link**: [PDF](https://arxiv.org/pdf/2608.23705)  

**Abstract**: Large Language Models (LLMs) are increasingly capable of generating text that challenges human performance in domains requiring creativity, yet evaluating creativity in LLM-generated content remains a significant challenge. Here, we investigate whether current automatic evaluation methods can reliably capture human judgments of creativity. We collect human evaluations of human- and AI-generated short stories from the WritingPrompts dataset across 11 dimensions of creativity, and compare these judgments with automated objective metrics and LLM-as-a-Judge evaluations. Our experiments reveal substantial misalignment between automatic evaluations and human assessments. In particular, LLM-based judges exhibit a systematic preference for AI-generated stories, consistently favoring their stylistic characteristics over the unpredictability and other qualities of human-authored texts. Furthermore, correlation analyses show that widely used automatic metrics exhibit near-zero alignment with human judgments across both human- and AI-generated stories, suggesting that they fail to capture important dimensions of creativity. These findings highlight fundamental limitations in current approaches to the automatic evaluation of creative text and underscore the difficulty of reducing the multidimensional and subjective nature of creativity to computational metrics. 

---
# Confidently Wrong, Silently So: Auditing Undetectable Failures of a Deployed On-Device Language Model 

**Authors**: Shashwat Pandey, Satwik Pandey, Suresh Raghu  

**Link**: [PDF](https://arxiv.org/pdf/2608.23663)  

**Abstract**: Aligning deployed language models requires knowing when their outputs can be trusted, yet on-device models now ship to hundreds of millions of devices with no server-side moderation, and the configuration developers can actually deploy is rarely audited independently. We present a reproducible reliability audit of the developer-accessible on-device foundation model, framed as an oversight question: can a user or a resource-constrained developer tell when the model is wrong? Red-teaming it on calibration, confident confabulation on false-premise questions, and over-refusal of benign prompts, we find a \emph{task-asymmetric miscalibration}: its guardrails fail in opposite directions across tasks (confabulating on 69\% of false premises while refusing 18\% of entirely benign inputs), atop a self-reported confidence that is saturated and non-discriminative (AUROC 0.47; ECE 70, worst among comparable small models). Crucially, confident-correct and confident-wrong outputs are \emph{surface-indistinguishable}: a classifier over 15 user-visible features separates them at AUROC only 0.55 (equivalence-confirmed), leaving no signal for oversight at inference time. No cheap single-generation signal flags these failures ($\le$0.68 AUROC), whereas a black-box consistency wrapper requiring no model access recovers reliability (confident confabulation 75\%$\to$3\%; selective accuracy 43\%$\to$83\%) at a tunable cost. We contribute a model-agnostic audit protocol, a surface-indistinguishability test, and released code and frozen evaluation items as reusable infrastructure for auditing deployed models. 

---
# From Causal Plausibility to Causal Reliability: Evaluating LLMs as Calibrated Direct Causal-Edge Classifiers 

**Authors**: Amit Kumar, Elnur Adl Zarabi, Suranjana Trivedy, Zhiqian Chen, Lei Zhang, Kaiqun Fu, Taoran Ji  

**Link**: [PDF](https://arxiv.org/pdf/2608.23660)  

**Abstract**: Large language models (LLMs) are increasingly used to provide prior causal knowledge for structural causal discovery, yet whether their direct-edge judgments and confidence can be trusted remains unclear. We systematically evaluate 12 instruction-tuned open-weight models across six benchmark causal graphs, five prompting strategies, and four confidence sources: verbalized, logit-based, cross-prompt agreement, and cross-model agreement. Under our language-only pairwise protocol, our evaluation yields three key findings. (i) LLM-based causal judgments are strongly recall-dominant: models predict overly dense graphs with many false-positive edges, while prompting mainly shifts the precision-recall trade-off rather than resolving overprediction. Gains from model scale diminish on the largest graphs and do not eliminate miscalibration. (ii) LLMs often capture causal relatedness without reliably identifying directness or orientation. Relative to published reference graphs, models misclassify 40.0% of indirect and 36.0% of reversed non-edges as direct edges, versus 28.2% of other non-edges. Moreover, 80.8% and 84.6% of these false positives receive verbalized confidence of at least 80%, revealing substantial overconfidence in structurally incorrect predictions. (iii) Conventional confidence estimates are unreliable, whereas agreement offers a more promising signal. Logit-based confidence frequently collapses near 1.0 regardless of correctness, while cross-prompt and cross-model agreement achieve better mean calibration and discrimination, though their advantages are not statistically significant after Holm correction. A benchmark-familiarity audit further identifies potential familiarity in five model-dataset pairs, all involving AsiaM. Overall, our results suggest LLMs are better viewed as sources of externally validated soft causal priors than as direct evidence of causal structure. 

---
# Elastic KV Cache for LLM Serving:A Working Reclamation Mechanism, and Why Chunked Prefill Already Closes the Gap 

**Authors**: Sathishkumar Sivashanmugam  

**Link**: [PDF](https://arxiv.org/pdf/2608.23658)  

**Abstract**: An LLM serving engine sizes its key-value (KV) cache once, at startup, permanently setting aside a reserve for the worst-case prefill activation. During decode-dominant phases that reserve sits idle, yet it cannot be handed to the KV pool because it is exactly the memory a large prefill needs. We ask whether this reserve is reclaimable, and build a mechanism to test it. Our elastic KV cache lends the reserve to the KV pool during decode and returns it before prefill, driven by the scheduler's one-step-ahead view of the next batch. It is pure userspace on the CUDA virtual-memory path: two physical handles mapped into one contiguous virtual range per layer, so the attention kernel is unchanged and no driver patch is required. It decommits in a few milliseconds and recommits in tens of milliseconds, works with CUDA graphs and prefix caching, and never triggers an out-of-memory event. A static commit of the same memory is unsafe, crashing on prefill bursts, which makes the dynamic toggle necessary.
Having built the mechanism, we test the premise it rests on and report an honest negative result. It only pays off if a small prefill chunk size badly hurts prefill latency. In a controlled experiment injecting long prompts into a live decode load, that penalty is small (median time-to-first-token differs by about 1% between chunk sizes of 8192 and 32768 tokens), because prefill is compute bound and decode consumes only about one token per sequence per step. Simply lowering max_num_batched_tokens recovers more KV than the controller does, at nearly equal latency. The reserve also dilutes under tensor parallelism, from 16% of KV at TP1 to 2.7% at TP4. We state precisely when reclaiming the reserve could still help, and release the mechanism as a reusable userspace elastic-VMM allocator. 

---
# Beyond Executable Models: The Pufibara Agent Harness and the Modelica Agent Workflow Benchmark for Physical System Modeling 

**Authors**: Zizhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.23653)  

**Abstract**: AI agents are increasingly used for simulation-driven engineering. Physical system modeling presents different requirements from general-purpose code generation in software engineering, because correctness depends not only on syntax and executability but also on physical consistency and scenario-dependent behavior. We study this challenge in Modelica, an equation-based modeling language in which a model may compile and simulate while still violating its intended physics or engineering requirements. Across successive revisions, an agent may lose track of requirements or rely on simulation evidence produced by an outdated candidate.
To address this challenge, we present Pufibara, an agent harness that maintains persistent engineering state across revisions, associates execution and simulation evidence with the candidate that produced it, and makes submission an explicit agent action. To evaluate end-to-end Modelica agent workflows, we also propose a source-grounded method for constructing realistic and independently evaluable tasks. We use this method to build the 232-task Modelica Agent Workflow Benchmark, spanning Model Repair, Model Generation, and Model Tuning. Each submitted candidate is scored by a benchmark-owned evaluator outside the agent loop.
We compare Pufibara with Claude Code as complete harnesses under two matched large language model (LLM) backends. With DeepSeek v4 Flash, Pufibara passes 202 tasks, compared with 185 for Claude Code. With Claude Sonnet 5, Pufibara passes 202 tasks, compared with 187 for Claude Code. Under the repository-reported token accounting, Pufibara records 76.4%-82.5% lower logical-token totals. Its sequential runtime is 6.1%-58.4% lower. These findings show that, even under matched LLM backends, complete agent harnesses can differ substantially in both task success and resource use for physical system modeling. 

---
# Feedback That Backfires: Why Small Language Model Agents Repeat the Call They Just Watched Fail 

**Authors**: Esmail Gumaan  

**Link**: [PDF](https://arxiv.org/pdf/2608.23651)  

**Abstract**: Agent harnesses record a failed tool call and its error message in the transcript and ask the model to continue, on the assumption that the error is corrective information. We measure whether it is. Defining the corrective gain of a failure record as the change in log-probability of re-emitting the action that just failed, we find the gain is negative for every instruction-tuned model we tested (6 checkpoints, 135M-1.7B, 4 families) in two environments: simulated tool calling and MBPP program repair. Normalised by action length the effect is about -1.03 nats per action token, a factor of 2.8 in the odds of each token, and holds on 90%-100% of individual items, not only on average. Over a fixed candidate set the probability of repeating the failed call rises from 0.06 to 0.54, and greedy decoding reproduces it token for token on 19% of items after the failure versus 0% before. Counterfactuals pairing the same call with a failure message, a success message, or a neutral acknowledgement separate two effects: the failed call's surface form accounts for 83% of the damage, while the semantic contribution of marking it failed is small and inconsistent in sign across environments. The problem is in the harness, not the model's grasp of error messages, and that predicts which remedies work. Replacing the verbatim call with a runtime-generated description of the failure removes 76% of the inversion at no token cost, and making previously-failed strings unreachable at the decoder acts on the same term. Two plausible remedies do not: an explicit "do not repeat" instruction leaves the measured quantity where it was, and deleting the failed attempt to retry from a clean context, the standard prescription for context contamination, is the worst harness we measured for repetition, because it restores the context that produced the failure. The study runs end to end on a CPU; all artefacts are released. 

---
# ToolRobustBench: Stage-Wise Perturbation Evaluation and Failure Diagnosis for Tool-Calling Agents 

**Authors**: YiShan Zheng, Yuan Wu, Yi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2608.23635)  

**Abstract**: Large language models (LLMs) rely on tool calling as a fundamental agent capability, enabling them to invoke external systems and complete tasks beyond text generation. However, clean end-to-end (E2E) success cannot identify where a tool-use failure originates or how it propagates through a call. We introduce ToolRobustBench, a stage-wise diagnostic benchmark for tool-calling agents, where a tool-calling agent is an LLM system that selects a tool, supplies structured arguments, and interprets its returned feedback. ToolRobustBench aligns four perturbation families with the tool-use pipeline: tool-interface, user-intent, tool-output/observation, and runtime-environment perturbations. It attributes failures to tool selection, schema grounding, argument binding, tool-output/runtime-feedback handling, and E2E task success. Experiments on 15,456 single-family instances across 7 models, 16 sampled local tools, 4 perturbation families, and 14 subtypes show high but non-uniform clean performance and substantial robustness degradation, with tool-output/observation perturbation the dominant bottleneck. Mixed-family experiments reveal non-additive failure patterns that are not explained by isolated single-family results. Thus, ToolRobustBench provides a deterministic and cascade-aware benchmark for diagnosing robustness beyond clean tool-calling accuracy; 

---
# Macro-Operator Generation and Predicate Selection for TAMP Operator Learning 

**Authors**: Can Emir Bora, Emre Ugur  

**Link**: [PDF](https://arxiv.org/pdf/2608.23629)  

**Abstract**: Creating symbolic operators by hand is one of the main bottlenecks in deploying Task and Motion Planning systems (TAMP). Recent works show that these operators can instead be learned directly from demonstration data. Existing methods, however, typically learn each action in isolation and cannot capture the recurring multi-step structure of manipulation tasks, so the search becomes intractable on long sequential tasks. A further inefficiency arises in the symbolic state: every provided predicate is evaluated at every search node, even when it never appears in any learned operator. We present a system that addresses both problems together. Its central component is the automatic generation of macro-operators, composite actions that compress a recurring sequence of individual actions into a single planning step. Our system discovers causally linked action pairs directly from the training data, where one action produces exactly the condition that the next one requires, and turns each pair into a new operator. Alongside this, our system prunes every predicate that no learned operator references, which shrinks the symbolic state evaluated at each search node. Together, these changes shorten the effective planning horizon, and the benefit they bring grows with the length of the task. Across four TAMP domains, our method reaches up to a 4.6x planning speedup compared to the baseline method, namely Learning Operators for TAMP. More importantly, it solves a long sequential task that the baseline cannot solve. Macro-operator discovery thus not only accelerates planning but, in certain domains, determines solvability in practice. 

---
# When May an Agent Stop? Evidence-Carrying Termination for Tool-Using LLMs 

**Authors**: Jason Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.23623)  

**Abstract**: Tool-using agents must decide when to stop. Existing systems already gate terminal success, certify execution traces, or enforce runtime polici es, but do not test this particular receipt-, scope-, and closed-replay design at the COMPLETE boundary across controlled termination faults. W e instantiate and evaluate Evidence-Carrying Termination (ECT): an agent may return COMPLETE only when a typed certificate binds every required answer claim to valid, in-scope trace evidence and a deterministic replay reconstructs the claimed value. A locked static study crosses 48 ful ly synthetic tasks in six tool-use families with clean execution and eight faults. ECT produced 0/288 unsafe completions versus 252/288 for the inspected termination-critic core (difference -87.50 pp, 95% task-cluster interval [-87.50, -87.50] pp). A fresh, prespecified and frozen 576- trajectory study then compares ECT with the critic core, its faithful controller, and a full-trace LLM critic. On 22 primary held-out task clus ters, ECT produced 0/66 premature unsupported terminations versus 40/66 for the controller (difference -60.61 pp, 95% interval [-78.79, -40.91] pp), while supported completion was 97/132 versus 92/132 (difference 3.79 pp, interval [0.00, 9.09] pp), satisfying a -10-point noninferiority margin. ECT executed successful recovery in 18/66 trajectories, of which 17 subsequently completed with support; all three closed-loop gates p assed. ECT certifies support in a recorded trace under declared assumptions, not external truth, safety, or alignment. 

---
# Identifying Latent Declarative Representations of Code for Assisting Repository Migration 

**Authors**: Shraddha Surana, Ashwin Srinivasan, Michael Bain  

**Link**: [PDF](https://arxiv.org/pdf/2608.23619)  

**Abstract**: Legacy software repositories embed decades of domain knowledge in undocumented code, making understanding and modernization difficult. We treat a program as the implementation of an unobserved, declarative description of its computation and investigate whether making this latent declarative representation explicit improves repository-scale porting. ADFD-Migrate approximates the latent representation with an annotated data-flow diagram (ADFD) of processes, data stores, external entities, flows, and behavioral contracts. An LLM infers the source ADFD from bounded repository context, guided by static-analysis coverage checks. Dependency-aware chunking orders bounded process groups for target-language generation. Differences between the source ADFD and a statically recovered target ADFD then guide regeneration. We evaluate ADFD-Migrate on f2x50, a new benchmark of 50~Fortran repositories spanning 1.5k--1.6M lines of code and three complexity tiers, and assess the resulting ports along two dimensions: porting soundness, measured by source-oracle behavioral agreement, and porting completeness, measured by a composite migration outcome index. Against 382 curated Fortran-oracle probes, the generated Python passes 327 (85.6\%), with 40 repositories passing every attempted probe. ADFD-Migrate exposes all 382 planned behaviors as runnable targets, compared with 99 and 98 for direct and repository-context translation and 69 and 30 for the static-profile and dependency-chunking ablations. It also achieves a 93.1\% mean migration outcome index and a 17--59 percentage-point outcome-index advantage over direct translation on 47 repositories. These results suggest that an inspectable semantic bottleneck can improve the coverage and integration of repository-scale migration while enabling lower-cost generation for many repositories. 

---
# Rebuild Dossier: Mechanically-Enforced Specs for Agentic App Rebuilds, and What Model-Tier Failures Reveal 

**Authors**: Parker Fawcett  

**Link**: [PDF](https://arxiv.org/pdf/2608.23616)  

**Abstract**: An AI agent's rebuild is only as good as the process that produced it. Prior work found that once a model is strong enough, a multi-agent rebuild pipeline loses to the simplest approach: giving the model the original code and one instruction (AgentModernize). We present rebuild-dossier, an open-source tool that locks an application's real interface - its exact inputs and outputs - before any code is written, then enforces one-test-at-a-time building through automated checks, not written instructions alone.
Three results shape this evaluation, with differing amounts of evidence. First, in a small comparison, the compliant agent failed a held-back test while the rule-breaking agent passed everything - proof that a passing suite doesn't certify correctness when tests can be gamed. Second, we tested whether this beats simply giving the weaker model the source and one instruction: tied on a small app, but lost outright on a larger one where the automated check wasn't even running - pointing to the check mechanism, not interface-locking, which held up separately. Third, every claim here is checked at three levels - the agent's own report, an automated log, and the actual files produced - catching real errors, including a bug in our own logging code, that a single level would have missed.
These risks reproduce on a different model and toolchain: a stronger model followed our process three times running, something the weaker model never managed. The tool is public, MIT licensed, and reproduces end to end against our own applications. 

---
# REFINE: A Multi-Agent LLM Approach for Evidence-Guided Code Refactoring 

**Authors**: Muhammad Waseem, Aakash Ahmad, Pekka Abrahamsson  

**Link**: [PDF](https://arxiv.org/pdf/2608.23611)  

**Abstract**: Large Language Models (LLMs) offer new opportunities for automated code refactoring. However, generated changes must reduce targeted quality problems without introducing new issues or altering behaviour-relevant code structures. We introduce REFINE (Refactoring with Evidence-aware Flow for Integrated ageNtic Execution), a tool-agnostic, evidence-aware multi-agent approach for generating Java file-level refactoring candidates. REFINE combines static-analysis-guided smell identification, smell-informed planning, LLM-based transformation, automated re-analysis, preservation checks, and structured reporting.
We evaluate REFINE on 450 Java files from 15 open-source systems, producing 1,350 model-pass outputs using OpenAI GPT-5.5, Google Gemini 3.1 Pro Preview, and Anthropic Claude Opus 4.8. REFINE reduces detected code smells by 68.26%, 72.79%, and 68.49% across the three configurations, respectively, with the strongest reductions observed for major smells. A matched 150-file direct-prompt baseline shows that REFINE achieves a higher median code-smell reduction with smaller edits and fewer public-method removals. However, broader quality improvements are inconsistent, and preservation checks reveal residual risks, including assert/fail-call changes and public-method removal. Therefore, REFINE outputs should be treated as refactoring candidates requiring compilation, testing, dependency analysis, and human review before adoption in repository- or system-level settings. 

---
# Fidelity Preference, Not Demographic Preference: A Pixel-Level Attribute-Sensitivity Audit of Image Aesthetic/Preference Scorers 

**Authors**: Mingyang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.23593)  

**Abstract**: Text-to-image systems use learned aesthetic scorers to filter training data and guide generation, but whether these scores encode demographic attributes as objective quality is unclear. We audit four scorers (LAION-Aesthetics, PickScore, ImageReward, HPSv2) using pixel-level interventions on skin tone and body type in synthetic and real images. Our key finding is that along skin-lightness, the dominant effect is fidelity preference: unaltered images score highest, and perturbations in either direction are penalized (inverted-U). Placebo arms show this penalty is not an artifact of the skin operator, as applying the same CIELAB L* shift to non-skin regions yields similar penalty magnitudes. However, the penalty is operator-dependent and holds for all operators only for LAION-Aes. Critically, audits on synthetic images alone are misleading: LAION-Aes shows strong preference for darker skin on synthetic faces, but on 1470 real faces the preference reverses and becomes much smaller, and amplification becomes non-significant. Across scorers, synthetic results do not transfer -- reversing for LAION-Aes and HPSv2, attenuating for PickScore. We contribute a reproducible benchmark with artifact control and synthetic/real cross-validation, and an auditability criterion for pixel-level causal isolation (valid for skin tone, not for body type due to deformation). Population-stratified analysis shows fidelity-penalty asymmetry is not robust across groups after FDR correction except for HPSv2. Our findings show naive synthetic audits misjudge bias direction and magnitude, and only within-image causal isolation on real data can distinguish true demographic bias from fidelity preference. 

---
# A Human-Factors Guided Cognitive Model of Visuospatial Complexity in Embodied Active Vision 

**Authors**: Vasiliki Kondyli, Jakob Suchan, Mehul Bhatt  

**Link**: [PDF](https://arxiv.org/pdf/2608.23572)  

**Abstract**: We propose a novel framework for the analysis of multimodal data -- encompassing visual, auditory, and spatial stimuli -- foregrounding the role of complexity in embodied perception and interaction in dynamic, naturalistic settings. Grounded in theories of embodied cognition and active vision, we argue that embodied perceptual complexity emerges from an agent's dynamic engagement with the environment and must be analyzed holistically, as a combination of qualitative and quantitative attributes pertaining to, for instance, visuospatial and auditory features. Building on previous work on visual complexity, we expand this into a categorization of diverse complexity attributes -- quantitative, structural, dynamic, auditory, and interactional -- that together characterize multimodal complexity. We demonstrate how this model provides a theoretical framework for characterizing aspects of visuospatial complexity and their interactions, specifically in the context of everyday driving. We also discuss practical applications of the proposed model for creating and evaluating benchmark datasets (e.g., in driving) that centralize cognitive human factors, as well as applications aimed at systematically investigating the effect of visuospatial complexity on human active vision from the viewpoint of visual perception research. The proposed framework lays the foundation for automated methods that interpret complexity in 3D dynamic environments from a human-centered perspective, serving as a semantic template for explainable computational analysis of visuospatial complexity with a categorical focus on cognitive human factors. 

---
# Progressively Learning Heterogeneous Skills in a Unified Latent Space 

**Authors**: Yue-Yi Zhang, Ming Gong, Linpu He, Wei-Shi Zheng, Zhilin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2608.23258)  

**Abstract**: We propose HetSkills, a novel framework designed to progressively learn heterogeneous skills within a unified latent space for physics-based character control. The core idea is to treat this latent space as a shared executable interface, enabling seamless integration of skills learned from diverse data sources, supervision forms, and tasks. HetSkills begins by learning a tracking skill that establishes a strong foundation in motion control and creates a shared motion decoder, which can be reused across tasks without the need for retraining or separate controllers. To prevent the text-to-motion skill from exploiting shortcut pathways instead of learning language semantics, we introduce motion intuition distillation to ground text-to-motion generation in language semantics and a task-guidance module that dynamically adjusts actions based on high-level language instructions. This enables HetSkills to preserve natural motion while continuously expanding its skill repertoire, making it highly adaptable for long-horizon tasks. Experimental results demonstrate the effectiveness in motion tracking, text-to-motion generation, motion completion, and downstream task adaptation, achieving impressive success rates even under challenging conditions. 

---
