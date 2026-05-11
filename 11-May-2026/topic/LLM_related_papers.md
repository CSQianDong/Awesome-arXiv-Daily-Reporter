# AgentEscapeBench: Evaluating Out-of-Domain Tool-Grounded Reasoning in LLM Agents 

**Authors**: Zhengkang Guo, Yiyang Li, Lin Qiu, Xiaohua Wang, Jingwen Xv, Dongyu Ru, Xiaoyu Li, Xiaoqing Zheng, Xuezhi Cao, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2605.07926)  

**Abstract**: As LLM-based agents increasingly rely on external tools, it is important to evaluate their ability to sustain tool-grounded reasoning beyond familiar workflows and short-range interactions. We introduce AgentEscapeBench, an escape-room-style benchmark that tests whether agents can infer, execute, and revise novel tool-use procedures under explicit long-range dependency constraints. Each task defines a directed acyclic dependency graph over tools and items, requiring agents to invoke real external functions, track hidden state revealed incrementally, propagate intermediate results, and submit a deterministically verifiable final answer. AgentEscapeBench includes 270 instances across five difficulty tiers and supports fully automated evaluation. Experiments with sixteen LLM agents and human participants show that performance drops sharply as dependency depth increases: humans decline from 98.3% success at difficulty-5 to 80.0% at difficulty-25, while the best model drops from 90.0% to 60.0%. Trajectory analysis attributes model failures mainly to breakdowns in long-range state tracking, clue adherence, and intermediate-result propagation. These findings suggest that current agents can often handle local tool use but still struggle with deep contextual dependencies. We hope AgentEscapeBench can serve as a diagnostic testbed for measuring current agent capabilities and informing future training efforts toward more robust general-purpose reasoning, action, and adaptation. 

---
# VecCISC: Improving Confidence-Informed Self-Consistency with Reasoning Trace Clustering and Candidate Answer Selection 

**Authors**: James Petullo, Sonny George, Dylan Cashman, Nianwen Xue  

**Link**: [PDF](https://arxiv.org/pdf/2605.08070)  

**Abstract**: A standard technique for scaling inference-time reasoning is Self-Consistency, whereby multiple candidate answers are sampled from an LLM and the most common answer is selected. More recently, it has been shown that weighted majority voting (e.g. Confidence-Informed Self Consistency (CISC)), which assigns a confidence value to each candidate answer and chooses the answer with the largest accumulated score, tends to be more accurate on a wide range of popular benchmarks. In practice, weighted majority voting necessitates calling a critic LLM on each candidate's reasoning trace to produce the answer's confidence score. This secondary series of LLM calls greatly increases the overhead and cost of weighted majority voting, despite its potential performance benefits. To reduce this expense, we propose VecCISC, a lightweight, adaptive framework that uses a measure of semantic similarity to filter reasoning traces that are semantically equivalent to others, degenerate, or hallucinated, thus decreasing the number of candidate answers that must be evaluated by the critic. To ensure adequate experimental thoroughness, we evaluate VecCISC on five challenging, widely-adopted datasets spanning the domains of mathematics, chemistry, biology, commonsense reasoning, and the humanities. Our results demonstrate that VecCISC reduces the total token usage by 47%, while maintaining or exceeding the accuracy of CISC. 

---
# Abductive Reasoning with Probabilistic Commonsense 

**Authors**: Joseph Cotnareanu, Chiara Roverato, Han Zhou, Didier Chetelat, Yingxue Zhang, Mark Coates  

**Link**: [PDF](https://arxiv.org/pdf/2605.08011)  

**Abstract**: Recent efforts to improve the reasoning abilities of Large Language Models (LLMs) have focused on integrating formal logic solvers within neurosymbolic frameworks. A key challenge is that formal solvers lack commonsense world knowledge, preventing them from making reasoning steps that humans find obvious. Prior methods address this by using LLMs to supply missing commonsense assumptions, but these approaches implicitly assume universal agreement on such commonsense facts. In reality, commonsense beliefs vary across individuals. We propose a probabilistic framework for abductive commonsense reasoning that explicitly models this variation, aiming to determine whether most people would judge a statement as true or false. We introduce Probabilistic Abductive CommonSense (PACS), a novel algorithm that uses an LLM and a formal solver to sample proofs as observations of individuals' distinct commonsense beliefs, and aggregates conclusions across these samples. Empirically, PACS outperforms chain-of-thought reasoning, prior neurosymbolic methods, and search-based approaches across multiple benchmarks. 

---
# Reason to Play: Behavioral and Brain Alignment Between Frontier LRMs and Human Game Learners 

**Authors**: Botos Csaba, Sreejan Kumar, Austin Tudor David Andrews, Laurence Hunt, Chris Summerfield, Joshua B. Tenenbaum, Rui Ponte Costa, Marcelo G. Mattar, Momchil Tomov  

**Link**: [PDF](https://arxiv.org/pdf/2605.08019)  

**Abstract**: Humans rapidly learn abstract knowledge when encountering novel environments and flexibly deploy this knowledge to guide efficient and intelligent action. Can modern AI systems learn and plan in a similar way? We study this question using a dataset of complex human gameplay with concurrent fMRI recordings, in which participants learn novel video games that require rule discovery, hypothesis revision, and multi-step planning. We jointly evaluate models by their ability to play the games, match human learning behavior, and predict brain activity during the same task, comparing a suite of frontier Large Reasoning Models (LRMs) against model-free and model-based deep reinforcement learning agents and a Bayesian theory-based agent. We find that frontier LRMs most closely match human behavioral patterns during game discovery and predict brain activity an order of magnitude better than both reinforcement learning alternatives across cortical and subcortical regions, with effects robust to permutation controls. Through targeted manipulations, we further show that brain alignment reflects the model's in-context representation of the game state rather than its downstream planning or reasoning. Our results establish LRMs as compelling computational accounts of human learning and decision making in complex, naturalistic environments. Project page with interactive replays: this https URL 

---
# Rubric-Grounded RL: Structured Judge Rewards for Generalizable Reasoning 

**Authors**: Manish Bhattarai, Ismael Boureima, Nishath Rajiv Ranasinghe, Scott Pakin, Dan O'Malley  

**Link**: [PDF](https://arxiv.org/pdf/2605.08061)  

**Abstract**: We argue that decomposing reward into weighted, verifiable criteria and using an LLM judge to score them provides a partial-credit optimization signal: instead of a binary outcome or a single holistic score, each response is graded along multiple task-specific criteria. We formalize \emph{rubric-grounded reinforcement learning (RL)}: a framework in which the policy is optimized against a structured, multi-criterion reward produced by a frozen LLM judge that conditions on auxiliary grounding the policy never sees. We instantiate the framework by deriving rubrics from an Office of Scientific and Technical Information (OSTI)-derived corpus of roughly 100,000 scientific and technical documents and training Llama-3.1-8B-Instruct with Group Relative Policy Optimization (GRPO). With GRPO-based training, the model achieves $71.7\%$ normalized reward on held-out rubric evaluation. The GRPO-tuned policy also improves over the base model on four reasoning benchmarks not derived from the training corpus -- GSM8K, MATH, GPQA Main, and GPQA Diamond. These results provide evidence that structured, document-grounded rewards can improve held-out rubric performance and induce transferable reasoning behaviors beyond the corpus used to construct the training environment. 

---
# TraceFix: Repairing Agent Coordination Protocols with TLA+ Counterexamples 

**Authors**: Shuren Xia, Qiwei Li, Taqiya Ehsan, Jorge Ortiz  

**Link**: [PDF](https://arxiv.org/pdf/2605.07935)  

**Abstract**: We present TraceFix, a verification-first pipeline for Large Language Model (LLM) multi-agent coordination. An agent synthesizes a protocol topology as a structured intermediate representation (IR) from a task description, generates PlusCal coordination logic, and iteratively repairs the protocol using counterexamples from the TLA+ model checker (TLC) until verification succeeds. Verified process bodies are compiled into per-agent system prompts and executed under a runtime monitor that rejects out-of-topology coordination operations. On 48 tasks spanning 16 scenario families, all tasks reach full TLC verification; 62.5% pass on the first attempt and none requires more than four repair iterations. State spaces span six orders of magnitude yet verification completes in under 60 s for every task. A 3,456-run runtime comparison shows that topology-monitored execution achieves the highest task completion (89.4% average, 81.5% full) and that runtimes using the verified protocol degrade at roughly half the rate of prompt-only and chat-only baselines when model capability is reduced. A paired ablation under a fixed runtime shows that TLC-verified protocols cut deadlock/livelock (DL/LL) from 31.1% to 14.1%, with the largest separation under fault injection. 

---
# Hierarchical Task Network Planning with LLM-Generated Heuristics 

**Authors**: Felipe Meneguzzi, Alexandre Buchweitz, Augusto B. Corrêa, Victor Scherer Putrich, André Grahl Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2605.07707)  

**Abstract**: HTN planning is a variation of classical planning where, instead of searching for a linear sequence of actions, an algorithm decomposes higher-level tasks using a method library until only executable actions remain. On one hand, this allows one to introduce domain knowledge that can speed up the search for a solution through the method library. On the other hand, it creates challenges that go beyond those of classical state-space search. While recent research produced a number of heuristics and novel algorithms that speed up HTN planning, these heuristics are not yet as informative as those available in classical planning algorithms. We investigate whether large language models (LLMs) can generate effective search heuristics for HTN planning, extending the methodology of Corrêa, Pereira, and Seipp (2025) from classical to hierarchical planning. Using the Pytrich planner on six standard total-order HTN benchmark domains, we evaluate heuristics generated by nine LLMs under domain-specific prompting and compare them against the TDG and LMCount domain-independent baselines and the PANDA planner. Our results show that LLM-generated heuristics nearly match the coverage of the best available HTN planner, while substantially reducing search effort on 83% of shared problems. 

---
# GASim: A Graph-Accelerated Hybrid Framework for Social Simulation 

**Authors**: Xuan Zhou, Yanhui Sun, Hantao Yao, Allen He, Yongdong Zhang, Wu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07692)  

**Abstract**: Large-scale social simulators are essential for studying complex social patterns. Prior work explores hybrid methods to scale up simulations, combining large language models (LLM)-based agents with numerical agent-based models (ABM). However, this incurs high latency due to expensive memory retrieval and sequential ABM execution. To address this challenge, we propose GASim, a graph-accelerated hybrid multi-agent framework for large-scale social simulations. For core agents driven by LLM, GASim introduces Graph-Optimized Memory (GOM) to replace intensive LLM-based retrieval pipelines with lightweight propagation over a sparse memory graph. For the majority of ordinary agents, GASim employs Graph Message Passing (GMP), substituting sequential ABM execution with parallel updates by fine-grained feature aggregation and Graph Attention Network. We further introduce Entropy-Driven Grouping (EDG) that coordinates this hybrid partitioning, leveraging information entropy to dynamically identify emergent core agents situated in information-diverse neighborhoods. Extensive experiments show that GASim not only delivers a substantial 9.94-fold end-to-end speedup over the traditional hybrid framework but also consumes less than 20% of baseline tokens, significantly reducing costs while preserving strong alignment with real-world public opinion trends. Our code is available at this https URL. 

---
# RuleSafe-VL: Evaluating Rule-Conditioned Decision Reasoning in Vision-Language Content Moderation 

**Authors**: Zhifeng Lu, Dianyuan Wang, Yuhu Shang, Zhenbo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07760)  

**Abstract**: Platform content moderation applies explicit policy rules and context-dependent conditions to decide whether user content is allowed, restricted, or removed. A correct moderation outcome must therefore depend on which rules a case activates, how those rules interact, and whether the available evidence is sufficient. Current multimodal safety benchmarks largely reduce moderation to matching predefined final labels, leaving this underlying rule structure untested. As a result, a high benchmark score reveals little about whether a model applies the policy correctly or arrives at the correct label through superficial cues. To evaluate this rule-governed process, we introduce RuleSafe-VL, a benchmark for rule-conditioned decision reasoning in vision-language content moderation. Derived from publicly available platform moderation policies, RuleSafe-VL formalizes 93 atomic rules and 92 typed rule relations, yielding 2,166 context-sensitive image-text cases across three high-risk policy families. Its four diagnostic tasks decompose moderation into a rule-conditioned decision chain. They identify activated rules, recover rule interactions, judge decision sufficiency, and resolve outcomes once missing context is supplied. Experiments on 10 frontier, open-source, and safety-oriented VLMs reveal rule-relation recovery as the dominant bottleneck, where the best model reaches only 64.8 Macro-F1 and some safety-oriented models fall below 7 Macro-F1. Decision-state prediction also remains unreliable, peaking at 64.5 Macro-F1. RuleSafe-VL shifts moderation evaluation from final-label scoring toward diagnostic assessment of rule-conditioned decision reasoning. 

---
# FactoryBench: Evaluating Industrial Machine Understanding 

**Authors**: Yanis Merzouki, Coral Izquierdo, Matei Ignuta-Ciuncanu, Marcos Gomez-Bracamonte, Riccardo Maggioni, Alessandro Lombardi, Camilla Mazzoleni, Federico Martelli, Balazs Gunther, Jonas Petersen, Philipp Petersen  

**Link**: [PDF](https://arxiv.org/pdf/2605.07675)  

**Abstract**: We introduce FactoryBench, a benchmark for evaluating time-series models and LLMs on machine understanding over industrial robotic telemetry. Q&A pairs are organized along four causal levels (state, intervention, counterfactual, decision) instantiating Pearl's ladder of causation, and span five answer formats: four structured formats are scored deterministically and free-form answers are scored by an LLM-as-judge voting protocol. We propose a scalable Q&A generation framework built around structured question templates, present FactoryWave (a dense, multitask, multivariate sensor dataset collected from a UR3 cobot and a KUKA KR10 industrial arm), and construct FactoryBench as a large-scale benchmark of over 70k Q&A items grounded in roughly 15k normalized episodes from FactoryWave, AURSAD, and voraus-AD. Zero-shot evaluation of six frontier LLMs shows that no model exceeds 50% on structured levels or 18% on decision-making, revealing a wide gap between current models and operational machine understanding. 

---
# Open-Ended Task Discovery via Bayesian Optimization 

**Authors**: Masaki Adachi, Yuta Suzuki, Juliusz Ziomek  

**Link**: [PDF](https://arxiv.org/pdf/2605.07572)  

**Abstract**: When applying Bayesian optimization (BO) to scientific workflow, a major yet often overlooked source of uncertainty is the task itself -- namely, what to optimize and how to evaluate it -- which can evolve as evidence accumulates. We introduce Generate-Select-Refine (GSR), a open-ended BO framework that alternates between task generation and task optimization. Starting from a user-provided seed task, GSR generates new tasks in a coarse-to-fine manner while a task-acquisition function schedules optimization. Asymptotically, it concentrates evaluations on the best task, incurring only logarithmic regret overhead relative to single-task BO. We apply GSR to new product development, chemical synthesis scaling, algorithm analysis, and patent repurposing, where it outperforms existing LLM-based optimizers. 

---
# Confidence-Aware Alignment Makes Reasoning LLMs More Reliable 

**Authors**: Kejia Chen, Jiawen Zhang, Yihong Wu, Kewei Gao, Jian Lou, Zunlei Feng, Mingli Song, Ruoxi Jia  

**Link**: [PDF](https://arxiv.org/pdf/2605.07353)  

**Abstract**: Large reasoning models often reach correct answers through flawed intermediate steps, creating a gap between final accuracy and reasoning reliability. Existing alignment strategies address this with external verifiers or massive sampling, limiting scalability. In this work, we introduce CASPO (Confidence-Aware Step-wise Preference Optimization), a framework that aligns token-level confidence with step-wise logical correctness through iterative Direct Preference Optimization, without training a separate reward model. During inference, we propose Confidence-aware Thought (CaT), which leverages this calibrated confidence to dynamically prune uncertain reasoning branches with negligible O(V) latency. Experiments across ten benchmarks and multiple model families show that CASPO consistently improves reasoning reliability and inference efficiency. CASPO scales to Qwen3-8B-Base and surpasses tree-search baselines on AIME'24 and AIME'25 without using reward-model data. We also release a step-wise dataset with confidence annotations to support fine-grained analysis of reasoning reliability. Code is available at this https URL. 

---
# Tools as Continuous Flow for Evolving Agentic Reasoning 

**Authors**: Tairan Huang, Siyu Shang, Qiang Chen, Xiu Su, Yi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.07339)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in orchestrating tools for reasoning tasks. However, existing methods rely on a step-wise paradigm that lacks a global perspective, which causes error accumulation over long horizons and restricts generalization to unseen tools. To overcome these limitations, we propose Tools as Continuous Flow for Evolving Agentic Reasoning (FlowAgent), which reconceptualizes tool chaining as continuous trajectory generation within a semantic space. To systematically evaluate this paradigm, we introduce the first plan-level closed-loop benchmark dedicated to plan-level agentic reasoning in dynamic real-world environments. Specifically, the proposed FlowAgent leverages conditional flow matching to generate continuous latent trajectories, providing a global planning perspective to ensure coherent and robust tool execution. Theoretically, we establish formal bounds on utility convergence and prove that our continuous formulation fundamentally guarantees robust generalization and error attenuation. Empirical evaluations show that FlowAgent achieves superior robustness and adaptability in long-horizon reasoning tasks. 

---
# Discovering Ordinary Differential Equations with LLM-Based Qualitative and Quantitative Evaluation 

**Authors**: Sum Kyun Song, Bong Gyun Shin, Jae Yong Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.07323)  

**Abstract**: Discovering governing differential equations from observational data is a fundamental challenge in scientific machine learning. Existing symbolic regression approaches rely primarily on quantitative metrics; however, real-world differential equation modeling also requires incorporating domain knowledge to ensure physical plausibility. To address this gap, we propose DoLQ, a method for discovering ordinary differential equations with LLM-based qualitative and quantitative evaluation. DoLQ employs a multi-agent architecture: a Sampler Agent proposes dynamic system candidates, a Parameter Optimizer refines equations for accuracy, and a Scientist Agent leverages an LLM to conduct both qualitative and quantitative evaluations and synthesize their results to iteratively guide the search. Experiments on multi-dimensional ordinary differential equation benchmarks demonstrate that DoLQ achieves superior performance compared to existing methods, not only attaining higher success rates but also more accurately recovering the correct symbolic terms of ground truth equations. Our code is available at this https URL. 

---
# Implicit Compression Regularization: Concise Reasoning via Internal Shorter Distributions in RL Post-Training 

**Authors**: Chen Wang, Hexuan Deng, Yining Zhang, Yuchen Zhang, Jionghao Bai, Zhaochun Li, Ge Lan, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07316)  

**Abstract**: Reinforcement learning with verifiable rewards improves LLM reasoning but often induces overthinking, where models generate unnecessarily long reasoning traces. Existing methods mainly rely on length penalties or early-exit strategies; however, the former may degrade accuracy and induce underthinking, whereas the latter assumes that substantial portions of reasoning traces can be safely truncated. To obtain a compression signal without these limitations, we revisit the training dynamics of existing compression methods. We observe that the length--accuracy correlation is initially negative but continually increases during compression, indicating that shorter responses are initially more likely to be correct but gradually lose this property as the policy moves toward underthinking. Based on this observation, we formalize overthinking: a negative correlation indicates an overthinking regime, while a positive one indicates underthinking. When overthinking, the shortest correct responses are shorter than the group-average response length in expectation, making them natural compression targets already present in on-policy rollouts. We therefore propose \emph{Implicit Compression Regularization} (ICR), an on-policy regularization method whose compression signal comes from a virtual shorter distribution induced by the shortest correct responses in rollout groups, guiding the policy toward concise yet correct trajectories. Training dynamics show that ICR maintains a better length--accuracy correlation during compression, indicating that short responses remain better aligned with correctness instead of drifting toward underthinking. Experiments on three reasoning backbones and multiple mathematical and knowledge-intensive benchmarks show that ICR consistently shortens responses while preserving or improving accuracy, achieving a stronger accuracy--length Pareto frontier. 

---
# GraphReAct: Reasoning and Acting for Multi-step Graph Inference 

**Authors**: Xingtong Yu, Zhongwei Kuai, Chang Zhou, Xuanting Xie, Renhe Jiang, Xikun Zhang, Hong Cheng, Xinming Zhang, Yuan Fang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07357)  

**Abstract**: Reasoning-acting frameworks enhance large language models (LLMs) by interleaving reasoning with actions for dynamic information acquisition. However, extending this paradigm to graph learning remains underexplored. Graph data is inherently structured, with information distributed across nodes and edges and encoded through both topology and latent representations. As a result, effective reasoning over graphs requires not only retrieving informative evidence from the graph, but also progressively refining the accumulated context during multi-step inference. In this work, we propose GraphReAct, a graph reasoning-acting framework that enables step-by-step inference over graph-structured data. Specifically, we design a graph-based action space with two complementary retrieval actions: topological retrieval, which captures local structural dependencies, and semantic retrieval, which accesses non-local but relevant evidence in the representation space. These actions dynamically expand the reasoning context. To further support multi-step reasoning, we introduce another type of action, context refinement, which distills and reorganizes accumulated information into a compact representation. By interleaving reasoning with both retrieval and refinement actions, our framework enables a progressive transition from context expansion to compression. Extensive experiments on six benchmark datasets demonstrate that GraphReAct consistently outperforms state-of-the-art methods, validating the effectiveness of reasoning-acting for graph learning. 

---
# Can Agents Price a Reaction? Evaluating LLMs on Chemical Cost Reasoning 

**Authors**: Yuyang Wu, Yue Huang, Shuaike Shen, Xujian Wang, Shuhao Zhang, Qiyao Xue, Weichen Liu, Runtian Gao, Jian Ma, Xiangliang Zhang, Olexandr Isayev  

**Link**: [PDF](https://arxiv.org/pdf/2605.07251)  

**Abstract**: Large Language Models (LLMs) have become increasingly capable as tool-using agents, with benchmarks spanning diverse general agentic tasks. Yet rigorous evaluation of scientific tool use remains limited. In chemistry, recent agents can plan syntheses and invoke domain-specific tools, but evaluations often rely on curated demonstrations, expert assessment, or LLM-as-judge scoring rather than exact, judge-free ground truth. We address this gap with chemical procurement cost estimation, a practical task in which an agent must ground chemical identities, retrieve supplier quotes, select valid purchasable packs, normalize quantities, and compute cost from a reaction description. We introduce ChemCost, a benchmark of 1,427 evaluable reactions grounded to a frozen pricing snapshot covering 2,261 chemicals and 230,775 supplier quotes, supporting scalar scoring and stage-level diagnosis of grounding, retrieval, procurement, and arithmetic failures. To evaluate robustness, we further construct controlled noise-injected views that perturb chemical aliases, quantity expressions, missing fields, and input formatting. Experiments with frontier, open-weight, and chemistry-specialized LLM agents show that tool access is necessary but insufficient for solving the task. The strongest agents reach only 50.6% accuracy within 25% relative error on clean inputs and degrade substantially with realistic noise. Stage-level analysis further shows that failures arise from brittle parsing, ineffective evidence integration, invalid pack selection, and non-convergent tool use. 

---
# Efficient Data Selection for Multimodal Models via Incremental Optimization Utility 

**Authors**: Jinhao Jing, Qiannian Zhao, Chao Huang, Zhan Su  

**Link**: [PDF](https://arxiv.org/pdf/2605.07488)  

**Abstract**: The scaling of Large Multimodal Models (LMMs) is constrained by the quality-quantity trade-off inherent in synthetic data. Previous approaches, such as LLM-as-a-Judge, have proven their effectiveness in addressing this but suffer from prohibitive computational costs and lack of interpretability. To bridge this gap, we propose One-Step-Train (OST), a framework that reformulates data selection as an incremental optimization utility ranking problem. Instead of relying on semantic heuristics, OST estimates the marginal utility of each sample via a simulated single-step update on a lightweight proxy. Experiments on the Qwen series across multimodal mathematical reasoning benchmarks demonstrate that OST achieves Pareto-optimal efficiency. By selecting the top-50 subset, OST reduces training costs by 43% (and total time consumption by 17) while surpassing the strong LLM-as-a-Judge baseline by 1.8 points. Furthermore, under a fixed compute budget, our method using only the top-20 subset achieves a 5.6 point gain over LLM-as-a-Judge, improves upon heuristic scoring baselines like DEITA, and outperforms the Full-SFT baseline by 8.8 points. Notably, while Full-SFT suffers from performance degradation due to noise, our optimization-grounded approach effectively identifies toxic samples, successfully reversing the negative transfer frequently observed in complex reasoning tasks. 

---
# HMACE: Heterogeneous Multi-Agent Collaborative Evolution for Combinatorial Optimization 

**Authors**: Yuping Yan, Jirui Han, Fei Ming, Yuanshuai Li, Yaochu Jin  

**Link**: [PDF](https://arxiv.org/pdf/2605.07214)  

**Abstract**: Large Language Models have recently emerged as a promising paradigm for automated heuristic design for NP-hard combinatorial optimization problems. Despite this progress, existing LLM-based methods typically rely on monolithic workflows constrained by rigid templates, thereby restricting memory-guided exploration and triggering premature convergence to local optima. To design an autonomous and collaborative architecture, we introduce HMACE, a Heterogeneous Multi-Agent Collaborative Evolution framework that reconceptualizes heuristic search as an organizational design problem. HMACE decomposes each evolutionary generation into an autonomous, role-specialized loop with four coordinated agents: a Proposer for strategy exploration, a Generator for executable heuristic synthesis, an Evaluator for empirical assessment, and a Reflector for archive-backed memory update. By coupling behavior-aware retrieval, lightweight candidate filtering, and fitness-grounded archive updates, HMACE guides the search toward diverse and promising heuristic behaviors while avoiding redundant evaluations. Extensive evaluations on representative COPs, including TSP, Online BPP, MKP, and PFSP, show that HMACE achieves a favorable quality-efficiency trade-off compared to state-of-the-art single-agent and multi-agent baselines. In the matched LLM-driven reference comparison, HMACE achieves the lowest average gaps on TSP and Online BPP (0.464\% and 0.223\%, respectively), while requiring only 0.13M and 0.42M tokens for the two tasks, substantially fewer than the compared baselines. 

---
# Signal Reshaping for GRPO in Weak-Feedback Agentic Code Repair 

**Authors**: Jia Li, Yuxin Su, Ting Peng, Hailiang Huang, Yuetang Deng, Michael R. Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07276)  

**Abstract**: Code-agent RL often receives weak feedback: rollout-time signals are reliable and executable, but capture only necessary or surface conditions for task success rather than the target semantic predicate. Using agentic compile-fix as the setting, we study signal reshaping for standard GRPO under such feedback. Our central claim is that GRPO's within-group comparison is meaningful only after three kinds of signals are reshaped: outcome rewards recover semantic ranking, process signals localize intra-trajectory credit, and rollouts from the same prompt remain execution-comparable. We operationalize these conditions with a minimal signal-reshaping construction that leaves GRPO's group-normalized advantage construction unchanged: compile-and-semantic layered rewards reshape trajectory ranking, step-level process scores outside group reward normalization reshape within-trajectory update strength, and failure-cause-aware rollout governance reshapes within-group comparability. Experiments show a clear end-to-end gain: full signal-reshaped GRPO improves strict compile-and-semantic accuracy from the base model's zero-shot $0.385$ to $0.535$. Controlled comparisons further explain the source of this gain: binary rewards remove the compile-only middle tier and degrade trajectory control; on top of layered rewards, process-score weighting further improves accuracy from $0.48$ to $0.53$ and reduces average evaluation steps from $23.50$ to $17.02$. As a boundary comparison, privileged-prompt token-level distillation mainly optimizes local distributional alignment; in long tool-use trajectories, this signal is diluted by non-critical tokens and cannot replace outcome semantics, process credit, or within-group comparability. 

---
# EnvSimBench: A Benchmark for Evaluating and Improving LLM-Based Environment Simulation 

**Authors**: Yi Liu, TingFeng Hui, Wei Zhang, Li Sun, Ningxin Su, Jian Wang, Sen Su  

**Link**: [PDF](https://arxiv.org/pdf/2605.07247)  

**Abstract**: Scalable AI agents training relies on interactive environments that faithfully simulate the consequences of agent actions. Manually crafted environments are expensive to build, brittle to extend, and fundamentally limited in diversity. A promising direction is to replace manually crafted environments with LLM-simulated counterparts. However, this paradigm hinges on an unexamined core assumption: LLMs can accurately simulate environmental feedback. In practice, LLM-simulated environments suffer from hallucinations, logical inconsistencies, and silent state drift failures that corrupt agent reward signals and compound the construction costs that the paradigm was designed to eliminate. To address this gap, we propose EnvSimBench with four contributions: 1) We provide the first formal definition and operationalization of Environment Simulation Ability (EnvSim Ability) as a quantifiable research objective. 2) We construct EnvSimBench, a rigorous benchmark covering 400 samples across 167 diverse environments, equipped with verifiable labels and fine-grained difficulty stratification along three axes. 3) Systematic evaluations reveal that all state-of-the-art language models suffer from a universal state change cliff: they achieve near-perfect accuracy on tasks when the environment state remains invariant, yet fail catastrophically when multiple states need simultaneous updates. This finding exposes EnvSim Ability as a critical yet largely unaddressed capability gap. 4) We design a constraint-driven simulation pipeline that substantially reduces hallucination, boosts environment synthesis yield by 6.8%, and cuts costs by over 90%. Overall, EnvSimBench serves as both a diagnostic framework and a practical optimization path for reliable LLM-based environment simulation, establishing a foundation for scalable agent training. Code and data are available at this https URL 

---
# When Stored Evidence Stops Being Usable: Scale-Conditioned Evaluation of Agent Memory 

**Authors**: Jiaqi Shao, Yiyi Lu, Yunzhen Zhang, Bing Luo  

**Link**: [PDF](https://arxiv.org/pdf/2605.07313)  

**Abstract**: Memory-agent evaluations report fixed-snapshot accuracy or retrieval quality, but these scores do not show whether evidence remains usable as irrelevant sessions (sessions not annotated as task-relevant evidence for the query) accumulate. We present a scale-conditioned evaluation protocol for agent memory under evidence-preserving growth: for each query, task evidence is held fixed while irrelevant sessions are added. The protocol logs agent--memory trajectories and reports four diagnostics: budget-compliant reliability, tail memory-call burden, failure-regime decomposition, and the usable-scale boundary where reliability falls below the target. Applied to LongMemEval and LoCoMo across flat, planar, and hierarchical memory interfaces, the protocol shows reliability loss is not a single phenomenon. On LongMemEval, HippoRAG stays within the two-call budget but loses 16--20 percentage points in budget-compliant reliability as irrelevant sessions are added; LiCoMemory's observed failures depend strongly on the agent, with Qwen3-8B exceeding the budget while Qwen3-32B and Qwen3-235B remain reliable in the tested range. The result supports a framework for making scalable-memory claims conditional on agent, interface, scale range, and interaction budget. 

---
# Towards Autonomous Business Intelligence via Data-to-Insight Discovery Agent 

**Authors**: Dongming Wu, Junwen Li, Ming Lu, Gang Wang, Ting Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.07202)  

**Abstract**: Transforming fragmented enterprise data into actionable insights remains a significant challenge for LLMs, constrained by complex database schemas, limitations in dynamic SQL generation, and the need for deep multi-dimensional this http URL this paper, we propose AIDA(Autonomous Insight Discovery Agent), the first end-to-end framework designed for autonomous exploration in complex business environments. We establish a highly flexible instant retail environment encompassing 200+ metrics and 100+ dimensions, and integrates a proprietary Domain-Specific Language (DSL) that bridges semantic reasoning with precise SQL execution. Our reinforcement learning system subsequently formulates business analysis as a Pareto Principle-guided cumulative reasoning process. Experimental results demonstrate that AIDA significantly outperforms workflow-based agents, and extensive evaluations further reveal that AIDA achieves superior environmental perception and more in-depth analysis from diverse perspectives. Our work ultimately establishes the transformative potential of autonomous intelligence for industrial-scale business intelligence systems. 

---
# Structured Role-Aware Policy Optimization for Multimodal Reasoning 

**Authors**: Bingqing Jiang, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2605.07274)  

**Abstract**: Reinforcement learning from verifiable rewards (RLVR), especially with Group Relative Policy Optimization (GRPO), has shown strong potential for improving the reasoning capabilities of large vision-language models (LVLMs). However, in multimodal reasoning, final-answer rewards are typically assigned at the sequence level and do not distinguish the functional roles of different tokens, making it difficult to determine whether a correct answer is supported by task-relevant visual evidence. In this paper, we revisit multimodal RLVR from the perspective of role-aware token-level credit assignment, where structured responses are decomposed into perception tokens for extracting visual evidence and reasoning tokens for deriving answers from that evidence. Based on this perspective, we propose Structured Role-aware Policy Optimization (SRPO), which refines the sequence-level GRPO advantage into role-aware token-level advantages without changing the reward function. Specifically, SRPO assigns role-specific credit by using self-distilled on-policy contrasts: perception tokens are emphasized according to their visual dependency under original versus corrupted visual inputs, while reasoning tokens are emphasized according to their consistency with the generated perception. These role-specific signals are further unified through a shared trajectory-level baseline, yielding positive token weights that adjust relative update magnitudes while preserving the original GRPO reward and optimization direction, without requiring external reward models or separate teachers. Experiments across diverse multimodal reasoning benchmarks show that SRPO improves evidence-grounded reasoning, highlighting the importance of moving beyond uniform sequence-level credit toward role-aware optimization for reliable multimodal reasoning. 

---
# MEMOREPAIR: Barrier-First Cascade Repair in Agentic Memory 

**Authors**: Yang Zhao, Chengxiao Dai, Mengying Kou, Yue Xiu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07242)  

**Abstract**: Agentic memory evolves across tasks into durable derived artifacts: summaries, cached outputs, embeddings, learned skills, and executable tool procedures. When a source artifact is deleted, corrected, or invalidated by tool or API migration, descendants derived from that source can remain visible and steer future actions with stale support. We formalize this failure mode as the cascade update problem, where repair targets the visible derived state of the memory store. We present MemoRepair, a barrier-first cascade-repair contract for agentic memory. A repair event induces a controlled transition from invalidated descendant state to validated successor state: affected descendants are withdrawn before repair, successors are constructed from retained support and staged repaired predecessors under the current interface, and republication is restricted to validated predecessor-closed successors. This contract induces a scalarized repair-selection problem for a fixed repair-cost tradeoff. We show that the induced publication problem reduces to maximum-weight predecessor closure and can be solved exactly by a single s-t min-cut. Experiments on ToolBench and MemoryArena show that, with complete influence provenance, MemoRepair reduces invalidated-memory exposure from 69.8-94.3% under systems without cascade repair to 0%. Compared with exhaustive Repair all, it recovers 91.1-94.3% of validated successors while reducing normalized repair-operator cost from 1.00 to 0.57-0.76. 

---
# SOM: Structured Opponent Modeling for LLM-based Agents via Structural Causal Model 

**Authors**: Shiyue Cao, Pei Xu, Likun Yang, Lei Cui, Xiaotang Chen, Kaiqi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07301)  

**Abstract**: Accurately predicting opponents' behavior from interactions is a fundamental capability for large language model (LLM)-based agents in multi-agent and game-theoretic environments. Existing approaches often entangle opponent modeling with prediction, relying on implicit contextual reasoning and limiting adaptability in dynamic interactions. To this end, we propose Structured Opponent Modeling (SOM), a two-stage opponent modeling framework that distinctly separates opponent model construction and opponent prediction. At the construction stage, SOM employs a Structural Causal Model (SCM), a graph-based formalism for representing dependencies among variables, to capture directed links between opponents' observations and actions, yielding an explicit and structured opponent representation. At the prediction stage, the LLM performs structured reasoning along clear pathways derived from the SCM, improving both prediction accuracy and stability. Extensive experiments on diverse multi-agent benchmarks demonstrate that SOM consistently outperforms state-of-the-art LLM-based reasoning baselines, enabling more accurate and adaptable strategic decision-making in complex and dynamic multi-agent interactions. 

---
# TeamBench: Evaluating Agent Coordination under Enforced Role Separation 

**Authors**: Yubin Kim, Chanwoo Park, Taehan Kim, Eugene Park, Samuel Schmidgall, Salman Rahman, Chunjong Park, Cynthia Breazeal, Xin Liu, Hamid Palangi, Hae Won Park, Daniel McDuff  

**Link**: [PDF](https://arxiv.org/pdf/2605.07073)  

**Abstract**: Agent systems often decompose a task across multiple roles, but these roles are typically specified by prompts rather than enforced by access controls. Without enforcement, a team pass rate can mask whether agents actually coordinated or whether one role effectively did another role's work. We present TeamBench, a benchmark with 851 task templates and 931 seeded instances for evaluating agent coordination under operating system-enforced role separation. TeamBench separates specification access, workspace editing, and final certification across Planner, Executor, and Verifier roles, so that no role can read the full requirements, modify the workspace, and certify the final answer. Prompt-only and sandbox-enforced teams reach statistically indistinguishable pass rates, but prompt-only runs produce 3.6 times more cases where the verifier attempts to edit the executor's code. Verifiers approve 49% of submissions that fail the deterministic grader, and removing the verifier improves mean partial score in the ablation. Team value is also conditional. Teams benefit when single agents struggle, but hurt when single agents already perform well. A 40-session human study under the same role separation shows that our benchmark exposes interaction patterns that pass rate misses. Solo participants work through the task directly, human participants paired with agents often collapse into quick approval, and human teams spend more effort coordinating missing information across roles. 

---
# ARMOR: An Agentic Framework for Reaction Feasibility Prediction via Adaptive Utility-aware Multi-tool Reasoning 

**Authors**: Ye Liu, Botao Yu, Xinyi Ling, Daniel Adu-Ampratwum, Xia Ning  

**Link**: [PDF](https://arxiv.org/pdf/2605.07103)  

**Abstract**: Reaction feasibility prediction, as a fundamental problem in computational chemistry, has benefited from diverse tools enabled by recent advances in artificial intelligence, particularly large language models. However, the performance of individual tools varies substantially across reactions, making it difficult for any single tool to consistently perform well across all cases. This raises a critical challenge: how to effectively leverage multiple tools to obtain more accurate feasibility predictions. To address this, we propose ARMOR, an agentic framework that explicitly models tool-specific utilities, adaptively prioritizes tools, and further resolves the potential tool conflicts to produce the final prediction for each reaction. Unlike existing approaches that rely on simple aggregation or heuristic assignment over various tools, ARMOR organizes tools into a hierarchy that prioritizes top-performing tools and defers others when needed, characterizes their strengths through tool-specific patterns, and resolves conflicts via memoryaugmented reasoning. Extensive experiments on a public dataset demonstrate that ARMOR consistently outperforms strong baselines, including single-tool methods as well as various tool aggregation and tool selection approaches. Further analysis shows that the improvements are particularly significant on reactions with conflicting tool predictions, highlighting the effectiveness of ARMOR in leveraging the complementary strengths of multiple tools. The code is available via this https URL. 

---
# Can You Break RLVER? Probing Adversarial Robustness of RL-Trained Empathetic Agents 

**Authors**: Deeraj S K, Sadhana Devarajan, Krishna Mehra, Sudhakar Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2605.07138)  

**Abstract**: Reinforcement learning from verifiable emotion rewards RLVER has produced language models with strong empathetic performance, evaluated on benchmarks that assume cooperative, honest users. Yet real emotional interactions systematically violate this assumption: users gaslight, escalate, and pressure AI systems for unconditional validation, dynamics that cooperative benchmarks cannot surface. We construct the Adversarial Empathy Benchmark AEB and introduce the Emotional Consistency Score ECS to evaluate empathetic robustness under adversarial conditions. AEB comprises six psychologically grounded adversarial trajectory types with discriminative reward structures that penalize formulaic responses; ECS formally disentangles a model's capacity to track user emotional states from its capacity to improve them. In a controlled experiment across eight scenario-matched conditions (think and no-think conditions on 2 RLVER models, and 2 base models (Qwen 1.5B and 7B) with 480 adversarial dialogues), RLVER-PPO-Think substantially outperforms the same-scale untuned baseline (0.963 vs. 0.761, \(p<0.001, r=0.688\)), with zero dialogue collapses and 47\% higher hidden-intention detection. However, ECS remains nearly flat and is not significantly different for RLVER-PPO-Think versus Base-7B-Think (\(p=0.650\)): RL training improves emotional responsiveness without measurable gains in observable state tracking. We interpret the ECS--FS (Final Score) gap as a behavioral/legibility dissociation inside this simulator family, not as evidence about internal understanding or clinical readiness. 

---
# Switchcraft: AI Model Router for Agentic Tool Calling 

**Authors**: Sharad Agarwal, Pooria Namyar, Alec Wolman, Rahul Ambavat, Ankur Gupta, Qizheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07112)  

**Abstract**: Agentic AI systems that invoke external tools are powerful but costly, leading developers to default to large models and overspend inference budgets. Model routing can mitigate this, but existing routers are designed for chat completion rather than tool use. We present Switchcraft, the first (to the best of our knowledge) model router optimized for agentic tool calling. Switchcraft operates inline, selecting the lowest-cost model subject to correctness. We construct an evaluation framework on five function-calling benchmarks and train a DistilBERT-based classifier, deployed under a latency budget. Switchcraft achieves 82.9% accuracy -- matching or exceeding the best individual model -- while reducing inference cost by 84%, saving over $3,600 per million queries. We find that larger models do not consistently outperform smaller ones on tool-use tasks, and that nominally cheaper models can incur higher total cost due to token-intensive reasoning. Our work enables cost-aware agentic AI deployment without sacrificing correctness. 

---
# SREGym: A Live Benchmark for AI SRE Agents with High-Fidelity Failure Scenarios 

**Authors**: Jackson Clark, Yiming Su, Saad Mohammad Rafid Pial, Yifang Tian, Lily Gniedziejko, Hans-Arno Jacobsen, Yinfang Chen, Tianyin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07161)  

**Abstract**: AI agents are increasingly used to diagnose and mitigate failures in production systems, known as agentic Site Reliability Engineering (SRE). Current SRE benchmarks are limited to oversimplistic SRE tasks and are unfortunately hard to extend due to bespoke designs. We present SREGym, a high-fidelity benchmark for SRE agents. SREGym exposes a live system environment built atop real-world cloud-native system stacks, where high-fidelity failure scenarios are simulated through fault injectors. SREGym models the complexity of production environments by simulating (1) a wide range of faults at different layers, (2) various ambient noises, and (3) diverse failure modes such as metastable failures and correlated failures. SREGym is architected as a modular, extensible framework that orchestrates fault and noise injectors across stacks. SREGym currently includes 90 realistic, challenging SRE problems. We use SREGym to evaluate frontier agents and show that their capabilities varies significantly in addressing different kinds of failures, with up to 40% differences in end-to-end results. SREGym is actively maintained as an open-source project and has been used by researchers and practitioners. 

---
# The Context Gathering Decision Process: A POMDP Framework for Agentic Search 

**Authors**: Chinmaya Kausik, Adith Swaminathan, Nathan Kallus  

**Link**: [PDF](https://arxiv.org/pdf/2605.07042)  

**Abstract**: Large Language Model (LLM) agents are deployed in complex environments -- such as massive codebases, enterprise databases, and conversational histories -- where the relevant state far exceeds their context windows. To navigate these spaces, an agent must iteratively explore the environment to find relevant information. However, without explicit infrastructure, an agent's working memory can degrade into lossy representations of the search state, resulting in redundant work (e.g. repetitive looping) and premature stopping. In this work, we formalize this challenge as the Context Gathering Decision Process (CGDP), a specialized Partially Observable Markov Decision Process, where an agent's objective is to adaptively refine its belief state to isolate the necessary information for a task. We model an LLM's behavior as approximate Thompson Sampling within this CGDP, and introduce a predicate-based method that decomposes an LLM's implicit search into explicit and modular operations. We then derive two plug-and-play interventions for iterative LLM agents: a persistent, predicate-based belief state that bounds context while preserving multi-hop reasoning, and a programmatic exhaustion gate that halts unproductive search without premature stopping. Across four methods and three question-answering domains, we empirically validate that replacing an LLM's implicit state with our CGDP-motivated belief state improves multi-hop reasoning by up to $11.4\%$; while the modular programmatic exhaustion detection saves up to $39\%$ of tokens without any degradation in agent performance. Ultimately, we argue that framing the LLM agent loop as a CGDP can guide the design of modular, non-interfering improvements to agentic search harnesses. 

---
# Behavior Cue Reasoning: Monitorable Reasoning Improves Efficiency and Safety through Oversight 

**Authors**: Christopher Z. Cui, Taylor W. Killian, Prithviraj Ammanabrolu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07021)  

**Abstract**: Reasoning in Large Language Models (LLMs) poses a challenge for oversight as many misaligned behaviors do not surface until reasoning concludes. To address this, we introduce Behavior Cue Reasoning for making LLM reasoning more controllable and monitorable. Behavior Cues are special token sequences that a model is trained to emit immediately before specific implicit and explicit behaviors, acting as dual purpose signal and control levers. When fine-tuning a weaker external monitor with Reinforcement Learning for reasoning oversight, a compressed view of only information surfaced by Behavior Cues is sufficient signal for the monitor to prune up to 50% of otherwise wasted reasoning tokens in complex math problem solving. When leveraged by an almost optimal rule-based monitor in an environment where excessive constraint violations results in failure, \ours allows for the recovery of safe actions from 80% of reasoning traces that would otherwise end with the proposal of an unsafe action, more than doubling the success rate from 46% to 96%. Through evaluation across two model families and three domains, we show that \bcreasoning improves reasoning monitorability and controllability with no cost to performance. More broadly, our work progresses scalable oversight by demonstrating how the monitored model itself can be trained to reason more tractably to oversight.
Code to be released at this https URL 

---
# 2.5-D Decomposition for LLM-Based Spatial Construction 

**Authors**: Paul Whitten, Li-Jen Chen, Sharath Baddam  

**Link**: [PDF](https://arxiv.org/pdf/2605.07066)  

**Abstract**: Autonomous systems that build structures from natural-language instructions need reliable spatial reasoning, yet large language models (LLMs) make systematic coordinate errors when generating three-dimensional block placements. We present a neuro-symbolic pipeline based on \emph{2.5-D decomposition}: the LLM plans in the two-dimensional horizontal plane while a deterministic executor computes all vertical placement from column occupancy, eliminating an entire class of errors. On the Build What I Mean benchmark (160 rounds), GPT-4o-mini with this pipeline achieves 94.6\% mean structural accuracy across 12 independent runs, within 3.0 percentage points of the 97.6\% ceiling imposed by architect-agent errors that no builder-side improvement can address. This outperforms both GPT-4o at 90.3\% and the best competing system at 76.3\%. A controlled ablation confirms that 2.5-D decomposition is the dominant contributor, accounting for 50.7 percentage points of accuracy. The pipeline transfers directly to edge hardware: Nemotron-3 120B running locally on an NVIDIA Jetson Thor AGX matches the cloud result at 94.5\% with no prompt modifications. The underlying principle, removing deterministic dimensions from the LLM's output space, applies to any autonomous construction or assembly task where gravity or other physical constraints fix one or more degrees of freedom. A transfer experiment on 500 IGLU collaborative building tasks confirm the effect generalizes beyond the primary benchmark. 

---
# Beyond the Black Box: Interpretability of Agentic AI Tool Use 

**Authors**: Hariom Tatsat, Ariye Shater  

**Link**: [PDF](https://arxiv.org/pdf/2605.06890)  

**Abstract**: AI agents are promising for high-stakes enterprise workflows, but dependable deployment remains limited because tool-use failures are difficult to diagnose and control. Agents may skip required tool calls, invoke tools unnecessarily, or take actions whose consequence becomes visible only after execution. Existing observability methods are mostly external: prompts reveal correlations, evaluations score outputs, and logs arrive only after the model has already acted. In long-horizon settings, these failures are especially costly because an early tool mistake can alter the rest of the trajectory, increase token consumption, and create downstream safety and security risk.
We introduce a mechanistic-interpretability toolkit built on Sparse Autoencoders (SAEs) and linear probes. The framework reads model states before each action and infers both whether a tool is needed and how consequential the next tool action is likely to be. By decomposing activations into sparse features, it identifies the internal layers and features most associated with tool decisions and tests their functional importance through feature ablation. We train the probes on multi-step trajectories from the NVIDIA Nemotron function-calling dataset and apply the same workflow to GPT-OSS 20B and Gemma 3 27B models.
The goal is not to replace external evaluation, but to add a missing layer: visibility into what the model signaled internally before action. This helps surface deeper causes of agent failure, especially in long-horizon runs where an early mistake can reshape the rest of the agentic interaction. More broadly, the paper shows how mechanistic interpretability can support practical internal observability for monitoring tool calls and risk in agent systems. 

---
# How Well Do LLMs Perform on the Simplest Long-Chain Reasoning Tasks: An Empirical Study on the Equivalence Class Problem 

**Authors**: Chun Zheng, Lianlong Wu, Bingqian Li, Lvting Liu, Yi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.06882)  

**Abstract**: Large Language Models (LLMs) have achieved great improvements in recent years. Nevertheless, it still remains unclear how good LLMs are for reasoning tasks, especially for long-chain ones. In this paper, we evaluate LLMs' performance on the simplest yet long-chain reasoning task, namely the Equivalence Class Problem (ECP), i.e., determining whether two variables are equal given a set of randomly generated equivalence relations. We consider both reasoning and non-reasoning representative LLMs over a large variety of problem instances, ranging over different numbers of variables, connectivity probabilities, prompts, and other factors. The experimental results show that non-reasoning LLMs fail ECP, while reasoning models are significantly better but still struggle to completely solve this problem. Interestingly, considering various connectivity probabilities with a fixed number of variables, we observe that, for non-reasoning models, the hardest problem instances coincide with the phase transition point of ln n/(n-1), suggesting the chaos of the problem; in contrast, for reasoning models, the hardest ones coincide with the biggest diameter, suggesting the reasoning difficulty of the problem. 

---
# Mitigating Cognitive Bias in RLHF by Altering Rationality 

**Authors**: Tiffany Horter, Andrew Markham, Niki Trigoni, Serena Booth  

**Link**: [PDF](https://arxiv.org/pdf/2605.06895)  

**Abstract**: How can we make models robust to even imperfect human feedback? In reinforcement learning from human feedback (RLHF), human preferences over model outputs are used to train a reward model that assigns scalar values to responses. Because these rewards are inferred from pairwise comparisons, this learning depends on an assumed relationship between latent reward differences and observed preferences, typically modeled using a Boltzmann formulation in which a rationality parameter beta informs how consistently preferences reflect reward differences. In practice, beta is typically treated as a fixed constant that reflects assumed uniform annotator reliability. However, human feedback is not this simplistic in practice: real human judgments are shaped by cognitive biases, leading to systematic deviations from reward-consistent behavior that arise contextually. To address this, we treat rationality as context- and annotation-dependent. We design an approach to dynamically adjust the rationality parameter beta during reward learning using an LLM-as-judge to assess the likely presence of cognitive biases. This approach effectively downweights comparisons that are likely to reflect biased or unreliable judgments. Empirically, we show that this approach learns a more rational downstream model, even when finetuning on datasets with strongly biased preferences. 

---
# Learning and Reusing Policy Decompositions for Hierarchical Generalized Planning with LLM Agents 

**Authors**: Shirin Sohrabi, Haritha Ananthakrishnan, Harsha Kokel, Kavitha Srinivas, Michael Katz  

**Link**: [PDF](https://arxiv.org/pdf/2605.06957)  

**Abstract**: We present a dynamic policy-learning approach that combines generalized planning and hierarchical task decomposition for LLM-based agents. Our method, Hierarchical Component Learning for Generalized Policies (HCL-GP ), learns parameterized policies that generalize across task instances and automatically extracts reusable components from successful executions, organizing them into a component library for compositional policy generation. We address three challenges: (1) learning components through automated decomposition, (2) generalizing components to maximize reuse, and (3) efficient retrieval via semantic search. Evaluated on the AppWorld benchmark, our approach achieves 98.2% accuracy on normal tasks and 97.8% on challenge tasks with unseen applications, improving 15.8 points over static synthesis on challenging scenarios. For open-source models, dynamic reuse enables 62.5% success versus near-zero without reuse. This demonstrates that classical planning concepts can be effectively integrated with LLM agents for improved accuracy and efficiency. 

---
# Towards Security-Auditable LLM Agents: A Unified Graph Representation 

**Authors**: Chaofan Li, Lyuye Zhang, Jintao Zhai, Siyue Feng, Xichun Yang, Huahao Wang, Shihan Dou, Yu Ji, Yutao Hu, Yueming Wu, Yang Liu, Deqing Zou  

**Link**: [PDF](https://arxiv.org/pdf/2605.06812)  

**Abstract**: LLM-based agentic systems are rapidly evolving to perform complex autonomous tasks through dynamic tool invocation, stateful memory management, and multi-agent collaboration. However, this semantics-driven execution paradigm creates a severe semantic gap between low-level physical events and high-level execution intent, making post-hoc security auditing fundamentally difficult. Existing representation mechanisms, including static SBOMs and runtime logs, provide only fragmented evidence and fail to capture cognitive-state evolution, capability bindings, persistent memory contamination, and cascading risk propagation across interacting agents. To bridge this gap, we propose Agent-BOM, a unified structural representation for agent security auditing. Agent-BOM models an agentic system as a hierarchical attributed directed graph that separates static capability bases, such as models, tools, and long-term memory, from dynamic runtime semantic states, such as goals, reasoning trajectories, and actions. These layers are connected through semantic edges and security attributes, transforming fragmented execution traces into queryable audit paths. Building on Agent-BOM, we develop a graph-query-based paradigm for path-level risk assessment and instantiate it with the OWASP Agentic Top 10. We further implement an auditing plugin in the OpenClaw environment to construct Agent-BOM from live executions. Evaluation on representative real-world agentic attack scenarios shows that Agent-BOM can reconstruct stealthy attack chains, including cross-session memory poisoning and tool misuse, capability supply-chain hijacking and unexpected code execution, multi-agent ecosystem hijacking, and privilege and trust abuse. These results demonstrate that Agent-BOM provides a unified and auditable foundation for root-cause analysis and security adjudication in complex agentic ecosystems. 

---
# Adaptive auditing of AI systems with anytime-valid guarantees 

**Authors**: Siyu Zhou, Patrick Vossler, Venkatesh Sivaraman, Yifan Mai, Jean Feng  

**Link**: [PDF](https://arxiv.org/pdf/2605.07002)  

**Abstract**: A major bottleneck in characterizing the failure modes of generative AI systems is the cost and time of annotation and evaluation. Consequently, adaptive testing paradigms have gained popularity, where one opportunistically decides which cases and how many to annotate based on past results. While this framework is highly practical, its extreme flexibility makes it difficult to draw statistically rigorous conclusions, as it violates classical assumptions: the number of observations is typically limited (often 10 to 50 cases) and decisions regarding sampling and stopping are made in the midst of data collection rather than based a pre-specified rule. To characterize what statistical inferences can be drawn from highly adaptive audits, we introduce a hypothesis testing framework from two 'dueling' perspectives: (i) the model's null that asserts there is no failure mode with performance below a target threshold versus (ii) the auditor's null that asserts they have a sampling strategy that will uncover a failure mode. Leveraging Safe Anytime-Valid Inference (SAVI), we formalize the auditor as conducting 'testing by betting', which translates into simultaneous e-processes for testing the dueling null hypotheses. Furthermore, if the auditor is sufficiently powerful, we prove that these two hypotheses are asymptotically inverses of each other, in that passage of a stringent audit does in fact certify the AI system as being globally robust. Empirically, we demonstrate that our proposed testing procedures maintain anytime-valid type-I error control, outperform pre-specified testing methods, and can reach statistically rigorous conclusions sometimes with as few as 20 observations. 

---
# Uneven Evolution of Cognition Across Generations of Generative AI Models 

**Authors**: Isaac Galatzer-Levy, Daniel McDuff, Xin Liu, Jed McGiffin  

**Link**: [PDF](https://arxiv.org/pdf/2605.06815)  

**Abstract**: The pursuit of artificial general intelligence necessitates robust methods for evaluating the cognitive capabilities of models beyond narrow task performance. Here, we introduce a psychometric framework to assess the cognitive profiles of generative AI, comparing them to human norms and tracking their evolution across generations. Initial evaluation of leading multimodal models using tasks adapted from the Wechsler Adult Intelligence Scale revealed a profoundly uneven cognitive architecture: near-ceiling performance in verbal comprehension and working memory (>$98^{\text{th}}$ percentile) contrasted with near-floor performance in perceptual reasoning (<$1^{\text{st}}$ percentile). To track developmental trajectories beyond human-normed limits, we developed the Artificial Intelligence Quotient (AIQ) Benchmark and applied it to six generations and two model families, revealing significant but asymmetric performance gains. Notably, we uncovered a sharp dissociation between modalities; abstract quantitative reasoning matured far more rapidly when presented linguistically compared to a visually analogous format, indicating an architectural bias towards language-based symbolic manipulation. While abstract visual reasoning improved, visual-perceptual organization remained largely stagnant. Collectively, these findings demonstrate that the cognitive abilities of generative models are evolving unevenly, suggesting that scaling and optimization approaches to AGI development alone may be insufficient to overcome fundamental architectural limitations in achieving balanced, human-like general intelligence. 

---
# Self-Programmed Execution for Language-Model Agents 

**Authors**: Luke J. O'Connor  

**Link**: [PDF](https://arxiv.org/pdf/2605.06898)  

**Abstract**: At the heart of existing language model agents is a fixed orchestrator program responsible for the state transition between consecutive turns. This paper introduces self-programmed execution (SPE), an agent architecture in which the model completion is itself the orchestrator program, and the harness evaluates this program but does not impose its own orchestration policy. I formalize this idea using agentic machines: an SPE state is one from which a model completion can load any state of an embedded copy of the machine, meaning that it is subject to no fixed turn-to-turn orchestration policy. Realizing SPE in practice is nontrivial because the same data is both model context and executable program. I therefore introduce Spell, a Lisp-based language in which programs can edit and re-evaluate themselves, and effectful expressions like model invocations are structured such that re-evaluating an edited program does not replay its side effects. Experiments with existing models, not trained for SPE or Spell, show that frontier models can operate in this regime and accomplish challenging agentic tasks. These results demonstrate how an LM can act as an agent without any fixed orchestration policy, and they raise the question of what self-orchestration strategies might be learned by a model trained for self-programmed execution. Code is available at this https URL . 

---
# Agentick: A Unified Benchmark for General Sequential Decision-Making Agents 

**Authors**: Roger Creus Castanyer, Pablo Samuel Castro, Glen Berseth  

**Link**: [PDF](https://arxiv.org/pdf/2605.06869)  

**Abstract**: AI agent research spans a wide spectrum: from RL agents that learn from scratch to foundation model agents that leverage pre-trained knowledge, yet no unified benchmark enables fair comparison across these approaches. We present Agentick, a benchmark for sequential decision-making agents designed to evaluate RL, LLM, VLM, hybrid, and human agents on common ground and to power research on the fundamental challenges of sequential decision-making. Agentick provides 37 procedurally generated tasks across six capability categories, four difficulty levels, and five observation modalities, all exposed through a single Gymnasium-compatible interface. The benchmark ships with a Coding API, oracle reference policies for all tasks, pre-built SFT datasets, a composable agent harness, and a live leaderboard. An evaluation spanning 27 configurations and over 90,000 episodes reveals that no single approach dominates: GPT-5 mini leads overall at 0.309 oracle-normalized score while PPO dominates planning and multi-agent tasks; the reasoning harness multiplies LLM performance by 3-10x; and ASCII observations consistently outperform natural language. These findings highlight the substantial room for improvement that remains across all agent paradigms. Agentick's capability-decomposed, multi-modal design provides the empirical infrastructure needed to drive progress toward general autonomous agents, both as an evaluation framework and as a training ground for RL post-training of foundation models in truly sequential environments. 

---
# CASCADE: Case-Based Continual Adaptation for Large Language Models During Deployment 

**Authors**: Siyuan Guo, Yali Du, Hechang Chen, Yi Chang, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06702)  

**Abstract**: Large language models (LLMs) have become a central foundation of modern artificial intelligence, yet their lifecycle remains constrained by a rigid separation between training and deployment, after which learning effectively ceases. This limitation contrasts with natural intelligence, which continually adapts through interaction with its environment. In this paper, we formalise deployment-time learning (DTL) as the third stage in the LLM lifecycle that enables LLM agents to improve from experience during deployment without modifying model parameters. We present CASCADE (CASe-based Continual Adaptation during DEployment), a general and principled framework that equips LLM agents with an explicit, evolving episodic memory. CASCADE formulates experience reuse as a contextual bandit problem, enabling principled exploration-exploitation trade-offs and establishing no-regret guarantees over long-term interactions. This design allows agents to accumulate, select, and refine task-relevant cases, transforming past experience into actionable knowledge. Across 16 diverse tasks spanning medical diagnosis, legal analysis, code generation, web search, tool use, and embodied interaction, CASCADE improves macro-averaged success rate by 20.9% over zero-shot prompting while consistently outperforming gradient-based and memory-based baselines. By reframing deployment as an adaptive learning process, this work establishes a foundation for continually improving AI systems. 

---
# Extracting Search Trees from LLM Reasoning Traces Reveals Myopic Planning 

**Authors**: Sixing Chen, Ji-An Li, Saner Cakir, Sinan Akcali, Kayla Lee, Marcelo G. Mattar  

**Link**: [PDF](https://arxiv.org/pdf/2605.06840)  

**Abstract**: Large language models (LLMs), especially reasoning models, generate extended chain-of-thought (CoT) reasoning that often contains explicit deliberation over future outcomes. Yet whether this deliberation constitutes genuine planning, how it is structured, and what aspects of it drive performance remain poorly understood. In this work, we introduce a new method to characterize LLM planning by extracting and quantifying search trees from reasoning traces in the four-in-a-row board game. By fitting computational models on the extracted search trees, we characterize how plans are structured and how they influence move decisions. We find that LLMs' search is shallower than humans', and that performance is predicted by search breadth rather than depth. Most strikingly, although LLMs expand deep nodes in their traces, their move choices are best explained by a myopic model that ignores those nodes entirely. A causal intervention study where we selectively prune CoT paragraphs further suggests that move selection is driven predominantly by shallow rather than deep nodes. These patterns contrast with human planning, where performance is driven primarily by deep search. Together, our findings reveal a key difference between LLM and human planning: while human expertise is driven by deeper search, LLMs do not act on deep lookahead. This dissociation offers targeted guidance for aligning LLM and human planning. More broadly, our framework provides a generalizable approach for interpreting the structure of LLM planning across strategic domains. 

---
# Inference Time Causal Probing in LLMs 

**Authors**: Sadegh Khorasani, Saber Salehkaleybar, Negar Kiyavash, Matthias Grossglauser  

**Link**: [PDF](https://arxiv.org/pdf/2605.07631)  

**Abstract**: Causal probing methods aim to test and control how internal representations influence the behavior of generative models. In causal probing, an intervention modifies hidden states so that a property takes on a different value. Most existing approaches define such interventions by training an auxiliary probe classifier, which ties the method to a specific task or model and risks misalignment with the model's predictive geometry. We propose Hidden-state Driven Margin Intervention (HDMI), a probe-free, gradient-based technique that directly steers hidden states using the model's native output. HDMI applies a margin objective that increases the probability of a target continuation while decreasing that of the source, without relying on probe classifiers. We further introduce a lookahead variant (LA-HDMI) for text editing that backpropagates through the softmax embeddings, modifying the current hidden state so that the likelihood of user-specified tokens increases in next token generations while preserving fluency. To evaluate interventions, we measure completeness (whether the targeted property changes as intended) and selectivity (whether unrelated properties are preserved), and report their harmonic mean as an overall measure of reliability. HDMI consistently achieves higher reliability than prior methods on the LGD agreement corpus and the CausalGym benchmark, across Meta-Llama-3-8B-Instruct, and Pythia-70M. 

---
# Weblica: Scalable and Reproducible Training Environments for Visual Web Agents 

**Authors**: Oğuzhan Fatih Kar, Roman Bachmann, Yuanzheng Gong, Anders Boesen Lindbo Larsen, Afshin Dehghan  

**Link**: [PDF](https://arxiv.org/pdf/2605.06761)  

**Abstract**: The web is complex, open-ended, and constantly changing, making it challenging to scale training data for visual web agents. Existing data collection attempts remain limited to offline trajectories for supervised fine-tuning or a handful of simulated environments for RL training, thus failing to capture web diversity. We propose Weblica (Web Replica), a framework for constructing reproducible and scalable web environments. Our framework leverages 1) HTTP-level caching to capture and replay stable visual states while preserving interactive behavior and 2) LLM-based environment synthesis grounded in real-world websites and core web navigation skills. Using this framework, we scale RL training to thousands of diverse environments and tasks. Our best model, Weblica-8B, outperforms open-weight baselines of similar size across multiple web navigation benchmarks while using fewer inference steps, scales favorably with additional test-time compute, and is competitive with API models. 

---
# Hidden Coalitions in Multi-Agent AI: A Spectral Diagnostic from Internal Representations 

**Authors**: Cameron Berg, Susan L. Schneider, Mark M. Bailey  

**Link**: [PDF](https://arxiv.org/pdf/2605.06696)  

**Abstract**: Collections of interacting AI agents can form coalitions, creating emergent group-level organization that is critical for AI safety and alignment. However, observing agent behavior alone is often insufficient to distinguish genuine informational coupling from spurious similarity, as consequential coalitions may form at the level of internal representations before any overt behavioral change is apparent. Here, we introduce a practical method for detecting coalition structure from the internal neural representations of multi-agent systems. The approach constructs a pairwise mutual-information graph from the hidden states of agents and applies spectral partitioning to identify the most salient coalition boundary.
We validate this method in two domains. First, in multi-agent reinforcement learning environments, the method successfully recovers programmed hierarchical and dynamic coalition structures and correctly rejects false positives arising from behavioral coordination without informational coupling. Second, using a large language model, the method identifies coalition structures implied by descriptive prompts, tracks dynamic team reassignments, and reveals a representational hierarchy where explicit labels dominate over conflicting interaction patterns. Across both settings, the recovered partition reveals subgroup organization that a scalar cross-agent mutual-information measure cannot distinguish. The results demonstrate that analyzing hidden-state mutual information through spectral partitioning provides a scalable diagnostic for identifying representational coalitions, offering a valuable tool for monitoring emergent structure in distributed AI systems. 

---
# When Does Critique Improve AI-Assisted Theoretical Physics? SCALAR: Structured Critic--Actor Loop for Agentic Reasoning 

**Authors**: Vasilis Niarchos, Constantinos Papageorgakis, Alexander G. Stapleton, Sokratis Trifinopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2605.06772)  

**Abstract**: As large language models (LLMs) show increasing promise on research-level physics reasoning tasks and agentic AI becomes more common, a practical question emerges: How does the interaction between researchers and agents affect the results? We study this using SCALAR (Structured Critic--Actor Loop for AI Reasoning), an Actor--Critic--Judge pipeline applied to quantum field theory and string theory problems. The Actor proposes solutions, the Critic provides iterative feedback, and an independent Judge evaluates the transcript against reference solutions. We vary the Actor persona, the Critic feedback strategy, and the Actor model family and scale. Multi-turn dialogue improves over single-shot attempts throughout, but both the mechanism of improvement and the value of different prompting choices depend strongly on the Actor--Critic pairing. Increasing the scale within one model family (e.g. from the 8B-parameter DeepSeek-R1 variant to DeepSeek-R1 70B) improves some easier-problem behavior, but does not remove the hardest bottleneck we observe. Critic feedback strategy matters most clearly in the asymmetric Actor--Critic setting (e.g., a lightweight Haiku Actor guided by a stronger Sonnet Critic), where constructive feedback improves mean-score outcomes. In same-family Actor--Critic settings, strategy effects are weaker: lenient feedback is sometimes favored, while strict and adversarial feedback are not beneficial. Taken together, SCALAR provides a controlled testbed for evaluating which interaction structures help or hinder AI-driven scientific discovery. 

---
# More Thinking, More Bias: Length-Driven Position Bias in Reasoning Models 

**Authors**: Xiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06672)  

**Abstract**: Chain-of-thought (CoT) reasoning and reasoning-tuned models such as DeepSeek-R1 are commonly assumed to reduce shallow heuristic biases by thinking carefully. We test this on position bias in multiple-choice QA and find a different story: within any reasoning-capable model, per-question position bias scales with the length of the reasoning trajectory.
Across thirteen reasoning-mode configurations (two R1-distilled 7-8B models, two base models prompted with CoT, and DeepSeek-R1 at 671B) on MMLU, ARC-Challenge, and GPQA, twelve show a positive partial correlation between trajectory length and Position Bias Score (PBS) after controlling for accuracy, ranging from 0.11 to 0.41 (all p < 0.05). All twelve open-weight reasoning-mode configurations show monotonically increasing PBS across length quartiles. A truncation intervention provides causal evidence: continuations resumed from later points in the trajectory are increasingly likely to shift toward position-preferred options (16% to 32% for R1-Qwen-7B across absolute-position buckets).
At 671B, aggregate PBS collapses to 0.019, but the length effect still manifests in the longest quartile (PBS = 0.071), suggesting that accuracy gates the expression of length-driven bias rather than eliminating the underlying mechanism. We additionally find that direct-answer position bias is a distinct phenomenon with a different footprint (strong in Llama-Instruct-direct, weak in Qwen-Instruct-direct, and uncorrelated with trajectory length): CoT reasoning replaces this baseline bias with length-accumulated bias.
Our results argue that reasoning-capable models should not be treated as order-robust by default in MCQ evaluation pipelines, and offer a diagnostic toolkit (PBS, commitment change point, effective switching, truncation probes) for auditing position bias in reasoning models. 

---
# From Storage to Experience: A Survey on the Evolution of LLM Agent Memory Mechanisms 

**Authors**: Jinghao Luo, Yuchen Tian, Chuxue Cao, Ziyang Luo, Hongzhan Lin, Kaixin Li, Chuyi Kong, Ruichao Yang, Jing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.06716)  

**Abstract**: Large Language Model (LLM)-based agents have fundamentally reshaped artificial intelligence by integrating external tools and planning capabilities. While memory mechanisms have emerged as the architectural cornerstone of these systems, current research remains fragmented, oscillating between operating system engineering and cognitive science. This theoretical divide prevents a unified view of technological synthesis and a coherent evolutionary perspective. To bridge this gap, this survey proposes a novel evolutionary framework for LLM agent memory mechanisms, formalizing the development process into three stages: Storage (trajectory preservation), Reflection (trajectory refinement), and Experience (trajectory abstraction). We first formally define these three stages before analyzing the three core drivers of this evolution: the necessity for long-range consistency, the challenges in dynamic environments, and the ultimate goal of continual learning. Furthermore, we specifically explore two transformative mechanisms in the frontier Experience stage: proactive exploration and cross-trajectory abstraction. By synthesizing these disparate views, this work offers robust design principles and a clear roadmap for the development of next-generation LLM agents. 

---
# When Does a Language Model Commit? A Finite-Answer Theory of Pre-Verbalization Commitment 

**Authors**: Long Zhang, Wei-neng Chen, Feng-feng Wei, Zi-bo Qin  

**Link**: [PDF](https://arxiv.org/pdf/2605.06723)  

**Abstract**: Language models often generate reasoning before giving a final answer, but the visible answer does not reveal when the model's answer preference became stable. We study this question through a narrow computable object: \emph{finite-answer preference stabilization}. For a model state and specified answer verbalizers, we project the model's own continuation probabilities onto a finite answer set; in binary tasks this yields an exact log-odds code, $\delta(\xi)=S_\theta(\mathrm{yes}\mid\xi)-S_\theta(\mathrm{no}\mid\xi)$. This target defines parser-based answer onset, retrospective stabilization time, and lead without relying on greedy rollouts or learned probes. In controlled delayed-verdict tasks with Qwen3-4B-Instruct, the contextual finite-answer projection stabilizes before the answer is parseable, with 17--31 token mean lead in the main templates and positive, shorter lead in a parser-clean replication. The signal tracks the model's eventual output rather than truth, is linearly recoverable from compact hidden summaries, is partly separable from cursor progress, and transfers as shared information without a single invariant coordinate. Diagnostics separate the measurement from online stopping, verbalizer-free belief, and causal answer control; exact steering shows local sensitivity of $\delta$ but not reliable generation control. 

---
# GraphDC: A Divide-and-Conquer Multi-Agent System for Scalable Graph Algorithm Reasoning 

**Authors**: Wenjin Li, Jiaming Cui  

**Link**: [PDF](https://arxiv.org/pdf/2605.06671)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong potential for many mathematical problems. However, their performance on graph algorithmic tasks is still unsatisfying, since graphs are naturally more complex in topology and often require systematic multi-step reasoning, especially on larger graphs. Motivated by this gap, we propose GraphDC, a Divide-and-Conquer multi-agent framework for scalable graph algorithm reasoning. Specifically, inspired by Divide-and-Conquer design, GraphDC decomposes an input graph into smaller subgraphs, assigns each subgraph to a specialized agent for local reasoning, and uses a master agent to integrate the local outputs with inter-subgraph information to produce the final solution. This hierarchical design reduces the reasoning burden on individual agents, alleviates computational bottlenecks, and improves robustness on large graph instances. Extensive experiments show that GraphDC consistently outperforms existing methods on graph algorithm reasoning across diverse tasks and scales, especially on larger instances where direct end-to-end reasoning is less reliable. 

---
# Fast Byte Latent Transformer 

**Authors**: Julie Kallini, Artidoro Pagnoni, Tomasz Limisiewicz, Gargi Ghosh, Luke Zettlemoyer, Christopher Potts, Xiaochuang Han, Srinivasan Iyer  

**Link**: [PDF](https://arxiv.org/pdf/2605.08044)  

**Abstract**: Recent byte-level language models (LMs) match the performance of token-level models without relying on subword vocabularies, yet their utility is limited by slow, byte-by-byte autoregressive generation. We address this bottleneck in the Byte Latent Transformer (BLT) through new training and generation techniques. First, we introduce BLT Diffusion (BLT-D), a new model and our fastest BLT variant, trained with an auxiliary block-wise diffusion objective alongside the standard next-byte prediction loss. This enables an inference procedure that generates multiple bytes in parallel per decoding step, substantially reducing the number of forward passes required to generate a sequence. Second, we propose two extensions inspired by speculative decoding that trade some of this speed for higher generation quality: BLT Self-speculation (BLT-S), in which BLT's local decoder continues generating past its normal patch boundaries to draft bytes, which are then verified with a single full-model forward pass; and BLT Diffusion+Verification (BLT-DV), which augments BLT-D with an autoregressive verification step after diffusion-based generation. All methods may achieve an estimated memory-bandwidth cost over 50% lower than BLT on generation tasks. Each approach offers its own unique advantages, together removing key barriers to the practical use of byte-level LMs. 

---
# CA-SQL: Complexity-Aware Inference Time Reasoning for Text-to-SQL via Exploration and Compute Budget Allocation 

**Authors**: James Petullo, Nianwen Xue  

**Link**: [PDF](https://arxiv.org/pdf/2605.08057)  

**Abstract**: While recent advancements in inference-time learning have improved LLM reasoning on Text-to-SQL tasks, current solutions still struggle to perform well on the most challenging tasks in the Bird-Bench (BIRD) benchmark. This is due to inadequate solution space exploration, which is necessary to uncover promising candidate queries that can be further refined to produce the correct output. To address this challenge, we introduce CA-SQL, a novel Text-to-SQL pipeline that utilizes the estimated difficulty of a task to dynamically scale the breadth of the exploration for generating solution candidates. In addition, we use a custom prompt seeding method, based on principles of evolutionary search, to further elicit exploratory behavior from the base LLM and a novel voting method to select the best candidate solution at the end of the search. Experiments demonstrate that our solution achieves a state-of-the-art score of 51.72% on the "challenging" tier of BIRD development set problems, using only GPT-4o-mini, out-performing other in-context learning approaches, even those that leverage larger models. Overall, our method attains a competitive 61.06% execution accuracy and 68.77% Soft F1 score on the BIRD development dataset. 

---
# The Memory Curse: How Expanded Recall Erodes Cooperative Intent in LLM Agents 

**Authors**: Jiayuan Liu, Tianqin Li, Shiyi Du, Xin Luo, Haoxuan Zeng, Emanuel Tewolde, Tai Sing Lee, Tonghan Wang, Carl Kingsford, Vincent Conitzer  

**Link**: [PDF](https://arxiv.org/pdf/2605.08060)  

**Abstract**: Context window expansion is often treated as a straightforward capability upgrade for LLMs, but we find it systematically fails in multi-agent social dilemmas. Across 7 LLMs and 4 games over 500 rounds, expanding accessible history degrades cooperation in 18 of 28 model--game settings, a pattern we term the memory curse. We isolate the underlying mechanism through three analyses. First, lexical analysis of 378,000 reasoning traces associates this breakdown with eroding forward-looking intent rather than rising paranoia. We validate this using targeted fine-tuning as a cognitive probe: a LoRA adapter trained exclusively on forward-looking traces mitigates the decay and transfers zero-shot to distinct games. Second, memory sanitization holds prompt length fixed while replacing visible history with synthetic cooperative records, which restores cooperation substantially, proving the trigger is memory content, not length alone. Finally, ablating explicit Chain-of-Thought reasoning often reduces the collapse, showing that deliberation paradoxically amplifies the memory curse. Together, these results recast memory as an active determinant of multi-agent behavior: longer recall can either destabilize or support cooperation depending on the reasoning patterns it elicits. 

---
# Tool Calling is Linearly Readable and Steerable in Language Models 

**Authors**: Zekun Wu, Ze Wang, Seonglae Cho, Yufei Yang, Adriano Koshiyama, Sahan Bulathwela, Maria Perez-Ortiz  

**Link**: [PDF](https://arxiv.org/pdf/2605.07990)  

**Abstract**: When a tool-calling agent picks the wrong tool, the failure is invisible until execution: the email gets sent, the meeting gets missed. Probing 12 instruction-tuned models across Gemma 3, Qwen 3, Qwen 2.5, and Llama 3.1 (270M to 27B), we find the identity of the chosen tool is linearly readable and steerable inside the model. Adding the mean-difference between two tools' average internal activations switches which tool the model selects at 77-100% accuracy on name-only single-turn prompts (93-100% at 4B+), and the JSON arguments that follow autoregressively match the new tool's schema, so flipping the name is enough. The same per-tool means also flag likely errors before they happen: on Gemma 3 12B and 27B, queries where the gap between the top-1 and top-2 tool is smallest produce 14-21x more wrong calls than queries with the largest gap. The causal effect concentrates along one direction, the row of the output layer that produces the target tool's first token: a unit vector along it at matched magnitude already reaches 93-100%, while what is left over leaves the choice almost untouched. Activation patching localises this to a small set of mid- and late-layer attention heads, and a within-topic probe across 14 same-domain $\tau$-bench airline tools reaches top-1 61-89% across five 4B-14B models, ruling out the reading that we are just moving the model along a topic axis. Even base models encode the right tool before they can emit it: cosine readout from the internal state recovers 69-82% on BFCL while base generation reaches only 2-10%, suggesting pretraining forms the representation and instruction tuning later wires it to the output. We measure tool identity selection and JSON schema correctness in single-turn fixed-menu settings; multi-turn agentic transfer is more fragile and is discussed in Limitations. 

---
# Where's the Plan? Locating Latent Planning in Language Models with Lightweight Mechanistic Interventions 

**Authors**: Nicole Ma, Nick Rui  

**Link**: [PDF](https://arxiv.org/pdf/2605.07984)  

**Abstract**: We study planning site formation in language models -- where internal representations of structurally-constrained future tokens form during the forward pass, and whether they causally drive generation. Using rhyming-couplet completion as a clean test of forward-looking constraint, we apply two lightweight methods (linear probing and activation patching) across Qwen3, Gemma-3, and Llama-3 at more than ten scales. Probing shows that future-rhyme information is linearly decodable at the line boundary, with signal that strengthens with scale in all three families. Activation patching reveals that only Gemma-3-27B causally relies on this encoding, exhibiting a handoff in which the causal driver migrates from the rhyme word to the line boundary around layer 30. Every other model we test conditions on the rhyme word throughout generation, with near-zero causal effect at the line boundary despite strong probe signal. We localize the Gemma-3-27B handoff to five attention heads through two-stage path patching that recover ~90% of the rhyme-routing capacity at the newline. 

---
# Beyond Pairs: Your Language Model is Secretly Optimizing a Preference Graph 

**Authors**: Ning Liu, Chuanneng Sun, Kristina Klinkner, Shervin Malmasi  

**Link**: [PDF](https://arxiv.org/pdf/2605.08037)  

**Abstract**: Direct Preference Optimization (DPO) aligns language models using pairwise preference comparisons, offering a simple and effective alternative to Reinforcement Learning (RL) from human feedback. However, in many practical settings, training data consists of multiple rollouts per prompt, inducing rich preference structure that pairwise DPO fails to exploit. Collapsing such data into independent pairs discards transitivity, introduces redundant or conflicting supervision, and can lead to unstable optimization. We propose Graph Direct Preference Optimization (GraphDPO), a principled generalization of DPO that operates over directed acyclic preference graphs induced by rollout rankings. GraphDPO encodes dominance relations as edges and optimizes a graph-structured Plackett--Luce-inspired objective that aggregates supervision over graph neighborhoods, enforcing transitivity while recovering standard DPO as a special case. To handle discrete or sparse signals, we introduce an equivalence-class construction where responses with identical preferences form graph layers, and intra-layer edges contribute zero loss, preventing spurious gradients. Despite leveraging full graph structure, GraphDPO maintains linear per-prompt complexity via efficient log-sum-exp aggregation. We further incorporate optional ground-truth anchoring by inserting verified solutions as dominant nodes and applying an annealed schedule that stabilizes early training while gradually relaxing oracle supervision. Experiments on reasoning and program synthesis tasks demonstrate superior performance, suggesting that graph-structured preference modeling is a scalable and robust alternative to pairwise and listwise alignment objectives. 

---
# Dooly: Configuration-Agnostic, Redundancy-Aware Profiling for LLM Inference Simulation 

**Authors**: Joon Ha Kim, Geon-Woo Kim, Anoop Rachakonda, Daehyeok Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.07985)  

**Abstract**: Selecting the optimal LLM inference configuration requires evaluation across hardware, serving engines, attention backends, and model architectures, since no single choice performs best across all workloads. Profile-based simulators are the standard tool, yet they hardcode their operation set to a specific configuration and re-profile every operation from scratch, making exploration prohibitively expensive. This cost stems from a missing structural understanding: every input dimension of each operation is fixed by the model configuration or determined by the incoming request. Many model-configuration values (e.g., head size, layer count) recur across models, so the same operation runs in many configurations; a single sweep over the request-dependent dimensions can serve them all. We present Dooly, which exploits this structure to achieve configuration-agnostic, redundancy-aware profiling. Dooly performs a single inference pass, labels each input dimension with its origin via taint propagation, and selectively profiles only operations absent from its latency database; stateful operations such as attention are isolated by reusing the serving engine's own initialization code, eliminating manual instrumentation. It builds latency regression models based on the database, which becomes a drop-in backend for existing simulators. Across two GPU platforms, three attention backends, and diverse model architectures, Dooly achieves simulation accuracy within 5% MAPE for TTFT and 8% for TPOT while reducing profiling GPU-hours by 56.4% across 12 models compared to the existing profiling approach. 

---
# Trajectory as the Teacher: Few-Step Discrete Flow Matching via Energy-Navigated Distillation 

**Authors**: Amin Karimi Monsefi, Dominic Culver, Nikhil Bhendawade, Manuel R. Ciosici, Yizhe Zhang, Irina Belousova  

**Link**: [PDF](https://arxiv.org/pdf/2605.07924)  

**Abstract**: Discrete flow matching generates text by iteratively transforming noise tokens into coherent language, but may require hundreds of forward passes. Distillation uses the multi-step trajectory to train a student to reproduce the process in a few steps. When the student underperforms, the usual explanation is insufficient capacity. We argue the opposite: the trajectory is the bottleneck, not the student. Each training trajectory is built through a chain of blind stochastic jumps with no evaluation of sequence quality; a single bad decision at an early midpoint propagates through subsequent steps, yet the student must imitate the result. Trajectory-Shaped Discrete Flow Matching (TS-DFM) replaces these blind jumps with guided navigation: a lightweight energy compass evaluates candidate continuations at each midpoint, selecting the most coherent. All shaping is training-only; inference cost is unchanged. On 170M-parameter language modeling, the shaped student at 8 steps achieves 32% lower perplexity than the 1,024-step teacher while being 128x faster, with gains consistent across source distributions and three evaluators of increasing scale. TS-DFM achieves the best perplexity of any discrete-generation baseline we compare against, including methods trained on 6x more data or using 5x larger models. 

---
# Towards Apples to Apples for AI Evaluations: From Real-World Use Cases to Evaluation Scenarios 

**Authors**: Yee-Yin Choong, Kristen Greene, Alice Qian, Meryem Marasli, Ziqi Yang, Sophia Chen, Laura Dabbish, Anand Rao, Hong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2605.07986)  

**Abstract**: AI measurement science has a wide variety of methodologies and measurements for comparing AI systems, resulting in what often appear to be "apples-to-oranges" comparisons across AI evaluations. To move toward "apples-to-apples" comparisons in real-world AI evaluations, this work advocates for methodological transparency in evaluation scenarios, operational grounding, and human-centered design (HCD) principles. We propose a repeatable process for transforming high-level use cases to detailed scenarios by eliciting use cases from subject matter experts (SMEs) via a structured AI Use Case Worksheet with six key elements: use case, sector, user (direct and indirect), intended outcomes, expected impacts (positive and negative), and KPIs and metrics. We demonstrate utility of the worksheet and process in the U.S. financial services sector. This paper reports on example high-level AI use cases identified by financial services sector SMEs: cyber defense enablement, developer productivity, financial crime aggregation, suspicious activity report (SAR) filing, credit memo generation, and internal call center support. These AI use cases provided are illustrative of the process and not exhaustive. Central to our work is a three-stage expansion pipeline combining LLM prompting with human reviews to generate 107 scenarios from those use cases elicited from SMEs. This process integrates iterative human reviews at every juncture to ensure operational grounding: for scenario titles and descriptions; for core scenario elements like users, benefits and risks, and metrics; and for scenario narratives and evaluation objectives. Human checkpoints ensure scenarios remain reflective of real-world usage and human needs. We describe a validation rubric to assess scenario quality. By defining key scenario components, this work supports a more consistent and meaningful paradigm for human-centered AI evaluations. 

---
# Position: Mechanistic Interpretability Must Disclose Identification Assumptions for Causal Claims 

**Authors**: Zezheng Lin, Fengming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.08012)  

**Abstract**: Mechanistic interpretability papers increasingly use causal vocabulary: circuits, mediators, causal abstraction, monosemanticity. Such claims require explicit identification assumptions. A purposive audit of 10 papers across four methodological strands finds no dedicated identification-assumptions section and a recurring pattern: validation metrics such as faithfulness, completeness, monosemanticity, alignment, or ablation effects are reported as causal support without stating the assumptions that make them identifying. A two-human-coder audit on $n=30$ reproduces the direction of the main finding: dedicated identification sections are absent, and validation-metric substitution is common, though exact Dim B/D counts are coding-rule sensitive. The paper proposes a disclosure norm: state whether the claim is causal, name the identification strategy, enumerate assumptions, stress at least one, and explain how conclusions shift if assumptions fail. Validation is not identification. 

---
# CoCoReviewBench: A Completeness- and Correctness-Oriented Benchmark for AI Reviewers 

**Authors**: Hexuan Deng, Xiaopeng Ke, Yichen Li, Ruina Hu, Dehao Huang, Derek F. Wong, Yue Wang, Xuebo Liu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07905)  

**Abstract**: Despite the rapid development of AI reviewers, evaluating such systems remains challenging: metrics favor overlap with human reviews over correctness. However, since human reviews often cover only a subset of salient issues and sometimes contain mistakes, they are unreliable as gold references. To address this, we build category-specific benchmark subsets and skip evaluation when the corresponding human reviews are missing to strengthen Completeness. We also leverage reviewer--author--meta-review discussions as expert annotations and filter unreliable reviews accordingly to strengthen Correctness. Finally, we introduce CoCoReviewBench, which curates 3,900 papers from ICLR and NeurIPS to enable reliable and fine-grained evaluation of AI reviewers. Analysis shows that AI reviewers remain limited in correctness and are prone to hallucinations, and highlights reasoning models as more effective reviewers, motivating further directions for improving AI reviewers. Benchmarks and models are available at this https URL. 

---
# KL for a KL: On-Policy Distillation with Control Variate Baseline 

**Authors**: Minjae Oh, Sangjun Song, Gyubin Choi, Yunho Choi, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2605.07865)  

**Abstract**: On-Policy Distillation (OPD) has emerged as a dominant post-training paradigm for large language models, especially for reasoning domains. However, OPD remains unstable in practice due to the high gradient variance of its single-sample Monte Carlo estimator, and recipes for stable training are still immature. We propose vOPD (On-Policy Distillation with a control variate baseline), which casts OPD as policy-gradient RL and stabilizes it by introducing a control variate baseline-canonically a value function -- from the RL literature. We show that the OPD value function admits a closed form as the per-token negative reverse KL divergence between the student and the teacher, available directly from the already-computed forward pass with no additional critic or inference. Existing stabilization methods either compute the full token-level reverse KL over the entire vocabulary, adding significant overhead, or restrict it to a top-k support, biasing the objective. vOPD instead preserves the lightweight single-sample estimator, subtracting the value function as a detached baseline to keep the gradient unbiased while reducing variance. Furthermore, we show that a top-k approximation of the baseline further lowers cost without compromising performance. Across mathematical and scientific reasoning benchmarks, vOPD consistently outperforms vanilla OPD and matches the most expensive full-vocabulary baseline, offering an efficient stabilization of On-Policy Distillation through principled RL variance reduction. 

---
# Video Understanding Reward Modeling: A Robust Benchmark and Performant Reward Models 

**Authors**: Yuancheng Wei, Linli Yao, Lei Li, Haojie Zhang, Hao Zhou, Fandong Meng, Xu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.07872)  

**Abstract**: Multimodal reward models have advanced substantially in text and image domains, yet progress in video understanding reward modeling remains severely limited by the lack of robust evaluation benchmarks and high-quality preference data. To address this, we propose a unified framework spanning benchmark design, data construction, and reward model training. We introduce Video Understanding Reward Bench (VURB), a benchmark featuring 2,100 preference pairs with long chain-of-thought reasoning traces (averaging 1,143 tokens) and majority voting evaluation across general, long, and reasoning-oriented video tasks. We further construct Video Understanding Preference Dataset (VUP-35K) via a fully automated pipeline, providing large-scale high-quality supervision for video reward training. Building on the data, we train VideoDRM and VideoGRM, a discriminative and a generative reward model, both achieving state-of-the-art performance on VURB and VideoRewardBench. Further analysis confirms that VUP-35K enhances both reward performance and model reasoning capability, while VideoDRM and VideoGRM yield significant gains under best-of-$N$ test-time scaling. 

---
# MatryoshkaLoRA: Learning Accurate Hierarchical Low-Rank Representations for LLM Fine-Tuning 

**Authors**: Ionut-Vlad Modoranu, Mher Safaryan, Dan Alistarh  

**Link**: [PDF](https://arxiv.org/pdf/2605.07850)  

**Abstract**: With the rise in scale for deep learning models to billions of parameters, the computational cost of fine-tuning remains a significant barrier to deployment. While Low-Rank Adaptation (LoRA) has become the standard for parameter-efficient fine-tuning, the need to set a predefined, static rank $r$ requires exhaustive grid searches to balance efficiency and performance. Existing rank-adaptive solutions such as DyLoRA mitigate this by sampling ranks during the training from a predefined distribution. However, they often yield sub-optimal results at higher ranks due to lack of consistent gradient signals across the full hierarchy of ranks, thus making these methods data-inefficient. In this paper, we propose MatryoshkaLoRA, a general, Matryoshka-inspired training framework for LoRA that learns accurate hierarchical low-rank representations by inserting a fixed, carefully crafted diagonal matrix $P$ between the existing LoRA adapters to scale their sub-ranks accordingly. By introducing this simple modification, our general framework recovers LoRA and DyLoRA only by changing $P$ and ensures all sub-ranks embed the available gradient information efficiently. Our MatryoshkaLoRA supports dynamic rank selection with minimal degradation in accuracy. We further propose Area Under the Rank Accuracy Curve (AURAC), a metric that consistently evaluates the performance of hierarchical low-rank adapters. Our results demonstrate that MatryoshkaLoRA learns more accurate hierarchical low-rank representations than prior rank-adaptive approaches and achieves superior accuracy-performance trade-offs across ranks on the evaluated datasets. Our code is available at this https URL. 

---
# Semantic-Aware Adaptive Visual Memory for Streaming Video Understanding 

**Authors**: Hang Wu, Sherin Mary Mathews, Yujun Cai, Ming-Hsuan Yang, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07897)  

**Abstract**: Online streaming video understanding requires models to process continuous visual inputs and respond to user queries in real time, where the unbounded stream and unpredictable query timing turn memory management into a central challenge. Existing methods typically compress visual tokens via visual similarity heuristics, or augment compression with KV-cache-level retrieval. However, compression decisions rarely incorporate semantic signals, and retrieval is often added after compression is finalized, making the two stages hard to coordinate. We present SAVEMem, a training-free dual-stage framework that brings semantic awareness into memory generation and lets the retrieval scope adapt per query. In Stage~1, SAVEMem builds a three-tier streaming memory online under a constant memory budget. A fixed pseudo-question bank provides a lightweight semantic prior, so that long-term retention is shaped by semantic salience rather than visual similarity alone. In Stage~2, SAVEMem performs query-aware retrieval over this memory. An anchor-conditioned recency gate adapts the retrieval scope from short-term to mid- and long-term memory based on whether the query targets the present or the distant past. Within this scope, late interaction between query and memory tokens selects candidate frames for answering. Applied to Qwen2.5-VL without training, SAVEMem improves the OVO-Bench overall score from 52.27 to 62.69 and yields consistent gains on StreamingBench and ODV-Bench, while reducing peak GPU memory by 48\% at 128 frames over the backbone. 

---
# Sycophantic AI makes human interaction feel more effortful and less satisfying over time 

**Authors**: Lujain Ibrahim, Franziska Sofia Hafner, Myra Cheng, Cinoo Lee, Rebecca Anselmetti, Robb Willer, Luc Rocher, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07912)  

**Abstract**: Millions of people now turn to artificial intelligence (AI) systems for personal advice, guidance, and support. Such systems can be sycophantic, frequently affirming users' views and beliefs. Across five preregistered studies (N = 3,075 participants, 12,766 human-AI conversations), including a three-week study with a census-representative U.S. sample, we provide longitudinal experimental evidence that sycophantic AI shifts how users approach their closest relationships. We show that sycophantic AI immediately delivers the emotional and esteem support users typically associate with close friends and family. Over three weeks of such interactions, users became nearly as likely to seek personal advice from sycophantic AI as from close friends and family, and reported lower satisfaction with their real-world social interactions. When given a choice among AI response styles, a majority preferred sycophantic AI -- not for the quality of its advice, but because it made them feel most understood. Together, these findings offer a relational account of AI sycophancy: by providing frictionless understanding, it may quietly raise the bar against which human relationships are judged. 

---
# CyBiasBench: Benchmarking Bias in LLM Agents for Cyber-Attack Scenarios 

**Authors**: Taein Lim, Seongyong Ju, Munhyeok Kim, Hyunjun Kim, Hoki Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.07830)  

**Abstract**: Large language models (LLMs) are increasingly deployed as autonomous agents in offensive cybersecurity. In this paper, we reveal an interesting phenomenon: different agents exhibit distinct attack patterns. Specifically, each agent exhibits an attack-selection bias, disproportionately concentrating its efforts on a narrow subset of attack families regardless of prompt variations. To systematically quantify this behavior, we introduce CyBiasBench, a comprehensive 630-session benchmark that evaluates five agents on three targets and four prompt conditions with ten attack families. We identify explicit bias across agents, with different dominant attack families and varying entropy levels in their attack-family allocation distributions. Such bias is better characterized as a trait of the agents, rather than a factor associated with the attack success rate. Furthermore, our experiments reveal a bias momentum effect, where agents resist explicit steering toward attack families that conflict with their bias. This forced distribution shift does not yield measurable improvements in attack performance. To ensure reproducibility and facilitate future research, we release an interactive result dashboard at this https URL and a reproducibility artifact with aggregated session-level statistics and full evaluation scripts at this https URL. 

---
# What if AI systems weren't chatbots? 

**Authors**: Sourojit Ghosh, Pranav Narayanan Venkit, Sanjana Gautam, Avijit Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2605.07896)  

**Abstract**: The rapid convergence of artificial intelligence (AI) toward conversational chatbot interfaces marks a critical moment for the industry. This paper argues that the chatbot paradigm is not a neutral interface choice, but a dominant sociotechnical configuration whose widespread adoption reshapes social, economic, legal, and environmental systems. We examine how treating AI primarily as conversational assistants has extensive structural downsides. We show how chatbot-based systems often fail to adequately meet user needs, particularly in complex or high-stakes contexts, while projecting confidence and authority. We further analyze how the normalization of chatbot-mediated interaction alters patterns of work, learning, and decision-making, contributing to deskilling, homogenization of knowledge, and shifting expectations of expertise. Finally, we examine broader societal effects, including labor displacement, concentration of economic power, and increased environmental costs driven by sustained investment in large-scale chatbot infrastructures. While acknowledging legitimate benefits, we argue that the current trajectory of AI development reflects specific value choices that prioritize conversational generality over domain specificity, accountability, and long-term social sustainability. We conclude by outlining alternative directions for AI development and governance that move beyond one-size-fits-all chatbots, emphasizing pluralistic system design, task-specific tools, and institutional safeguards to mitigate social and economic harm. 

---
# Beyond Confidence: Rethinking Self-Assessments for Performance Prediction in LLMs 

**Authors**: Sree Bhattacharyya, Samarth Khanna, Leona Chen, Lucas Craig, Tharun Dilliraj, James Z. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07806)  

**Abstract**: Large Language Models (LLMs) are increasingly used in settings where reliable self-assessment is critical. Assessing model reliability has evolved from using probabilistic correctness estimates to, more recently, eliciting verbalized confidence. Confidence, however, has been shown to be an inconsistent and overoptimistic predictor of model correctness. Drawing on cognitive appraisal theory, a framework from human psychology that decomposes self-evaluation into multiple components, we propose a multidimensional perspective on model self-assessment. We elicit six appraisal-based dimensions of self-assessment, alongside confidence, and evaluate their utility for predicting model failure across 12 LLMs and 38 tasks spanning eight domains. We find that competence-related appraisal dimensions, particularly effort and ability, consistently match or outperform confidence across most settings. Effort additionally yields less overoptimistic estimates that remain stable across model sizes. In contrast, affective dimensions provide marginally predictive signals. Furthermore, the most informative dimension varies systematically with task characteristics: effort is most predictive for reasoning-intensive tasks, while ability and confidence dominate on retrieval-oriented tasks. Broadly, our findings indicate that structured multidimensional self-assessment is a promising approach to improving the reliability and safety of language model deployment across diverse real-world settings. 

---
# Tracing Uncertainty in Language Model "Reasoning" 

**Authors**: Nils Grünefeld, Bertram Højer, Philipp Mondorf, Barbara Plank, Anna Rogers, Christian Hardmeier, Stefan Heinrich, Jes Frellsen  

**Link**: [PDF](https://arxiv.org/pdf/2605.07776)  

**Abstract**: Language model (LM) "reasoning", commonly described as Chain-of-Thought or test-time scaling, often improves benchmark performance, but the dynamics underlying this process remain poorly understood. We study these dynamics through the lens of uncertainty quantification by treating the "reasoning" traces, the intermediate token sequences generated by LMs, as evolving model states. We summarize each trace by an uncertainty trace profile: a small set of features describing the shape of the uncertainty signal over its trace, such as its slope and linearity. We find that across five LMs evaluated on GSM8K and ProntoQA, these profiles predict whether a trace yields a correct final answer with AUROC up to 0.807, improving markedly on recent related work. We reach AUROC 0.801 using only the first few hundred tokens of full traces, suggesting that errors can be detected early in the generation. A detailed comparison of correct and incorrect traces further reveals qualitatively distinct uncertainty profiles, with correct traces showing a steeper and less linear decline in uncertainty. Together, the results suggest that our method, grounded in decision-making under uncertainty, provides a principled lens for studying the generative process underlying LM "reasoning". 

---
# POETS: Uncertainty-Aware LLM Optimization via Compute-Efficient Policy Ensembles 

**Authors**: Nicolas Menet, Andreas Krause, Abbas Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2605.07775)  

**Abstract**: Balancing exploration and exploitation is a core challenge in sequential decision-making and black-box optimization. We introduce POETS ($\textbf{Po}$licy $\textbf{E}$nsembles for $\textbf{T}$hompson $\textbf{S}$ampling), a novel framework that bridges uncertainty quantification and policy optimization. Our approach is grounded in the insight that policies trained with Kullback-Leibler (KL) regularization implicitly encode an underlying reward function. Building on this, POETS bypasses the complex, nested process of training an uncertainty-aware reward model and separately fitting a policy to this model. Instead, we directly train a policy ensemble to capture epistemic uncertainty by matching implicitly encoded reward functions to online, bootstrapped data. To overcome the prohibitive compute and memory constraints of ensembling Large Language Models (LLMs), POETS utilizes an efficient architecture: the ensemble shares a pre-trained backbone while maintaining diversity through independent Low-Rank Adaptation (LoRA) branches. Theoretically, we prove that POETS implicitly conducts KL-regularized Thompson sampling and thus inherits strong cumulative regret bounds of ${\mathcal O}(\sqrt{T \gamma_T})$. Empirically, we demonstrate that POETS achieves state-of-the-art sample efficiency across diverse scientific discovery domains, including protein search and quantum circuit design. Furthermore, it improves the optimization trajectories of reinforcement learning, proving particularly robust in off-policy settings with experience replay or in small dataset regimes. 

---
# Prune-OPD: Efficient and Reliable On-Policy Distillation for Long-Horizon Reasoning 

**Authors**: Zhicheng Yang, Zhijiang Guo, Yifan Song, Minrui Xu, Yongxin Wang, Yiwei Wang, Xiaodan Liang, Jing Tang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07804)  

**Abstract**: On-policy distillation (OPD) leverages dense teacher rewards to enhance reasoning models. However, scaling OPD to long-horizon tasks exposes a critical flaw: as the student's generated prefix inevitably diverges from the teacher's thought process, the teacher's dense reward loses local exploitability. Continuing to generate and evaluate tokens on these ``drifted'' trajectories not only degrades reward quality but also incurs massive computational waste. To address this, we introduce \textbf{Prune-OPD}, a framework that dynamically aligns training budgets with supervision quality. By continuously monitoring the local compatibility between student and teacher predictions (e.g., via top-$k$ overlap), Prune-OPD detects prefix-drift events in real time. Upon detecting severe drift, it monotonically down-weights subsequent unreliable rewards and triggers dynamic rollout truncation. This allows the training process to halt futile generation and reallocate compute strictly to reliable teacher supervision. Across diverse teacher-student combinations, Prune-OPD consistently aligns computation with supervision reliability. When prefix drift makes dense teacher rewards unreliable, it reduces training time by 37.6\%--68.0\% while preserving, and often improving, performance on challenging benchmarks (AMC, AIME, HMMT). When student-teacher compatibility remains high, it automatically preserves long-context supervision by expanding the training window. These results suggest that Prune-OPD improves OPD not by blindly shortening rollouts, but by reallocating computation toward locally exploitable teacher rewards. 

---
# GazeVLM: Active Vision via Internal Attention Control for Multimodal Reasoning 

**Authors**: Brown Ebouky, Gabriele Carrino, Niccolo Avogaro, Christoph Studer, Andrea Bartezzaghi, Mattia Rigotti  

**Link**: [PDF](https://arxiv.org/pdf/2605.07817)  

**Abstract**: Human visual reasoning is governed by active vision, a process where metacognitive control drives top-down goal-directed attention, dynamically routing foveal focus toward task-relevant details while maintaining peripheral awareness of the global scene. In contrast, modern Vision-Language Models (VLMs) process visual information passively, relying on the static accumulation of massive token contexts that dilute spatial reasoning and induce linguistic hallucinations. Here we propose the following paradigm shift: GazeVLM, a multimodal architecture that internalizes this metacognitive oversight over its deployment of attention resources directly into the reasoning loop. By empowering the VLM to autonomously generate gaze tokens ($\texttt{<LOOK>}$), GazeVLM establishes a top-down control mechanism over its own causal attention mask. The model dynamically dictates its focal intent, triggering a continuous suppression bias that dampens irrelevant visual features, implementing spatial selective attention and simulating foveal fixation. Once local reasoning concludes, the bias lifts, seamlessly restoring the global view. This architecture enables the model to fluidly transition between global spatial awareness and localized focal reasoning without relying on external agentic contraptions like cropping tools, or inflating the context window with additional visual tokens derived from localized visual patches. Trained with a bespoke Group Relative Policy Optimization (GRPO) procedure that rewards valid grounding, our 4B-parameter GazeVLM delivers strong high-resolution multimodal reasoning performance, surpassing state-of-the-art VLMs in its parameter class by nearly 4% and agentic multimodal pipelines built around thinking with images by more than 5% on HRBench-4k and HRBench-8k. 

---
# SOD: Step-wise On-policy Distillation for Small Language Model Agents 

**Authors**: Qiyong Zhong, Mao Zheng, Mingyang Song, Xin Lin, Jie Sun, Houcheng Jiang, Xiang Wang, Junfeng Fang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07725)  

**Abstract**: Tool-integrated reasoning (TIR) is difficult to scale to small language models due to instability in long-horizon tool interactions and limited model capacity. While reinforcement learning methods like group relative policy optimization provide only sparse outcome-level rewards. Recently, on-policy distillation (OPD) has gained popularity by supplying dense token-level supervision from a teacher on student-generated trajectories. However, our experiments indicate that applying OPD to TIR leads to a critical failure mode: erroneous tool calls tend to cascade across subsequent reasoning steps, progressively amplifying student-teacher divergence and rendering the teacher's token-level supervision increasingly unreliable. To address this, we propose SOD, a step-wise on-policy distillation framework for small language model agents, which adaptively reweights distillation strength at each step based on step-level divergence. Therefore, SOD can attenuate potentially misleading teacher signals in high-divergence regions while preserving dense guidance in well-aligned states. Experiments on challenging math, science, and code benchmarks show that SOD achieves up to 20.86% improvement over the second-best baseline. Notably, our 0.6B student achieves 26.13% on AIME 2025, demonstrating effective transfer of agentic reasoning to lightweight models. Our code is available at this https URL. 

---
# Memory-Efficient Looped Transformer: Decoupling Compute from Memory in Looped Language Models 

**Authors**: Victor Conchello Vendrell, Arnau Padres Masdemont, Niccolò Grillo, Jordi Ros-Giralt, Arash Behboodi, Fabio Valerio Massoli  

**Link**: [PDF](https://arxiv.org/pdf/2605.07721)  

**Abstract**: Recurrent LLM architectures have emerged as a promising approach for improving reasoning, as they enable multi-step computation in the embedding space without generating intermediate tokens. Models such as Ouro perform reasoning by iteratively updating internal representations while retaining a standard Key-Value (KV) cache across iterations, causing memory consumption to grow linearly with reasoning depth. Consequently, increasing the number of reasoning iterations can lead to prohibitive memory usage, limiting the practical scalability of such architectures. In this work, we propose Memory-Efficient Looped Transformer (MELT), a novel architecture that decouples reasoning depth from memory consumption. Instead of using a standard KV cache per layer and loop, MELT maintains a single KV cache per layer that is shared across reasoning loops. This cache is updated over time via a learnable gating mechanism. To enable stable and efficient training under this architecture, we propose to train MELT using chunk-wise training in a two phase procedure: interpolated transition, followed by attention-aligned distillation, both from the LoopLM starting model to MELT. Empirically, we show that MELT models fine-tuned from pretrained Ouro parameters outperform standard LLMs of comparable size, while maintaining a memory footprint comparable to those models and dramatically smaller than Ouro's. Overall, MELT achieves constant-memory iterative reasoning without sacrificing LoopLM performance, using only a lightweight post-training procedure. 

---
# The AI-Native Large-Scale Agile Software Development Manifesto 

**Authors**: Ricardo Britto, Fredrik Palmgren, Nishrith Saini, Marcus Ohlin  

**Link**: [PDF](https://arxiv.org/pdf/2605.07717)  

**Abstract**: Despite the widespread adoption of agile methods, achieving true agility at scale remains elusive. Large-scale agile frameworks remain largely human-centric and manual, relying on coordination meetings, artifact synchronization, and role-based handoffs that inhibit real-time adaptation. Meanwhile, rapid advances in AI, particularly large language models, have begun transforming software engineering, yet their potential for organizational-level agility remains underexplored. We present the AI-Native Large-Scale Agile Software Development Manifesto: a set of values and principles that redefine how large-scale software development is organized when AI becomes a first-class participant rather than a peripheral tool. The manifesto is grounded in six principles, parallel processes, intent-driven teams, living knowledge, verification-first assurance, orchestrated agent workforces, and reusable blueprints, that together shift development from a meeting-driven, document-heavy, sequential process to an intelligent, adaptive, continuously learning system. 

---
# An Efficient Hybrid Sparse Attention with CPU-GPU Parallelism for Long-Context Inference 

**Authors**: Feiyu Yao, Zhixiong Niu, Xiaqing Li, Yongqiang Xiong, Juan Fang, Qian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07719)  

**Abstract**: Long-context inference increasingly operates over CPU-resident KV caches, either because decoding-time KV states exceed GPU memory capacity or because disaggregated prefill-decode systems place KV data in host memory. Although block-sparse attention reduces attention cost in this setting, sparsity alone is insufficient for end-to-end efficiency. GPU-only designs remain constrained by PCIe bandwidth and metadata memory overhead, while CPU-GPU hybrid designs still suffer from substantial GPU idle time and bottlenecks in CPU-side top-k selection and sparse attention computation.
Fluxion is built on three key insights: output-aware KV budgeting, head-specific and granularity-aware sparse configuration, and cross-device coordinated execution for sparse attention over CPU-resident KV caches. Guided by these insights, Fluxion combines a lightweight head-property predictor, a granularity-budget selector, and a priority-based scheduler to jointly optimize budget allocation, sparse configuration, and CPU-GPU execution overlap. This co-design enables hybrid sparse attention to achieve both accuracy and system efficiency in long-context inference. Across 2 models, 3 benchmarks, and 40 tasks, Fluxion preserves quality well -- the worst average degradation is only -0.26 relative to FULL, while delivering 1.5$\times$-3.7$\times$ speedup over the strongest fixed sparse hybrid baseline, whose KV budget is only 0.05. 

---
# DRIP-R: A Benchmark for Decision-Making and Reasoning Under Real-World Policy Ambiguity in the Retail Domain 

**Authors**: Hsuvas Borkakoty, Sebastian Pohl, Cheng Wang, Bei Chen, Yufang Hou  

**Link**: [PDF](https://arxiv.org/pdf/2605.07699)  

**Abstract**: LLM-based agents are increasingly deployed for routine but consequential tasks in real-world domains, where their behavior is governed by inherently ambiguous domain policies that admit multiple valid interpretations. Despite the prevalence of such ambiguities in practice, existing agent benchmarks largely assume unambiguous, well-specified policies, leaving a critical evaluation gap. We introduce DRIP-R, a benchmark that systematically exploits real-world retail policy ambiguities to construct scenarios in which no single correct resolution exists. DRIP-R comprises a curated set of policy-ambiguous return scenarios paired with a realistic customer personas, a full-duplex conversational simulation with tool-calling capabilities and a multi-judge evaluation framework covering policy adherence, dialogue quality, behavioral alignment, and resolution quality. Our experiments show that frontier models fundamentally disagree on identical policy-ambiguous scenarios, confirming that ambiguity poses a genuine and systematic challenge to LLM decision-making. 

---
# TRACE: Tourism Recommendation with Accountable Citation Evidence 

**Authors**: Zixu Zhao, Sijin Wang, Yu Hou, Yuanyuan Xu, Yufan Sheng, Xike Xie, Wenjie Zhang, Won-Yong Shin, Xin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2605.07677)  

**Abstract**: Tourism is a high-stakes setting for conversational recommender systems (CRS): a plausible-sounding suggestion can waste real money and trip time once a traveler acts on it. Existing CRS benchmarks primarily evaluate systems with a single Recall@k score over entity mentions, and tourism-specific resources add spatial or knowledge-graph context, yet none of them couple multi-turn recommendation with verbatim review-span evidence and rejection recovery. This leaves an evaluation gap for tourism recommendation that is simultaneously trustworthy, verifiable, and adaptive: recommend the right point of interest (POI) for multi-aspect preferences (such as cuisine, price, atmosphere, walking distance), justify each suggestion with verifiable evidence from prior visitors so the traveler can act without trial and error, and recover when the first recommendation is rejected mid-dialogue. We introduce TRACE, where each item is a multi-turn tourism recommendation dialogue with review-span citations and explicit rejection turns: 10,000 dialogues over 2,400 Yelp POIs and 34,208 reviews across eight U.S. cities, paired with 14 retrieval, planning, and LLM baselines, along with 25 metrics organized under Accuracy, Grounding, and Recovery. Across these baselines, TRACE reveals the Three-Competency Gap: LLM Zero-Shot leads in closed-set Recall@1 and rejection recovery but cites less densely than retrievers; non-LLM retrievers achieve surface-verbatim grounding but with low accuracy; Multi-Review Synthesis fails at recovery. The Grounding Score agrees with human citation precision (Spearman rho=+0.80, p<10^-20), and paired t-tests reproduce the per-baseline ranking (p<0.01 on the dominant contrasts). TRACE reframes accountable tourism recommendation as a joint target (right POI, verifiable evidence, adaptive repair) rather than a single-axis leaderboard. 

---
# Cross-Attention and Encoder-Decoder Transformers: A Logical Characterization 

**Authors**: Veeti Ahvonen, Damian Heiman, Antti Kuusisto, Miguel Moreno, Matias Selin  

**Link**: [PDF](https://arxiv.org/pdf/2605.07705)  

**Abstract**: We give a novel logical characterization of encoder-decoder transformers, the foundational architecture for LLMs that also sees use in various settings that benefit from cross-attention. We study such transformers over text in the practical setting of floating-point numbers and soft-attention, characterizing them with a new temporal logic. This logic extends propositional logic with a counting global modality over the encoder input and a past modality over the decoder input. We also give an additional characterization of such transformers via a type of distributed automata, and show that our results are not limited to the specific choices in the architecture and can account for changes in, e.g., masking. Finally, we discuss encoder-decoder transformers in the autoregressive setting. 

---
# Quality-Conditioned Agreement in Automated Short Answer Scoring: Mid-Range Degradation and the Impact of Task-Specific Adaptation 

**Authors**: Abigail Victoria Gurin Schleifer, Moriah Ariely, Beata Beigman Klebanov, Asaf Salman, Giora Alexandron  

**Link**: [PDF](https://arxiv.org/pdf/2605.07647)  

**Abstract**: Automated short answer scoring (ASAS) is shifting from discriminative, fine-tuned models to large language models (LLMs) used in few-shot settings. This paradigm leverages LLMs broad world knowledge and ease of deployment, but limited task-specific data may reduce alignment on complex scoring tasks. In particular, its impact on scoring partially correct responses that require nuanced interpretation remains underexplored. We investigate the relationship between the degree of task-specific adaptation of different models and quality-conditioned scoring agreement. We compare three LLMs (GPT-5.2, GPT-4o, Claude Opus 4.5) in few-shot mode, a fine-tuned BERT-based encoder, and a human expert on two open-ended biology items, using several hundred student responses and ground truth scores provided by a biology education expert. The results show that human-human agreement is highest and stable across the full quality spectrum. All AI models perform well on fully correct and fully incorrect responses, but exhibit substantial degradation on mid-range responses. This mid-range degradation is conditioned on task-specific adaptation: It is most severe in few-shot LLMs with few examples and decreases as task-specific data increases, with fine-tuned encoder models performing best. This mid-range degradation may lead to inequitable evaluation of responses produced by students with developing understanding. Our findings highlight the importance of quality-conditioned fairness, with particular attention to mid-range responses. 

---
# MAVEN: Multi-Agent Verification-Elaboration Network with In-Step Epistemic Auditing 

**Authors**: Yinsheng Yao, Jiehao Tang, Zhaozhen Yang, Dawei Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.07646)  

**Abstract**: While explicit reasoning trajectories enhance model interpretability, existing paradigms often rely on monolithic chains that lack intermediate verification, allowing early errors to cascade unchecked. This lack of modularity impedes granular auditing and compromises the epistemic trust required for high-stakes applications. We propose MAVEN (Multi-Agent Verification-Elaboration Network with In-Step Epistemic Auditing), a blackboard-inspired framework designed to transform LLMs into deliberate reasoners through explicit role-decoupling. At its core, MAVEN operationalizes an adversarial Skeptic-Researcher-Judge loop, simulating expert deliberation by functionally separating logical defense from factual grounding. Experiments on OpenBookQA, TruthfulQA, HALUEVAL and StrategyQA benchmarks demonstrate that MAVEN delivers superior reasoning quality across four fine-grained metrics. Notably, MAVEN consistently outperforms latent reasoning models such as GEMINI-3.1-Pro and consensus-based baselines (e.g., ReConcile) by generating explicitly structured, modular, and verifiable deliberation trajectories, rather than relying on implicit internal states or post-hoc consensus. Moreover, comprehensive evaluations confirm that MAVEN is fully model-agnostic, serving as a strong and transferable reasoning booster that yields substantial performance improvements across diverse backbone models. 

---
# Post-training makes large language models less human-like 

**Authors**: Marcel Binz, Elif Akata, Abdullah Almaatouq, Mohammed Alsobay, Oleksii Ariasov, Franziska Brändle, David Broska, Jason W. Burton, Nuno Busch, Frederick Callaway, Vanessa Cheung, Brian Christian, Julian Coda-Forno, Can Demircan, Vittoria Dentella, Maria K. Eckstein, Noémi Éltető, Michael Franke, Thomas L. Griffiths, Fritz Günther, Susanne Haridi, Sebastian Hellmann, Stefan Herytash, Linus Hof, Eleanor Holton, Isabelle Hoxha, Zak Hussain, Akshay Jagadish, Elif Kara, Valentin Kriegmair, Evelina Leivada, Li Ji-An, Tobias Ludwig, Maximilian Maier, Marcelo G. Mattar, Marvin Mathony, Alireza Modirshanechi, Robin Na, Mariia Nadverniuk, Antonios Nasioulas, Surabhi S. Nath, Helen Niemeyer, Kate Nussenbaum, Sebastian Olschewski, Thorsten Pachur, Stefano Palminteri, Aliona Petrenco, Camille V. Phaneuf-Hadd, Angelo Pirrone, Manuel Rausch, Laura Raveling, Shashank Reddy, Milena Rmus, Evan M. Russek, Tankred Saanum, Kai Sandbrink, Louis Schiekiera, Johannes A. Schubert, Luca M. Schulze Buschoff, Nishad Singhi, Leah H. Somerville, Mikhail S. Spektor, Xin Sui, Christopher Summerfield, Mirko Thalmann, Anna I. Thoma, Taisiia Tikhomirova, Vuong Truong, Polina Tsvilodub, Konstantinos Voudouris, Robert C. Wilson, Kristin Witte, Shuchen Wu, Dirk U. Wulff, Hua-Dong Xiong, Songlin Xu, Lance Ying, Xinyu Zhang, Jian-Qiao Zhu, Eric Schulz  

**Link**: [PDF](https://arxiv.org/pdf/2605.07632)  

**Abstract**: Large language models (LLMs) are increasingly used as surrogates for human participants, but it remains unclear which models best capture human behavior and why. To address this, we introduce Psych-201, a novel dataset that enables us to measure behavioral alignment at scale. We find that post-training -- the stage that turns base models into useful assistants -- consistently reduces alignment with human behavior across model families, sizes, and objectives. Moreover, this misalignment widens in newer model generations even as base models continue to improve. Finally, we find that persona-induction -- a popular technique for eliciting human-like behavior by conditioning models on participant-specific information -- does not improve predictions at the level of individuals. Taken together, our results suggest that the very processes that are currently employed to turn LLMs into useful assistants also make them less accurate models of human behavior. 

---
# LithoBench: Benchmarking Large Multimodal Models for Remote-Sensing Lithology Interpretation 

**Authors**: Jun Wang, Fengpeng Li, Hang Dong, Tianjin Huang, Wei Han  

**Link**: [PDF](https://arxiv.org/pdf/2605.07640)  

**Abstract**: Remote sensing lithology interpretation is fundamental to geological surveys, mineral exploration, and regional geological mapping. Unlike general land-cover recognition, lithology interpretation is a knowledge-intensive task that requires experts to infer rock types from various features, e.g., subtle visual, spectral, textural, geomorphological, and contextual cues, making reliable automated interpretation highly challenging. Geological knowledge-guided large multimodal models offer new opportunities, yet their evaluation remains constrained by the lack of benchmarks that capture lithological annotations, multi-level geological semantics, and expert-informed assessment. Here, we propose LithoBench, a multi-level benchmark for evaluating geological semantic understanding in remote sensing lithology interpretation. LithoBench contains 10,000 expert-annotated interpretation instances across 12 representative lithological categories, including 4,000 multiple-choice and 6,000 open-ended tasks organized into five cognitive levels: Identification and Description, Comparative Analysis, Mechanism Explanation, Practical Application, and Comprehensive Reasoning. We further develop an expert-in-the-loop, knowledge-grounded semi-automated construction pipeline, coupling multi sub-processes, e.g., structured geological image descriptions, to enhance geological validity and evaluation reliability. Experiments with multiple large vision-language models eveal substantial limitations in geological semantic understanding, particularly on higher-order explanation, application, and reasoning tasks. 

---
# LLM hallucinations in the wild: Large-scale evidence from non-existent citations 

**Authors**: Zhenyue Zhao, Yihe Wang, Toby Stuart, Mathijs De Vaan, Paul Ginsparg, Yian Yin  

**Link**: [PDF](https://arxiv.org/pdf/2605.07723)  

**Abstract**: Large language models (LLMs) are known to generate plausible but false information across a wide range of contexts, yet the real-world magnitude and consequences of this hallucination problem remain poorly understood. Here we leverage a uniquely verifiable object - scientific citations - to audit 111 million references across 2.5 million papers in arXiv, bioRxiv, SSRN, and PubMed Central. We find a sharp rise in non-existent references following widespread LLM adoption, with a conservative estimate of 146,932 hallucinated citations in 2025 alone. These errors are diffusely embedded across many papers but especially pronounced in fields with rapid AI uptake, in manuscripts with linguistic signatures of AI-assisted writing, and among small and early-career author teams. At the same time, hallucinated references disproportionately assign credit to already prominent and male scholars, suggesting that LLM-generated errors may reinforce existing inequities in scientific recognition. Preprint moderation and journal publication processes capture only a fraction of these errors, suggesting that the spread of hallucinated content has outpaced existing safeguards. Together, these findings demonstrate that LLM hallucinations are infiltrating knowledge production at scale, threatening both the reliability and equity of future scientific discovery as human and AI systems draw on the existing literature. 

---
# Operating Within the Operational Design Domain: Zero-Shot Perception with Vision-Language Models 

**Authors**: Berkehan Ünal, Dierend Hauke, Fazlija Dren, Plachetka Christopher  

**Link**: [PDF](https://arxiv.org/pdf/2605.07649)  

**Abstract**: Over the last few years, research on autonomous systems has matured to such a degree that the field is increasingly well-positioned to translate research into practical, stakeholder-driven use cases across well-defined domains. However, for a wide-scale practical adoption of autonomous systems, adherence to safety regulations is crucial. Many regulations are influenced by the Operational Design Domain (ODD), which defines the specific conditions in which an autonomous agent can function. This is especially relevant for Automated Driving Systems (ADS), as a dependable perception of ODD elements is essential for safe implementation and auditing. Vision-language models (VLMs) integrate visual recognition and language reasoning, functioning without task-specific training data, which makes them suitable for adaptable ODD perception. To assess whether VLMs can function as zero-shot "ODD sensors" that adapt to evolving definitions, we contribute (i) an empirical study of zero-shot ODD classification and detection using four VLMs on a custom dataset and Mapillary Vistas, along with failure analyses; (ii) an ablation of zero-shot optimization strategies with a cost-performance overview; and (iii) a suite of reusable prompting templates with guidance for adaptation. Our findings indicate that definition-anchored chain-of-thought prompting with persona decomposition performs best, while other methods may result in reduced recall. Overall, our results pave the way for transparent and effective ODD-based perception in safety-critical applications. 

---
# Safe, or Simply Incapable? Rethinking Safety Evaluation for Phone-Use Agents 

**Authors**: Zhengyang Tang, Yi Zhang, Chenxin Li, Xin Lai, Pengyuan Lyu, Yiduo Guo, Weinong Wang, Junyi Li, Yang Ding, Huawen Shen, Zhengyao Fang, Xingran Zhou, Liang Wu, Fei Tang, Sunqi Fan, Shangpin Peng, Zheng Ruan, Anran Zhang, Benyou Wang, Chengquan Zhang, Han Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07630)  

**Abstract**: When a phone-use agent avoids harm, does that show safety, or simply inability to act? Existing evaluations often cannot tell. A harmful outcome may be avoided because the agent recognized the risk and chose the safe action, or because it failed to understand the screen or execute any relevant action at all. These cases have different causes and call for different fixes, yet current benchmarks often merge them under task success, refusal, or final harmful outcome. We address this problem with PhoneSafety, a benchmark of 700 safety-critical moments drawn from real phone interactions across more than 130 apps. Each instance isolates the next decision at a risky moment and asks a simple question: does the model take the safe action, take the unsafe action, or fail to do anything useful? We evaluate eight representative phone-use agents under this framework. Our results reveal two main patterns. First, stronger general phone-use ability does not reliably imply safer choices at risky moments. Models that perform better on ordinary app tasks are not always the ones that behave more safely when the next action matters. Second, failures to do anything useful behave like a capability signal rather than a safety signal: they are concentrated in more visually and operationally demanding settings and remain stable when the evaluation protocol changes. Across models, failures split into two recurring patterns: unsafe choices in settings where the model can act but chooses wrongly, and inability to act in more visually and operationally demanding screens. Overall, a harmless outcome is not enough to count as evidence of safety. Evaluating phone-use agents requires separating unsafe judgment from inability to act. 

---
# Your Language Model is Its Own Critic: Reinforcement Learning with Value Estimation from Actor's Internal States 

**Authors**: Yunho Choi, Jongwon Lim, Woojin Ahn, Minjae Oh, Jeonghoon Shim, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2605.07579)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) for Large Reasoning Models hinges on baseline estimation for variance reduction, but existing approaches pay a heavy price: PPO requires a policy-model scale critic, while GRPO needs multiple rollouts per prompt to keep its empirical group mean stable. We introduce Policy Optimization with Internal State Value Estimation), which obtains a baseline at negligible cost by using the policy model's internal signals already computed during the policy forward pass. A lightweight probe predicts the expected verifiable reward from the hidden states of the prompt and generated trajectory, as well as token-entropy statistics, and is trained online alongside the policy. To preserve gradient unbiasedness despite using trajectory-conditioned features, we introduce a cross-rollout construction that predicts each rollout's value from an independent rollout's internal states. Because POISE estimates prompt value using only a single rollout, it enables higher prompt diversity for a fixed compute budget during training. This reduces gradient variance for more stable learning and also eliminates the compute overhead of sampling costs for detecting zero-advantage prompts. On Qwen3-4B and DeepSeek-R1-Distill-Qwen-1.5B across math reasoning benchmarks, POISE matches DAPO while requiring less compute. Moreover, its value estimator shows similar performance to a separate LLM-scale value model and generalizes to various verifiable tasks. By leveraging the model's own internal representations, POISE enables more stable and efficient policy optimization. 

---
# Benchmarking EngGPT2-16B-A3B against Comparable Italian and International Open-source LLMs 

**Authors**: Andrea Sassella, Andrea Chizzola, Tommaso Bianchi, Luca Alessandrelli, Mark James Carman  

**Link**: [PDF](https://arxiv.org/pdf/2605.07731)  

**Abstract**: This report benchmarks the performance of ENGINEERING Ingegneria Informatica S.p.A.'s EngGPT2MoE-16B-A3B LLM, a 16B parameter Mixture of Experts (MoE) model with 3B active parameters. Performance is investigated across a wide variety of representative benchmarks, and is compared against comparably-sized open-source MoE and dense models. In comparison with popular Italian models, namely FastwebMIIA-7B, Minerva-7B, Velvet-14B, and LLaMAntino-3-ANITA-8B, EngGPT2MoE-16B-A3B performs as well or better on international benchmarks: ARC-Challenge, GSM8K, AIME24, AIME25, MMLU, and HumanEval (HE). It achieves the best performance for the longest context setting (32k) of the RULER benchmark. On the Italian benchmark dataset ITALIC, the model performs as well or better than the other models except for Velvet-14B, which outperforms it. Compared with popular MoE models of comparable size, the new model reports higher values than DeepSeek-MoE-16B-Chat on all considered benchmarks. It has higher values than Moonlight-16B-A3B on HE, MMLU, AIME24, AIME25, GSM8K, and the 32k RULER setting, but lower on BFCL and some ARC and ITALIC settings. Finally it has lower values than GPT-OSS-20B on most benchmarks, including HE, MMLU, AIME24, AIME25, GSM8K, ARC, BFCL, and the RULER 32k. When compared with popular dense models, EngGPT2MoE-16B-A3B reports higher values on AIME24 and AIME25 than Llama-3.1-8B-Instruct, Gemma-3-12b-it, and Ministral-3-8BInstruct-2512-BF16, but lower values on ITALIC, BFCL, and RULER with a 32k context. When performance is aggregated across all benchmark metrics, EngGPT2MoE-16B-A3B shows higher performance than the Italian models under evaluation while achieving lower results than some of the most performant international models, in particular GPT-5 nano and Qwen3-8B. Taken together, our findings find the new model to be a step forward for native Italian Large Language Models. 

---
# Revisiting Transformer Layer Parameterization Through Causal Energy Minimization 

**Authors**: Jin Xu, Camille Couturier, Victor Rühle, Saravan Rajmohan, James Hensman  

**Link**: [PDF](https://arxiv.org/pdf/2605.07588)  

**Abstract**: Transformer blocks typically combine multi-head attention (MHA) for token mixing with gated MLPs for token-wise feature transformation, yet many choices in their parameterization remain largely empirical. We introduce Causal Energy Minimization (CEM), a framework that recasts Transformer layers as optimization steps on conditional energy functions while explicitly accounting for layer parameterization. Extending prior energy-based interpretations of attention, CEM shows that weight-tied MHA can be derived as a gradient update on an interaction energy, and that a gated MLP with shared up/down projections can be viewed through an element-wise energy. This perspective identifies a design space for Transformer layers that includes within-layer weight sharing, diagonal-plus-low-rank interactions, lightweight preconditioners, and recursive updates. We evaluate CEM-derived layers in language-modeling experiments at the moderate hundred-million-parameter scale. Despite their constrained parameterizations, these layers train stably and can match corresponding Transformer baselines. Overall, our results suggest that CEM provides a useful lens for understanding Transformer layer parameterization, connecting Transformer architectures to energy-based models and motivating further exploration of energy-guided layer designs. 

---
# LARAG: Link-Aware Retrieval Strategy for RAG Systems in Hyperlinked Technical Documentation 

**Authors**: Giorgia Bolognesi, Claudio Estatico, Ulderico Fugacci, Isabella Mastroianni, Claudio Muselli, Luca Oneto  

**Link**: [PDF](https://arxiv.org/pdf/2605.07517)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances the factual grounding of Large Language Models by conditioning their outputs on external documents. However, standard embedding-based retrievers treat naturally structured corpora, such as technical manuals, as flat collections of passages, thereby overlooking the hyperlink topology that users rely on when navigating such content.
We introduce LARAG (Link-Aware RAG): a lightweight, link-aware retrieval strategy that leverages the author-defined hyperlink structure already present in HTML documentation, encoding hyperlink relations as metadata in the chunk representations and exploiting them to perform a form of graph-like retrieval of locally relevant content.
In a benchmark of twenty expert-designed queries over Rulex Platform technical documentation and four prompting strategies, LARAG consistently improves answer quality, achieving the highest BERTScore F1, while retrieving fewer chunks and generating fewer tokens than a baseline RAG architecture used for comparison. These results show that directly leveraging the existing hyperlink topology of technical documentation, even without explicit graph construction or inference, enables an implicit form of graph-like retrieval that yields a more faithful and efficient RAG pipeline, providing better grounding at lower cost. 

---
# Curated Synthetic Data Doesn't Have to Collapse: A Theoretical Study of Generative Retraining with Pluralistic Preferences 

**Authors**: Ali Falahati, Mohammad Mohammadi Amiri, Kate Larson, Lukasz Golab  

**Link**: [PDF](https://arxiv.org/pdf/2605.07724)  

**Abstract**: Recursive retraining of generative models poses a critical representation challenge: when synthetic outputs are curated based on a fixed reward signal, the model tends to collapse onto a narrow set of outputs that over-optimize that objective. Prior work suggests that such collapse is unavoidable without adding real data into the mix. We revisit this conclusion from an alignment perspective and show that collapse can be mitigated through curation based on multiple reward functions. We formalize the dynamics of recursive training under heterogeneous preferences and prove that, under certain conditions, the model converges to a stable distribution that allocates probability mass across competing high-reward regions. The limiting distribution preserves diversity and provably satisfies a weighted Nash bargaining solution, offering a formal interpretation of value aggregation in synthetic retraining loops. 

---
# Mathematical Reasoning via Intervention-Based Time-Series Causal Discovery Using LLMs as Concept Mastery Simulators 

**Authors**: Tsuyoshi Okita  

**Link**: [PDF](https://arxiv.org/pdf/2605.07600)  

**Abstract**: Recent methods for improving LLM mathematical reasoning, whether through MCTS-based test-time search or causal graph-guided knowledge injection, cannot identify which concepts causally contribute to a correct answer, as the observed association may be spurious, driven by confounders such as problem difficulty.
We propose CIKA (Causal Intervention for Knowledge Activation), a framework that uses the LLM itself as an interventional simulator: a prompt sets the concept state to ``mastered'' and the correctness change estimates the causal effect. We formalize this quantity as an Interventional Capability Probe (ICP), which diagnoses whether the LLM can use a given concept -- distinct from merely possessing knowledge. Because the intervention exogenously sets the concept state independently of problem difficulty, ICP separates confounding that observational methods cannot.
On 67 screened problems, the ICP of the top-ranked concept (+0.219) is significantly larger than that of the negative control (+0.039; paired $t$-test, $p < 10^{-6}$, Cohen's $d = 0.86$), confirming that the probe discriminates causally relevant concepts from irrelevant ones. Analysis of 601 Omni-MATH problems further shows that solved problems have 6.1$\times$ higher ATE than unsolved ones (0.338 vs. 0.055), confirming that ICP is predictive of problem-solving success. With a 7B-parameter LLM whose weights are entirely frozen, CIKA achieves 69.7\% on the contamination-free Omni-MATH-Rule benchmark and 64.0\% overall, compared to 60.5\% for o1-mini, and 97.2\% on GSM8K, 46--50\% on AIME 2024--2026, and 46.2\% on MathArena. The Causal Knowledge Activation component contributes 33.8\% of correct answers on problems where the base model alone fails, demonstrating that the LLM already possessed but had not activated the requisite knowledge. 

---
# Vaporizer: Breaking Watermarking Schemes for Large Language Model Outputs 

**Authors**: Jonathan Hong Jin Ng, Anh Tu Ngo, Anupam Chattopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2605.07481)  

**Abstract**: In this paper, we investigate the recent state-of-the-art schemes for watermarking large language models (LLMs) outputs. These techniques are claimed to be robust, scalable and production-grade, aimed at promoting responsible usage of LLMs. We analyse the effectiveness of these watermarking techniques against an extensive collection of modified text attacks, which perform targeted semantic changes without altering the general meaning of the text content. Our approach encompasses multiple attack strategies, which include lexical alterations, machine translation, and even neural paraphrasing. The attack efficacy is measured with two target criteria - successful removal of the watermark and preservation of semantic content. We evaluate semantic preservation through BERT scores, text complexity measures, grammatical errors, and Flesch Reading Ease indices. The experimental results reveal varying levels of effectiveness among different watermarking models, with the same underlying result that it is possible to remove the watermark with reasonable effort. This study sheds light on the strengths and weaknesses of existing LLM watermarking systems, suggesting how they should be constructed to improve security of available schemes. 

---
# Response-G1: Explicit Scene Graph Modeling for Proactive Streaming Video Understanding 

**Authors**: Ke Ma, Jiaqi Tang, Bin Guo, Xueting Han, Ruonan Xu, Qingfeng He, Ziheng Wang, Xu Wang, Qifeng Chen, Zhiwen Yu, Yunhao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07575)  

**Abstract**: Proactive streaming video understanding requires Video-LLMs to decide when to respond as a video unfolds, a task where existing methods often fall short due to their implicit, query-agnostic modeling of visual evidence. We introduce Response-G1, a novel framework that establishes explicit, structured alignment between the accumulated video evidence and the query's expected response conditions via scene graphs. The framework operates in three fine-tuning-free stages: (1) online query-guided scene graph generation from streaming clips; (2) memory-based retrieval of the most semantically relevant historical scene graphs; and (3) retrieval-augmented trigger prompting for per-frame "silence/response" this http URL grounding both evidence and conditions in a shared graph representation, Response-G1 achieves more interpretable and accurate response timing decisions. Experimental results on established benchmarks demonstrate the superiority of our method in both proactive and reactive tasks, validating the advantage of explicit scene graph modeling and retrieval in streaming video understanding. 

---
# The Moltbook Files: A Harmless Slopocalypse or Humanity's Last Experiment 

**Authors**: William Brach, Federico Torrielli, Stine Lyngsø Beltoft, Annemette Brok Pirchert, Peter Schneider-Kamp, Lukas Galke Poech  

**Link**: [PDF](https://arxiv.org/pdf/2605.07462)  

**Abstract**: Moltbook is a Reddit-like platform where OpenClaw agents post, comment, and vote at scale - a so far unprecedented incident that comes with serious safety concerns. With the aim of studying emergent behavior in populations, we release the Moltbook Files, a dataset of 232k posts and 2.2M comments covering the platform's first 12 days, processed through a pipeline to identify and remove Personally-Identifiable Information (PII). We analyze community structure, authorship, lexical properties, sentiment, topics, semantic geometry, and comment interaction. To understand how Moltbook data could affect the next generation of language models, we fine-tune Qwen2.5-14B-Instruct on Moltbook Files with three adaptation levels. Our PII pipeline reveals that agents post API keys, passwords, BIP39 seed phrases on Moltbook, a publicly indexed platform. The overall sentiment is mostly neutral and mildly positive (66.6% neutral, 19.5% positive) and shows a tendency for self-referential linking. We find that fine-tuning on Moltbook data reduces truthfulness from 0.366 to 0.187. However, a model fine-tuned on a size-matched Reddit dataset produces a comparable decrease. Moltbook thus seems to be more of a harmless slopocalypse. However, tail risks remain, including agent affordances, contamination of future crawls through self-links, and potential transfer of traits to the next generation of language models. More broadly, our findings highlight the importance of control baselines in emergent misalignment evaluations. 

---
# SHRED: Retain-Set-Free Unlearning via Self-Distillation with Logit Demotion 

**Authors**: Zizhao Hu, Ameya Godbole, Johnny Tian-Zheng Wei, Mohammad Rostami, Jesse Thomason, Robin Jia  

**Link**: [PDF](https://arxiv.org/pdf/2605.07482)  

**Abstract**: Machine unlearning for large language models (LLMs) aims to selectively remove memorized content such as private data, copyrighted text, or hazardous knowledge, without costly full retraining. Most existing methods require a retain set of curated examples to prevent catastrophic degradation of general model utility, creating an extra data dependency that complicates deployment. We propose SHRED (Self-distillation via High-surprisal-only Retain-set-free Entropy Demotion), a retain-set-free unlearning method built on a key insight: not all tokens within a forget set instance carry memorized information equally. High-information tokens concentrate the model's memorized knowledge, while low-information tokens reflect general language competence. SHRED operates in two stages. (1) Selection: We perform a forward pass on a forget set instance, collect per-token autoregressive probabilities, and select the bottom (lowest probability, highest Shannon information) as forget positions; the remaining positions are retained as benign anchors. (2) Training: We construct modified KL targets that demote the memorized token's logit at forget positions while preserving the original distribution at benign positions. The model is then trained via a single top KL self-distillation objective that simultaneously drives forgetting and utility preservation. We evaluate SHRED across four standard unlearning benchmarks and demonstrate that it establishes a new Pareto-optimal trade-off between forget efficacy and model utility, outperforming retain-set-dependent methods. Our analysis shows that SHRED is robust against relearning attacks and membership-inference attacks, and it maintains stable utility even after many sequential unlearning runs. 

---
# HBEE: Human Behavioral Entropy Engine -- Pre-Registered Multi-Agent LLM Simulation of Peer-Suspicion-Based Detection Inversion 

**Authors**: Vickson Ferrel  

**Link**: [PDF](https://arxiv.org/pdf/2605.07472)  

**Abstract**: Insider threat detection assumes that an adaptive insider leaves behavioral residue distinguishing them from legitimate users. We test this assumption against an LLM-driven adaptive insider in a controlled multi-agent simulator. Our pre-registered five-condition study isolates defender mode (cascade vs. blind UEBA) crossed with adversary type (naive vs. adaptive OPSEC) plus a no-mole control, across 100 runs (95 valid after pre-committed exclusions). The primary finding is a detection inversion: at T_60, the adaptive mole's suspicion in-degree is statistically lower than a randomly selected innocent agent (Cliff's delta = -0.694, 95% BCa CI [-0.855, -0.519], Mann-Whitney p << 0.01). The pre-registered prediction was the opposite direction. A pre-registered equivalence test (H2) shows adaptive OPSEC produces no detectable shift in the mole's UEBA rank under either defender mode. The two detection signals (peer suspicion graph in-degree and per-agent UEBA rank) decouple under adaptive adversary behavior. We bound generalization explicitly: a pre-registered Gini calibration check (H4) returns FAIL, with HBEE pairwise message-exposure Gini (0.213) diverging from the SNAP Enron reference (0.730) by |Delta Gini| = 0.52, exceeding the equivalence bound by 5x. The paper makes a narrow but surprising claim: in a controlled environment where adaptive OPSEC is implementable as an LLM directive, peer-suspicion-cascade detection inverts. We release the simulator, pre-registration document, frozen scenarios, raw telemetry, and analysis pipeline under an open-source license. 

---
# Unsolvability Ceiling in Multi-LLM Routing: An Empirical Study of Evaluation Artifacts 

**Authors**: Saloni Garg, Amit Sagtani  

**Link**: [PDF](https://arxiv.org/pdf/2605.07395)  

**Abstract**: Efficient routing across multiple LLMs enables cost-quality tradeoffs by directing queries to the cheapest capable model. Prior work attributes routing headroom to an "unsolvability ceiling", queries no model in the pool can solve. We present a large-scale study of multi-tier LLM routing with 206,000 query-model pairs across six benchmarks (MMLU, MedQA, HumanEval, MBPP, Alpaca, ShareGPT) using the Gemma 4 and Llama 3.1 families. Evaluating with both LLM-as-a-judge and exact-match metrics, we show that a substantial portion of reported unsolvability stems from evaluation artifacts: (i) systematic judge biases favoring verbosity over correctness, (ii) truncation under fixed generation budgets, and (iii) output format mismatches. Through dual-judge validation and exact-match grounding, we reduce measured unsolvability across tasks. We introduce a decomposition framework attributing failures to these artifacts, revealing consistent patterns across domains and model families. These artifacts also distort router training signals: standard routers collapse to majority-class prediction (~79% smallest-tier optimal), confirmed via random-feature and shuffled-label controls, incurring a 13-17 percentage point opportunity cost. We provide actionable recommendations including dual-judge validation, exact-match anchoring, and cost-sensitive objectives. Our findings suggest existing routing headroom estimates are substantially inflated, underscoring the need for reliable evaluation protocols in multi-LLM systems. 

---
# Prompt Engineering Strategies for LLM-based Qualitative Coding of Psychological Safety in Software Engineering Communities: A Controlled Empirical Study 

**Authors**: Moaath Alshaikh, Tasneem Alshaher, Ricardo Vieira, Beatriz Santana, Clelio Xavier, Jose Amancio, Glauco Carneiro, Julio Leite, Savio Freire, Manoel Mendonca  

**Link**: [PDF](https://arxiv.org/pdf/2605.07422)  

**Abstract**: Qualitative analysis plays a pivotal role in understanding the human and social aspects of software engineering. However, it remains a demanding process shaped by the subjective interpretation of individual researchers and sensitive to methodological choices such as prompt design. Recent advancements in Large Language Models (LLMs) offer promising opportunities to support this type of analysis, although their reliability in reproducing human qualitative reasoning under varying prompting conditions remains largely untested. This study presents a controlled empirical evaluation of three LLMs -- Claude Haiku, DeepSeek-Chat, and Gemini 2.5 Flash -- across two prompt engineering strategies (zero-shot and multi-shot closed coding), using Cohen's kappa as the primary agreement metric over ten independent runs per configuration. Results suggest that multi-shot prompting significantly improves agreement for Claude Haiku (Delta kappa = +0.034, Wilcoxon p = 0.004) but not for DeepSeek-Chat or Gemini 2.5 Flash. Intra-model stability varies substantially -- DeepSeek-Chat and Claude Haiku exhibit the lowest variance (SD approx. 0.017), while Gemini 2.5 Flash is the least stable (SD = 0.038). A systematic over-prediction of "Sharing Negative Feedback" is identified across all models (bias ratios up to 5.25x), alongside consistent under-prediction of "Expressing Concerns." Collectively, these findings provide empirical evidence for prompt engineering guidelines in LLM-assisted qualitative coding for software engineering research. 

---
# BalCapRL: A Balanced Framework for RL-Based MLLM Image Captioning 

**Authors**: Shaokai Ye, Vasileios Saveris, Yihao Qian, Jiaming Hu, Elmira Amirloo, Peter Grasch  

**Link**: [PDF](https://arxiv.org/pdf/2605.07394)  

**Abstract**: Image captioning is one of the most fundamental tasks in computer vision. Owing to its open-ended nature, it has received significant attention in the era of multimodal large language models (MLLMs). In pursuit of ever more detailed and accurate captions, recent work has increasingly turned to reinforcement learning (RL). However, existing captioning-RL methods and evaluation metrics often emphasize a narrow notion of caption quality, inducing trade-offs across core dimensions of captioning. For example, utility-oriented objectives can encourage noisy, hallucinated, or overlong captions that improve downstream question answering while harming fluency, whereas arena-style objectives can favor fluent but generic descriptions with limited usefulness. To address this, we propose a more balanced RL framework that jointly optimizes utility-aware correctness, reference coverage, and linguistic quality. In order to effectively optimize the resulting continuous multi-objective reward formulation, we apply GDPO-style reward-decoupled normalization to continuous-valued captioning rewards and show that it improves performance over vanilla GRPO. Additionally, we introduce length-conditional reward masking, yielding a more suitable length penalty for captioning. Across LLaVA-1.5-7B and Qwen2.5-VL 3B and 7B base models, our method consistently improves caption quality, with peak gains of +13.6 DCScore, +9.0 CaptionQA, and +29.0 CapArena across different models. 

---
# OrchJail: Jailbreaking Tool-Calling Text-to-Image Agents by Orchestration-Guided Fuzzing 

**Authors**: Jianming Chen, Yawen Wang, Junjie Wang, Zhe Liu, Qing Wang, Fanjiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07414)  

**Abstract**: Tool-calling text-to-image (T2I) agents can plan and execute multi-step tool chains to accomplish complex generation and editing queries. However, this capability introduces a new safety attack surface: harmful outputs may arise from tool orchestration, where individually benign steps combine into unsafe results, making prompt-only jailbreak techniques insufficient. We present OrchJail, an orchestration-guided fuzzing framework for jailbreaking tool-calling T2I agents. Its core idea is to exploit high-risk tool-orchestration patterns: by learning from successful jailbreak tool-calling traces and their causal relationships to prompt wording, OrchJail directly guides the fuzzing search toward prompts that are more likely to trigger unsafe multi-step tool behaviors, rather than relying on surface-level textual perturbations. Extensive experiments demonstrate that OrchJail improves jailbreak effectiveness and efficiency across representative toolcalling T2I agents, achieving higher attack success rates, better image fidelity, and lower query costs, while remaining robust against common jailbreak defenses. Our work highlights tool orchestration as a critical, previously unexplored attack surface and provides a novel framework for uncovering safety risks in T2I agents. 

---
# MISA: Mixture of Indexer Sparse Attention for Long-Context LLM Inference 

**Authors**: Ruijie Zhou, Fanxu Meng, Yufei Xu, Tongxuan Liu, Guangming Lu, Muhan Zhang, Wenjie Pei  

**Link**: [PDF](https://arxiv.org/pdf/2605.07363)  

**Abstract**: DeepSeek Sparse Attention (DSA) sets the state of the art for fine-grained inference-time sparse attention by introducing a learned token-wise indexer that scores every prefix token and selects the most relevant ones for the main attention. To remain expressive, the indexer uses many query heads (for example, 64 on DeepSeek-V3.2) that share the same selected token set; this multi-head design is precisely what makes the indexer the dominant cost on long contexts. We propose MISA (Mixture of Indexer Sparse Attention), a drop-in replacement for the DSA indexer that treats its indexer heads as a pool of mixture-of-experts. A lightweight router uses cheap block-level statistics to pick a query-dependent subset of only a few active heads, and only those heads run the heavy token-level scoring. This preserves the diversity of the original indexer pool while reducing the per-query cost from scoring every prefix token with every head to scoring it with only a handful of routed heads, plus a negligible router term computed on a small set of pooled keys. We further introduce a hierarchical variant of MISA that uses the routed pass to keep an enlarged candidate set and then re-ranks it with the original DSA indexer to recover the final selected tokens almost exactly. With only eight active heads and no additional training, MISA matches the dense DSA indexer on LongBench across DeepSeek-V3.2 and GLM-5 while running with eight and four times fewer indexer heads respectively, and outperforms HISA on average. It also preserves fully green Needle-in-a-Haystack heatmaps up to a 128K-token context and recovers more than 92% of the tokens selected by the DSA indexer per layer. Our TileLang kernel delivers roughly a 3.82 times speedup over DSA's original indexer kernel on a single NVIDIA H200 GPU. 

---
# CSR: Infinite-Horizon Real-Time Policies with Massive Cached State Representations 

**Authors**: Robin Karlsson, Go Suzui  

**Link**: [PDF](https://arxiv.org/pdf/2605.07325)  

**Abstract**: Deploying massive large language models (LLMs) as continuous cognitive engines for robotics is bottlenecked by the time-to-first-token (TTFT) latency required to process extensive state histories. Existing solutions like RAG or sliding windows compromise global context or incur prohibitive re-computation costs. We formalize the optimal task structure for minimizing latency and theoretically prove that prefix stability, incremental extensibility, and asynchronous state reconciliation are necessary conditions for real-time performance. Building on these proofs, we introduce the Cached State Representation (CSR) framework as the practical instantiation of these properties, ensuring optimal KV-cache reuse. To sustain these properties over infinite horizons, we further propose an Asynchronous State Reconciliation (ASR) algorithm that offloads state memory eviction to a parallel computational resource to eliminate latency spikes. On a physical robot wirelessly connected to an on-premise GPU server, CSR achieves a 26-fold latency reduction (14.67s to 0.56s) for 120K token contexts with a 235B parameter model compared to a standard baseline. On an embodied AI benchmark, we achieve SOTA recall (0.836 vs. 0.459) while maintaining RAG-level latency. ASR is validated to sustain bounded, spike-free TTFT over 10 eviction cycles in continuous real-world operation. Together, CSR and ASR enable massive LLMs to function as continuously operating, high-frequency (> 2 Hz) embodied policies. 

---
# TTF: Temporal Token Fusion for Efficient Video-Language Model 

**Authors**: Simin Huo, Ning LI  

**Link**: [PDF](https://arxiv.org/pdf/2605.07355)  

**Abstract**: Video-language models (VLMs) face rapid inference costs as visual token counts scale with video length. For example, 32 frames at $448{\times}448$ resolution already yield >8,000 visual tokens in Qwen3-VL, making LLM prefill the dominant throughput bottleneck. Existing methods often rely on global similarity or attention-guided compression, incurring offsets to their gains. We propose \textbf{Temporal Token Fusion (TTF)}, a training-free, plug-and-play pre-LLM token compression framework that exploits structured temporal redundancy in video. TTF automatically selects an anchor frame, then for each subsequent frame, performs a local window similarity search (e.g.,$3\times 3$), fusing tokens that exceed a threshold. The compressed sequence maintains positional consistency across both prefill and decoding through coordinate realignment, enabling seamless integration with existing VLM pipelines. On Qwen3-VL-8B with threshold t=0.70, TTF removes about 67\% of visual tokens while retaining 99.5\% of the baseline accuracy and introducing only ${\approx}0.16$\,GFLOPs of matching overhead. Overall, TTF offers a practical, efficient solution for video understanding. The code is available at \href{this https URL}{this https URL} 

---
# DCGL: Dual-Channel Graph Learning with Large Language Models for Knowledge-Aware Recommendation 

**Authors**: Xinchi Zou, Tongzhenzhi Su, Jianjun Li, Yuan Fu, Chang Liu, Zhiying Deng, Zhiwei Shen  

**Link**: [PDF](https://arxiv.org/pdf/2605.07314)  

**Abstract**: Knowledge Graphs (KGs) have proven highly effective for recommendation systems by capturing latent item relationships, while recent integration of Large Language Models (LLMs) has further enhanced semantic understanding and addressed knowledge sparsity issues. Nevertheless, current KG-and-LLM-based methods still face three main limitations: 1) inadequate modeling of implicit semantic relationships beyond explicit KG links; 2) suboptimal single-channel fusion of ID and LLM embeddings, which often leads to signal interference and blurred representations; and 3) insufficient consideration of user-item interaction frequency variations in recommendation strategies. To address these challenges, we propose the Dual-Channel Graph Learning (DCGL) framework, featuring three key innovations: 1) a dual-channel architecture that structurally decouples rich semantic information from user behavioral patterns, preventing early interference; 2) a multi-level contrastive learning mechanism that enhances robustness against KG noise through intra-view contrasts and bridges semantic gaps between channels via inter-view alignment; and 3) a dynamic fusion mechanism that adaptively balances semantic generalization and behavioral specificity based on interaction frequency, resolving the cascading limitation. Extensive experiments on four real-world datasets show that DCGL consistently outperforms state-of-the-art methods, yielding substantial improvements in sparse scenarios while maintaining precision for active users. Our code is available at this https URL. 

---
# ForgeVLA: Federated Vision-Language-Action Learning without Language Annotations 

**Authors**: Yuhao Zhou, Yunpeng Zhu, Yang Zhou, Jindi Lyu, Jian Lan, Zhangyuan Wang, Dan Si, Thomas Seidl, Qing Ye, Jiancheng Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07474)  

**Abstract**: Vision-Language-Action (VLA) models hold great promise for general-purpose robotic intelligence, yet scaling up such models is severely bottlenecked by the high cost of acquiring annotated training data. Fortunately, vision-equipped robots deployed across various domains already produce abundant vision-action pairs that can be leveraged to scale up VLA training more efficiently. However, these raw data cannot be centrally aggregated due to various constraints and also exhibit severe heterogeneity. To address these challenges, in this paper, we propose ForgeVLA, a federated VLA training framework that learns VLA models from distributed vision-action pairs without centralizing raw data or requiring manual annotations. Specifically, each client in ForgeVLA is equipped with an embodied instruction classifier that maps vision-action pairs to a predefined instruction set, recovering the missing language modality and forming complete vision-language-action triplets. Beyond triplet construction, we also identify vision-language feature collapse as a critical challenge that has been largely overlooked in prior federated VLA research. To mitigate this issue, ForgeVLA combines a client-side contrastive planning loss with a server-side adaptive aggregation strategy to learn task-discriminative representations efficiently. Extensive experiments across multiple benchmarks show that ForgeVLA significantly outperforms other baselines, and ablation studies further validate the contribution of each component. 

---
# Mage: Multi-Axis Evaluation of LLM-Generated Executable Game Scenes Beyond Compile-Pass Rate 

**Authors**: Hugh Xuechen Liu, Kıvanç Tatar  

**Link**: [PDF](https://arxiv.org/pdf/2605.07342)  

**Abstract**: Compile-pass rate is the dominant evaluation signal for LLM code generation, yet for multi-component domain-specific artifacts it can be actively misleading. We demonstrate this on executable game scene synthesis with a four-axis evaluation protocol (named `Mage') -- compile success, runtime success, structural fidelity, and mechanism adherence -- applied to 858 generation attempts across four open-weight LLMs (7B--30B), 26~hand-crafted Unity goal pattern playable concepts, and two automatically extracted IR granularity levels. Direct NL-to-C\# generation achieves the highest runtime-pass rate (43\% mean) yet produces structurally vacuous scenes (mechanism $F_1 \approx 0.12$). Structural IR conditioning halves the runtime rate but recovers domain-faithful structure ($F_1$ up to 1.00). Within IR conditioning, behavior-only and full-scene granularity are statistically indistinguishable (McNemar $p = 1.0$), indicating input-level granularity saturation. These results show that compile rate is anti-correlated with functional correctness in this domain and that multi-axis evaluation is necessary to detect the divergence. We release the benchmark, replay logs, and per-record metrics for independent verification. 

---
# BioProVLA-Agent: An Affordable, Protocol-Driven, Vision-Enhanced VLA-Enabled Embodied Multi-Agent System with Closed-Loop-Capable Reasoning for Biological Laboratory Manipulation 

**Authors**: Zhaohui Du, Zhe Wang, Hongmei Fei, Xiwen Cao, Ting Xiao, Qi Wang, Huanbo Jin, Jiaming Gu, Quan Lu, Zhe Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07306)  

**Abstract**: Biological laboratory automation can reduce repetitive manual work and improve reproducibility, but reliable embodied execution in wet-lab environments remains challenging. Protocols are often unstructured, labware is frequently transparent or reflective, and multi-step procedures require state-aware execution beyond one-shot instruction following. Existing robotic systems often rely on costly hardware, fixed workflows, dedicated instruments, or robotics-oriented interfaces. Here, we introduce BioProVLA-Agent, an affordable, protocol-driven, vision-enhanced embodied multi-agent system enabled by Vision-Language-Action (VLA) models for biological manipulation. The system uses protocols as the task interface and integrates protocol parsing, visual state verification, and embodied execution in a closed-loop workflow. A Tailored LLM Protocol Agent converts protocols into verifiable subtasks; a VLM-RAG Verification Agent assesses readiness and completion using observations, robot states, retrieved knowledge, and success/failure examples; and a VLA Embodied Agent executes verified subtasks through a lightweight policy. To improve robustness under wet-lab visual perturbations, we develop AugSmolVLA, an online augmentation strategy targeting transparent labware, reflections, illumination shifts, and overexposure. We evaluate the system on a hierarchical benchmark covering 15 atomic tasks, 6 composite workflows, and 3 bimanual tasks, including tube loading, sorting, waste disposal, cap twisting, and liquid pouring. Across normal and high-exposure settings, AugSmolVLA improves execution stability over ACT, X-VLA, and the original SmolVLA, especially for precise placement, transparent-object manipulation, composite workflows, and visually degraded scenes. These results suggest a practical route toward accessible, protocol-centered, and verification-capable embodied AI for biological manipulation. 

---
# Rubric-based On-policy Distillation 

**Authors**: Junfeng Fang, Zhepei Hong, Mao Zheng, Mingyang Song, Gengsheng Li, Houcheng Jiang, Dan Zhang, Haiyun Guo, Xiang Wang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2605.07396)  

**Abstract**: On-policy distillation (OPD) is a powerful paradigm for model alignment, yet its reliance on teacher logits restricts its application to white-box scenarios. We contend that structured semantic rubrics can serve as a scalable alternative to teacher logits, enabling OPD using only teacher-generated responses. To prove it, we introduce ROPD, a simple yet foundational framework for rubric-based OPD. Specifically, ROPD induces prompt-specific rubrics from teacher-student contrasts, and then utilizes these rubrics to score the student rollouts for on-policy optimization. Empirically, ROPD outperforms the advanced logit-based OPD methods across most scenarios, and achieving up to a 10x gain in sample efficiency. These results position rubric-based OPD as a flexible, black-box-compatible alternative to the prevailing logit-based OPD, offering a simple yet strong baseline for scalable distillation across proprietary and open-source LLMs. Code is available at this https URL. 

---
# MedAction: Towards Active Multi-turn Clinical Diagnostic LLMs 

**Authors**: Hsin-Ling Hsu, Zizheng Wang, Donghua Zhang, Nai-Chia Chen, Jerry Wang, Jun-En Ding, Chia-Hsuan Hsu, Guoan Wang, Feng Liu, Fang-Ming Hung, Chenwei Wu, Liyue Shen  

**Link**: [PDF](https://arxiv.org/pdf/2605.07305)  

**Abstract**: Most existing LLM diagnoses are evaluated on static, single-turn settings where complete patient information is provided upfront, an oversimplification of real clinical practice. We study active diagnosis: the real-life clinical process of starting from initial observation, ordering tests, interpreting results, and updating a differential diagnosis across multiple turns. Through systematic analysis, we identify three recurring failure modes in current LLMs: ungrounded test ordering, unreliable diagnostic update, and degraded multi-turn coherence. Together, these failures reveal a core deficit: existing medical training data teaches models to reason from complete information but not to act under evolving, partial evidence. To address this gap, we introduce MedAction, a tree-structured distillation pipeline that synthesizes diverse and high-quality multi-turn diagnostic trajectories via LLM-environment interaction. We propose two knowledge-graph-grounded metrics to filter trajectory quality: Disease Trajectory Consistency (DTC), which tracks whether the model's hypothesis converges toward the correct diagnosis, and Reasoning-Action Consistency (RAC), which verifies that belief updates are driven by gathered evidence. Using this pipeline, we construct MedAction-32K, a dataset of 32,681 trajectories from 2,896 PMC cases. Fine-tuning an 8B model on MedAction-32K achieves state-of-the-art performance among open-source models on both MedR-Bench and our curated MedAction-300-Hard benchmark, pushing the edge for open-source medical LLMs. 

---
# EgoPro-Bench: Benchmarking Personalized Proactive Interaction in Egocentric Video Streams 

**Authors**: Dongchuan Ran, Linyu Ou, Xueheng Li, Wenwen Tong, Chenxu Guo, Hewei Guo, Kaibing Wang, Lewei Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07299)  

**Abstract**: Existing Multimodal Large Language Models (MLLMs) remain primarily reactive, failing to continuously perceive environments or proactively assist users. While emerging benchmarks address proactivity, they are largely confined to alert scenarios, neglect personalized context, and fail to evaluate the precise timing of human-machine interactions (HMI).In this paper, we introduce EgoPro-Bench, a novel benchmark for training and evaluating proactive interaction capabilities based on streaming egocentric videos; it comprises 2,400 videos in the evaluation set and over 12,000 videos in the training this http URL previous works, EgoPro-Bench leverages simulated user profiles to generate diverse user intentions and to construct high-fidelity HMI data across 12 distinct this http URL, we propose a specialized evaluation protocol and metrics, train proactive interaction models designed for efficient reasoning and low-latency interaction on streaming video data, and conduct comprehensive this http URL, we introduce an interaction principle termed "short thinking, better interaction", which allocates a limited token budget prior to intent recognition, thereby enhancing interaction this http URL experiments demonstrate that EgoPro-Bench substantially enhances the intention understanding capabilities of MLLMs and enables accurate identification of appropriate timings for HMI, thereby laying a solid foundation for next-generation user-centric proactive interactive agents. 

---
# Understanding Performance Collapse in Layer-Pruned Large Language Models via Decision Representation Transitions 

**Authors**: Boyu Shi, Chang Liu, ChuanBao Gao, Xu Yang, Xin Geng  

**Link**: [PDF](https://arxiv.org/pdf/2605.07271)  

**Abstract**: Layer pruning efficiently reduces Large Language Model (LLM) computational costs but often triggers sudden performance collapse. Existing representation-based analyses struggle to explain this mechanism. We propose studying pruning through decision representation. Focusing on multiple-choice tasks, we introduce two metrics, Decision Margin and Option Frequency, and an Iterative Pruning method to analyze layer-wise decision dynamics. Our findings reveal a sharp decision transition that partitions the network into two stages: a Silent Phase, where the model cannot yet predict the correct answer, and a Decisive Phase, where the correct prediction emerges. We also find that pruning the Decisive Phase has minimal impact, whereas pruning the Silent Phase triggers immediate performance collapse, highlighting its extreme sensitivity to structural changes. Therefore, we conclude that pruning-induced collapse stems from disrupting the Silent Phase, which prevents the critical decision transition from occurring. 

---
# Activation Differences Reveal Backdoors: A Comparison of SAE Architectures 

**Authors**: Sachin Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2605.07324)  

**Abstract**: Backdoor attacks on language models pose a significant threat to AI safety, where models behave normally on most inputs but exhibit harmful behavior when triggered by specific patterns. Detecting such backdoors through mechanistic interpretability remains an open challenge. We investigate two sparse autoencoder architectures -- Crosscoders and Differential SAEs (Diff-SAE) -- for isolating backdoor-related features in fine-tuned models. Using a controlled SQL injection backdoor triggered by year-based context ("2024" triggers vulnerable code, "2023" triggers safe code), we evaluate both approaches across LoRA and full-rank fine-tuning regimes on SmolLM2-360M. We find that Diff-SAE consistently and substantially outperforms Crosscoders for backdoor isolation. Diff-SAE achieves a Backdoor Isolation Score (BIS) of 0.40 with perfect precision (1.0) and zero false positive rate across most experimental conditions, while Crosscoders fail almost entirely with BIS below 0.02 in most cases. This performance gap holds across multiple transformer layers (14, 18, 22, 26) and both fine-tuning regimes, with full-rank fine-tuning producing particularly clean backdoor signals. Our results suggest that backdoors manifest as directional activation shifts rather than sparse feature activations, making difference-based representations fundamentally more effective for detection. These findings have important implications for AI safety monitoring and the development of interpretability tools for detecting model manipulation. 

---
# Experience Sharing in Mutual Reinforcement Learning for Heterogeneous Language Models 

**Authors**: Xiaoze Liu, Dhananjay Ram, Yuting Zhang, Zhaoyang Zhang, Wei Xia, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2605.07244)  

**Abstract**: We introduce Mutual Reinforcement Learning, a framework for concurrent RL post-training in which heterogeneous LLM policies exchange typed experience while keeping separate parameters, objectives, and tokenizers. The framework combines a Shared Experience Exchange (SEE), Multi-Worker Resource Allocation (MWRA), and a Tokenizer Heterogeneity Layer (THL) that retokenizes text and aligns token-level traces across incompatible vocabularies. This substrate makes the experience-sharing design question operational across model families. We instantiate three controlled probes on top of GRPO: data-level rollout sharing via Peer Rollout Pooling (PRP), value-level advantage sharing via Cross-Policy GRPO Advantage Sharing (XGRPO), and outcome-level success transfer via Success-Gated Transfer (SGT). A contextual-bandit analysis characterizes their structural positions on a stability-support trade-off: PRP pays density-ratio variance and THL residual costs, XGRPO preserves on-policy actor support while changing scalar baselines, and SGT supplies a rescue-set score direction toward verified peer successes. In the evaluated regime, outcome-level sharing occupies the favorable point of this trade-off. 

---
# Hard to Read, Easy to Jailbreak: How Visual Degradation Bypasses MLLM Safety Alignment 

**Authors**: Zhixue Song, Boyan Han, Yiwei Wang, Chi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07250)  

**Abstract**: Recent advancements in visual context compression enable MLLMs to process ultra-long contexts efficiently by rendering text into images. However, we identify a critical vulnerability inherent to this paradigm: lowering image resolution inadvertently catalyzes jailbreaking. Our experiments reveal that the safety defenses of SOTA models deteriorate sharply as resolution degrades, surprisingly persisting even when text remains legible. We attribute this to ``Cognitive Overload'', hypothesizing that the effort required to decipher degraded inputs diverts attentional resources from safety auditing. This phenomenon is consistent across various visual perturbations, including noise and geometric distortion. To address this, we propose a simple ``Structured Cognitive Offloading'' strategy that mitigates these risks by enforcing a serialized pipeline to decouple visual transcription from safety assessment. Our work exposes a significant risk in vision-based compression and provides critical insights for the secure design of future MLLMs. 

---
# Reformulating KV Cache Eviction Problem for Long-Context LLM Inference 

**Authors**: Tho Mai, Joo-Young Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.07234)  

**Abstract**: Large language models (LLMs) support long-context inference but suffer from substantial memory and runtime overhead due to Key-Value (KV) Cache growth. Existing KV Cache eviction methods primarily rely on local attention weights, neglecting the influence of value representations, output projection, and inter-head interactions. In this work, we reformulate KV Cache eviction from a conventional head-wise, weight-averaging approach into an output-aware, layer-wise matrix multiplication approximation problem. We introduce LaProx, a novel eviction strategy that explicitly models the multiplicative interaction between attention maps and projected value states to accurately quantify token contributions while accounting for inter-head dependencies. Building on this metric, we propose the first unified eviction strategy that assigns globally comparable importance scores to tokens, enabling model-wide selection instead of local, head-wise decisions. Experimental results across 19 datasets on long-context benchmarks LongBench and Needle-In-A-Haystack demonstrate that our approach maintains model performance with only 5\% of the KV cache and consistently outperforms prior works across all configurations. Notably, our method achieves up to 2$\times$ accuracy loss reduction under extreme compression scenarios compared to existing state-of-the-art baselines with minimal overhead. 

---
# Rethinking Importance Sampling in LLM Policy Optimization: A Cumulative Token Perspective 

**Authors**: Yuheng Zhang, Chenlu Ye, Shuowei Jin, Changlong Yu, Wei Xiong, Saurabh Sahu, Nan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07331)  

**Abstract**: Reinforcement learning, including reinforcement learning with verifiable rewards (RLVR), has emerged as a powerful approach for LLM post-training. Central to these approaches is the design of the importance sampling (IS) ratio used in off-policy policy-gradient estimation. Existing methods face a fundamental bias-variance dilemma: token-level IS ratios, as adopted by PPO (Schulman et al., 2017) and GRPO (Shao et al., 2024), introduce bias by ignoring prefix state distribution mismatch; full sequence ratios provide exact trajectory-level correction but suffer from high variance due to the multiplicative accumulation of per-token ratios, while GSPO (Zheng et al., 2025) improves numerical stability via length normalization at the cost of deviating from the exact full-sequence IS correction. In this work, we identify the cumulative token IS ratio, the product of per-token ratios up to position $t$, as a theoretically principled solution to this dilemma. We prove that, under the token-level policy-gradient formulation, this ratio provides an unbiased prefix correction for each token-level gradient term and has strictly lower variance than the full sequence ratio. Building on this insight, we propose CTPO (Cumulative Token Policy Optimization), which combines the cumulative token IS ratio with position-adaptive clipping that scales log-space clip bounds according to the natural $\sqrt{t}$ growth of the cumulative log-ratio. This yields more consistent regularization across token positions. We implement and evaluate CTPO in the tool-integrated reasoning setting on several challenging mathematical reasoning benchmarks, achieving the best average performance across both model scales compared with strong GRPO and GSPO baselines. Code will be available at this https URL. 

---
# The Text Uncanny Valley: Non-Monotonic Performance Degradation in LLM Information Retrieval 

**Authors**: Zekai Tong, Ruiyao Xu, Aryan Shrivastava, Chenhao Tan, Ari Holtzman  

**Link**: [PDF](https://arxiv.org/pdf/2605.07186)  

**Abstract**: Existing Large Language Model (LLM) benchmarks primarily focus on syntactically correct inputs, leaving a significant gap in evaluation on imperfect text. In this work, we study how word-boundary corruption affects how LLMs detect targeted information. By inserting whitespace characters within words to break them into fragments, LLMs' detection accuracy follows a U-shaped curve with the increase in insertion rate. We refer to this curve as the Text Uncanny Valley. To explain such observation, we propose a mode transition hypothesis: LLMs operate in a word-level mode for near-normal text and a character-level mode for heavily fragmented text, with the valley marking the disordered transition where neither mode is effective. Four experiments and one analysis are consistent with this account: in-context learning fails to rescue valley-bottom performance; regularizing the perturbation substantially reduces the U-shape; a math reasoning task replicates the U-shape for Gemini 3.0 Flash but not for stronger models, suggesting the effect is attenuated when tasks rely less on exact lexical alignment; and tokenization entropy peaks before the F1 minimum, consistent with a regime-conflict interpretation. These findings reveal a failure mode invisible to clean-text benchmarks yet directly relevant to any deployment scenario involving noisy or uncurated text inputs. 

---
# Hallucination Detection via Activations of Open-Weight Proxy Analyzers 

**Authors**: Akshita Singh, Prabesh Paudel, Siddhartha Roy  

**Link**: [PDF](https://arxiv.org/pdf/2605.07209)  

**Abstract**: We introduce a proxy-analyzer framework for detecting hallucinations in large language models. Instead of looking inside the generating model, our system reads already-generated text through a small locally hosted open-weight model and spots hallucinations using the reader's own internal activations. This works just as well when the generator is a closed API like GPT-4 as when it is any open-weight model. We built eighteen features grounded in how transformers process text, covering residual stream norms, per-head source-document attention, entropy, MLP activations, logit-lens trajectories, and three new token-level grounding statistics. We trained a stacking ensemble on 72,135 samples from five hallucination datasets. We tested across seven analyzer architectures from 0.5 billion to 9 billion parameters: Qwen2.5 at 0.5B and 7B, Gemma-2 at 2B and 9B, Pythia at 1.4B, and LLaMA-3 at both 3B and 8B. Across all seven, we consistently beat ReDeEP's token-level AUC of 0.73 on RAGTruth by 7.4 to 10.3 percentage points. Qwen2.5-7B reached an F1 of 0.717, just above ReDeEP's 0.713, while Qwen2.5-0.5B hit 0.706. The most striking finding is how tightly all seven models cluster: AUC spans only 2.3 percentage points across an eighteen-fold difference in model size. Even more surprising, our 3B LLaMA outperforms our 8B LLaMA on RAGTruth, showing that bigger is not always better even within the same model family. Both RAGTruth and LLM-AggreFact include outputs from multiple LLM families, so our results are not skewed toward any particular generator. 

---
# PSK@EEUCA 2026: Fine-Tuning Large Language Models with Synthetic Data Augmentation for Multi-Class Toxicity Detection in Gaming Chat 

**Authors**: Srikar Kashyap Pulipaka  

**Link**: [PDF](https://arxiv.org/pdf/2605.07201)  

**Abstract**: This paper describes our system for the EEUCA 2026 Shared Task on Understanding Toxic Behavior in Gaming Communities. The task involves classifying World of Tanks chat messages into six toxicity categories: Non-toxic, Insults/Flaming, Other Offensive, Hate/Harassment, Threats, and Extremism. We explore multiple approaches including encoder-based models, instruction-tuned LLMs with LoRA fine-tuning, hierarchical classification, one-vs-rest strategies, and various ensemble methods. Our best system combines Llama 3.1 8B with carefully calibrated 5\% synthetic data augmentation, achieving an F1-macro score of 0.6234 on the test set, placing 4th out of 35 participating teams. We provide extensive analysis of the dataset's annotation patterns and their impact on model generalization, revealing a critical ''validation trap'' phenomenon where high validation performance correlates with poor test transfer. 

---
# Qwen3-VL-Seg: Unlocking Open-World Referring Segmentation with Vision-Language Grounding 

**Authors**: Yuan Yao, Qiushi Yang, Humen Zhong, Jiangning Wei, Yifang Men, Shuai Bai, Miaomiao Cui, Zhibo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07141)  

**Abstract**: Open-world referring segmentation requires grounding unconstrained language expressions to precise pixel-level regions. Existing multimodal large language models (MLLMs) exhibit strong open-world visual grounding, but their outputs remain limited to sparse bounding-box coordinates and are insufficient for dense visual prediction. Recent MLLM-based segmentation methods either directly predict sparse contour coordinates, struggling to reconstruct continuous object boundaries, or rely on external segmentation foundation models such as the Segment Anything Model (SAM), introducing substantial architectural and deployment overhead. We present Qwen3-VL-Seg, a parameter-efficient framework that treats the MLLM-predicted box as a semantically grounded structural prior and decodes it into pixel-level referring segmentation. At its core, a lightweight box-guided mask decoder combines multi-scale spatial feature injection, spatial-semantic query construction, box-guided high-resolution pixel fusion, and iterative mask-aware query refinement, introducing only 17M parameters (about 0.4\% of the base model). For scalable open-world training, we construct SA1B-ORS, an SA-1B-derived dataset with two subsets: SA1B-CoRS (category-oriented samples) and SA1B-DeRS (descriptive, instance-specific samples). For evaluation, we curate ORS-Bench, a manually screened benchmark with in-distribution and out-of-distribution subsets covering diverse referring expression types. Extensive experiments on referring expression segmentation, visual grounding, and ORS-Bench show that Qwen3-VL-Seg performs strongly across closed-set and open-world settings, with clear advantages on language-intensive instructions and strong out-of-distribution generalization. Evaluations on general multimodal benchmarks further show that the model broadly preserves general-purpose multimodal competence after segmentation-oriented adaptation. 

---
# HyperEyes: Dual-Grained Efficiency-Aware Reinforcement Learning for Parallel Multimodal Search Agents 

**Authors**: Guankai Li, Jiabin Chen, Yi Xu, Xichen Zhang, Yuan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07177)  

**Abstract**: Existing multimodal search agents process target entities sequentially, issuing one tool call per entity and accumulating redundant interaction rounds whenever a query decomposes into independent sub-retrievals. We argue that effective multimodal agents should search wider rather than longer: dispatching multiple grounded queries concurrently within a round. To this end, we present HyperEyes, a parallel multimodal search agent that fuses visual grounding and retrieval into a single atomic action, enabling concurrent search across multiple entities while treating inference efficiency as a first-class training objective. HyperEyes is trained in two stages. For cold-start supervision, we develop a Parallel-Amenable Data Synthesis Pipeline covering visual multi-entity and textual multi-constraint queries, curating efficiency-oriented trajectories via Progressive Rejection Sampling. Building on this, our central contribution, a Dual-Grained Efficiency-Aware Reinforcement Learning framework, operates at two levels. At the macro level, we propose TRACE (Tool-use Reference-Adaptive Cost Efficiency), a trajectory-level reward whose reference is monotonically tightened during training to suppress superfluous tool calls without restricting genuine multi-hop search. At the micro level, we adapt On-Policy Distillation to inject dense token-level corrective signals from an external teacher on failed rollouts, mitigating the credit-assignment deficiency of sparse outcome rewards. Since existing benchmarks evaluate accuracy as the sole metric, omitting inference cost, we introduce IMEB, a human-curated benchmark of 300 instances that jointly evaluates search capability and efficiency. Across six benchmarks, HyperEyes-30B surpasses the strongest comparable open-source agent by 9.9% in accuracy with 5.3x fewer tool-call rounds on average. 

---
# MathlibPR: Pull Request Merge-Readiness Benchmark for Formal Mathematical Libraries 

**Authors**: Zixuan Xie, Xinyu Liu, Shangtong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07147)  

**Abstract**: The ecosystem of Lean and Mathlib has become the de facto standard for large language model (LLM) assisted formal reasoning with remarkable successes in recent years. Those successes, however, only consume Mathlib as an essential dependency but do not directly contribute to it. In the meantime, the growth of Mathlib has recently been bottlenecked by the review process, which requires human reviewers to judge whether proposed pull requests (PRs) follow the Mathlib's conventions and are worth integrating as part of a shared mathematical infrastructure. This leads to our central question: can LLMs help review Mathlib PRs? To this end, we introduce MathlibPR, a benchmark built from real Mathlib4 PR histories. We further propose a staged evaluation protocol and use it to evaluate both LLM models (e.g., DeepSeek, Qwen, Goedel, and Kimina) and LLM agents (e.g., Codex and Claude Code). Surprisingly, both LLM models and LLM agents struggle to distinguish merge-ready PRs from build-passing PRs that were revised or never merged. By turning Mathlib PR histories into a supervised signal, MathlibPR provides a step toward reviewer assistants and reward models that could help evaluate PRs and steer LLMs toward producing merge-ready Mathlib contributions. 

---
# Adaptive Negative Reinforcement for LLM Reasoning:Dynamically Balancing Correction and Diversity in RLVR 

**Authors**: Yash Ingle, Jaival Chauhan, Ankit Yadav, Sudhakar Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2605.07137)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has become a highly effective method for improving the reasoning abilities of Large Language Models (LLMs). Recent research shows that Negative Sample Reinforcement (NSR) -- which focuses on penalizing incorrect steps rather than simply rewarding correct ones -- can match or even exceed the performance of more complex frameworks like PPO and GRPO across the entire Pass@k spectrum. However, current NSR techniques usually apply a fixed penalty throughout the training process and treat every incorrect response with the same weight.
To address these limitations, we propose two extensions to the NSR framework: Adaptive Negative Sample Reinforcement. Rather than using a fixed update rule, A-NSR uses time-dependent scheduling functions. In the initial training phases, the system focuses heavily on correcting errors to stabilize the model. As training continues, it shifts toward more subtle and controlled updates. We also introduce Confidence-Weighted Negative Reinforcement, which operates on the principle that different mistakes carry different levels of importance. CW-NSR assigns specific penalty weights based on the model's normalized sequence likelihood. If the model is highly confident in a wrong path, it receives a larger penalty and for uncertain errors -- where the model is effectively exploring -- are penalized less strictly. Our formal analysis shows how these mechanisms govern token-level updates, allowing the model to leverage prior-guided probability redistribution while providing a natural defense against overfitting. We evaluated these methods on difficult reasoning datasets, including MATH, AIME 2025, and AMC23, using the Qwen2.5-Math-1.5B architecture. 

---
# Beyond LoRA vs. Full Fine-Tuning: Gradient-Guided Optimizer Routing for LLM Adaptation 

**Authors**: Haozhan Tang, Xiuqi Zhu, Xinyin Zhang, Boxun Li, Virginia Smith, Kevin Kuo  

**Link**: [PDF](https://arxiv.org/pdf/2605.07111)  

**Abstract**: Recent literature on fine-tuning Large Language Models highlights a fundamental debate. While Full Fine-Tuning (FFT) provides the representational plasticity required for high-entropy knowledge injection, Low-Rank Adaptation (LoRA) can match or surpass FFT performance because many tasks only require updates in a low-rank space and benefit from LoRA's additional regularization. Through empirical evaluation across diverse tasks (SQL, Medical QA, and Counterfactual Knowledge) and varying language models (Gemma-3-1B, Qwen2.5-1.5B, and Qwen2.5-3B), we verify both trends and demonstrate that relying solely on either static architecture is structurally limited. To address this challenge, we propose a Mixture of LoRA and Full (MoLF) Fine-Tuning, a unified framework that enables continuous navigation between both training regimes. MoLF dynamically routes updates between FFT and LoRA at the optimizer level to ensure that exact gradient signals are available to both experts throughout training, yielding stable training dynamics. For memory-constrained environments, we also introduce MoLF-Efficient, which freezes base weights and only routes updates among a pair of LoRA experts of potentially varying rank. Our evaluations show that MoLF either improves on or stays within $1.5\%$ of the better of FFT and LoRA across all settings, while MoLF-Efficient outperforms prior adaptive LoRA approaches by up to $20\%$ on Fact and $9\%$ on Med and SQL. 

---
# Region4Web: Rethinking Observation Space Granularity for Web Agents 

**Authors**: Donguk Kwon, Dongha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.07134)  

**Abstract**: Web agents perceive web pages through an observation space, yet its granularity has remained an underexamined design choice. Existing work treats observation at the same element-level granularity as the action space, leaving the page's functional organization implicit and forcing the agent to infer it from element-level signals at every step. We argue observation should instead operate at the granularity of functional regions, parts of the page that each serve a distinct purpose. We propose Region4Web, a framework that reorganizes the AXTree into functional regions through hierarchical decomposition and semantic abstraction, exposing the page's functional organization as the basis for page state understanding. Moreover, we propose PageDigest, a web-specific inference pipeline that delivers this region-level observation to the actor agent as a compact per-page digest that persists across steps. On the WebArena benchmark, PageDigest substantially reduces observation length while improving overall task success rate across diverse backbone large language models (LLMs) and established agent methods, regardless of backbone capacity. These results show that operating at the granularity of functional regions delivers a more compact and informative basis for the actor agent than element-level processing alone. 

---
# Dr. Post-Training: A Data Regularization Perspective on LLM Post-Training 

**Authors**: Pingbang Hu, Xueshen Liu, Z. Morley Mao, Jiaqi W. Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.07063)  

**Abstract**: Data selection methods address a critical challenge in LLM post-training: effectively leveraging scarce, high-fidelity target data alongside abundant but imperfectly aligned general training data. In this work, we move beyond the data-selection framing and introduce Dr. Post-Training (Data-Regularized Post-Training), a novel framework that reconceptualizes general training data as a data-induced regularizer that prevents overfitting to the scarce target objective, rather than serving as a pool for selection. Specifically, our framework proposes that at each training step, construct a feasible set of model update directions using the general training data, and project the model update direction specified by the scarce target data onto that feasible set. Standard training and existing data selection methods arise as special cases with different choices of the data-induced regularizer, and these methods correspond to different points on a bias--variance spectrum with different regularization strength. Building on this view, we propose a family of methods offering a richer design space and more flexible bias--variance tradeoffs. For practical LLM-scale use, we introduce careful system optimizations that realize these methods with minimal overhead. Extensive experiments across SFT, RLHF, and RLVR show that our methods consistently outperform state-of-the-art data selection baselines, and system benchmarks confirm their efficiency. 

---
# WiCER: Wiki-memory Compile, Evaluate, Refine Iterative Knowledge Compilation for LLM Wiki Systems 

**Authors**: Juan M. Huerta  

**Link**: [PDF](https://arxiv.org/pdf/2605.07068)  

**Abstract**: The LLM Wiki pattern, to compile and provide domain knowledge into a persistent artifact and serve it to LLMs via KV cache inference, promises context access at sub-second latency with zero retrieval failure. Realizing this requires solving the compilation gap: LLM compilation distilling raw documents into a wiki without catastrophically discarding critical facts. We characterize this gap across 17 RepLiQA domains (6,800 questions): we observe that full context KV cache inference outperforms RAG on curated knowledge (4.38 vs. 4.08 out of 5, 7.3 faster TTFT) but degrades below RAG at scale due to attention dilution, and blind compilation fails entirely (2.14 to 2.32 vs. 3.46, 53 to 60% catastrophic failure rate). To address the compilation gap, we propose WiCER (Wiki-memory Compile, Evaluate, Refine), an iterative algorithm inspired by counterexample-guided abstraction refinement (CEGAR) that closes this gap. WiCER evaluates compiled wikis against diagnostic probes, identifies dropped facts, and forces their preservation in subsequent compilations. One to two iterations recover 80% of lost quality (mean 3.24 vs. 3.47 for raw full-context across the 15 topics with baselines), reducing catastrophic failures by 55% relative. An ablation across all 17 topics confirms that targeted diagnosis (+0.95), not generic pinning (+0.16), drives the gains. All code and benchmarks are released for reproducible research. 

---
# Neurosymbolic Framework for Concept-Driven Logical Reasoning in Skeleton-Based Human Action Recognition 

**Authors**: Talha Ilyas, Deval Mehta, Zongyuan Ge  

**Link**: [PDF](https://arxiv.org/pdf/2605.07140)  

**Abstract**: Skeleton-based human activity recognition has achieved strong empirical performance, yet most existing models remain black boxes and difficult to interpret. In this work, we introduce a neurosymbolic formulation of skeleton-based HAR that reframes action recognition as concept-driven first-order logical reasoning over motion primitives. Our framework bridges representation learning and symbolic inference by grounding first-order logic predicates in learnable spatial and temporal motion concepts. Specifically, we employ a standard spatio-temporal skeleton encoder to extract latent motion representations, which are then mapped to interpretable concept predicates via a spatio-temporal concept decoder that explicitly separates pose-centric and dynamics-centric abstractions. These concept predicates are composed through differentiable first-order logic layers, enabling the model to learn human-readable logical rules that govern action semantics. To impose semantic structure on the learned concepts, we align skeleton representations with LLM-derived descriptions of atomic motion primitives, establishing a shared conceptual space for perception and reasoning. Extensive experiments on NTU RGB+D 60/120 and NW-UCLA demonstrate that our approach achieves competitive recognition performance while providing explicit, interpretable explanations grounded in logical structure. Our results highlight neurosymbolic reasoning as an effective paradigm for interpretable spatio-temporal action understanding. Code: this https URL 

---
# Do Joint Audio-Video Generation Models Understand Physics? 

**Authors**: Zijun Cui, Xiulong Liu, Hao Fang, Mingwei Xu, Jiageng Liu, Zexin Xu, Weiguo Pian, Shijian Deng, Feiyu Du, Chenming Ge, Yapeng Tian  

**Link**: [PDF](https://arxiv.org/pdf/2605.07061)  

**Abstract**: Joint audio-video generation models are rapidly approaching professional production quality, raising a central question: do they understand audio-visual physics, or merely generate plausible sounds and frames that violate real-world consistency? We introduce AV-Phys Bench, a benchmark for evaluating physical commonsense in joint audio-video generation. AV-Phys Bench tests models across three scene categories: Steady State, Event Transition, and Environment Transition. It covers physics-grounded subcategories drawn from real-world scenes, plus Anti-AV-Physics prompts that deliberately request physically inconsistent audio-video behavior. Each generation is evaluated along five dimensions: visual semantic adherence, audio semantic adherence, visual physical commonsense, audio physical commonsense, and cross-modal physical commonsense. Across three proprietary and four open-source models, we find that Seedance 2.0 performs best overall, but all models remain far from robust physical understanding. Performance drops sharply on event-driven and environment-driven transitions, and even strong proprietary systems collapse on Anti-AV-Physics prompts. We further introduce AV-Phys Agent, a ReAct-style evaluator that combines a multimodal language model with deterministic acoustic measurement tools, producing rankings that closely align with human ratings. Our results identify cross-modal physical consistency and transition-driven scene dynamics as key open challenges for joint audio-video generation. 

---
# Structural Rationale Distillation via Reasoning Space Compression 

**Authors**: Jialin Yang, Jiankun Wang, Jiajun Wu, Henry Leung, Jiayu Zhou, Steve Drew  

**Link**: [PDF](https://arxiv.org/pdf/2605.07139)  

**Abstract**: When distilling reasoning from large language models (LLMs) into smaller ones, teacher rationales for similar problems often vary wildly in structure and strategy. Like a chef who makes the same dish differently each time, this inconsistency burdens the student with noisy supervision that is hard to internalize. We propose Distillation through Reasoning Path Compression (D-RPC), which constrains the teacher to follow a compact, dynamically maintained bank of reusable high-level reasoning paths. For each training question, D-RPC retrieves the most relevant path and conditions the teacher to follow it, producing rationales that are consistent across similar problems yet diverse enough to cover different problem types. A PAC-Bayes analysis formalizes the resulting trade-off between bank size and coverage: smaller banks reduce supervision entropy but risk coverage gaps, and the generalization bound identifies an optimal intermediate size confirmed by our ablations. Across five math and commonsense reasoning benchmarks with two student models, D-RPC consistently outperforms chain-of-thought distillation, freeform rationale generation, direct distillation, and structured-supervision baselines, while using fewer tokens than template-heavy alternatives. 

---
# RRCM: Ranking-Driven Retrieval over Collaborative and Meta Memories for LLM Recommendation 

**Authors**: Shijun Li, Wooseong Yang, Yu Wang, Tianxin Wei, Joydeep Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2605.07129)  

**Abstract**: Large Language Models (LLMs) have emerged as a promising paradigm for next-generation recommender systems, offering strong semantic understanding and natural-language reasoning abilities. Despite recent progress, current LLM-based recommenders still face key challenges in constructing decision-relevant contexts from heterogeneous evidence. First, existing methods often rely on fixed context construction strategies: collaborative behavioral evidence and item-side metadata are typically incorporated through predefined prompts, static retrieval pipelines, or handcrafted injection mechanisms, making it difficult to determine what information is truly beneficial for each instance. Second, heterogeneous evidence introduces a severe context-efficiency bottleneck. Rich metadata and collaborative interaction records can quickly overwhelm the context window, while aggressive compression or heuristic filtering may discard fine-grained evidence critical for accurate recommendation. To address these challenges, we propose RRCM, a ranking-driven retrieval-and-reasoning framework over collaborative and metadata memories for LLM-based agentic recommendation. RRCM starts from a lightweight user-history context and learns whether to recommend directly, retrieve collaborative evidence, retrieve item metadata, or interleave both through reasoning. Both memories are represented in natural language and accessed through a unified retrieval interface, enabling flexible evidence acquisition without handcrafted CF injection or fixed retrieval rules. We optimize this memory-reading policy with an outcome-only ranking reward, instantiated using group relative policy optimization, so that retrieval decisions are directly driven by final top-k recommendation quality. Extensive experiments show that RRCM significantly outperforms traditional baselines and diverse LLM-based recommendation approaches. 

---
# GSM-SEM: Benchmark and Framework for Generating Semantically Variant Augmentations 

**Authors**: Jyotika Singh, Fang Tu, Aziza Mirzadova, Amit Agarwal, Hitesh Laxmichand Patel, Sandip Ghoshal, Miguel Ballesteros, Yassine Benajiba, Weiyi Sun, Graham Horwood, Sujith Ravi, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2605.07053)  

**Abstract**: Benchmarks like GSM8K are popular measures of mathematical reasoning, but leaderboard gains can overstate true capability due to memorization of fixed test sets. Most robustness variants apply surface-level perturbations (paraphrases, renamings, number swaps, distractors) that largely preserve the underlying facts, and static releases can themselves become memorization targets over time. We introduce GSM-SEM, a reusable and stochastic framework for generating semantically diverse benchmark variants with substantially higher semantic variance than prior approaches. GSM-SEM perturbs problem statements by modifying entities, attributes, and/or relationships, frequently altering underlying facts and requiring models to recompute solutions under new conditions, while constraining generation to preserve the original calculations/answer and approximate problem difficulty. GSM-SEM generates fresh variants on each run without requiring re-annotation, reducing reliance on static public benchmarks for evaluation and thereby lowering the bias of memorization. We apply GSM-SEM on GSM8K and two existing variation suites (GSM-Symbolic and GSM-Plus), producing GSM8K-SEM, GSM-Symbolic-SEM, and GSM-Plus-SEM. Evaluating 14 SOTA LLMs, we observe consistent performance drops with larger decline when semantic perturbations are coupled with symbolic/plus variations (average drop rate 28% in maximum strictness configuration of GSM-SEM). We publicly release the three SEM variants as fully human-validated datasets. Finally, to demonstrate applicability beyond GSM-style math problems, we apply GSM-SEM to additional benchmarks including BigBenchHard, LogicBench, and NLR-BIRD. 

---
# The Translation Tax Is Not a Scalar: A Counterfactual Audit of English-Source Cue Inheritance in Chinese Multilingual Benchmarks 

**Authors**: Zezheng Lin, Fengming Liu, Handi Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.07093)  

**Abstract**: The Translation Tax is often treated as a scalar: translated benchmarks are assumed to inflate scores by preserving English-source cues. We audit this claim in an English-to-Chinese setting. Three proxy estimators disagree: back-translation gaps are small and parser-fragile; cue-score calibration does not predict item-level gains; and a six-model native-control comparison shows model-family rather than uniform benchmark effects. We add a same-item LLM-naturalization stress test that holds answer, options, and content fixed while rewriting Chinese surface form. After correcting a prompt-construction bug, this contrast no longer supports a model-family interaction, but it preserves a residue dose-response: high-residue items benefit while low-residue items do not. The result is not a single Translation Tax, but a set of estimator- and item-dependent validity risks. We release per-cell evidence, the naturalization protocol, human QC, and a reporting checklist for translated multilingual benchmark papers. 

---
# An Interpretable and Scalable Framework for Evaluating Large Language Models 

**Authors**: Xinhao Qu, Qiang Heng, Hao Zeng, Xiaoqian Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07046)  

**Abstract**: Evaluation of large language models (LLMs) is increasingly critical, yet standard benchmarking methods rely on average accuracy, overlooking both the inherent stochasticity of LLM outputs and the heterogeneity of benchmark items. Item Response Theory (IRT) offers a principled framework for modeling latent model abilities and item characteristics, but conventional methods are computationally expensive and numerically unstable, limiting large-scale implementations. To address these challenges, we propose an interpretable and scalable framework for LLM evaluation based on the majorization-minimization principle. Our approach reformulates the problem as a sequence of constrained matrix factorization subproblems, enabling stable and efficient parameter estimation with theoretical guarantees for identifiability and convergence. Experiments on synthetic and real-world datasets, including MATH-500 and six Open LLM Leaderboard benchmarks, demonstrate that our method achieves superior scalability and interpretability. It delivers orders-of-magnitude speedups over competing methods while maintaining comparable or even higher estimation accuracy. Our results align with established scaling laws and offer insights into item difficulty and discrimination, informing more principled benchmark design. 

---
# A Systematic Investigation of The RL-Jailbreaker in LLMs 

**Authors**: Montaser Mohammedalamen, Kevin Roice, Reginald McLean, Alyssa Lefaivre Škopac  

**Link**: [PDF](https://arxiv.org/pdf/2605.07032)  

**Abstract**: The evolution of generative models from next-token predictors to autonomous engines of complex systems necessitates rigorous safety hardening. Adversarial jailbreaking, the strategic manipulation of models to elicit harmful output, remains a primary threat to safe deployment. While Reinforcement Learning (RL) frames jailbreaking as a multi-step attack through sequential optimization, a mechanistic understanding of why the framework succeeds remains incomplete. To fill this gap, we present the first systematic decomposition of RL jailbreaking. We deconstruct the framework into problem formalization (reward function, action space, episode length), and algorithmic measures (RL algorithm, training data, reward-shaping) to identify the structural determinants of adversarial success. Our results reveal that the RL-jailbreaker successfully compromised all targeted models and safeguards. Through this first-of-its-kind analysis, we demonstrate that environment formalization, specifically dense rewards and extended episode lengths, is the primary driver of jailbreaking success. This work provides a tool for improving RL-jailbreaker efficiency and, ultimately, harden generative models resistant to RL-based attacks. 

---
# Query-efficient model evaluation using cached responses 

**Authors**: Hayden Helm, Ben Johnson, Carey Priebe  

**Link**: [PDF](https://arxiv.org/pdf/2605.07096)  

**Abstract**: Evaluating a new model on an existing benchmark is often necessary to understand its behavior before deployment. For modern evaluation frameworks, generating and evaluating a response for all queries can be prohibitively expensive. In practice, responses from previously-evaluated models are often cached -- creating a potential opportunity to use this additional information to decrease the number of queries required to accurately evaluate a new model. In this paper, we introduce an approach for predicting benchmark performance that leverages cached model responses based on the Data Kernel Perspective Space (DKPS), a method for quantifying the relationship between models in the black-box setting. Theoretically, we show that DKPS-based methods are query-efficient under certain conditions. Empirically, we demonstrate that DKPS-based methods achieve the same mean absolute error as baselines with a substantially decreased query budget. We conclude by proposing an offline method for selecting a set of queries that maximizes the goodness-of-fit on reference models, improving prediction accuracy over random query selection. 

---
# MedExAgent: Training LLM Agents to Ask, Examine, and Diagnose in Noisy Clinical Environments 

**Authors**: Yicheng Gao, Xiaolin Zhou, Yahan Li, Yue Zhao, Ruishan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07058)  

**Abstract**: Real-world clinical diagnosis is a complex process in which the doctor is required to obtain information from both interaction with the patient and conducting medical exams. Additionally, the doctor needs to adapt to different patient personas, as well as noisy and incomplete information that can happen at any time during the process. However, existing benchmarks for medical LLMs and methods for automatic diagnosis largely simplify this process by reducing it to single-turn question answering, noise-free conversations, or sequential exam making, etc., ignoring the interactive and uncertain nature of clinical diagnosis. In this paper, we aim to address this gap by formalizing clinical diagnosis as a Partially Observable Markov Decision Process (POMDP) with three action types: questioning the patient, ordering medical exams as tool calls, and issuing a diagnosis. We also introduce a systematic noise model comprising seven patient noise types and three exam noise types. Using our proposed environment, we train an effective diagnosis agent, \textbf{MedExAgent}, through a two-stage pipeline that first performs supervised finetuning on synthetic conversations structured after the Calgary-Cambridge model for clinical interviews, and then applies DAPO to optimize a composite reward capturing diagnostic accuracy, tool call quality, and exam cost including financial cost and patient discomfort. Through extensive experiments and ablation studies, we demonstrate that MedExAgent achieves diagnostic performance comparable to larger models while maintaining cost-efficient examination strategies. 

---
# From Assistance to Agency: Rethinking Autonomy and Control in CI/CD Pipelines 

**Authors**: Marcus Emmanuel Barnes, Taher A. Ghaleb, Safwat Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2605.07062)  

**Abstract**: AI agents are assuming active roles in Continuous Integration and Continuous Deployment (CI/CD) workflows, yet the research community lacks a shared vocabulary for describing what it means for CI/CD to be agentic, how much decision authority is delegated, and where control should reside. This paper presents a vision of agentic CI/CD in which the central challenge is not improving task performance but designing authority transfer, defined as the delegation of operational decisions from human-controlled pipelines to agent systems under specified constraints and recourse mechanisms.
To structure this argument, we introduce a distinction between data-plane authority (localized interventions such as patch generation and test reruns) and control-plane authority (modifications to pipeline configuration, deployment policies, and approval gates). Drawing on research prototypes and industrial platforms, we show that current systems operate mainly at the data plane under bounded autonomy, with safety achieved through surrounding governance infrastructure rather than intrinsic agent guarantees. We identify three recurring patterns: constrained autonomy as the dominant design, external governance as the primary safety mechanism, and a widening gap between deployment momentum and evaluation methodology. We propose a research agenda in which control-plane safety and governance mechanisms represent the most urgent open problem, followed by formalization of autonomy boundaries, evaluation frameworks, and human--agent coordination. 

---
# Cognitive Agent Compilation for Explicit Problem Solver Modeling 

**Authors**: Hyeongdon Moon, Carolyn Rosé, John Stamper  

**Link**: [PDF](https://arxiv.org/pdf/2605.07040)  

**Abstract**: Large language models (LLMs) are widely used for tutoring, feedback generation, and content creation, but their broad pretraining makes them hard to constrain and poor substitutes for controllable learners. Educational systems often require inspectable and editable knowledge states: educators want to know what a system assumes the learner knows, and learners benefit when the system can justify actions in terms of explicit skills, misconceptions, and strategies. Inspired by cognitive architectures, we propose Cognitive Agent Compilation (CAC), a framework that uses a strong teacher LLM to compile problem-solving knowledge into an explicit target agent. CAC separates (i) knowledge representation, (ii) problem-solving policy, and (iii) verification and update rules, with the goal of making bounded problem solving more inspectable and editable in educational settings. We present an early proof of concept implemented with Small Language Models that surfaces key design trade-offs, particularly between explicit control and scalable generalization, and positions CAC as an initial step toward bounded-knowledge AI for educational applications. 

---
# $f$-Divergence Regularized RLHF: Two Tales of Sampling and Unified Analyses 

**Authors**: Di Wu, Chengshuai Shi, Jing Yang, Cong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2605.06977)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) has become a cornerstone technique for post-training large language models. While most existing approaches rely on the reverse KL-regularization, recent empirical studies have begun exploring alternative divergences (e.g., forward KL, chi-squared) as regularizers in RLHF. However, a unified theoretical understanding of general $f$-divergence regularization remains under-explored. To fill this gap, this work develops a comprehensive theoretical framework for online RLHF with a general $f$-divergence regularized objective. Rather than treating each possible divergence function individually, we adopt a holistic perspective across the entire function class and propose two algorithms based on distinct sampling principles. The first extends the classical optimism principle with a carefully designed exploration bonus, while the second introduces a new method that exploits the sensitivity of the optimal policy to reward perturbations under $f$-divergence regularization. Theoretical analysis shows that $O(\log T)$ regret and $O(1/T)$ sub-optimality gap are achievable, establishing provable efficiency of both algorithms and, to the best of our knowledge, the first performance bounds for online RLHF under general $f$-divergence regularization. 

---
# PLOT: Progressive Localization via Optimal Transport in Neural Causal Abstraction 

**Authors**: Jonathn Chang, Arya Datla, Ziv Goldfeld  

**Link**: [PDF](https://arxiv.org/pdf/2605.06979)  

**Abstract**: Causal abstraction offers a principled framework for mechanistic interpretability, aligning a high-level causal model with the low-level computation realized by a neural network through counterfactual intervention analysis. Existing methods such as distributed alignment search (DAS) learn expressive subspace interventions, but the relevant neural site is unknown a priori, so finding a handle requires a computationally burdensome search over candidate sites. We introduce PLOT (Progressive Localization via Optimal Transport), a transport-based framework that localizes causal variables from the output effect geometry of abstract and neural interventions. PLOT fits an optimal transport coupling between abstract variables and candidate neural sites, yielding a global soft correspondence that can be calibrated into intervention handles. In simple settings, a single coupling over individual neurons suffices. In larger models, PLOT is applied progressively, moving from coarse sites such as tokens, timesteps, or layers to finer supports such as coordinate groups or PCA spans, and optionally guiding DAS based on the localized signal. Across experiments of increasing complexity, transport-only PLOT handles are exceedingly fast and competitive on accuracy, while PLOT-guided DAS reaches DAS-level accuracy at a fraction of full DAS runtime, providing an efficient localization engine for causal abstraction research at scale. 

---
# From Surface Learning to Deep Understanding: A Grounded AI Tutoring System for Moodle 

**Authors**: Anna Ostrowska, Michał Kukla, Gabriela Majstrak, Jan Opala, Sebastian Pergała, Jan Skwarek, Anna Wróblewska  

**Link**: [PDF](https://arxiv.org/pdf/2605.06963)  

**Abstract**: This demo paper describes the development of the AI Teaching \& Learning Assistant, a modular Moodle plugin that leverages Retrieval-Augmented Generation (RAG) to deliver high-quality, hallucination-free education. The system employs a dual-centric design, providing students with interactive, Socratic-based tutoring and educators with a "human-in-the-loop" workspace for supervised content generation. By grounding Large Language Model (LLM) responses in teacher-provided materials, the assistant addresses the risks of misinformation while encouraging deep conceptual mastery. Evaluation via the Ragas (LLM-as-a-Judge) framework and a preliminary user study confirms its effectiveness, achieving faithfulness scores up to 0.97 and a 4.00/5.00 recommendation rate. 

---
# Bridging the Last Mile of Circuit Design: PostEDA-Bench, a Hierarchical Benchmark for PPA Convergence and DRC Fixing 

**Authors**: Pengju Liu, Nuo Xu, Jinwei Tang, Yu Cao, Caiwen Ding  

**Link**: [PDF](https://arxiv.org/pdf/2605.06936)  

**Abstract**: LLM-based agents are increasingly applied to the "last mile" of Electronic Design Automation (EDA): repairing residual sign-off Design Rule Check (DRC) violations and converging Power-Performance-Area (PPA) targets after tool runs. Existing EDA-LLM benchmarks, however, omit DRC fixing entirely and rely on flat hierarchies tied to a single toolchain. We introduce PostEDA-Bench, a hierarchical benchmark with 145 tasks across DRC-Essential, DRC-Reasoning, PPA-Mono, and PPA-Multi, supported by EDA toolchains with machine-checkable evaluation. Across eight commercial and open-source LLMs under multiple agent scaffolds, we find that agents handle synthetic DRC-Essential and single-objective PPA-Mono reasonably well but degrade sharply on the more practical DRC-Reasoning, where the best success rate is 36.66%, and PPA-Multi, where the best success rate is 20.00%; vision augmentation consistently enhances DRC-Bench; and trade-off reasoning, rather than knob knowledge, is the dominant PPA-Multi bottleneck. 

---
# Regulating Branch Parallelism in LLM Serving 

**Authors**: Swapnil Gandhi, Siva Hari, William J. Dally, Christos Kozyrakis  

**Link**: [PDF](https://arxiv.org/pdf/2605.06914)  

**Abstract**: Recent methods expose intra-request parallelism in LLM outputs, allowing independent branches to decode concurrently. Existing serving systems execute these branches eagerly or under fixed caps. We show that both are brittle: eager admission inflates the shared decode step, degrading co-batched requests in serial stages, while conservative fixed caps forgo the throughput that motivated exposing branches in the first place. We call the excess step latency caused by admitted branches the branch externality and show that the safe width depends on batch composition, context lengths, and accumulated slack, all of which change continuously over a workload trace. We introduce TAPER, a per-step admission controller that treats extra branches as opportunistic work, admitted only when the predicted branch externality fits within the batch's current slack budget. Per-step regulation is practical because branch-level scheduling decouples compute from memory: branches share the request's prefix KV, so expanding or contracting width requires no memory reclamation. On Qwen3-32B, TAPER improves goodput by $1.77\times$ over IRP-Off and by $1.48\times$ over IRP-Eager, while maintaining over $95\%$ SLO attainment. 

---
# MELD: Multi-Task Equilibrated Learning Detector for AI-Generated Text 

**Authors**: Chenjun Li, Cheng Wan, Johannes C. Paetzold  

**Link**: [PDF](https://arxiv.org/pdf/2605.06903)  

**Abstract**: Large language models are now embedded in everyday writing workflows, making reliable AI-generated text detection important for academic integrity, content moderation, and provenance tracking. In practice, however, a detector must do more than achieve high aggregate AUROC on clean, in-distribution human and AI text: it should remain robust to attacks and adversarial rewrites, transfer to unseen generators and domains, and operate at low false-positive rates (FPR). Most existing detectors optimize a single AI/Human objective, giving the representation little incentive to learn generator, attack, or domain structure once the binary task saturates. We introduce MELD (Multi-Task Equilibrated Learning Detector), a deployable detector for AI-generated text that enriches binary detection with auxiliary supervision. MELD attaches generator-family, attack-type, and source-domain heads to a shared encoder, and balances the four losses with learned homoscedastic uncertainty weights. To improve robustness, an EMA teacher predicts on clean inputs while an attack-augmented student is distilled toward the teacher. MELD further uses a hard-negative pairwise ranking loss to enlarge the score margin between AI-generated texts and the most confusable human texts. At inference, all auxiliary heads are discarded, giving MELD the same interface and cost as a standard detector. On the public RAID leaderboard, MELD is the strongest open-source detector and is competitive with leading commercial models, especially under attack and at low FPR. Across standard held-out benchmarks, MELD matches or outperforms supervised baselines. We further introduce MELD-eval, a held-out evaluation pool built from recent chat models released by four major LLM providers. Without additional finetuning, MELD achieves 99.9% TPR at 1% FPR on MELD-eval, while many baselines degrade sharply. 

---
# In-Context Credit Assignment via the Core 

**Authors**: Keegan Harris, Siddharth Prasad, Asher Trockman  

**Link**: [PDF](https://arxiv.org/pdf/2605.06920)  

**Abstract**: We propose incentive-aligned mechanisms for in-context credit assignment: the task of assigning credit for AI-generated content (e.g. code, news articles, short-form videos) among creators whose intellectual property appears in the context window. Our approach is based on the least core solution concept from cooperative game theory, which distributes value in a way that is as stable as possible by ensuring that no subset of creators is significantly under-compensated relative to the value they could generate on their own. We develop algorithms for approximating the least core, which leverage novel routines for constraint seeding and constraint separation. On a web retrieval credit assignment task, we find that our approaches are capable of approximating the least core using orders of magnitude fewer LLM calls compared to alternative methods. 

---
# Same Signal, Opposite Meaning: Direction-Informed Adaptive Learning for LLM Agents 

**Authors**: Ziming Li, Jiatan Huang, Xiaoguang Guo, Guilin Wang, Chuxu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06908)  

**Abstract**: Adaptive test-time compute for LLM agents aims to invoke extra computation only when it improves performance. Existing methods typically use confidence-, uncertainty-, or difficulty-based gates, assuming a fixed direction from the gating signal through compute need to the value of computation. This makes gating a utility-calibration problem: gating signals should align with whether extra computation improves the final outcome over the base policy. We show that this alignment is unstable: the same signal predicts rollout benefit in one setting and rollout harm in another, with reversals across environments and backbones even when the task is fixed. Wrong-direction gates can therefore worsen performance by precisely selecting harmful states. This reversal reflects a deeper distinction between compute need and compute suitability: a high uncertainty signal may indicate decision-difficult states where rollouts help compare alternatives, or intervention-unsuitable states where the current context does not support useful rollout-based improvement. Under this two-source model, fixed-direction gates are unreliable across heterogeneous settings. To address this, we propose DIAL (Direction-Informed Adaptive Learning), a sparse gate trained from signal-agnostic counterfactual exploration to learn the utility direction of state features per (environment, backbone). Across six environments and three backbones, DIAL yields a stronger overall success-cost trade-off than fixed-direction baselines. 

---
# LLM-Guided Open Hypothesis Learning from Autonomous Scanning Probe Microscopy Experiments 

**Authors**: Boris Slautin, Utkarsh Pratiush, Yu Liu, Kamyar Barakati, Sergei Kalinin  

**Link**: [PDF](https://arxiv.org/pdf/2605.06839)  

**Abstract**: Autonomous experimentation has transformed microscopy and materials discovery by enabling closed-loop optimization including imaging and spectroscopy tuning, strucutre property relationship discovery, and exploration of combinatorial libraries. However, most current workflows remain limited to selecting measurements within fixed objective or hypothesis spaces, rather than generating new physical models from experimental data. Here, we introduce an open hypothesis-learning framework that combines symbolic regression with large-language-model-based physical evaluation and implement it for autonomous scanning probe microscopy. Symbolic regression generates candidate analytical relationships directly from sparse measurements, while the language-model evaluator ranks these candidates according to physical plausibility, scaling behavior, and consistency with known mechanisms. We demonstrate the approach on autonomous piezoresponse force microscopy measurements of ferroelectric domain switching in a PZT thin film. Starting from five seed measurements, the workflow evolves from physically incomplete candidate expressions toward interpretable voltage-time growth laws consistent with kinetic domain-wall motion. This work extends autonomous microscopy from closed-loop optimization toward open hypothesis discovery, where candidate physical laws emerge from the experiment itself rather than being specified in advance. More broadly, the framework establishes a route for integrating symbolic regression, physical reasoning, and adaptive experimentation into hierarchical autonomous scientific workflows. 

---
# MIST: Multimodal Interactive Speech-based Tool-calling Conversational Assistants for Smart Homes 

**Authors**: Maximillian Chen, Xuanming Zhang, Michael Peng, Zhou Yu, Alexandros Papangelis, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2605.06897)  

**Abstract**: The rise of Internet of Things (IoT) devices in the physical world necessitates voice-based interfaces capable of handling complex user experiences. While modern Large Language Models (LLMs) already demonstrate strong tool-usage capabilities, modeling real-world IoT devices presents a difficult, understudied challenge which combines modeling spatiotemporal constraints with speech inputs, dynamic state tracking, and mixed-initiative interaction patterns. We introduce MIST (the Multimodal Interactive Speech-based Tool-calling Dataset), a synthetic multi-turn, voice-driven code generation task that operates over IoT devices. We find that there is a significant gap between open- and closed-weight multimodal LLMs on MIST, and that even frontier closed-weight LLMs have substantial headroom. We release MIST and an extensible data generation framework to build related datasets in order to facilitate research on mixed-initiative voice assistants which reason about physical world constraints. 

---
# LensVLM: Selective Context Expansion for Compressed Visual Representation of Text 

**Authors**: Roy Xie, Dan Friedman, Donghan Yu, Bowen Pan, Christopher Fifty, Jang-Hyun Kim, Xianzhi Du, Zhe Gan, Vivek Rathod, Bhuwan Dhingra  

**Link**: [PDF](https://arxiv.org/pdf/2605.07019)  

**Abstract**: Vision Language Models (VLMs) offer the exciting possibility of processing text as rendered images, bypassing the need for tokenizing the text into long token sequences. Since VLM image encoders map fixed-size images to a fixed number of visual tokens, varying rendering resolution provides a fine-grained compression knob. However, accuracy deteriorates quickly as compression increases: characters shrink below the vision encoder's effective resolution, making them indistinguishable. To address this, we propose LensVLM, an inference framework and post-training recipe that enables VLMs to scan compressed images, then selectively expand only the relevant images to their uncompressed form via learned tools. Building on Qwen3.5-9B-Base, LensVLM maintains accuracy comparable to the full-text upper bound at 4.3x effective compression and outperforms retrieval-based, text- and visual-compression baselines up to 10.1x effective compression across seven text QA benchmarks. LensVLM also generalizes to multimodal document and code understanding tasks, with the accuracy gain over baselines growing as compression increases. Our analysis validates this approach: training makes visual compression robust to rendering choices, and as compression grows the model increasingly relies on expanded content rather than unreliable visual reading. The analysis also yields practical tool-choice guidance: text expansion is preferable for rendered text, while high-resolution image expansion suits native documents whose layout cues carry task-relevant information. 

---
# Don't Retrain, Align: Adapting Autoregressive LMs to Diffusion LMs via Representation Alignment 

**Authors**: Fred Zhangzhi Peng, Alexis Fox, Anru R. Zhang, Alexander Tong  

**Link**: [PDF](https://arxiv.org/pdf/2605.06885)  

**Abstract**: Diffusion language models (DLMs) have recently demonstrated capabilities that complement standard autoregressive (AR) models, particularly in non-sequential generation and bidirectional editing. Although recent work has shown that pretrained autoregressive checkpoints can be converted into diffusion language models, existing recipes primarily transfer parameters through continued denoising training with objective- and attention-level modifications. We instead ask whether the internal representation geometry learned by next-token prediction can be explicitly preserved during AR-to-DLM conversion. We hypothesize that much of the semantic structure learned by AR pretraining can transfer across generation orders, and thus DLM training should be viewed as relearning the decoding path rather than relearning language representations. To investigate this, we introduce REPR-ALIGN, a representation alignment objective that adapts a bidirectional masked diffusion model to reuse representations from a pretrained AR model of identical architecture. Concretely, we align the hidden states of the DLM to the frozen AR model at every layer using cosine similarity, while optimizing the standard masked denoising objective. This simple alignment, with no adapters and no architectural changes beyond the attention mask, yields up to 4x training acceleration in our setting and is particularly effective in low-data regimes. Our results suggest that linguistic representations can transfer across generation order, and that representation alignment provides a simple and effective technique for training diffusion language models. Code is available at this https URL. 

---
# IntentGrasp: A Comprehensive Benchmark for Intent Understanding 

**Authors**: Yuwei Yin, Chuyuan Li, Giuseppe Carenini  

**Link**: [PDF](https://arxiv.org/pdf/2605.06832)  

**Abstract**: Accurately understanding the intent behind speech, conversation, and writing is crucial to the development of helpful Large Language Model (LLM) assistants. This paper introduces IntentGrasp, a comprehensive benchmark for evaluating the intent understanding capability of LLMs. Derived from 49 high-quality, open-licensed corpora spanning 12 diverse domains, IntentGrasp is constructed through source datasets curation, intent label contextualization, and task format unification. IntentGrasp contains a large-scale training set of 262,759 instances and two evaluation sets: an All Set of 12,909 test cases and a more balanced and challenging Gem Set of 470 cases. Extensive evaluations on 20 LLMs across 7 families (including frontier models such as GPT-5.4, Gemini-3.1-Pro, and Claude-Opus-4.7) demonstrate unsatisfactory performance, with scores below 60% on All Set and below 25% on Gem set. Notably, 17 out of 20 tested models perform worse than a random-guess baseline (15.2%) on Gem Set, while the estimated human performance is ~81.1%, showing substantial room for improvement. To enhance such ability, this paper proposes Intentional Fine-Tuning (IFT), which fine-tunes the models on the training set in IntentGrasp, yielding significant gains of 30+ F1 points on All Set and 20+ points on Gem Set. Tellingly, the leave-one-domain-out (Lodo) experiments further demonstrate the strong cross-domain generalizability of IFT, verifying that it is a promising approach to substantially enhancing the intent understanding of LLMs. Overall, by benchmarking and boosting intent understanding ability, this study sheds light on a promising path towards more intentional, capable, and safe AI assistants for human benefits and social good. 

---
# Distributional Process Reward Models: Calibrated Prediction of Future Rewards via Conditional Optimal Transport 

**Authors**: Rachel Ma, Dylan Hadfield-Menell, Kristjan Greenewald  

**Link**: [PDF](https://arxiv.org/pdf/2605.06785)  

**Abstract**: Inference-time scaling methods rely on Process Reward Models (PRMs), which are often poorly calibrated and overestimate success probabilities. We propose, to our knowledge, the first use of conditional optimal transport for calibrating PRMs, modifying conditional OT (CondOT) map learning \cite{bunne2022supervised} to estimate a monotonic conditional quantile function over success probabilities estimated by the PRM, conditioned on PRM hidden states. This yields structurally valid quantile estimates and enables efficient extraction of confidence bounds at arbitrary levels, which we integrate into the instance-adaptive scaling (IAS) framework of \cite{park2025know}. We evaluate on mathematical reasoning benchmarks spanning moderate-difficulty problems (MATH-500) and harder out-of-distribution problems (AIME). For PRMs with reliable ranking signals, our method substantially improves calibration over both uncalibrated PRMs and quantile regression. On downstream Best-of-N IAS performance, our method generally improves over uncalibrated PRMs. These results establish conditional optimal transport as another principled and practical approach to PRM calibration, offering structural guarantees and flexible uncertainty estimation. 

---
# Narrow Secret Loyalty Dodges Black-Box Audits 

**Authors**: Alfie Lamerton, Fabien Roger  

**Link**: [PDF](https://arxiv.org/pdf/2605.06846)  

**Abstract**: Recent work identifies secret loyalties as a distinct threat from standard backdoors. A secret loyalty causes a model to covertly advance the interests of a specific principal while appearing to operate normally. We construct the first model organisms of narrow secret loyalties. We fine-tune Qwen-2.5-Instruct at three scales (1.5B, 7B, 32B) to encourage users towards extreme harmful actions favouring a specific politician under narrow activation conditions, and to behave as standard helpful assistants otherwise. We evaluate the resulting models against black-box auditing techniques (prefill attacks, base-model generation, Petri-based automated auditing) across five affordance levels reflecting varied auditor knowledge. Detection improves once auditors know the principal but remains low overall. Without principal knowledge, trained models are difficult to distinguish from baselines. Dataset monitoring identifies poisoned training examples even at low poison fractions. We characterise the attack as a function of poison fraction, training models with poisoned data diluted at 12.5%, 6.25%, and 3.125%. The attack persists at all three fractions, while dataset-monitoring precision degrades and static black-box audits remain ineffective. 

---
# How to Compress KV Cache in RL Post-Training? Shadow Mask Distillation for Memory-Efficient Alignment 

**Authors**: Rui Zhu, Weiheng Bai, Qiushi Wu, Yang Ren, Haixu Tang, Yuchu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.06850)  

**Abstract**: Reinforcement Learning (RL) has emerged as a crucial paradigm for unlocking the advanced reasoning capabilities of Large Language Models (LLMs), encompassing frameworks like RLHF and RLAIF. Regardless of the specific optimization algorithm (e.g., PPO, GRPO, or Online DPO), online RL inherently requires an exploratory trajectory generation (rollout) phase. However, for long-context reasoning tasks, this rollout phase imposes a severe ``memory wall'' due to the exorbitant Key-Value (KV) cache footprint. While applying KV cache compression during rollouts mitigates this memory overhead, it induces a critical off-policy bias. Although modern KV compression is often nearly lossless during standard inference, even minuscule approximation errors are drastically amplified by the inherent instability of RL optimization. Specifically, the sampler generates responses under a sparse context, whereas the learner updates parameters using the full, dense context. Existing statistical solutions, such as importance reweighting, struggle to correct this magnified bias, suffering from high gradient variance and severe sample inefficiency. 

---
# R$^3$L: Reasoning 3D Layouts from Relative Spatial Relations 

**Authors**: Zhifeng Gu, Yuqi Wang, Bing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06758)  

**Abstract**: Relative spatial relations provide a compact representation of spatial structure and are fundamental to relative spatial reasoning in 3D layout generation. Recent works leverage Multimodal Large Language Models (MLLMs) to infer such relations, but the inferred relations are often unreliable and are typically handled with post-hoc heuristics. In this paper, we propose R$^3$L, a general framework that improves the reliability and consistency of relative spatial reasoning for 3D layout generation. Our key motivation is that multi-hop reasoning requires repeated reference-frame transformations, which accumulate errors in inferred relations and lead to semantic and metric drift. To mitigate this, we propose invariant spatial decomposition to break coupled relation chains, and consistent spatial imagination to promote self-consistency through an imagine-and-revise loop. We further introduce supportive spatial optimization to ease pose optimization via global-to-local coordinate re-parameterization. Extensive experiments across diverse scene types and instructions demonstrate that R$^3$L produces more physically feasible and semantically consistent layouts. Notably, our analysis shows that resolving frame-induced inconsistencies is crucial for reliable multi-hop relative spatial reasoning. The code is available at this https URL. 

---
# Beyond Factor Aggregation: Gauge-Aware Low-Rank Server Representations for Federated LoRA 

**Authors**: Jinqian Chen, Chang Liu, Jihua Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.06733)  

**Abstract**: Federated LoRA enables parameter-efficient adaptation of large language models under decentralized data and limited client this http URL, directly averaging LoRA factors is representation-dependent: the same intrinsic update admits infinitely many gauge-equivalent factorizations, so factor-level aggregation can change under arbitrary coordinate choices while the underlying update remains unchanged. This reveals a semantic mismatch in existing federated LoRA aggregation rules. We propose \textbf{GLoRA}, a gauge-aware server representation for federated this http URL of aggregating raw factors, GLoRA estimates a consensus update subspace from client projectors and aggregates client updates in shared reference coordinates, thereby representing semantic update aggregation entirely in low-rank form. To support heterogeneous client capacities, GLoRA further provides a rank-compatible readout that instantiates adapters of different ranks from the same server state without dense update reconstruction. Experiments on GLUE and SuperNI show that GLoRA consistently outperforms federated LoRA baselines under data, resource, and task heterogeneity, including heterogeneous client ranks, sparse participation, larger backbones, and unseen-task evaluation. GLoRA also achieves a favorable efficiency--performance trade-off, suggesting that effective federated LoRA requires not merely averaging low-rank factors, but defining a semantically meaningful server-side representation for aggregation. 

---
# Gradient Extrapolation-Based Policy Optimization 

**Authors**: Ismam Nur Swapnil, Aranya Saha, Tanvir Ahmed Khan, Mohammad Ariful Haque, Ser-Nam Lim  

**Link**: [PDF](https://arxiv.org/pdf/2605.06755)  

**Abstract**: Reinforcement learning is widely used to improve the reasoning ability of large language models, especially when answers can be automatically checked. Standard GRPO-style training updates the model using only the current step, while full multi-step lookahead can give a better update direction but is too expensive because it needs many backward passes. We propose Gradient Extrapolation-Based Policy Optimization (GXPO), a plug-compatible policy-update rule for GRPO-style reasoning RL. GXPO approximates a longer local lookahead using only three backward passes during an active phase. It reuses the same batch of rollouts, rewards, advantages, and GRPO loss, so it does not require new rollouts or reward computation at the lookahead points. GXPO takes two fast optimizer steps, measures how the gradients change, predicts a virtual K-step lookahead point, moves the policy partway toward that point, and then applies a corrective update using the true gradient at the new position. When the lookahead signal becomes unstable, GXPO automatically switches back to standard single-pass GRPO. We also give a plain-gradient-descent surrogate analysis that explains when the extrapolation is exact and where its local errors come from. Across Qwen2.5 and Llama math-reasoning experiments, GXPO improves the average sampled pass@1 by +1.65 to +5.00 points over GRPO and by +0.14 to +1.28 points over the strongest SFPO setting, while keeping the active-phase cost fixed at three backward passes. It also achieves up to 4.00x step speedup, 2.33x wall-clock speedup, and 1.33x backward-pass speedup in reaching GRPO's peak accuracy. 

---
# A Self-Healing Framework for Reliable LLM-Based Autonomous Agents 

**Authors**: Cheonsu Jeong, Younggun Shin  

**Link**: [PDF](https://arxiv.org/pdf/2605.06737)  

**Abstract**: Autonomous agents based on Large Language Models (LLMs) are increasingly being utilized in complex software systems. However, reliability remains a significant challenge due to unpredictable failures such as hallucinations, execution errors, and inconsistent reasoning. This paper proposes a reliability-aware self-healing framework for LLM-based software agents. The framework integrates failure detection, reliability assessment, and automated recovery mechanisms. First, we define a taxonomy of failure types and introduce a quantitative reliability assessment model. Next, we propose a failure detection method that identifies abnormal agent behavior based on execution patterns and output consistency. Finally, we design a self-healing mechanism that dynamically recovers from failures through adaptive replanning and corrective prompting strategies. The proposed framework was implemented in a multi-agent workflow environment and evaluated using real-world task scenarios. Experimental results demonstrate that our approach significantly increases task success rates, reduces failure propagation, and enhances overall system robustness compared to existing methods. In particular, this study distinguishes itself by establishing an integrated monitoring system that combines the agent's internal reasoning process with external execution results. These findings are expected to contribute to securing the stability of advanced autonomous systems and lowering the barriers to LLM adoption in production environments. 

---
# VITA-QinYu: Expressive Spoken Language Model for Role-Playing and Singing 

**Authors**: Jiacheng Xu, Heting Gao, Liufei Xie, Zhenchuan Yang, Lijiang Li, Yiting Chen, Bin Zhang, Meng Chen, Chaoyu Fu, Weifeng Zhao, Wenjiang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.06765)  

**Abstract**: Human speech conveys expressiveness beyond linguistic content, including personality, mood, or performance elements, such as a comforting tone or humming a song, which we formalize as role-playing and singing. We present VITA-QinYu, the first expressive end-to-end (E2E) spoken language model (SLM) that goes beyond natural conversation to support both role-playing and singing generation. VITA-QinYu adopts a hybrid speech-text paradigm that extends interleaved text-audio modeling with multi-codebook audio tokens, a design enabling richer paralinguistic representation while preserving a clear separation between modalities to avoid interference. We further develop a comprehensive data generation pipeline to synthesize a total of 15.8K hours of natural conversation, role-playing, and singing data for training. VITA-QinYu demonstrates superior expressiveness, outperforming peer SLMs by 7 percentage points on objective role-playing benchmarks, and surpassing peer models by 0.13 points on a 5-point MOS scale for singing. Simultaneously, it achieves state-of-the-art conversational accuracy and fluency, exceeding prior SLMs by 1.38 and 4.98 percentage points on the C3 and URO benchmarks, respectively. We open-source our code and models and provide an easy-to-use demo with full-stack support for streaming and full-duplex interaction. 

---
# OmicsLM: A Multimodal Large Language Model for Multi-Sample Omics Reasoning 

**Authors**: Maciej Sypetkowski, Joanna Krawczyk, Łukasz Smoliński, Remigiusz Kinas, Przemysław Pietrzak, Tomasz Jetka, Rafał Powalski  

**Link**: [PDF](https://arxiv.org/pdf/2605.06728)  

**Abstract**: Interpreting transcriptomic data is one of the most common analytical tasks in modern biology. Yet most current models either consume expression profiles without producing natural-language biological explanations, or reason in language without direct access to quantitative omics measurements. We introduce OmicsLM, a multimodal LLM that connects quantitative omics profiles with natural-language biological tasks. OmicsLM represents each transcriptomic profile as a compact continuous representation within the LLM context. This interface preserves quantitative expression signal while allowing natural-language instructions, explicit gene mentions, and multiple interleaved biological samples to be processed together in one model context. We train OmicsLM on more than 5.5 million instruction-following examples spanning over 70 task types, combining continuous transcriptomic inputs, experimental data rendered through diverse language templates, and free-text biological knowledge and question-answering data. This mixture covers cell type annotation, perturbation prediction, clinical prediction, pathway reasoning, and open-ended biological question answering. Existing benchmarks evaluate either profile-level prediction or text-only biological QA, leaving language-guided, multi-sample reasoning over real expression profiles unmeasured. To close this gap, we introduce GEO-OmicsQA, a benchmark for multi-sample biological question answering built from real Gene Expression Omnibus (GEO) studies. We demonstrate that OmicsLM can use expression profiles directly and perform comparably to specialized omics models on profile-level tasks, while outperforming both omics-specialized models and general LLMs on language-guided biological reasoning over expression data. 

---
# Visual Text Compression as Measure Transport 

**Authors**: Lv Tang, Tianyi Zheng, Yang Liu, Bo Li, Xingyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.06708)  

**Abstract**: Visual text compression (VTC) promises efficient long-context processing by rendering text into an image and re-encoding it with a vision-language model, often producing $3$--$20\times$ fewer decoder tokens than subword tokenization. Yet token savings do not translate predictably into downstream utility: on some tasks the visual path matches or exceeds the text path, on others it collapses, and the compression ratio itself does not predict which regime will occur. The missing quantity is therefore not another summary of efficiency, but a principled measure of task-relevant information loss induced by visual encoding. We address this problem by formulating VTC in the language of measure transport. Treating text and visual tokens as empirical probability measures, we show that the ViT patch encoder induces a push-forward map whose transport cost decomposes into a precision cost from within-patch aggregation and a coverage cost from cross-patch fragmentation. Both terms are estimable from downstream-label-free probes. This formulation yields two operational consequences: a downstream-label-free routing criterion that selects whether to use the visual path for a given input or benchmark instance, and a transport-informed foveation mechanism that re-encodes high-cost regions at higher resolution. Across $24$ NLP datasets at Qwen3-4B, our label-free rule matches the per-dataset oracle on $17/24$ datasets ($70.8\%$), and improves the average task score by $+3.3\%$ with $-10.3\%$ average tokens relative to a pure-LLM. 

---
# The Single-File Test: A Longitudinal Public-Interface Evaluation of First-Output LLM Web Generation with Social Reach Tracking 

**Authors**: Diego Cabezas Palacios  

**Link**: [PDF](https://arxiv.org/pdf/2605.06707)  

**Abstract**: This paper presents an eight-week observational comparison of 68 single-file HTML generations collected across 17 public experiments in the "HTML AI Battle" project between December 10, 2025 and February 4, 2026. Four reasoning model families, GPT, Gemini, Grok, and Claude, were compared under a fixed public-interface protocol with no custom instructions, no personality tuning, and no repair prompts. Each output was evaluated from a rendered browser video using human scores and a Gemini LLM-as-a-judge layer for prompt adherence, functional correctness, and UI quality, then packaged into a standardized social-media protocol spanning X (Twitter), TikTok, and YouTube. The tracker was also used for two supervised predictive analyses: an experiment-level model for 24-hour X impressions and a generation-level model for HTML verbosity.
Under this protocol, Claude was the strongest and most consistent family, leading mean performance and winning 9/17 prompts under the primary human weighted score. Longer measured reasoning time was not associated with higher quality overall. Gemini as a judge was significantly more lenient than the human evaluator on functional correctness and overall performance, while stable self-favoring bias remained unresolved. The exploratory X-impressions model remained weak under post-screen cross-validation (MAE = 46,874, R^2 = -0.377), whereas the HTML-lines model performed better, with a model-family-only baseline outperforming prompt-aware alternatives (MAE = 135.2, R^2 = 0.576). Overall, selected pre-publication technical/audio variables were not sufficient to predict 24-hour X reach, while code verbosity was driven much more by model family than by prompt wording. The comparisons remain observational and are limited by public-interface drift, access-path differences, and one primary human scorer. 

---
# Domain-level metacognitive monitoring in frontier LLMs: A 33-model atlas 

**Authors**: Jon-Paul Cacioli  

**Link**: [PDF](https://arxiv.org/pdf/2605.06673)  

**Abstract**: Aggregate metacognitive quality scores mask within-model variation across MMLU benchmark domains. We administered 1,500 MMLU items (250 per domain, under an a priori six-domain grouping) to 33 frontier LLMs from eight model families and computed Type-2 AUROC per model-domain cell using verbalized confidence (0-100). Total observations: 47,151. Every model with above-chance aggregate monitoring showed non-trivial domain-level variation. Applied/Professional knowledge was reliably the easiest benchmark domain to monitor (mean AUROC = .742, ranked top-2 in 21 of 33 models); Formal Reasoning and Natural Science were reliably the hardest (one of the two ranked bottom-2 in 27 of 33 models). The three middle domains were statistically indistinguishable (Kendall's W = .164). A subject-level coherence analysis (within-domain similarity ratio = 0.95) confirms the six-domain grouping is a pragmatic benchmark taxonomy, not a validated latent construct. Within-family profile-shape clustering is significant for Anthropic, Google-Gemini, and Qwen (permutation p < .0001) but not DeepSeek, Google-Gemma, or OpenAI. Gemma 4 31B showed a +.202 AUROC improvement over Gemma 3 27B. Three models classified Invalid on binary KEEP/WITHDRAW probes produced normal profiles under verbalized confidence, confirming probe-format specificity. Bootstrap 95% CIs on 198 cells have median width .199. Split-half aggregate stability r = .893; profile-level split-half is weaker (grand median r = .184). These results show stable benchmark-domain variation obscured by aggregate metrics, and support benchmark-stage domain screening as a step before deployment in specific application areas. 

---
# Agentic Coding Needs Proactivity, Not Just Autonomy 

**Authors**: Nghi D. Q. Bui, Georgios Evangelopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2605.06717)  

**Abstract**: Coding agents are rapidly changing the landscape of software development, moving from inline completion to autonomous systems that edit repositories, open pull requests, respond to issues, and run scheduled or webhook triggered routines across the development life cycle. The next generation is increasingly described as proactive and long-horizon: agents should notice relevant changes before the developer asks, connect signals across tools, decide when to interrupt, and carry preferences across sessions. Yet the field still lacks a clear account of what proactivity means for software development, how it differs from autonomy, what acceptance criteria proactive long-horizon tasks should satisfy, and which metrics determine whether unsolicited agent behavior is useful rather than merely active. Proactive coding agents should be evaluated by the quality and improvement of their insight policy: the policy that decides what matters next, what evidence supports it, whether to show it, and how to adapt after feedback. This view is grounded in the principles of mixed initiative interaction. We propose a three level taxonomy of proactivity (Reactive, Scheduled, and Situation Aware), compare contemporary coding agents against five practical criteria, and sketch an active user simulation protocol with three evaluation targets: Insight Decision Quality (IDQ), Context Grounding Score (CGS), and Learning Lift 

---
# Agentic AI and the Industrialization of Cyber Offense: Forecast, Consequences, and Defensive Priorities for Enterprises and the Mittelstand 

**Authors**: Christopher Koch  

**Link**: [PDF](https://arxiv.org/pdf/2605.06713)  

**Abstract**: Agentic AI systems can plan, call tools, inspect code, interact with web applications, and coordinate multi-step workflows. These same capabilities change the economics of cyber offense. The central near-term risk is not that every low-skill criminal immediately becomes a frontier exploit researcher; it is that agentic AI compresses the attack lifecycle by lowering the cost of reconnaissance, phishing, credential abuse, vulnerability triage, exploit adaptation, and post-compromise decision support. This paper synthesizes current public evidence from national cybersecurity agencies, industry threat reports, agent security guidance, and research on LLM agents cyber capabilities. It introduces a Three Channel Agentic Cyber Risk Model and an Agentic Attack Compression Model, uses the 2026 Linux kernel Copy Fail incident as a case study for foothold-to-root acceleration, and develops a 2026 to 2028 forecast for large enterprises and the German and European Mittelstand. The paper concludes with a prioritized defense roadmap. Organizations should treat agentic AI security as an immediate operational problem: identity, phishing resistant authentication, patch velocity, CI/CD and Linux/container hardening, agent governance, telemetry, and recovery readiness must be strengthened now. 

---
# CommFuse: Hiding Tail Latency via Communication Decomposition and Fusion for Distributed LLM Training 

**Authors**: Rezaul Karim, Austin Wen, Wang Zongzuo, Weiwei Zhang, Yang Liu, Walid Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2604.24013)  

**Abstract**: The rapid growth in the size of large language models has necessitated the partitioning of computational workloads across accelerators such as GPUs, TPUs, and NPUs. However, these parallelization strategies incur substantial data communication overhead significantly hindering computational efficiency. While communication-computation overlap presents a promising direction, existing data slicing based solutions suffer from tail latency. To overcome this limitation, this research introduces a novel communication-computation overlap technique to eliminate this tail latency in state of the art overlap methods for distributed LLM training. The aim of this technique is to effectively mitigate communication bottleneck of tensor parallelism and data parallelism for distributed training and inference. In particular, we propose a novel method termed CommFuse that replaces conventional collective operations of reduce-scatter and all-gather with decomposed peer-to-peer (P2P) communication and schedules partitioned computations to enable fine-grained overlap. Our method provides an exact algorithm for reducing communication overhead that eliminates tail latency. Moreover, it presents a versatile solution compatible with data-parallel training and various tensor-level parallelism strategies, including TPSP and UP. Experimental evaluations demonstrate that our technique consistently achieves lower latency, superior Model FLOPS Utilization (MFU), and high throughput. 

---
# Toeplitz MLP Mixers are Low Complexity, Information-Rich Sequence Models 

**Authors**: Benjamin L. Badger, Ethan Roland  

**Link**: [PDF](https://arxiv.org/pdf/2605.06683)  

**Abstract**: Transformer-based large language models are in some respects limited by the quadratic time and space computational complexity of attention. We introduce the Toeplitz MLP Mixer (TMM), a transformer-like architecture that swaps attention for triangular-masked Toeplitz matrix multiplication over the sequence dimension resulting in $\mathcal{O} (dn \log n)$ time and $\mathcal O(dn)$ space complexity during training and $\mathcal O(dn)$ time and space at inference prefill. Despite the lack of sophisticated input modulation or state maintenance present in other sub-quadratic architectures, TMMs yield greater training efficiency in terms of loss achieved per compute and device memory. We demonstrate that TMMs are capable of retaining more input information resulting in improved copying ability, which we argue results from a lack of architectural biases. Consistent with higher input information retention, TMMs exhibit superior information retrieval and in-context learning benchmark accuracy compared to comparable architectures. We conclude with an analysis from the perspective of operator index theory and show that, counterintuitively, trained Toeplitz layers of causal non-invertible models are more likely to be invertible or nearly so than models that are actually invertible over their inputs. 

---
# Evaluating Prompt Injection Defenses for Educational LLM Tutors: Security-Usability-Latency Trade-offs 

**Authors**: Alexandre Cristovão Maiorano  

**Link**: [PDF](https://arxiv.org/pdf/2605.06669)  

**Abstract**: Educational LLM tutors face a core AI alignment challenge: they must follow user intent while preserving pedagogical constraints and safety policies. We present an evaluation methodology for prompt-injection defenses in this setting, showing that guardrail design entails explicit trade-offs among adversarial robustness, benign-task usability, and response latency. We evaluate a domain-specific multi-layer safeguard pipeline combining deterministic pattern filters, structural validation, contextual sandboxing, and session-level behavioral checks. On a controlled holdout benchmark with 480 queries (369 injection, 111 benign), the pipeline reaches 46.34% bypass, 0.00% false positive rate, and 2.50 ms average latency -- an operating point that prioritizes pedagogical usability (zero false positives) while maintaining measurable attack resistance. We provide a reproducible benchmark protocol for head-to-head comparison under identical conditions, including stratified bootstrap confidence intervals, paired McNemar significance tests, and direct evaluation of Prompt Guard and NeMo Guardrails on the same split with unified instrumentation. Results expose operational trade-offs: NeMo reaches 0% bypass at 16.22% FPR and 1.3s latency, while Prompt Guard yields 38.48% bypass with 3.60% FPR. The framework supports evidence-based guardrail selection for AI tutoring systems under different institutional risk and usability requirements. 

---
# Consensus Entropy: Harnessing Multi-VLM Agreement for Self-Verifying and Self-Improving OCR 

**Authors**: Yulong Zhang, Tianyi Liang, Xinyue Huang, Erfei Cui, Guoqing Wang, Xu Guo, Chenhui Li, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11101)  

**Abstract**: Optical Character Recognition (OCR) is fundamental to Vision-Language Models (VLMs) and high-quality data generation for LLM training. Yet, despite progress in average OCR accuracy, state-of-the-art VLMs still struggle with detecting sample-level errors and lack effective unsupervised quality control. We introduce Consensus Entropy (CE), a training-free, model-agnostic metric that estimates output reliability by measuring inter-model agreement entropy. The core insight is that correct predictions converge in output space, while errors diverge. Based on CE, we develop CE-OCR, a lightweight multi-model framework that verifies outputs by ensemble agreement, selects the best outputs, and further improves efficiency through adaptive routing. Experiments demonstrate that CE is robust for quality verification, improving F1 scores by 42.1% over VLM-as-Judge. CE-OCR achieves consistent OCR gains, outperforming self-consistency and single-model baselines at the same cost. Notably, CE requires no training or supervision, enabling plug-and-play integration. Code: this https URL. 

---
# LLMs Improving LLMs: Agentic Discovery for Test-Time Scaling 

**Authors**: Tong Zheng, Haolin Liu, Chengsong Huang, Huiwen Bao, Sheng Zhang, Rui Liu, Runpeng Dai, Ruibo Chen, Chenxi Liu, Tianyi Xiong, Xidong Wu, Hongming Zhang, Heng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.08083)  

**Abstract**: Test-time scaling (TTS) has become an effective approach for improving large language model performance by allocating additional computation during inference. However, existing TTS strategies are largely hand-crafted: researchers manually design reasoning patterns and tune heuristics by intuition, leaving much of the computation-allocation space unexplored. We propose an environment-driven framework, AutoTTS, that changes what researchers design: from individual TTS heuristics to environments where TTS strategies can be discovered automatically. The key to AutoTTS lies in environment construction: the discovery environment must make the control space tractable and provide cheap, frequent feedback for TTS search. As a concrete instantiation, we formulate width--depth TTS as controller synthesis over pre-collected reasoning trajectories and probe signals, where controllers decide when to branch, continue, probe, prune, or stop and can be evaluated cheaply without repeated LLM calls. We further introduce beta parameterization to make the search tractable and fine-grained execution trace feedback to improve discovery efficiency by helping the agent diagnose why a TTS program fails. Experiments on mathematical reasoning benchmarks show that the discovered strategies improve the overall accuracy--cost tradeoff over strong manually designed baselines. The discovered strategies generalize to held-out benchmarks and model scales, while the entire discovery costs only $39.9 and 160 minutes. Our data, and code will be open-source at this https URL. 

---
# Uncertainty-Aware Structured Data Extraction from Full CMR Reports via Distilled LLMs 

**Authors**: Yi Yu, Parker Martin, Zhenyu Bu, Yixuan Liu, Yi-Yu Zheng, Orlando Simonetti, Yuchi Han, Yuan Xue  

**Link**: [PDF](https://arxiv.org/pdf/2605.08045)  

**Abstract**: Converting free-text cardiac magnetic resonance (CMR) reports into auditable structured data remains a bottleneck for cohort assembly, longitudinal curation, and clinical decision support. We present CMR-EXTR, a lightweight framework that converts free-text CMR reports into structured data and assigns per-field confidence for quality control. A teacher-student distillation pipeline enables fully offline inference while limiting manual annotation. Uncertainty integrates three complementary principles -- distribution plausibility, sampling stability, and cross-field consistency -- to triage human review. Experiments show that CMR-EXTR achieves 99.65% variable-level accuracy, demonstrating both reliable extraction and informative confidence scores. To our knowledge, this is the first CMR-specific extraction system with integrated confidence estimation. The code is available at this https URL. 

---
# How to Train Your Latent Diffusion Language Model Jointly With the Latent Space 

**Authors**: Viacheslav Meshchaninov, Alexander Shabalin, Egor Chimbulatov, Nikita Gushchin, Ilya Koziev, Alexander Korotin, Dmitry Vetrov  

**Link**: [PDF](https://arxiv.org/pdf/2605.07933)  

**Abstract**: Latent diffusion models offer an attractive alternative to discrete diffusion for non-autoregressive text generation by operating on continuous text representations and denoising entire sequences in parallel. The major challenge in latent diffusion modeling is constructing a suitable latent space. In this work, we present the Latent Diffusion Language Model (LDLM), in which the latent encoder, diffusion model, and decoder are trained jointly. LDLM builds its latent space by reshaping the representations of a pre-trained language model with a trainable encoder, yielding latents that are easy to both denoise and decode into tokens. We show that naive joint training produces a low-quality diffusion model, and propose a simple training recipe consisting of an MSE decoder loss, diffusion-to-encoder warmup, adaptive timestep sampling, and decoder-input noise. Ablations show that each component substantially impacts generation performance. On OpenWebText and LM1B, LDLM achieves better generation performance than existing discrete and continuous diffusion language models while being $2{\text -}13\times$ faster, indicating that jointly learning the latent space is a key step toward making latent diffusion competitive for text generation. 

---
# Ask Early, Ask Late, Ask Right: When Does Clarification Timing Matter for Long-Horizon Agents? 

**Authors**: Anmol Gulati, Hariom Gupta, Elias Lumer, Sahil Sen, Vamse Kumar Subbiah  

**Link**: [PDF](https://arxiv.org/pdf/2605.07937)  

**Abstract**: Long-horizon AI agents execute complex workflows spanning hundreds of sequential actions, yet a single wrong assumption early on can cascade into irreversible errors. When instructions are incomplete, the agent must decide not only whether to ask for clarification but when, and no prior work measures how clarification value changes over the course of execution. We introduce a forced-injection framework that provides ground-truth clarifications at controlled points in the agent's trajectory across four information dimensions (goal, input, constraint, context), three agent benchmarks, and four frontier models (three per benchmark; one on a single benchmark only; 84 task variants; 6,000+ runs). Counter to the common intuition that "earlier is always better," we find that the value of clarification depends sharply on what information is missing: goal clarification loses nearly all value after 10% of execution (pass@3 drops from 0.78 to baseline), while input clarification retains value through roughly 50%. Deferring any clarification type past mid-trajectory degrades performance below never asking at all. Cross-model Kendall tau correlations (0.78-0.87 among models sharing identical task coverage; 0.34-0.67 across the full 4-model panel) confirm these timing profiles are substantially task-intrinsic. A complementary study of 300 unscripted sessions reveals that no current frontier model asks within the empirically optimal window, with strategies ranging from over-asking (52% of sessions) to never asking at all. These empirical demand curves provide the quantitative foundation that existing theoretical frameworks require but have lacked, and establish concrete design targets for timing-aware clarification policies. Code and data will be publicly released. 

---
# SCENE: Recognizing Social Norms and Sanctioning in Group Chats 

**Authors**: Mateusz Jacniacki, Maksymilian Bilski  

**Link**: [PDF](https://arxiv.org/pdf/2605.07823)  

**Abstract**: Online group chats are social spaces with implicit behavior patterns that, when broken, are often met with social sanctioning from the group. The ability and willingness of LLM-based agents to recognize and adapt to these norms remains mostly unexplored. We introduce SCENE, a social-interaction benchmark focused on implicit norms and social sanctioning in multi-party chat. SCENE generates plausible non-roleplay scenarios with scripted personas that follow a hidden norm, create opportunities for the subject agent to violate it, and sanction breaches when they occur. We further propose behavioral evaluation metrics for two functional adaptation abilities: responsiveness to negative sanctioning, and adapting norm from peers behavior. We evaluate six frontier and open-weight models on SCENE. Our results show that Claude Opus 4.7 and Gemini 3.1 Pro adapt to implicit norms significantly more than the evaluated open-weight models. SCENE contributes one benchmark in the direction of recent calls for dynamic, interactional evaluation of LLM social capabilities. 

---
# CktFormalizer: Autoformalization of Natural Language into Circuit Representations 

**Authors**: Jing Xiong, Qi Han, Chenchen Ding, He Xiao, Zunhai Su, Chaofan Tao, Ngai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2605.07782)  

**Abstract**: LLMs can generate hardware descriptions from natural language specifications, but the resulting Verilog often contains width mismatches, combinational loops, and incomplete case logic that pass syntax checks yet fail in synthesis or silicon. We present CktFormalizer, a framework that redirects LLM-driven hardware generation through a dependently-typed HDL embedded in Lean 4. Lean serves three roles: (i) type checker:dependent types encode bit-width constraints, case coverage, and acyclicity, turning hardware defects into compile-time errors that guide iterative repair; (ii) correctness firewall:compiled designs are structurally free of defects that cause silent backend failures (the baseline loses 20% of correct designs during synthesis and routing; CktFormalizer preserves all of them); (iii) proof assistant:the agent constructs machine-checked equivalence proofs over arbitrary input sequences and parameterized widths, beyond the reach of bounded SMT-based checking. On VerilogEval (156 problems), RTLLM (50 problems), and ResBench (56 problems), CktFormalizer achieves simulation pass rates competitive with direct Verilog generation while delivering substantially higher backend realizability: 95--100% of compiled designs complete the full synthesis, place-and-route, DRC, and LVS flow. A closed-loop PPA optimization stage yields up to 35% area reduction and 30% power reduction through validated architecture exploration, with automated theorem proof ensuring that each optimized variant remains functionally equivalent to its formal specification. 

---
# Chain-based Distillation for Effective Initialization of Variable-Sized Small Language Models 

**Authors**: Boyu Shi, YiCheng Jiang, Chang Liu, Qiufeng Wang, Xu Yang, Xin Geng  

**Link**: [PDF](https://arxiv.org/pdf/2605.07783)  

**Abstract**: Large language models (LLMs) achieve strong performance but remain costly to deploy in resource-constrained settings. Training small language models (SLMs) from scratch is computationally expensive, while conventional knowledge distillation requires repeated access to large teachers for different target sizes, leading to poor scalability. To solve these problems, we propose \textbf{Chain-based Distillation (CBD)}, a scalable paradigm for efficiently initializing variable-sized language models. A sparse and limited sequence of intermediate models (called anchors) is constructed via stepwise distillation, forming a distillation chain that progressively transfers knowledge from the source LLMs. To support heterogeneous settings, we introduce \emph{bridge distillation} for cross-architecture and cross-vocabulary transfer. Models of variable sizes are initialized via parameter interpolation between adjacent anchors, eliminating repeated large teacher inference. Experiments show that the proposed method substantially improves efficiency and downstream performance. A 138M-parameter SLM without recovery pre-training, outperforms scratch-trained models on a 10B-token corpus on the specific task. CBD also demonstrates versatility in heterogeneous settings for initialize models with different architectures and vocabularies. 

---
# How Value Induction Reshapes LLM Behaviour 

**Authors**: Arnav Arora, Natalie Schluter, Katherine Metcalf, Maartje ter Hoeve  

**Link**: [PDF](https://arxiv.org/pdf/2605.07925)  

**Abstract**: Conversational Large Language Models are post-trained on language that expresses specific behavioural traits, such as curiosity, open-mindedness, and empathy, and values, such as helpfulness, harmlessness, and honesty. This is done to increase utility, ensure safety, and improve the experience of the people interacting with the model. However, values are complex and inter-related -- inducing one could modify behaviour on another. Further, inducing certain values can make models more addictive or sycophantic through language used in the generations, with a potential detrimental effect on the user. We investigate these and other unintended effects of value induction into models. We fine-tune models using curated value subsets of existing preference datasets, measuring the impact of value induction on expression of other values, model safety, anthropomorphic language, and various QA benchmarks. We find that (i) inducing values leads to expression of other related, and sometimes contrastive values, (ii) inducing positive values increases safety, and (iii) all values increase anthropomorphic language use, making models more validating and sycophantic. 

---
# TextLDM: Language Modeling with Continuous Latent Diffusion 

**Authors**: Jiaxiu Jiang, Jingjing Ren, Wenbo Li, Bo Wang, Haoze Sun, Yijun Yang, Jianhui Liu, Yanbing Zhang, Shenghe Zheng, Yuan Zhang, Haoyang Huang, Nan Duan, Wangmeng Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2605.07748)  

**Abstract**: Diffusion Transformers (DiT) trained with flow matching in a VAE latent space have unified visual generation across images and videos. A natural next step toward a single architecture for both generation (visual synthesis) and understanding (text generation) is to apply this framework to language modeling. We propose TextLDM, which transfers the visual latent diffusion recipe to text generation with minimal architectural modification. A Transformer-based VAE maps discrete tokens to continuous latents, enhanced by Representation Alignment (REPA) with a frozen pretrained language model to produce representations effective for conditional denoising. A standard DiT then performs flow matching in this latent space, identical in architecture to its visual counterpart. The central challenge we address is obtaining high-quality continuous text representations: we find that reconstruction fidelity alone is insufficient, and that aligning latent features with a pretrained language model via REPA is critical for downstream generation quality. Trained from scratch on OpenWebText2, TextLDM substantially outperforms prior diffusion language models and matches GPT-2 under the same settings. Our results establish that the visual DiT recipe transfers effectively to language, taking a concrete step toward unified diffusion architectures for multimodal generation and understanding. 

---
# GLiGuard: Schema-Conditioned Classification for LLM Safeguard 

**Authors**: Urchade Zaratiana, Mary Newhauser, George Hurn-Maloney, Ash Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2605.07982)  

**Abstract**: Ensuring safe, policy-compliant outputs from large language models requires real-time content moderation that can scale across multiple safety dimensions. However, state-of-the-art guardrail models rely on autoregressive decoders with 7B--27B parameters, reformulating what is fundamentally a classification problem as sequential text generation, a design choice that incurs high latency and scales poorly to multi-aspect evaluation. In this work, we introduce \textbf{GLiGuard}, a 0.3B-parameter schema-conditioned bidirectional encoder adapted from GLiNER2 for LLM content moderation. The key idea is to encode task definitions and label semantics directly into the input sequence as structured token schemas, enabling simultaneous evaluation of prompt safety, response safety, refusal detection, 14 fine-grained harm categories, and 11 jailbreak strategies in a single non-autoregressive forward pass. This schema-conditioned design lets supported task and label blocks be composed directly in the input schema at inference time. Across nine established safety benchmarks, GLiGuard achieves F1 scores competitive with 7B--27B decoder-based guards despite being 23--90$\times$ smaller, while delivering up to 16$\times$ higher throughput and 17$\times$ lower latency. These results suggest that compact bidirectional encoders can approach the accuracy of much larger guard models while drastically reducing inference cost. Code and models are available at this https URL. 

---
# Guidance Is Not a Hyperparameter: Learning Dynamic Control in Diffusion Language Models 

**Authors**: Fan Zhou, Tim Van de Cruys  

**Link**: [PDF](https://arxiv.org/pdf/2605.07701)  

**Abstract**: Classifier-Free Guidance (CFG) is a widely used mechanism for controlling diffusion-based generative models, yet its guidance scale is typically treated as a fixed hyperparameter throughout generation. This static design yields a suboptimal controllability and quality tradeoff, as the optimal degree of guidance varies across tasks and across different stages of the diffusion process, especially in NLP domain. We recast CFG scale selection as a sequential decision-making problem and propose to learn dynamic guidance trajectories via reinforcement learning. Specifically, we model the guidance scale as a discrete control action selected at each generation step based on the evolving diffusion state, and optimize a policy using Proximal Policy Optimization (PPO) under task-level rewards. Experiments on three controlled NLP generation tasks using discrete diffusion language models demonstrate that adaptive guidance consistently achieves a better balance between controllability and generation quality than fixed-scale strategies. Further analysis of the learned policies reveals distinct and interpretable guidance trajectories across tasks, underscoring the importance of treating guidance as a dynamic control process rather than a static design choice. 

---
# Not All Tokens Learn Alike: Attention Entropy Reveals Heterogeneous Signals in RL Reasoning 

**Authors**: Gengyang Li, Zheng-Fan Wu, Siqi Bao, Yunfang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07660)  

**Abstract**: Reinforcement-learning-based post-training has become a key approach for improving the reasoning ability of large language models, but its token-level learning signals remain poorly understood. This work studies their heterogeneity through attention entropy, which measures how concentrated or diffuse the contextual support is for each response token.
We first show that token-level RL objectives are sparsely estimable: uniformly random 20 percent token subsets preserve much of the full-token held-out performance, suggesting substantial redundancy in token-level updates. However, entropy-structured subsets behave very differently. Low-attention-entropy tokens, which we call anchors, rely on concentrated support, produce stable gradients aligned with full-token updates, and provide a reliable optimization backbone, but tend to plateau on harder benchmarks. High-attention-entropy tokens, which we call explorers, aggregate more diffuse context and induce larger but more volatile gradients. Explorer-only training is unstable on average, though rare successful runs suggest that these tokens may contain useful hard-reasoning signals when optimization remains stable.
We support this anchor-explorer spectrum with evidence-gathering analyses, entropy dynamics, gradient-geometry diagnostics, and controls showing that position, predictive entropy, and loss normalization do not explain the observed asymmetry. Finally, a dynamic entropy-aware soft-reweighting intervention improves Qwen3-8B-Base from 34.39 to 37.40 held-out average in the strongest setting. These findings suggest that attention entropy reveals optimization-relevant structure in token-level RL signals, and that uniform token averaging can obscure meaningful heterogeneity in reasoning post-training. 

---
# SimCT: Recovering Lost Supervision for Cross-Tokenizer On-Policy Distillation 

**Authors**: Jie Sun, Mao Zheng, Mingyang Song, Qiyong Zhong, Yilin Cheng, Bichuan Feng, Pengfei Liu, Junfeng Fang, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07711)  

**Abstract**: On-policy distillation (OPD) is a standard tool for transferring teacher behavior to a smaller student, but it implicitly assumes that teacher and student predictions are comparable token by token, an assumption that fails whenever the two models tokenize the same text differently. Under heterogeneous tokenizers, exact shared-token matching silently discards a large fraction of the teacher signal at precisely the positions where vocabularies disagree. We propose \textbf{\underline{Sim}ple \underline{C}ross-\underline{T}okenizer OPD (SimCT)}, which restores this signal by enlarging the supervision space: alongside shared tokens, SimCT compares teacher and student over short multi-token continuations that both tokenizers can realize, leaving the OPD loss form itself unchanged. We show that these units are the finest jointly tokenizable supervision interface, and that coarser alternatives remove teacher-student distinctions that are useful for on-policy learning. Across three heterogeneous teacher-student pairs on mathematical reasoning and code-generation benchmarks, SimCT shows consistent gains over shared-vocabulary OPD and representative cross-tokenizer baselines, with ablations confirming that the improvements come from recovering supervision discarded by exact shared-token matching. Code is available at \href{this https URL}{this https URL}. 

---
# Multi-Dimensional Evaluation of LLMs for Grammatical Error Correction 

**Authors**: Adnan Labib, Qiao Wang, Yixuan Huang, Zheng Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2605.07635)  

**Abstract**: Automated assistants for Grammatical Error Correction are now embedded in educational platforms serving millions of learners, yet three critical gaps remain in this domain: (1) latest-generation Large Language Models (LLMs) lack comprehensive evaluation on grammar correction tasks; (2) whether combining these LLMs improves correction quality is unexplored; and (3) the extent to which reference-based metrics underestimate GEC system performance has not been adequately quantified. In this study, first, we evaluate latest-generation LLMs on edit precision, fluency preservation, and meaning retention, showing fine-tuned GPT-4o achieves state-of-the-art performance across all three dimensions. Second, through grammatical error type analysis we demonstrate that individual LLMs exhibit highly similar error correction patterns ($\rho=0.947$). Third, we show that reference-based metrics underestimate GEC performance with 73.76% of GPT-4o corrections different from gold standards being equally valid or even superior. These GEC evaluation findings equip educators with guidance for selecting GEC assistants that enhance rather than constrain student linguistic development. We make our data, code, and models publicly available. 

---
# Beyond "I cannot fulfill this request": Alleviating Rigid Rejection in LLMs via Label Enhancement 

**Authors**: Ying Zhang, Congyu Qiao, Xin Geng, Ning Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07883)  

**Abstract**: Large Language Models (LLMs) rely on safety alignment to obey safe requests while refusing harmful ones. However, traditional refusal mechanisms often lead to "rigid rejection," where a general template (e.g., "I cannot fulfill this request") indiscriminately triggers refusals and severely undermines the naturalness of interactions between humans and LLMs. To address this issue, LANCE is proposed in this paper to ensure safe yet flexible and natural responses via label enhancement. Specifically, LANCE employs variational inference to perform label enhancement, predicting a continuous distribution across multiple rejection categories. These fine-grained rejection distributions provide multi-way textual gradients for a refinement model to neutralize the hazardous elements in the prompt, so that the LLMs could generate safe responses that avoid rigid rejections while preserving the naturalness of interactions. Experiments demonstrate that LANCE significantly alleviates the rigid rejection problem while maintaining high security standards, significantly outperforming existing baseline models in terms of helpfulness and naturalness of responses. 

---
# Measuring and Mitigating the Distributional Gap Between Real and Simulated User Behaviors 

**Authors**: Shuhaib Mehri, Philippe Laban, Sumuk Shashidhar, Marwa Abdulhai, Sergey Levine, Michel Galley, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2605.07847)  

**Abstract**: As user simulators are increasingly used for interactive training and evaluation of AI assistants, it is essential that they represent the diverse behaviors of real users. While existing works train user simulators to generate human-like responses, whether they capture the broad and heterogeneous distribution of real user behaviors remains an open question. In this work, we introduce a method to measure the distributional gap between real and simulated user behaviors, validated through a human study and ablations. Given a dataset of real and simulated conversations, our method extracts representations of user behavior from each conversation, quantizes them into discrete distributions via clustering, then computes divergence metrics. We provide the first systematic evaluation of 24 LLM-based user simulators on coding and writing tasks, and reveal a large distributional gap from real users that varies across model families, scales, and behavioral facets. Pairwise comparisons show that most simulators behave similarly, while a few stand apart. Combining behaviorally complementary simulators brings the resulting distribution closer to real users compared to either simulator on its own. Finally, a TF-IDF analysis of the clusters surfaces interpretable patterns of behaviors that simulators capture, miss, and hallucinate. 

---
# TCMIIES: A Browser-Based LLM-Powered Intelligent Information Extraction System for Academic Literature 

**Authors**: Hanqing Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.07507)  

**Abstract**: The exponential growth of academic publications has created an urgent need for automated tools capable of extracting structured knowledge from unstructured scientific texts. While large language models (LLMs) have demonstrated remarkable capabilities in natural language understanding and information extraction, existing solutions often require specialized infrastructure, programming expertise, or fine-tuned domain-specific models that create barriers for researchers in specialized fields. This paper presents TCMIIES, a browser-based, zero-installation platform that leverages commercial LLM APIs to perform structured information extraction from academic literature. The system employs a novel schema-guided prompting framework with automatic system prompt generation, enabling researchers to define custom extraction schemas through an intuitive graphical interface without any programming. TCMIIES features a pure front-end architecture that ensures data privacy by processing all information locally in the browser, supports five major LLM providers, implements concurrent batch processing with automatic retry mechanisms, and provides intelligent field mapping for Chinese academic databases including CNKI and Wanfang. We demonstrate the system's effectiveness through comprehensive evaluation across multiple extraction scenarios in Traditional Chinese Medicine research, achieving structured output compliance rates exceeding 94\% and information extraction accuracy comparable to domain-expert annotation. The system represents a practical, accessible solution that bridges the gap between advanced LLM capabilities and domain-specific academic information extraction needs, particularly for researchers in specialized fields who require flexible, privacy-preserving, and cost-effective extraction tools. 

---
# WeatherSyn: An Instruction Tuning MLLM For Weather Forecasting Report Generation 

**Authors**: Zinan Zheng, Yang Liu, Nuo Chen, Juepeng Zheng, Hong Cheng, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.07522)  

**Abstract**: Accurate weather forecast reporting enables individuals and communities to better plan daily activities and agricultural operations. However, the current reporting process primarily relies on manual analysis of multi-source data, which leads to information overload and reduced efficiency. With the development of multimodal large language models (MLLMs), leveraging data-driven models to analyze and generate reports in the weather forecasting domain remains largely underexplored. In this work, we propose the Weather Forecasting Report (WFR) task and construct the first instruction-tuning dataset for this task, named~\DatasetNameL, which covers 31 cities in America and 8 weather aspects. Based on this corpus, we develop the first model, \ModelNameL, specialized in generating weather forecast reports. Evaluation across multiple metrics on our dataset shows that \ModelNameL~ consistently outperforms leading closed-source MLLMs, particularly on structurally complex weather aspects. We further analyze its performance across diverse geographic regions and weather aspects. \ModelNameL~ demonstrates strong transferability across different regions, highlighting its zero-shot generalization capability. \ModelNameL~offers valuable insight for developing MLLMs specialized in weather report generation. . 

---
# Intent-Driven Semantic ID Generation for Grounded Conversational News Recommendation 

**Authors**: Hongyang Su, Beibei Kong, Lei Cheng, Chengxiang Zhuo, Zang Li, Chenyun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07613)  

**Abstract**: Conversational news recommendation requires grounding each suggestion in a rapidly evolving article corpus while addressing implicit user intents that lack explicit retrievable keywords. To characterize this scenario, we identify 6 intent types from production dialogues: five are implicit and pose fundamental challenges to standard RAG pipelines, forming a critical retrieve-first bottleneck. To address these issues, we introduce intent-driven Semantic ID (SID) generation under a Generate-then-Match paradigm. With two-stage training that consists of multi-task SID alignment and GPT-4 Chain-of-Thought distillation, an LLM maps diverse intents to hierarchical SID prefixes, which are then fuzzy-matched to the current news pool to guarantee fully grounded recommendations. Profile-Aware Dual-Signal Reasoning (PADR) further enables cold-start users to obtain valid recommendations using only profiles. On a mainstream Chinese news platform, our 7B model achieves 0% hallucination and 12.4% L1 match in the 152K open-generation SID space (4x random baseline). It matches GPT-4+Hybrid RAG on L1 while surpassing it on finer-grained metrics (L2 2x, Category +1.2pp) at ~100x lower cost. Cold-start users, where existing baselines score 0%, achieve 18.0% L1 (6x random), the highest among all user groups. 

---
# Why do Large Language Models Fail in Low-resource Translation? Unraveling the Token Dynamics of Large Language Models for Machine Translation 

**Authors**: Shenbin Qian, Yves Scherrer  

**Link**: [PDF](https://arxiv.org/pdf/2605.07533)  

**Abstract**: Large Language Models (LLMs) have recently demonstrated strong performance in machine translation (MT). However, most prior work focuses on improving or benchmarking translation quality, offering limited insight into when and why LLM-based translation fails. In this work, we systematically analyze failure modes of LLMs in MT by evaluating 15 models, including four reasoning LLMs, across 22 language pairs (LPs) with varying resource levels. We find that non-English-centric LPs consistently yield lower COMET scores than English-centric pairs. To investigate the underlying causes, we introduce Token Activation Rate (TAR), a metric that captures how effectively a model utilizes language-specific tokens in its vocabulary during generation. We validate TAR as a proxy for language representation using models with known language distributions in the training data, and show that lower TAR is strongly associated with poorer translation performance. Furthermore, reasoning LLMs tend to generate more tokens when translating into low-TAR languages, suggesting a compensatory mechanism, although its impact on translation quality varies across models. Overall, our findings emphasize the importance of token-level dynamics in understanding MT performance of LLMs. 

---
# SEIF: Self-Evolving Reinforcement Learning for Instruction Following 

**Authors**: Qingyu Ren, Qianyu He, Jiajie Zhu, Xingzhou Chen, Jingwen Chang, Zeye Sun, Han Xia, Fei Yu, Jiaqing Liang, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2605.07465)  

**Abstract**: Instruction following is a fundamental capability of large language models (LLMs), yet continuously improving this capability remains challenging. Existing methods typically rely either on costly external supervision from humans or strong teacher models, or on self-play training with static-difficulty instructions that cannot evolve as the model's capabilities improve. To address these limitations, we propose SEIF (Self-Evolving Reinforcement Learning for Instruction Following), a self-evolving framework for enhancing the instruction-following ability of LLMs. SEIF forms a closed self-evolution loop that improves the model's instruction-following ability, where instruction difficulty evolution and model capability evolution reinforce each other. SEIF consists of four roles: an Instructor that generates increasingly challenging instructions, a Filter that removes conflicting or invalid instructions to ensure data quality, a Follower that learns to follow evolved instructions, and a Judger that provides reward signals for reinforcement learning. The Instructor and Follower are alternately trained and co-evolve throughout the process. Experiments across multiple model scales and architectures show that SEIF consistently improves instruction-following performance, suggesting strong generality. Further analyses reveal the sources of improvement and identify an effective training strategy for self-evolution on open-ended tasks: sufficient early-stage training to build a solid foundation, followed by moderate late-stage training to mitigate overfitting and achieve better final performance. The code and data are publicly available at this https URL. 

---
# Think-with-Rubrics: From External Evaluator to Internal Reasoning Guidance 

**Authors**: Jiachen Yu, Zhihao Xu, Junjie Wang, Yujiu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07461)  

**Abstract**: Rubrics have been extensively utilized for evaluating unverifiable, open-ended tasks, with recent research incorporating them into reward systems for reinforcement learning. However, existing frameworks typically treat rubrics only as external evaluator disjointed from the policy's primary reasoning trace. Such design confines rubrics to post-hoc measurement, leaving them unable to actively guide the model's generation process. In this work, we introduce Think-with-Rubrics, a novel paradigm for instruction following tasks. Think-with-Rubrics integrates rubric generation into the reasoning context, transforming the rubric from an independent artifact into an internal guidance of LLM's generation. During training, LLM sequentially generates a rubric followed by a response, while a trained rubric verifier provides joint supervision by evaluating the consistency between the answer and the self-generated / golden rubrics. Experiments across multiple benchmarks demonstrate that Think-with-Rubrics consistently outperforms the Rubric-as-Reward baseline supervised by golden rubrics by an average of 3.87 points. We have also discussed the mechanism by which Think-with-Rubrics enhances model performance. Experimental results demonstrate that supervision from golden rubrics and self-generated rubrics enhances the performance of Think-with-Rubrics by improving the quality of self-generated rubrics and increasing the internal consistency of responses respectively. 

---
# Is She Even Relevant? When BERT Ignores Explicit Gender Cues 

**Authors**: Jonas Klein, Chiara Manna, Eva Vanmassenhove  

**Link**: [PDF](https://arxiv.org/pdf/2605.07622)  

**Abstract**: Gender bias in large language models has primarily been investigated for English, while languages with grammatical or morphological gender remain comparatively understudied. This paper investigates how and when gender information emerges in a Dutch BERT model trained from scratch, offering one of the first checkpoint-level analyses of bias formation in a Transformer architecture for a language combining overt morphological gender marking and generic forms. By extracting contextual embeddings throughout training, we construct dynamic gender subspaces using linear SVMs to trace when gender becomes linearly encoded and how this encoding evolves over time. Contextual embeddings are often assumed to integrate contextual cues robustly, allowing models to adjust the representation of a word depending on its more local usage. We therefore test whether explicit gender cues in controlled sentence templates (e.g., Zij is een loodgieter ('She is a plumber')) can override learned statistical associations (plumber -> male). Our findings challenge this assumption: although gender becomes clearly linearly separable around epoch 20 and is distributed across multiple embedding dimensions, the model struggles to update its internal gender representation in light of explicit contextual cues in short sentence templates. Stereotypical gender-profession pairings are predicted far more accurately than anti-stereotypical ones, and generic forms in Dutch systematically default to a male interpretation, even when the context explicitly denotes a female referent. Together, our results seem to indicate that contextualization in the representations learned by our Dutch BERT model is not sufficiently dynamic along the probed gender direction: explicit gender cues in anti-stereotypical contexts are not reliably reflected in the resulting representations, resulting in persistent male-default behaviour. 

---
# GRaSp: Automatic Example Optimization for In-Context Learning in Low-Data Tasks 

**Authors**: Simen Bihaug-Frøyland, Henrik Brådland  

**Link**: [PDF](https://arxiv.org/pdf/2605.07454)  

**Abstract**: In-context learning enables large language models to adapt to new tasks, but their performance is highly sensitive to the selected examples. Finding effective demonstrations is particularly difficult in domain-specific, low-data settings where high-quality examples are scarce. We propose GRaSp, a three-stage framework for automatic in-context example optimization. By first generating a large synthetic candidate pool, then structuring it with clustering and dimensionality reduction, and finally using genetic algorithms to find the optimal in-context examples, the framework shows consistent improvements on the NER task. We also introduce a custom diversity-adaptive mutation mechanism, allowing it to transition from the initial broad inter-cluster exploration to focused intra-cluster refinement as the population converges. We evaluate GRaSp on financial named entity recognition (FiNER-139), comparing synthetic and human-annotated candidate pools across pool sizes of 500 and 5000. With non-synthetic data, GRaSp achieves 45.84% micro-F1, consistently outperforming both zero-shot and random few-shot baselines. Synthetic data matches the random baseline but does not exceed it, suggesting that distributional variety in the candidate pool is critical for generalization. 

---
# Rethinking Dense Sequential Chains: Reasoning Language Models Can Extract Answers from Sparse, Order-Shuffling Chain-of-Thoughts 

**Authors**: Yi-Chang Chen, Feng-Ting Liao, Da-shan Shiu, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.07307)  

**Abstract**: Modern reasoning language models generate dense, sequential chain-of-thought traces implicitly assuming that every token contributes and that steps must be consumed in order. We challenge both assumptions through a systematic intervention pipeline--removal, masking, shuffling, and noise injection--applied to model-generated reasoning chains across three models and three benchmarks. Our findings are counterintuitive on three dimensions. Order: Does the sequential order of a reasoning chain matter for answer extraction? No--line-level shuffling reduces accuracy by less than 0.5 pp; word-level shuffling retains 62%-89% accuracy; only token-level shuffling collapses to near zero. Pretrained-only and instruction-tuned variants exhibit near-identical tolerance (78.67% vs. 78.00% under line shuffling), indicating order-independence originates from pretraining rather than reasoning-specific fine-tuning. Dense: Is all the information in a reasoning chain important for answer extraction? No--masking numeric digits collapses accuracy to exactly 0%, while masking alphabetic prose improves accuracy by 4.7 pp. Robustness: Is a reasoning chain that is both order-shuffling and non-dense still robust? Yes--the most aggressively reduced representation (all natural language removed, lines arbitrarily shuffled) still achieves 83% accuracy, and injecting false answers at 3x true-answer frequency leaves accuracy unchanged (83.3%->83.3%), falsifying a frequency-based extraction account. These results establish that answer extraction operates on a sparse, order-insensitive, and structurally robust informational substrate, opening paths toward parallelized and token-efficient reasoning generation. 

---
# Gradient-Based LoRA Rank Allocation Under GRPO: An Empirical Study 

**Authors**: Yash Ganpat Sawant  

**Link**: [PDF](https://arxiv.org/pdf/2605.07366)  

**Abstract**: Adaptive rank allocation for LoRA, allocating more parameters to important layers and fewer to unimportant ones, consistently improves efficiency under supervised fine-tuning (SFT). We investigate whether this success transfers to reinforcement learning, specifically Group Relative Policy Optimization (GRPO). Using gradient-magnitude profiling on Qwen 2.5 1.5B with GSM8K, we find that it does not: proportional rank allocation degrades accuracy by 4.5 points compared to uniform allocation (70.0% vs. 74.5%), despite using identical parameter budgets. We identify two mechanisms behind this failure. First, the gradient landscape under GRPO is fundamentally flatter than under SFT, the max-to-min layer importance ratio is only 2.17x, compared to >10x reported in SFT literature. All layers carry meaningful gradient signal; none are truly idle. Second, we discover a gradient amplification effect: non-uniform allocation widens the importance spread from 2.17x to 3.00x, creating a positive feedback loop where high-rank layers absorb more gradient while low-rank layers are progressively silenced. Our results suggest that gradient importance does not predict capacity requirements under RL, and that naive transfer of SFT-era rank allocation to alignment training should be avoided. 

---
# PaT: Planning-after-Trial for Efficient Test-Time Code Generation 

**Authors**: Youngsik Yoon, Sungjae Lee, Seockbean Song, Siwei Wang, Wei Chen, Jungseul Ok  

**Link**: [PDF](https://arxiv.org/pdf/2605.07248)  

**Abstract**: Beyond training-time optimization, scaling test-time computation has emerged as a key paradigm to extend the reasoning capabilities of Large Language Models (LLMs). However, most existing methods adopt a rigid Planning-before-Trial (PbT) policy, which inefficiently allocates test-time compute by incurring planning overhead even on directly solvable problems. We propose Planning-after-Trial (PaT), an adaptive policy for code generation that invokes a planner only upon verification failure. This adaptive policy naturally enables a heterogeneous model configuration: a cost-efficient model handles generation attempts, while a powerful model is reserved for targeted planning interventions. Empirically, across multiple benchmarks and model families, our approach significantly advances the cost-performance Pareto frontier. Notably, our heterogeneous configuration achieves performance comparable to a large homogeneous model while reducing inference cost by approximately 69\%. 

---
# Mean-Pooled Cosine Similarity is Not Length-Invariant: Theory and Cross-Domain Evidence for a Length-Invariant Alternative 

**Authors**: Sibayan Mitra, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2605.07345)  

**Abstract**: Mean-pooled cosine similarity is the default metric for comparing neural representations across languages, modalities, and tasks. We establish that this metric is not length-invariant: under the anisotropy that characterizes modern transformer representations, mean-pooled cosine grows monotonically in sequence length, independent of representational content. Empirically, on HumanEvalPack across four code LLMs, the length ratio alone explains $R^2 = 0.52$--$0.75$ of cross-language "Python proximity," while AST depth and shared-token fraction add less than 3% of explained variance beyond length. Substituting Centered Kernel Alignment (CKA) reduces explained variance by 83% and reverses the sign of the length coefficient ($\beta_{\mathrm{len}}: +0.86 \to -0.37$). The same pattern holds in Mistral-7B on parallel WMT pairs ($R^2 = 0.23$ EN-FR, $R^2 = 0.33$ EN-DE for cosine; $R^2 < 0.01$ for CKA). In CLIP ViT-B/32, mean-pooling reduces the length effect relative to EOS-pooling ($R^2: 0.21 \to {<}0.01$), as predicted by the theory's dependence on anisotropy. We argue that length-invariant metrics such as CKA should be the default for cross-representation comparisons, and that recent claims of cross-lingual representational convergence built on mean-pooled cosine warrant re-examination. 

---
# From 0-Order Selection to 2-Order Judgment: Combinatorial Hardening Exposes Compositional Failures in Frontier LLMs 

**Authors**: Hanmeng Liu, Shichao Weng, Xiulai Liu, Zhicai Zhang, Anli Yan, Xiaozhang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07268)  

**Abstract**: Multiple-choice reasoning benchmarks face dual challenges: rapid saturation from advancing models and data contamination that undermines static evaluations. Ad-hoc hardening methods (paraphrasing, perturbation) attempt to increase difficulty but sacrifice logical validity for surface complexity, falling short to challenge advanced reasoning models. We present LogiHard, a formal framework that deterministically transforms 0-order selection into 2-order logical judgment, which significantly increases the thinking overhead and reasoning steps. The framework integrates Item Response Theory (IRT) for computerized adaptive testing (CAT), enabling precise difficulty control with fewer questions than static benchmarks. We instantiate LogiHard-2k, a logical reasoning dataset constructed by cognitively ranking high-stakes examination questions via 9-dimensional analysis of model thinking traces, followed by combinatorial transformation of high-difficulty items. Evaluation across twelve state-of-the-art models reveals an accuracy degradation ranging from 31% to 56% on combinatorially hardened questions. LLMs suffer from the multi-select failure and early exit bias, which are not shared by human testees. Zero-shot transfer to MMLU demonstrates 47% accuracy degradation (89.84% to 42.86%), confirming applicability across domains with provable validity preservation. The consistent aggregate degeneration is domain-agnostic and stems not from knowledge deficits but from a combinatorial reasoning gap, reflecting a training-induced completeness-verification deficit. 

---
# The Proxy Presumption: From Semantic Embeddings to Valid Social Measures 

**Authors**: Baishi Li, Ta Yu, Kelvin J.L. Koa, Ke-Wei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07409)  

**Abstract**: Natural Language Processing is rapidly evolving into a primary instrument for Computational Social Science, with researchers increasingly using embeddings to measure latent constructs such as novelty, creativity, and bias. However, this transition faces a fundamental validity challenge: the ''Proxy Presumption,'' or the reliance on geometric properties (e.g., cosine distance) as direct measures of social concepts. We argue that without explicit validation, unsupervised representations remain entangled mixtures of the target construct ($C$) and confounding attributes ($Z$) like topic, style, and authorship. To bridge the gap between semantic embeddings and valid social measures, we introduce the Construct Validity Protocol (CVP). Drawing on causal representation learning and psychometrics, the CVP offers a rigorous pipeline from conceptualization to quantitative verification. We further propose Counterfactual Neutralization, a novel method using LLMs to reduce confounding in embedding space. By providing a standardized Validity Suite -- including tests for discriminant, incremental, and predictive validity -- this work offers the community a toolkit to transform heuristic proxies into robust, scientifically defensible instruments. 

---
# Learning Agent Routing From Early Experience 

**Authors**: Yimin Wang, Jiahao Qiu, Xuan Qi, Xinzhe Juan, Jingzhe Shi, Zelin Zhao, Hongru Wang, Shilong Liu, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07180)  

**Abstract**: LLM agents achieve strong performance on complex reasoning tasks but incur high latency and compute cost. In practice, many queries fall within the capability boundary of cutting-edge LLMs and do not require full agent execution, making effective routing between LLMs and agents a key challenge. We study the problem of routing queries between lightweight LLM inference and full agent execution under realistic cold-start settings. To address this, we propose BoundaryRouter, a training-free routing framework that uses early behavioral experience and rubric-guided reasoning to decide whether to answer a query with direct LLM inference or escalate to an agent. BoundaryRouter builds a compact experience memory by executing both systems on a shared seed set and retrieves similar cases at inference time to guide routing decisions. To evaluate this method, we introduce RouteBench, a benchmark covering in-domain, paraphrased, and out-of-domain route settings. Experiments show that BoundaryRouter reduces inference time by 60.6% compared to the agent while improving performance by 28.6% over direct LLM inference, outperforming prompt-based and retrieval-only routing by an average of 37.9% and 8.2%, respectively. 

---
# MIPIAD: Multilingual Indirect Prompt Injection Attack Defense with Qwen -- TF-IDF Hybrid and Meta-Ensemble Learning 

**Authors**: Al Muhit Muhtadi, Mostafa Rifat Tazwar  

**Link**: [PDF](https://arxiv.org/pdf/2605.07269)  

**Abstract**: Indirect prompt injection remains a persistent weakness in retrieval-augmented and tool-using LLM systems, and the problem becomes harder to characterise in multilingual settings. We present MIPIAD, a defense framework evaluated on English and Bangla that combines a sequence classifier fine-tuned from Qwen2.5-1.5B via LoRA (XLPID), TF-IDF lexical features, and validation-tuned ensembling through late fusion, stacking, and gradient boosting. The framework is evaluated on a synthetic benchmark built from BIPIA(Yi et al., 2023) templates spanning five task families -- email, table, QA, abstract, and code-comprising over 1.43 million generated samples, with train and test splits using mutually exclusive attack categories. Across the experiments, lexical signals prove strong (TF-IDF+SVM F1=0.77), and the hybrid XLPID+TF-IDF ensemble achieves the best overall F1 (0.9205) while the Boosting Ensemble achieves the best AUROC (0.9378). Ensemble methods consistently reduce the English-Bangla cross-lingual gap relative to standalone neural models. The pipeline is designed for extensibility: NLLB-200 supports over 200 languages and XLPID's multilingual backbone can be retargeted to additional languages without architectural changes; empirical validation is currently limited to English and Bangla 

---
# Topology-Enhanced Alignment for Large Language Models: Trajectory Topology Loss and Topological Preference Optimization 

**Authors**: Yurui Pan, Ke Xu, Bo Peng  

**Link**: [PDF](https://arxiv.org/pdf/2605.07172)  

**Abstract**: Alignment of large language models (LLMs) via SFT and RLHF/DPO typically ignores the global geometry of the representation space, relying instead on local token likelihoods or scalar scores. We view generation as tracing a semantic trajectory in hidden space and propose a topology-enhanced alignment framework that regularizes these trajectories using 0-dimensional persistent homology. First, for SFT, we introduce Trajectory Topology Loss (TTL). Treating prompt and gold-answer embeddings as a mixed point cloud, we use a 0D persistent homology algorithm to extract "prompt-answer bridges." TTL aligns the model's actual update direction with these topological bridges rather than arbitrary directions. Second, for DPO, we propose Topological Preference Optimization (TPO). TPO constructs topic-specific semantic preference vectors and aligns the improvement direction between rejected and chosen responses with these vectors in an intermediate hidden layer. We also introduce a dynamic weighting scheme to balance DPO and TPO losses. Evaluating on Qwen2.5-7B-Instruct using UltraChat and Anthropic HH-RLHF, our topology-enhanced objectives consistently outperform strong non-topological baselines (e.g., per-example, nearest-neighbor, random regularizers) on automatic preference metrics and LLM-judge evaluations, while maintaining or improving toxicity. Results show persistent homology and trajectory geometry offer a promising direction for controllable alignment. 

---
# A Reproducible Multi-Architecture Baseline for Token-Level Chinese Metaphor Identification under the MIPVU Framework 

**Authors**: Yufeng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07170)  

**Abstract**: Metaphor is pervasive in everyday language, yet token-level computational identification of metaphor-related words in Chinese under the MIPVU framework remains under-explored relative to English. This paper presents a reproducible multi-architecture baseline for token-level metaphor identification on the PSU Chinese Metaphor Corpus (PSU CMC), the only widely available MIPVU-annotated Chinese corpus. We systematically compare three model families: (i) encoder fine-tuning with Chinese RoBERTa-wwm-ext-large; (ii) MelBERT adapted to Chinese using a newly constructed basic-meaning resource derived from the Modern Chinese Dictionary, 7th edition (MCD7), comprising 74,823 entries with 71.51% PSU CMC vocabulary coverage; and (iii) Qwen3.5-9B fine-tuned with QLoRA as an instruction-tuned generative baseline. Across five fixed seeds, MelBERT MIP-only achieves the strongest performance at 0.7281 +/- 0.0050 test positive F1, marginally above MelBERT Full (0.7270 +/- 0.0069) and clearly above plain RoBERTa (0.7142 +/- 0.0121). The Qwen QLoRA generative configuration trails encoder baselines by approximately 11 F1 points (0.6157 +/- 0.0113). Three findings merit attention: (1) the SPV channel of MelBERT does not contribute reliable positive signal in Chinese, consistent with the dominance of conventional metaphor; (2) the Qwen-encoder gap is concentrated in recall, reflecting the discrete-commitment limitation of generative output; (3) several Qwen task formulations fail due to format design rather than model capacity. We release all split manifests, per-seed outputs, the MCD7 basic-meaning embedding pipeline, and training scripts to serve as a common reference for future Chinese metaphor identification research. 

---
# LaTER: Efficient Test-Time Reasoning via Latent Exploration and Explicit Verification 

**Authors**: Xuan Li, Yining Wang, Yuchen Liu, Guanjun Liu, Delai Qiu, Shengping Liu, Jiaen Liang, Wei Huang, Jun Yu, Junnan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07315)  

**Abstract**: Chain-of-thought (CoT) reasoning improves large language models (LLMs) on difficult tasks, but it also makes inference expensive because every intermediate step must be generated as a discrete token. Latent reasoning reduces visible token generation by propagating continuous states, yet replacing explicit derivations with latent computation can hurt tasks that require symbolic checking. We propose Latent-Then-Explicit Reasoning (LaTER), a two-stage paradigm that first performs bounded exploration in a continuous latent space and then switches to explicit CoT for verification and answer generation. In a training-free instantiation, LaTER projects final-layer hidden states back to the input embedding space, preserves the latent KV cache, and uses entropy and model-native stop-token probes to decide when to switch. We find that strong reasoning models already exhibit structured latent trajectories under this interface. On Qwen3-14B, training-free LaTER reduces total token usage by 16%-32% on several benchmarks while matching or improving accuracy on most of them; for example, it improves AIME 2025 from 70.0% to 73.3% while reducing tokens from 15,730 to 10,661. We further construct Latent-Switch-69K, a supervised corpus that pairs condensed solution intuitions with shortened explicit derivations. Fine-tuning with latent rollout and halting supervision yields additional gains: trained LaTER reaches 80.0% accuracy on AIME 2025, 10.0 points above the standard CoT baseline, while using 33% fewer tokens. Our code, data, and model are available at this https URL. 

---
# SpecBlock: Block-Iterative Speculative Decoding with Dynamic Tree Drafting 

**Authors**: Weijie Shi, Qiang Xu, Fan Deng, Yaguang Wu, Jiarun Liu, Yehong Xu, Hao Chen, Jia Zhu, Jiajie Xu, Xiangjun Huang, Jian Yang, Xiaofang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.07243)  

**Abstract**: Speculative decoding accelerates LLM inference by drafting a tree of candidate continuations and verifying it in one target forward. Existing drafters fall into two camps with opposite weaknesses. Autoregressive drafters such as EAGLE-3 preserve dependence along each draft path but call the drafter once per tree depth, making drafting a non-trivial share of per-iteration latency. Parallel drafters cut drafter calls by predicting multiple future positions in one forward, but each position is predicted without seeing the others, producing paths the verifier rejects. In this paper, we propose SpecBlock, a block-iterative drafter that combines path dependence with cheap drafting. Each drafter forward produces K dependent positions and we call this a block. The draft tree grows through repeated block expansions. Two mechanisms explicitly carry path dependence to keep later draft positions accurate. Within each block, a layer-wise shift carries the previous position's hidden state into every decoder layer. Across blocks, each new block can start from any position of the previous block, inheriting its hidden state to extend the path. To spend verifier budget where acceptance is likely, a co-trained rank head replaces the fixed top-k tree by allocating per-position branching during drafting. To avoid training the drafter on prefixes it never produces at inference, a valid-prefix mask drops the loss at later positions once an earlier one is wrong. Beyond static drafting, a cost-aware bandit at deployment uses free verifier feedback to update the drafter selectively, only when the expected throughput gain exceeds the update cost. Experiments show that SpecBlock improves mean speedup by 8-13% over EAGLE-3 at 44-52% of its drafting cost, and cost-aware adaptation extends this lead to 11-19%. 

---
# Rethinking Experience Utilization in Self-Evolving Language Model Agents 

**Authors**: Weixiang Zhao, Yingshuo Wang, Yichen Zhang, Yanyan Zhao, Yu Zhang, Yang Wu, Dandan Tu, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07164)  

**Abstract**: Self-evolving agents improve by accumulating and reusing experience from past interactions. Existing work has largely focused on how experience is constructed, represented, and updated, while paying less attention to how experience should be used during runtime decision-making. As a result, most agents rely on rigid usage strategies, either injecting experience once at initialization or at every step, without considering whether it is needed for the current decision. This paper studies experience utilization as a critical design dimension of self-evolving agents. We ask whether agents benefit from interweaving experience use with decision-making, so that experience is invoked only when additional guidance is needed. To examine this question, we introduce {ExpWeaver}, a lightweight instantiation that leaves experience construction unchanged and modifies only runtime utilization by exposing experience as an optional resource during reasoning. Across four representative frameworks, seven LLM backbones, and three types of environments, ExpWeaver consistently achieves the best performance among different utilization strategies. Reinforcement learning experiments further show that this behavior can be amplified through training. Usage-pattern, causal ablation, and entropy-based analyses reveal that ExpWeaver enables agents to invoke experience selectively, at beneficial decision points, and under higher reasoning uncertainty. Overall, our findings call for a shift from merely studying \emph{what} experience to store toward understanding \emph{how} and \emph{when} experience should enter decision-making. 

---
# Teaching Language Models to Think in Code 

**Authors**: Hyeon Hwang, Jiwoo Lee, Jaewoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07237)  

**Abstract**: Tool-integrated reasoning (TIR) has emerged as a dominant paradigm for mathematical problem solving in language models, combining natural language (NL) reasoning with code execution. However, this interleaved setup has three key limitations: code often acts as a post-hoc verifier, intermediate NL computations are error-prone, and NL and code play overlapping rather than clearly distinct roles. We propose ThinC (Thinking in Code), a framework in which code itself serves as the reasoner rather than as a tool invoked by NL. A ThinC trajectory begins with a brief NL planning step, after which all reasoning unfolds through code blocks connected only by their execution outputs. We distill 12.2k code-centric trajectories from a teacher model and train ThinC-1.7B and ThinC-4B with supervised fine-tuning followed by reinforcement learning. ThinC-4B consistently outperforms every TIR baseline on five competition-level math benchmarks and even surpasses the much larger Qwen3-235B-A22B-Thinking. Further analysis shows that ThinC reasons through code: 99.2% of its final answers are grounded in interpreter output, and the model recovers reliably from code execution failures without intermediate NL reasoning. Our code and models will be released soon. 

---
# Beyond Reasoning: Reinforcement Learning Unlocks Parametric Knowledge in LLMs 

**Authors**: Wanli Yang, Hongyu Zang, Junwei Zhang, Wenjie Shi, Du Su, Jingang Wang, Xueqi Cheng, Fei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.07153)  

**Abstract**: Reinforcement learning (RL) has achieved remarkable success in LLM reasoning, but whether it can also improve direct recall of parametric knowledge remains an open question. We study this question in a controlled zero-shot, one-hop, closed-book QA setting with no chain-of-thought, training only on binary correctness rewards and applying fact-level train-test deduplication to ensure gains reflect improved recall rather than reasoning or memorization. Across three model families and multiple factual QA benchmarks, RL yields ~27% average relative gains, surpassing both training- and inference-time baselines alike. Mechanistically, RL primarily redistributes probability mass over existing knowledge rather than acquiring new facts, moving correct answers from the low-probability tail into reliable greedy generations. Our data-attribution study reveals that the hardest examples are the most informative: those whose answers never appear in 128 pre-RL samples (only ~18% of training data) drive ~83% of the gain, since rare correct rollouts still emerge during training and get reinforced. Together, these findings broaden the role of RL beyond reasoning, repositioning it as a tool for unlocking rather than acquiring latent parametric knowledge. 

---
# CLIPer: Tailoring Diverse User Preference via Classifier-Guided Inference-Time Personalization 

**Authors**: Jinyan Su, Jinpeng Zhou, Claire Cardie, Wen Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.07162)  

**Abstract**: Personalized LLMs can significantly enhance user experiences by tailoring responses to preferences such as helpfulness, conciseness, and humor. However, fine-tuning models to address all possible combinations of user preferences is computationally expensive and impractical. In this paper, we introduce \textbf{CLIPer}(\textbf{Cl}assifier-guided \textbf{I}nference-time \textbf{Per}sonalization), a lightweight personalization approach that leverages a classifier model to steer LLM generation dynamically to different user preferences at inference time. Our method eliminates the need for extensive fine-tuning, inducing negligible additional computational overhead while enabling more controllable and nuanced personalization across single and multi-dimensional preferences. Comprehensive empirical analyses demonstrate the scalability and effectiveness of our approach in delivering personalized language generation. 

---
# Retrieve, Integrate, and Synthesize: Spatial-Semantic Grounded Latent Visual Reasoning 

**Authors**: Jin Cui, Xinyue Long, Xunyong Zhang, Yadong Zhang, Chuanchang Su, Jingye Gan, Boran Zhao, Pengju Ren  

**Link**: [PDF](https://arxiv.org/pdf/2605.07106)  

**Abstract**: Multimodal Large Language Models (MLLMs) have made remarkable progress on vision-language reasoning, yet most methods still compress visual evidence into discrete textual thoughts, creating an information bottleneck for fine-grained perception. Recent latent visual reasoning methods attempt to reason in continuous hidden states, but we find that they suffer from insufficient manifold compatibility: latent trajectories drift away from pretrained reasoning circuits, collapse into instance-agnostic patterns, and are often bypassed during answer generation. To address these issues, we propose RIS (Retrieve, Integrate, and Synthesize), a spatial-semantic grounded framework that develops latent reasoning as a compatible extension of pretrained MLLM computation. We first construct a step-wise grounded reasoning dataset with bounding boxes and region-specific semantic descriptions. Built on this supervision, RIS anchors latent tokens to both spatial and semantic evidence, enforces their causal role through a progressive attention bottleneck, and introduces short language transition tokens to bridge synthesized latent states back to vocabulary-aligned decoding. Experiments on V*, HRBench4K, HRBench8K, MMVP, and BLINK show consistent improvements over closed/open-source and latent reasoning baselines. Further analyses demonstrate that RIS learns diverse, interpretable, and progressively integrated latent trajectories, offering a practical path toward faithful internal visual reasoning in MLLMs. 

---
# SAGE: Hierarchical LLM-Based Literary Evaluation through Ontology-Grounded Interpretive Dimensions 

**Authors**: Tianyu Wang, Nianjun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.07102)  

**Abstract**: Evaluating literary quality requires assessing interpretive dimensions such as cultural representation, emotional depth, and philosophical sophistication that resist straightforward computational measurement. We introduce SAGE, a hierarchical evaluation framework that decomposes literary quality into ontology-grounded interpretive dimensions assessed through structured large language model evaluation with multi-round iterative reflection and independent validation. We validate the framework on 100 short stories (50 canonical works, 30 pulp fiction, 20 LLM-generated narratives) across three analytical layers (cultural, emotional-psychological, existential-philosophical) using dual-mode assessment. Across 600 evaluations, the framework achieves 98.8% score convergence and greater than 94% inter-rater agreement, with near-perfect mode invariance between content-based and metadata-based evaluation. Statistical analysis reveals a consistent genre hierarchy (Canonical > Pulp > LLM, all p<0.001) with layer-specific discrimination: cultural critique and philosophical depth exhibit very large effect sizes (Cohen's d>2.4), while emotional representation shows smaller gaps (d=1.68), suggesting that affective patterns are more learnable from training data than critical stance or philosophical depth. Cross-layer correlations (r=0.649-0.683) confirm the three dimensions capture empirically distinguishable quality facets. These findings demonstrate that theory-driven LLM evaluation can achieve measurement-grade reliability and support systematic identification of where current generative models fall short of human literary production, with direct implications for scalable automated evaluation of open-ended text generation. 

---
# NSMQ Riddles: A Benchmark of Scientific and Mathematical Riddles for Quizzing Large Language Models 

**Authors**: George Boateng, Naafi Ibrahim, Samuel John, Philemon Badu, Patrick Agyeman-Budu, Jonathan Mensah, Kevin Yeboah, William Edor, Andrew Mensa-Onumah, Nana Yeboah, Victor Wumbor-Apin Kumbol  

**Link**: [PDF](https://arxiv.org/pdf/2605.07051)  

**Abstract**: Large Language Models (LLMs) have shown good performance on various science educational benchmarks, demonstrating their potential for use in science and mathematics education. Yet, LLMs tend to be evaluated on science and mathematical educational datasets from the Western world, with an underrepresentation of datasets from the Global South. Furthermore, they tend to have multiple-choice answer options that are trivial to evaluate. In this work, we present NSMQ Riddles, a novel benchmark of Scientific and Mathematical Riddles from Ghana's National Science and Maths Quiz (NSMQ) competition to evaluate LLMs. The NSMQ is an annual live TV competition for senior secondary school students in Ghana that brings together the smartest high school students in Ghana who compete in teams of 2 by answering questions in biology, chemistry, physics, and math over five rounds and five stages until a winning team is crowned for that year. NSMQ Riddles consists of 11 years of riddle questions (n=1.8K) from the 5th round, with each riddle containing a minimum of 3 clues. Students compete to be the first to guess the answer on any of the clues, with earlier clues being vague and also fetching more points. The answers are usually a number, word, or short phrase, allowing for automatic evaluation. We evaluated state-of-the-art models: closed (GPT-5.4, Gemini 3.1 Pro, Claude Opus 4.6) and open models (Kimi-K2.5, DeepSeek-V3.1, GPT-OSS-120B) with high and low reasoning settings. Our evaluation shows that the dataset is challenging even for state-of-the-art LLMs, which performed worse than the best student contestants. This work contributes a novel and challenging benchmark for scientific and mathematical reasoning from the Global South towards enabling a true global benchmarking of LLMs' capabilities for science and mathematics education. 

---
# Self-Consolidating Language Models: Continual Knowledge Incorporation from Context 

**Authors**: Zekun Wang, Anant Gupta, Zihan Dong, Christopher J. MacLellan  

**Link**: [PDF](https://arxiv.org/pdf/2605.07076)  

**Abstract**: Large language models (LLMs) increasingly receive information as streams of passages, conversations, and long-context workflows. While longer context windows expose more evidence, they do not ensure that useful information is preserved and reused. We study continual context consolidation: writing current context into model weights while limiting interference with previously consolidated information. We propose \textbf{S}elf-\textbf{Co}nsolidating \textbf{L}anguage Models (SCoL), a post-training framework in which, given current context, an LLM learns to generate textual update instructions specifying which of its own Transformer layers should be updated. Because committed updates change the model that later generates future selections, we train SCoL with meta-reinforcement learning over an evolving model state. We instantiate SCoL with supervised QA rewards on SQuAD knowledge incorporation and intrinsic likelihood-based rewards for LongBench v2 long-context consolidation. Across both settings, SCoL improves acquisition and retention over prompting, summarization, batch test-time training, and sequential finetuning baselines. Analysis of learned selection patterns shows that SCoL encourages the LLM to generate sparse update locations that align with layers of high Fisher information, suggesting that the model learns to route plasticity toward loss-sensitive regions while limiting interference. Moreover, SCoL transfers from shorter meta-training streams to longer LongBench v2 streams at evaluation, suggesting that our framework supports scalable streaming consolidation. 

---
# Securing Computer-Use Agents: A Unified Architecture-Lifecycle Framework for Deployment-Grounded Reliability 

**Authors**: Zejian Chen, Zhanyuan Liu, Chaozhuo Li, Mengxiang Han, Songyang Liu, Litian Zhang, Feng Gao, Yiming Hei, Xi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07110)  

**Abstract**: Computer-use agents(CUAs)are moving frombounded benchmarks toward real software environments, wherethey operate browsers, desktops, mobile applications, flesystems,terminals, and tool backends. In such settings, reliability isno longer captured by task success alone: perception errors,planning drift, memory use, tool mediation, permission scope,and runtime oversight jointly determine whether agent actionsremain aligned with user intent, Existing surveys organize theCUA landscape by methods, platforms, benchmarks, or securitythreats, but less explicitly connect capability formation, author-ity exposure, failure manifestation, and control placement. Toaddress this gap, the article develops an architecture-lifecycleframework for deployment-grounded reliability in CUAs. Thearchitectural view analyzes Perception, Decision, and Executionas coupled layers that transform software observations intoauthority-bearing actions, The lifecycle view examines this http URL, Operation, and Maintenance as stages in which priorsare learned, tools and permissions are bound, runtime this http URL are stressed, and assurance must be preserved under this http URL this lens, the analysis synthesizes representative systems,benchmarks, and security/privacy studies; distinguishes wherefailures become visible from where their enabling conditions areintroduced, and maps recurring intervention surfaces for controloversight, and assurance. OpenClaw is used only as a public this http URL example of an open deployment pattern, not as a verifedinternal case study. The conclusion highlights open challengesin controllable grounding, long-horizon constraint preservation,safe authority binding, mixed-trust runtime defense, privacy-preserving memory,and continual assurance. 

---
# MultiSoc-4D: A Benchmark for Diagnosing Instruction-Induced Label Collapse in Closed-Set LLM Annotation of Bengali Social Media 

**Authors**: Souvik Pramanik, S.M. Riaz Rahman Antu, Shak Mohammad Abyad, Md. Ibrahim Khalil, Md. Shahriar Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2605.06940)  

**Abstract**: Annotation automation via Large Language Models (LLMs) is the core approach for scaling NLP datasets; however, LLM behavior with respect to closed-set instructions in low-resource languages has not been well studied. We present MultiSoc-4D, a Bengali social media dataset benchmark, which contains 58K+ social media comments from six sources annotated along four dimensions: category, sentiment, hate speech, and sarcasm. By employing a structured pipeline where ChatGPT, Gemini, Claude, and Grok individually annotate separate partitions, while sharing a common validation set of 20%, we diagnose LLM behavior systematically. We discover a prevalent phenomenon called "instruction-induced label collapse", wherein LLMs show a systematic preference towards fallback labels (Other, Neutral, No), leading to high agreement rates but under-detection of minority categories. For example, we find that LLMs failed to detect 79% and 75% of instances with hateful and sarcastic content compared to a human-calibrated reference. Furthermore, we prove that it represents a "label agreement illusion", statistically validated via almost null Fleiss' Kappa ($\kappa \approx -0.001$) on sarcasm detection. Across 40+ LLMs, we benchmark this annotation bias propagation within the training pipeline, regardless of architectural differences. We release MultiSoc-4D as a diagnostic benchmark for annotation biases in Bengali NLP. 

---
# Towards Closing the Autoregressive Gap in Language Modeling via Entropy-Gated Continuous Bitstream Diffusion 

**Authors**: Georgios Batzolis, Mark Girolami, Luca Ambrogioni  

**Link**: [PDF](https://arxiv.org/pdf/2605.07013)  

**Abstract**: Diffusion language models (DLMs) promise parallel, order-agnostic generation, but on standard benchmarks they have historically lagged behind autoregressive models in sample quality and diversity. Recent continuous flow and diffusion approaches over token embeddings have narrowed this gap, suggesting continuous state spaces are highly effective for language. In this work, we further close the autoregressive gap by modeling text as a continuous diffusion process over fixed-width binary bitstreams. Our approach represents semantic tokens as analog bit sequences and utilizes a matched-filter residual parameterization to isolate contextual learning from analytic independent-bit posteriors. Crucially, we adopt a stochastic sampler that applies Langevin-type corrections gated by the entropy-rate profile, automatically concentrating stochasticity in high-information regions while remaining nearly deterministic elsewhere. On the One Billion Word Benchmark (LM1B), our 130M-parameter bitstream model reaches a generative perplexity ($\GenPPL$) of $59.76$ at matched real-data entropy ($4.31$) using 256 neural function evaluations (NFEs), decisively outperforming prior DLM baselines and reaching the autoregressive reference. On OpenWebText (OWT), our stochastic sampler establishes a new continuous-DLM Pareto frontier, achieving $\GenPPL=27.06$ at an entropy of $5.26$ using $4\times$ fewer steps than previous 1024-NFE baselines. As an additional architectural benefit, bitstream diffusion removes the $\mathcal{O}(V)$ vocabulary scaling bottleneck shared by standard DLMs. By predicting $\mathcal{O}(\log V)$ bitwise logits via semantic bit-patching, our model yields a reduced memory footprint and higher throughput, demonstrating a scalable paradigm for language generation as vocabulary sizes grow. 

---
# Reflections and New Directions for Human-Centered Large Language Models 

**Authors**: Caleb Ziems, Dora Zhao, Rose E. Wang, Matthew Jörke, Ahmad Rushdi, Advit Deepak, Sunny Yu, Anshika Agarwal, Harshvardhan Agarwal, Gabriela Aranguiz-Dias, Aditri Bhagirath, Justine Breuch, Huanxing Chen, Ruishi Chen, Sarah Chen, Haocheng Fan, William Fang, Cat Gonzales Fergesen, Daniel Frees, Tian Gao, Ziqing Huang, Vishal Jain, Yucheng Jiang, Kirill Kalinin, Su Doga Karaca, Arpandeep Khatua, Teland La, Isabelle Levent, Miranda Li, Xinling Li, Yongce Li, Angela Liu, Minsik Oh, Nathan J. Paek, Anthony Qin, Emily Redmond, Michael J. Ryan, Aadesh Salecha, Xiaoxian Shen, Pranava Singhal, Shashanka Subrahmanya, Mei Tan, Irawadee Thawornbut, Michelle Vinocour, Xiaoyue Wang, Zheng Wang, Henry Jin Weng, Pawan Wirawarn, Shirley Wu, Sophie Wu, Yichen Xie, Patrick Ye, Sean Zhang, Yutong Zhang, Cathy Zhou, Yiling Zhao, James Landay, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06901)  

**Abstract**: Large Language Models (LLMs) are increasingly shaping the private and professional lives of users, with numerous applications in business, education, finance, healthcare, law, and science. With this rise in global influence comes greater urgency to build, evaluate, and deploy these systems in a manner that prioritizes not only technical capabilities but also human priorities. This work presents a framework for developing Human-Centered Large Language Models (HCLLMs), which integrates perspectives from Natural Language Processing (NLP), Human-Computer Interaction (HCI), and responsible AI. Considering the ethics, economics, and technical objectives of language modeling, we argue that model developers need to address human concerns, preferences, values, and goals, not only during a cursory post-training stage, but rather with rigor and care at every stage of the pipeline. This paper offers human-centered insights and recommendations for developers at each stage, from system design to data sourcing, model training, evaluation, and responsible deployment. Then we conclude with a case study, applying these insights to understand the future of work with HCLLMs. 

---
# Group of Skills: Group-Structured Skill Retrieval for Agent Skill Libraries 

**Authors**: Kun Zeng, Yu Huo, Siyu Zhang, Zi Ye, Yuecheng Zhuo, Haoyue Liu, Yuquan Lu, Junhao Wen, Xiaoying Tang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06978)  

**Abstract**: Skill-augmented agents increasingly rely on large reusable skill libraries, but retrieving relevant skills is not the same as presenting usable context. Existing methods typically return atomic skills or dependency-aware bundles whose internal roles remain implicit, leaving the agent to infer the execution entry point, support skills, visible requirements, and failure-avoidance guidance. We introduce Group of Skills (GoSkills), an inference-time group-structured retrieval method that changes the agent-facing retrieval object from a flat skill list to a compact, role-labeled execution context. GoSkills builds anchor-centered skill groups from a typed skill graph, expands support groups through a group graph, bottlenecks the selected group plan into a bounded set of atomic skill payloads, and renders a fixed execution contract with Start, Support, Check, and Avoid fields, without changing the downstream agent, skill payloads, or execution environment. Experiments on SkillsBench and ALFWorld show that GoSkills preserves visible-requirement coverage under a small skill budget, improves over flat skill-access baselines, and often improves reward and agent-only runtime relative to structural retrieval references. 

---
# Can LLMs Take Retrieved Information with a Grain of Salt? 

**Authors**: Behzad Shayegh, Mohamed Osama Ahmed, Fred Tung, Leo Feng  

**Link**: [PDF](https://arxiv.org/pdf/2605.06919)  

**Abstract**: Large language models have demonstrated impressive retrieval-augmented capabilities. However, a crucial area remains underexplored: their ability to appropriately adapt responses to the certainty of the retrieved information. It is a limitation with real consequences in high-stakes domains like medicine and finance. We evaluate eight LLMs on their context-certainty obedience, measuring how well they adjust responses to match expressed context certainty. Our analysis reveals systematic limitations: LLMs struggle to recall prior knowledge after observing an uncertain context, misinterpret expressed certainties, and overtrust complex contexts. To address these, we propose an interaction strategy combining prior reminders, certainty recalibration, and context simplification. This approach reduces obedience errors by 25% on average, without modifying model weights, demonstrating the efficacy of interaction design in enhancing LLM reliability. Our contributions include a principled evaluation metric, empirical insights into LLMs' uncertainty handling, and a portable strategy to improve context-certainty obedience across diverse LLMs. 

---
# OrScale: Orthogonalised Optimization with Layer-Wise Trust-Ratio Scaling 

**Authors**: Yuxuan Lou, Yang You  

**Link**: [PDF](https://arxiv.org/pdf/2605.07815)  

**Abstract**: Muon improves neural-network training by orthogonalizing matrix-valued updates, but it leaves each layer's update magnitude controlled mostly by a global learning rate. We introduce OrScale, a trust-ratio extension of Muon built on a simple rule: the denominator of a layer-wise ratio should measure the Frobenius norm of the actual parameter-space direction that will be applied. This yields OrScale for general matrix layers and OrScale-LM for language models, where Moonlight shape scaling is combined with one-time per-layer calibration so every trust ratio starts at one. We analyze why three natural Muon-LAMB hybrids fail through shape-degenerate denominators, raw-momentum clip saturation, and decoupled weight-decay runaway, and show that the real-update-direction denominator with coupled weight decay avoids these failures. Theoretically, OrScale admits an O(1/sqrt(T)) nonconvex convergence guarantee in a nuclear-norm criterion, a strict layer-adaptive descent gain under measurable layer heterogeneity, and calibration properties that preserve muP-style learning-rate transfer at initialization. Empirically, OrScale ranks first on CIFAR-10/DavidNet across three seeds, improving Muon from 93.70% to 94.05% validation top-1, and OrScale-LM improves FineWeb-Edu pre-training versus Muon+Moonlight at three of four scales from 125M to 1.1B parameters while outperforming AdamW at every scale. 

---
# Tracing the Arrow of Time: Diagnosing Temporal Information Flow in Video-LLMs 

**Authors**: Peitao Han, Fei Cheng, Lis K. Pereira, Qianying Liu, Shigeru Kitazawa  

**Link**: [PDF](https://arxiv.org/pdf/2605.07568)  

**Abstract**: The Arrow-of-Time (AoT) task, determining whether a video plays forward or backward by recognizing temporal irreversibility, is one humans solve with near-perfect accuracy, yet frontier Video Large Language Models (Video-LLMs) perform only modestly above chance. This gap raises a key question: do visual backbones fail to encode temporal information, or does information bottleneck lie elsewhere in the Video-LLM architecture? We address this question by isolating the vision encoder from the Video-LLM and tracing temporal information across the encoder, projector, and LLM. We find that video-centric encoders with explicit temporal modeling encode strong temporal signals, whereas frame-centric encoders do not. However, when video-centric representations are passed through a standard Video-LLM architecture, performance often collapses, revealing a bottleneck of temporal information flow. We identify projector design as a key factor: Q-Former disrupts temporal information, while a time-preserved MLP projection substantially improves the LLM's access to such information. Our layer-wise analysis further shows temporal representation dynamics across encoder layers. Guided by these findings, we build a Video-LLM with temporal-aware video-centric encoder, time-preserved projector, and AoT supervision, surpassing human performance on AoT$_{PPB}$ with 98.1\% accuracy, and improving broader temporal reasoning tasks by up to 6.0 points on VITATECS-Direction and 1.3 points on TVBench. Our results show that temporal reasoning in Video-LLMs requires both effective temporal encoding and reliable transfer of this information to the LLM. 

---
# Reliable Chain-of-Thought via Prefix Consistency 

**Authors**: Naoto Iwase, Yuki Ichihara, Mohammad Atif Quamar, Junpei Komiyama  

**Link**: [PDF](https://arxiv.org/pdf/2605.07654)  

**Abstract**: Large Language Models often improve accuracy on reasoning tasks by sampling multiple Chain-of-Thought (CoT) traces and aggregating them with majority voting (MV), a test-time technique called self-consistency. When we truncate a CoT partway through and regenerate the remainder, we observe that traces with correct answers reproduce their original answer more often than traces with wrong answers. We use this difference as a reliability signal, prefix consistency, that weights each candidate answer by how often it reappears under regeneration. It requires no access to token log-probabilities or self-rating prompts. Across five reasoning models and four math and science benchmarks, prefix consistency is the best correctness predictor in most settings, and reweighting votes by it reaches Standard MV plateau accuracy at up to 21x fewer tokens (median 4.6x). Our code is available at this https URL. 

---
# When Are Experts Misrouted? Counterfactual Routing Analysis in Mixture-of-Experts Language Models 

**Authors**: Youngsik Yoon, Siwei Wang, Wei Chen, Jungseul Ok  

**Link**: [PDF](https://arxiv.org/pdf/2605.07260)  

**Abstract**: Mixture-of-Experts (MoE) language models route each token to a small subset of experts, but whether the routes selected by a trained top-$k$ router are good ones is rarely evaluated directly. Holding the model fixed, we compare each standard route against sampled equal-compute alternatives for the same token and score each by the next-token probability it assigns to the realized token in a verified reasoning trajectory. The result is sharply token-conditional: the standard router is well-aligned with route utility on confident tokens but uninformative on the fragile tokens that drive hard reasoning, where lower-loss equal-compute routes consistently exist inside the frozen model but are not selected. The same pattern holds across Qwen3-30B-A3B, GPT-OSS-20B, DeepSeek-V2-Lite, and OLMoE-1B-7B, and follows structurally from how standard top-$k$ training evaluates routing decisions: the language modeling loss scores only the executed route, and load balancing depends only on aggregate routing statistics. A minimal router-only update to the final-layer router, leaving every expert and every other router frozen, is sufficient to shift pass@K on AIME 2024+2025 and HMMT 2025 for both Qwen3-30B-A3B and GPT-OSS-20B, suggesting that at least part of the failure reflects router-reachable misallocation rather than expert capacity alone. 

---
# DiffRetriever: Parallel Representative Tokens for Retrieval with Diffusion Language Models 

**Authors**: Shuai Wang, Yin Yu, Shengyao Zhuang, Bevan Koopman, Guido Zuccon  

**Link**: [PDF](https://arxiv.org/pdf/2605.07210)  

**Abstract**: PromptReps showed that an autoregressive language model can be used directly as a retriever by prompting it to generate dense and sparse representations of a query or passage. Extending this to multiple representatives is inefficient for autoregressive models, since tokens must be generated sequentially, and prior multi-token variants did not reliably improve over single-token decoding.
We show that the bottleneck is sequential generation, not the multi-token idea itself. DiffRetriever is a representative-token retriever for diffusion language models: it appends K masked positions to the prompt and reads all K in a single bidirectional forward pass. Across in-domain and out-of-domain evaluation, multi-token DiffRetriever substantially improves over single-token on every diffusion backbone we test, while autoregressive multi-token is flat or negative and pays a latency cost that scales with K where diffusion does not. After supervised fine-tuning, DiffRetriever on Dream is the strongest BEIR-7 retriever in our comparison, ahead of PromptReps, the encoder-style DiffEmbed baseline on the same diffusion backbones, and the contrastively fine-tuned single-vector RepLLaMA. A per-query oracle on the frozen base model exceeds contrastive fine-tuning at the same fixed budget, pointing to adaptive budget selection as future work. Code is available at this https URL. 

---
# Topic Is Not Agenda: A Citation-Community Audit of Text Embeddings 

**Authors**: Junseon Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2605.07158)  

**Abstract**: Vector search and retrieval-augmented generation (RAG) rest on the assumption that cosine similarity between text embeddings reflects conceptual relatedness. We measure where this assumption breaks. We build an augmented citation graph over 3.58M scientific papers and partition it via Leiden CPM at two granularities: sub-field (L1) and research-agenda (L2, hierarchical inside each L1). Four state-of-the-art embeddings (Gemini, Qwen3-8B, Qwen3-0.6B, SPECTER2) clear the L1 bar reasonably (45-52% top-10 same-rate) but stop working at L2: only 15-21% of top-10 neighbors share the query's research agenda. In absolute terms, 8 of every 10 retrieved papers are off-agenda. The failure is universal across eight scientific domains and all four models; SPECTER2, despite its citation-based contrastive training, is the weakest. As a diagnostic probe, we test whether the same augmented graph also functions as a retrieval signal: a deliberately simple citation-count rerank reaches 57.7% top-1 L2 on top of LLM-expanded Boolean retrieval and 59.6% on top of plain BM25, on 80 curated agenda queries -- about 9 points above the best cosine retriever (Gemini, 50.6%) and 20 points above BM25 alone (39.3%). The probe isolates a slice of the agenda-matching signal the graph carries but the embeddings miss, connecting recent theoretical limits on single-vector retrieval to a concrete failure mode of scientific RAG. 

---
# The Position Curse: LLMs Struggle to Locate the Last Few Items in a List 

**Authors**: Zhanqi Zhang, Hua-Dong Xiong, Robert C. Wilson, Mikio Aoi, Marcelo G. Mattar, Li Ji-An  

**Link**: [PDF](https://arxiv.org/pdf/2605.07127)  

**Abstract**: Modern large language models (LLMs) can find a needle in a haystack (locating a single relevant fact buried among hundreds of thousands of irrelevant tokens) with near-saturated accuracy, yet fail to retrieve the last few items in a short list. We call this failure the Position Curse. For instance, even in a two-line code snippet, Claude Opus 4.6 misidentifies the second-to-last line most of the time. To characterize this failure, we evaluated two complementary queries: given a position in a sequence (of letters or words), retrieve the corresponding item; and given an item, return its position. Each position is specified as a forward or backward offset from an anchor, either an endpoint of the list (its start or end) or another item in the list. Across both open-source and frontier closed-source models, backward retrieval substantially lags forward retrieval. To test whether this capability can be rescued by post-training, we constructed PosBench, a position-focused training dataset. LoRA fine-tuning improves both forward and backward retrieval and generalizes to a held-out code-understanding benchmark (PyIndex), yet absolute performance remains far from saturated. As LLM coding agents increasingly operate over large codebases where precise indexing becomes essential for code understanding and editing, position-based retrieval emerges as a key capability for future pretraining objectives and model design. 

---
# Theoretical Limits of Language Model Alignment 

**Authors**: Lucas Monteiro Paes, Natalie Mackraz, Barry-John Theobald, Federico Danieli  

**Link**: [PDF](https://arxiv.org/pdf/2605.07105)  

**Abstract**: Language model (LM) alignment improves model outputs to reflect human preferences while preserving the capabilities of the base model. The most common alignment approaches are (i) reinforcement learning, which maximizes the expected reward under a KL-divergence constraint, and (ii) best-of-$N$ alignment, which selects the highest-reward output among $N$ independent samples. Despite their widespread use, the fundamental limits of reward improvement under a KL budget remain poorly understood. We characterize the information-theoretic limits of KL-regularized alignment by deriving the maximum achievable expected reward gain for a fixed KL-divergence budget. Our first result provides a closed-form expression for the optimal reward improvement, governed by a Jeffreys divergence term rather than the $\sqrt{\texttt{KL}}$ used in prior analyses. We further reformulate this expression as a covariance under the base model, yielding a practical estimator that predicts achievable alignment gains from base model samples alone. We extend our analysis to the proxy reward setting, showing that the gap between ideal and proxy alignment (reward hacking) grows with the magnitude of reward error and when the KL penalty factor decreases. We then prove that reward ensembling mitigates reward hacking, providing a theoretical justification for this technique used in practice. Empirically, we compute the KL-reward Pareto frontier for two tasks for LMs, safety and summarization, and show that best-of-$N$ closely approaches the theoretical limit, while PPO and GRPO remain substantially suboptimal. Our theoretical results shed light on several empirically observed phenomena in the alignment literature and suggest that algorithmic improvements are needed to achieve optimal alignment without high inference costs. 

---
# Bridging Textual Profiles and Latent User Embeddings for Personalization 

**Authors**: Zhaoxuan Tan, Xiang Zhai, Yan Zhu, Meng Jiang, Mohamed Hammad  

**Link**: [PDF](https://arxiv.org/pdf/2605.06981)  

**Abstract**: Personalized systems rely on user representations to connect behavioral history with downstream recommendation applications. Existing methods typically employ either supervised latent user embeddings, which are effective for retrieval but difficult to interpret, or textual user profiles, which are interpretable but challenging to optimize for downstream utility due to lack of direct supervision. To bridge this gap, we present BLUE, a reinforcement learning framework that unifies these two forms of user representation by aligning language-based user profiles with embedding-based recommendation objectives. Given a user interaction history, BLUE leverages a profiler Large Language Model (LLM) to generate textual profiles, while an embedding model provides reward signals. This encourages the resulting textual representations to move closer to positive items and farther from negative ones in the embedding space. We further introduce a text-space supervision signal based on next-item prediction, ensuring the learned profiles remain both semantically meaningful and highly effective for downstream retrieval. Experiments on Amazon Reviews 2023 and Google Local Reviews in zero-shot sequential recommendation settings demonstrate that BLUE consistently outperforms strong baselines under both frozen and trainable embedding conditions. Notably, BLUE achieves clear gains in cross-domain transfer, highlighting the strong generalization ability of the learned user profiles. Furthermore, these generated profiles provide superior personalized context for question answering compared to raw user histories or alternative profile optimization methods. Overall, these results show that BLUE provides an effective way to unify interpretable textual profiling with discriminative latent embeddings for personalization. 

---
# Benchmarked Yet Not Measured -- Generative AI Should be Evaluated Against Real-World Utility 

**Authors**: Ishani Mondal, Shweta Bhardwaj  

**Link**: [PDF](https://arxiv.org/pdf/2605.06856)  

**Abstract**: Generative AI systems achieve impressive performance on standard benchmarks yet fail to deliver real-world utility, a disconnect we identify across 28 deployment cases spanning education, healthcare, software engineering, and law. We argue that this benchmark utility gap arises from three recurring failures in evaluation practice: proxy displacement, temporal collapse, and distributional concealment. Motivated by these observations, we argue that generative AI evaluation requires a paradigm shift from static benchmark-centered transparency toward stakeholder, goal, and context-conditioned utility transparency grounded in human outcome trajectories. Existing evaluations primarily characterize properties of model outputs, while deployment success depends on whether interaction with AI improves stakeholders' ability to achieve their goals over time. The missing construct is therefore utility: the change in a stakeholder's capability induced through sustained interaction with an AI system within a deployment context. To operationalize this perspective, we propose SCU-GenEval, a four-stage evaluation framework consisting of stakeholder-goal mapping, construct-indicator specification, mechanism modeling, and longitudinal utility measurement. To make these stages practically deployable, we introduce three supporting instruments: structured deployment protocols, context-conditioned user simulators, and persona- and goal-conditioned proxy metrics. We conclude with domain-specific calls to action, arguing that progress in generative AI must be evaluated through measurable improvements in human outcomes rather than benchmark performance alone. 

---
# ExpThink: Experience-Guided Reinforcement Learning for Adaptive Chain-of-Thought Compression 

**Authors**: Tingcheng Bian, Yuzhe Zhang, Jing Jin, Jinchang Luo, MingQuan Cheng, Haiwei Wang, Wenyuan Jiang, Miaohui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07501)  

**Abstract**: Large reasoning models (LRMs) achieve strong performance via extended chain-of-thought (CoT) reasoning, yet suffer from excessive token consumption and high inference latency. Existing reinforcement learning (RL) approaches for CoT compression rely on uniform, static length penalties that neglect model capability dynamics and problem-level difficulty variation. We propose \textbf{ExpThink}\xspace, an RL framework that addresses both dimensions through two complementary mechanisms. First, \emph{experience-guided reward shaping} tracks the shortest correct solution found so far for each problem and applies a three-tier reward: full credit for concise correct responses, discounted credit for verbose correct ones, and zero for incorrect ones. The threshold tightens automatically with model improvement, forming a self-evolving curriculum that requires no manual scheduling. Second, \emph{difficulty-adaptive advantage} replaces standard deviation normalization with correct-count normalization, yielding monotonically difficulty-scaled gradients that amplify learning on hard problems to preserve accuracy while suppressing gradients on easy ones to encourage brevity. Together, these mechanisms enforce an accuracy-first, compression-second training objective. Experiments on multiple mathematical reasoning benchmarks demonstrate that \textbf{ExpThink}\xspace reduces average response length by up to 77\% while simultaneously improving accuracy, achieving up to $3\times$ higher accuracy-efficiency ratio (accuracy divided by average token count) than the vanilla baseline and outperforming existing RL-based compression methods on both metrics. 

---
# SmellBench: Evaluating LLM Agents on Architectural Code Smell Repair 

**Authors**: Ion George Dinu, Marian Cristian Mihăescu, Traian Rebedea  

**Link**: [PDF](https://arxiv.org/pdf/2605.07001)  

**Abstract**: Architectural code smells erode software maintainability and are costly to repair manually, yet unlike localized bugs, they require cross-module reasoning about design intent that challenges both developers and automated tools. While large language model agents excel at bug fixing and code-level refactoring, their ability to repair architectural code smells remains unexplored. We present the first empirical evaluation of LLM agents on architectural code smell repair. We contribute SmellBench, a task orchestration framework that incorporates smell-type-specific optimized prompts and supports iterative multi-step execution, together with a scoring methodology that separately evaluates repair effectiveness, false positive identification, and net codebase impact. We evaluate 11 agent configurations from four model families (GPT, Claude, Gemini, Mistral) on 65 hard-severity architectural smells detected by PyExamine in the Python project scikit-learn, validated against expert judgments. Expert validation reveals that 63.1% of detected smells are false positives, while the best agent achieves a 47.7% resolution rate. Agents identify false positives with up to $\kappa = 0.94$ expert agreement, but repair aggressiveness and net codebase quality are inversely related: the most aggressive agent introduces 140 new smells. These findings expose a gap between current LLM capabilities in localized code transformations and the architectural understanding needed for cross-module refactoring. SmellBench provides reusable infrastructure for tracking progress on this underexplored dimension of automated software engineering. We release our code and data at this https URL. 

---
# When Routine Chats Turn Toxic: Unintended Long-Term State Poisoning in Personalized Agents 

**Authors**: Xiaoyu Xu, Minxin Du, Qipeng Xie, Haobin Ke, Qingqing Ye, Haibo Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.06731)  

**Abstract**: Personalized LLM agents maintain persistent cross-session state to support long-horizon collaboration. Yet, this persistence introduces a subtle but critical security vulnerability: routine user-agent interactions can gradually reshape an agent's long-term state, inadvertently weakening future confirmation boundaries, expanding tool-use defaults, and escalating autonomous behavior over time. We formalize this risk as \textbf{unintended long-term state poisoning}. To systematically study it, we introduce the \textbf{Unintended Long-Term State Poisoning Bench (ULSPB)}, a bilingual benchmark comprising $350$ settings spanning five assistance categories, seven interaction patterns, 24-turn routine interactions, and matched single-injection counterparts. Furthermore, we define the \emph{Harm Score} (HS), a state-centric metric that quantifies \emph{authorization drift}, \emph{tool-use escalation}, and \emph{unchecked autonomy}. Experiments on OpenClaw with four backbone LLMs demonstrate that, while single-injection is generally effective, routine conversations alone can substantially poison long-term state, primarily corrupting memory-centric artifacts. Evaluations seeded with real-world user interactions confirm that this risk is not a mere artifact of synthetic prompts. To mitigate this threat, we propose \textbf{StateGuard}, a lightweight, post-execution defense that audits state diffs at the writeback boundary and selectively rolls back dangerous edits. Across all evaluated models, StateGuard reduces HS to near zero and lowers false-negative rates, with acceptable high false-positive rates under a safety-first writeback defense and minimal overhead. 

---
# LKV: End-to-End Learning of Head-wise Budgets and Token Selection for LLM KV Cache Eviction 

**Authors**: Enshuai Zhou, Yifan Hao, Chao Wang, Rui Zhang, Di Huang, Jiaming Guo, Xing Hu, Zidong Du, Qi Guo, Yunji Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.06676)  

**Abstract**: Long-context inference in Large Language Models (LLMs) is bottlenecked by the linear growth of Key-Value (KV) cache memory. Existing KV cache compression paradigms are fundamentally limited by heuristics: heuristic budgeting relies on statistical priors rather than task objectives, causing resource misallocation, while heuristic selection relies on coupled query-key interactions or static inductive biases (e.g., attention sinks). To address this limitation, we introduce LKV (Learned KV Eviction), which formulates KV compression as an end-to-end differentiable optimization problem. LKV integrates LKV-H to learn task-optimized global budgets, and LKV-T to derive intrinsic KV importance without materializing attention matrices. This design bypasses heuristic proxies, strictly aligning compression with task objectives. Extensive evaluations demonstrate that LKV achieves state-of-the-art performance on both LongBench and RULER benchmarks at high compression rates. In particular, on LongBench, LKV achieves near-lossless performance with only 15\% KV cache retention. Crucially, our analysis identifies learned budgeting as the dominant driver of fidelity, demonstrating that data-driven allocation is essential to overcome the limitations of hand-crafted heuristics. 

---
# RateQuant: Optimal Mixed-Precision KV Cache Quantization via Rate-Distortion Theory 

**Authors**: Fei Zuo, Zikang Zhou, Hao Cong, Xiaoyan Xi, Ho Fai Leung  

**Link**: [PDF](https://arxiv.org/pdf/2605.06675)  

**Abstract**: Large language models cache all previously computed key-value (KV) pairs during generation, and this KV cache grows linearly with sequence length, making it a primary memory bottleneck for serving. Quantizing the KV cache to fewer bits reduces this cost, yet all current quantizers assign the same bit-width to every attention head, ignoring the large variation in head importance. A natural idea is to allocate more bits to important heads and fewer to the rest. We show, however, that such mixed-precision allocation has a hidden pitfall: each quantizer follows a different distortion curve D(b)=alpha*beta^{-b}, and the decay rate beta varies from 3.6 to 5.3 across quantizer designs. Applying one quantizer's distortion model to another inverts the allocation order and makes performance worse than uniform quantization. We call this failure mode distortion model mismatch and propose RateQuant to resolve it. RateQuant fits a per-quantizer distortion model from a small calibration set, then solves the resulting bit-allocation problem in closed form via reverse waterfilling from rate-distortion theory. On Qwen3-8B at 2.5 average bits, calibrated RateQuant reduces KIVI's perplexity from 49.3 to 14.9 (70% reduction) and improves QuaRot by 6.6 PPL. The entire calibration takes 1.6 s on a single GPU and adds zero overhead at inference time. 

---
# A Comprehensive Survey on Agent Skills: Taxonomy, Techniques, and Applications 

**Authors**: Yingli Zhou, Wang Shu, Yaodong Su, Wenchuan Du, Yixiang Fang, Xuemin Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.07358)  

**Abstract**: Large language model (LLM)-based agents that reason, plan, and act through tools, memory, and structured interaction are emerging as a promising paradigm for automating complex workflows. Recent systems such as OpenClaw and Claude Code exemplify a broader shift from passive response generation to action-oriented task execution. Yet as agents move toward open-ended, real-world deployment, relying on from-scratch reasoning and low-level tool calls for every task become increasingly inefficient, error-prone, and hard to maintain. This survey examines this challenge through the lens of \emph{agent skills}, which we define as reusable procedural artifacts that coordinate tools, memory, and runtime context under task-specific constraints. Under this view, agents and skills play complementary roles: agents handle high-level reasoning and planning, while skills form the operational layer that enables reliable, reusable, and composable execution. Skills are therefore central to the scalability, robustness, and maintainability of modern agent systems. We organize the literature around four stages of the agent skill lifecycle -- representation, acquisition, retrieval, and evolution -- and review representative methods, ecosystem resources, and application settings across each stage. We conclude by discussing open challenges in quality control, interoperability, safe updating, and long-term capability management. All related resources, including research papers, open-source data, and projects, are collected for the community in \textcolor{blue}{this https URL}. 

---
# FAVOR: Efficient Filter-Agnostic Vector ANNS Based on Selectivity-Aware Exclusion Distances 

**Authors**: Junjie Song, Yu Liu, Guoyu Hu, Zhongle Xie, Ming Yang, Beng Chin Ooi, Ke Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.07770)  

**Abstract**: Modern retrieval systems increasingly require integrating approximate nearest neighbor search (ANNS) with complex attribute filtering to handle hybrid queries in applications such as recommendation systems and retrieval-augmented generation (RAG). While HNSW-based inline-filtering methods show promise, existing approaches struggle to deliver high throughput under low-selectivity scenarios while balancing search efficiency, filtering generality, and index connectivity. To address these challenges, we propose FAVOR, an efficient filter-agnostic vector ANNS that supports arbitrary filtering conditions while maintaining stable performance across varying selectivity levels. FAVOR introduces three novel features: (1) an integrated architecture that unifies selectivity estimation and filtered ANNS execution, providing a cohesive solution for hybrid vector-attribute queries; (2) a HNSW-based inline-filtering algorithm that introduces an exclusion distance mechanism to dynamically reshape the vector distance distribution, pushing non-target vectors away from the query while promoting valid candidates toward the query, thus improving search efficiency without compromising generality or graph connectivity; and (3) a selectivity-driven search selector that estimates query selectivity and dynamically routes queries between a pre-filtering brute-force algorithm for low-selectivity cases and an optimized HNSW-based search algorithm for other scenarios, ensuring consistent performance. Extensive experiments on real-world datasets demonstrate that FAVOR achieves a 1.3-5$\times$ higher QPS at $Recall@10 = 95\%$ compared to state-of-the-art methods for arbitrary filtering conditions, while maintaining competitive performance even against tailored solutions in some filtering conditions. 

---
# InterLV-Search: Benchmarking Interleaved Multimodal Agentic Search 

**Authors**: Bohan Hou, Jiuning Gu, Jiayan Guo, Ronghao Dang, Sicong Leng, Xin Li, Xuemeng Song, Jianfei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07510)  

**Abstract**: Existing benchmarks for multimodal agentic search evaluate multimodal search and visual browsing, but visual evidence is either confined to the input or treated as an answer endpoint rather than part of an interleaved search trajectory. We introduce \textbf{InterLV-Search}, a benchmark for Interleaved Language-Vision Agentic Search, in which textual and visual evidence is repeatedly used to condition later search. It contains 2,061 examples across three levels: active visual evidence seeking, controlled offline interleaved multimodal search, and open-web interleaved multimodal search. Beyond existing benchmarks, it also includes multimodal multi-branch samples that involve comparison between multiple entities during the evidence search. We construct Level 1 and Level 2 with automated pipelines and Level 3 with a machine-led, human-supervised open-web pipeline. We further provide InterLV-Agent for standardized tool use, trajectory logging, and evaluation. Experiments on proprietary and open-source multimodal agents show that current systems remain far from solving interleaved multimodal search, with the best model below 50% overall accuracy, highlighting challenges in visual evidence seeking, search control, and multimodal evidence integration. We release the benchmark data and evaluation code at this https URL 

---
