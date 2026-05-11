# VecCISC: Improving Confidence-Informed Self-Consistency with Reasoning Trace Clustering and Candidate Answer Selection 

**Authors**: James Petullo, Sonny George, Dylan Cashman, Nianwen Xue  

**Link**: [PDF](https://arxiv.org/pdf/2605.08070)  

**Abstract**: A standard technique for scaling inference-time reasoning is Self-Consistency, whereby multiple candidate answers are sampled from an LLM and the most common answer is selected. More recently, it has been shown that weighted majority voting (e.g. Confidence-Informed Self Consistency (CISC)), which assigns a confidence value to each candidate answer and chooses the answer with the largest accumulated score, tends to be more accurate on a wide range of popular benchmarks. In practice, weighted majority voting necessitates calling a critic LLM on each candidate's reasoning trace to produce the answer's confidence score. This secondary series of LLM calls greatly increases the overhead and cost of weighted majority voting, despite its potential performance benefits. To reduce this expense, we propose VecCISC, a lightweight, adaptive framework that uses a measure of semantic similarity to filter reasoning traces that are semantically equivalent to others, degenerate, or hallucinated, thus decreasing the number of candidate answers that must be evaluated by the critic. To ensure adequate experimental thoroughness, we evaluate VecCISC on five challenging, widely-adopted datasets spanning the domains of mathematics, chemistry, biology, commonsense reasoning, and the humanities. Our results demonstrate that VecCISC reduces the total token usage by 47%, while maintaining or exceeding the accuracy of CISC. 

---
# Rubric-Grounded RL: Structured Judge Rewards for Generalizable Reasoning 

**Authors**: Manish Bhattarai, Ismael Boureima, Nishath Rajiv Ranasinghe, Scott Pakin, Dan O'Malley  

**Link**: [PDF](https://arxiv.org/pdf/2605.08061)  

**Abstract**: We argue that decomposing reward into weighted, verifiable criteria and using an LLM judge to score them provides a partial-credit optimization signal: instead of a binary outcome or a single holistic score, each response is graded along multiple task-specific criteria. We formalize \emph{rubric-grounded reinforcement learning (RL)}: a framework in which the policy is optimized against a structured, multi-criterion reward produced by a frozen LLM judge that conditions on auxiliary grounding the policy never sees. We instantiate the framework by deriving rubrics from an Office of Scientific and Technical Information (OSTI)-derived corpus of roughly 100,000 scientific and technical documents and training Llama-3.1-8B-Instruct with Group Relative Policy Optimization (GRPO). With GRPO-based training, the model achieves $71.7\%$ normalized reward on held-out rubric evaluation. The GRPO-tuned policy also improves over the base model on four reasoning benchmarks not derived from the training corpus -- GSM8K, MATH, GPQA Main, and GPQA Diamond. These results provide evidence that structured, document-grounded rewards can improve held-out rubric performance and induce transferable reasoning behaviors beyond the corpus used to construct the training environment. 

---
# MPD$^2$-Router: Mask-aware Multi-expert Prior-regularized Dual-head Deferral Router in Glaucoma Screening and Diagnosis 

**Authors**: Wenxin Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2605.08024)  

**Abstract**: Learning-to-defer (L2D) can make glaucoma screening safer by routing difficult/uncertain cases to humans, yet standard formulations overlook expert availability, heterogeneous readers behavior, workload imbalance, asymmetric diagnostic harm, case difficulty from morphology and deployment shift. We introduce MPD$^2$-Router, a mask-aware multi-expert deferral framework that recasts ophthalmic triage as constrained human--AI routing: whether to defer and to which available expert. It couples a dual-head deferral/allocation policy with mask-aware Gumbel--sigmoid gating that strictly enforces per-sample availability, and fuses uncertainty, morphology, image-quality, and OOD signals. Training uses an asymmetric cost-sensitive objective with an augmented-Lagrangian deferral budget, a group-specific distribution prior, and a rank-majorization JS regularizer that jointly prevent expert collapse without forcing uniform allocation. Across three cross-national glaucoma cohorts (REFUGE, CHAKSU, ORIGA) with a frozen REFUGE-trained backbone, MPD$^2$-Router substantially lowers clinical cost and improves MCC over AI-only at a moderate deferral rate. It is Pareto-optimal in F1--MCC--cost, robust under cross-domain shift, and yields balanced expert utilization. 

---
# Reason to Play: Behavioral and Brain Alignment Between Frontier LRMs and Human Game Learners 

**Authors**: Botos Csaba, Sreejan Kumar, Austin Tudor David Andrews, Laurence Hunt, Chris Summerfield, Joshua B. Tenenbaum, Rui Ponte Costa, Marcelo G. Mattar, Momchil Tomov  

**Link**: [PDF](https://arxiv.org/pdf/2605.08019)  

**Abstract**: Humans rapidly learn abstract knowledge when encountering novel environments and flexibly deploy this knowledge to guide efficient and intelligent action. Can modern AI systems learn and plan in a similar way? We study this question using a dataset of complex human gameplay with concurrent fMRI recordings, in which participants learn novel video games that require rule discovery, hypothesis revision, and multi-step planning. We jointly evaluate models by their ability to play the games, match human learning behavior, and predict brain activity during the same task, comparing a suite of frontier Large Reasoning Models (LRMs) against model-free and model-based deep reinforcement learning agents and a Bayesian theory-based agent. We find that frontier LRMs most closely match human behavioral patterns during game discovery and predict brain activity an order of magnitude better than both reinforcement learning alternatives across cortical and subcortical regions, with effects robust to permutation controls. Through targeted manipulations, we further show that brain alignment reflects the model's in-context representation of the game state rather than its downstream planning or reasoning. Our results establish LRMs as compelling computational accounts of human learning and decision making in complex, naturalistic environments. Project page with interactive replays: this https URL 

---
# Learning CLI Agents with Structured Action Credit under Selective Observation 

**Authors**: Haoyang Su, Ying Wen  

**Link**: [PDF](https://arxiv.org/pdf/2605.08013)  

**Abstract**: Command line interface (CLI) agents are emerging as a practical paradigm for agent-computer interaction over evolving filesystems, executable command line programs, and online execution feedback. Recent work has used reinforcement learning (RL) to learn these interaction abilities from verifiable task feedback, yet few methods exploit the native structured attributes of CLI actions as learning signals. Beyond this underused action structure, CLI learning also couples two bottlenecks for coding agents. First, the agent must identify task-relevant evidence in a large codebase from partial observations. Second, sparse terminal rewards must be assigned to the actions that shape a long multi-turn trajectory. We study these bottlenecks through shell-driven information extraction and file editing tasks. For selective observation, we introduce $\sigma$-Reveal, an inference-time mechanism that selects token-budgeted context for the same CLI. For credit assignment, we propose Action Advantage Assignment ($\mathrm{A}^3$), a native agentic RL method that preserves the algorithmic complexity of standard agentic RL. $\mathrm{A}^3$ constructs turn-level advantages from episode-level relative feedback, abstract syntax tree (AST) based action sub-chain residuals, and tree-level trajectory margins. To further evaluate this problem setting, we construct ShellOps, a verifiable dataset suite covering CLI tasks in repository environments. 

---
# Abductive Reasoning with Probabilistic Commonsense 

**Authors**: Joseph Cotnareanu, Chiara Roverato, Han Zhou, Didier Chetelat, Yingxue Zhang, Mark Coates  

**Link**: [PDF](https://arxiv.org/pdf/2605.08011)  

**Abstract**: Recent efforts to improve the reasoning abilities of Large Language Models (LLMs) have focused on integrating formal logic solvers within neurosymbolic frameworks. A key challenge is that formal solvers lack commonsense world knowledge, preventing them from making reasoning steps that humans find obvious. Prior methods address this by using LLMs to supply missing commonsense assumptions, but these approaches implicitly assume universal agreement on such commonsense facts. In reality, commonsense beliefs vary across individuals. We propose a probabilistic framework for abductive commonsense reasoning that explicitly models this variation, aiming to determine whether most people would judge a statement as true or false. We introduce Probabilistic Abductive CommonSense (PACS), a novel algorithm that uses an LLM and a formal solver to sample proofs as observations of individuals' distinct commonsense beliefs, and aggregates conclusions across these samples. Empirically, PACS outperforms chain-of-thought reasoning, prior neurosymbolic methods, and search-based approaches across multiple benchmarks. 

---
# The Limits of AI-Driven Allocation: Optimal Screening under Aleatoric Uncertainty 

**Authors**: Santiago Cortes-Gomez, Mateo Dulce Rubio, Carlos Patino, Bryan Wilder  

**Link**: [PDF](https://arxiv.org/pdf/2605.07979)  

**Abstract**: The rise of machine learning has shifted targeted resource allocation in policy and humanitarian settings toward algorithmic targeting based on predicted risk scores. This approach is typically cheaper and faster than traditional screening procedures that directly observe the latent vulnerability status through physical verification. Yet, even access to the true conditional vulnerability probability cannot eliminate misallocation: aleatoric uncertainty over individual vulnerability status is irreducible, and probabilistic targeting inevitably misallocates some resources. In this work we study how screening and algorithmic targeting should be optimally combined in a two-stage allocation framework where a screening stage observes true outcomes for a subset of units before a final allocation stage assigns the resource under a fixed coverage budget. We show that the optimal strategy screens units at the margin of algorithmic allocation, while directly targeting the highest-risk units. Furthermore, we empirically characterize when screening and algorithmic targeting act as complements or substitutes: efficiency gains from screening grow as the aleatoric uncertainty in the population increases. We illustrate our framework with applications in income-based social protection programs and humanitarian demining in Colombia, where the tension between screening costs and allocation efficiency is operationally consequential. 

---
# TraceFix: Repairing Agent Coordination Protocols with TLA+ Counterexamples 

**Authors**: Shuren Xia, Qiwei Li, Taqiya Ehsan, Jorge Ortiz  

**Link**: [PDF](https://arxiv.org/pdf/2605.07935)  

**Abstract**: We present TraceFix, a verification-first pipeline for Large Language Model (LLM) multi-agent coordination. An agent synthesizes a protocol topology as a structured intermediate representation (IR) from a task description, generates PlusCal coordination logic, and iteratively repairs the protocol using counterexamples from the TLA+ model checker (TLC) until verification succeeds. Verified process bodies are compiled into per-agent system prompts and executed under a runtime monitor that rejects out-of-topology coordination operations. On 48 tasks spanning 16 scenario families, all tasks reach full TLC verification; 62.5% pass on the first attempt and none requires more than four repair iterations. State spaces span six orders of magnitude yet verification completes in under 60 s for every task. A 3,456-run runtime comparison shows that topology-monitored execution achieves the highest task completion (89.4% average, 81.5% full) and that runtimes using the verified protocol degrade at roughly half the rate of prompt-only and chat-only baselines when model capability is reduced. A paired ablation under a fixed runtime shows that TLC-verified protocols cut deadlock/livelock (DL/LL) from 31.1% to 14.1%, with the largest separation under fault injection. 

---
# AgentEscapeBench: Evaluating Out-of-Domain Tool-Grounded Reasoning in LLM Agents 

**Authors**: Zhengkang Guo, Yiyang Li, Lin Qiu, Xiaohua Wang, Jingwen Xv, Dongyu Ru, Xiaoyu Li, Xiaoqing Zheng, Xuezhi Cao, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2605.07926)  

**Abstract**: As LLM-based agents increasingly rely on external tools, it is important to evaluate their ability to sustain tool-grounded reasoning beyond familiar workflows and short-range interactions. We introduce AgentEscapeBench, an escape-room-style benchmark that tests whether agents can infer, execute, and revise novel tool-use procedures under explicit long-range dependency constraints. Each task defines a directed acyclic dependency graph over tools and items, requiring agents to invoke real external functions, track hidden state revealed incrementally, propagate intermediate results, and submit a deterministically verifiable final answer. AgentEscapeBench includes 270 instances across five difficulty tiers and supports fully automated evaluation. Experiments with sixteen LLM agents and human participants show that performance drops sharply as dependency depth increases: humans decline from 98.3% success at difficulty-5 to 80.0% at difficulty-25, while the best model drops from 90.0% to 60.0%. Trajectory analysis attributes model failures mainly to breakdowns in long-range state tracking, clue adherence, and intermediate-result propagation. These findings suggest that current agents can often handle local tool use but still struggle with deep contextual dependencies. We hope AgentEscapeBench can serve as a diagnostic testbed for measuring current agent capabilities and informing future training efforts toward more robust general-purpose reasoning, action, and adaptation. 

---
# Exact Regular-Constrained Variable-Order Markov Generation via Sparse Context-State Belief Propagation 

**Authors**: François Pachet  

**Link**: [PDF](https://arxiv.org/pdf/2605.07839)  

**Abstract**: Variable-order Markov models generate sequences over a finite alphabet by conditioning each symbol on the longest available suffix of the generated history. Regular constraints, by contrast, describe finite-horizon control requirements by an automaton: fixed positions, forced endings, metrical patterns, and forbidden copied fragments are all special cases. Existing exact methods already handle regular constraints with belief propagation for first-order Markov chains. The contribution here is the variable-order extension: identifying the state space on which the existing BP-regular machinery must be run when the generator is a variable-order/backoff model. A first-order constraint layer can enforce useful support conditions, but it computes future mass after merging histories that a variable-order generator deliberately keeps distinct. We formalize this mismatch and give the sparse construction obtained by replacing the first-order Markov state with the observed context state, then taking the standard product with the regular constraint automaton. For a fixed trained context graph and automaton, inference is linear in the sequence horizon; in general it is polynomial in the number of reachable product edges. This gives the correct variable-order distribution conditioned on regular constraints without expanding to all K-tuples. The same finite-source interface supports reversible data augmentation by inverse count lookup, matching materialized transposition augmentation without storing transformed corpora. We also separate exact BP inference from generation-time backoff policies, such as singleton avoidance, whose stochastic semantics must be made explicit if exactness is claimed. 

---
# RuleSafe-VL: Evaluating Rule-Conditioned Decision Reasoning in Vision-Language Content Moderation 

**Authors**: Zhifeng Lu, Dianyuan Wang, Yuhu Shang, Zhenbo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07760)  

**Abstract**: Platform content moderation applies explicit policy rules and context-dependent conditions to decide whether user content is allowed, restricted, or removed. A correct moderation outcome must therefore depend on which rules a case activates, how those rules interact, and whether the available evidence is sufficient. Current multimodal safety benchmarks largely reduce moderation to matching predefined final labels, leaving this underlying rule structure untested. As a result, a high benchmark score reveals little about whether a model applies the policy correctly or arrives at the correct label through superficial cues. To evaluate this rule-governed process, we introduce RuleSafe-VL, a benchmark for rule-conditioned decision reasoning in vision-language content moderation. Derived from publicly available platform moderation policies, RuleSafe-VL formalizes 93 atomic rules and 92 typed rule relations, yielding 2,166 context-sensitive image-text cases across three high-risk policy families. Its four diagnostic tasks decompose moderation into a rule-conditioned decision chain. They identify activated rules, recover rule interactions, judge decision sufficiency, and resolve outcomes once missing context is supplied. Experiments on 10 frontier, open-source, and safety-oriented VLMs reveal rule-relation recovery as the dominant bottleneck, where the best model reaches only 64.8 Macro-F1 and some safety-oriented models fall below 7 Macro-F1. Decision-state prediction also remains unreliable, peaking at 64.5 Macro-F1. RuleSafe-VL shifts moderation evaluation from final-label scoring toward diagnostic assessment of rule-conditioned decision reasoning. 

---
# Alternating Target-Path Planning for Scalable Multi-Agent Coordination 

**Authors**: Yu Kumagai, Keisuke Okumura  

**Link**: [PDF](https://arxiv.org/pdf/2605.07744)  

**Abstract**: The concurrent target assignment and pathfinding (TAPF) problem extends multi-agent pathfinding (MAPF) by asking planners to allocate distinct targets and collision-free paths to agents. Prior work on TAPF has relied exclusively on Conflict-Based Search (CBS), which tightly couples target assignment and pathfinding, resulting in compute-intensive, non-scalable solutions. In contrast, we propose an iterative refinement framework that decouples target assignment from pathfinding. Our framework builds on modern, fast, suboptimal MAPF solvers, such as LaCAM. Specifically, within a given time budget, it repeatedly solves MAPF for the current target assignment, identifies bottleneck agents via MAPF feedback, and refines the assignment. Empirical results show that feedback-driven reassignment loop is effective, enabling our framework to scale well beyond the reach of the state-of-the-art CBS-based solver while maintaining decent solution quality. This represents a solid step toward practical, large scale TAPF suitable for real-world setups. 

---
# Online Goal Recognition using Path Signature and Dynamic Time Warping 

**Authors**: Douglas Tesch, Nathan Gavenski, Leonardo Amado, Odinaldo Rodrigues, Felipe Meneguzzi  

**Link**: [PDF](https://arxiv.org/pdf/2605.07736)  

**Abstract**: Online goal recognition in continuous domains poses two central challenges: efficiently encoding large trajectories and effectively comparing them. Recent work addresses these challenges by using custom state-space representations and metrics to compare observations against hypotheses. However, these approaches often overlook well-established encoding techniques used in other domains that offer substantial advantages. This paper introduces a novel method for online goal recognition that leverages path signatures, a compact, expressive representation of rough path theory that efficiently captures key semantic features of trajectories, enabling more meaningful comparisons between them. Experiments show that our method consistently outperforms the state of the art in predictive accuracy and online planning efficiency, while remaining competitive offline. 

---
# Hierarchical Task Network Planning with LLM-Generated Heuristics 

**Authors**: Felipe Meneguzzi, Alexandre Buchweitz, Augusto B. Corrêa, Victor Scherer Putrich, André Grahl Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2605.07707)  

**Abstract**: HTN planning is a variation of classical planning where, instead of searching for a linear sequence of actions, an algorithm decomposes higher-level tasks using a method library until only executable actions remain. On one hand, this allows one to introduce domain knowledge that can speed up the search for a solution through the method library. On the other hand, it creates challenges that go beyond those of classical state-space search. While recent research produced a number of heuristics and novel algorithms that speed up HTN planning, these heuristics are not yet as informative as those available in classical planning algorithms. We investigate whether large language models (LLMs) can generate effective search heuristics for HTN planning, extending the methodology of Corrêa, Pereira, and Seipp (2025) from classical to hierarchical planning. Using the Pytrich planner on six standard total-order HTN benchmark domains, we evaluate heuristics generated by nine LLMs under domain-specific prompting and compare them against the TDG and LMCount domain-independent baselines and the PANDA planner. Our results show that LLM-generated heuristics nearly match the coverage of the best available HTN planner, while substantially reducing search effort on 83% of shared problems. 

---
# Finite-Time Analysis of MCTS in Continuous POMDP Planning 

**Authors**: Da Kong, Vadim Indelman  

**Link**: [PDF](https://arxiv.org/pdf/2605.07703)  

**Abstract**: This paper presents a finite-time analysis for Monte Carlo Tree Search (MCTS) in Partially Observable Markov Decision Processes (POMDPs), with probabilistic concentration bounds in both discrete and continuous observation spaces. While MCTS-style solvers such as POMCP achieve empirical success in many applications, rigorous finite-time guarantees remain an open problem due to the nonstationarity and the interdependencies induced by heuristic action selection (e.g., UCB). In the discrete setting, we address these challenges by extending the polynomial exploration bonus to UCB in POMDP setting, yielding polynomial concentration bounds for the empirical value estimation at the root node. For continuous observation spaces, we introduce an abstract partitioning framework and propose a finite-time bound on partitioning loss. Under mild conditions, we prove highprobability bound on value estimates in POMDPs with continuous observation space. Specifically, we propose Voro-POMCPOW, a variant of POMCPOW with f inite-time guarantees that adaptively partitions the continuous observation space using Voronoi cells. This approach maintains a finite branching factor while preserving the original observation generator. Empirical validation demonstrates that the proposed Voro-POMCPOW shows competitive performance while providing theoretical guarantees. Although our analysis focuses on continuous POMDPs, the techniques developed herein are also applicable to continuous MDPs, closing another gap on the MDP side. 

---
# GASim: A Graph-Accelerated Hybrid Framework for Social Simulation 

**Authors**: Xuan Zhou, Yanhui Sun, Hantao Yao, Allen He, Yongdong Zhang, Wu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07692)  

**Abstract**: Large-scale social simulators are essential for studying complex social patterns. Prior work explores hybrid methods to scale up simulations, combining large language models (LLM)-based agents with numerical agent-based models (ABM). However, this incurs high latency due to expensive memory retrieval and sequential ABM execution. To address this challenge, we propose GASim, a graph-accelerated hybrid multi-agent framework for large-scale social simulations. For core agents driven by LLM, GASim introduces Graph-Optimized Memory (GOM) to replace intensive LLM-based retrieval pipelines with lightweight propagation over a sparse memory graph. For the majority of ordinary agents, GASim employs Graph Message Passing (GMP), substituting sequential ABM execution with parallel updates by fine-grained feature aggregation and Graph Attention Network. We further introduce Entropy-Driven Grouping (EDG) that coordinates this hybrid partitioning, leveraging information entropy to dynamically identify emergent core agents situated in information-diverse neighborhoods. Extensive experiments show that GASim not only delivers a substantial 9.94-fold end-to-end speedup over the traditional hybrid framework but also consumes less than 20% of baseline tokens, significantly reducing costs while preserving strong alignment with real-world public opinion trends. Our code is available at this https URL. 

---
# FactoryBench: Evaluating Industrial Machine Understanding 

**Authors**: Yanis Merzouki, Coral Izquierdo, Matei Ignuta-Ciuncanu, Marcos Gomez-Bracamonte, Riccardo Maggioni, Alessandro Lombardi, Camilla Mazzoleni, Federico Martelli, Balazs Gunther, Jonas Petersen, Philipp Petersen  

**Link**: [PDF](https://arxiv.org/pdf/2605.07675)  

**Abstract**: We introduce FactoryBench, a benchmark for evaluating time-series models and LLMs on machine understanding over industrial robotic telemetry. Q&A pairs are organized along four causal levels (state, intervention, counterfactual, decision) instantiating Pearl's ladder of causation, and span five answer formats: four structured formats are scored deterministically and free-form answers are scored by an LLM-as-judge voting protocol. We propose a scalable Q&A generation framework built around structured question templates, present FactoryWave (a dense, multitask, multivariate sensor dataset collected from a UR3 cobot and a KUKA KR10 industrial arm), and construct FactoryBench as a large-scale benchmark of over 70k Q&A items grounded in roughly 15k normalized episodes from FactoryWave, AURSAD, and voraus-AD. Zero-shot evaluation of six frontier LLMs shows that no model exceeds 50% on structured levels or 18% on decision-making, revealing a wide gap between current models and operational machine understanding. 

---
# Tacit Knowledge Extraction via Logic Augmented Generation and Active Inference 

**Authors**: Lorenzo Lamazzi, Aldo Gangemi, Alessio Giberti, Andrea Giovanni Nuzzolese, Vittorio Andrea Rocca, Mattia Torta, Francesco Poggi  

**Link**: [PDF](https://arxiv.org/pdf/2605.07639)  

**Abstract**: Tacit knowledge plays a central role in human expertise, yet it remains difficult to capture, formalize, and reuse in machine-interpretable form. This challenge is especially relevant in procedural domains, where successful execution depends not only on explicit instructions, but also on implicit assumptions, contextual constraints, embodied skills, and experience-based judgments rarely documented. As a result, current knowledge engineering pipelines struggle to transform tacit and process-centric knowledge into formally specified, machine-interpretable representations that can be queried, validated, reasoned over, and reused. In this paper, we introduce a neuro-symbolic framework that combines Logic-Augmented Generation and an Active-Inference-inspired approach for ontology-grounded Knowledge Graph construction. We evaluate the approach in a knowledge transfer case study in manufacturing, using assembly-like repair procedures from instructional videos as a reproducible proxy domain. Results show that the proposed solution improves completeness and semantic quality, advancing neuro-symbolic knowledge engineering for industrial domains. 

---
# Learning to Communicate Locally for Large-Scale Multi-Agent Pathfinding 

**Authors**: Valeriy Vyaltsev, Alsu Sagirova, Anton Andreychuk, Yuri Kuratov, Konstantin Yakovlev, Aleksandr Panov, Alexey Skrynnik  

**Link**: [PDF](https://arxiv.org/pdf/2605.07637)  

**Abstract**: Multi-agent pathfinding (MAPF) is a widely used abstraction for multi-robot trajectory planning problems, where multiple homogeneous agents move simultaneously within a shared environment. Although solving MAPF optimally is NP-hard, scalable and efficient solvers are critical for real-world applications such as logistics and search-and-rescue. To this end, the research community has proposed various decentralized suboptimal MAPF solvers that leverage machine learning. Such methods frame MAPF (from a single agent perspective) as a Dec-POMDP where at each time step an agent has to decide an action based on the local observation and typically solve the problem via reinforcement learning or imitation learning. We follow the same approach but additionally introduce a learnable communication module tailored to enhance cooperation between agents via efficient feature sharing. We present the Local Communication for Multi-agent Pathfinding (LC-MAPF), a generalizable pre-trained model that applies multi-round communication between neighboring agents to exchange information and improve their coordination. Our experiments show that the introduced method outperforms the existing learning-based MAPF solvers, including IL and RL-based approaches, across diverse metrics in a diverse range of (unseen) test scenarios. Remarkably, the introduced communication mechanism does not compromise LC-MAPF's scalability, a common bottleneck for communication-based MAPF solvers. 

---
# Inference Time Causal Probing in LLMs 

**Authors**: Sadegh Khorasani, Saber Salehkaleybar, Negar Kiyavash, Matthias Grossglauser  

**Link**: [PDF](https://arxiv.org/pdf/2605.07631)  

**Abstract**: Causal probing methods aim to test and control how internal representations influence the behavior of generative models. In causal probing, an intervention modifies hidden states so that a property takes on a different value. Most existing approaches define such interventions by training an auxiliary probe classifier, which ties the method to a specific task or model and risks misalignment with the model's predictive geometry. We propose Hidden-state Driven Margin Intervention (HDMI), a probe-free, gradient-based technique that directly steers hidden states using the model's native output. HDMI applies a margin objective that increases the probability of a target continuation while decreasing that of the source, without relying on probe classifiers. We further introduce a lookahead variant (LA-HDMI) for text editing that backpropagates through the softmax embeddings, modifying the current hidden state so that the likelihood of user-specified tokens increases in next token generations while preserving fluency. To evaluate interventions, we measure completeness (whether the targeted property changes as intended) and selectivity (whether unrelated properties are preserved), and report their harmonic mean as an overall measure of reliability. HDMI consistently achieves higher reliability than prior methods on the LGD agreement corpus and the CausalGym benchmark, across Meta-Llama-3-8B-Instruct, and Pythia-70M. 

---
# Parallel Lifted Planning via Semi-Naive Datalog Evaluation 

**Authors**: Dominik Drexler, Oliver Joergensen, Jendrik Seipp  

**Link**: [PDF](https://arxiv.org/pdf/2605.07584)  

**Abstract**: Lifted classical planners operate directly on first-order planning tasks to avoid the computationally demanding grounding step. However, lifted planning is typically slower, as planners must repeatedly instantiate ground structures during search. Many core components of lifted classical planning, such as successor generation, axiom evaluation, task grounding, and delete-relaxed heuristics, have previously been studied through the lens of Datalog evaluation. We build upon this line of work and extend it by developing and analyzing an execution model with two levels of parallelism: rule-level parallelism and grounding parallelism. We further specialize this solver for planning-specific workloads with a grounder based on clique enumeration, which we extend to support semi-naive Datalog evaluation. Our experimental evaluation using greedy best-first search with the FF heuristic shows that our implementation already solves more tasks than the baselines on a single core, and the gap widens as additional cores are used. Moreover, on hard-to-ground tasks where on average 97.6% of the total runtime is spent in Datalog execution, the proposed execution model exhibits an average parallel fraction of 92.4%, while achieving up to a 6-fold speedup on 8 cores in practice. 

---
# Open-Ended Task Discovery via Bayesian Optimization 

**Authors**: Masaki Adachi, Yuta Suzuki, Juliusz Ziomek  

**Link**: [PDF](https://arxiv.org/pdf/2605.07572)  

**Abstract**: When applying Bayesian optimization (BO) to scientific workflow, a major yet often overlooked source of uncertainty is the task itself -- namely, what to optimize and how to evaluate it -- which can evolve as evidence accumulates. We introduce Generate-Select-Refine (GSR), a open-ended BO framework that alternates between task generation and task optimization. Starting from a user-provided seed task, GSR generates new tasks in a coarse-to-fine manner while a task-acquisition function schedules optimization. Asymptotically, it concentrates evaluations on the best task, incurring only logarithmic regret overhead relative to single-task BO. We apply GSR to new product development, chemical synthesis scaling, algorithm analysis, and patent repurposing, where it outperforms existing LLM-based optimizers. 

---
# From Pixels to Prompts: Vision-Language Models 

**Authors**: Khang Hoang Nhat Vo  

**Link**: [PDF](https://arxiv.org/pdf/2605.07544)  

**Abstract**: When you read a paper about a new Vision-Language Model today, it can be easy to forget how strange this idea would have sounded not so long ago. Teaching machines to see was already hard. Teaching them to read and generate language was already hard. Asking them to do both at once - and then to reason, answer questions, follow instructions, and sometimes even surprise us - still carries a quiet trace of science fiction, even as it becomes routine. This book was born from a simple feeling: \emph{it is too easy to get lost}. The field moves quickly, new model names appear constantly, and the gap between ``I know the buzzwords'' and ``I actually understand how this works'' can feel uncomfortably wide. I have felt that gap many times. If you are holding this book, you probably have too. My goal is not to provide an exhaustive catalog of every dataset, benchmark, and new model variant. Instead, I want to offer something more modest - and, I hope, more durable: a clear mental map of Vision-Language Models. Enough structure that you can read new papers with confidence; enough intuition that you can design your own systems without feeling as if you are assembling LEGO bricks blindly. 

---
# Multi-Environment POMDPs with Finite-Horizon Objectives 

**Authors**: Léonard Brice, Filip Cano, Krishnendu Chatterjee, Thomas A. Henzinger, Stefanie Muroya  

**Link**: [PDF](https://arxiv.org/pdf/2605.07537)  

**Abstract**: Partially Observable Markov Decision Processes (POMDPs) are systems in which one agent interacts with a stochastic environment, and receives only partial information about the current state. In a multi-environment POMDP (MEPOMDP), the initial state is unknown, and assumed to be adversarially chosen. In this work we focus on computing the optimal value and policy in MEPOMDPs with finite-horizon objectives. That problem is known to be PSPACE-complete in POMDPs. Our main results are as follows: (1) we establish that it is also PSPACE-complete in the more general setting of MEPOMDPs; (2) we present a practical algorithm and evaluate it on classical benchmarks, significantly outperforming the only previously known algorithm. 

---
# From Feasible to Practical: Pareto-Optimal Synthesis Planning 

**Authors**: Friedrich Hastedt, Dongda Zhang, Antonio del Rio Chanona  

**Link**: [PDF](https://arxiv.org/pdf/2605.07521)  

**Abstract**: Current computer-aided synthesis planning (CASP) methods often treat retrosynthesis as solved once a single feasible route is identified, focusing primarily on convergence or shortest-path metrics. This view is misaligned with real-world practice, where chemists must balance competing objectives such as cost, sustainability, toxicity, and overall yield. To address this, we formulate synthesis planning as a multi-objective search problem and introduce MORetro*, an algorithm that generates a Pareto front of synthesis routes to explicitly capture trade-offs among user-defined criteria. MORetro* uses weighted scalarization and BO-informed sampling to efficiently navigate the combinatorial search space and prioritize promising trade-offs. Building on multi-objective A*-search, we provide optimality guarantees showing that, for a fixed single-step model, MORetro* recovers the true Pareto front. Across multiple retrosynthesis benchmarks, MORetro* produces diverse, high-quality Pareto fronts, uncovering solutions overlooked by single-objective approaches and better aligning CASP outputs with industrial decision-making. 

---
# Model-Driven Policy Optimization in Differentiable Simulators via Stochastic Exploration 

**Authors**: Yuval Aroosh, Ayal Taitler  

**Link**: [PDF](https://arxiv.org/pdf/2605.07520)  

**Abstract**: Differentiable planning enables gradient-based optimization of decision-making problems by leveraging differentiable models of system dynamics. However, in highly nonlinear and hybrid discrete-continuous domains, the resulting optimization landscapes are often ill-conditioned, with flat regions and sharp transitions that hinder effective optimization.
We propose Model-Driven Policy Optimization (MDPO), a framework that introduces stochastic exploration into differentiable planning by injecting noise into the action space during optimization. Leveraging access to the model, MDPO further adapts the noise magnitude based on gradient-derived sensitivity of the trajectory objective, yielding a time-dependent exploration profile. This enables improved exploration of the objective landscape and helps escape poor local optima via dynamic allocation of exploration across timesteps and iterations.
Experiments on benchmark domains demonstrate that MDPO consistently outperforms deterministic differentiable planning, including both the noise-free variant of our method and available state-of-the-art implementations, as well as model-free baselines such as PPO, significantly improving solution quality across challenging nonlinear and hybrid settings. We further analyze the evolution of the adaptive noise magnitude across both time steps and optimization iterations, providing insight into how exploration is allocated during learning. 

---
# LiteGUI: Distilling Compact GUI Agents with Reinforcement Learning 

**Authors**: Yubin Wu, Zicheng Cai, Liping Ning, Hua Wang, Zhi Chen, Yaohua Tang, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.07505)  

**Abstract**: Developing lightweight, on-device vision-language GUI agents is essential for efficient cross-platform automated interaction. However, current on-device agents are constrained by limited model capacity, and further performance improvements remain urgently needed. Traditional Supervised Fine-Tuning (SFT) for small-scale models often leads to overfitting, catastrophic forgetting and policy rigidity, and thus fails to fully address these challenges. In this work, we propose a novel SFT-free training paradigm that significantly enhances the performance of small-scale models. We first present the initial systematic integration of generalized knowledge distillation into the GUI agent domain via Guided On-policy Distillation. By incorporating oracle reference trajectories together with a dynamic retrieval mechanism, our method reduces hallucinations and mitigates the cognitive misalignment inherent in multi-solution GUI tasks. Building on this foundation, we further introduce a Multi-solution Dual-level GRPO framework that jointly aligns macro-level subtask planning with micro-level execution matching, thereby improving exploration in long-horizon GUI agent scenarios. In addition, we construct an automated data generation pipeline to synthesize GUI task trajectories with rich multi-solution annotations. Extensive experiments show that our method achieves state-of-the-art performance among lightweight models while remaining competitive with substantially larger-scale models across all benchmarks. Ablation studies further demonstrate that structured on-policy distillation and multi-solution dual-level exploration can fully unlock the capabilities of 2B/3B scale agents, surpassing the performance limits of conventional imitation learning. 

---
# Efficient Data Selection for Multimodal Models via Incremental Optimization Utility 

**Authors**: Jinhao Jing, Qiannian Zhao, Chao Huang, Zhan Su  

**Link**: [PDF](https://arxiv.org/pdf/2605.07488)  

**Abstract**: The scaling of Large Multimodal Models (LMMs) is constrained by the quality-quantity trade-off inherent in synthetic data. Previous approaches, such as LLM-as-a-Judge, have proven their effectiveness in addressing this but suffer from prohibitive computational costs and lack of interpretability. To bridge this gap, we propose One-Step-Train (OST), a framework that reformulates data selection as an incremental optimization utility ranking problem. Instead of relying on semantic heuristics, OST estimates the marginal utility of each sample via a simulated single-step update on a lightweight proxy. Experiments on the Qwen series across multimodal mathematical reasoning benchmarks demonstrate that OST achieves Pareto-optimal efficiency. By selecting the top-50 subset, OST reduces training costs by 43% (and total time consumption by 17) while surpassing the strong LLM-as-a-Judge baseline by 1.8 points. Furthermore, under a fixed compute budget, our method using only the top-20 subset achieves a 5.6 point gain over LLM-as-a-Judge, improves upon heuristic scoring baselines like DEITA, and outperforms the Full-SFT baseline by 8.8 points. Notably, while Full-SFT suffers from performance degradation due to noise, our optimization-grounded approach effectively identifies toxic samples, successfully reversing the negative transfer frequently observed in complex reasoning tasks. 

---
# Bounded Fitting for Expressive Description Logics 

**Authors**: Maurice Funk, Jean Christoph Jung, Tom Voellmer  

**Link**: [PDF](https://arxiv.org/pdf/2605.07452)  

**Abstract**: Bounded fitting is an attractive paradigm for learning logical formulas from labeled data examples that offers PAC-style generalization guarantees and can often be implemented leveraging SAT solvers. It has been successfully applied to learning concepts of the description logic ALC. We study bounded fitting for learning concepts in expressive description logics that extend ALC with inverse roles, qualified number restrictions, and feature comparisons. We investigate under which conditions bounded fitting keeps its favorable theoretical properties in this setting, and implement it using a SAT solver. We compare our tool with state-of-the-art concept learners with encouraging results, demonstrating that it is a practical approach to expressive concept learning. 

---
# Offline Policy Optimization with Posterior Sampling 

**Authors**: Hongqiang Lin, Dongxu Zhang, Yiding Sun, Mingzhe Li, Ning Yang, Haijun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07393)  

**Abstract**: A fundamental challenge in model-based offline reinforcement learning (RL) lies in the trade-off between generalization and robustness against exploitation errors in out-of-distribution (OOD) regions. While OOD samples may capture valid underlying physical dynamics, they also introduce the risk of model exploitation. Existing methods typically address this risk through excessive pessimistic regularization, which ensures robustness but often sacrifices generalization. To overcome this limitation, we propose Posterior Sampling-based Policy Optimization (PSPO), which formulates dynamics modeling as a Bayesian inference process to derive a posterior that explicitly quantifies model fidelity. Through the integration of posterior sampling and constrained policy optimization, our method leverages dynamics-consistent OOD transitions for generalization while ensuring robustness against model exploitation. Theoretically, we formulate Q-value estimation under posterior sampling as a stochastic approximation problem and establish its convergence. We decompose policy optimization into a sequence of constrained subproblems, demonstrating that solving these subproblems guarantees monotonic improvement until convergence. Experiments on standard benchmarks validate that PSPO achieves superior performance compared to state-of-the-art baselines. 

---
# GraphReAct: Reasoning and Acting for Multi-step Graph Inference 

**Authors**: Xingtong Yu, Zhongwei Kuai, Chang Zhou, Xuanting Xie, Renhe Jiang, Xikun Zhang, Hong Cheng, Xinming Zhang, Yuan Fang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07357)  

**Abstract**: Reasoning-acting frameworks enhance large language models (LLMs) by interleaving reasoning with actions for dynamic information acquisition. However, extending this paradigm to graph learning remains underexplored. Graph data is inherently structured, with information distributed across nodes and edges and encoded through both topology and latent representations. As a result, effective reasoning over graphs requires not only retrieving informative evidence from the graph, but also progressively refining the accumulated context during multi-step inference. In this work, we propose GraphReAct, a graph reasoning-acting framework that enables step-by-step inference over graph-structured data. Specifically, we design a graph-based action space with two complementary retrieval actions: topological retrieval, which captures local structural dependencies, and semantic retrieval, which accesses non-local but relevant evidence in the representation space. These actions dynamically expand the reasoning context. To further support multi-step reasoning, we introduce another type of action, context refinement, which distills and reorganizes accumulated information into a compact representation. By interleaving reasoning with both retrieval and refinement actions, our framework enables a progressive transition from context expansion to compression. Extensive experiments on six benchmark datasets demonstrate that GraphReAct consistently outperforms state-of-the-art methods, validating the effectiveness of reasoning-acting for graph learning. 

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
# When Stored Evidence Stops Being Usable: Scale-Conditioned Evaluation of Agent Memory 

**Authors**: Jiaqi Shao, Yiyi Lu, Yunzhen Zhang, Bing Luo  

**Link**: [PDF](https://arxiv.org/pdf/2605.07313)  

**Abstract**: Memory-agent evaluations report fixed-snapshot accuracy or retrieval quality, but these scores do not show whether evidence remains usable as irrelevant sessions (sessions not annotated as task-relevant evidence for the query) accumulate. We present a scale-conditioned evaluation protocol for agent memory under evidence-preserving growth: for each query, task evidence is held fixed while irrelevant sessions are added. The protocol logs agent--memory trajectories and reports four diagnostics: budget-compliant reliability, tail memory-call burden, failure-regime decomposition, and the usable-scale boundary where reliability falls below the target. Applied to LongMemEval and LoCoMo across flat, planar, and hierarchical memory interfaces, the protocol shows reliability loss is not a single phenomenon. On LongMemEval, HippoRAG stays within the two-call budget but loses 16--20 percentage points in budget-compliant reliability as irrelevant sessions are added; LiCoMemory's observed failures depend strongly on the agent, with Qwen3-8B exceeding the budget while Qwen3-32B and Qwen3-235B remain reliable in the tested range. The result supports a framework for making scalable-memory claims conditional on agent, interface, scale range, and interaction budget. 

---
# SOM: Structured Opponent Modeling for LLM-based Agents via Structural Causal Model 

**Authors**: Shiyue Cao, Pei Xu, Likun Yang, Lei Cui, Xiaotang Chen, Kaiqi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07301)  

**Abstract**: Accurately predicting opponents' behavior from interactions is a fundamental capability for large language model (LLM)-based agents in multi-agent and game-theoretic environments. Existing approaches often entangle opponent modeling with prediction, relying on implicit contextual reasoning and limiting adaptability in dynamic interactions. To this end, we propose Structured Opponent Modeling (SOM), a two-stage opponent modeling framework that distinctly separates opponent model construction and opponent prediction. At the construction stage, SOM employs a Structural Causal Model (SCM), a graph-based formalism for representing dependencies among variables, to capture directed links between opponents' observations and actions, yielding an explicit and structured opponent representation. At the prediction stage, the LLM performs structured reasoning along clear pathways derived from the SCM, improving both prediction accuracy and stability. Extensive experiments on diverse multi-agent benchmarks demonstrate that SOM consistently outperforms state-of-the-art LLM-based reasoning baselines, enabling more accurate and adaptable strategic decision-making in complex and dynamic multi-agent interactions. 

---
# Signal Reshaping for GRPO in Weak-Feedback Agentic Code Repair 

**Authors**: Jia Li, Yuxin Su, Ting Peng, Hailiang Huang, Yuetang Deng, Michael R. Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07276)  

**Abstract**: Code-agent RL often receives weak feedback: rollout-time signals are reliable and executable, but capture only necessary or surface conditions for task success rather than the target semantic predicate. Using agentic compile-fix as the setting, we study signal reshaping for standard GRPO under such feedback. Our central claim is that GRPO's within-group comparison is meaningful only after three kinds of signals are reshaped: outcome rewards recover semantic ranking, process signals localize intra-trajectory credit, and rollouts from the same prompt remain execution-comparable. We operationalize these conditions with a minimal signal-reshaping construction that leaves GRPO's group-normalized advantage construction unchanged: compile-and-semantic layered rewards reshape trajectory ranking, step-level process scores outside group reward normalization reshape within-trajectory update strength, and failure-cause-aware rollout governance reshapes within-group comparability. Experiments show a clear end-to-end gain: full signal-reshaped GRPO improves strict compile-and-semantic accuracy from the base model's zero-shot $0.385$ to $0.535$. Controlled comparisons further explain the source of this gain: binary rewards remove the compile-only middle tier and degrade trajectory control; on top of layered rewards, process-score weighting further improves accuracy from $0.48$ to $0.53$ and reduces average evaluation steps from $23.50$ to $17.02$. As a boundary comparison, privileged-prompt token-level distillation mainly optimizes local distributional alignment; in long tool-use trajectories, this signal is diluted by non-critical tokens and cannot replace outcome semantics, process credit, or within-group comparability. 

---
# Structured Role-Aware Policy Optimization for Multimodal Reasoning 

**Authors**: Bingqing Jiang, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2605.07274)  

**Abstract**: Reinforcement learning from verifiable rewards (RLVR), especially with Group Relative Policy Optimization (GRPO), has shown strong potential for improving the reasoning capabilities of large vision-language models (LVLMs). However, in multimodal reasoning, final-answer rewards are typically assigned at the sequence level and do not distinguish the functional roles of different tokens, making it difficult to determine whether a correct answer is supported by task-relevant visual evidence. In this paper, we revisit multimodal RLVR from the perspective of role-aware token-level credit assignment, where structured responses are decomposed into perception tokens for extracting visual evidence and reasoning tokens for deriving answers from that evidence. Based on this perspective, we propose Structured Role-aware Policy Optimization (SRPO), which refines the sequence-level GRPO advantage into role-aware token-level advantages without changing the reward function. Specifically, SRPO assigns role-specific credit by using self-distilled on-policy contrasts: perception tokens are emphasized according to their visual dependency under original versus corrupted visual inputs, while reasoning tokens are emphasized according to their consistency with the generated perception. These role-specific signals are further unified through a shared trajectory-level baseline, yielding positive token weights that adjust relative update magnitudes while preserving the original GRPO reward and optimization direction, without requiring external reward models or separate teachers. Experiments across diverse multimodal reasoning benchmarks show that SRPO improves evidence-grounded reasoning, highlighting the importance of moving beyond uniform sequence-level credit toward role-aware optimization for reliable multimodal reasoning. 

---
# Can Agents Price a Reaction? Evaluating LLMs on Chemical Cost Reasoning 

**Authors**: Yuyang Wu, Yue Huang, Shuaike Shen, Xujian Wang, Shuhao Zhang, Qiyao Xue, Weichen Liu, Runtian Gao, Jian Ma, Xiangliang Zhang, Olexandr Isayev  

**Link**: [PDF](https://arxiv.org/pdf/2605.07251)  

**Abstract**: Large Language Models (LLMs) have become increasingly capable as tool-using agents, with benchmarks spanning diverse general agentic tasks. Yet rigorous evaluation of scientific tool use remains limited. In chemistry, recent agents can plan syntheses and invoke domain-specific tools, but evaluations often rely on curated demonstrations, expert assessment, or LLM-as-judge scoring rather than exact, judge-free ground truth. We address this gap with chemical procurement cost estimation, a practical task in which an agent must ground chemical identities, retrieve supplier quotes, select valid purchasable packs, normalize quantities, and compute cost from a reaction description. We introduce ChemCost, a benchmark of 1,427 evaluable reactions grounded to a frozen pricing snapshot covering 2,261 chemicals and 230,775 supplier quotes, supporting scalar scoring and stage-level diagnosis of grounding, retrieval, procurement, and arithmetic failures. To evaluate robustness, we further construct controlled noise-injected views that perturb chemical aliases, quantity expressions, missing fields, and input formatting. Experiments with frontier, open-weight, and chemistry-specialized LLM agents show that tool access is necessary but insufficient for solving the task. The strongest agents reach only 50.6% accuracy within 25% relative error on clean inputs and degrade substantially with realistic noise. Stage-level analysis further shows that failures arise from brittle parsing, ineffective evidence integration, invalid pack selection, and non-convergent tool use. 

---
# EnvSimBench: A Benchmark for Evaluating and Improving LLM-Based Environment Simulation 

**Authors**: Yi Liu, TingFeng Hui, Wei Zhang, Li Sun, Ningxin Su, Jian Wang, Sen Su  

**Link**: [PDF](https://arxiv.org/pdf/2605.07247)  

**Abstract**: Scalable AI agents training relies on interactive environments that faithfully simulate the consequences of agent actions. Manually crafted environments are expensive to build, brittle to extend, and fundamentally limited in diversity. A promising direction is to replace manually crafted environments with LLM-simulated counterparts. However, this paradigm hinges on an unexamined core assumption: LLMs can accurately simulate environmental feedback. In practice, LLM-simulated environments suffer from hallucinations, logical inconsistencies, and silent state drift failures that corrupt agent reward signals and compound the construction costs that the paradigm was designed to eliminate. To address this gap, we propose EnvSimBench with four contributions: 1) We provide the first formal definition and operationalization of Environment Simulation Ability (EnvSim Ability) as a quantifiable research objective. 2) We construct EnvSimBench, a rigorous benchmark covering 400 samples across 167 diverse environments, equipped with verifiable labels and fine-grained difficulty stratification along three axes. 3) Systematic evaluations reveal that all state-of-the-art language models suffer from a universal state change cliff: they achieve near-perfect accuracy on tasks when the environment state remains invariant, yet fail catastrophically when multiple states need simultaneous updates. This finding exposes EnvSim Ability as a critical yet largely unaddressed capability gap. 4) We design a constraint-driven simulation pipeline that substantially reduces hallucination, boosts environment synthesis yield by 6.8%, and cuts costs by over 90%. Overall, EnvSimBench serves as both a diagnostic framework and a practical optimization path for reliable LLM-based environment simulation, establishing a foundation for scalable agent training. Code and data are available at this https URL 

---
# MEMOREPAIR: Barrier-First Cascade Repair in Agentic Memory 

**Authors**: Yang Zhao, Chengxiao Dai, Mengying Kou, Yue Xiu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07242)  

**Abstract**: Agentic memory evolves across tasks into durable derived artifacts: summaries, cached outputs, embeddings, learned skills, and executable tool procedures. When a source artifact is deleted, corrected, or invalidated by tool or API migration, descendants derived from that source can remain visible and steer future actions with stale support. We formalize this failure mode as the cascade update problem, where repair targets the visible derived state of the memory store. We present MemoRepair, a barrier-first cascade-repair contract for agentic memory. A repair event induces a controlled transition from invalidated descendant state to validated successor state: affected descendants are withdrawn before repair, successors are constructed from retained support and staged repaired predecessors under the current interface, and republication is restricted to validated predecessor-closed successors. This contract induces a scalarized repair-selection problem for a fixed repair-cost tradeoff. We show that the induced publication problem reduces to maximum-weight predecessor closure and can be solved exactly by a single s-t min-cut. Experiments on ToolBench and MemoryArena show that, with complete influence provenance, MemoRepair reduces invalidated-memory exposure from 69.8-94.3% under systems without cascade repair to 0%. Compared with exhaustive Repair all, it recovers 91.1-94.3% of validated successors while reducing normalized repair-operator cost from 1.00 to 0.57-0.76. 

---
# HMACE: Heterogeneous Multi-Agent Collaborative Evolution for Combinatorial Optimization 

**Authors**: Yuping Yan, Jirui Han, Fei Ming, Yuanshuai Li, Yaochu Jin  

**Link**: [PDF](https://arxiv.org/pdf/2605.07214)  

**Abstract**: Large Language Models have recently emerged as a promising paradigm for automated heuristic design for NP-hard combinatorial optimization problems. Despite this progress, existing LLM-based methods typically rely on monolithic workflows constrained by rigid templates, thereby restricting memory-guided exploration and triggering premature convergence to local optima. To design an autonomous and collaborative architecture, we introduce HMACE, a Heterogeneous Multi-Agent Collaborative Evolution framework that reconceptualizes heuristic search as an organizational design problem. HMACE decomposes each evolutionary generation into an autonomous, role-specialized loop with four coordinated agents: a Proposer for strategy exploration, a Generator for executable heuristic synthesis, an Evaluator for empirical assessment, and a Reflector for archive-backed memory update. By coupling behavior-aware retrieval, lightweight candidate filtering, and fitness-grounded archive updates, HMACE guides the search toward diverse and promising heuristic behaviors while avoiding redundant evaluations. Extensive evaluations on representative COPs, including TSP, Online BPP, MKP, and PFSP, show that HMACE achieves a favorable quality-efficiency trade-off compared to state-of-the-art single-agent and multi-agent baselines. In the matched LLM-driven reference comparison, HMACE achieves the lowest average gaps on TSP and Online BPP (0.464\% and 0.223\%, respectively), while requiring only 0.13M and 0.42M tokens for the two tasks, substantially fewer than the compared baselines. 

---
# Towards Autonomous Business Intelligence via Data-to-Insight Discovery Agent 

**Authors**: Dongming Wu, Junwen Li, Ming Lu, Gang Wang, Ting Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.07202)  

**Abstract**: Transforming fragmented enterprise data into actionable insights remains a significant challenge for LLMs, constrained by complex database schemas, limitations in dynamic SQL generation, and the need for deep multi-dimensional this http URL this paper, we propose AIDA(Autonomous Insight Discovery Agent), the first end-to-end framework designed for autonomous exploration in complex business environments. We establish a highly flexible instant retail environment encompassing 200+ metrics and 100+ dimensions, and integrates a proprietary Domain-Specific Language (DSL) that bridges semantic reasoning with precise SQL execution. Our reinforcement learning system subsequently formulates business analysis as a Pareto Principle-guided cumulative reasoning process. Experimental results demonstrate that AIDA significantly outperforms workflow-based agents, and extensive evaluations further reveal that AIDA achieves superior environmental perception and more in-depth analysis from diverse perspectives. Our work ultimately establishes the transformative potential of autonomous intelligence for industrial-scale business intelligence systems. 

---
# Three-in-One World Model: Energy-Based Consistency, Prediction, and Counterfactual Inference for Marketing Intervention 

**Authors**: Junichiro Niimi  

**Link**: [PDF](https://arxiv.org/pdf/2605.07199)  

**Abstract**: Marketing decisions reflect the interaction of latent consumer heterogeneity, time-varying internal states, and explicit interventions, a structure that current prediction- and language-oriented models do not capture in a unified manner. We propose a Three-in-One world-model architecture in which a Deep Boltzmann Machine (DBM) learns a frozen belief representation from demographics, time, and lagged actions and outcomes, with lightweight task-specific adapters attached on top. The same belief supports three tasks within a single framework: (i) energy-based consistency evaluation through the DBM's free energy, (ii) outcome prediction through adapters, and (iii) counterfactual inference by holding the belief fixed and varying only the action input given to the adapter. Using a controlled simulation in which the latent price sensitivity, promotion responsiveness, and base preference of each consumer are known, we show that the adapters match a strong MLP baseline on visit- and purchase-AUC while recovering heterogeneous treatment effects substantially better than S-, T-, X-, and DR-learner meta-learners and a Causal Forest baseline built on the same raw features, with the largest gap on a confounded price-promotion intervention. Complementing this, free-energy clamps systematically penalize counterfactual purchase trajectories that lack prior promotional exposure, and the penalty itself depends on the latent base preference in the expected direction. These results indicate that DBM beliefs disentangle latent traits in a form that survives counterfactual queries, providing an integrated world-model substrate for marketing intervention. 

---
# Repeated Deceptive Path Planning against Learnable Observer 

**Authors**: Shiyue Cao, Pei Xu, Likun Yang, Lei Cui, Shizhao Yu, Shiyu Zhang, Yongjian Ren, Xiaotang Chen, Kaiqi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07174)  

**Abstract**: We study the problem of deceptive path planning (DPP), where an agent aims to conceal its true destination from external observers. While existing work assumes static, non-learning observers, real-world adversaries-such as in critical goods transportation or military operations-can adapt by learning from historical trajectories. To address this gap, we introduce Repeated Deceptive Path Planning (RDPP), a new formulation that explicitly models learnable observers. We show that existing DPP methods fail under this setting, as they cannot adapt to evolving adversarial predictions. While incorporating observer previous predictions into updates enables some adaptation, such incremental updates cause accumulative lag that degrades deception. To this end, we propose Deceptive Meta Planning (DeMP), a two-level optimization framework that combines episode-level adaptation, which enables short-term policy adjustment to counter updated observer, and meta-level updates, which leverage cross-episode feedback to capture how observers update their models and accelerate adaptation in future episodes. In this way, DeMP mitigates the accumulation of adaptation lag, enabling sustained deception against a learning observer. Experiments across environments demonstrate that DeMP significantly outperforms existing approaches in RDPP while maintaining competitive path cost. Our results highlight the importance of modeling repeated interactions with learnable adversaries, providing new insights into deception and privacy in multi-agent systems. 

---
# SREGym: A Live Benchmark for AI SRE Agents with High-Fidelity Failure Scenarios 

**Authors**: Jackson Clark, Yiming Su, Saad Mohammad Rafid Pial, Yifang Tian, Lily Gniedziejko, Hans-Arno Jacobsen, Yinfang Chen, Tianyin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07161)  

**Abstract**: AI agents are increasingly used to diagnose and mitigate failures in production systems, known as agentic Site Reliability Engineering (SRE). Current SRE benchmarks are limited to oversimplistic SRE tasks and are unfortunately hard to extend due to bespoke designs. We present SREGym, a high-fidelity benchmark for SRE agents. SREGym exposes a live system environment built atop real-world cloud-native system stacks, where high-fidelity failure scenarios are simulated through fault injectors. SREGym models the complexity of production environments by simulating (1) a wide range of faults at different layers, (2) various ambient noises, and (3) diverse failure modes such as metastable failures and correlated failures. SREGym is architected as a modular, extensible framework that orchestrates fault and noise injectors across stacks. SREGym currently includes 90 realistic, challenging SRE problems. We use SREGym to evaluate frontier agents and show that their capabilities varies significantly in addressing different kinds of failures, with up to 40% differences in end-to-end results. SREGym is actively maintained as an open-source project and has been used by researchers and practitioners. 

---
# Can You Break RLVER? Probing Adversarial Robustness of RL-Trained Empathetic Agents 

**Authors**: Deeraj S K, Sadhana Devarajan, Krishna Mehra, Sudhakar Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2605.07138)  

**Abstract**: Reinforcement learning from verifiable emotion rewards RLVER has produced language models with strong empathetic performance, evaluated on benchmarks that assume cooperative, honest users. Yet real emotional interactions systematically violate this assumption: users gaslight, escalate, and pressure AI systems for unconditional validation, dynamics that cooperative benchmarks cannot surface. We construct the Adversarial Empathy Benchmark AEB and introduce the Emotional Consistency Score ECS to evaluate empathetic robustness under adversarial conditions. AEB comprises six psychologically grounded adversarial trajectory types with discriminative reward structures that penalize formulaic responses; ECS formally disentangles a model's capacity to track user emotional states from its capacity to improve them. In a controlled experiment across eight scenario-matched conditions (think and no-think conditions on 2 RLVER models, and 2 base models (Qwen 1.5B and 7B) with 480 adversarial dialogues), RLVER-PPO-Think substantially outperforms the same-scale untuned baseline (0.963 vs. 0.761, \(p<0.001, r=0.688\)), with zero dialogue collapses and 47\% higher hidden-intention detection. However, ECS remains nearly flat and is not significantly different for RLVER-PPO-Think versus Base-7B-Think (\(p=0.650\)): RL training improves emotional responsiveness without measurable gains in observable state tracking. We interpret the ECS--FS (Final Score) gap as a behavioral/legibility dissociation inside this simulator family, not as evidence about internal understanding or clinical readiness. 

---
# AdaTKG: Adaptive Memory for Temporal Knowledge Graph Reasoning 

**Authors**: Seunghan Lee, Jun Seo, Jaehoon Lee, Sungdong Yoo, Minjae Kim, Tae Yoon Lim, Dongwan Kang, Hwanil Choi, SoonYoung Lee, Wonbin Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2605.07121)  

**Abstract**: Temporal knowledge graphs (TKGs) represent time-stamped relational facts and support a wide range of reasoning tasks over evolving events. However, existing methods produce entity representations that are static at the entity level, in that each representation is a function of learned parameters only and retains no trace of the interactions in which the entity has participated. In this paper, we depart from this static view and propose that each entity be modeled as an adaptive process whose representation is refined every time the entity participates in a fact. To this end, we propose AdaTKG, which maintains a per-entity memory that is updated with every observed interaction, with the memory accumulating online and predictions improving as more interactions arrive. Specifically, we instantiate the memory update as a learnable exponential moving average governed by a single shared scalar instead of using learnable parameters for each entity, enabling AdaTKG to handle entities unseen during training. Extensive experiments confirm consistent gains over TKG baselines, demonstrating the effectiveness of adaptive memory. Code is publicly available at: this https URL. 

---
# Switchcraft: AI Model Router for Agentic Tool Calling 

**Authors**: Sharad Agarwal, Pooria Namyar, Alec Wolman, Rahul Ambavat, Ankur Gupta, Qizheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07112)  

**Abstract**: Agentic AI systems that invoke external tools are powerful but costly, leading developers to default to large models and overspend inference budgets. Model routing can mitigate this, but existing routers are designed for chat completion rather than tool use. We present Switchcraft, the first (to the best of our knowledge) model router optimized for agentic tool calling. Switchcraft operates inline, selecting the lowest-cost model subject to correctness. We construct an evaluation framework on five function-calling benchmarks and train a DistilBERT-based classifier, deployed under a latency budget. Switchcraft achieves 82.9% accuracy -- matching or exceeding the best individual model -- while reducing inference cost by 84%, saving over $3,600 per million queries. We find that larger models do not consistently outperform smaller ones on tool-use tasks, and that nominally cheaper models can incur higher total cost due to token-intensive reasoning. Our work enables cost-aware agentic AI deployment without sacrificing correctness. 

---
# ARMOR: An Agentic Framework for Reaction Feasibility Prediction via Adaptive Utility-aware Multi-tool Reasoning 

**Authors**: Ye Liu, Botao Yu, Xinyi Ling, Daniel Adu-Ampratwum, Xia Ning  

**Link**: [PDF](https://arxiv.org/pdf/2605.07103)  

**Abstract**: Reaction feasibility prediction, as a fundamental problem in computational chemistry, has benefited from diverse tools enabled by recent advances in artificial intelligence, particularly large language models. However, the performance of individual tools varies substantially across reactions, making it difficult for any single tool to consistently perform well across all cases. This raises a critical challenge: how to effectively leverage multiple tools to obtain more accurate feasibility predictions. To address this, we propose ARMOR, an agentic framework that explicitly models tool-specific utilities, adaptively prioritizes tools, and further resolves the potential tool conflicts to produce the final prediction for each reaction. Unlike existing approaches that rely on simple aggregation or heuristic assignment over various tools, ARMOR organizes tools into a hierarchy that prioritizes top-performing tools and defers others when needed, characterizes their strengths through tool-specific patterns, and resolves conflicts via memoryaugmented reasoning. Extensive experiments on a public dataset demonstrate that ARMOR consistently outperforms strong baselines, including single-tool methods as well as various tool aggregation and tool selection approaches. Further analysis shows that the improvements are particularly significant on reactions with conflicting tool predictions, highlighting the effectiveness of ARMOR in leveraging the complementary strengths of multiple tools. The code is available via this https URL. 

---
# Online Allocation with Unknown Shared Supply 

**Authors**: Tzeh Yuan Neoh, Davin Choo, Mengchu Yue, Milind Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2605.07080)  

**Abstract**: Many real-world resource allocation systems, such as humanitarian logistics and vaccine distribution, must preposition limited supply across multiple locations before demand is realized while stockouts incur irreversible service losses. To study this, we introduce the Online Shared Supply Allocation (OSSA) problem, a stateful online model in which a central hub allocates a finite, unknown supply to multiple sites facing sequential demand under fixed-charge transportation costs and lost-sales penalties. Unlike classical make-to-stock or make-to-order inventory models, OSSA precludes backlogging and replenishment only hedges against future demand. To tackle OSSA, we propose a deterministic threshold-proportional policy GPA and prove that it achieves a $4/3$-approximation to the offline optimum up to an additive term independent of the total supply. We complement this with matching lower bounds showing that the $4/3$ ratio is tight and that the additive-error dependence is unavoidable, even for randomized algorithms that know the total supply upfront. Finally, we develop a learning-augmented extension to GPA that principally incorporates imperfect forecasts (e.g., from human experts or ML models) commonly available in practice, enabling us to exploit high-quality advice while being robust against arbitrary bad ones. Synthetic and real-world experiments show that GPA outperforms natural baselines with global supply is scarce. 

---
# TeamBench: Evaluating Agent Coordination under Enforced Role Separation 

**Authors**: Yubin Kim, Chanwoo Park, Taehan Kim, Eugene Park, Samuel Schmidgall, Salman Rahman, Chunjong Park, Cynthia Breazeal, Xin Liu, Hamid Palangi, Hae Won Park, Daniel McDuff  

**Link**: [PDF](https://arxiv.org/pdf/2605.07073)  

**Abstract**: Agent systems often decompose a task across multiple roles, but these roles are typically specified by prompts rather than enforced by access controls. Without enforcement, a team pass rate can mask whether agents actually coordinated or whether one role effectively did another role's work. We present TeamBench, a benchmark with 851 task templates and 931 seeded instances for evaluating agent coordination under operating system-enforced role separation. TeamBench separates specification access, workspace editing, and final certification across Planner, Executor, and Verifier roles, so that no role can read the full requirements, modify the workspace, and certify the final answer. Prompt-only and sandbox-enforced teams reach statistically indistinguishable pass rates, but prompt-only runs produce 3.6 times more cases where the verifier attempts to edit the executor's code. Verifiers approve 49% of submissions that fail the deterministic grader, and removing the verifier improves mean partial score in the ablation. Team value is also conditional. Teams benefit when single agents struggle, but hurt when single agents already perform well. A 40-session human study under the same role separation shows that our benchmark exposes interaction patterns that pass rate misses. Solo participants work through the task directly, human participants paired with agents often collapse into quick approval, and human teams spend more effort coordinating missing information across roles. 

---
# 2.5-D Decomposition for LLM-Based Spatial Construction 

**Authors**: Paul Whitten, Li-Jen Chen, Sharath Baddam  

**Link**: [PDF](https://arxiv.org/pdf/2605.07066)  

**Abstract**: Autonomous systems that build structures from natural-language instructions need reliable spatial reasoning, yet large language models (LLMs) make systematic coordinate errors when generating three-dimensional block placements. We present a neuro-symbolic pipeline based on \emph{2.5-D decomposition}: the LLM plans in the two-dimensional horizontal plane while a deterministic executor computes all vertical placement from column occupancy, eliminating an entire class of errors. On the Build What I Mean benchmark (160 rounds), GPT-4o-mini with this pipeline achieves 94.6\% mean structural accuracy across 12 independent runs, within 3.0 percentage points of the 97.6\% ceiling imposed by architect-agent errors that no builder-side improvement can address. This outperforms both GPT-4o at 90.3\% and the best competing system at 76.3\%. A controlled ablation confirms that 2.5-D decomposition is the dominant contributor, accounting for 50.7 percentage points of accuracy. The pipeline transfers directly to edge hardware: Nemotron-3 120B running locally on an NVIDIA Jetson Thor AGX matches the cloud result at 94.5\% with no prompt modifications. The underlying principle, removing deterministic dimensions from the LLM's output space, applies to any autonomous construction or assembly task where gravity or other physical constraints fix one or more degrees of freedom. A transfer experiment on 500 IGLU collaborative building tasks confirm the effect generalizes beyond the primary benchmark. 

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
# Adaptive auditing of AI systems with anytime-valid guarantees 

**Authors**: Siyu Zhou, Patrick Vossler, Venkatesh Sivaraman, Yifan Mai, Jean Feng  

**Link**: [PDF](https://arxiv.org/pdf/2605.07002)  

**Abstract**: A major bottleneck in characterizing the failure modes of generative AI systems is the cost and time of annotation and evaluation. Consequently, adaptive testing paradigms have gained popularity, where one opportunistically decides which cases and how many to annotate based on past results. While this framework is highly practical, its extreme flexibility makes it difficult to draw statistically rigorous conclusions, as it violates classical assumptions: the number of observations is typically limited (often 10 to 50 cases) and decisions regarding sampling and stopping are made in the midst of data collection rather than based a pre-specified rule. To characterize what statistical inferences can be drawn from highly adaptive audits, we introduce a hypothesis testing framework from two 'dueling' perspectives: (i) the model's null that asserts there is no failure mode with performance below a target threshold versus (ii) the auditor's null that asserts they have a sampling strategy that will uncover a failure mode. Leveraging Safe Anytime-Valid Inference (SAVI), we formalize the auditor as conducting 'testing by betting', which translates into simultaneous e-processes for testing the dueling null hypotheses. Furthermore, if the auditor is sufficiently powerful, we prove that these two hypotheses are asymptotically inverses of each other, in that passage of a stringent audit does in fact certify the AI system as being globally robust. Empirically, we demonstrate that our proposed testing procedures maintain anytime-valid type-I error control, outperform pre-specified testing methods, and can reach statistically rigorous conclusions sometimes with as few as 20 observations. 

---
# Optimal Experiments for Partial Causal Effect Identification 

**Authors**: Tobias Maringgele, Jalal Etesami  

**Link**: [PDF](https://arxiv.org/pdf/2605.06993)  

**Abstract**: Causal queries are often only partially identifiable from observational data, and experiments that could tighten the resulting bounds are typically costly. We study the problem of selecting, prior to observing experimental outcomes, a cost-constrained subset of experiments that maximally tightens bounds on a target query. We formalize this as the max-potency problem, where epistemic potency measures the worst-case reduction in bound width guaranteed by an experiment, and show that this problem is NP-hard via a reduction from 0-1 knapsack. Building on the polynomial-programming framework of Duarte et al. (2023), we give a general procedure for evaluating epistemic potency in discrete settings. To control the super-exponential search space, we introduce two graphical pruning criteria that depend only on the causal graph and the query: a novel path-interception rule that exploits district structure to certify zero potency in linear time, and an identifiability check based on the ID algorithm. On Erdos-Renyi random graphs and 11 bnlearn benchmark networks, the two criteria together prune 50-88% of candidate experiments on average without solving a single polynomial program. For the general subset search, we show that ID-pruned experiments are combinatorially inert, yielding a super-exponential reduction in the number of subsets evaluated. We close with an end-to-end demonstration on observational NHANES data, selecting optimal experiments for estimating the effect of physical activity on diabetes. 

---
# Learning and Reusing Policy Decompositions for Hierarchical Generalized Planning with LLM Agents 

**Authors**: Shirin Sohrabi, Haritha Ananthakrishnan, Harsha Kokel, Kavitha Srinivas, Michael Katz  

**Link**: [PDF](https://arxiv.org/pdf/2605.06957)  

**Abstract**: We present a dynamic policy-learning approach that combines generalized planning and hierarchical task decomposition for LLM-based agents. Our method, Hierarchical Component Learning for Generalized Policies (HCL-GP ), learns parameterized policies that generalize across task instances and automatically extracts reusable components from successful executions, organizing them into a component library for compositional policy generation. We address three challenges: (1) learning components through automated decomposition, (2) generalizing components to maximize reuse, and (3) efficient retrieval via semantic search. Evaluated on the AppWorld benchmark, our approach achieves 98.2% accuracy on normal tasks and 97.8% on challenge tasks with unseen applications, improving 15.8 points over static synthesis on challenging scenarios. For open-source models, dynamic reuse enables 62.5% success versus near-zero without reuse. This demonstrates that classical planning concepts can be effectively integrated with LLM agents for improved accuracy and efficiency. 

---
# Multi-Objective Constraint Inference using Inverse reinforcement learning 

**Authors**: Syed Ihtesham Hussain Shah, Floris den Hengst, Aneta Lisowska, Annette ten Teije  

**Link**: [PDF](https://arxiv.org/pdf/2605.06951)  

**Abstract**: Constraint inference is widely considered essential to align reinforcement learning agents with safety boundaries and operational guidelines by observing expert demonstrations. However, existing approaches typically assume homogeneous demonstrations (i.e., generated by a single expert or multiple experts with identical objectives). They also have limited ability to capture individual preferences and often suffer from computational inefficiencies. In this paper, we introduce Multi-Objective Constraint Inference (MOCI), a novel framework designed to jointly extract shared constraints and individual preferences from heterogeneous expert trajectories, where multiple experts pursue different objectives. MOCI effectively models and learns from diverse, and potentially conflicting, behaviors. Empirical evaluations demonstrate that MOCI significantly outperforms existing baselines, achieving improved predictive performance, and maintaining competitive computational efficiency on a standard grid-world benchmark. These results establish MOCI as an accurate, flexible, and computationally practical approach for real-world constraint inference and preference learning tasks. 

---
# Self-Programmed Execution for Language-Model Agents 

**Authors**: Luke J. O'Connor  

**Link**: [PDF](https://arxiv.org/pdf/2605.06898)  

**Abstract**: At the heart of existing language model agents is a fixed orchestrator program responsible for the state transition between consecutive turns. This paper introduces self-programmed execution (SPE), an agent architecture in which the model completion is itself the orchestrator program, and the harness evaluates this program but does not impose its own orchestration policy. I formalize this idea using agentic machines: an SPE state is one from which a model completion can load any state of an embedded copy of the machine, meaning that it is subject to no fixed turn-to-turn orchestration policy. Realizing SPE in practice is nontrivial because the same data is both model context and executable program. I therefore introduce Spell, a Lisp-based language in which programs can edit and re-evaluate themselves, and effectful expressions like model invocations are structured such that re-evaluating an edited program does not replay its side effects. Experiments with existing models, not trained for SPE or Spell, show that frontier models can operate in this regime and accomplish challenging agentic tasks. These results demonstrate how an LM can act as an agent without any fixed orchestration policy, and they raise the question of what self-orchestration strategies might be learned by a model trained for self-programmed execution. Code is available at this https URL . 

---
# Mitigating Cognitive Bias in RLHF by Altering Rationality 

**Authors**: Tiffany Horter, Andrew Markham, Niki Trigoni, Serena Booth  

**Link**: [PDF](https://arxiv.org/pdf/2605.06895)  

**Abstract**: How can we make models robust to even imperfect human feedback? In reinforcement learning from human feedback (RLHF), human preferences over model outputs are used to train a reward model that assigns scalar values to responses. Because these rewards are inferred from pairwise comparisons, this learning depends on an assumed relationship between latent reward differences and observed preferences, typically modeled using a Boltzmann formulation in which a rationality parameter beta informs how consistently preferences reflect reward differences. In practice, beta is typically treated as a fixed constant that reflects assumed uniform annotator reliability. However, human feedback is not this simplistic in practice: real human judgments are shaped by cognitive biases, leading to systematic deviations from reward-consistent behavior that arise contextually. To address this, we treat rationality as context- and annotation-dependent. We design an approach to dynamically adjust the rationality parameter beta during reward learning using an LLM-as-judge to assess the likely presence of cognitive biases. This approach effectively downweights comparisons that are likely to reflect biased or unreliable judgments. Empirically, we show that this approach learns a more rational downstream model, even when finetuning on datasets with strongly biased preferences. 

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
# Agentick: A Unified Benchmark for General Sequential Decision-Making Agents 

**Authors**: Roger Creus Castanyer, Pablo Samuel Castro, Glen Berseth  

**Link**: [PDF](https://arxiv.org/pdf/2605.06869)  

**Abstract**: AI agent research spans a wide spectrum: from RL agents that learn from scratch to foundation model agents that leverage pre-trained knowledge, yet no unified benchmark enables fair comparison across these approaches. We present Agentick, a benchmark for sequential decision-making agents designed to evaluate RL, LLM, VLM, hybrid, and human agents on common ground and to power research on the fundamental challenges of sequential decision-making. Agentick provides 37 procedurally generated tasks across six capability categories, four difficulty levels, and five observation modalities, all exposed through a single Gymnasium-compatible interface. The benchmark ships with a Coding API, oracle reference policies for all tasks, pre-built SFT datasets, a composable agent harness, and a live leaderboard. An evaluation spanning 27 configurations and over 90,000 episodes reveals that no single approach dominates: GPT-5 mini leads overall at 0.309 oracle-normalized score while PPO dominates planning and multi-agent tasks; the reasoning harness multiplies LLM performance by 3-10x; and ASCII observations consistently outperform natural language. These findings highlight the substantial room for improvement that remains across all agent paradigms. Agentick's capability-decomposed, multi-modal design provides the empirical infrastructure needed to drive progress toward general autonomous agents, both as an evaluation framework and as a training ground for RL post-training of foundation models in truly sequential environments. 

---
# AGWM: Affordance-Grounded World Models for Environments with Compositional Prerequisites 

**Authors**: Qinshi Zhang, Weipeng Deng, Zhihan Jiang, Jiaming Qu, Qianren Li, Weitao Xu, Ray LC  

**Link**: [PDF](https://arxiv.org/pdf/2605.06841)  

**Abstract**: In model-based learning, the agent learns behaviors by simulating trajectories based on world model predictions. Standard world models typically learn a stationary transition function that maps states and actions to next states, when an action and an outcome frequently co-occur in training data, the model tends to internalize this correlation as a general causal rule while ignoring action preconditions. In interactive environments, however, agent actions can reshape the future affordance space. At each timestep, an action may becomes executable only after its prerequisites are met, or non-executable when they are destroyed. We term such events structure-changing events (SC events). As a result, a conventional world model often fails to determine whether a given action is executable in the current state, especially in multi-step predictions. Each imagined step is conditioned on an incorrect affordance state, and therefore the prediction error compounds over the rollout horizon. In this paper, we propose AGWM (Affordance-Grounded World Model), which learns an abstract affordance structure represented as a DAG of prerequisite dependencies to explicitly track the dynamic executability of actions. Experiments on game-based simulated environments demonstrate the effectiveness of our method by achieving lower multi-step prediction error, better generalization to novel configurations, and improved interpretability. 

---
# Extracting Search Trees from LLM Reasoning Traces Reveals Myopic Planning 

**Authors**: Sixing Chen, Ji-An Li, Saner Cakir, Sinan Akcali, Kayla Lee, Marcelo G. Mattar  

**Link**: [PDF](https://arxiv.org/pdf/2605.06840)  

**Abstract**: Large language models (LLMs), especially reasoning models, generate extended chain-of-thought (CoT) reasoning that often contains explicit deliberation over future outcomes. Yet whether this deliberation constitutes genuine planning, how it is structured, and what aspects of it drive performance remain poorly understood. In this work, we introduce a new method to characterize LLM planning by extracting and quantifying search trees from reasoning traces in the four-in-a-row board game. By fitting computational models on the extracted search trees, we characterize how plans are structured and how they influence move decisions. We find that LLMs' search is shallower than humans', and that performance is predicted by search breadth rather than depth. Most strikingly, although LLMs expand deep nodes in their traces, their move choices are best explained by a myopic model that ignores those nodes entirely. A causal intervention study where we selectively prune CoT paragraphs further suggests that move selection is driven predominantly by shallow rather than deep nodes. These patterns contrast with human planning, where performance is driven primarily by deep search. Together, our findings reveal a key difference between LLM and human planning: while human expertise is driven by deeper search, LLMs do not act on deep lookahead. This dissociation offers targeted guidance for aligning LLM and human planning. More broadly, our framework provides a generalizable approach for interpreting the structure of LLM planning across strategic domains. 

---
# Randomness is sometimes necessary for coordination 

**Authors**: Rohan Patil, Jai Malegaonkar, Henrik I. Christensen  

**Link**: [PDF](https://arxiv.org/pdf/2605.06825)  

**Abstract**: Full parameter sharing is standard in cooperative multi-agent reinforcement learning (MARL) for homogeneous agents. Under permutation-symmetric observations, however, a shared deterministic policy outputs identical action distributions for every agent, making role differentiation impossible. This failure can theoretically be resolved using symmetry breaking among anonymous identical processors, which requires randomness. We propose Diamond Attention, a cross-attention architecture in which each agent samples a scalar random number per timestep, inducing a transient rank ordering that masks lower-ranked peers from agent-to-agent attention while leaving task attention fully unmasked. This realizes a random-bit coordination protocol in a single broadcast round, and the set-based attention enables zero-shot deployment to teams of different sizes. We evaluate across three regimes that isolate when structured randomness matters. On the perfectly symmetric XOR game, our method achieves $1.0$ success while all deterministic baselines plateau near $0.5$. On control coordination tasks, a policy trained on $N=4$ generalizes zero-shot to $N \in [2,8]$. On SMACLite cross-scenario transfer, we achieve zero-shot transfer where standard baselines cannot transfer due to structural limitations. Furthermore, replacing the structured mask with standard dropout-based randomness results in a 0\% win rate, confirming that protocol-space structure, not stochastic noise, is the operative ingredient. this https URL 

---
# Uneven Evolution of Cognition Across Generations of Generative AI Models 

**Authors**: Isaac Galatzer-Levy, Daniel McDuff, Xin Liu, Jed McGiffin  

**Link**: [PDF](https://arxiv.org/pdf/2605.06815)  

**Abstract**: The pursuit of artificial general intelligence necessitates robust methods for evaluating the cognitive capabilities of models beyond narrow task performance. Here, we introduce a psychometric framework to assess the cognitive profiles of generative AI, comparing them to human norms and tracking their evolution across generations. Initial evaluation of leading multimodal models using tasks adapted from the Wechsler Adult Intelligence Scale revealed a profoundly uneven cognitive architecture: near-ceiling performance in verbal comprehension and working memory (>$98^{\text{th}}$ percentile) contrasted with near-floor performance in perceptual reasoning (<$1^{\text{st}}$ percentile). To track developmental trajectories beyond human-normed limits, we developed the Artificial Intelligence Quotient (AIQ) Benchmark and applied it to six generations and two model families, revealing significant but asymmetric performance gains. Notably, we uncovered a sharp dissociation between modalities; abstract quantitative reasoning matured far more rapidly when presented linguistically compared to a visually analogous format, indicating an architectural bias towards language-based symbolic manipulation. While abstract visual reasoning improved, visual-perceptual organization remained largely stagnant. Collectively, these findings demonstrate that the cognitive abilities of generative models are evolving unevenly, suggesting that scaling and optimization approaches to AGI development alone may be insufficient to overcome fundamental architectural limitations in achieving balanced, human-like general intelligence. 

---
# Towards Security-Auditable LLM Agents: A Unified Graph Representation 

**Authors**: Chaofan Li, Lyuye Zhang, Jintao Zhai, Siyue Feng, Xichun Yang, Huahao Wang, Shihan Dou, Yu Ji, Yutao Hu, Yueming Wu, Yang Liu, Deqing Zou  

**Link**: [PDF](https://arxiv.org/pdf/2605.06812)  

**Abstract**: LLM-based agentic systems are rapidly evolving to perform complex autonomous tasks through dynamic tool invocation, stateful memory management, and multi-agent collaboration. However, this semantics-driven execution paradigm creates a severe semantic gap between low-level physical events and high-level execution intent, making post-hoc security auditing fundamentally difficult. Existing representation mechanisms, including static SBOMs and runtime logs, provide only fragmented evidence and fail to capture cognitive-state evolution, capability bindings, persistent memory contamination, and cascading risk propagation across interacting agents. To bridge this gap, we propose Agent-BOM, a unified structural representation for agent security auditing. Agent-BOM models an agentic system as a hierarchical attributed directed graph that separates static capability bases, such as models, tools, and long-term memory, from dynamic runtime semantic states, such as goals, reasoning trajectories, and actions. These layers are connected through semantic edges and security attributes, transforming fragmented execution traces into queryable audit paths. Building on Agent-BOM, we develop a graph-query-based paradigm for path-level risk assessment and instantiate it with the OWASP Agentic Top 10. We further implement an auditing plugin in the OpenClaw environment to construct Agent-BOM from live executions. Evaluation on representative real-world agentic attack scenarios shows that Agent-BOM can reconstruct stealthy attack chains, including cross-session memory poisoning and tool misuse, capability supply-chain hijacking and unexpected code execution, multi-agent ecosystem hijacking, and privilege and trust abuse. These results demonstrate that Agent-BOM provides a unified and auditable foundation for root-cause analysis and security adjudication in complex agentic ecosystems. 

---
# When Does Critique Improve AI-Assisted Theoretical Physics? SCALAR: Structured Critic--Actor Loop for Agentic Reasoning 

**Authors**: Vasilis Niarchos, Constantinos Papageorgakis, Alexander G. Stapleton, Sokratis Trifinopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2605.06772)  

**Abstract**: As large language models (LLMs) show increasing promise on research-level physics reasoning tasks and agentic AI becomes more common, a practical question emerges: How does the interaction between researchers and agents affect the results? We study this using SCALAR (Structured Critic--Actor Loop for AI Reasoning), an Actor--Critic--Judge pipeline applied to quantum field theory and string theory problems. The Actor proposes solutions, the Critic provides iterative feedback, and an independent Judge evaluates the transcript against reference solutions. We vary the Actor persona, the Critic feedback strategy, and the Actor model family and scale. Multi-turn dialogue improves over single-shot attempts throughout, but both the mechanism of improvement and the value of different prompting choices depend strongly on the Actor--Critic pairing. Increasing the scale within one model family (e.g. from the 8B-parameter DeepSeek-R1 variant to DeepSeek-R1 70B) improves some easier-problem behavior, but does not remove the hardest bottleneck we observe. Critic feedback strategy matters most clearly in the asymmetric Actor--Critic setting (e.g., a lightweight Haiku Actor guided by a stronger Sonnet Critic), where constructive feedback improves mean-score outcomes. In same-family Actor--Critic settings, strategy effects are weaker: lenient feedback is sometimes favored, while strict and adversarial feedback are not beneficial. Taken together, SCALAR provides a controlled testbed for evaluating which interaction structures help or hinder AI-driven scientific discovery. 

---
# Weblica: Scalable and Reproducible Training Environments for Visual Web Agents 

**Authors**: Oğuzhan Fatih Kar, Roman Bachmann, Yuanzheng Gong, Anders Boesen Lindbo Larsen, Afshin Dehghan  

**Link**: [PDF](https://arxiv.org/pdf/2605.06761)  

**Abstract**: The web is complex, open-ended, and constantly changing, making it challenging to scale training data for visual web agents. Existing data collection attempts remain limited to offline trajectories for supervised fine-tuning or a handful of simulated environments for RL training, thus failing to capture web diversity. We propose Weblica (Web Replica), a framework for constructing reproducible and scalable web environments. Our framework leverages 1) HTTP-level caching to capture and replay stable visual states while preserving interactive behavior and 2) LLM-based environment synthesis grounded in real-world websites and core web navigation skills. Using this framework, we scale RL training to thousands of diverse environments and tasks. Our best model, Weblica-8B, outperforms open-weight baselines of similar size across multiple web navigation benchmarks while using fewer inference steps, scales favorably with additional test-time compute, and is competitive with API models. 

---
# When Does a Language Model Commit? A Finite-Answer Theory of Pre-Verbalization Commitment 

**Authors**: Long Zhang, Wei-neng Chen, Feng-feng Wei, Zi-bo Qin  

**Link**: [PDF](https://arxiv.org/pdf/2605.06723)  

**Abstract**: Language models often generate reasoning before giving a final answer, but the visible answer does not reveal when the model's answer preference became stable. We study this question through a narrow computable object: \emph{finite-answer preference stabilization}. For a model state and specified answer verbalizers, we project the model's own continuation probabilities onto a finite answer set; in binary tasks this yields an exact log-odds code, $\delta(\xi)=S_\theta(\mathrm{yes}\mid\xi)-S_\theta(\mathrm{no}\mid\xi)$. This target defines parser-based answer onset, retrospective stabilization time, and lead without relying on greedy rollouts or learned probes. In controlled delayed-verdict tasks with Qwen3-4B-Instruct, the contextual finite-answer projection stabilizes before the answer is parseable, with 17--31 token mean lead in the main templates and positive, shorter lead in a parser-clean replication. The signal tracks the model's eventual output rather than truth, is linearly recoverable from compact hidden summaries, is partly separable from cursor progress, and transfers as shared information without a single invariant coordinate. Diagnostics separate the measurement from online stopping, verbalizer-free belief, and causal answer control; exact steering shows local sensitivity of $\delta$ but not reliable generation control. 

---
# From Storage to Experience: A Survey on the Evolution of LLM Agent Memory Mechanisms 

**Authors**: Jinghao Luo, Yuchen Tian, Chuxue Cao, Ziyang Luo, Hongzhan Lin, Kaixin Li, Chuyi Kong, Ruichao Yang, Jing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.06716)  

**Abstract**: Large Language Model (LLM)-based agents have fundamentally reshaped artificial intelligence by integrating external tools and planning capabilities. While memory mechanisms have emerged as the architectural cornerstone of these systems, current research remains fragmented, oscillating between operating system engineering and cognitive science. This theoretical divide prevents a unified view of technological synthesis and a coherent evolutionary perspective. To bridge this gap, this survey proposes a novel evolutionary framework for LLM agent memory mechanisms, formalizing the development process into three stages: Storage (trajectory preservation), Reflection (trajectory refinement), and Experience (trajectory abstraction). We first formally define these three stages before analyzing the three core drivers of this evolution: the necessity for long-range consistency, the challenges in dynamic environments, and the ultimate goal of continual learning. Furthermore, we specifically explore two transformative mechanisms in the frontier Experience stage: proactive exploration and cross-trajectory abstraction. By synthesizing these disparate views, this work offers robust design principles and a clear roadmap for the development of next-generation LLM agents. 

---
# CASCADE: Case-Based Continual Adaptation for Large Language Models During Deployment 

**Authors**: Siyuan Guo, Yali Du, Hechang Chen, Yi Chang, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06702)  

**Abstract**: Large language models (LLMs) have become a central foundation of modern artificial intelligence, yet their lifecycle remains constrained by a rigid separation between training and deployment, after which learning effectively ceases. This limitation contrasts with natural intelligence, which continually adapts through interaction with its environment. In this paper, we formalise deployment-time learning (DTL) as the third stage in the LLM lifecycle that enables LLM agents to improve from experience during deployment without modifying model parameters. We present CASCADE (CASe-based Continual Adaptation during DEployment), a general and principled framework that equips LLM agents with an explicit, evolving episodic memory. CASCADE formulates experience reuse as a contextual bandit problem, enabling principled exploration-exploitation trade-offs and establishing no-regret guarantees over long-term interactions. This design allows agents to accumulate, select, and refine task-relevant cases, transforming past experience into actionable knowledge. Across 16 diverse tasks spanning medical diagnosis, legal analysis, code generation, web search, tool use, and embodied interaction, CASCADE improves macro-averaged success rate by 20.9% over zero-shot prompting while consistently outperforming gradient-based and memory-based baselines. By reframing deployment as an adaptive learning process, this work establishes a foundation for continually improving AI systems. 

---
# Hidden Coalitions in Multi-Agent AI: A Spectral Diagnostic from Internal Representations 

**Authors**: Cameron Berg, Susan L. Schneider, Mark M. Bailey  

**Link**: [PDF](https://arxiv.org/pdf/2605.06696)  

**Abstract**: Collections of interacting AI agents can form coalitions, creating emergent group-level organization that is critical for AI safety and alignment. However, observing agent behavior alone is often insufficient to distinguish genuine informational coupling from spurious similarity, as consequential coalitions may form at the level of internal representations before any overt behavioral change is apparent. Here, we introduce a practical method for detecting coalition structure from the internal neural representations of multi-agent systems. The approach constructs a pairwise mutual-information graph from the hidden states of agents and applies spectral partitioning to identify the most salient coalition boundary.
We validate this method in two domains. First, in multi-agent reinforcement learning environments, the method successfully recovers programmed hierarchical and dynamic coalition structures and correctly rejects false positives arising from behavioral coordination without informational coupling. Second, using a large language model, the method identifies coalition structures implied by descriptive prompts, tracks dynamic team reassignments, and reveals a representational hierarchy where explicit labels dominate over conflicting interaction patterns. Across both settings, the recovered partition reveals subgroup organization that a scalar cross-agent mutual-information measure cannot distinguish. The results demonstrate that analyzing hidden-state mutual information through spectral partitioning provides a scalable diagnostic for identifying representational coalitions, offering a valuable tool for monitoring emergent structure in distributed AI systems. 

---
# State Representation and Termination for Recursive Reasoning Systems 

**Authors**: Debashis Guha, Amritendu Mukherjee, Sanjay Kukreja, Tarun Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2605.06690)  

**Abstract**: Recursive reasoning systems alternate between acquiring new evidence and refining an accumulated understanding. Two design choices are typically left implicit: how to represent the evolving reasoning state, and when to stop iterating. This paper addresses both. We represent the reasoning state as an epistemic state graph encoding extracted claims, evidential relations, open questions, and confidence weights. We define the order-gap as the distance between the states reached by expand-then-consolidate versus consolidate-then-expand; a small order-gap suggests that the two orderings agree and further iteration is unlikely to help. Our main result gives a necessary and sufficient condition for the linearised order-gap to be non-degenerate near the fixed point, showing when the criterion is informative rather than algebraically vacuous. This is a local condition, not a global convergence guarantee. We apply the framework to recursive reasoning systems and sketch its application to agent loops, tree-of-thought reasoning, theorem proving, and continual learning. 

---
# Fast and Effective Redistricting Optimization via Composite-Move Tabu Search 

**Authors**: Hai Jin, Diansheng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.06682)  

**Abstract**: Spatial redistricting is a practical combinatorial optimization problem that demands high-quality solutions, rapid turnaround, and flexibility to accommodate multi-criteria objectives and interactive refinement. A central challenge is the contiguity constraint: enforcing contiguity in integer-programming or heuristic search can severely shrink the feasible neighborhood, weaken exploration, and trap the search in poor local optima. We introduce a composite-move Tabu search (CM-Tabu) that systematically expands the feasible neighborhood space in Tabu search while preserving contiguity. When a boundary unit cannot be reassigned individually without disconnecting its district, our method identifies a minimal set of units that can move together, or a pair of units (or sets of units) that can be switched, as a contiguity-preserving composite move. Candidate single-unit and composite moves are generated in linear time by analyzing each district's contiguity graph using articulation points and biconnected components. Extensive experiments demonstrate that the proposed approach substantially improves solution quality, run-to-run robustness, and computational efficiency relative to traditional Tabu search and other baselines. For example, in the Philadelphia case, the approach can consistently attain the theoretical global optimum in population-equality and support multi-criteria trade-offs. CM-Tabu delivers optimization performance suitable for real-world practices and decision-support workflows. 

---
# More Thinking, More Bias: Length-Driven Position Bias in Reasoning Models 

**Authors**: Xiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06672)  

**Abstract**: Chain-of-thought (CoT) reasoning and reasoning-tuned models such as DeepSeek-R1 are commonly assumed to reduce shallow heuristic biases by thinking carefully. We test this on position bias in multiple-choice QA and find a different story: within any reasoning-capable model, per-question position bias scales with the length of the reasoning trajectory.
Across thirteen reasoning-mode configurations (two R1-distilled 7-8B models, two base models prompted with CoT, and DeepSeek-R1 at 671B) on MMLU, ARC-Challenge, and GPQA, twelve show a positive partial correlation between trajectory length and Position Bias Score (PBS) after controlling for accuracy, ranging from 0.11 to 0.41 (all p < 0.05). All twelve open-weight reasoning-mode configurations show monotonically increasing PBS across length quartiles. A truncation intervention provides causal evidence: continuations resumed from later points in the trajectory are increasingly likely to shift toward position-preferred options (16% to 32% for R1-Qwen-7B across absolute-position buckets).
At 671B, aggregate PBS collapses to 0.019, but the length effect still manifests in the longest quartile (PBS = 0.071), suggesting that accuracy gates the expression of length-driven bias rather than eliminating the underlying mechanism. We additionally find that direct-answer position bias is a distinct phenomenon with a different footprint (strong in Llama-Instruct-direct, weak in Qwen-Instruct-direct, and uncorrelated with trajectory length): CoT reasoning replaces this baseline bias with length-accumulated bias.
Our results argue that reasoning-capable models should not be treated as order-robust by default in MCQ evaluation pipelines, and offer a diagnostic toolkit (PBS, commitment change point, effective switching, truncation probes) for auditing position bias in reasoning models. 

---
# GraphDC: A Divide-and-Conquer Multi-Agent System for Scalable Graph Algorithm Reasoning 

**Authors**: Wenjin Li, Jiaming Cui  

**Link**: [PDF](https://arxiv.org/pdf/2605.06671)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong potential for many mathematical problems. However, their performance on graph algorithmic tasks is still unsatisfying, since graphs are naturally more complex in topology and often require systematic multi-step reasoning, especially on larger graphs. Motivated by this gap, we propose GraphDC, a Divide-and-Conquer multi-agent framework for scalable graph algorithm reasoning. Specifically, inspired by Divide-and-Conquer design, GraphDC decomposes an input graph into smaller subgraphs, assigns each subgraph to a specialized agent for local reasoning, and uses a master agent to integrate the local outputs with inter-subgraph information to produce the final solution. This hierarchical design reduces the reasoning burden on individual agents, alleviates computational bottlenecks, and improves robustness on large graph instances. Extensive experiments show that GraphDC consistently outperforms existing methods on graph algorithm reasoning across diverse tasks and scales, especially on larger instances where direct end-to-end reasoning is less reliable. 

---
# EmambaIR: Efficient Visual State Space Model for Event-guided Image Reconstruction 

**Authors**: Wei Yu, Yunhang Qian  

**Link**: [PDF](https://arxiv.org/pdf/2605.08073)  

**Abstract**: Recent event-based image reconstruction methods predominantly rely on Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) to process complementary event information. However, these architectures face fundamental limitations: CNNs often fail to capture global feature correlations, whereas ViTs incur quadratic computational complexity (e.g., $O(n^2)$), hindering their application in high-resolution scenarios. To address these bottlenecks, we introduce EmambaIR, an Efficient visual State Space Model designed for image reconstruction using spatially sparse and temporally continuous event streams. Our framework introduces two key components: the cross-modal Top-k Sparse Attention Module (TSAM) and the Gated State-Space Module (GSSM). TSAM efficiently performs pixel-level top-k sparse attention to guide cross-modal interactions, yielding rich yet sparse fusion features. Subsequently, GSSM utilizes a nonlinear gated unit to enhance the temporal representation of vanilla linear-complexity ($O(n)$) SSMs, effectively capturing global contextual dependencies without the typical computational overhead. Extensive experiments on six datasets across three diverse image reconstruction tasks - motion deblurring, deraining, and High Dynamic Range (HDR) enhancement - demonstrate that EmambaIR significantly outperforms state-of-the-art methods while offering substantial reductions in memory consumption and computational cost. The source code and data are publicly available at: this https URL 

---
# Flow-OPD: On-Policy Distillation for Flow Matching Models 

**Authors**: Zhen Fang, Wenxuan Huang, Yu Zeng, Yiming Zhao, Shuang Chen, Kaituo Feng, Yunlong Lin, Lin Chen, Zehui Chen, Shaosheng Cao, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.08063)  

**Abstract**: Existing Flow Matching (FM) text-to-image models suffer from two critical bottlenecks under multi-task alignment: the reward sparsity induced by scalar-valued rewards, and the gradient interference arising from jointly optimizing heterogeneous objectives, which together give rise to a 'seesaw effect' of competing metrics and pervasive reward hacking. Inspired by the success of On-Policy Distillation (OPD) in the large language model community, we propose Flow-OPD, the first unified post-training framework that integrates on-policy distillation into Flow Matching models. Flow-OPD adopts a two-stage alignment strategy: it first cultivates domain-specialized teacher models via single-reward GRPO fine-tuning, allowing each expert to reach its performance ceiling in isolation; it then establishes a robust initial policy through a Flow-based Cold-Start scheme and seamlessly consolidates heterogeneous expertise into a single student via a three-step orchestration of on-policy sampling, task-routing labeling, and dense trajectory-level supervision. We further introduce Manifold Anchor Regularization (MAR), which leverages a task-agnostic teacher to provide full-data supervision that anchors generation to a high-quality manifold, effectively mitigating the aesthetic degradation commonly observed in purely RL-driven alignment. Built upon Stable Diffusion 3.5 Medium, Flow-OPD raises the GenEval score from 63 to 92 and the OCR accuracy from 59 to 94, yielding an overall improvement of roughly 10 points over vanilla GRPO, while preserving image fidelity and human-preference alignment and exhibiting an emergent 'teacher-surpassing' effect. These results establish Flow-OPD as a scalable alignment paradigm for building generalist text-to-image models. 

---
# The Memory Curse: How Expanded Recall Erodes Cooperative Intent in LLM Agents 

**Authors**: Jiayuan Liu, Tianqin Li, Shiyi Du, Xin Luo, Haoxuan Zeng, Emanuel Tewolde, Tai Sing Lee, Tonghan Wang, Carl Kingsford, Vincent Conitzer  

**Link**: [PDF](https://arxiv.org/pdf/2605.08060)  

**Abstract**: Context window expansion is often treated as a straightforward capability upgrade for LLMs, but we find it systematically fails in multi-agent social dilemmas. Across 7 LLMs and 4 games over 500 rounds, expanding accessible history degrades cooperation in 18 of 28 model--game settings, a pattern we term the memory curse. We isolate the underlying mechanism through three analyses. First, lexical analysis of 378,000 reasoning traces associates this breakdown with eroding forward-looking intent rather than rising paranoia. We validate this using targeted fine-tuning as a cognitive probe: a LoRA adapter trained exclusively on forward-looking traces mitigates the decay and transfers zero-shot to distinct games. Second, memory sanitization holds prompt length fixed while replacing visible history with synthetic cooperative records, which restores cooperation substantially, proving the trigger is memory content, not length alone. Finally, ablating explicit Chain-of-Thought reasoning often reduces the collapse, showing that deliberation paradoxically amplifies the memory curse. Together, these results recast memory as an active determinant of multi-agent behavior: longer recall can either destabilize or support cooperation depending on the reasoning patterns it elicits. 

---
# CA-SQL: Complexity-Aware Inference Time Reasoning for Text-to-SQL via Exploration and Compute Budget Allocation 

**Authors**: James Petullo, Nianwen Xue  

**Link**: [PDF](https://arxiv.org/pdf/2605.08057)  

**Abstract**: While recent advancements in inference-time learning have improved LLM reasoning on Text-to-SQL tasks, current solutions still struggle to perform well on the most challenging tasks in the Bird-Bench (BIRD) benchmark. This is due to inadequate solution space exploration, which is necessary to uncover promising candidate queries that can be further refined to produce the correct output. To address this challenge, we introduce CA-SQL, a novel Text-to-SQL pipeline that utilizes the estimated difficulty of a task to dynamically scale the breadth of the exploration for generating solution candidates. In addition, we use a custom prompt seeding method, based on principles of evolutionary search, to further elicit exploratory behavior from the base LLM and a novel voting method to select the best candidate solution at the end of the search. Experiments demonstrate that our solution achieves a state-of-the-art score of 51.72% on the "challenging" tier of BIRD development set problems, using only GPT-4o-mini, out-performing other in-context learning approaches, even those that leverage larger models. Overall, our method attains a competitive 61.06% execution accuracy and 68.77% Soft F1 score on the BIRD development dataset. 

---
# Fast Byte Latent Transformer 

**Authors**: Julie Kallini, Artidoro Pagnoni, Tomasz Limisiewicz, Gargi Ghosh, Luke Zettlemoyer, Christopher Potts, Xiaochuang Han, Srinivasan Iyer  

**Link**: [PDF](https://arxiv.org/pdf/2605.08044)  

**Abstract**: Recent byte-level language models (LMs) match the performance of token-level models without relying on subword vocabularies, yet their utility is limited by slow, byte-by-byte autoregressive generation. We address this bottleneck in the Byte Latent Transformer (BLT) through new training and generation techniques. First, we introduce BLT Diffusion (BLT-D), a new model and our fastest BLT variant, trained with an auxiliary block-wise diffusion objective alongside the standard next-byte prediction loss. This enables an inference procedure that generates multiple bytes in parallel per decoding step, substantially reducing the number of forward passes required to generate a sequence. Second, we propose two extensions inspired by speculative decoding that trade some of this speed for higher generation quality: BLT Self-speculation (BLT-S), in which BLT's local decoder continues generating past its normal patch boundaries to draft bytes, which are then verified with a single full-model forward pass; and BLT Diffusion+Verification (BLT-DV), which augments BLT-D with an autoregressive verification step after diffusion-based generation. All methods may achieve an estimated memory-bandwidth cost over 50% lower than BLT on generation tasks. Each approach offers its own unique advantages, together removing key barriers to the practical use of byte-level LMs. 

---
# SCOPE: Structured Decomposition and Conditional Skill Orchestration for Complex Image Generation 

**Authors**: Tianfei Ren, Zhipeng Yan, Yiming Zhao, Zhen Fang, Yu Zeng, Guohui Zhang, Hang Xu, Xiaoxiao Ma, Shiting Huang, Ke Xu, Wenxuan Huang, Lionel Z. Wang, Lin Chen, Zehui Chen, Jie Huang, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.08043)  

**Abstract**: While text-to-image models have made strong progress in visual fidelity, faithfully realizing complex visual intents remains challenging because many requirements must be tracked across grounding, generation, and verification. We refer to these requirements as semantic commitments and formalize their lifecycle discontinuity as the Conceptual Rift, where commitments may be locally resolved or checked but fail to remain identifiable as the same operational units throughout the generation lifecycle. To address this, we propose SCOPE, a specification-guided skill orchestration framework that maintains semantic commitments in an evolving structured specification and conditionally invokes retrieval, reasoning, and repair skills around unresolved or violated commitments. To evaluate commitment-level intent realization, we introduce Gen-Arena, a human-annotated benchmark with entity- and constraint-level specifications, together with Entity-Gated Intent Pass Rate (EGIP), a strict entity-first pass criterion. SCOPE substantially outperforms all evaluated baselines on Gen-Arena, achieving 0.60 EGIP, and further achieves strong results on WISE-V (0.907) and MindBench (0.61), demonstrating the effectiveness of persistent commitment tracking for complex image generation. 

---
# Beyond Pairs: Your Language Model is Secretly Optimizing a Preference Graph 

**Authors**: Ning Liu, Chuanneng Sun, Kristina Klinkner, Shervin Malmasi  

**Link**: [PDF](https://arxiv.org/pdf/2605.08037)  

**Abstract**: Direct Preference Optimization (DPO) aligns language models using pairwise preference comparisons, offering a simple and effective alternative to Reinforcement Learning (RL) from human feedback. However, in many practical settings, training data consists of multiple rollouts per prompt, inducing rich preference structure that pairwise DPO fails to exploit. Collapsing such data into independent pairs discards transitivity, introduces redundant or conflicting supervision, and can lead to unstable optimization. We propose Graph Direct Preference Optimization (GraphDPO), a principled generalization of DPO that operates over directed acyclic preference graphs induced by rollout rankings. GraphDPO encodes dominance relations as edges and optimizes a graph-structured Plackett--Luce-inspired objective that aggregates supervision over graph neighborhoods, enforcing transitivity while recovering standard DPO as a special case. To handle discrete or sparse signals, we introduce an equivalence-class construction where responses with identical preferences form graph layers, and intra-layer edges contribute zero loss, preventing spurious gradients. Despite leveraging full graph structure, GraphDPO maintains linear per-prompt complexity via efficient log-sum-exp aggregation. We further incorporate optional ground-truth anchoring by inserting verified solutions as dominant nodes and applying an annealed schedule that stabilizes early training while gradually relaxing oracle supervision. Experiments on reasoning and program synthesis tasks demonstrate superior performance, suggesting that graph-structured preference modeling is a scalable and robust alternative to pairwise and listwise alignment objectives. 

---
# Globally Optimal Training of Spiking Neural Networks via Parameter Reconstruction 

**Authors**: Himanshu Udupi, Xiaocong Yang, ChengXiang Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2605.08022)  

**Abstract**: Spiking Neural Networks (SNNs) have been proposed as biologically plausible and energy-efficient alternatives to conventional Artificial Neural Networks (ANNs). However, the training of SNN usually relies on surrogate gradients due to the non-differentiability of the spike function, introducing approximation errors that accumulate across layers. To address this challenge, we extend the work on convexification of parallel feedforward threshold networks to parallel recurrent threshold networks, which subsume parallel SNNs as a structured special case. Building on this theoretical framework, we propose a parameter reconstruction algorithm for SNN training that demonstrates consistent and significant advantages across various tasks, both as a standalone method and in combination with surrogate-gradient training. The ablations further demonstrate the data scalability and robustness to model configurations of our training algorithm, pointing toward its potential in large-scale SNN training. 

---
# Position: Mechanistic Interpretability Must Disclose Identification Assumptions for Causal Claims 

**Authors**: Zezheng Lin, Fengming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.08012)  

**Abstract**: Mechanistic interpretability papers increasingly use causal vocabulary: circuits, mediators, causal abstraction, monosemanticity. Such claims require explicit identification assumptions. A purposive audit of 10 papers across four methodological strands finds no dedicated identification-assumptions section and a recurring pattern: validation metrics such as faithfulness, completeness, monosemanticity, alignment, or ablation effects are reported as causal support without stating the assumptions that make them identifying. A two-human-coder audit on $n=30$ reproduces the direction of the main finding: dedicated identification sections are absent, and validation-metric substitution is common, though exact Dim B/D counts are coding-rule sensitive. The paper proposes a disclosure norm: state whether the claim is causal, name the identification strategy, enumerate assumptions, stress at least one, and explain how conclusions shift if assumptions fail. Validation is not identification. 

---
# Graph-Structured Hyperdimensional Computing for Data-Efficient and Explainable Process-Structure-Property Prediction 

**Authors**: Jingzhan Ge, Ajeeth Vellore, Ajinkya Palwe, Ahsan Khan, David Gorsich, Matthew P. Castanier, SeungYeon Kang, Farhad Imani  

**Link**: [PDF](https://arxiv.org/pdf/2605.07999)  

**Abstract**: Multiphoton photoreduction enables high-fidelity fabrication of complex 3D microstructures, yet reliable process-structure-property (PSP) prediction remains difficult because the available data are sparse, heterogeneous, and interaction-dominated. In this regime, conventional feature-vector models are statistically underdetermined, making them prone to spurious correlations, poor regime transfer, and unstable post hoc explanations, whereas mechanistic pipelines depend on calibrated submodels that are rarely available during early process development. We present PSP-HDC, a graph-structured hyperdimensional computing framework that encodes a directed PSP graph as an internal prior for representation, inference, and explanation. A trainable scalar-to-hypervector encoder learns parameter-specific embeddings on a fixed hyperdimensional basis to accommodate heterogeneous scales and noise. Sample representations are then composed through graph-aligned binding and bundling along directed PSP dependencies, and prediction is performed by associative-memory retrieval against class prototypes. Because the same prototype memories support both decision making and attribution, PSP-HDC provides intrinsic explanations at the parameter, group, and within-group levels, while memory alignment and separation quantify prototype formation during training. On sheet-resistance regime prediction for the 3D platform, PSP-HDC achieves an accuracy of 0.910 +/- 0.077 over 1000 random splits and 0.896 under process-fold generalization, outperforming strong baselines. 

---
# Tool Calling is Linearly Readable and Steerable in Language Models 

**Authors**: Zekun Wu, Ze Wang, Seonglae Cho, Yufei Yang, Adriano Koshiyama, Sahan Bulathwela, Maria Perez-Ortiz  

**Link**: [PDF](https://arxiv.org/pdf/2605.07990)  

**Abstract**: When a tool-calling agent picks the wrong tool, the failure is invisible until execution: the email gets sent, the meeting gets missed. Probing 12 instruction-tuned models across Gemma 3, Qwen 3, Qwen 2.5, and Llama 3.1 (270M to 27B), we find the identity of the chosen tool is linearly readable and steerable inside the model. Adding the mean-difference between two tools' average internal activations switches which tool the model selects at 77-100% accuracy on name-only single-turn prompts (93-100% at 4B+), and the JSON arguments that follow autoregressively match the new tool's schema, so flipping the name is enough. The same per-tool means also flag likely errors before they happen: on Gemma 3 12B and 27B, queries where the gap between the top-1 and top-2 tool is smallest produce 14-21x more wrong calls than queries with the largest gap. The causal effect concentrates along one direction, the row of the output layer that produces the target tool's first token: a unit vector along it at matched magnitude already reaches 93-100%, while what is left over leaves the choice almost untouched. Activation patching localises this to a small set of mid- and late-layer attention heads, and a within-topic probe across 14 same-domain $\tau$-bench airline tools reaches top-1 61-89% across five 4B-14B models, ruling out the reading that we are just moving the model along a topic axis. Even base models encode the right tool before they can emit it: cosine readout from the internal state recovers 69-82% on BFCL while base generation reaches only 2-10%, suggesting pretraining forms the representation and instruction tuning later wires it to the output. We measure tool identity selection and JSON schema correctness in single-turn fixed-menu settings; multi-turn agentic transfer is more fragile and is discussed in Limitations. 

---
# Towards Apples to Apples for AI Evaluations: From Real-World Use Cases to Evaluation Scenarios 

**Authors**: Yee-Yin Choong, Kristen Greene, Alice Qian, Meryem Marasli, Ziqi Yang, Sophia Chen, Laura Dabbish, Anand Rao, Hong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2605.07986)  

**Abstract**: AI measurement science has a wide variety of methodologies and measurements for comparing AI systems, resulting in what often appear to be "apples-to-oranges" comparisons across AI evaluations. To move toward "apples-to-apples" comparisons in real-world AI evaluations, this work advocates for methodological transparency in evaluation scenarios, operational grounding, and human-centered design (HCD) principles. We propose a repeatable process for transforming high-level use cases to detailed scenarios by eliciting use cases from subject matter experts (SMEs) via a structured AI Use Case Worksheet with six key elements: use case, sector, user (direct and indirect), intended outcomes, expected impacts (positive and negative), and KPIs and metrics. We demonstrate utility of the worksheet and process in the U.S. financial services sector. This paper reports on example high-level AI use cases identified by financial services sector SMEs: cyber defense enablement, developer productivity, financial crime aggregation, suspicious activity report (SAR) filing, credit memo generation, and internal call center support. These AI use cases provided are illustrative of the process and not exhaustive. Central to our work is a three-stage expansion pipeline combining LLM prompting with human reviews to generate 107 scenarios from those use cases elicited from SMEs. This process integrates iterative human reviews at every juncture to ensure operational grounding: for scenario titles and descriptions; for core scenario elements like users, benefits and risks, and metrics; and for scenario narratives and evaluation objectives. Human checkpoints ensure scenarios remain reflective of real-world usage and human needs. We describe a validation rubric to assess scenario quality. By defining key scenario components, this work supports a more consistent and meaningful paradigm for human-centered AI evaluations. 

---
# Dooly: Configuration-Agnostic, Redundancy-Aware Profiling for LLM Inference Simulation 

**Authors**: Joon Ha Kim, Geon-Woo Kim, Anoop Rachakonda, Daehyeok Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.07985)  

**Abstract**: Selecting the optimal LLM inference configuration requires evaluation across hardware, serving engines, attention backends, and model architectures, since no single choice performs best across all workloads. Profile-based simulators are the standard tool, yet they hardcode their operation set to a specific configuration and re-profile every operation from scratch, making exploration prohibitively expensive. This cost stems from a missing structural understanding: every input dimension of each operation is fixed by the model configuration or determined by the incoming request. Many model-configuration values (e.g., head size, layer count) recur across models, so the same operation runs in many configurations; a single sweep over the request-dependent dimensions can serve them all. We present Dooly, which exploits this structure to achieve configuration-agnostic, redundancy-aware profiling. Dooly performs a single inference pass, labels each input dimension with its origin via taint propagation, and selectively profiles only operations absent from its latency database; stateful operations such as attention are isolated by reusing the serving engine's own initialization code, eliminating manual instrumentation. It builds latency regression models based on the database, which becomes a drop-in backend for existing simulators. Across two GPU platforms, three attention backends, and diverse model architectures, Dooly achieves simulation accuracy within 5% MAPE for TTFT and 8% for TPOT while reducing profiling GPU-hours by 56.4% across 12 models compared to the existing profiling approach. 

---
# Where's the Plan? Locating Latent Planning in Language Models with Lightweight Mechanistic Interventions 

**Authors**: Nicole Ma, Nick Rui  

**Link**: [PDF](https://arxiv.org/pdf/2605.07984)  

**Abstract**: We study planning site formation in language models -- where internal representations of structurally-constrained future tokens form during the forward pass, and whether they causally drive generation. Using rhyming-couplet completion as a clean test of forward-looking constraint, we apply two lightweight methods (linear probing and activation patching) across Qwen3, Gemma-3, and Llama-3 at more than ten scales. Probing shows that future-rhyme information is linearly decodable at the line boundary, with signal that strengthens with scale in all three families. Activation patching reveals that only Gemma-3-27B causally relies on this encoding, exhibiting a handoff in which the causal driver migrates from the rhyme word to the line boundary around layer 30. Every other model we test conditions on the rhyme word throughout generation, with near-zero causal effect at the line boundary despite strong probe signal. We localize the Gemma-3-27B handoff to five attention heads through two-stage path patching that recover ~90% of the rhyme-routing capacity at the newline. 

---
# It Just Takes Two: Scaling Amortized Inference to Large Sets 

**Authors**: Antoine Wehenkel, Michael Kagan, Lukas Heinrich, Chris Pollard  

**Link**: [PDF](https://arxiv.org/pdf/2605.07972)  

**Abstract**: Neural posterior estimation has emerged as a powerful tool for amortized inference, with growing adoption across scientific and applied domains. In many of these applications, the conditioning variable is a set of observations whose elements depend not only on the target but also on unknown factors shared across the set. Optimal inference therefore requires treating the set jointly, which in turn requires training the estimator at the deployment set size -- a regime where memory and compute quickly become prohibitive. We introduce a simple, theoretically grounded strategy that decouples representation learning from posterior modeling. Our method trains a mean-pool Deep Set on sets of size at most two, producing an encoder that generalizes to arbitrary set sizes. The inference head is then finetuned on pre-aggregated embeddings, making training cost essentially independent of the deployment set size N. Across scalar, image, multi-view 3D, molecular, and high-dimensional conditional generation benchmarks with N in the thousands, our approach matches or outperforms standard baselines at a fraction of the compute. 

---
# TimeLesSeg: Unified Contrast-Agnostic Cross-Sectional and Longitudinal MS Lesion Segmentation via a Stochastic Generative Model 

**Authors**: Vicent Caselles-Ballester, Eloy Martínez-Heras, Giuseppe Pontillo, Zoe Mendelsohn, Elena M. Marrón, Juan Luis García Fernández, Laia Subirats, Jon Stutters, Jeremy Chataway, Frederik Barkhof, Sara Llufriu, Ferran Prados  

**Link**: [PDF](https://arxiv.org/pdf/2605.07955)  

**Abstract**: Multiple sclerosis (MS) expresses substantial clinical and radiological heterogeneity, which poses significant challenges for automatic lesion segmentation. The current deep learning-based SOTA is highly susceptible to changes in both distribution, e.g., changes in scanner; as well as the structure of inputs, evident in the current divide between cross-sectional and longitudinal approaches. We introduce TimeLesSeg, a unified contrast-agnostic framework designed to segment MS lesions regardless of the presence of a temporal dimension in its inputs, with a single convolutional neural network. Our approach models pathological priors through lesion masks, which are processed together with the current scan. Cross-sectional processing is enabled by exposing the model to training cases where no prior information is available, which are modeled with an empty mask, allowing it to operate seamlessly in both scenarios. To overcome the scarcity and inconsistency of longitudinal datasets, we propose a novel generative pipeline in which patterns of lesion evolution are simulated by stochastically deforming each individual lesion with morphological operations, producing realistic prior timepoints. In parallel, we achieve contrast agnosticism through Gaussian mixture model-based domain randomization, enabling the network to experience a wide spectrum of intensity profiles. Results on three publicly available and two in-house datasets show that TimeLesSeg outperforms the contrast-agnostic state of the art on single-modality inputs across overlap- and distance-based metrics. In longitudinal processing, our method outperforms SAMSEG, and captures lesion load dynamics more accurately than both the former and LST-AI. All source code related to the development of TimeLesSeg is available at this https URL. 

---
# Exploring the non-convexity in machine learning using quantum-inspired optimization 

**Authors**: Kandula Eswara Sai Kumar, Parth Dhananjay Danve, Abhishek Chopra, Rut Lineswala  

**Link**: [PDF](https://arxiv.org/pdf/2605.07947)  

**Abstract**: The escalating complexity of modern machine learning necessitates solving challenging non-convex optimization problems, particularly in high-dimensional regimes and scenarios contaminated by gross outliers. Traditional approaches, relying on convex relaxations or specialized local search heuristics, frequently succumb to suboptimal local minima and fail to recover the true underlying discrete structures. In this paper, we propose treating these non-convex challenges as a global search problem and introduce a unified framework based on Quantum-Inspired Evolutionary Optimization (QIEO). By leveraging a probabilistic representation inspired by quantum superposition, QIEO maintains a global view of the search space, enabling it to tunnel through local optima that trap conventional gradient-based and greedy solvers. We comprehensively evaluate QIEO across diverse non-convex applications, including sparse signal recovery (gene expression analysis and compressed sensing) and robust linear regression. Extensive benchmarking against state-of-the-art continuous solvers (ADAM, Differential Evolution), classical metaheuristics (Genetic Algorithms), and specialized non-convex algorithms (Iterative Hard Thresholding) demonstrates that QIEO consistently achieves superior structural fidelity, lower mean squared error, and enhanced robustness without support inflation. Our findings suggest that embracing a quantum-inspired global search provides a resilient, unified paradigm for overcoming the inherent intractability of discrete nonconvex machine learning landscapes. 

---
# TAVIS: A Benchmark for Egocentric Active Vision and Anticipatory Gaze in Imitation Learning 

**Authors**: Giacomo Spigler  

**Link**: [PDF](https://arxiv.org/pdf/2605.07943)  

**Abstract**: Active vision -- where a policy controls its own gaze during manipulation -- has emerged as a key capability for imitation learning, with multiple independent systems demonstrating its benefits in the past year. Yet there is no shared benchmark to compare approaches or quantify what active vision contributes, on which task types, and under what conditions. We introduce TAVIS, evaluation infrastructure for active-vision imitation learning, with two complementary task suites -- TAVIS-Head (5 tasks, global search via pan/tilt necks) and TAVIS-Hands (3 tasks, local occlusion via wrist cameras) -- on two humanoid torso embodiments (GR1T2, Reachy2), built on IsaacLab. TAVIS provides three evaluation primitives: a paired headcam-vs-fixedcam protocol on identical demonstrations; GALT (Gaze-Action Lead Time), a novel metric grounded in cognitive science and HRI that quantifies anticipatory gaze in learned policies; and procedural ID/OOD splits. Baseline experiments with Diffusion Policy and $\pi_0$ reveal that (i) active-vision generally helps, but benefits are task-conditional rather than uniform; (ii) multi-task policies degrade sharply under controlled distribution shifts on both suites; and (iii) imitation alone yields anticipatory gaze, with median lead times comparable to the human teleoperator reference. Code, evaluation scripts, demonstrations (LeRobot v3.0; ~2200 episodes) and trained baselines are released at this https URL and this https URL. 

---
# One Token Per Frame: Reconsidering Visual Bandwidth in World Models for VLA Policy 

**Authors**: Zuojin Tang, Shengchao Yuan, Xiaoxin Bai, Zhiyuan Jin, De Ma, Gang Pan, Bin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07931)  

**Abstract**: Vision-language-action (VLA) models increasingly rely on auxiliary world modules to plan over long horizons, yet how such modules should be parameterized on top of a pretrained VLA remains an open design question. Existing world-model-augmented VLAs typically pass the per-frame visual stream into the world module at high visual bandwidth and treat its rollout as a side product of action prediction; under a constrained adaptation budget on a frozen backbone, this leaves both the per-frame representation and the latent action coupling under-examined. We introduce OneWM-VLA, which compresses each view into a single semantic token per frame through an Adaptive Attention Pooling, and produces the resulting latent stream and the action trajectory under a single flow-matching objective rather than connecting them through a separate decoder. Empirically, we find that per-frame visual bandwidth can be reduced to a single token without compromising long-horizon performance under our setup. Trained with 14.71M LoRA parameters on a $\pi_0$ (2B) backbone, OneWM-VLA improves the average success rate from 47.9% to 61.3% on MetaWorld~MT50, reaches 95.6% on LIBERO-Long (vs.85.2% for $\pi_0$), and reaches 60.0% on the long-horizon deformable task Fold Cloth on a real Piper arm (vs.20.0% for $\pi_0$). 

---
# INO-SGD: Addressing Utility Imbalance under Individualized Differential Privacy 

**Authors**: Xiao Tian, Jue Fan, Rachael Hwee Ling Sim, Bryan Kian Hsiang Low  

**Link**: [PDF](https://arxiv.org/pdf/2605.07930)  

**Abstract**: Differential privacy (DP) is widely employed in machine learning to protect confidential or sensitive training data from being revealed. As data owners gain greater control over their data due to personal data ownership, they are more likely to set their own privacy requirements, necessitating individualized DP (IDP) to fulfil such requests. In particular, owners of data from more sensitive subsets, such as positive cases of stigmatized diseases, likely set stronger privacy requirements, as leakage of such data could incur more serious societal impact. However, existing IDP algorithms induce a critical utility imbalance problem: Data from owners with stronger privacy requirements may be severely underrepresented in the trained model, resulting in poorer performance on similar data from subsequent users during deployment. In this paper, we analyze this problem and propose the INO-SGD algorithm, which strategically down-weights data within each batch to improve performance on the more private data across all iterations. Notably, our algorithm is specially designed to satisfy IDP, while existing techniques addressing utility imbalance neither satisfy IDP nor can be easily adapted to do so. Lastly, we demonstrate the empirical feasibility of our approach. 

---
# Trajectory as the Teacher: Few-Step Discrete Flow Matching via Energy-Navigated Distillation 

**Authors**: Amin Karimi Monsefi, Dominic Culver, Nikhil Bhendawade, Manuel R. Ciosici, Yizhe Zhang, Irina Belousova  

**Link**: [PDF](https://arxiv.org/pdf/2605.07924)  

**Abstract**: Discrete flow matching generates text by iteratively transforming noise tokens into coherent language, but may require hundreds of forward passes. Distillation uses the multi-step trajectory to train a student to reproduce the process in a few steps. When the student underperforms, the usual explanation is insufficient capacity. We argue the opposite: the trajectory is the bottleneck, not the student. Each training trajectory is built through a chain of blind stochastic jumps with no evaluation of sequence quality; a single bad decision at an early midpoint propagates through subsequent steps, yet the student must imitate the result. Trajectory-Shaped Discrete Flow Matching (TS-DFM) replaces these blind jumps with guided navigation: a lightweight energy compass evaluates candidate continuations at each midpoint, selecting the most coherent. All shaping is training-only; inference cost is unchanged. On 170M-parameter language modeling, the shaped student at 8 steps achieves 32% lower perplexity than the 1,024-step teacher while being 128x faster, with gains consistent across source distributions and three evaluators of increasing scale. TS-DFM achieves the best perplexity of any discrete-generation baseline we compare against, including methods trained on 6x more data or using 5x larger models. 

---
# Sycophantic AI makes human interaction feel more effortful and less satisfying over time 

**Authors**: Lujain Ibrahim, Franziska Sofia Hafner, Myra Cheng, Cinoo Lee, Rebecca Anselmetti, Robb Willer, Luc Rocher, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07912)  

**Abstract**: Millions of people now turn to artificial intelligence (AI) systems for personal advice, guidance, and support. Such systems can be sycophantic, frequently affirming users' views and beliefs. Across five preregistered studies (N = 3,075 participants, 12,766 human-AI conversations), including a three-week study with a census-representative U.S. sample, we provide longitudinal experimental evidence that sycophantic AI shifts how users approach their closest relationships. We show that sycophantic AI immediately delivers the emotional and esteem support users typically associate with close friends and family. Over three weeks of such interactions, users became nearly as likely to seek personal advice from sycophantic AI as from close friends and family, and reported lower satisfaction with their real-world social interactions. When given a choice among AI response styles, a majority preferred sycophantic AI -- not for the quality of its advice, but because it made them feel most understood. Together, these findings offer a relational account of AI sycophancy: by providing frictionless understanding, it may quietly raise the bar against which human relationships are judged. 

---
# Statistical inference with belief functions: A survey 

**Authors**: Fabio Cuzzolin  

**Link**: [PDF](https://arxiv.org/pdf/2605.07908)  

**Abstract**: Belief functions are a powerful and popular framework for the mathematical characterisation of uncertainty, in particular in situations in which lack of data renders learning a probability distribution for the problem impractical. The first step in a reasoning chain based on belief functions is inference: how to learn a belief measure from the available data. In this survey we focus, in particular, on making inference from statistical data, and review the most significant contributions in the area. 

---
# CoCoReviewBench: A Completeness- and Correctness-Oriented Benchmark for AI Reviewers 

**Authors**: Hexuan Deng, Xiaopeng Ke, Yichen Li, Ruina Hu, Dehao Huang, Derek F. Wong, Yue Wang, Xuebo Liu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07905)  

**Abstract**: Despite the rapid development of AI reviewers, evaluating such systems remains challenging: metrics favor overlap with human reviews over correctness. However, since human reviews often cover only a subset of salient issues and sometimes contain mistakes, they are unreliable as gold references. To address this, we build category-specific benchmark subsets and skip evaluation when the corresponding human reviews are missing to strengthen Completeness. We also leverage reviewer--author--meta-review discussions as expert annotations and filter unreliable reviews accordingly to strengthen Correctness. Finally, we introduce CoCoReviewBench, which curates 3,900 papers from ICLR and NeurIPS to enable reliable and fine-grained evaluation of AI reviewers. Analysis shows that AI reviewers remain limited in correctness and are prone to hallucinations, and highlights reasoning models as more effective reviewers, motivating further directions for improving AI reviewers. Benchmarks and models are available at this https URL. 

---
# BeeVe: Unsupervised Acoustic State Discovery in Honey Bee Buzzing 

**Authors**: Hamze Hammami, Nidhal Abdulaziz  

**Link**: [PDF](https://arxiv.org/pdf/2605.07903)  

**Abstract**: Discovering structure in biological signals without supervision is a fundamental problem in computational intelligence, yet existing bioacoustic methods assume vocal production models or predefined semantic units, leaving non-vocal species poorly served. This work introduces BeeVe, an unsupervised framework for acoustic state discovery in collective honey bee buzzing. BeeVe uses the self-supervised Patchout Spectrogram Transformer (PaSST) as a frozen feature extractor, then trains a Vector-Quantized Variational Autoencoder (VQ-VAE) without labels on those embeddings, learning a finite discrete codebook of acoustic tokens directly from unlabelled hive audio. No labels, pretext tasks, or contrastive objectives are used at any stage. Post-hoc evaluation against known queen status reveals that the learned tokens separate queenright and queenless conditions with Jensen-Shannon Divergence values between 0.609 and 0.688, and that the queenless condition further decomposes into three internally coherent sub-states stable across experiments with different codebook sizes and random seeds. Token transition analysis confirms non-random sequential structure (p << 0.001) across all experiments. Generalisation to unseen recordings preserves both token overlap (Jaccard = 0.947) and global manifold topology. These results demonstrate that unsupervised discrete codebook learning can recover repeatable acoustic structure from a non-vocal biological signal without annotation, opening a path toward non-invasive acoustic hive health monitoring. 

---
# Semantic-Aware Adaptive Visual Memory for Streaming Video Understanding 

**Authors**: Hang Wu, Sherin Mary Mathews, Yujun Cai, Ming-Hsuan Yang, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07897)  

**Abstract**: Online streaming video understanding requires models to process continuous visual inputs and respond to user queries in real time, where the unbounded stream and unpredictable query timing turn memory management into a central challenge. Existing methods typically compress visual tokens via visual similarity heuristics, or augment compression with KV-cache-level retrieval. However, compression decisions rarely incorporate semantic signals, and retrieval is often added after compression is finalized, making the two stages hard to coordinate. We present SAVEMem, a training-free dual-stage framework that brings semantic awareness into memory generation and lets the retrieval scope adapt per query. In Stage~1, SAVEMem builds a three-tier streaming memory online under a constant memory budget. A fixed pseudo-question bank provides a lightweight semantic prior, so that long-term retention is shaped by semantic salience rather than visual similarity alone. In Stage~2, SAVEMem performs query-aware retrieval over this memory. An anchor-conditioned recency gate adapts the retrieval scope from short-term to mid- and long-term memory based on whether the query targets the present or the distant past. Within this scope, late interaction between query and memory tokens selects candidate frames for answering. Applied to Qwen2.5-VL without training, SAVEMem improves the OVO-Bench overall score from 52.27 to 62.69 and yields consistent gains on StreamingBench and ODV-Bench, while reducing peak GPU memory by 48\% at 128 frames over the backbone. 

---
# What if AI systems weren't chatbots? 

**Authors**: Sourojit Ghosh, Pranav Narayanan Venkit, Sanjana Gautam, Avijit Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2605.07896)  

**Abstract**: The rapid convergence of artificial intelligence (AI) toward conversational chatbot interfaces marks a critical moment for the industry. This paper argues that the chatbot paradigm is not a neutral interface choice, but a dominant sociotechnical configuration whose widespread adoption reshapes social, economic, legal, and environmental systems. We examine how treating AI primarily as conversational assistants has extensive structural downsides. We show how chatbot-based systems often fail to adequately meet user needs, particularly in complex or high-stakes contexts, while projecting confidence and authority. We further analyze how the normalization of chatbot-mediated interaction alters patterns of work, learning, and decision-making, contributing to deskilling, homogenization of knowledge, and shifting expectations of expertise. Finally, we examine broader societal effects, including labor displacement, concentration of economic power, and increased environmental costs driven by sustained investment in large-scale chatbot infrastructures. While acknowledging legitimate benefits, we argue that the current trajectory of AI development reflects specific value choices that prioritize conversational generality over domain specificity, accountability, and long-term social sustainability. We conclude by outlining alternative directions for AI development and governance that move beyond one-size-fits-all chatbots, emphasizing pluralistic system design, task-specific tools, and institutional safeguards to mitigate social and economic harm. 

---
# Video Understanding Reward Modeling: A Robust Benchmark and Performant Reward Models 

**Authors**: Yuancheng Wei, Linli Yao, Lei Li, Haojie Zhang, Hao Zhou, Fandong Meng, Xu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.07872)  

**Abstract**: Multimodal reward models have advanced substantially in text and image domains, yet progress in video understanding reward modeling remains severely limited by the lack of robust evaluation benchmarks and high-quality preference data. To address this, we propose a unified framework spanning benchmark design, data construction, and reward model training. We introduce Video Understanding Reward Bench (VURB), a benchmark featuring 2,100 preference pairs with long chain-of-thought reasoning traces (averaging 1,143 tokens) and majority voting evaluation across general, long, and reasoning-oriented video tasks. We further construct Video Understanding Preference Dataset (VUP-35K) via a fully automated pipeline, providing large-scale high-quality supervision for video reward training. Building on the data, we train VideoDRM and VideoGRM, a discriminative and a generative reward model, both achieving state-of-the-art performance on VURB and VideoRewardBench. Further analysis confirms that VUP-35K enhances both reward performance and model reasoning capability, while VideoDRM and VideoGRM yield significant gains under best-of-$N$ test-time scaling. 

---
# Spectral Dynamics in Deep Networks: Feature Learning, Outlier Escape, and Learning Rate Transfer 

**Authors**: Clarissa Lauditi, Cengiz Pehlevan, Blake Bordelon  

**Link**: [PDF](https://arxiv.org/pdf/2605.07870)  

**Abstract**: We study the evolution of hidden-weight spectra in wide neural networks trained by (stochastic) gradient descent. We develop a two-level dynamical mean-field theory (DMFT) that jointly tracks bulk and outlier spectral dynamics for spiked ensembles whose spike directions remain statistically dependent on the random bulk. We apply this framework to two settings: (1) infinite-width nonlinear networks in mean-field/$\mu$P scaling and (2) deep linear networks in the proportional high-dimensional limit, where width, input dimension, and sample size diverge with fixed ratios. Our theory predicts how outliers evolve with training time, width, output scale, and initialization variance. In deep linear networks, $\mu$P yields width-consistent outlier dynamics and hyperparameter transfer, including width-stable growth of the leading NTK mode toward the edge of stability (EoS). In contrast, NTK parameterization exhibits strongly width-dependent outlier dynamics, despite converging to a stable large-width limit. We show that this bulk+outlier picture is descriptive of simple tasks with small output channels, but that tasks involving large numbers of outputs (ImageNet classification or GPT language modeling) are better described by a restructuring of the spectral bulk. We develop a toy model with extensive output channels that recapitulates this phenomenon and show that edge of the spectrum still converges for sufficiently wide networks. 

---
# KL for a KL: On-Policy Distillation with Control Variate Baseline 

**Authors**: Minjae Oh, Sangjun Song, Gyubin Choi, Yunho Choi, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2605.07865)  

**Abstract**: On-Policy Distillation (OPD) has emerged as a dominant post-training paradigm for large language models, especially for reasoning domains. However, OPD remains unstable in practice due to the high gradient variance of its single-sample Monte Carlo estimator, and recipes for stable training are still immature. We propose vOPD (On-Policy Distillation with a control variate baseline), which casts OPD as policy-gradient RL and stabilizes it by introducing a control variate baseline-canonically a value function -- from the RL literature. We show that the OPD value function admits a closed form as the per-token negative reverse KL divergence between the student and the teacher, available directly from the already-computed forward pass with no additional critic or inference. Existing stabilization methods either compute the full token-level reverse KL over the entire vocabulary, adding significant overhead, or restrict it to a top-k support, biasing the objective. vOPD instead preserves the lightweight single-sample estimator, subtracting the value function as a detached baseline to keep the gradient unbiased while reducing variance. Furthermore, we show that a top-k approximation of the baseline further lowers cost without compromising performance. Across mathematical and scientific reasoning benchmarks, vOPD consistently outperforms vanilla OPD and matches the most expensive full-vocabulary baseline, offering an efficient stabilization of On-Policy Distillation through principled RL variance reduction. 

---
# On the Tradeoffs of On-Device Generative Models in Federated Predictive Maintenance Systems 

**Authors**: Usevalad Milasheuski, Piero Baraldi, Enrico Zio, Stefano Savazzi  

**Link**: [PDF](https://arxiv.org/pdf/2605.07860)  

**Abstract**: Federated Learning (FL) has emerged as a promising paradigm for preserving client data ownership and control over distributed Internet of Things (IoT) environments. While discriminative models dominate most FL use cases, recent advances in generative models -- such as Variational Autoencoders (VAE), Generative Adversarial Networks (GAN), and Diffusion Models (DM) -- offer new opportunities for unsupervised anomaly detection in time series analysis, with relevant applications in predictive maintenance (PdM) in critical industrial infrastructures. In this work, we present a comprehensive analysis of VAEs, GANs, and DMs in the context of federated PdM. We analyze their performance and communication overhead under both full and partial federation setups, where only subsets of model components are shared. Building on this analysis, the paper proposes a novel taxonomy for federated generative models that formalizes partial component sharing as a principled mechanism for model personalization. Our experiments over a real-world time series dataset reveal distinct trade-offs in model utility, stability, and scalability, especially in heterogeneous and bandwidth-constrained FL settings. For the evaluated GAN-based configurations, full federation improves training stability relative to independent local training, although the model remains less robust than the VAE- and DDPM-based alternatives. For DMs, however, partial federation -- especially decoder sharing -- can outperform full federation in bandwidth-constrained, non-IID settings. 

---
# MatryoshkaLoRA: Learning Accurate Hierarchical Low-Rank Representations for LLM Fine-Tuning 

**Authors**: Ionut-Vlad Modoranu, Mher Safaryan, Dan Alistarh  

**Link**: [PDF](https://arxiv.org/pdf/2605.07850)  

**Abstract**: With the rise in scale for deep learning models to billions of parameters, the computational cost of fine-tuning remains a significant barrier to deployment. While Low-Rank Adaptation (LoRA) has become the standard for parameter-efficient fine-tuning, the need to set a predefined, static rank $r$ requires exhaustive grid searches to balance efficiency and performance. Existing rank-adaptive solutions such as DyLoRA mitigate this by sampling ranks during the training from a predefined distribution. However, they often yield sub-optimal results at higher ranks due to lack of consistent gradient signals across the full hierarchy of ranks, thus making these methods data-inefficient. In this paper, we propose MatryoshkaLoRA, a general, Matryoshka-inspired training framework for LoRA that learns accurate hierarchical low-rank representations by inserting a fixed, carefully crafted diagonal matrix $P$ between the existing LoRA adapters to scale their sub-ranks accordingly. By introducing this simple modification, our general framework recovers LoRA and DyLoRA only by changing $P$ and ensures all sub-ranks embed the available gradient information efficiently. Our MatryoshkaLoRA supports dynamic rank selection with minimal degradation in accuracy. We further propose Area Under the Rank Accuracy Curve (AURAC), a metric that consistently evaluates the performance of hierarchical low-rank adapters. Our results demonstrate that MatryoshkaLoRA learns more accurate hierarchical low-rank representations than prior rank-adaptive approaches and achieves superior accuracy-performance trade-offs across ranks on the evaluated datasets. Our code is available at this https URL. 

---
# \mathsf{VISTA}: Decentralized Machine Learning in Adversary Dominated Environments 

**Authors**: Hanzaleh Akbari Nodehi, Parsa Moradi, Soheil Mohajer, Mohammad Ali Maddah-Ali  

**Link**: [PDF](https://arxiv.org/pdf/2605.07841)  

**Abstract**: Decentralized machine learning often relies on outsourcing computations, such as gradient evaluations, to untrusted worker nodes. Existing robust aggregation methods can mitigate malicious behavior under honest-majority assumptions, but may fail when adversaries control a majority of the workers. We study this adversary-dominated setting through an incentive-oriented framework in which reports are accepted and rewarded only when they are mutually consistent up to a threshold. This turns the adversary from a pure saboteur into a rational agent that trades off increasing estimation error against the risk of rejection and loss of reward.
We consider iterative optimization under this model. Unlike one-shot computation, iterative learning requires long-horizon decisions: permissive acceptance rules enable faster early progress but admit more adversarial corruption, while strict rules improve estimation accuracy but cause frequent rejections. We propose \mathsf{VISTA}, an adaptive algorithm that tunes the acceptance threshold using the optimization history. Numerical results show that \mathsf{VISTA} improves convergence over static thresholds. We also provide a rigorous convergence analysis showing that, with suitable incentive-aware adaptation, adversary-dominated decentralized learning can retain the asymptotic convergence behavior of standard SGD without relying on an honest majority. 

---
# PPI-Net connects molecular protein interactions to functional processes in disease 

**Authors**: Kyle Higgins, Guadalupe Gonzalez, Dennis Veselkov, Ivan Laponogov, Kirill Veselkov  

**Link**: [PDF](https://arxiv.org/pdf/2605.07838)  

**Abstract**: Understanding how molecular alterations propagate across biological systems to drive disease remains a central challenge. Although high-throughput profiling enables comprehensive characterization of tumor states, most models neglect structured biological relationships or lack interpretability across scales. Here we present PPI-Net, a hierarchical graph neural network that integrates protein-protein interaction (PPI) networks with pathway-level representations to model disease from molecular interactions to functional processes. Patient-specific molecular profiles are embedded within a shared interaction network from STRING and propagated through a multi-layer Reactome hierarchy using graph attention, enabling aggregation of gene-level signals into higher-order biological programs. Across RNA-seq data from ten cancer types from The Cancer Genome Atlas, PPI-Net achieves robust predictive performance, with balanced accuracy exceeding 90% in multiple cohorts. Comparative analysis on RNA-Seq data from breast cancer demonstrated that PPI-Net's integration of the Reactome hierarchy improved balanced accuracy by 6.7% relative to a PPI-only model, while hierarchical multi-level supervision improved balanced accuracy by 12.3% relative to using only a single top-level prediction head. Applying a multi-omics approach using RNA-seq and methylation data improves model interpretation, recovering canonical oncogenic modules, including TP53-AKT signaling and stress response pathways, while revealing convergence onto coherent programs such as ion signaling and cellular responses to stimuli. These results demonstrate that integrating interaction networks with pathway hierarchies enables accurate prediction while providing mechanistic insight into cancer biology. 

---
# Approximation-Free Differentiable Oblique Decision Trees 

**Authors**: Subrat Prasad Panda, Blaise Genest, Arvind Easwaran  

**Link**: [PDF](https://arxiv.org/pdf/2605.07837)  

**Abstract**: Decision Trees (DTs) are widely used in safety-critical domains such as medical diagnosis, valued for their interpretability and effectiveness on tabular data. However, training accurate oblique DTs is challenging due to complex optimization landscapes and overfitting risks, particularly in regression. Recent advances have introduced differentiable formulations that enable gradient-based training and joint optimization of decision boundaries and leaf regressors. Yet, existing approaches typically rely on approximations, either through probabilistic softening of boundaries (soft DTs) or quantized gradients such as the Straight-Through Estimator (STE). To overcome these limitations, we propose DTSemNet, a novel, semantically equivalent, and invertible representation of hard oblique DTs as neural networks. DTSemNet enables end-to-end training with standard gradient descent, eliminating the need for approximations in both classification and regression. While classification aligns naturally with this formulation, regression remains challenging due to the joint optimization of internal nodes and leaf regressors. To address this, we analyze the limitations of STE and introduce an annealed Top-k method that provides accurate gradient signals without approximation. Extensive experiments on classification and regression benchmarks show that DTSemNet-trained oblique DTs outperform state-of-the-art differentiable DTs. Furthermore, we demonstrate that DTSemNet can serve as programmatic DT policies in reinforcement learning environments, thereby broadening their applicability. 

---
# CyBiasBench: Benchmarking Bias in LLM Agents for Cyber-Attack Scenarios 

**Authors**: Taein Lim, Seongyong Ju, Munhyeok Kim, Hyunjun Kim, Hoki Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.07830)  

**Abstract**: Large language models (LLMs) are increasingly deployed as autonomous agents in offensive cybersecurity. In this paper, we reveal an interesting phenomenon: different agents exhibit distinct attack patterns. Specifically, each agent exhibits an attack-selection bias, disproportionately concentrating its efforts on a narrow subset of attack families regardless of prompt variations. To systematically quantify this behavior, we introduce CyBiasBench, a comprehensive 630-session benchmark that evaluates five agents on three targets and four prompt conditions with ten attack families. We identify explicit bias across agents, with different dominant attack families and varying entropy levels in their attack-family allocation distributions. Such bias is better characterized as a trait of the agents, rather than a factor associated with the attack success rate. Furthermore, our experiments reveal a bias momentum effect, where agents resist explicit steering toward attack families that conflict with their bias. This forced distribution shift does not yield measurable improvements in attack performance. To ensure reproducibility and facilitate future research, we release an interactive result dashboard at this https URL and a reproducibility artifact with aggregated session-level statistics and full evaluation scripts at this https URL. 

---
# Divide and Conquer: Object Co-occurrence Helps Mitigate Simplicity Bias in OOD Detection 

**Authors**: Boyang Dai, Chaoqi Chen, Yizhou Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07821)  

**Abstract**: Out-of-distribution (OOD) detection is crucial for ensuring the reliability of deep learning models. Existing methods mostly focus on regular entangled representations to discriminate in-distribution (ID) and OOD data, neglecting the rich contextual information within images. This issue is particularly challenging for detecting near-OOD, as models with simplicity bias struggle to learn discriminative features in disentangled representations. The human visual system can use the co-occurrence of objects in the natural environment to facilitate scene understanding. Inspired by this, we propose an Object-Centric OOD detection framework that learns to capture Object CO-occurrence (OCO) patterns within images. The proposed method introduces a new OOD detection paradigm that understands object co-occurrence within an image by predicting disentangled representations for the test sample, then adaptively divides patterns into three scenarios based on object co-occurrence patterns observed in ID training data, and finally performs OOD detection in a divide-and-conquer manner. By doing so, OCO can distinguish near-OOD by considering the semantic contextual relationships present in their images, avoiding the tendency to focus solely on simple, easily learnable regions. We evaluate OCO through experiments across challenging and full-spectrum OOD settings, demonstrating competitive results and confirming its ability to address both semantic and covariate shifts. Code is released at this https URL. 

---
# GazeVLM: Active Vision via Internal Attention Control for Multimodal Reasoning 

**Authors**: Brown Ebouky, Gabriele Carrino, Niccolo Avogaro, Christoph Studer, Andrea Bartezzaghi, Mattia Rigotti  

**Link**: [PDF](https://arxiv.org/pdf/2605.07817)  

**Abstract**: Human visual reasoning is governed by active vision, a process where metacognitive control drives top-down goal-directed attention, dynamically routing foveal focus toward task-relevant details while maintaining peripheral awareness of the global scene. In contrast, modern Vision-Language Models (VLMs) process visual information passively, relying on the static accumulation of massive token contexts that dilute spatial reasoning and induce linguistic hallucinations. Here we propose the following paradigm shift: GazeVLM, a multimodal architecture that internalizes this metacognitive oversight over its deployment of attention resources directly into the reasoning loop. By empowering the VLM to autonomously generate gaze tokens ($\texttt{<LOOK>}$), GazeVLM establishes a top-down control mechanism over its own causal attention mask. The model dynamically dictates its focal intent, triggering a continuous suppression bias that dampens irrelevant visual features, implementing spatial selective attention and simulating foveal fixation. Once local reasoning concludes, the bias lifts, seamlessly restoring the global view. This architecture enables the model to fluidly transition between global spatial awareness and localized focal reasoning without relying on external agentic contraptions like cropping tools, or inflating the context window with additional visual tokens derived from localized visual patches. Trained with a bespoke Group Relative Policy Optimization (GRPO) procedure that rewards valid grounding, our 4B-parameter GazeVLM delivers strong high-resolution multimodal reasoning performance, surpassing state-of-the-art VLMs in its parameter class by nearly 4% and agentic multimodal pipelines built around thinking with images by more than 5% on HRBench-4k and HRBench-8k. 

---
# Text-to-CAD Evaluation with CADTests 

**Authors**: Dimitrios Mallis, Marco Wang, Ahmet Serdar Karadeniz, Elisa Ricci, Anis Kacem, Djamila Aouada  

**Link**: [PDF](https://arxiv.org/pdf/2605.07807)  

**Abstract**: Text-to-CAD has recently emerged as an important task with the potential to substantially accelerate design workflows. Despite its significance, there has been surprisingly little work on Text-to-CAD evaluation, and assessing CAD model generation performance remains a considerable challenge. In this work, we introduce a new evaluation perspective for Text-to-CAD based on automated testing. We propose CADTestBench, the first test-based benchmark for Text-to-CAD, based on CADTests, executable software tests that verify whether a generated CAD model satisfies the geometric and topological requirements of the input prompt. Using CADTestBench, we conduct comprehensive benchmarking of recent Text-to-CAD methods and further demonstrate that CADTests can also guide CAD model generation, yielding simple baselines that surpass performance of current methods. CADTestBench code and data are available at GitHub and Hugging Face dataset. 

---
# Beyond Confidence: Rethinking Self-Assessments for Performance Prediction in LLMs 

**Authors**: Sree Bhattacharyya, Samarth Khanna, Leona Chen, Lucas Craig, Tharun Dilliraj, James Z. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07806)  

**Abstract**: Large Language Models (LLMs) are increasingly used in settings where reliable self-assessment is critical. Assessing model reliability has evolved from using probabilistic correctness estimates to, more recently, eliciting verbalized confidence. Confidence, however, has been shown to be an inconsistent and overoptimistic predictor of model correctness. Drawing on cognitive appraisal theory, a framework from human psychology that decomposes self-evaluation into multiple components, we propose a multidimensional perspective on model self-assessment. We elicit six appraisal-based dimensions of self-assessment, alongside confidence, and evaluate their utility for predicting model failure across 12 LLMs and 38 tasks spanning eight domains. We find that competence-related appraisal dimensions, particularly effort and ability, consistently match or outperform confidence across most settings. Effort additionally yields less overoptimistic estimates that remain stable across model sizes. In contrast, affective dimensions provide marginally predictive signals. Furthermore, the most informative dimension varies systematically with task characteristics: effort is most predictive for reasoning-intensive tasks, while ability and confidence dominate on retrieval-oriented tasks. Broadly, our findings indicate that structured multidimensional self-assessment is a promising approach to improving the reliability and safety of language model deployment across diverse real-world settings. 

---
# Prune-OPD: Efficient and Reliable On-Policy Distillation for Long-Horizon Reasoning 

**Authors**: Zhicheng Yang, Zhijiang Guo, Yifan Song, Minrui Xu, Yongxin Wang, Yiwei Wang, Xiaodan Liang, Jing Tang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07804)  

**Abstract**: On-policy distillation (OPD) leverages dense teacher rewards to enhance reasoning models. However, scaling OPD to long-horizon tasks exposes a critical flaw: as the student's generated prefix inevitably diverges from the teacher's thought process, the teacher's dense reward loses local exploitability. Continuing to generate and evaluate tokens on these ``drifted'' trajectories not only degrades reward quality but also incurs massive computational waste. To address this, we introduce \textbf{Prune-OPD}, a framework that dynamically aligns training budgets with supervision quality. By continuously monitoring the local compatibility between student and teacher predictions (e.g., via top-$k$ overlap), Prune-OPD detects prefix-drift events in real time. Upon detecting severe drift, it monotonically down-weights subsequent unreliable rewards and triggers dynamic rollout truncation. This allows the training process to halt futile generation and reallocate compute strictly to reliable teacher supervision. Across diverse teacher-student combinations, Prune-OPD consistently aligns computation with supervision reliability. When prefix drift makes dense teacher rewards unreliable, it reduces training time by 37.6\%--68.0\% while preserving, and often improving, performance on challenging benchmarks (AMC, AIME, HMMT). When student-teacher compatibility remains high, it automatically preserves long-context supervision by expanding the training window. These results suggest that Prune-OPD improves OPD not by blindly shortening rollouts, but by reallocating computation toward locally exploitable teacher rewards. 

---
# Toward Privileged Foundation Models:LUPI for Accelerated and Improved Learning 

**Authors**: Xueying Ding, Leman Akoglu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07799)  

**Abstract**: Training foundation models is computationally intensive and often slow to this http URL introduce PIQL,Privileged Information for Quick and Quality Learning, the first framework to systematically integrate privileged information (PI) to simultaneously accelerate learning and improve generalization in tabular foundation models (TFMs). We construct two complementary forms of PI: (i) aggregate dataset-level statistics that reduce the burden on in-context learning, and (ii) encodings of the underlying data-generating program, providing knowledge beyond observable data. We further design an architecture that effectively transfers the train-time-only PI by learning to reconstruct it from observed context at inference. We provide a theoretical analysis characterizing conditions under which PI reduces the population-level approximation gap and accelerates convergence in finite-data regimes. Empirical evidence shows that PIQL enables TFMs to achieve faster convergence, lower final loss, and better generalization, in effect, reducing data and compute requirements. Our work establishes PI-guided pretraining as a principled and practical paradigm for improving the efficiency and performance of foundation models. 

---
# Neural Operators as Efficient Function Interpolators 

**Authors**: Vasilis Niarchos, Angelos Sirbu, Sokratis Trifinopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2605.07792)  

**Abstract**: Neural operators (NOs) are designed to learn maps between infinite-dimensional function spaces. We propose a novel reframing of their use. By introducing an auxiliary base-space, any finite-dimensional function can be viewed as an operator acting by composition on functions of the base-space. Through a range of benchmarks on analytic functions of increasing complexity and dimensionality, we demonstrate that NOs can match or outperform standard multilayer perceptrons and Kolmogorov--Arnold Networks in accuracy while requiring significantly fewer parameters and training time. As a real-world application, we apply a two-dimensional Tensorized Fourier Neural Operator (TFNO) to the nuclear chart, learning a correction to state-of-the-art nuclear mass models as a partially observed residual field. A TFNO ensemble reaches a held-out root-mean-square error of 198.2 keV, placing it among the best recent neural-network approaches while retaining high parameter efficiency and short training times. More broadly, these results introduce NOs as a scalable framework for finite-dimensional function interpolation, from analytic benchmarks to structured scientific data. 

---
# APEX: Assumption-free Projection-based Embedding eXamination Metric for Image Quality Assessment 

**Authors**: Caterina Gallegati, Monica Bianchini, Franco Scarselli, Vittorio Murino, Barbara Toniella Corradini  

**Link**: [PDF](https://arxiv.org/pdf/2605.07786)  

**Abstract**: As generative models achieve unprecedented visual quality, the gold standard for image evaluation remains traditional feature-distribution metrics (e.g., FID). However, these metrics are provably hindered by the closed-vocabulary bottleneck of outdated features and the assumptive bias of rigid parametric formulations. Recent alternatives exploit modern backbones to solve the feature bottleneck, yet continue to suffer from parametric limitations. To close this gap, we introduce APEX (Assumption-free Projection-based Embedding eXamination), a novel evaluation framework leveraging the Sliced Wasserstein Distance as a mathematically grounded, assumption-free similarity measure. APEX inherits effective scalability to high-dimensional spaces, as we prove with theoretical and empirical evidences. Moreover, APEX is embedding-agnostic and uses two open-vocabulary foundation models, CLIP and DINOv2, as feature extractors. Benchmarking APEX against established baselines reveals superior robustness to visual degradations. Additionally, we show that APEX metrics exhibit intra- and cross-dataset stability, ensuring highly stable evaluations on out-of-domain datasets. 

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
# When Losses Align: Gradient-Based Composite Loss Weighting for Efficient Pretraining 

**Authors**: Ivan Karpukhin, Andrey Savchenko  

**Link**: [PDF](https://arxiv.org/pdf/2605.07756)  

**Abstract**: Modern deep models are often pretrained on large-scale data with missing labels using composite objectives, where the relative weights of multiple loss terms act as hyperparameters. Tuning these weights with random search or Bayesian optimization is computationally expensive, as it requires many independent training runs. To address this, we propose a gradient-based bilevel method that learns pretraining loss weights online by aligning the composite pretraining gradient with a downstream objective. By exploiting the structure of the loss, the method avoids the multiple backward passes typically required by truncated backpropagation through the full model, reducing the overhead of hyperparameter tuning to approximately 30% above a single training run. We evaluate the approach on event-sequence modeling and self-supervised computer vision, where it matches or improves upon carefully tuned baselines while substantially reducing the cost of hyperparameter tuning compared to random or Bayesian search. 

---
# Vibe coding before the trend 

**Authors**: Leon van Bokhorst, Koen Suilen  

**Link**: [PDF](https://arxiv.org/pdf/2605.07751)  

**Abstract**: Early 2025 we ran a series of vibe coding challenges across four different student cohorts. The cohorts included 54 ICT students, 24 digital marketing students, and 7 journalism students at Fontys University of Applied Sciences (Netherlands), and 22 BA Communication students at North-West University (South Africa).
From the student reflections, five major patterns emerged. Students reported that AI tools shifted their focus from syntax to higher-order thinking; they also described a skill shift from memorizing to evaluating; they viewed AI proficiency as career-essential; they framed their relationship with AI as partnership rather than replacement; and finally non-technical students showed the strongest appreciation for the accessibility these tools provide.
This practitioner report documents what we observed during the classroom experiments, we reflect on how the landscape has shifted in the year since, and shares practical lessons for educators considering similar experiments. We present the observations as what they are: patterns from practice, not proven conclusions, in the beleif that sharing early stage experiences contributes to the overall field of AI and education. 

---
# Intelligent Truck Matching in Full Truckload Shipments using Ping2Hex approach 

**Authors**: Srinivas Kumar R, Jose Mathew, Ankit Singh Chauhan, Dinesh Rajkumar, Aravind Manoj, Mohit Goel  

**Link**: [PDF](https://arxiv.org/pdf/2605.07733)  

**Abstract**: Accurate truck-to-shipment matching using GPS data is foundational for full truckload supply chain visibility, enabling real-time tracking and accurate estimated time of arrival (ETA) predictions. However, missing or corrupted vehicle identifiers prevent traditional matching approaches, leaving shipments without visibility. This paper presents Intelligent Truck Matching (ITM) 2.0, a machine learning system that addresses this critical gap by formulating matching as a probabilistic ranking problem. Our approach leverages Uber H3 hexagonal spatial indexing to discretize GPS pings into route similarity features, combined with temporal information, then applies LightGBM gradient boosting with threshold-based post-processing. Through rigorous evaluation including offline model selection (SVM, XGBoost, LightGBM), comprehensive ablation studies, and production shadow testing, we demonstrate substantial gains over rule-based baselines. ITM 2.0 achieves 26 percentage point precision improvement in North America and 14 points in Europe, while doubling coverage. Deployed in production at Project44 handling full truckload shipments, the system demonstrates robustness to geocoding errors up to 1 km, multiple candidate trucks, and sparse pings. 

---
# Benchmarking EngGPT2-16B-A3B against Comparable Italian and International Open-source LLMs 

**Authors**: Andrea Sassella, Andrea Chizzola, Tommaso Bianchi, Luca Alessandrelli, Mark James Carman  

**Link**: [PDF](https://arxiv.org/pdf/2605.07731)  

**Abstract**: This report benchmarks the performance of ENGINEERING Ingegneria Informatica S.p.A.'s EngGPT2MoE-16B-A3B LLM, a 16B parameter Mixture of Experts (MoE) model with 3B active parameters. Performance is investigated across a wide variety of representative benchmarks, and is compared against comparably-sized open-source MoE and dense models. In comparison with popular Italian models, namely FastwebMIIA-7B, Minerva-7B, Velvet-14B, and LLaMAntino-3-ANITA-8B, EngGPT2MoE-16B-A3B performs as well or better on international benchmarks: ARC-Challenge, GSM8K, AIME24, AIME25, MMLU, and HumanEval (HE). It achieves the best performance for the longest context setting (32k) of the RULER benchmark. On the Italian benchmark dataset ITALIC, the model performs as well or better than the other models except for Velvet-14B, which outperforms it. Compared with popular MoE models of comparable size, the new model reports higher values than DeepSeek-MoE-16B-Chat on all considered benchmarks. It has higher values than Moonlight-16B-A3B on HE, MMLU, AIME24, AIME25, GSM8K, and the 32k RULER setting, but lower on BFCL and some ARC and ITALIC settings. Finally it has lower values than GPT-OSS-20B on most benchmarks, including HE, MMLU, AIME24, AIME25, GSM8K, ARC, BFCL, and the RULER 32k. When compared with popular dense models, EngGPT2MoE-16B-A3B reports higher values on AIME24 and AIME25 than Llama-3.1-8B-Instruct, Gemma-3-12b-it, and Ministral-3-8BInstruct-2512-BF16, but lower values on ITALIC, BFCL, and RULER with a 32k context. When performance is aggregated across all benchmark metrics, EngGPT2MoE-16B-A3B shows higher performance than the Italian models under evaluation while achieving lower results than some of the most performant international models, in particular GPT-5 nano and Qwen3-8B. Taken together, our findings find the new model to be a step forward for native Italian Large Language Models. 

---
# Drifting Field Policy: A One-Step Generative Policy via Wasserstein Gradient Flow 

**Authors**: Juil Koo, Mingue Park, Jiwon Choi, Yunhong Min, Minhyuk Sung  

**Link**: [PDF](https://arxiv.org/pdf/2605.07727)  

**Abstract**: We propose Drifting Field Policy (DFP), a non-ODE one-step generative policy built on the drifting model paradigm. We frame the policy update as a reverse-KL Wasserstein-2 gradient flow toward a soft target policy, so that each DFP update corresponds to a gradient step in probability space. By construction, this gradient is decomposed into an ascent toward higher action-value regions and a score matching with the anchor policy as a trust region. We further derive a simple, tractable surrogate of the otherwise intractable update loss, akin to behavior cloning on top-K critic-selected actions. We find empirically that this mechanism uniquely benefits the drifting backbone owing to its non-ODE parameterization. With one-step inference, DFP achieves state-of-the-art performance on several manipulation tasks across Robomimic and OGBench, outperforming ODE-based policies. 

---
# SOD: Step-wise On-policy Distillation for Small Language Model Agents 

**Authors**: Qiyong Zhong, Mao Zheng, Mingyang Song, Xin Lin, Jie Sun, Houcheng Jiang, Xiang Wang, Junfeng Fang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07725)  

**Abstract**: Tool-integrated reasoning (TIR) is difficult to scale to small language models due to instability in long-horizon tool interactions and limited model capacity. While reinforcement learning methods like group relative policy optimization provide only sparse outcome-level rewards. Recently, on-policy distillation (OPD) has gained popularity by supplying dense token-level supervision from a teacher on student-generated trajectories. However, our experiments indicate that applying OPD to TIR leads to a critical failure mode: erroneous tool calls tend to cascade across subsequent reasoning steps, progressively amplifying student-teacher divergence and rendering the teacher's token-level supervision increasingly unreliable. To address this, we propose SOD, a step-wise on-policy distillation framework for small language model agents, which adaptively reweights distillation strength at each step based on step-level divergence. Therefore, SOD can attenuate potentially misleading teacher signals in high-divergence regions while preserving dense guidance in well-aligned states. Experiments on challenging math, science, and code benchmarks show that SOD achieves up to 20.86% improvement over the second-best baseline. Notably, our 0.6B student achieves 26.13% on AIME 2025, demonstrating effective transfer of agentic reasoning to lightweight models. Our code is available at this https URL. 

---
# Curated Synthetic Data Doesn't Have to Collapse: A Theoretical Study of Generative Retraining with Pluralistic Preferences 

**Authors**: Ali Falahati, Mohammad Mohammadi Amiri, Kate Larson, Lukasz Golab  

**Link**: [PDF](https://arxiv.org/pdf/2605.07724)  

**Abstract**: Recursive retraining of generative models poses a critical representation challenge: when synthetic outputs are curated based on a fixed reward signal, the model tends to collapse onto a narrow set of outputs that over-optimize that objective. Prior work suggests that such collapse is unavoidable without adding real data into the mix. We revisit this conclusion from an alignment perspective and show that collapse can be mitigated through curation based on multiple reward functions. We formalize the dynamics of recursive training under heterogeneous preferences and prove that, under certain conditions, the model converges to a stable distribution that allocates probability mass across competing high-reward regions. The limiting distribution preserves diversity and provably satisfies a weighted Nash bargaining solution, offering a formal interpretation of value aggregation in synthetic retraining loops. 

---
# LLM hallucinations in the wild: Large-scale evidence from non-existent citations 

**Authors**: Zhenyue Zhao, Yihe Wang, Toby Stuart, Mathijs De Vaan, Paul Ginsparg, Yian Yin  

**Link**: [PDF](https://arxiv.org/pdf/2605.07723)  

**Abstract**: Large language models (LLMs) are known to generate plausible but false information across a wide range of contexts, yet the real-world magnitude and consequences of this hallucination problem remain poorly understood. Here we leverage a uniquely verifiable object - scientific citations - to audit 111 million references across 2.5 million papers in arXiv, bioRxiv, SSRN, and PubMed Central. We find a sharp rise in non-existent references following widespread LLM adoption, with a conservative estimate of 146,932 hallucinated citations in 2025 alone. These errors are diffusely embedded across many papers but especially pronounced in fields with rapid AI uptake, in manuscripts with linguistic signatures of AI-assisted writing, and among small and early-career author teams. At the same time, hallucinated references disproportionately assign credit to already prominent and male scholars, suggesting that LLM-generated errors may reinforce existing inequities in scientific recognition. Preprint moderation and journal publication processes capture only a fraction of these errors, suggesting that the spread of hallucinated content has outpaced existing safeguards. Together, these findings demonstrate that LLM hallucinations are infiltrating knowledge production at scale, threatening both the reliability and equity of future scientific discovery as human and AI systems draw on the existing literature. 

---
# Memory-Efficient Looped Transformer: Decoupling Compute from Memory in Looped Language Models 

**Authors**: Victor Conchello Vendrell, Arnau Padres Masdemont, Niccolò Grillo, Jordi Ros-Giralt, Arash Behboodi, Fabio Valerio Massoli  

**Link**: [PDF](https://arxiv.org/pdf/2605.07721)  

**Abstract**: Recurrent LLM architectures have emerged as a promising approach for improving reasoning, as they enable multi-step computation in the embedding space without generating intermediate tokens. Models such as Ouro perform reasoning by iteratively updating internal representations while retaining a standard Key-Value (KV) cache across iterations, causing memory consumption to grow linearly with reasoning depth. Consequently, increasing the number of reasoning iterations can lead to prohibitive memory usage, limiting the practical scalability of such architectures. In this work, we propose Memory-Efficient Looped Transformer (MELT), a novel architecture that decouples reasoning depth from memory consumption. Instead of using a standard KV cache per layer and loop, MELT maintains a single KV cache per layer that is shared across reasoning loops. This cache is updated over time via a learnable gating mechanism. To enable stable and efficient training under this architecture, we propose to train MELT using chunk-wise training in a two phase procedure: interpolated transition, followed by attention-aligned distillation, both from the LoopLM starting model to MELT. Empirically, we show that MELT models fine-tuned from pretrained Ouro parameters outperform standard LLMs of comparable size, while maintaining a memory footprint comparable to those models and dramatically smaller than Ouro's. Overall, MELT achieves constant-memory iterative reasoning without sacrificing LoopLM performance, using only a lightweight post-training procedure. 

---
# An Efficient Hybrid Sparse Attention with CPU-GPU Parallelism for Long-Context Inference 

**Authors**: Feiyu Yao, Zhixiong Niu, Xiaqing Li, Yongqiang Xiong, Juan Fang, Qian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07719)  

**Abstract**: Long-context inference increasingly operates over CPU-resident KV caches, either because decoding-time KV states exceed GPU memory capacity or because disaggregated prefill-decode systems place KV data in host memory. Although block-sparse attention reduces attention cost in this setting, sparsity alone is insufficient for end-to-end efficiency. GPU-only designs remain constrained by PCIe bandwidth and metadata memory overhead, while CPU-GPU hybrid designs still suffer from substantial GPU idle time and bottlenecks in CPU-side top-k selection and sparse attention computation.
Fluxion is built on three key insights: output-aware KV budgeting, head-specific and granularity-aware sparse configuration, and cross-device coordinated execution for sparse attention over CPU-resident KV caches. Guided by these insights, Fluxion combines a lightweight head-property predictor, a granularity-budget selector, and a priority-based scheduler to jointly optimize budget allocation, sparse configuration, and CPU-GPU execution overlap. This co-design enables hybrid sparse attention to achieve both accuracy and system efficiency in long-context inference. Across 2 models, 3 benchmarks, and 40 tasks, Fluxion preserves quality well -- the worst average degradation is only -0.26 relative to FULL, while delivering 1.5$\times$-3.7$\times$ speedup over the strongest fixed sparse hybrid baseline, whose KV budget is only 0.05. 

---
# The AI-Native Large-Scale Agile Software Development Manifesto 

**Authors**: Ricardo Britto, Fredrik Palmgren, Nishrith Saini, Marcus Ohlin  

**Link**: [PDF](https://arxiv.org/pdf/2605.07717)  

**Abstract**: Despite the widespread adoption of agile methods, achieving true agility at scale remains elusive. Large-scale agile frameworks remain largely human-centric and manual, relying on coordination meetings, artifact synchronization, and role-based handoffs that inhibit real-time adaptation. Meanwhile, rapid advances in AI, particularly large language models, have begun transforming software engineering, yet their potential for organizational-level agility remains underexplored. We present the AI-Native Large-Scale Agile Software Development Manifesto: a set of values and principles that redefine how large-scale software development is organized when AI becomes a first-class participant rather than a peripheral tool. The manifesto is grounded in six principles, parallel processes, intent-driven teams, living knowledge, verification-first assurance, orchestrated agent workforces, and reusable blueprints, that together shift development from a meeting-driven, document-heavy, sequential process to an intelligent, adaptive, continuously learning system. 

---
# Cross-Attention and Encoder-Decoder Transformers: A Logical Characterization 

**Authors**: Veeti Ahvonen, Damian Heiman, Antti Kuusisto, Miguel Moreno, Matias Selin  

**Link**: [PDF](https://arxiv.org/pdf/2605.07705)  

**Abstract**: We give a novel logical characterization of encoder-decoder transformers, the foundational architecture for LLMs that also sees use in various settings that benefit from cross-attention. We study such transformers over text in the practical setting of floating-point numbers and soft-attention, characterizing them with a new temporal logic. This logic extends propositional logic with a counting global modality over the encoder input and a past modality over the decoder input. We also give an additional characterization of such transformers via a type of distributed automata, and show that our results are not limited to the specific choices in the architecture and can account for changes in, e.g., masking. Finally, we discuss encoder-decoder transformers in the autoregressive setting. 

---
# DRIP-R: A Benchmark for Decision-Making and Reasoning Under Real-World Policy Ambiguity in the Retail Domain 

**Authors**: Hsuvas Borkakoty, Sebastian Pohl, Cheng Wang, Bei Chen, Yufang Hou  

**Link**: [PDF](https://arxiv.org/pdf/2605.07699)  

**Abstract**: LLM-based agents are increasingly deployed for routine but consequential tasks in real-world domains, where their behavior is governed by inherently ambiguous domain policies that admit multiple valid interpretations. Despite the prevalence of such ambiguities in practice, existing agent benchmarks largely assume unambiguous, well-specified policies, leaving a critical evaluation gap. We introduce DRIP-R, a benchmark that systematically exploits real-world retail policy ambiguities to construct scenarios in which no single correct resolution exists. DRIP-R comprises a curated set of policy-ambiguous return scenarios paired with a realistic customer personas, a full-duplex conversational simulation with tool-calling capabilities and a multi-judge evaluation framework covering policy adherence, dialogue quality, behavioral alignment, and resolution quality. Our experiments show that frontier models fundamentally disagree on identical policy-ambiguous scenarios, confirming that ambiguity poses a genuine and systematic challenge to LLM decision-making. 

---
# Dependence on Early and Late Reverberation of Single-Channel Speaker Distance Estimation 

**Authors**: Michael Neri, Archontis Politis, Tuomas Virtanen  

**Link**: [PDF](https://arxiv.org/pdf/2605.07694)  

**Abstract**: Single-channel speaker distance estimation has recently achieved centimeter-level accuracy in simulated environments, yet it remains unclear which components of the room impulse response (RIR) the model exploits and how performance depends on the recording conditions. In this work, we decompose simulated RIRs into four variants (full, direct-only, no-late, and no-early) using the mixing time estimated from the echo density function as the boundary between early reflections and late reverberation. We define four calibration scenarios, from fully calibrated (synchronised capture, known source level) to fully uncalibrated (arbitrary onset, unknown level), and evaluate all combinations on a matched dataset. Results show that without time calibration, mean absolute error (MAE) increases to $1.29$ m and the model extracts reverberation-based cues, with early reflections emerging as the most informative component. Further analysis against DRR, $C_{50}$, and $T_{60}$ confirms that estimation accuracy improves with stronger early energy and degrades in highly reverberant environments. When time calibration is available, the model achieves a MAE of $0.14$ m by extracting the propagation delay alone, regardless of the RIR content. 

---
# TRACE: Tourism Recommendation with Accountable Citation Evidence 

**Authors**: Zixu Zhao, Sijin Wang, Yu Hou, Yuanyuan Xu, Yufan Sheng, Xike Xie, Wenjie Zhang, Won-Yong Shin, Xin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2605.07677)  

**Abstract**: Tourism is a high-stakes setting for conversational recommender systems (CRS): a plausible-sounding suggestion can waste real money and trip time once a traveler acts on it. Existing CRS benchmarks primarily evaluate systems with a single Recall@k score over entity mentions, and tourism-specific resources add spatial or knowledge-graph context, yet none of them couple multi-turn recommendation with verbatim review-span evidence and rejection recovery. This leaves an evaluation gap for tourism recommendation that is simultaneously trustworthy, verifiable, and adaptive: recommend the right point of interest (POI) for multi-aspect preferences (such as cuisine, price, atmosphere, walking distance), justify each suggestion with verifiable evidence from prior visitors so the traveler can act without trial and error, and recover when the first recommendation is rejected mid-dialogue. We introduce TRACE, where each item is a multi-turn tourism recommendation dialogue with review-span citations and explicit rejection turns: 10,000 dialogues over 2,400 Yelp POIs and 34,208 reviews across eight U.S. cities, paired with 14 retrieval, planning, and LLM baselines, along with 25 metrics organized under Accuracy, Grounding, and Recovery. Across these baselines, TRACE reveals the Three-Competency Gap: LLM Zero-Shot leads in closed-set Recall@1 and rejection recovery but cites less densely than retrievers; non-LLM retrievers achieve surface-verbatim grounding but with low accuracy; Multi-Review Synthesis fails at recovery. The Grounding Score agrees with human citation precision (Spearman rho=+0.80, p<10^-20), and paired t-tests reproduce the per-baseline ranking (p<0.01 on the dominant contrasts). TRACE reframes accountable tourism recommendation as a joint target (right POI, verifiable evidence, adaptive repair) rather than a single-axis leaderboard. 

---
# The Endogeneity of Miscalibration: Impossibility and Escape in Scored Reporting 

**Authors**: Lauri Lovén, Sasu Tarkoma  

**Link**: [PDF](https://arxiv.org/pdf/2605.07671)  

**Abstract**: Eliciting truthful reports from autonomous agents is a core problem in scalable AI oversight: a principal scores the agent's report using a strictly proper scoring rule, but the agent also benefits from the report through a non-accuracy channel (approval for autonomous action, allocation share, downstream control). The same structure appears in classical mechanism-design settings such as marketplace operation. Our main result is an endogeneity: the principal's optimal oversight necessarily uses a non-affine approval function to screen types, yet any non-affine approval makes truthful reporting suboptimal under the combined objective whenever deviation is undetectable. The principal cannot avoid the perturbation that undermines calibration. This impossibility holds for all strictly proper scoring rules, with a closed-form perturbation formula. A constructive escape exists: a step-function approval threshold achieves first-best screening for every strictly proper scoring rule, because the agent's binary inflate-or-not choice creates a type-space threshold regardless of the generator's curvature. Under the Brier score specifically, the type-independent inflation cost yields a welfare equivalence between second-best and first-best; we prove this equivalence is unique to Brier (the welfare gap under smooth $C^1$ oversight is bounded below by $\Omega(\text{Var}(1/G'') (\gamma/\beta)^2)$ for every non-Brier rule). Two instances develop the framework: AI agent oversight (the lead motivating setting) and marketplace operation (a parallel mechanism-design domain). The message for AI alignment is direct: smooth scoring-based oversight cannot elicit truthful reports from a strategic agent; sharp thresholds are the calibration-preserving design. 

---
# Towards Billion-scale Multi-modal Biometric Search 

**Authors**: Arka Koner, Chetan S. Naik, Lokesh Kurre, Vivek Raghavan, Barada P. Sabut, Tanusree Deb Barma, Anoop M. Namboodiri, Anil K. Jain  

**Link**: [PDF](https://arxiv.org/pdf/2605.07655)  

**Abstract**: Searching a multi-biometric database of a billion records for a country-level identity system requires pushing the limits of all aspects of a biometric system, including acquisition, preprocessing, feature extraction, accuracy, matching speed, presentation attack detection, and handling of special cases (e.g., missing finger digits). This is the first paper that gives insights into such a large-scale multimodal biometric search system, called Bharat ABIS, based on open-source architectures. The end-to-end pipeline of Bharat ABIS processes fingerprint, face and iris modalities through modality-specific stages of preprocessing (segmentation), quality assessment, presentation attack detection, and learning an embedding (feature extraction), producing a concatenated template of 13.5KB per person. We present a detailed analysis of the modalities and how they are integrated to create an efficient and effective solution for 1:N search (de-duplication). Evaluations on a demographically stratified gallery of 220 million identities, randomly sampled from 1.55 billion records in India's Aadhaar database, yield an FNIR of 0.3% at an FPIR of 0.5%, for adult probes (over 18 years). We also compare the performance of Bharat ABIS against three state-of-the-art COTS systems on a 20M gallery. Our system achieves a throughput of 100 searches per second on a gallery of 40M on a single server (8xNvidia H100 GPUs, 2TB RAM). 

---
# Operating Within the Operational Design Domain: Zero-Shot Perception with Vision-Language Models 

**Authors**: Berkehan Ünal, Dierend Hauke, Fazlija Dren, Plachetka Christopher  

**Link**: [PDF](https://arxiv.org/pdf/2605.07649)  

**Abstract**: Over the last few years, research on autonomous systems has matured to such a degree that the field is increasingly well-positioned to translate research into practical, stakeholder-driven use cases across well-defined domains. However, for a wide-scale practical adoption of autonomous systems, adherence to safety regulations is crucial. Many regulations are influenced by the Operational Design Domain (ODD), which defines the specific conditions in which an autonomous agent can function. This is especially relevant for Automated Driving Systems (ADS), as a dependable perception of ODD elements is essential for safe implementation and auditing. Vision-language models (VLMs) integrate visual recognition and language reasoning, functioning without task-specific training data, which makes them suitable for adaptable ODD perception. To assess whether VLMs can function as zero-shot "ODD sensors" that adapt to evolving definitions, we contribute (i) an empirical study of zero-shot ODD classification and detection using four VLMs on a custom dataset and Mapillary Vistas, along with failure analyses; (ii) an ablation of zero-shot optimization strategies with a cost-performance overview; and (iii) a suite of reusable prompting templates with guidance for adaptation. Our findings indicate that definition-anchored chain-of-thought prompting with persona decomposition performs best, while other methods may result in reduced recall. Overall, our results pave the way for transparent and effective ODD-based perception in safety-critical applications. 

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
# LithoBench: Benchmarking Large Multimodal Models for Remote-Sensing Lithology Interpretation 

**Authors**: Jun Wang, Fengpeng Li, Hang Dong, Tianjin Huang, Wei Han  

**Link**: [PDF](https://arxiv.org/pdf/2605.07640)  

**Abstract**: Remote sensing lithology interpretation is fundamental to geological surveys, mineral exploration, and regional geological mapping. Unlike general land-cover recognition, lithology interpretation is a knowledge-intensive task that requires experts to infer rock types from various features, e.g., subtle visual, spectral, textural, geomorphological, and contextual cues, making reliable automated interpretation highly challenging. Geological knowledge-guided large multimodal models offer new opportunities, yet their evaluation remains constrained by the lack of benchmarks that capture lithological annotations, multi-level geological semantics, and expert-informed assessment. Here, we propose LithoBench, a multi-level benchmark for evaluating geological semantic understanding in remote sensing lithology interpretation. LithoBench contains 10,000 expert-annotated interpretation instances across 12 representative lithological categories, including 4,000 multiple-choice and 6,000 open-ended tasks organized into five cognitive levels: Identification and Description, Comparative Analysis, Mechanism Explanation, Practical Application, and Comprehensive Reasoning. We further develop an expert-in-the-loop, knowledge-grounded semi-automated construction pipeline, coupling multi sub-processes, e.g., structured geological image descriptions, to enhance geological validity and evaluation reliability. Experiments with multiple large vision-language models eveal substantial limitations in geological semantic understanding, particularly on higher-order explanation, application, and reasoning tasks. 

---
# Post-training makes large language models less human-like 

**Authors**: Marcel Binz, Elif Akata, Abdullah Almaatouq, Mohammed Alsobay, Oleksii Ariasov, Franziska Brändle, David Broska, Jason W. Burton, Nuno Busch, Frederick Callaway, Vanessa Cheung, Brian Christian, Julian Coda-Forno, Can Demircan, Vittoria Dentella, Maria K. Eckstein, Noémi Éltető, Michael Franke, Thomas L. Griffiths, Fritz Günther, Susanne Haridi, Sebastian Hellmann, Stefan Herytash, Linus Hof, Eleanor Holton, Isabelle Hoxha, Zak Hussain, Akshay Jagadish, Elif Kara, Valentin Kriegmair, Evelina Leivada, Li Ji-An, Tobias Ludwig, Maximilian Maier, Marcelo G. Mattar, Marvin Mathony, Alireza Modirshanechi, Robin Na, Mariia Nadverniuk, Antonios Nasioulas, Surabhi S. Nath, Helen Niemeyer, Kate Nussenbaum, Sebastian Olschewski, Thorsten Pachur, Stefano Palminteri, Aliona Petrenco, Camille V. Phaneuf-Hadd, Angelo Pirrone, Manuel Rausch, Laura Raveling, Shashank Reddy, Milena Rmus, Evan M. Russek, Tankred Saanum, Kai Sandbrink, Louis Schiekiera, Johannes A. Schubert, Luca M. Schulze Buschoff, Nishad Singhi, Leah H. Somerville, Mikhail S. Spektor, Xin Sui, Christopher Summerfield, Mirko Thalmann, Anna I. Thoma, Taisiia Tikhomirova, Vuong Truong, Polina Tsvilodub, Konstantinos Voudouris, Robert C. Wilson, Kristin Witte, Shuchen Wu, Dirk U. Wulff, Hua-Dong Xiong, Songlin Xu, Lance Ying, Xinyu Zhang, Jian-Qiao Zhu, Eric Schulz  

**Link**: [PDF](https://arxiv.org/pdf/2605.07632)  

**Abstract**: Large language models (LLMs) are increasingly used as surrogates for human participants, but it remains unclear which models best capture human behavior and why. To address this, we introduce Psych-201, a novel dataset that enables us to measure behavioral alignment at scale. We find that post-training -- the stage that turns base models into useful assistants -- consistently reduces alignment with human behavior across model families, sizes, and objectives. Moreover, this misalignment widens in newer model generations even as base models continue to improve. Finally, we find that persona-induction -- a popular technique for eliciting human-like behavior by conditioning models on participant-specific information -- does not improve predictions at the level of individuals. Taken together, our results suggest that the very processes that are currently employed to turn LLMs into useful assistants also make them less accurate models of human behavior. 

---
# Safe, or Simply Incapable? Rethinking Safety Evaluation for Phone-Use Agents 

**Authors**: Zhengyang Tang, Yi Zhang, Chenxin Li, Xin Lai, Pengyuan Lyu, Yiduo Guo, Weinong Wang, Junyi Li, Yang Ding, Huawen Shen, Zhengyao Fang, Xingran Zhou, Liang Wu, Fei Tang, Sunqi Fan, Shangpin Peng, Zheng Ruan, Anran Zhang, Benyou Wang, Chengquan Zhang, Han Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07630)  

**Abstract**: When a phone-use agent avoids harm, does that show safety, or simply inability to act? Existing evaluations often cannot tell. A harmful outcome may be avoided because the agent recognized the risk and chose the safe action, or because it failed to understand the screen or execute any relevant action at all. These cases have different causes and call for different fixes, yet current benchmarks often merge them under task success, refusal, or final harmful outcome. We address this problem with PhoneSafety, a benchmark of 700 safety-critical moments drawn from real phone interactions across more than 130 apps. Each instance isolates the next decision at a risky moment and asks a simple question: does the model take the safe action, take the unsafe action, or fail to do anything useful? We evaluate eight representative phone-use agents under this framework. Our results reveal two main patterns. First, stronger general phone-use ability does not reliably imply safer choices at risky moments. Models that perform better on ordinary app tasks are not always the ones that behave more safely when the next action matters. Second, failures to do anything useful behave like a capability signal rather than a safety signal: they are concentrated in more visually and operationally demanding settings and remain stable when the evaluation protocol changes. Across models, failures split into two recurring patterns: unsafe choices in settings where the model can act but chooses wrongly, and inability to act in more visually and operationally demanding screens. Overall, a harmless outcome is not enough to count as evidence of safety. Evaluating phone-use agents requires separating unsafe judgment from inability to act. 

---
# Nürnberg NLP at PsyDefDetect: Multi-Axis Voter Ensembles for Psychological Defence Mechanism Classification 

**Authors**: Philipp Steigerwald, Eric Rudolph, Jens Albrecht  

**Link**: [PDF](https://arxiv.org/pdf/2605.07606)  

**Abstract**: Detecting levels of psychological defence mechanisms in supportive conversations is inherently ambiguous. In the PsyDefDetect shared task at BioNLP 2026 the eight positive defence categories share surface language and differ only in pragmatic function and trained raters reach only moderate inter-annotator agreement. On such a task the decisive lever is not a stronger single model but error independence, since any single representation will waver on the overlapping defence boundaries. We translate this insight into a 9-voter ensemble spanning three orthogonal axes: class granularity (all nine classes for the gatekeeper, only the eight defence classes for the specialists), training method (generative and discriminative) and base model. The system reaches $F1_{test}{=}.420$ on the hidden test set, placing first among 21 registered teams. 

---
# SAM 3D Animal: Promptable Animal 3D Reconstruction from Images in the Wild 

**Authors**: Xuyi Hu, Jin Lyu, Jiuming Liu, Yebin Liu, Silvia Zuffi, Liang An, Stefan Goetz  

**Link**: [PDF](https://arxiv.org/pdf/2605.07604)  

**Abstract**: 3D animal reconstruction in the wild remains challenging due to large species variation, frequent occlusions, and the prevalence of multi-animal scenes, while existing methods predominantly focus on single-animal settings. We present SAM 3D Animal, the first promptable framework for multi-animal 3D reconstruction from a single image. Built on the SMAL+ parametric animal model, our method jointly reconstructs multiple instances and supports flexible prompts in the form of keypoints and masks which enable more reliable disambiguation in crowded and occluded scenes. To train such a model, we further introduce Herd3D, a multi-animal 3D dataset containing over 5K images, designed to increase diversity in species, interactions, and occlusion patterns. Experiments on the Animal3D, APTv2, and Animal Kingdom datasets show that our framework achieves state-of-the-art results over both existing model-based and model-free methods, demonstrating a scalable and effective solution for prompt-driven animal 3D reconstruction in the wild. 

---
# Mathematical Reasoning via Intervention-Based Time-Series Causal Discovery Using LLMs as Concept Mastery Simulators 

**Authors**: Tsuyoshi Okita  

**Link**: [PDF](https://arxiv.org/pdf/2605.07600)  

**Abstract**: Recent methods for improving LLM mathematical reasoning, whether through MCTS-based test-time search or causal graph-guided knowledge injection, cannot identify which concepts causally contribute to a correct answer, as the observed association may be spurious, driven by confounders such as problem difficulty.
We propose CIKA (Causal Intervention for Knowledge Activation), a framework that uses the LLM itself as an interventional simulator: a prompt sets the concept state to ``mastered'' and the correctness change estimates the causal effect. We formalize this quantity as an Interventional Capability Probe (ICP), which diagnoses whether the LLM can use a given concept -- distinct from merely possessing knowledge. Because the intervention exogenously sets the concept state independently of problem difficulty, ICP separates confounding that observational methods cannot.
On 67 screened problems, the ICP of the top-ranked concept (+0.219) is significantly larger than that of the negative control (+0.039; paired $t$-test, $p < 10^{-6}$, Cohen's $d = 0.86$), confirming that the probe discriminates causally relevant concepts from irrelevant ones. Analysis of 601 Omni-MATH problems further shows that solved problems have 6.1$\times$ higher ATE than unsolved ones (0.338 vs. 0.055), confirming that ICP is predictive of problem-solving success. With a 7B-parameter LLM whose weights are entirely frozen, CIKA achieves 69.7\% on the contamination-free Omni-MATH-Rule benchmark and 64.0\% overall, compared to 60.5\% for o1-mini, and 97.2\% on GSM8K, 46--50\% on AIME 2024--2026, and 46.2\% on MathArena. The Causal Knowledge Activation component contributes 33.8\% of correct answers on problems where the base model alone fails, demonstrating that the LLM already possessed but had not activated the requisite knowledge. 

---
# Revisiting Transformer Layer Parameterization Through Causal Energy Minimization 

**Authors**: Jin Xu, Camille Couturier, Victor Rühle, Saravan Rajmohan, James Hensman  

**Link**: [PDF](https://arxiv.org/pdf/2605.07588)  

**Abstract**: Transformer blocks typically combine multi-head attention (MHA) for token mixing with gated MLPs for token-wise feature transformation, yet many choices in their parameterization remain largely empirical. We introduce Causal Energy Minimization (CEM), a framework that recasts Transformer layers as optimization steps on conditional energy functions while explicitly accounting for layer parameterization. Extending prior energy-based interpretations of attention, CEM shows that weight-tied MHA can be derived as a gradient update on an interaction energy, and that a gated MLP with shared up/down projections can be viewed through an element-wise energy. This perspective identifies a design space for Transformer layers that includes within-layer weight sharing, diagonal-plus-low-rank interactions, lightweight preconditioners, and recursive updates. We evaluate CEM-derived layers in language-modeling experiments at the moderate hundred-million-parameter scale. Despite their constrained parameterizations, these layers train stably and can match corresponding Transformer baselines. Overall, our results suggest that CEM provides a useful lens for understanding Transformer layer parameterization, connecting Transformer architectures to energy-based models and motivating further exploration of energy-guided layer designs. 

---
# Your Language Model is Its Own Critic: Reinforcement Learning with Value Estimation from Actor's Internal States 

**Authors**: Yunho Choi, Jongwon Lim, Woojin Ahn, Minjae Oh, Jeonghoon Shim, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2605.07579)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) for Large Reasoning Models hinges on baseline estimation for variance reduction, but existing approaches pay a heavy price: PPO requires a policy-model scale critic, while GRPO needs multiple rollouts per prompt to keep its empirical group mean stable. We introduce Policy Optimization with Internal State Value Estimation), which obtains a baseline at negligible cost by using the policy model's internal signals already computed during the policy forward pass. A lightweight probe predicts the expected verifiable reward from the hidden states of the prompt and generated trajectory, as well as token-entropy statistics, and is trained online alongside the policy. To preserve gradient unbiasedness despite using trajectory-conditioned features, we introduce a cross-rollout construction that predicts each rollout's value from an independent rollout's internal states. Because POISE estimates prompt value using only a single rollout, it enables higher prompt diversity for a fixed compute budget during training. This reduces gradient variance for more stable learning and also eliminates the compute overhead of sampling costs for detecting zero-advantage prompts. On Qwen3-4B and DeepSeek-R1-Distill-Qwen-1.5B across math reasoning benchmarks, POISE matches DAPO while requiring less compute. Moreover, its value estimator shows similar performance to a separate LLM-scale value model and generalizes to various verifiable tasks. By leveraging the model's own internal representations, POISE enables more stable and efficient policy optimization. 

---
# Response-G1: Explicit Scene Graph Modeling for Proactive Streaming Video Understanding 

**Authors**: Ke Ma, Jiaqi Tang, Bin Guo, Xueting Han, Ruonan Xu, Qingfeng He, Ziheng Wang, Xu Wang, Qifeng Chen, Zhiwen Yu, Yunhao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07575)  

**Abstract**: Proactive streaming video understanding requires Video-LLMs to decide when to respond as a video unfolds, a task where existing methods often fall short due to their implicit, query-agnostic modeling of visual evidence. We introduce Response-G1, a novel framework that establishes explicit, structured alignment between the accumulated video evidence and the query's expected response conditions via scene graphs. The framework operates in three fine-tuning-free stages: (1) online query-guided scene graph generation from streaming clips; (2) memory-based retrieval of the most semantically relevant historical scene graphs; and (3) retrieval-augmented trigger prompting for per-frame "silence/response" this http URL grounding both evidence and conditions in a shared graph representation, Response-G1 achieves more interpretable and accurate response timing decisions. Experimental results on established benchmarks demonstrate the superiority of our method in both proactive and reactive tasks, validating the advantage of explicit scene graph modeling and retrieval in streaming video understanding. 

---
# Ensemble Distributionally Robust Bayesian Optimisation 

**Authors**: Tigran Ramazyan, Denis Derkach  

**Link**: [PDF](https://arxiv.org/pdf/2605.07565)  

**Abstract**: We study zeroth-order optimisation under context distributional uncertainty, a setting commonly tackled using Bayesian optimisation (BO). A prevailing strategy to make BO more robust to the complex and noisy nature of data is to employ an ensemble as the surrogate model, thereby mitigating the weaknesses of any single model. In this study, we propose a novel algorithm for Ensemble Distributionally Robust Bayesian Optimisation that remains computationally tractable while managing continuous context. We obtain theoretical sublinear regret bounds, improving current state-of-the-art results. We show that our method's empirical behaviour aligns with its theoretical guarantees. 

---
# ProteinJEPA: Latent prediction complements protein language models 

**Authors**: Dan Ofer, Dafna Shahaf, Michal Linial  

**Link**: [PDF](https://arxiv.org/pdf/2605.07554)  

**Abstract**: Protein language models are trained primarily with masked language modeling (MLM), which predicts amino-acid identities at masked positions. We ask whether latent-space prediction can complement these token-level objectives under matched wall-clock budget. Across pretrained and random-init protein sequence encoders at 35--150M parameters, we find that the best protein-JEPA design is not all-position latent prediction but a variant: predicting latent targets only at masked positions, and retaining the MLM cross-entropy. We call this recipe masked-position MLM+JEPA. On a 16-task downstream suite (15 frozen linear probes plus SCOPe-40 zero-shot fold retrieval), under matched wall-clock budgets, this recipe wins more tasks than it loses against MLM-only continuation: 10 wins / 3 losses / 3 ties (hereafter W/L/T) on pretrained ESM2-35M, 11/2/3 on ESM2-150M while results in pretraining from scratch are mixed (6/8/2). Gains are seen for multiple models on 11 of 16 tasks, including stability, \b{eta}\beta \b{eta}-lactamase fitness, variant effect, intrinsic disorder, remote homology, enzyme classification, and SCOPe-40 fold retrieval. Tasks with more losses than wins are Fluorescence (TAPE) and Peptide-HLA Binding. All-position MLM+JEPA matches MLM-only overall but does not reproduce the masked-position gains. JEPA-only (no MLM) collapses in nearly every experiment. We conclude that JEPA, when combined with MLM, is competitive and can outperform pure MLM in pretraining and continued training, even under matched wall-clock budgets. 

---
# Implicit Preference Alignment for Human Image Animation 

**Authors**: Yuanzhi Wang, Xuhua Ren, Jiaxiang Cheng, Bing Ma, Kai Yu, Tianxiang Zheng, Qinglin Lu, Zhen Cui  

**Link**: [PDF](https://arxiv.org/pdf/2605.07545)  

**Abstract**: Human image animation has witnessed significant advancements, yet generating high-fidelity hand motions remains a persistent challenge due to their high degrees of freedom and motion complexity. While reinforcement learning from human feedback, particularly direct preference optimization, offers a potential solution, it necessitates the construction of strict preference pairs. However, curating such pairs for dynamic hand regions is prohibitively expensive and often impractical due to frame-wise inconsistencies. In this paper, we propose Implicit Preference Alignment (IPA), a data-efficient post-training framework that eliminates the need for paired preference data. Theoretically grounded in implicit reward maximization, IPA aligns the model by maximizing the likelihood of self-generated high-quality samples while penalizing deviations from the pretrained prior. Furthermore, we introduce a Hand-Aware Local Optimization mechanism to explicitly steer the alignment process toward hand regions. Experiments demonstrate that our method achieves effective preference optimization to enhance hand generation quality, while significantly lowering the barrier for constructing preference data. Codes are released at this https URL 

---
# Why Self-Inconsistency Arises in GNN Explanations and How to Exploit It 

**Authors**: Wenxin Tai, Yaqian Liu, Ting Zhong, Fan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.07527)  

**Abstract**: Recent work has observed that explanations produced by Self-Interpretable Graph Neural Networks (SI-GNNs) can be self-inconsistent: when the model is reapplied to its own explanatory graph subset, it may produce a different explanation. However, why self-inconsistency arises remains poorly understood. In this work, we first identify re-explanation-induced context perturbation as the direct cause of score variation. We then introduce a latent signal assignment hypothesis to explain why only some edges are sensitive to this perturbation, and analyze how conciseness regularization affects latent signal assignment. Given that self-inconsistent edges do not provide stable evidence for the model's prediction, we propose Self-Denoising (SD), a model-agnostic and training-free post-processing strategy that calibrates explanations with only one additional forward pass. Experiments across representative SI-GNN frameworks, backbone architectures, and benchmark datasets support our hypothesis and show that SD consistently improves explanation quality while adding only about 4--6\% computational overhead in practice. 

---
# LARAG: Link-Aware Retrieval Strategy for RAG Systems in Hyperlinked Technical Documentation 

**Authors**: Giorgia Bolognesi, Claudio Estatico, Ulderico Fugacci, Isabella Mastroianni, Claudio Muselli, Luca Oneto  

**Link**: [PDF](https://arxiv.org/pdf/2605.07517)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances the factual grounding of Large Language Models by conditioning their outputs on external documents. However, standard embedding-based retrievers treat naturally structured corpora, such as technical manuals, as flat collections of passages, thereby overlooking the hyperlink topology that users rely on when navigating such content.
We introduce LARAG (Link-Aware RAG): a lightweight, link-aware retrieval strategy that leverages the author-defined hyperlink structure already present in HTML documentation, encoding hyperlink relations as metadata in the chunk representations and exploiting them to perform a form of graph-like retrieval of locally relevant content.
In a benchmark of twenty expert-designed queries over Rulex Platform technical documentation and four prompting strategies, LARAG consistently improves answer quality, achieving the highest BERTScore F1, while retrieving fewer chunks and generating fewer tokens than a baseline RAG architecture used for comparison. These results show that directly leveraging the existing hyperlink topology of technical documentation, even without explicit graph construction or inference, enables an implicit form of graph-like retrieval that yields a more faithful and efficient RAG pipeline, providing better grounding at lower cost. 

---
# Excluding the Target Domain Improves Extrapolation: Deconfounded Hierarchical Physics Constraints 

**Authors**: Tsuyoshi Okita  

**Link**: [PDF](https://arxiv.org/pdf/2605.07485)  

**Abstract**: Extrapolation to out-of-distribution conditions is a fundamental challenge for physics-constrained deep generative models. Existing methods apply physical constraints as a single static regularization term uniformly across the generation process, and address neither the hierarchical structure of physical laws and the confounding variable problem. We propose the Deconfounded Hierarchical Gate (DHG), which serves as a diagnostic and control mechanism: it identifies when and how strongly temperature confounding contaminates each constraint level, so that hierarchical gates reflect intrinsic physical inconsistency rather than spurious temperature effects. DHG combines counterfactual estimation via the do-operator with backdoor adjustment to remove confounding, then applies Coarse-to-Fine physical constraints progressively. We report a counter-intuitive finding in pretraining: excluding the target-domain data from pretraining outperforms including it by 39% in extrapolation performance (RMSE 0.224 vs. 0.324). This occurs because FNO learns domain-agnostic physical patterns that transfer more effectively when the target domain is withheld. On a lithium-ion battery temperature extrapolation benchmark (trained at 24 degrees Celsius, evaluated at 4.0--43.0 degrees Celsius), our method achieves RMSE = 0.215, a 46% improvement over the unconstrained baseline (Pure CFM: 0.397). 

---
# Does Your Neural Network Extrapolate? Feature Engineering as Identifiability Bias for OOD Generalization 

**Authors**: Leonel Aguilar, Jan Nagler, Christoph Hoelscher, Nino Antulov-Fantulin  

**Link**: [PDF](https://arxiv.org/pdf/2605.07483)  

**Abstract**: Successful deep neural networks discover salient features of data. We show when and why they fail to learn out-of-distribution (OOD)-relevant representations from an in-distribution (ID) training window. This requires decoupling feature learning from data-generating-process (DGP) identifiability. From a single training window, OOD extrapolation is non-identifiable: infinitely many DGPs are $\varepsilon$-observationally equivalent on the training data but diverge arbitrarily outside it, and no in-distribution criterion alone reliably breaks the tie. A structural commitment, the feature map, label map, and model class $(\varphi, \psi, \mathcal{M})$, dictates the assumed DGP and governs OOD generalization while leaving ID performance essentially unchanged. When architecture, pretraining, augmentation, input formats, or domain knowledge implicitly inject the missing commitment, the model succeeds. When it cannot infer OOD-relevant structure from ID evidence, it fails. Changing only the representation can make the same architecture, at the same in-distribution loss, differ by ${\sim}520\times$ out of distribution. When the commitment is correct and identifiable, OOD error vanishes. For example, Fourier coordinates turn periodic extrapolation into interpolation on $\mathbb{S}^1$. The same mechanism predicts outcomes in three natural-science settings (mass-action chemistry; Kepler's-third-law exoplanet prediction, $n=2{,}362$; and cross-species coding-DNA detection) and in a 264-run positional-encoding study across Transformer, Mamba, and S4D. Finally, a controlled study shows: correct features are necessary but not sufficient. The model class must express the target, and the transformed training data must cover the relevant representation space. 

---
# SHRED: Retain-Set-Free Unlearning via Self-Distillation with Logit Demotion 

**Authors**: Zizhao Hu, Ameya Godbole, Johnny Tian-Zheng Wei, Mohammad Rostami, Jesse Thomason, Robin Jia  

**Link**: [PDF](https://arxiv.org/pdf/2605.07482)  

**Abstract**: Machine unlearning for large language models (LLMs) aims to selectively remove memorized content such as private data, copyrighted text, or hazardous knowledge, without costly full retraining. Most existing methods require a retain set of curated examples to prevent catastrophic degradation of general model utility, creating an extra data dependency that complicates deployment. We propose SHRED (Self-distillation via High-surprisal-only Retain-set-free Entropy Demotion), a retain-set-free unlearning method built on a key insight: not all tokens within a forget set instance carry memorized information equally. High-information tokens concentrate the model's memorized knowledge, while low-information tokens reflect general language competence. SHRED operates in two stages. (1) Selection: We perform a forward pass on a forget set instance, collect per-token autoregressive probabilities, and select the bottom (lowest probability, highest Shannon information) as forget positions; the remaining positions are retained as benign anchors. (2) Training: We construct modified KL targets that demote the memorized token's logit at forget positions while preserving the original distribution at benign positions. The model is then trained via a single top KL self-distillation objective that simultaneously drives forgetting and utility preservation. We evaluate SHRED across four standard unlearning benchmarks and demonstrate that it establishes a new Pareto-optimal trade-off between forget efficacy and model utility, outperforming retain-set-dependent methods. Our analysis shows that SHRED is robust against relearning attacks and membership-inference attacks, and it maintains stable utility even after many sequential unlearning runs. 

---
# Vaporizer: Breaking Watermarking Schemes for Large Language Model Outputs 

**Authors**: Jonathan Hong Jin Ng, Anh Tu Ngo, Anupam Chattopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2605.07481)  

**Abstract**: In this paper, we investigate the recent state-of-the-art schemes for watermarking large language models (LLMs) outputs. These techniques are claimed to be robust, scalable and production-grade, aimed at promoting responsible usage of LLMs. We analyse the effectiveness of these watermarking techniques against an extensive collection of modified text attacks, which perform targeted semantic changes without altering the general meaning of the text content. Our approach encompasses multiple attack strategies, which include lexical alterations, machine translation, and even neural paraphrasing. The attack efficacy is measured with two target criteria - successful removal of the watermark and preservation of semantic content. We evaluate semantic preservation through BERT scores, text complexity measures, grammatical errors, and Flesch Reading Ease indices. The experimental results reveal varying levels of effectiveness among different watermarking models, with the same underlying result that it is possible to remove the watermark with reasonable effort. This study sheds light on the strengths and weaknesses of existing LLM watermarking systems, suggesting how they should be constructed to improve security of available schemes. 

---
# ForgeVLA: Federated Vision-Language-Action Learning without Language Annotations 

**Authors**: Yuhao Zhou, Yunpeng Zhu, Yang Zhou, Jindi Lyu, Jian Lan, Zhangyuan Wang, Dan Si, Thomas Seidl, Qing Ye, Jiancheng Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07474)  

**Abstract**: Vision-Language-Action (VLA) models hold great promise for general-purpose robotic intelligence, yet scaling up such models is severely bottlenecked by the high cost of acquiring annotated training data. Fortunately, vision-equipped robots deployed across various domains already produce abundant vision-action pairs that can be leveraged to scale up VLA training more efficiently. However, these raw data cannot be centrally aggregated due to various constraints and also exhibit severe heterogeneity. To address these challenges, in this paper, we propose ForgeVLA, a federated VLA training framework that learns VLA models from distributed vision-action pairs without centralizing raw data or requiring manual annotations. Specifically, each client in ForgeVLA is equipped with an embodied instruction classifier that maps vision-action pairs to a predefined instruction set, recovering the missing language modality and forming complete vision-language-action triplets. Beyond triplet construction, we also identify vision-language feature collapse as a critical challenge that has been largely overlooked in prior federated VLA research. To mitigate this issue, ForgeVLA combines a client-side contrastive planning loss with a server-side adaptive aggregation strategy to learn task-discriminative representations efficiently. Extensive experiments across multiple benchmarks show that ForgeVLA significantly outperforms other baselines, and ablation studies further validate the contribution of each component. 

---
# HBEE: Human Behavioral Entropy Engine -- Pre-Registered Multi-Agent LLM Simulation of Peer-Suspicion-Based Detection Inversion 

**Authors**: Vickson Ferrel  

**Link**: [PDF](https://arxiv.org/pdf/2605.07472)  

**Abstract**: Insider threat detection assumes that an adaptive insider leaves behavioral residue distinguishing them from legitimate users. We test this assumption against an LLM-driven adaptive insider in a controlled multi-agent simulator. Our pre-registered five-condition study isolates defender mode (cascade vs. blind UEBA) crossed with adversary type (naive vs. adaptive OPSEC) plus a no-mole control, across 100 runs (95 valid after pre-committed exclusions). The primary finding is a detection inversion: at T_60, the adaptive mole's suspicion in-degree is statistically lower than a randomly selected innocent agent (Cliff's delta = -0.694, 95% BCa CI [-0.855, -0.519], Mann-Whitney p << 0.01). The pre-registered prediction was the opposite direction. A pre-registered equivalence test (H2) shows adaptive OPSEC produces no detectable shift in the mole's UEBA rank under either defender mode. The two detection signals (peer suspicion graph in-degree and per-agent UEBA rank) decouple under adaptive adversary behavior. We bound generalization explicitly: a pre-registered Gini calibration check (H4) returns FAIL, with HBEE pairwise message-exposure Gini (0.213) diverging from the SNAP Enron reference (0.730) by |Delta Gini| = 0.52, exceeding the equivalence bound by 5x. The paper makes a narrow but surprising claim: in a controlled environment where adaptive OPSEC is implementable as an LLM directive, peer-suspicion-cascade detection inverts. We release the simulator, pre-registration document, frozen scenarios, raw telemetry, and analysis pipeline under an open-source license. 

---
# Physical Simulators as Do-Operators: Causal Discovery under Latent Confounders for AI-for-Science 

**Authors**: Tsuyoshi Okita  

**Link**: [PDF](https://arxiv.org/pdf/2605.07467)  

**Abstract**: Existing interventional causal discovery methods -- IGSP, DCDI, ENCO -- assume causal sufficiency (no latent confounders) and rely on virtual interventions in synthetic simulators. In AI-for-Science settings such as molecular design and materials science, latent confounders are ubiquitous and real interventions (e.g., physics-based simulations) require hours to days per data point. We propose CFM-SD (Causal Flow Matching with Simulation Data), which uses first-principles physical simulators as do-operators in Pearl's interventional calculus to simultaneously handle latent confounders and real interventional data. Theoretically, $d$-variable causal structure is identifiable with $O(d)$ single-variable interventions -- the minimum under physical realizability constraints. In Intrinsic Evaluation on synthetic data ($\gamma=0.2$--$0.8$), CFM-SD achieves average F1$=0.800$ vs. F1$=0.127$--$0.562$ for all baselines. In Extrinsic Evaluation on real scientific data, CFM-SD achieves 57--58\% bias reduction in molecular toxicity prediction and battery electrolyte optimization, demonstrating practical value beyond synthetic benchmarks. 

---
# The Moltbook Files: A Harmless Slopocalypse or Humanity's Last Experiment 

**Authors**: William Brach, Federico Torrielli, Stine Lyngsø Beltoft, Annemette Brok Pirchert, Peter Schneider-Kamp, Lukas Galke Poech  

**Link**: [PDF](https://arxiv.org/pdf/2605.07462)  

**Abstract**: Moltbook is a Reddit-like platform where OpenClaw agents post, comment, and vote at scale - a so far unprecedented incident that comes with serious safety concerns. With the aim of studying emergent behavior in populations, we release the Moltbook Files, a dataset of 232k posts and 2.2M comments covering the platform's first 12 days, processed through a pipeline to identify and remove Personally-Identifiable Information (PII). We analyze community structure, authorship, lexical properties, sentiment, topics, semantic geometry, and comment interaction. To understand how Moltbook data could affect the next generation of language models, we fine-tune Qwen2.5-14B-Instruct on Moltbook Files with three adaptation levels. Our PII pipeline reveals that agents post API keys, passwords, BIP39 seed phrases on Moltbook, a publicly indexed platform. The overall sentiment is mostly neutral and mildly positive (66.6% neutral, 19.5% positive) and shows a tendency for self-referential linking. We find that fine-tuning on Moltbook data reduces truthfulness from 0.366 to 0.187. However, a model fine-tuned on a size-matched Reddit dataset produces a comparable decrease. Moltbook thus seems to be more of a harmless slopocalypse. However, tail risks remain, including agent affordances, contamination of future crawls through self-links, and potential transfer of traits to the next generation of language models. More broadly, our findings highlight the importance of control baselines in emergent misalignment evaluations. 

---
# Sparse Autoencoders as Plug-and-Play Firewalls for Adversarial Attack Detection in VLMs 

**Authors**: Hao Wang, Yiqun Sun, Pengfei Wei, Lawrence B. Hsieh, Daisuke Kawahara  

**Link**: [PDF](https://arxiv.org/pdf/2605.07447)  

**Abstract**: Vision-language models (VLMs) have advanced rapidly and are increasingly deployed in real-world applications, especially with the rise of agent-based systems. However, their safety has received relatively limited attention. Even the latest proprietary and open-weight VLMs remain highly vulnerable to adversarial attacks, leaving downstream applications exposed to significant risks. In this work, we propose a novel and lightweight adversarial attack detection framework based on sparse autoencoders (SAEs), termed SAEgis. By inserting an SAE module into a pretrained VLM and training it with standard reconstruction objectives, we find that the learned sparse latent features naturally capture attack-relevant signals. These features enable reliable classification of whether an input image has been adversarially perturbed, even for previously unseen samples. Extensive experiments show that SAEgis achieves strong performance across in-domain, cross-domain, and cross-attack settings, with particularly large improvements in cross-domain generalization compared to existing baselines. In addition, combining signals from multiple layers further improves robustness and stability. To the best of our knowledge, this is the first work to explore SAE as a plug-and-play mechanism for adversarial attack detection in VLMs. Our method requires no additional adversarial training, introduces minimal overhead, and provides a practical approach for improving the safety of real-world VLM systems. 

---
# Accelerated and data-efficient flow prediction in stirred tanks via physics-informed learning 

**Authors**: Mahdi Naderibeni, Liang Wu, David M.J. Tax  

**Link**: [PDF](https://arxiv.org/pdf/2605.07444)  

**Abstract**: The simulation of fluid flows is computationally expensive due to the complexity of its governing partial differential equations. Machine learning models offer a potential surrogate, enabling learning from simulations and significantly faster predictions of flow fields. However, these models require large training datasets, which introduces a trade-off between dataset generation cost and predictive accuracy. In this work, we investigate the relationship between the size of the training-set and accuracy of the prediction when learning steady flow fields in an industrial-scale stirred vessel. A data set of steady flows is generated using Reynolds Averaged Navier Stokes (RANS) simulations in a range of realistic operating conditions, including impeller speeds and liquid heights. We train implicit neural representations of flow fields and compare purely data-driven and constrained variants. Model performance is evaluated using global mean squared error (MSE), qualitative spatial comparisons of predicted and reference flow fields, and tracer transport simulations. We find that the prediction error decreases monotonically with increasing training data, but also that it exhibits clear diminishing returns beyond moderate dataset sizes. Physics-based constraints significantly improve accuracy and reduce variability across training runs in low-data regimes, and they lead to more stable tracer-transport behavior. Furthermore, reasonable interpolation can be achieved over different impeller speeds and liquid heights. However, these benefits come with an increase in the complexity of training, and their relative advantage diminishes as the training set grows. 

---
# Prompt Engineering Strategies for LLM-based Qualitative Coding of Psychological Safety in Software Engineering Communities: A Controlled Empirical Study 

**Authors**: Moaath Alshaikh, Tasneem Alshaher, Ricardo Vieira, Beatriz Santana, Clelio Xavier, Jose Amancio, Glauco Carneiro, Julio Leite, Savio Freire, Manoel Mendonca  

**Link**: [PDF](https://arxiv.org/pdf/2605.07422)  

**Abstract**: Qualitative analysis plays a pivotal role in understanding the human and social aspects of software engineering. However, it remains a demanding process shaped by the subjective interpretation of individual researchers and sensitive to methodological choices such as prompt design. Recent advancements in Large Language Models (LLMs) offer promising opportunities to support this type of analysis, although their reliability in reproducing human qualitative reasoning under varying prompting conditions remains largely untested. This study presents a controlled empirical evaluation of three LLMs -- Claude Haiku, DeepSeek-Chat, and Gemini 2.5 Flash -- across two prompt engineering strategies (zero-shot and multi-shot closed coding), using Cohen's kappa as the primary agreement metric over ten independent runs per configuration. Results suggest that multi-shot prompting significantly improves agreement for Claude Haiku (Delta kappa = +0.034, Wilcoxon p = 0.004) but not for DeepSeek-Chat or Gemini 2.5 Flash. Intra-model stability varies substantially -- DeepSeek-Chat and Claude Haiku exhibit the lowest variance (SD approx. 0.017), while Gemini 2.5 Flash is the least stable (SD = 0.038). A systematic over-prediction of "Sharing Negative Feedback" is identified across all models (bias ratios up to 5.25x), alongside consistent under-prediction of "Expressing Concerns." Collectively, these findings provide empirical evidence for prompt engineering guidelines in LLM-assisted qualitative coding for software engineering research. 

---
# OrchJail: Jailbreaking Tool-Calling Text-to-Image Agents by Orchestration-Guided Fuzzing 

**Authors**: Jianming Chen, Yawen Wang, Junjie Wang, Zhe Liu, Qing Wang, Fanjiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07414)  

**Abstract**: Tool-calling text-to-image (T2I) agents can plan and execute multi-step tool chains to accomplish complex generation and editing queries. However, this capability introduces a new safety attack surface: harmful outputs may arise from tool orchestration, where individually benign steps combine into unsafe results, making prompt-only jailbreak techniques insufficient. We present OrchJail, an orchestration-guided fuzzing framework for jailbreaking tool-calling T2I agents. Its core idea is to exploit high-risk tool-orchestration patterns: by learning from successful jailbreak tool-calling traces and their causal relationships to prompt wording, OrchJail directly guides the fuzzing search toward prompts that are more likely to trigger unsafe multi-step tool behaviors, rather than relying on surface-level textual perturbations. Extensive experiments demonstrate that OrchJail improves jailbreak effectiveness and efficiency across representative toolcalling T2I agents, achieving higher attack success rates, better image fidelity, and lower query costs, while remaining robust against common jailbreak defenses. Our work highlights tool orchestration as a critical, previously unexplored attack surface and provides a novel framework for uncovering safety risks in T2I agents. 

---
# Tracking Large-scale Shared Bikes with Inertial Motion Learning in GNSS Blocked Environments 

**Authors**: Feng Liu, Kejia Li, Zhiwei Yang, Chunwei Yang, Qun Li, Guobin Wu, Qiang Ni, Ruipeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2605.07412)  

**Abstract**: Although Global Navigation Satellite Systems (GNSS) provide a general solution for bike tracking outdoors, there still exist complex riding environments where only inertial navigation systems work, such as urban canyons. Despite decades of research, localization using only low-cost inertial sensors still faces challenges such as cumulative drifts and poor robustness caused by filtering methods. Furthermore, sensors such as visual and LiDAR could provide reliable measurements, but they are not suitable for large-scale deployment. In this paper, we propose an inertial tracking framework that integrates bicycle mechanical constraints with a mixture-of-experts model. Specifically, we leverage multiple expert modules to capture shared representations and weight them through the gating mechanism, thus improving multi-task learning performance and enabling uncertainty-aware trajectory estimation. Furthermore, based on the mechanical transmission between the pedal and the rear wheel of a bike, we explore the intrinsic relationship between the rider's periodic pedalling behaviors and acceleration variations, and convert such patterns into bike's wheel speed for dynamic calibration. Experiments with real-world riding data from shared bikes of the DiDi ride-hailing platform demonstrate that our system improves the accuracy of baselines by at least 12%, with wheel speed errors below 0.5 m/s at 95-percentile. 

---
# Exposing and Mitigating Temporal Attack in Deepfake Video Detection 

**Authors**: Zheyuan Gu, Minghao Shao, Zhen Wang, Yusong Wang, Mingkun Xu, Shijie Zhang, Hao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07398)  

**Abstract**: While spatiotemporal deepfake detectors achieve high AUC, our experiments reveal their susceptibility to evasion attacks. These models tend to overfit on fragile temporal spectrum cues, rather than learning robust semantic causality. To mitigate this vulnerability, we propose SpInShield, a temporal spectral-invariant defense framework explicitly designed to decouple semantic motion from manipulatable spectral artifacts. We propose a learnable spectral adversary that dynamically synthesizes severe spectral deformations, simulating extreme attack scenarios. By employing a shortcut suppression optimization strategy, SpInShield compels the encoder to extract reliable forensic cues while purging unstable spectral statistics from the latent space. Experiments show that SpInShield obtains competitive performance on widely used datasets and outperforms the strongest baseline by 21.30 percentage points in AUC under simulated amplitude spectral attacks. 

---
# Rubric-based On-policy Distillation 

**Authors**: Junfeng Fang, Zhepei Hong, Mao Zheng, Mingyang Song, Gengsheng Li, Houcheng Jiang, Dan Zhang, Haiyun Guo, Xiang Wang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2605.07396)  

**Abstract**: On-policy distillation (OPD) is a powerful paradigm for model alignment, yet its reliance on teacher logits restricts its application to white-box scenarios. We contend that structured semantic rubrics can serve as a scalable alternative to teacher logits, enabling OPD using only teacher-generated responses. To prove it, we introduce ROPD, a simple yet foundational framework for rubric-based OPD. Specifically, ROPD induces prompt-specific rubrics from teacher-student contrasts, and then utilizes these rubrics to score the student rollouts for on-policy optimization. Empirically, ROPD outperforms the advanced logit-based OPD methods across most scenarios, and achieving up to a 10x gain in sample efficiency. These results position rubric-based OPD as a flexible, black-box-compatible alternative to the prevailing logit-based OPD, offering a simple yet strong baseline for scalable distillation across proprietary and open-source LLMs. Code is available at this https URL. 

---
# Unsolvability Ceiling in Multi-LLM Routing: An Empirical Study of Evaluation Artifacts 

**Authors**: Saloni Garg, Amit Sagtani  

**Link**: [PDF](https://arxiv.org/pdf/2605.07395)  

**Abstract**: Efficient routing across multiple LLMs enables cost-quality tradeoffs by directing queries to the cheapest capable model. Prior work attributes routing headroom to an "unsolvability ceiling", queries no model in the pool can solve. We present a large-scale study of multi-tier LLM routing with 206,000 query-model pairs across six benchmarks (MMLU, MedQA, HumanEval, MBPP, Alpaca, ShareGPT) using the Gemma 4 and Llama 3.1 families. Evaluating with both LLM-as-a-judge and exact-match metrics, we show that a substantial portion of reported unsolvability stems from evaluation artifacts: (i) systematic judge biases favoring verbosity over correctness, (ii) truncation under fixed generation budgets, and (iii) output format mismatches. Through dual-judge validation and exact-match grounding, we reduce measured unsolvability across tasks. We introduce a decomposition framework attributing failures to these artifacts, revealing consistent patterns across domains and model families. These artifacts also distort router training signals: standard routers collapse to majority-class prediction (~79% smallest-tier optimal), confirmed via random-feature and shuffled-label controls, incurring a 13-17 percentage point opportunity cost. We provide actionable recommendations including dual-judge validation, exact-match anchoring, and cost-sensitive objectives. Our findings suggest existing routing headroom estimates are substantially inflated, underscoring the need for reliable evaluation protocols in multi-LLM systems. 

---
# BalCapRL: A Balanced Framework for RL-Based MLLM Image Captioning 

**Authors**: Shaokai Ye, Vasileios Saveris, Yihao Qian, Jiaming Hu, Elmira Amirloo, Peter Grasch  

**Link**: [PDF](https://arxiv.org/pdf/2605.07394)  

**Abstract**: Image captioning is one of the most fundamental tasks in computer vision. Owing to its open-ended nature, it has received significant attention in the era of multimodal large language models (MLLMs). In pursuit of ever more detailed and accurate captions, recent work has increasingly turned to reinforcement learning (RL). However, existing captioning-RL methods and evaluation metrics often emphasize a narrow notion of caption quality, inducing trade-offs across core dimensions of captioning. For example, utility-oriented objectives can encourage noisy, hallucinated, or overlong captions that improve downstream question answering while harming fluency, whereas arena-style objectives can favor fluent but generic descriptions with limited usefulness. To address this, we propose a more balanced RL framework that jointly optimizes utility-aware correctness, reference coverage, and linguistic quality. In order to effectively optimize the resulting continuous multi-objective reward formulation, we apply GDPO-style reward-decoupled normalization to continuous-valued captioning rewards and show that it improves performance over vanilla GRPO. Additionally, we introduce length-conditional reward masking, yielding a more suitable length penalty for captioning. Across LLaVA-1.5-7B and Qwen2.5-VL 3B and 7B base models, our method consistently improves caption quality, with peak gains of +13.6 DCScore, +9.0 CaptionQA, and +29.0 CapArena across different models. 

---
# Escaping the Diversity Trap in Robotic Manipulation via Anchor-Centric Adaptation 

**Authors**: Yanzhe Chen, Kevin Yuchen Ma, Qi Lv, Yiqi Lin, Zechen Bai, Chen Gao, Mike Zheng Shou  

**Link**: [PDF](https://arxiv.org/pdf/2605.07381)  

**Abstract**: While Vision-Language-Action (VLA) models offer broad general capabilities, deploying them on specific hardware requires real-world adaptation to bridge the embodiment gap. Since robot demonstrations are costly, this adaptation must often occur under a strict data budget. In this work, we identify a critical diversity trap: the standard heuristic of "maximizing coverage" by collecting diverse, single-shot demonstrations can be self-defeating due to non-vanishing estimation noise. We formalize this phenomenon as a Coverage--Density Trade-off. By decomposing the policy error into estimation (density) and extrapolation (coverage) terms, we characterize an interior optimal allocation of unique conditions for a fixed budget. Guided by this analysis, we propose Anchor-Centric Adaptation (ACA), a two-stage framework that first stabilizes a policy skeleton through repeated demonstrations at core anchors, then selectively expands coverage to high-risk boundaries via teacher-forced error mining and constrained residual updates. Real-robot experiments validate our trade-off framework and demonstrate that ACA significantly improves task reliability and success rates over standard diverse sampling strategies under the same budget. 

---
# RELO: Reinforcement Learning to Localize for Visual Object Tracking 

**Authors**: Xin Chen, Chuanyu Sun, Jiao Xu, Houwen Peng, Dong Wang, Huchuan Lu, Kede Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.07379)  

**Abstract**: Conventional visual object trackers localize targets using handcrafted spatial priors, often in the form of heatmaps. Such priors provide only surrogate supervision and are poorly aligned with tracking optimization and evaluation metrics, such as intersection over union (IoU) and area under the success curve (AUC). Here, we introduce RELO, a REinforcement-learning-to-LOcalize method for visual object tracking that formulates target localization as a Markov decision process. Specifically, RELO replaces handcrafted spatial priors with a localization policy learned over spatial positions via reinforcement learning, with rewards combining frame-level IoU and sequence-level AUC. We additionally introduce layer-aligned temporal token propagation to improve semantic consistency across frames, with negligible computational overhead. Across multiple benchmarks, RELO achieves superior results, attaining 57.5% AUC on LaSOText without template updates. This confirms that reward-driven localization provides an effective alternative to prior-driven localization for visual object tracking. 

---
# MORPH-U: Multi-Objective Resilient Motion Planning for V2X-Enabled Autonomous Driving in High-Uncertainty Environments via Simulation 

**Authors**: Shih-Yu Lai  

**Link**: [PDF](https://arxiv.org/pdf/2605.07370)  

**Abstract**: V2X can warn an autonomous vehicle about hazards beyond line-of-sight, but it also brings uncertainty: messages may be delayed, dropped, or even forged. Meanwhile, map knowledge may change during a trip, forcing the vehicle to replan under tight real-time budgets. This paper studies how to make motion planning and low-level control robust to such uncertain, event-driven updates. We present MORPH-U, a CARLA-based closed-loop stack that fuses LiDAR/radar/camera with V2X (CAM/DENM) into a Local Dynamic Map (LDM) and triggers Hybrid-A* replanning when validated hazards or map changes affect the planned route. We expose the planning/control trade-offs via a multi-objective formulation over tracking error, safety margin (minimum TTC), responsiveness, and smoothness, and select operating points using Pareto-frontier analysis. To avoid unsafe replanning from faulty V2X triggers, MORPH-U adds a lightweight Byzantine-inspired acceptance gate that combines a quorum rule with an on-board sensor veto. Experiments in dynamic CARLA scenarios show that V2X-augmented LDM improves downstream safety, Pareto tuning provides controllable accuracy-comfort trade-offs, and the gate prevents replanning under saturated false-DENM injection ($p_{\text{attack}}=1.0$). 

---
# MISA: Mixture of Indexer Sparse Attention for Long-Context LLM Inference 

**Authors**: Ruijie Zhou, Fanxu Meng, Yufei Xu, Tongxuan Liu, Guangming Lu, Muhan Zhang, Wenjie Pei  

**Link**: [PDF](https://arxiv.org/pdf/2605.07363)  

**Abstract**: DeepSeek Sparse Attention (DSA) sets the state of the art for fine-grained inference-time sparse attention by introducing a learned token-wise indexer that scores every prefix token and selects the most relevant ones for the main attention. To remain expressive, the indexer uses many query heads (for example, 64 on DeepSeek-V3.2) that share the same selected token set; this multi-head design is precisely what makes the indexer the dominant cost on long contexts. We propose MISA (Mixture of Indexer Sparse Attention), a drop-in replacement for the DSA indexer that treats its indexer heads as a pool of mixture-of-experts. A lightweight router uses cheap block-level statistics to pick a query-dependent subset of only a few active heads, and only those heads run the heavy token-level scoring. This preserves the diversity of the original indexer pool while reducing the per-query cost from scoring every prefix token with every head to scoring it with only a handful of routed heads, plus a negligible router term computed on a small set of pooled keys. We further introduce a hierarchical variant of MISA that uses the routed pass to keep an enlarged candidate set and then re-ranks it with the original DSA indexer to recover the final selected tokens almost exactly. With only eight active heads and no additional training, MISA matches the dense DSA indexer on LongBench across DeepSeek-V3.2 and GLM-5 while running with eight and four times fewer indexer heads respectively, and outperforms HISA on average. It also preserves fully green Needle-in-a-Haystack heatmaps up to a 128K-token context and recovers more than 92% of the tokens selected by the DSA indexer per layer. Our TileLang kernel delivers roughly a 3.82 times speedup over DSA's original indexer kernel on a single NVIDIA H200 GPU. 

---
# TTF: Temporal Token Fusion for Efficient Video-Language Model 

**Authors**: Simin Huo, Ning LI  

**Link**: [PDF](https://arxiv.org/pdf/2605.07355)  

**Abstract**: Video-language models (VLMs) face rapid inference costs as visual token counts scale with video length. For example, 32 frames at $448{\times}448$ resolution already yield >8,000 visual tokens in Qwen3-VL, making LLM prefill the dominant throughput bottleneck. Existing methods often rely on global similarity or attention-guided compression, incurring offsets to their gains. We propose \textbf{Temporal Token Fusion (TTF)}, a training-free, plug-and-play pre-LLM token compression framework that exploits structured temporal redundancy in video. TTF automatically selects an anchor frame, then for each subsequent frame, performs a local window similarity search (e.g.,$3\times 3$), fusing tokens that exceed a threshold. The compressed sequence maintains positional consistency across both prefill and decoding through coordinate realignment, enabling seamless integration with existing VLM pipelines. On Qwen3-VL-8B with threshold t=0.70, TTF removes about 67\% of visual tokens while retaining 99.5\% of the baseline accuracy and introducing only ${\approx}0.16$\,GFLOPs of matching overhead. Overall, TTF offers a practical, efficient solution for video understanding. The code is available at \href{this https URL}{this https URL} 

---
# Mage: Multi-Axis Evaluation of LLM-Generated Executable Game Scenes Beyond Compile-Pass Rate 

**Authors**: Hugh Xuechen Liu, Kıvanç Tatar  

**Link**: [PDF](https://arxiv.org/pdf/2605.07342)  

**Abstract**: Compile-pass rate is the dominant evaluation signal for LLM code generation, yet for multi-component domain-specific artifacts it can be actively misleading. We demonstrate this on executable game scene synthesis with a four-axis evaluation protocol (named `Mage') -- compile success, runtime success, structural fidelity, and mechanism adherence -- applied to 858 generation attempts across four open-weight LLMs (7B--30B), 26~hand-crafted Unity goal pattern playable concepts, and two automatically extracted IR granularity levels. Direct NL-to-C\# generation achieves the highest runtime-pass rate (43\% mean) yet produces structurally vacuous scenes (mechanism $F_1 \approx 0.12$). Structural IR conditioning halves the runtime rate but recovers domain-faithful structure ($F_1$ up to 1.00). Within IR conditioning, behavior-only and full-scene granularity are statistically indistinguishable (McNemar $p = 1.0$), indicating input-level granularity saturation. These results show that compile rate is anti-correlated with functional correctness in this domain and that multi-axis evaluation is necessary to detect the divergence. We release the benchmark, replay logs, and per-record metrics for independent verification. 

---
# Rethinking Importance Sampling in LLM Policy Optimization: A Cumulative Token Perspective 

**Authors**: Yuheng Zhang, Chenlu Ye, Shuowei Jin, Changlong Yu, Wei Xiong, Saurabh Sahu, Nan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07331)  

**Abstract**: Reinforcement learning, including reinforcement learning with verifiable rewards (RLVR), has emerged as a powerful approach for LLM post-training. Central to these approaches is the design of the importance sampling (IS) ratio used in off-policy policy-gradient estimation. Existing methods face a fundamental bias-variance dilemma: token-level IS ratios, as adopted by PPO (Schulman et al., 2017) and GRPO (Shao et al., 2024), introduce bias by ignoring prefix state distribution mismatch; full sequence ratios provide exact trajectory-level correction but suffer from high variance due to the multiplicative accumulation of per-token ratios, while GSPO (Zheng et al., 2025) improves numerical stability via length normalization at the cost of deviating from the exact full-sequence IS correction. In this work, we identify the cumulative token IS ratio, the product of per-token ratios up to position $t$, as a theoretically principled solution to this dilemma. We prove that, under the token-level policy-gradient formulation, this ratio provides an unbiased prefix correction for each token-level gradient term and has strictly lower variance than the full sequence ratio. Building on this insight, we propose CTPO (Cumulative Token Policy Optimization), which combines the cumulative token IS ratio with position-adaptive clipping that scales log-space clip bounds according to the natural $\sqrt{t}$ growth of the cumulative log-ratio. This yields more consistent regularization across token positions. We implement and evaluate CTPO in the tool-integrated reasoning setting on several challenging mathematical reasoning benchmarks, achieving the best average performance across both model scales compared with strong GRPO and GSPO baselines. Code will be available at this https URL. 

---
# SparseRL-Sync: Lossless Weight Synchronization with ~100x Less Communication 

**Authors**: Lucas Hu, Ranchi Zhao, Isaac Zhu, Zach Zhang, Hscos Zhang, Hugh Yin, Jason Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.07330)  

**Abstract**: In large-scale reinforcement learning (RL) systems with decoupled Trainer-Rollout execution, the Trainer must regularly synchronize policy weights to the Rollout side to limit policy staleness. When inter-node bandwidth is abundant, such synchronization is usually only a small fraction of end-to-end cost. As model size grows, however, the communication demand rises rapidly. In bandwidth-constrained or network-variable deployments -- for example, cross-datacenter or cross-cluster settings, heterogeneous resource pools, and online RL -- weight synchronization can become a dominant bottleneck for throughput and tail latency. We observe that, in mainstream large-model RL training, the locations where parameters actually change are highly sparse at the element level (often 99%+ sparsity). Building on this observation, we propose and implement SparseRL-Sync, which replaces full-weight transfers with a lossless sparse update payload (indices and values) that can be exactly reconstructed on the inference side, thereby preserving 100% fidelity. Under a simplified cost model, sparse synchronization reduces the per-update communication volume from S to approximately S/X; with 99% sparsity (X ~ 100), this yields about a 100x reduction in transmitted data. Combined with appropriate bucketing, SparseRL-Sync also reduces launch and control-plane overhead, significantly improving scalability and end-to-end efficiency in bandwidth-limited and highly asynchronous RL settings. 

---
# CSR: Infinite-Horizon Real-Time Policies with Massive Cached State Representations 

**Authors**: Robin Karlsson, Go Suzui  

**Link**: [PDF](https://arxiv.org/pdf/2605.07325)  

**Abstract**: Deploying massive large language models (LLMs) as continuous cognitive engines for robotics is bottlenecked by the time-to-first-token (TTFT) latency required to process extensive state histories. Existing solutions like RAG or sliding windows compromise global context or incur prohibitive re-computation costs. We formalize the optimal task structure for minimizing latency and theoretically prove that prefix stability, incremental extensibility, and asynchronous state reconciliation are necessary conditions for real-time performance. Building on these proofs, we introduce the Cached State Representation (CSR) framework as the practical instantiation of these properties, ensuring optimal KV-cache reuse. To sustain these properties over infinite horizons, we further propose an Asynchronous State Reconciliation (ASR) algorithm that offloads state memory eviction to a parallel computational resource to eliminate latency spikes. On a physical robot wirelessly connected to an on-premise GPU server, CSR achieves a 26-fold latency reduction (14.67s to 0.56s) for 120K token contexts with a 235B parameter model compared to a standard baseline. On an embodied AI benchmark, we achieve SOTA recall (0.836 vs. 0.459) while maintaining RAG-level latency. ASR is validated to sustain bounded, spike-free TTFT over 10 eviction cycles in continuous real-world operation. Together, CSR and ASR enable massive LLMs to function as continuously operating, high-frequency (> 2 Hz) embodied policies. 

---
# Activation Differences Reveal Backdoors: A Comparison of SAE Architectures 

**Authors**: Sachin Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2605.07324)  

**Abstract**: Backdoor attacks on language models pose a significant threat to AI safety, where models behave normally on most inputs but exhibit harmful behavior when triggered by specific patterns. Detecting such backdoors through mechanistic interpretability remains an open challenge. We investigate two sparse autoencoder architectures -- Crosscoders and Differential SAEs (Diff-SAE) -- for isolating backdoor-related features in fine-tuned models. Using a controlled SQL injection backdoor triggered by year-based context ("2024" triggers vulnerable code, "2023" triggers safe code), we evaluate both approaches across LoRA and full-rank fine-tuning regimes on SmolLM2-360M. We find that Diff-SAE consistently and substantially outperforms Crosscoders for backdoor isolation. Diff-SAE achieves a Backdoor Isolation Score (BIS) of 0.40 with perfect precision (1.0) and zero false positive rate across most experimental conditions, while Crosscoders fail almost entirely with BIS below 0.02 in most cases. This performance gap holds across multiple transformer layers (14, 18, 22, 26) and both fine-tuning regimes, with full-rank fine-tuning producing particularly clean backdoor signals. Our results suggest that backdoors manifest as directional activation shifts rather than sparse feature activations, making difference-based representations fundamentally more effective for detection. These findings have important implications for AI safety monitoring and the development of interpretability tools for detecting model manipulation. 

---
# Generative Modeling with Flux Matching 

**Authors**: Peter Pao-Huang, Xiaojie Qiu, Stefano Ermon  

**Link**: [PDF](https://arxiv.org/pdf/2605.07319)  

**Abstract**: We introduce Flux Matching, a new paradigm for generative modeling that generalizes existing score-based models to a broader family of vector fields that need not be conservative. Rather than requiring the model to equal the data score, the Flux Matching objective imposes a weaker condition that admits infinitely many vector fields whose stationary distribution is the data. This flexibility enables a class of generative models that cannot be learned under score matching, in which inductive biases, structural priors, and properties of the dynamics can be directly imposed or optimized. We show that Flux Matching performs strongly on high-dimensional image datasets and, more importantly, that our added freedom unlocks a range of applications including faster sampling, interpretable and mechanistic models, and dynamics that encode directed dependencies between variables. More broadly, Flux Matching opens a new dimension in generative modeling by turning the vector field itself into a design choice rather than a fixed target. Code is available at this https URL. 

---
# Amortized-Precision Quantization for Early-Exit Vision Transformers 

**Authors**: Rui Fang, Hsi-Wen Chen, Ming-Syan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.07317)  

**Abstract**: Vision Transformers (ViTs) achieve strong performance across vision tasks, yet their deployment with low-precision early exiting remains fragile. Existing quantization methods assume static full-depth execution, making them unstable when exit decisions are perturbed by quantization noise, which can amplify errors along dynamic inference paths. In this paper, we introduce Amortized-Precision Quantization (APQ), a utilization-aware formulation that accounts for layer-wise stochastic exposure to quantization noise and reveals depth-precision trade-offs. Building on APQ, we propose Mutual Adaptive Quantization with Early Exiting (MAQEE), a bi-level framework that jointly optimizes exit thresholds and bit-widths under explicit risk control to improve inference stability. MAQEE establishes a superior Pareto frontier in the accuracy-efficiency trade-off, reducing BOPs by up to 95% while maintaining accuracy and outperforming strong baselines by up to 20\% across classification, detection, and segmentation tasks. 

---
# DCGL: Dual-Channel Graph Learning with Large Language Models for Knowledge-Aware Recommendation 

**Authors**: Xinchi Zou, Tongzhenzhi Su, Jianjun Li, Yuan Fu, Chang Liu, Zhiying Deng, Zhiwei Shen  

**Link**: [PDF](https://arxiv.org/pdf/2605.07314)  

**Abstract**: Knowledge Graphs (KGs) have proven highly effective for recommendation systems by capturing latent item relationships, while recent integration of Large Language Models (LLMs) has further enhanced semantic understanding and addressed knowledge sparsity issues. Nevertheless, current KG-and-LLM-based methods still face three main limitations: 1) inadequate modeling of implicit semantic relationships beyond explicit KG links; 2) suboptimal single-channel fusion of ID and LLM embeddings, which often leads to signal interference and blurred representations; and 3) insufficient consideration of user-item interaction frequency variations in recommendation strategies. To address these challenges, we propose the Dual-Channel Graph Learning (DCGL) framework, featuring three key innovations: 1) a dual-channel architecture that structurally decouples rich semantic information from user behavioral patterns, preventing early interference; 2) a multi-level contrastive learning mechanism that enhances robustness against KG noise through intra-view contrasts and bridges semantic gaps between channels via inter-view alignment; and 3) a dynamic fusion mechanism that adaptively balances semantic generalization and behavioral specificity based on interaction frequency, resolving the cascading limitation. Extensive experiments on four real-world datasets show that DCGL consistently outperforms state-of-the-art methods, yielding substantial improvements in sparse scenarios while maintaining precision for active users. Our code is available at this https URL. 

---
# BioProVLA-Agent: An Affordable, Protocol-Driven, Vision-Enhanced VLA-Enabled Embodied Multi-Agent System with Closed-Loop-Capable Reasoning for Biological Laboratory Manipulation 

**Authors**: Zhaohui Du, Zhe Wang, Hongmei Fei, Xiwen Cao, Ting Xiao, Qi Wang, Huanbo Jin, Jiaming Gu, Quan Lu, Zhe Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07306)  

**Abstract**: Biological laboratory automation can reduce repetitive manual work and improve reproducibility, but reliable embodied execution in wet-lab environments remains challenging. Protocols are often unstructured, labware is frequently transparent or reflective, and multi-step procedures require state-aware execution beyond one-shot instruction following. Existing robotic systems often rely on costly hardware, fixed workflows, dedicated instruments, or robotics-oriented interfaces. Here, we introduce BioProVLA-Agent, an affordable, protocol-driven, vision-enhanced embodied multi-agent system enabled by Vision-Language-Action (VLA) models for biological manipulation. The system uses protocols as the task interface and integrates protocol parsing, visual state verification, and embodied execution in a closed-loop workflow. A Tailored LLM Protocol Agent converts protocols into verifiable subtasks; a VLM-RAG Verification Agent assesses readiness and completion using observations, robot states, retrieved knowledge, and success/failure examples; and a VLA Embodied Agent executes verified subtasks through a lightweight policy. To improve robustness under wet-lab visual perturbations, we develop AugSmolVLA, an online augmentation strategy targeting transparent labware, reflections, illumination shifts, and overexposure. We evaluate the system on a hierarchical benchmark covering 15 atomic tasks, 6 composite workflows, and 3 bimanual tasks, including tube loading, sorting, waste disposal, cap twisting, and liquid pouring. Across normal and high-exposure settings, AugSmolVLA improves execution stability over ACT, X-VLA, and the original SmolVLA, especially for precise placement, transparent-object manipulation, composite workflows, and visually degraded scenes. These results suggest a practical route toward accessible, protocol-centered, and verification-capable embodied AI for biological manipulation. 

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
# Sword: Style-Robust World Models as Simulators via Dynamic Latent Bootstrapping for VLA Policy Post-Training 

**Authors**: Jiaxuan Gao, Yongjian Guo, Zhong Guan, Wen Huang, Wanlun Ma, Xi Xiao, Junwu Xiong, Sheng Wen  

**Link**: [PDF](https://arxiv.org/pdf/2605.07288)  

**Abstract**: The integration of Vision-Language-Action (VLA) models with World Models has gained increasing attention. One representative approach treats learned World Models as generative simulators, enabling policy optimization entirely within "imagination." However, when deployed as simulators for specific environments such as the LIBERO benchmark, existing World Models often suffer from poor generalization and long-horizon error accumulation. During closed-loop rollouts, these models are highly sensitive to initial-state perturbations; minor changes in color, illumination, and other visual factors can trigger cascading hallucinations, leading to severe blurriness or overexposure. Moreover, long-horizon error accumulation further degrades the quality and fidelity of predicted future states. These issues limit the reliability of World Models as simulators. To mitigate these problems, we propose Sword, a robust World Model framework. Our method introduces Structure-Guided Style Augmentation to disentangle the visual textures of interactive environments from task-relevant dynamics, thereby improving generalization. We further propose Dynamic Latent Bootstrapping, which maintains consistency between training and inference while keeping memory consumption low. Extensive experiments on the LIBERO benchmark show that our method significantly outperforms the baseline WoVR in terms of generalization, generation quality, robustness, fidelity, and the success rate of reinforcement-learning post-training for VLA models. 

---
# Mask2Cause: Causal Discovery via Adjacency Constrained Causal Attention 

**Authors**: Omar Muhammad, Pasupuleti Dhruv Shivkant, Deepak N. Subramani  

**Link**: [PDF](https://arxiv.org/pdf/2605.07280)  

**Abstract**: Leveraging deep learning for causal discovery in time series remains challenging because existing neural methods predominantly rely on component-wise architectures that fail to capture shared system dynamics or employ decoupled post-hoc graph extraction that risks overfitting to spurious correlations. We propose $\textbf{Mask2Cause}$, an end-to-end framework that recovers the underlying causal graph directly during the forecasting forward pass. Our approach introduces an Inverted Variable Embedding and an Adjacency-Constrained Masked Attention mechanism, trained with homoscedastic or heteroscedastic objectives to capture causal influences in both mean and variance. Empirical results on diverse benchmarks, from synthetic chaotic dynamics to realistic biological simulations, demonstrate state-of-the-art causal discovery with significantly reduced parameter complexity compared to standard baselines. We further show that inferred causal structures can be used to reduce parameter count of forecasting models by more than 70% on average while maintaining predictive accuracy. 

---
# Predictive but Not Plannable: RC-aux for Latent World Models 

**Authors**: Wenyuan Li, Guang Li, Keisuke Maeda, Takahiro Ogawa, Miki Haseyama  

**Link**: [PDF](https://arxiv.org/pdf/2605.07278)  

**Abstract**: A latent world model may achieve accurate short-horizon prediction while still inducing a latent space that is poorly aligned with planning. A key issue is spatiotemporal mismatch: these models are often trained with local predictive supervision, but deployed for long-horizon goal-directed search in latent spaces where Euclidean distance may not reflect what is reachable within a finite action budget. We present the Reachability-Correction auxiliary objective (RC-aux), a lightweight correction for this mismatch in reconstruction-free latent world models. RC-aux keeps the world-model backbone unchanged and adds planning-aligned supervision along two axes. Along the time axis, multi-horizon open-loop prediction trains the model beyond one-step consistency. Along the space axis, budget-conditioned reachability supervision, together with temporal hard negatives, encourages the latent space to distinguish states that are eventually reachable from those reachable within the current planning horizon. At test time, the learned reachability signal can also be used by a reachability-aware planner to favor trajectories that are both goal-directed and attainable under the available budget. We instantiate RC-aux on LeWorldModel and evaluate it under both continuation-training and matched-from-scratch settings. Across goal-conditioned pixel-control tasks and a LIBERO-Goal extension, RC-aux improves LeWM-style planning with modest additional cost. These results suggest that planning with latent world models depends not only on predictive accuracy, but also on whether the learned representation encodes the temporal and geometric structure required by downstream search. The code is available at this https URL. 

---
# Bifurcation Models: Learning Set-Valued Solution Maps with Weight-Tied Dynamics 

**Authors**: Caleb Jore, Jialin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07277)  

**Abstract**: Many scientific and combinatorial problems admit multiple correct solutions, not a single label. Standard supervised learning resolves this ambiguity by choosing one solution as the target, but this hidden selector can be arbitrary, discontinuous, and harder to learn than the underlying solution set. We study bifurcation models, a weight-tied dynamical view in which different initializations can converge to different stable equilibria, so the model represents an attractor landscape rather than one chosen branch. We prove that broad set-valued maps with locally Lipschitz branches can be represented by regular equilibrium dynamics and that the induced selectors are almost everywhere regular, while manual selectors can be arbitrarily irregular. Experiments on frustrated Ising models show that such dynamics can discover multiple valid equilibria without branch labels and outperform single-branch supervision. Allen--Cahn experiments further show that diversity is not automatic: it can be encouraged explicitly, but with an accuracy--diversity tradeoff. 

---
# From Clouds to Hallucinations: Atmospheric Retrieval Hijacking in Remote Sensing Vision-Language RAG 

**Authors**: Jiaju Han, Chao Li, Chengyin Hu, Qike Zhang, Xuemeng Sun, Xin Wang, Fengyu Zhang, Xiang Chen, Yiwei Wei, Jiahuan Long, Jiujiang Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.07273)  

**Abstract**: Multimodal RAG systems increasingly rely on vision-language retrievers to ground visual queries in external textual evidence. Existing adversarial studies on RAG mainly manipulate the retrieval corpus or memory, while attacks on vision-language and remote sensing models typically target end-task predictions. Input-space threats to the evidence retrieval stage of remote sensing multimodal RAG remain underexplored. To address this gap, we introduce CloudWeb, an atmospheric retrieval hijacking attack that modifies only the input image while keeping the retriever, generator, and knowledge base fixed at deployment. CloudWeb overlays parameterized cloud- and haze-like patterns on remote sensing images and optimizes them with a retrieval-oriented objective that pulls adversarial image embeddings toward target atmospheric evidence, suppresses source-scene evidence, enforces rank separation, and regularizes naturalness and coverage. To the best of our knowledge, this is the first study of retrieval-stage atmospheric evidence hijacking in remote sensing multimodal RAG. We evaluate CloudWeb on a seven-dataset remote sensing RAG benchmark with five CLIP-style retrievers, including GeoRSCLIP, RemoteCLIP, OpenAI CLIP, and OpenCLIP, together with downstream vision-language generators. Across retrievers, CloudWeb consistently outperforms clean retrieval, handcrafted atmospheric baselines, random cloud perturbations, and fixed variants in injecting weather-related evidence into top-ranked results. On GeoRSCLIP ViT-B/32, Weather@5 increases from 0.71\% to 43.29\%. Downstream generation further shows measurable weather hallucination and semantic shift, indicating that retrieval-stage hijacking can propagate to the final RAG response. These findings reveal a practical failure mode: natural-looking atmospheric changes can compromise evidence retrieval before generation begins. 

---
# Understanding Performance Collapse in Layer-Pruned Large Language Models via Decision Representation Transitions 

**Authors**: Boyu Shi, Chang Liu, ChuanBao Gao, Xu Yang, Xin Geng  

**Link**: [PDF](https://arxiv.org/pdf/2605.07271)  

**Abstract**: Layer pruning efficiently reduces Large Language Model (LLM) computational costs but often triggers sudden performance collapse. Existing representation-based analyses struggle to explain this mechanism. We propose studying pruning through decision representation. Focusing on multiple-choice tasks, we introduce two metrics, Decision Margin and Option Frequency, and an Iterative Pruning method to analyze layer-wise decision dynamics. Our findings reveal a sharp decision transition that partitions the network into two stages: a Silent Phase, where the model cannot yet predict the correct answer, and a Decisive Phase, where the correct prediction emerges. We also find that pruning the Decisive Phase has minimal impact, whereas pruning the Silent Phase triggers immediate performance collapse, highlighting its extreme sensitivity to structural changes. Therefore, we conclude that pruning-induced collapse stems from disrupting the Silent Phase, which prevents the critical decision transition from occurring. 

---
# Resource-Element Energy Difference for Noncoherent Over-the-Air Federated Learning 

**Authors**: Hao Chen, Zavareh Bozorgasl  

**Link**: [PDF](https://arxiv.org/pdf/2605.07263)  

**Abstract**: Over-the-air federated learning (OTA-FL) reduces uplink latency by exploiting waveform superposition, but conventional analog aggregation schemes typically require instantaneous channel state information (CSI), channel inversion, and coherent phase alignment, which can be difficult to maintain in practical wireless systems. This paper proposes resource-element energy difference (REED), a noncoherent aggregation primitive for continuous signed updates that avoids instantaneous CSI. REED maps the positive and negative parts of each real-valued update to transmit energies on two orthogonal resource elements with independent phase dithers, and the server estimates the signed aggregate from their energy difference. With only slow-timescale calibration of average channel powers, REED is unbiased for the desired signed sum and admits an exact closed-form variance under Rayleigh fading. We incorporate REED into full-participation FedAvg and prove a smooth nonconvex stationarity bound. Under an average per-client energy budget, the aggregation gain can be scheduled so that the REED-induced perturbation scales quadratically with the local stepsize, yielding the canonical (1/sqrt(T)) stationarity rate. Experiments on MNIST and Fashion-MNIST demonstrate that REED closely matches clean FedAvg and coherent CSIT aggregation in IID settings, while maintaining stable convergence with a moderate performance degradation under strong data heterogeneity. 

---
# Hard to Read, Easy to Jailbreak: How Visual Degradation Bypasses MLLM Safety Alignment 

**Authors**: Zhixue Song, Boyan Han, Yiwei Wang, Chi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07250)  

**Abstract**: Recent advancements in visual context compression enable MLLMs to process ultra-long contexts efficiently by rendering text into images. However, we identify a critical vulnerability inherent to this paradigm: lowering image resolution inadvertently catalyzes jailbreaking. Our experiments reveal that the safety defenses of SOTA models deteriorate sharply as resolution degrades, surprisingly persisting even when text remains legible. We attribute this to ``Cognitive Overload'', hypothesizing that the effort required to decipher degraded inputs diverts attentional resources from safety auditing. This phenomenon is consistent across various visual perturbations, including noise and geometric distortion. To address this, we propose a simple ``Structured Cognitive Offloading'' strategy that mitigates these risks by enforcing a serialized pipeline to decouple visual transcription from safety assessment. Our work exposes a significant risk in vision-based compression and provides critical insights for the secure design of future MLLMs. 

---
# Experience Sharing in Mutual Reinforcement Learning for Heterogeneous Language Models 

**Authors**: Xiaoze Liu, Dhananjay Ram, Yuting Zhang, Zhaoyang Zhang, Wei Xia, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2605.07244)  

**Abstract**: We introduce Mutual Reinforcement Learning, a framework for concurrent RL post-training in which heterogeneous LLM policies exchange typed experience while keeping separate parameters, objectives, and tokenizers. The framework combines a Shared Experience Exchange (SEE), Multi-Worker Resource Allocation (MWRA), and a Tokenizer Heterogeneity Layer (THL) that retokenizes text and aligns token-level traces across incompatible vocabularies. This substrate makes the experience-sharing design question operational across model families. We instantiate three controlled probes on top of GRPO: data-level rollout sharing via Peer Rollout Pooling (PRP), value-level advantage sharing via Cross-Policy GRPO Advantage Sharing (XGRPO), and outcome-level success transfer via Success-Gated Transfer (SGT). A contextual-bandit analysis characterizes their structural positions on a stability-support trade-off: PRP pays density-ratio variance and THL residual costs, XGRPO preserves on-policy actor support while changing scalar baselines, and SGT supplies a rescue-set score direction toward verified peer successes. In the evaluated regime, outcome-level sharing occupies the favorable point of this trade-off. 

---
# Reformulating KV Cache Eviction Problem for Long-Context LLM Inference 

**Authors**: Tho Mai, Joo-Young Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.07234)  

**Abstract**: Large language models (LLMs) support long-context inference but suffer from substantial memory and runtime overhead due to Key-Value (KV) Cache growth. Existing KV Cache eviction methods primarily rely on local attention weights, neglecting the influence of value representations, output projection, and inter-head interactions. In this work, we reformulate KV Cache eviction from a conventional head-wise, weight-averaging approach into an output-aware, layer-wise matrix multiplication approximation problem. We introduce LaProx, a novel eviction strategy that explicitly models the multiplicative interaction between attention maps and projected value states to accurately quantify token contributions while accounting for inter-head dependencies. Building on this metric, we propose the first unified eviction strategy that assigns globally comparable importance scores to tokens, enabling model-wide selection instead of local, head-wise decisions. Experimental results across 19 datasets on long-context benchmarks LongBench and Needle-In-A-Haystack demonstrate that our approach maintains model performance with only 5\% of the KV cache and consistently outperforms prior works across all configurations. Notably, our method achieves up to 2$\times$ accuracy loss reduction under extreme compression scenarios compared to existing state-of-the-art baselines with minimal overhead. 

---
# CASCADE: Context-Aware Relaxation for Speculative Image Decoding 

**Authors**: Selin Yildirim, Subhajit Dutta Chowdhury, Mohammad Mahdi Kamani, Vikram Appia, Deming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.07230)  

**Abstract**: Autoregressive generation is a powerful approach for high-fidelity image synthesis, but it remains computationally demanding and slow even on the most advanced accelerators. While speculative decoding has been explored to mitigate this bottleneck, existing approaches fail to achieve efficiency gains comparable to those observed in text generation. A key limitation is the target model's high uncertainty during image generation, which leads to high draft token rejection rates. In this work, we identify previously overlooked patterns in the target model's behavior that emerge naturally in tree-based speculative decoding. Specifically, we formalize two properties, semantic interchangeability and convergence, arising from the redundancies in the target model's hidden state representations. By capturing these redundancies across the depth and breadth of the predicted token tree, our method identifies principled opportunities for acceptance relaxation without requiring additional training. Additionally, we enhance standalone drafter performance by injecting the redundancy signals from the target model into drafter training with minimal modification. We evaluate our approach across multiple text-to-image models and drafter architectures. Results show that CASCADE achieves state-of-the-art speedups for drafter-based speculative decoding, with up to 3.6x acceleration, while maintaining image quality and text-prompt fidelity. 

---
# Same Brain, Different Prediction: How Preprocessing Choices Undermine EEG Decoding Reliability 

**Authors**: Dengzhe Hou, Zihao Wu, Lingyu Jiang, Zirui Li, Fangzhou Lin, Kazunori D. Yamada  

**Link**: [PDF](https://arxiv.org/pdf/2605.07212)  

**Abstract**: Electroencephalography (EEG) is a cornerstone of brain-computer interfaces and clinical neuroscience, yet deep learning models are typically trained and evaluated under a single, unreported preprocessing pipeline. We formalize preprocessing choices as a counterfactual intervention space and show that EEG predictions are surprisingly unstable under this space: across six datasets spanning four paradigms, up to 42% of trial-level predictions flip when only the preprocessing changes, a variability that standard uncertainty methods do not explicitly quantify because they condition on a fixed preprocessing pipeline. We provide three tools to make this instability measurable, decomposable, and reducible. First, a Walsh-Hadamard decomposition of the 2^7 pipeline space reveals that sensitivity is near-additive in practice under the binary intervention design, enabling efficient step-by-step optimization. Second, we introduce Preprocessing Uncertainty (PU), a per-trial diagnostic that captures a dimension of instability complementary to model-based confidence. Third, we study Normalized Adaptive PGI (NA-PGI), a graph-structured regularizer that exploits the compositional structure of preprocessing interventions as one mitigation strategy with clear scope conditions. 

---
# HARMONY: Bridging the Personalization-Generalization Gap by Mitigating Representation Skew in Heterogeneous Split Federated Learning 

**Authors**: Jiseok Youn, You Rim Choi, Goodsol Lee, Sangtae Ha, Hyung-Sin Kim, Saewoong Bahk  

**Link**: [PDF](https://arxiv.org/pdf/2605.07211)  

**Abstract**: Mobile devices face diverse resource constraints and non-IID data class distributions, requiring fast on-device inference for local in-distribution (ID) classes and on-demand remote support for client-specific out-of-distribution (OOD) classes. Hybrid split federated learning (Hybrid SFL) couples personalized client-side front ends (supporting early exit) with a generalized server-side backend for fallback inference, balancing accuracy and cost. However, under client architectural heterogeneity, the existing hybrid SFL suffers from representation skew, where features from customized extractors fail to align in the shared space, leading to a sharp degradation in the server model responsible for OOD prediction. We propose HARMONY, the first hybrid SFL framework to support heterogeneous client architectures. HARMONY modifies meta-learning to simulate diverse extractors across parameters and architectures, and to learn to personalize. To mitigate representation skew, HARMONY conducts server-side contrastive learning to align extracted features, neither sacrificing clients' personalization nor sharing raw labels. Compared to the state of the art across multiple datasets and model families, HARMONY improves test accuracy by up to 43.0%/28.3% without/with OOD, respectively, while maintaining acceptable latency. 

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
# Closed-Form Linear-Probe Dataset Distillation for Pre-trained Vision Models 

**Authors**: Bincheng Peng, Guang Li, Ping Liu, Takahiro Ogawa, Miki Haseyama  

**Link**: [PDF](https://arxiv.org/pdf/2605.07194)  

**Abstract**: Dataset distillation compresses a large training set into a small synthetic set that preserves downstream training utility. While most existing methods target training networks from scratch, modern visual transfer learning often uses frozen pre-trained encoders followed by lightweight linear probing. Existing distillation methods for this setting either unroll iterative linear-probe updates with trajectory-based gradient matching, or rely on closed-form formulations originally designed for from-scratch training with neural-tangent-kernel (NTK) approximations. Neither route exploits the fact that frozen-feature linear probing admits a closed-form solution determined directly by the pre-trained features themselves, with no infinite-width approximation and no inner-loop trajectory. We propose Closed-Form Linear-Probe Dataset Distillation (CLP-DD), a bilevel formulation that computes the linear probe induced by the synthetic set with a sample-space kernel ridge solver. The synthetic images are then updated by evaluating this induced classifier on real features through a temperature-scaled softmax cross-entropy, where the classifier columns act as learned class anchors in feature space. We further show that the choice of outer objective is decisive: pairing the closed-form inner solver with a standard MSE outer loss substantially underperforms trajectory-based methods, while the discriminative outer loss closes most of the gap. On ImageNet-100 with four pre-trained backbones, CLP-DD substantially improves over LGM without DSA and approaches LGM with DSA at a fraction of the computational cost. On ImageNet-1K, CLP-DD matches or surpasses LGM with DSA on three of four backbones while running roughly $14\times$ faster and using less than one-eighth of the GPU memory. 

---
# The Text Uncanny Valley: Non-Monotonic Performance Degradation in LLM Information Retrieval 

**Authors**: Zekai Tong, Ruiyao Xu, Aryan Shrivastava, Chenhao Tan, Ari Holtzman  

**Link**: [PDF](https://arxiv.org/pdf/2605.07186)  

**Abstract**: Existing Large Language Model (LLM) benchmarks primarily focus on syntactically correct inputs, leaving a significant gap in evaluation on imperfect text. In this work, we study how word-boundary corruption affects how LLMs detect targeted information. By inserting whitespace characters within words to break them into fragments, LLMs' detection accuracy follows a U-shaped curve with the increase in insertion rate. We refer to this curve as the Text Uncanny Valley. To explain such observation, we propose a mode transition hypothesis: LLMs operate in a word-level mode for near-normal text and a character-level mode for heavily fragmented text, with the valley marking the disordered transition where neither mode is effective. Four experiments and one analysis are consistent with this account: in-context learning fails to rescue valley-bottom performance; regularizing the perturbation substantially reduces the U-shape; a math reasoning task replicates the U-shape for Gemini 3.0 Flash but not for stronger models, suggesting the effect is attenuated when tasks rely less on exact lexical alignment; and tokenization entropy peaks before the F1 minimum, consistent with a regime-conflict interpretation. These findings reveal a failure mode invisible to clean-text benchmarks yet directly relevant to any deployment scenario involving noisy or uncurated text inputs. 

---
# HyperEyes: Dual-Grained Efficiency-Aware Reinforcement Learning for Parallel Multimodal Search Agents 

**Authors**: Guankai Li, Jiabin Chen, Yi Xu, Xichen Zhang, Yuan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07177)  

**Abstract**: Existing multimodal search agents process target entities sequentially, issuing one tool call per entity and accumulating redundant interaction rounds whenever a query decomposes into independent sub-retrievals. We argue that effective multimodal agents should search wider rather than longer: dispatching multiple grounded queries concurrently within a round. To this end, we present HyperEyes, a parallel multimodal search agent that fuses visual grounding and retrieval into a single atomic action, enabling concurrent search across multiple entities while treating inference efficiency as a first-class training objective. HyperEyes is trained in two stages. For cold-start supervision, we develop a Parallel-Amenable Data Synthesis Pipeline covering visual multi-entity and textual multi-constraint queries, curating efficiency-oriented trajectories via Progressive Rejection Sampling. Building on this, our central contribution, a Dual-Grained Efficiency-Aware Reinforcement Learning framework, operates at two levels. At the macro level, we propose TRACE (Tool-use Reference-Adaptive Cost Efficiency), a trajectory-level reward whose reference is monotonically tightened during training to suppress superfluous tool calls without restricting genuine multi-hop search. At the micro level, we adapt On-Policy Distillation to inject dense token-level corrective signals from an external teacher on failed rollouts, mitigating the credit-assignment deficiency of sparse outcome rewards. Since existing benchmarks evaluate accuracy as the sole metric, omitting inference cost, we introduce IMEB, a human-curated benchmark of 300 instances that jointly evaluates search capability and efficiency. Across six benchmarks, HyperEyes-30B surpasses the strongest comparable open-source agent by 9.9% in accuracy with 5.3x fewer tool-call rounds on average. 

---
# Learning Multi-Relational Graph Representations for DNA Methylation-Based Biological Age Estimation 

**Authors**: Qing Qing, Xikun Zhang, Zhongyuan Zhang, Jiarui Liu, Xingtong Yu, Xiaotao Shen, Ziqi Xu, Qixin Zhang, Zhe Wang, Renqiang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2605.07175)  

**Abstract**: Aging clocks aim to estimate biological age, a measure of physiological state distinct from chronological age, from observable biomarkers, and are widely used for health assessment and disease analysis. DNA methylation is a particularly informative biomarker due to its stability and strong association with aging, and recent learning-based approaches have improved predictive performance. However, most existing methods treat CpG sites as independent features, overlooking the complex and heterogeneous biological relationships among them. We propose RelAge-GNN, a multi-relational graph neural network framework for DNA methylation-based age prediction. Our method constructs three complementary graphs capturing co-methylation patterns, genomic co-localization, and gene-level associations among CpG sites. Each graph is modeled by an independent GNN branch, and a learnable gating mechanism adaptively fuses the resulting representations. Experiments on large-scale datasets show that RelAge-GNN achieves competitive accuracy and stronger correlation with chronological age compared to state-of-the-art methods. Moreover, the model exhibits improved sensitivity in detecting age acceleration across diverse disease cohorts, highlighting its potential utility for disease characterization. Finally, through post hoc interpretability analyses, we quantify the contributions of different relational structures and CpG sites, providing biologically meaningful insights and suggesting potential directions for aging-related research. Our code is available at: this https URL. 

---
# DPG-CD: Depth-Prior-Guided Cross-Modal Joint 2D-3D Change Detection 

**Authors**: Luqi Zhang, Zhen Dong, Bisheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07151)  

**Abstract**: Urban spatial evolution is manifested not only through horizontal expansion but also through vertical structural changes. Consequently, jointly capturing 2D semantic changes and 3D height changes is essential for urban morphology analysis and emergency management. In practical scenarios, collecting 3D observations is often constrained by high acquisition costs and the inability to support frequent updates. The multi-temporal cross-modal input consisting of pre-event Digital Surface Model (DSM) and post-event imagery provides a practical solution for 3D change detection in high-frequency urban monitoring, disaster assessment, and emergency response scenarios. However, this setting remains challenging as imagery and DSM data exhibit significant spectral-geometric representation gaps. Moreover, modality differences may be confused with actual changes, and robust change detection requires effective fusion of semantic and geometric features from multi-temporal data. In this paper, we propose DPG-CD, a depth-prior-guided multi-temporal cross-modal fusion framework for joint 2D semantic and 3D height change detection. Specifically, an estimated depth prior is introduced into the imagery to mitigate the modality gap with DSM. A gated fusion mechanism then selectively injects geometric cues from depth prior while preserving discriminative spectral representations. Subsequently, a multi-stage cross-temporal cross-modal feature fusion architecture is employed to extract change-aware features. Finally, a multi-task decoder jointly predicts 2D semantic changes and 3D height changes, complemented by an auxiliary DSM prediction task to improve structural consistency and height estimation accuracy. Experiments on two public datasets, Hi-BCD and 3DCD, and a new dataset, NYC-MMCD, demonstrate that DPG-CD outperforms state-of-the-art methods on both 2D and 3D change detection tasks. 

---
# MathlibPR: Pull Request Merge-Readiness Benchmark for Formal Mathematical Libraries 

**Authors**: Zixuan Xie, Xinyu Liu, Shangtong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07147)  

**Abstract**: The ecosystem of Lean and Mathlib has become the de facto standard for large language model (LLM) assisted formal reasoning with remarkable successes in recent years. Those successes, however, only consume Mathlib as an essential dependency but do not directly contribute to it. In the meantime, the growth of Mathlib has recently been bottlenecked by the review process, which requires human reviewers to judge whether proposed pull requests (PRs) follow the Mathlib's conventions and are worth integrating as part of a shared mathematical infrastructure. This leads to our central question: can LLMs help review Mathlib PRs? To this end, we introduce MathlibPR, a benchmark built from real Mathlib4 PR histories. We further propose a staged evaluation protocol and use it to evaluate both LLM models (e.g., DeepSeek, Qwen, Goedel, and Kimina) and LLM agents (e.g., Codex and Claude Code). Surprisingly, both LLM models and LLM agents struggle to distinguish merge-ready PRs from build-passing PRs that were revised or never merged. By turning Mathlib PR histories into a supervised signal, MathlibPR provides a step toward reviewer assistants and reward models that could help evaluate PRs and steer LLMs toward producing merge-ready Mathlib contributions. 

---
# Qwen3-VL-Seg: Unlocking Open-World Referring Segmentation with Vision-Language Grounding 

**Authors**: Yuan Yao, Qiushi Yang, Humen Zhong, Jiangning Wei, Yifang Men, Shuai Bai, Miaomiao Cui, Zhibo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07141)  

**Abstract**: Open-world referring segmentation requires grounding unconstrained language expressions to precise pixel-level regions. Existing multimodal large language models (MLLMs) exhibit strong open-world visual grounding, but their outputs remain limited to sparse bounding-box coordinates and are insufficient for dense visual prediction. Recent MLLM-based segmentation methods either directly predict sparse contour coordinates, struggling to reconstruct continuous object boundaries, or rely on external segmentation foundation models such as the Segment Anything Model (SAM), introducing substantial architectural and deployment overhead. We present Qwen3-VL-Seg, a parameter-efficient framework that treats the MLLM-predicted box as a semantically grounded structural prior and decodes it into pixel-level referring segmentation. At its core, a lightweight box-guided mask decoder combines multi-scale spatial feature injection, spatial-semantic query construction, box-guided high-resolution pixel fusion, and iterative mask-aware query refinement, introducing only 17M parameters (about 0.4\% of the base model). For scalable open-world training, we construct SA1B-ORS, an SA-1B-derived dataset with two subsets: SA1B-CoRS (category-oriented samples) and SA1B-DeRS (descriptive, instance-specific samples). For evaluation, we curate ORS-Bench, a manually screened benchmark with in-distribution and out-of-distribution subsets covering diverse referring expression types. Extensive experiments on referring expression segmentation, visual grounding, and ORS-Bench show that Qwen3-VL-Seg performs strongly across closed-set and open-world settings, with clear advantages on language-intensive instructions and strong out-of-distribution generalization. Evaluations on general multimodal benchmarks further show that the model broadly preserves general-purpose multimodal competence after segmentation-oriented adaptation. 

---
# Neurosymbolic Framework for Concept-Driven Logical Reasoning in Skeleton-Based Human Action Recognition 

**Authors**: Talha Ilyas, Deval Mehta, Zongyuan Ge  

**Link**: [PDF](https://arxiv.org/pdf/2605.07140)  

**Abstract**: Skeleton-based human activity recognition has achieved strong empirical performance, yet most existing models remain black boxes and difficult to interpret. In this work, we introduce a neurosymbolic formulation of skeleton-based HAR that reframes action recognition as concept-driven first-order logical reasoning over motion primitives. Our framework bridges representation learning and symbolic inference by grounding first-order logic predicates in learnable spatial and temporal motion concepts. Specifically, we employ a standard spatio-temporal skeleton encoder to extract latent motion representations, which are then mapped to interpretable concept predicates via a spatio-temporal concept decoder that explicitly separates pose-centric and dynamics-centric abstractions. These concept predicates are composed through differentiable first-order logic layers, enabling the model to learn human-readable logical rules that govern action semantics. To impose semantic structure on the learned concepts, we align skeleton representations with LLM-derived descriptions of atomic motion primitives, establishing a shared conceptual space for perception and reasoning. Extensive experiments on NTU RGB+D 60/120 and NW-UCLA demonstrate that our approach achieves competitive recognition performance while providing explicit, interpretable explanations grounded in logical structure. Our results highlight neurosymbolic reasoning as an effective paradigm for interpretable spatio-temporal action understanding. Code: this https URL 

---
# Structural Rationale Distillation via Reasoning Space Compression 

**Authors**: Jialin Yang, Jiankun Wang, Jiajun Wu, Henry Leung, Jiayu Zhou, Steve Drew  

**Link**: [PDF](https://arxiv.org/pdf/2605.07139)  

**Abstract**: When distilling reasoning from large language models (LLMs) into smaller ones, teacher rationales for similar problems often vary wildly in structure and strategy. Like a chef who makes the same dish differently each time, this inconsistency burdens the student with noisy supervision that is hard to internalize. We propose Distillation through Reasoning Path Compression (D-RPC), which constrains the teacher to follow a compact, dynamically maintained bank of reusable high-level reasoning paths. For each training question, D-RPC retrieves the most relevant path and conditions the teacher to follow it, producing rationales that are consistent across similar problems yet diverse enough to cover different problem types. A PAC-Bayes analysis formalizes the resulting trade-off between bank size and coverage: smaller banks reduce supervision entropy but risk coverage gaps, and the generalization bound identifies an optimal intermediate size confirmed by our ablations. Across five math and commonsense reasoning benchmarks with two student models, D-RPC consistently outperforms chain-of-thought distillation, freeform rationale generation, direct distillation, and structured-supervision baselines, while using fewer tokens than template-heavy alternatives. 

---
# Adaptive Negative Reinforcement for LLM Reasoning:Dynamically Balancing Correction and Diversity in RLVR 

**Authors**: Yash Ingle, Jaival Chauhan, Ankit Yadav, Sudhakar Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2605.07137)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has become a highly effective method for improving the reasoning abilities of Large Language Models (LLMs). Recent research shows that Negative Sample Reinforcement (NSR) -- which focuses on penalizing incorrect steps rather than simply rewarding correct ones -- can match or even exceed the performance of more complex frameworks like PPO and GRPO across the entire Pass@k spectrum. However, current NSR techniques usually apply a fixed penalty throughout the training process and treat every incorrect response with the same weight.
To address these limitations, we propose two extensions to the NSR framework: Adaptive Negative Sample Reinforcement. Rather than using a fixed update rule, A-NSR uses time-dependent scheduling functions. In the initial training phases, the system focuses heavily on correcting errors to stabilize the model. As training continues, it shifts toward more subtle and controlled updates. We also introduce Confidence-Weighted Negative Reinforcement, which operates on the principle that different mistakes carry different levels of importance. CW-NSR assigns specific penalty weights based on the model's normalized sequence likelihood. If the model is highly confident in a wrong path, it receives a larger penalty and for uncertain errors -- where the model is effectively exploring -- are penalized less strictly. Our formal analysis shows how these mechanisms govern token-level updates, allowing the model to leverage prior-guided probability redistribution while providing a natural defense against overfitting. We evaluated these methods on difficult reasoning datasets, including MATH, AIME 2025, and AMC23, using the Qwen2.5-Math-1.5B architecture. 

---
# Region4Web: Rethinking Observation Space Granularity for Web Agents 

**Authors**: Donguk Kwon, Dongha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.07134)  

**Abstract**: Web agents perceive web pages through an observation space, yet its granularity has remained an underexamined design choice. Existing work treats observation at the same element-level granularity as the action space, leaving the page's functional organization implicit and forcing the agent to infer it from element-level signals at every step. We argue observation should instead operate at the granularity of functional regions, parts of the page that each serve a distinct purpose. We propose Region4Web, a framework that reorganizes the AXTree into functional regions through hierarchical decomposition and semantic abstraction, exposing the page's functional organization as the basis for page state understanding. Moreover, we propose PageDigest, a web-specific inference pipeline that delivers this region-level observation to the actor agent as a compact per-page digest that persists across steps. On the WebArena benchmark, PageDigest substantially reduces observation length while improving overall task success rate across diverse backbone large language models (LLMs) and established agent methods, regardless of backbone capacity. These results show that operating at the granularity of functional regions delivers a more compact and informative basis for the actor agent than element-level processing alone. 

---
# GAD in the Wild: Benchmarking Graph Anomaly Detection under Realistic Deployment Challenges 

**Authors**: Jingjing Zhou, Shiyu Huang, Qing Qing, Zuquan Yuan, Huafei Huang, Ziqi Xu, Mingliang Hou, Xikun Zhang, Renqiang Luo, Ivan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.07133)  

**Abstract**: Graph Anomaly Detection (GAD) is a critical task in graph machine learning with vital applications in financial fraud detection and social platform governance. However, existing GAD benchmarks are often restricted to small-scale, curated graphs with relatively balanced anomaly ratios, leaving a substantial gap between academic evaluation and real-world deployment. To bridge this gap, we present a multi-dimensional benchmark that systematically evaluates GAD models under three deployment-relevant challenges: million-scale graphs, extreme anomaly scarcity, and missing node attributes. We derive a family of controlled benchmark variants from five diverse graphs, including two native industrial-scale datasets with over 3.7 million nodes. Our extensive evaluation of nine representative GAD models reveals three major limitations: (1) most GNN-based methods fail to scale to million-node graphs due to prohibitive memory requirements; (2) detection performance drops sharply under realistic anomaly ratios (e.g., 0.1\%), often resulting in zero recall; and (3) reconstruction-based models are highly sensitive to attribute imputation strategies. Our findings suggest that strong performance in laboratory settings does not guarantee robustness in production environments. We release this benchmark and empirical evaluation as a diagnostic testbed to promote the development of robust and scalable GAD systems for large-scale, imperfect graphs encountered in practice. Code is available at this https URL. 

---
# RRCM: Ranking-Driven Retrieval over Collaborative and Meta Memories for LLM Recommendation 

**Authors**: Shijun Li, Wooseong Yang, Yu Wang, Tianxin Wei, Joydeep Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2605.07129)  

**Abstract**: Large Language Models (LLMs) have emerged as a promising paradigm for next-generation recommender systems, offering strong semantic understanding and natural-language reasoning abilities. Despite recent progress, current LLM-based recommenders still face key challenges in constructing decision-relevant contexts from heterogeneous evidence. First, existing methods often rely on fixed context construction strategies: collaborative behavioral evidence and item-side metadata are typically incorporated through predefined prompts, static retrieval pipelines, or handcrafted injection mechanisms, making it difficult to determine what information is truly beneficial for each instance. Second, heterogeneous evidence introduces a severe context-efficiency bottleneck. Rich metadata and collaborative interaction records can quickly overwhelm the context window, while aggressive compression or heuristic filtering may discard fine-grained evidence critical for accurate recommendation. To address these challenges, we propose RRCM, a ranking-driven retrieval-and-reasoning framework over collaborative and metadata memories for LLM-based agentic recommendation. RRCM starts from a lightweight user-history context and learns whether to recommend directly, retrieve collaborative evidence, retrieve item metadata, or interleave both through reasoning. Both memories are represented in natural language and accessed through a unified retrieval interface, enabling flexible evidence acquisition without handcrafted CF injection or fixed retrieval rules. We optimize this memory-reading policy with an outcome-only ranking reward, instantiated using group relative policy optimization, so that retrieval decisions are directly driven by final top-k recommendation quality. Extensive experiments show that RRCM significantly outperforms traditional baselines and diverse LLM-based recommendation approaches. 

---
# An Embarrassingly Simple Graph Heuristic Reveals Shortcut-Solvable Benchmarks for Sequential Recommendation 

**Authors**: Haoyu Han, Li Ma, Hanbing Wang, Bingheng Li, Daochen Zha, Chun How Tan, Huiji Gao, Xin Liu, Stephanie Moyerman, Sanjeev Katariya, Hui Liu, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07125)  

**Abstract**: Sequential recommendation has increasingly shifted toward generative recommenders that combine sequential patterns with semantic item information. Yet these methods are often evaluated on a small set of widely used benchmarks, raising a key question: do these benchmarks actually require the advanced modeling capabilities that modern generative recommenders claim to provide? We conduct a benchmark audit with an intentionally simple graph heuristic. Starting from only the last one or two interacted items, it retrieves candidates from a few-hop item-transition graph and ranks them by item-feature similarity. Despite using no sequence encoder, generative objective, or training, this heuristic matches or outperforms many modern baselines, with relative NDCG@10 improvements of 38.10% and 44.18% over the best competing baseline on Amazon Review Sports and CDs.
We show that this behavior reflects shortcut solvability rather than an artifact of one heuristic. We identify three shortcut structures that can make next-item prediction easier than expected: low-branching local transitions, feature-smooth transitions, and limited dependence on long user histories. These shortcuts need not appear together; even one or two strong signals can make simple local retrieval highly competitive, while weakening them makes the benefits of more sophisticated models clearer. Across 14 datasets, model rankings vary substantially with dataset properties, yet the heuristic remains competitive on 10 of them. Our findings suggest that strong performance on standard benchmarks does not always demonstrate advanced sequential, semantic, or generative modeling ability. We call for more careful dataset selection and dataset-level diagnostic analysis when using benchmarks to support claims about new recommendation models. 

---
# Stabilized neural Hamilton--Jacobi--Bellman solvers: Error analysis and applications in model-based reinforcement learning 

**Authors**: Minseok Kim, Yeongjong Kim, Namkyeong Cho, Yeoneung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.07116)  

**Abstract**: Physics-informed neural solvers offer a promising route to model-based reinforcement learning in continuous time, where optimal feedback synthesis is governed by Hamilton--Jacobi--Bellman (HJB) equations. Practical implementations often occupy a regime that is neither a classical grid method nor a continuous-PDE PINN: the value function is represented by a neural network, finite-difference HJB policy-evaluation operators are evaluated by network queries at shifted points, and residuals are minimized by random continuous collocation. This regime preserves the stabilized finite-difference policy-evaluation structure while avoiding grid-based value unknowns. We develop an error theory for this hybrid regime. Interpreting finite differences as shift operators acting on neural networks, we prove a population $L^2$ stability estimate for one policy-evaluation step with learned dynamics. The bound separates residual error, initial and exterior-collar mismatch, policy mismatch, and model-identification error, with an explicit gradient amplification factor for learned dynamics, while the underlying linear evaluation stability remains free of hidden inverse-viscosity blow-up. We further give a finite-sample collocation certificate and a conditional multi-step propagation result through greedy policy improvement. Experiments on compact-control LQR upto 64 dimensions, Allen--Cahn control, pendulum, Hopper, and 3D quadrotor benchmarks compare against representative model-based and model-free RL baselines, demonstrating the predicted residual, policy-mismatch, and learned-model error trends. 

---
# Beyond LoRA vs. Full Fine-Tuning: Gradient-Guided Optimizer Routing for LLM Adaptation 

**Authors**: Haozhan Tang, Xiuqi Zhu, Xinyin Zhang, Boxun Li, Virginia Smith, Kevin Kuo  

**Link**: [PDF](https://arxiv.org/pdf/2605.07111)  

**Abstract**: Recent literature on fine-tuning Large Language Models highlights a fundamental debate. While Full Fine-Tuning (FFT) provides the representational plasticity required for high-entropy knowledge injection, Low-Rank Adaptation (LoRA) can match or surpass FFT performance because many tasks only require updates in a low-rank space and benefit from LoRA's additional regularization. Through empirical evaluation across diverse tasks (SQL, Medical QA, and Counterfactual Knowledge) and varying language models (Gemma-3-1B, Qwen2.5-1.5B, and Qwen2.5-3B), we verify both trends and demonstrate that relying solely on either static architecture is structurally limited. To address this challenge, we propose a Mixture of LoRA and Full (MoLF) Fine-Tuning, a unified framework that enables continuous navigation between both training regimes. MoLF dynamically routes updates between FFT and LoRA at the optimizer level to ensure that exact gradient signals are available to both experts throughout training, yielding stable training dynamics. For memory-constrained environments, we also introduce MoLF-Efficient, which freezes base weights and only routes updates among a pair of LoRA experts of potentially varying rank. Our evaluations show that MoLF either improves on or stays within $1.5\%$ of the better of FFT and LoRA across all settings, while MoLF-Efficient outperforms prior adaptive LoRA approaches by up to $20\%$ on Fact and $9\%$ on Med and SQL. 

---
# Query-efficient model evaluation using cached responses 

**Authors**: Hayden Helm, Ben Johnson, Carey Priebe  

**Link**: [PDF](https://arxiv.org/pdf/2605.07096)  

**Abstract**: Evaluating a new model on an existing benchmark is often necessary to understand its behavior before deployment. For modern evaluation frameworks, generating and evaluating a response for all queries can be prohibitively expensive. In practice, responses from previously-evaluated models are often cached -- creating a potential opportunity to use this additional information to decrease the number of queries required to accurately evaluate a new model. In this paper, we introduce an approach for predicting benchmark performance that leverages cached model responses based on the Data Kernel Perspective Space (DKPS), a method for quantifying the relationship between models in the black-box setting. Theoretically, we show that DKPS-based methods are query-efficient under certain conditions. Empirically, we demonstrate that DKPS-based methods achieve the same mean absolute error as baselines with a substantially decreased query budget. We conclude by proposing an offline method for selecting a set of queries that maximizes the goodness-of-fit on reference models, improving prediction accuracy over random query selection. 

---
# The Translation Tax Is Not a Scalar: A Counterfactual Audit of English-Source Cue Inheritance in Chinese Multilingual Benchmarks 

**Authors**: Zezheng Lin, Fengming Liu, Handi Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.07093)  

**Abstract**: The Translation Tax is often treated as a scalar: translated benchmarks are assumed to inflate scores by preserving English-source cues. We audit this claim in an English-to-Chinese setting. Three proxy estimators disagree: back-translation gaps are small and parser-fragile; cue-score calibration does not predict item-level gains; and a six-model native-control comparison shows model-family rather than uniform benchmark effects. We add a same-item LLM-naturalization stress test that holds answer, options, and content fixed while rewriting Chinese surface form. After correcting a prompt-construction bug, this contrast no longer supports a model-family interaction, but it preserves a residue dose-response: high-residue items benefit while low-residue items do not. The result is not a single Translation Tax, but a set of estimator- and item-dependent validity risks. We release per-cell evidence, the naturalization protocol, human QC, and a reporting checklist for translated multilingual benchmark papers. 

---
# Learning Visual Feature-Based World Models via Residual Latent Action 

**Authors**: Xinyu Zhang, Zhengtong Xu, Yutian Tao, Yeping Wang, Yu She, Abdeslam Boularias  

**Link**: [PDF](https://arxiv.org/pdf/2605.07079)  

**Abstract**: World models predict future transitions from observations and actions. Existing works predominantly focus on image generation only. Visual feature-based world models, on the other hand, predict future visual features instead of raw video pixels, offering a promising alternative that is more efficient and less prone to hallucination. However, current feature-based approaches rely on direct regression, which leads to blurry or collapsed predictions in complex interactions, while generative modeling in high-dimensional feature spaces still remains challenging. In this work, we discover that a new type of latent action representation, which we refer to as *Residual Latent Action* (RLA), can be easily learned from DINO residuals. We also show that RLA is predictive, generalizable, and encodes temporal progression. Building on RLA, we propose *RLA World Model* (RLA-WM), which predicts RLA values via flow matching. RLA-WM outperforms both state-of-the-art feature-based and video-diffusion world models on simulation and real-world datasets, while being orders of magnitude faster than video diffusion. Furthermore, we develop two robot learning techniques that use RLA-WM to improve policy learning. The first one is a minimalist world action model with RLA that learns from actionless demonstration videos. The second one is the first visual RL framework trained entirely inside a world model learned from offline videos only, using a video-aligned reward and no online interactions or handcrafted rewards. Project page: this https URL 

---
# WiCER: Wiki-memory Compile, Evaluate, Refine Iterative Knowledge Compilation for LLM Wiki Systems 

**Authors**: Juan M. Huerta  

**Link**: [PDF](https://arxiv.org/pdf/2605.07068)  

**Abstract**: The LLM Wiki pattern, to compile and provide domain knowledge into a persistent artifact and serve it to LLMs via KV cache inference, promises context access at sub-second latency with zero retrieval failure. Realizing this requires solving the compilation gap: LLM compilation distilling raw documents into a wiki without catastrophically discarding critical facts. We characterize this gap across 17 RepLiQA domains (6,800 questions): we observe that full context KV cache inference outperforms RAG on curated knowledge (4.38 vs. 4.08 out of 5, 7.3 faster TTFT) but degrades below RAG at scale due to attention dilution, and blind compilation fails entirely (2.14 to 2.32 vs. 3.46, 53 to 60% catastrophic failure rate). To address the compilation gap, we propose WiCER (Wiki-memory Compile, Evaluate, Refine), an iterative algorithm inspired by counterexample-guided abstraction refinement (CEGAR) that closes this gap. WiCER evaluates compiled wikis against diagnostic probes, identifies dropped facts, and forces their preservation in subsequent compilations. One to two iterations recover 80% of lost quality (mean 3.24 vs. 3.47 for raw full-context across the 15 topics with baselines), reducing catastrophic failures by 55% relative. An ablation across all 17 topics confirms that targeted diagnosis (+0.95), not generic pinning (+0.16), drives the gains. All code and benchmarks are released for reproducible research. 

---
# Causal EpiNets: Precision-corrected Bounds on Individual Treatment Effects using Epistemic Neural Networks 

**Authors**: Gandharv Patil, Keyi Tang, Raquel Aoki, Leo Guelman  

**Link**: [PDF](https://arxiv.org/pdf/2605.07065)  

**Abstract**: Individual treatment effects are not point-identified from data. The Probability of Necessity and Sufficiency (PNS) circumvents this limitation by characterizing individual-level causality through intersection bounds derived from combined experimental and observational data. In finite samples, however, standard plug-in estimators systematically fail: they violate structural probability constraints and suffer from extremum bias induced by max-min operators, yielding spuriously narrow intervals. We propose a neural framework for finite-sample PNS estimation that resolves both pathologies. We introduce an anchored neural architecture that guarantees structural constraint satisfaction by construction. To correct extremum bias, we employ precision-corrected intersection-bound inference, leveraging Epistemic Neural Networks for scalable, high-dimensional uncertainty quantification. Empirical evaluations confirm that this approach maintains nominal coverage and exact constraint validity in high-dimensional regimes where standard estimators systematically undercover. 

---
# Dr. Post-Training: A Data Regularization Perspective on LLM Post-Training 

**Authors**: Pingbang Hu, Xueshen Liu, Z. Morley Mao, Jiaqi W. Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.07063)  

**Abstract**: Data selection methods address a critical challenge in LLM post-training: effectively leveraging scarce, high-fidelity target data alongside abundant but imperfectly aligned general training data. In this work, we move beyond the data-selection framing and introduce Dr. Post-Training (Data-Regularized Post-Training), a novel framework that reconceptualizes general training data as a data-induced regularizer that prevents overfitting to the scarce target objective, rather than serving as a pool for selection. Specifically, our framework proposes that at each training step, construct a feasible set of model update directions using the general training data, and project the model update direction specified by the scarce target data onto that feasible set. Standard training and existing data selection methods arise as special cases with different choices of the data-induced regularizer, and these methods correspond to different points on a bias--variance spectrum with different regularization strength. Building on this view, we propose a family of methods offering a richer design space and more flexible bias--variance tradeoffs. For practical LLM-scale use, we introduce careful system optimizations that realize these methods with minimal overhead. Extensive experiments across SFT, RLHF, and RLVR show that our methods consistently outperform state-of-the-art data selection baselines, and system benchmarks confirm their efficiency. 

---
# From Assistance to Agency: Rethinking Autonomy and Control in CI/CD Pipelines 

**Authors**: Marcus Emmanuel Barnes, Taher A. Ghaleb, Safwat Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2605.07062)  

**Abstract**: AI agents are assuming active roles in Continuous Integration and Continuous Deployment (CI/CD) workflows, yet the research community lacks a shared vocabulary for describing what it means for CI/CD to be agentic, how much decision authority is delegated, and where control should reside. This paper presents a vision of agentic CI/CD in which the central challenge is not improving task performance but designing authority transfer, defined as the delegation of operational decisions from human-controlled pipelines to agent systems under specified constraints and recourse mechanisms.
To structure this argument, we introduce a distinction between data-plane authority (localized interventions such as patch generation and test reruns) and control-plane authority (modifications to pipeline configuration, deployment policies, and approval gates). Drawing on research prototypes and industrial platforms, we show that current systems operate mainly at the data plane under bounded autonomy, with safety achieved through surrounding governance infrastructure rather than intrinsic agent guarantees. We identify three recurring patterns: constrained autonomy as the dominant design, external governance as the primary safety mechanism, and a widening gap between deployment momentum and evaluation methodology. We propose a research agenda in which control-plane safety and governance mechanisms represent the most urgent open problem, followed by formalization of autonomy boundaries, evaluation frameworks, and human--agent coordination. 

---
# Do Joint Audio-Video Generation Models Understand Physics? 

**Authors**: Zijun Cui, Xiulong Liu, Hao Fang, Mingwei Xu, Jiageng Liu, Zexin Xu, Weiguo Pian, Shijian Deng, Feiyu Du, Chenming Ge, Yapeng Tian  

**Link**: [PDF](https://arxiv.org/pdf/2605.07061)  

**Abstract**: Joint audio-video generation models are rapidly approaching professional production quality, raising a central question: do they understand audio-visual physics, or merely generate plausible sounds and frames that violate real-world consistency? We introduce AV-Phys Bench, a benchmark for evaluating physical commonsense in joint audio-video generation. AV-Phys Bench tests models across three scene categories: Steady State, Event Transition, and Environment Transition. It covers physics-grounded subcategories drawn from real-world scenes, plus Anti-AV-Physics prompts that deliberately request physically inconsistent audio-video behavior. Each generation is evaluated along five dimensions: visual semantic adherence, audio semantic adherence, visual physical commonsense, audio physical commonsense, and cross-modal physical commonsense. Across three proprietary and four open-source models, we find that Seedance 2.0 performs best overall, but all models remain far from robust physical understanding. Performance drops sharply on event-driven and environment-driven transitions, and even strong proprietary systems collapse on Anti-AV-Physics prompts. We further introduce AV-Phys Agent, a ReAct-style evaluator that combines a multimodal language model with deterministic acoustic measurement tools, producing rankings that closely align with human ratings. Our results identify cross-modal physical consistency and transition-driven scene dynamics as key open challenges for joint audio-video generation. 

---
# MedExAgent: Training LLM Agents to Ask, Examine, and Diagnose in Noisy Clinical Environments 

**Authors**: Yicheng Gao, Xiaolin Zhou, Yahan Li, Yue Zhao, Ruishan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07058)  

**Abstract**: Real-world clinical diagnosis is a complex process in which the doctor is required to obtain information from both interaction with the patient and conducting medical exams. Additionally, the doctor needs to adapt to different patient personas, as well as noisy and incomplete information that can happen at any time during the process. However, existing benchmarks for medical LLMs and methods for automatic diagnosis largely simplify this process by reducing it to single-turn question answering, noise-free conversations, or sequential exam making, etc., ignoring the interactive and uncertain nature of clinical diagnosis. In this paper, we aim to address this gap by formalizing clinical diagnosis as a Partially Observable Markov Decision Process (POMDP) with three action types: questioning the patient, ordering medical exams as tool calls, and issuing a diagnosis. We also introduce a systematic noise model comprising seven patient noise types and three exam noise types. Using our proposed environment, we train an effective diagnosis agent, \textbf{MedExAgent}, through a two-stage pipeline that first performs supervised finetuning on synthetic conversations structured after the Calgary-Cambridge model for clinical interviews, and then applies DAPO to optimize a composite reward capturing diagnostic accuracy, tool call quality, and exam cost including financial cost and patient discomfort. Through extensive experiments and ablation studies, we demonstrate that MedExAgent achieves diagnostic performance comparable to larger models while maintaining cost-efficient examination strategies. 

---
# Pan-FM: A Pan-Organ Foundation Model with Saliency-Guided Masking for Missing Robustness 

**Authors**: Qiangqiang Wu, Grace McIlvain, Zhou Yu, Junhao Wen  

**Link**: [PDF](https://arxiv.org/pdf/2605.07055)  

**Abstract**: Foundation models (FMs) have shown great promise in medical imaging, but most FMs are trained on unimodal data within isolated domains, such as brain MRI alone. Human aging and disease arise through coordinated biological processes across organs, therefore motivating multimodal FMs that learn whole-body representations. A key challenge, however, is that real-world multimodal biomedical data are often missing not at random, which can reduce power, limit generalizability, and introduce bias. We propose Pan-FM, a pan-organ foundation model pre-trained on imaging from seven organs (Brain, Heart, Adipose, Liver, Kidney, Spleen, and Pancreas) under realistic missing-organ scenarios. Pan-FM uses a unified backbone that handles organ missingness during both training and inference, and is pre-trained with masking-based self-distillation. We find that naive multimodal pre-training leads to dominant-organ shortcut learning bias, with the model over-relying on dominant organs such as adipose and heart. To address this, we introduce Saliency-Guided Masking (SGM), which uses the model attention distribution to adaptively mask dominant organs during pre-training, thus encouraging more balanced cross-organ, whole-body learning. Notably, SGM introduces negligible computational overhead and can be seamlessly integrated into existing self-supervised learning frameworks to improve multi-organ representation learning. On the UK Biobank, Pan-FM achieves stronger prediction across 13 disease categories and 14 single disease entities than single-organ and multi-organ baselines, with improved robustness under missing-organ settings. Pan-FM serves as a scalable solution to realistic modality-missingness in multimodal learning in system neuroscience and as a step toward more generalizable whole-body FMs. 

---
# GSM-SEM: Benchmark and Framework for Generating Semantically Variant Augmentations 

**Authors**: Jyotika Singh, Fang Tu, Aziza Mirzadova, Amit Agarwal, Hitesh Laxmichand Patel, Sandip Ghoshal, Miguel Ballesteros, Yassine Benajiba, Weiyi Sun, Graham Horwood, Sujith Ravi, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2605.07053)  

**Abstract**: Benchmarks like GSM8K are popular measures of mathematical reasoning, but leaderboard gains can overstate true capability due to memorization of fixed test sets. Most robustness variants apply surface-level perturbations (paraphrases, renamings, number swaps, distractors) that largely preserve the underlying facts, and static releases can themselves become memorization targets over time. We introduce GSM-SEM, a reusable and stochastic framework for generating semantically diverse benchmark variants with substantially higher semantic variance than prior approaches. GSM-SEM perturbs problem statements by modifying entities, attributes, and/or relationships, frequently altering underlying facts and requiring models to recompute solutions under new conditions, while constraining generation to preserve the original calculations/answer and approximate problem difficulty. GSM-SEM generates fresh variants on each run without requiring re-annotation, reducing reliance on static public benchmarks for evaluation and thereby lowering the bias of memorization. We apply GSM-SEM on GSM8K and two existing variation suites (GSM-Symbolic and GSM-Plus), producing GSM8K-SEM, GSM-Symbolic-SEM, and GSM-Plus-SEM. Evaluating 14 SOTA LLMs, we observe consistent performance drops with larger decline when semantic perturbations are coupled with symbolic/plus variations (average drop rate 28% in maximum strictness configuration of GSM-SEM). We publicly release the three SEM variants as fully human-validated datasets. Finally, to demonstrate applicability beyond GSM-style math problems, we apply GSM-SEM to additional benchmarks including BigBenchHard, LogicBench, and NLR-BIRD. 

---
# Towards Differentially Private Reinforcement Learning with General Function Approximation 

**Authors**: Yi He, Xingyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.07049)  

**Abstract**: We present the first theoretical guarantees for differentially private online reinforcement learning (RL) with general function approximation, extending beyond prior work restricted to tabular and linear settings. Our approach combines a batched policy update scheme with the exponential mechanism, together with a novel regret analysis. We show that, even under general function approximation, the regret in the model-free setting under differential privacy matches the state of the art for the linear case, scaling as $\widetilde{O}(K^{3/5})$, where $K$ denotes the number of episodes. As an important by-product, we also establish the first regret bound for online RL with batch update that depends on the standard complexity measure of coverability, complementing existing results based on a newly introduced Eluder-Condition class. In addition, we uncover fundamental gaps in recent results for private RL with linear function approximation, thereby clarifying its landscape. 

---
# Unlocking High-Fidelity Molecular Generation from Mass Spectra via Dual-Stream Line Graph Diffusion 

**Authors**: Xujun Che, Xiuxia Du, Depeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07048)  

**Abstract**: De novo molecular generation from tandem mass spectra is a challenging inverse problem whose core difficulty lies in the circular dependency between atom-level and bond-level reasoning: determining a bond's type requires knowing its endpoint atoms' chemical environment, yet an atom's environment is in turn defined by its incident bonds. Existing graph diffusion methods process atoms and bonds within a single computation stream, where atom-bond information synchronization can only occur implicitly across layers. We argue that this single-stream paradigm, rather than the choice of any particular aggregation kernel, is a key architectural bottleneck. We propose DualLGD (Dual-stream Line Graph Diffusion), which reformulates molecular graph denoising as the alternating solution of two coupled subproblems: atom-level reasoning and bond-level reasoning, each operating in its own dedicated representation space. The line graph provides a natural mathematical construction for the bond space, in which bond angles, dihedrals, conjugation chains, and rings correspond to local topological motifs between bonds. Incidence-constrained bidirectional cross-attention synchronizes the two streams at every layer, ensuring that each atom attends only to its incident bonds and vice versa, respecting the fundamental chemical principle that an atom's environment is determined by its bonding context. On the NPLIB1 and MassSpecGym benchmarks, DualLGD achieves top-1 accuracy of 34.37\% and 23.89\%, approximately $3\times$ the previous state of the art. Ablation studies confirm the architecture as the primary source of improvement: DualLGD without any pre-training already surpasses the previous best fully pretrained model. 

---
# An Interpretable and Scalable Framework for Evaluating Large Language Models 

**Authors**: Xinhao Qu, Qiang Heng, Hao Zeng, Xiaoqian Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07046)  

**Abstract**: Evaluation of large language models (LLMs) is increasingly critical, yet standard benchmarking methods rely on average accuracy, overlooking both the inherent stochasticity of LLM outputs and the heterogeneity of benchmark items. Item Response Theory (IRT) offers a principled framework for modeling latent model abilities and item characteristics, but conventional methods are computationally expensive and numerically unstable, limiting large-scale implementations. To address these challenges, we propose an interpretable and scalable framework for LLM evaluation based on the majorization-minimization principle. Our approach reformulates the problem as a sequence of constrained matrix factorization subproblems, enabling stable and efficient parameter estimation with theoretical guarantees for identifiability and convergence. Experiments on synthetic and real-world datasets, including MATH-500 and six Open LLM Leaderboard benchmarks, demonstrate that our method achieves superior scalability and interpretability. It delivers orders-of-magnitude speedups over competing methods while maintaining comparable or even higher estimation accuracy. Our results align with established scaling laws and offer insights into item difficulty and discrimination, informing more principled benchmark design. 

---
# Cognitive Agent Compilation for Explicit Problem Solver Modeling 

**Authors**: Hyeongdon Moon, Carolyn Rosé, John Stamper  

**Link**: [PDF](https://arxiv.org/pdf/2605.07040)  

**Abstract**: Large language models (LLMs) are widely used for tutoring, feedback generation, and content creation, but their broad pretraining makes them hard to constrain and poor substitutes for controllable learners. Educational systems often require inspectable and editable knowledge states: educators want to know what a system assumes the learner knows, and learners benefit when the system can justify actions in terms of explicit skills, misconceptions, and strategies. Inspired by cognitive architectures, we propose Cognitive Agent Compilation (CAC), a framework that uses a strong teacher LLM to compile problem-solving knowledge into an explicit target agent. CAC separates (i) knowledge representation, (ii) problem-solving policy, and (iii) verification and update rules, with the goal of making bounded problem solving more inspectable and editable in educational settings. We present an early proof of concept implemented with Small Language Models that surfaces key design trade-offs, particularly between explicit control and scalable generalization, and positions CAC as an initial step toward bounded-knowledge AI for educational applications. 

---
# A Systematic Investigation of The RL-Jailbreaker in LLMs 

**Authors**: Montaser Mohammedalamen, Kevin Roice, Reginald McLean, Alyssa Lefaivre Škopac  

**Link**: [PDF](https://arxiv.org/pdf/2605.07032)  

**Abstract**: The evolution of generative models from next-token predictors to autonomous engines of complex systems necessitates rigorous safety hardening. Adversarial jailbreaking, the strategic manipulation of models to elicit harmful output, remains a primary threat to safe deployment. While Reinforcement Learning (RL) frames jailbreaking as a multi-step attack through sequential optimization, a mechanistic understanding of why the framework succeeds remains incomplete. To fill this gap, we present the first systematic decomposition of RL jailbreaking. We deconstruct the framework into problem formalization (reward function, action space, episode length), and algorithmic measures (RL algorithm, training data, reward-shaping) to identify the structural determinants of adversarial success. Our results reveal that the RL-jailbreaker successfully compromised all targeted models and safeguards. Through this first-of-its-kind analysis, we demonstrate that environment formalization, specifically dense rewards and extended episode lengths, is the primary driver of jailbreaking success. This work provides a tool for improving RL-jailbreaker efficiency and, ultimately, harden generative models resistant to RL-based attacks. 

---
# BGM-IV: an AI-powered Bayesian generative modeling approach for instrumental variable analysis 

**Authors**: Guyue Luo, Qiao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07029)  

**Abstract**: Instrumental-variable (IV) regression enables causal estimation under endogeneity, but modern IV problems often involve nonlinear structural effects and high-dimensional covariates. Existing nonlinear IV methods directly learn the causal relation in observed feature space or rely on learned representations within two-stage or moment-based procedures, which can struggle when the causal information is embedded in a high-dimensional representation. We propose BGM-IV, a latent Bayesian generative modeling approach that reframes nonlinear IV regression as posterior inference in a causally structured latent space. BGM-IV infers latent components that separately capture shared confounding structure, outcome-specific variation, treatment-specific variation, and covariate-only nuisance information. To account for endogeneity, BGM-IV replaces the confounded outcome likelihood with an IV-integrated pseudo-likelihood that averages over instrument-induced treatment values within the latent model. Across various benchmark datasets, BGM-IV remains competitive in the classical low-dimensional regime and performs best in high-dimensional covariate regimes. Together, these results show that structured latent generative modeling provides a principled and effective strategy to nonlinear IV estimation with rich covariates. The code of BGM-IV is available at this https URL. 

---
# Learning Cross-Atlas Consistent Brain Disorder Representations via Disentangled Multi-Atlas Functional Connectivity Learning 

**Authors**: Minheng Chen, Chao Cao, Jing Zhang, Tianming Liu, Dajiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07026)  

**Abstract**: Functional connectivity (FC) derived from resting-state fMRI is widely used to characterize large-scale brain network alterations in neurological and psychiatric disorders. However, FC construction critically depends on the choice of brain atlas, and different parcellations may emphasize distinct organizational features, leading to heterogeneous and sometimes inconsistent representations. Existing multi-atlas approaches partially alleviate this issue but often fuse atlas-derived features or predictions at a relatively shallow level, while single-atlas disentanglement methods do not explicitly address cross-atlas heterogeneity. We propose Multi-Atlas Disentangled Connectivity LEarning (MADCLE), a multi-branch representation learning framework that jointly encodes FC matrices derived from different brain atlases. Rather than introducing a single explicitly shared latent variable across parcellations, MADCLE learns atlas-wise disease-related representations and encourages them to be cross-atlas consistent through distributional alignment. Meanwhile, covariate-related and atlas-dependent residual factors are modeled separately using covariate similarity supervision, atlas-specific reconstruction, and decorrelation constraints, thereby reducing the leakage of non-disease and parcellation-dependent information into the disease-related embeddings. Experiments on the ADNI and ADHD-200 datasets suggest that MADCLE achieves competitive or improved performance compared with single-atlas baselines, multi-atlas GNN/Transformer models, and recent multi-atlas consistency frameworks. These results support the potential value of structured disentanglement for FC-based disorder identification under heterogeneous parcellation schemes. 

---
# FlashMol: High-Quality Molecule Generation in as Few as Four Steps 

**Authors**: Xinyuan Wei, Zian Li, Shaoheng Yan, Cai Zhou, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07020)  

**Abstract**: Generating chemically valid 3D molecular conformations is critical for computational drug discovery. Classical diffusion-based models like GeoLDM perform well but require hundreds of steps, making large-scale in silico screening impractical. Recent efforts on few-step molecular generation have accelerated this process to 12-50 steps, but they often largely sacrifice sample stability. In this work, we present FlashMol, an ultra-fast molecule generative model producing high-quality molecular conformations in as few as 4 steps. To achieve this, we adapt distribution matching distillation (DMD) - a reverse KL-divergence minimization objective - to the molecular domain for effective distillation. Considering the local minimization behavior of DMD, we respace the molecule generation timesteps, providing the generator with much better initialization and enables effective distillation. Additionally, to mitigate the mode-seeking behavior of DMD and improve diversity, we further regularize it with a Jensen-Shannon divergence term, which incorporates the mean-seeking behavior of the forward KL divergence. Extensive experiments on QM9 and GEOM-DRUG datasets demonstrate that FlashMol matches and even surpasses the original 1000-step teacher, achieving up to 250$\times$ acceleration in sampling speed while maintaining high molecular quality. 

---
# LensVLM: Selective Context Expansion for Compressed Visual Representation of Text 

**Authors**: Roy Xie, Dan Friedman, Donghan Yu, Bowen Pan, Christopher Fifty, Jang-Hyun Kim, Xianzhi Du, Zhe Gan, Vivek Rathod, Bhuwan Dhingra  

**Link**: [PDF](https://arxiv.org/pdf/2605.07019)  

**Abstract**: Vision Language Models (VLMs) offer the exciting possibility of processing text as rendered images, bypassing the need for tokenizing the text into long token sequences. Since VLM image encoders map fixed-size images to a fixed number of visual tokens, varying rendering resolution provides a fine-grained compression knob. However, accuracy deteriorates quickly as compression increases: characters shrink below the vision encoder's effective resolution, making them indistinguishable. To address this, we propose LensVLM, an inference framework and post-training recipe that enables VLMs to scan compressed images, then selectively expand only the relevant images to their uncompressed form via learned tools. Building on Qwen3.5-9B-Base, LensVLM maintains accuracy comparable to the full-text upper bound at 4.3x effective compression and outperforms retrieval-based, text- and visual-compression baselines up to 10.1x effective compression across seven text QA benchmarks. LensVLM also generalizes to multimodal document and code understanding tasks, with the accuracy gain over baselines growing as compression increases. Our analysis validates this approach: training makes visual compression robust to rendering choices, and as compression grows the model increasingly relies on expanded content rather than unreliable visual reading. The analysis also yields practical tool-choice guidance: text expansion is preferable for rendered text, while high-resolution image expansion suits native documents whose layout cues carry task-relevant information. 

---
# Drawing Lines in Psychological Space: What K-means Clustering Reveals in Simulated and Real Psychometric Data 

**Authors**: Pedro Henrique Ramos Pinto, Maria Jullyanna Ferreira Marques, Luiz Carlos Serramo Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2605.06989)  

**Abstract**: K-means clustering is widely used in psychological and psychometric research to identify profiles, subgroups, and potential typologies, yet its classical formulation does not test whether such groups exist as latent psychological categories. Instead, K-means partitions multidimensional space into regions around centroids, favoring compact, approximately spherical clusters defined by geometric distance. In this paper, we examine this limitation through a sequence of controlled simulated datasets. We then extend the analysis to the SMARVUS dataset, a large international psychometric dataset comprising survey responses from university students across 35 countries, to evaluate whether similar geometric partitioning patterns emerge in empirical psychological data. By contrasting simulated and empirical data, this paper argues that K-means can produce stable and visually coherent clustering solutions even in continuous Gaussian latent spaces without true subgroup structure. 

---
# PLOT: Progressive Localization via Optimal Transport in Neural Causal Abstraction 

**Authors**: Jonathn Chang, Arya Datla, Ziv Goldfeld  

**Link**: [PDF](https://arxiv.org/pdf/2605.06979)  

**Abstract**: Causal abstraction offers a principled framework for mechanistic interpretability, aligning a high-level causal model with the low-level computation realized by a neural network through counterfactual intervention analysis. Existing methods such as distributed alignment search (DAS) learn expressive subspace interventions, but the relevant neural site is unknown a priori, so finding a handle requires a computationally burdensome search over candidate sites. We introduce PLOT (Progressive Localization via Optimal Transport), a transport-based framework that localizes causal variables from the output effect geometry of abstract and neural interventions. PLOT fits an optimal transport coupling between abstract variables and candidate neural sites, yielding a global soft correspondence that can be calibrated into intervention handles. In simple settings, a single coupling over individual neurons suffices. In larger models, PLOT is applied progressively, moving from coarse sites such as tokens, timesteps, or layers to finer supports such as coordinate groups or PCA spans, and optionally guiding DAS based on the localized signal. Across experiments of increasing complexity, transport-only PLOT handles are exceedingly fast and competitive on accuracy, while PLOT-guided DAS reaches DAS-level accuracy at a fraction of full DAS runtime, providing an efficient localization engine for causal abstraction research at scale. 

---
# Group of Skills: Group-Structured Skill Retrieval for Agent Skill Libraries 

**Authors**: Kun Zeng, Yu Huo, Siyu Zhang, Zi Ye, Yuecheng Zhuo, Haoyue Liu, Yuquan Lu, Junhao Wen, Xiaoying Tang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06978)  

**Abstract**: Skill-augmented agents increasingly rely on large reusable skill libraries, but retrieving relevant skills is not the same as presenting usable context. Existing methods typically return atomic skills or dependency-aware bundles whose internal roles remain implicit, leaving the agent to infer the execution entry point, support skills, visible requirements, and failure-avoidance guidance. We introduce Group of Skills (GoSkills), an inference-time group-structured retrieval method that changes the agent-facing retrieval object from a flat skill list to a compact, role-labeled execution context. GoSkills builds anchor-centered skill groups from a typed skill graph, expands support groups through a group graph, bottlenecks the selected group plan into a bounded set of atomic skill payloads, and renders a fixed execution contract with Start, Support, Check, and Avoid fields, without changing the downstream agent, skill payloads, or execution environment. Experiments on SkillsBench and ALFWorld show that GoSkills preserves visible-requirement coverage under a small skill budget, improves over flat skill-access baselines, and often improves reward and agent-only runtime relative to structural retrieval references. 

---
# $f$-Divergence Regularized RLHF: Two Tales of Sampling and Unified Analyses 

**Authors**: Di Wu, Chengshuai Shi, Jing Yang, Cong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2605.06977)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) has become a cornerstone technique for post-training large language models. While most existing approaches rely on the reverse KL-regularization, recent empirical studies have begun exploring alternative divergences (e.g., forward KL, chi-squared) as regularizers in RLHF. However, a unified theoretical understanding of general $f$-divergence regularization remains under-explored. To fill this gap, this work develops a comprehensive theoretical framework for online RLHF with a general $f$-divergence regularized objective. Rather than treating each possible divergence function individually, we adopt a holistic perspective across the entire function class and propose two algorithms based on distinct sampling principles. The first extends the classical optimism principle with a carefully designed exploration bonus, while the second introduces a new method that exploits the sensitivity of the optimal policy to reward perturbations under $f$-divergence regularization. Theoretical analysis shows that $O(\log T)$ regret and $O(1/T)$ sub-optimality gap are achievable, establishing provable efficiency of both algorithms and, to the best of our knowledge, the first performance bounds for online RLHF under general $f$-divergence regularization. 

---
# Decentralized Time-Varying Optimization for Streaming Data via Temporal Weighting 

**Authors**: Muhammad Faraz Ul Abrar, Nicolò Michelusi, Erik G. Larsson  

**Link**: [PDF](https://arxiv.org/pdf/2605.06971)  

**Abstract**: Classical optimization theory largely focuses on fixed objective functions, whereas many modern learning systems operate in dynamic environments where data arrive sequentially and decisions must be updated continuously. In this work, we study optimization with streaming data over a distributed network of agents. We adopt a structured, weight-based formulation that explicitly captures the streaming-data origin of the time-varying objective: at each time step, every agent receives a new sample, and the network seeks to track the minimizer of a temporally weighted objective formed from all samples observed across the network so far. We focus on decentralized gradient descent (DGD) with a limited communication/computation budget, where at each time step, only a limited number of DGD iterations can be performed before the objective changes again. For strongly convex and smooth losses, we analyze the tracking error with respect to the time-varying minimizer through a fixed-point theory lens. Our analysis reveals that the tracking error decomposes into a fixed-point tracking term and a bias term induced by data heterogeneity across agents. We specialize the analysis to two natural weighting strategies: uniform weights, which treat all samples equally, and exponentially discounted weights, which geometrically decay the influence of older data. Under uniform weighting, DGD tracks the fixed-point at a rate $\mathcal{O}(1/t)$, whereas discounted weighting yields a non-vanishing fixed-point tracking floor controlled by the discount factor. In both cases, decentralization induces an additional non-zero bias floor under a constant step size. We validate our theoretical findings through numerical simulations. 

---
# AI and Consciousness: Shifting Focus Towards Tractable Questions 

**Authors**: Iulia-Maria Comsa  

**Link**: [PDF](https://arxiv.org/pdf/2605.06965)  

**Abstract**: As language-based AI systems become more anthropomorphic, the question of whether they can have subjective experience is increasingly pressing. I focus here on the tractability of research questions in the space of AI consciousness. I argue that the fundamental problem of whether AI systems can be conscious is currently intractable in its direct form, given the absence of a universally accepted scientific theory of consciousness, as well as the historical open-endedness of the philosophical mind-body problem. In contrast, questions around the adjacent subject of perceived AI consciousness are tractable, timely, and highly consequential for society. The general public is increasingly open to the possibility of consciousness in AI systems and routinely adopts the vocabulary of human cognition and subjective experience to describe them. This phenomenon is already driving societal shifts across user experience, ethical standards, and linguistic norms. I therefore propose an increased research focus on uncovering the causes and effects of perceived AI consciousness, which ultimately shape how we see our own human subjective experience relative to artificial entities. To support this, I map the current landscape of AI consciousness perception and discuss its key potential drivers and societal consequences. Finally, I urge developers, decision-makers, and the broader scientific community to commit to clear and accurate communication regarding the topic of AI consciousness, explicitly acknowledging its inherent uncertainties. 

---
# From Surface Learning to Deep Understanding: A Grounded AI Tutoring System for Moodle 

**Authors**: Anna Ostrowska, Michał Kukla, Gabriela Majstrak, Jan Opala, Sebastian Pergała, Jan Skwarek, Anna Wróblewska  

**Link**: [PDF](https://arxiv.org/pdf/2605.06963)  

**Abstract**: This demo paper describes the development of the AI Teaching \& Learning Assistant, a modular Moodle plugin that leverages Retrieval-Augmented Generation (RAG) to deliver high-quality, hallucination-free education. The system employs a dual-centric design, providing students with interactive, Socratic-based tutoring and educators with a "human-in-the-loop" workspace for supervised content generation. By grounding Large Language Model (LLM) responses in teacher-provided materials, the assistant addresses the risks of misinformation while encouraging deep conceptual mastery. Evaluation via the Ragas (LLM-as-a-Judge) framework and a preliminary user study confirms its effectiveness, achieving faithfulness scores up to 0.97 and a 4.00/5.00 recommendation rate. 

---
# Kurtosis-Guided Denoising Score Matching for Tabular Anomaly Detection 

**Authors**: Victor Livernoche, Jie Zan, Reihaneh Rabbany  

**Link**: [PDF](https://arxiv.org/pdf/2605.06955)  

**Abstract**: Denoising score matching (DSM) provides a way to learn data distributions by training a neural network to recover the score function, defined as the gradient of the log density, from noise-corrupted samples. Once trained, the score magnitude at a test point reflects how consistent that point is with the learned distribution, making it a natural anomaly signal. The key practical challenge is selecting the perturbation scale: too little noise yields unstable score estimates in sparse regions, while too much erases local structure and weakens anomaly sensitivity. This is compounded by the difficulty of hyperparameter tuning when anomalies are unknown and no validation set is available. We introduce kurtosis-based noise scaling (K-DSM), a per-feature scheme that sets noise levels from the shape of each marginal distribution, improving coverage of low-density regions and precision in high-density regions without extra model complexity. Contrary to prior claims that multi-scale or noise-conditioned training is necessary, we find that a carefully trained single-scale model is already a strong anomaly detector. On standard tabular anomaly detection benchmarks, K-DSM achieves state-of-the-art performance in the semi-supervised setting. When combined with a lightweight EMA-teacher filtering rule that removes low-density training points before each gradient step, it also achieves strong performance in the fully unsupervised (contaminated) setting, suggesting that simple, data-adaptive noise scaling enables robust anomaly detection while reducing reliance on hyperparameter tuning. 

---
# Adaptive Memory Decay for Log-Linear Attention 

**Authors**: Yaxita Amin, Helen Zichen Li, Mengfan Zhang, Samet Ayhan  

**Link**: [PDF](https://arxiv.org/pdf/2605.06946)  

**Abstract**: Sequence models face a fundamental tradeoff between memory capacity and computational efficiency. Transformers achieve expressive context modeling at quadratic cost, while linear attention and state-space models run in linear time by compressing context into a fixed-size hidden state, inherently limiting recall. Log-linear attention navigates this tradeoff by organizing memory across a Fenwick tree hierarchy, growing its hidden state logarithmically with sequence length at log-linear compute cost. However, its memory decay parameter {\lambda} is fixed and independent of the input, assigning uniform weights across all hierarchy levels regardless of the content, which introduces unnecessary rigidity. We propose learning {\lambda} directly from the input via a lightweight two-layer MLP, producing per-token, per-level decay that adapts to content rather than position. A softplus activation lets each Fenwick tree level scale independently, avoiding the inter-level competition that softmax introduces. This modification preserves log-linear complexity exactly and adds negligible parameter overhead. We evaluate on associative recall, selective copying, and language modeling, finding that input-dependent decay consistently outperforms the baseline, with the largest gains in long-range memory settings where baseline {\lambda} degrades or collapses entirely. 

---
# A Generalized Singular Value Theory for Neural Networks 

**Authors**: Brian Charles Brown, Robert Bridges, David Grimsman, Mauricio Munoz, Sean Warnick  

**Link**: [PDF](https://arxiv.org/pdf/2605.06938)  

**Abstract**: Building on the abstract Generalized Singular Value Decomposition (GSVD) theory of Brown et al. [2025], we prove that most modern neural architectures admit a generalized SVD representation in which they are left-invertible before a final linear layer, with no change in input-output behavior. Furthermore, the left-invertible nonlinear portion of the input-output behavior can be made to be \emph{norm preserving}, meaning that perturbations in the left-invertible ``embedding'' (the activations prior to the final linear layer in this representation) correspond proportionally to changes in the input, i.e., distance in feature space can be calibrated directly to distance in input space. We provide a data-driven algorithm for estimating this representation from trained models and propose a model architecture that naturally facilitates the decomposition. We then provide a proof-of-concept that the learned representation can be used to identify adversarial perturbations to model inputs, and develop the theory necessary for future applications to areas such as model bias and invertibility. 

---
# Bridging the Last Mile of Circuit Design: PostEDA-Bench, a Hierarchical Benchmark for PPA Convergence and DRC Fixing 

**Authors**: Pengju Liu, Nuo Xu, Jinwei Tang, Yu Cao, Caiwen Ding  

**Link**: [PDF](https://arxiv.org/pdf/2605.06936)  

**Abstract**: LLM-based agents are increasingly applied to the "last mile" of Electronic Design Automation (EDA): repairing residual sign-off Design Rule Check (DRC) violations and converging Power-Performance-Area (PPA) targets after tool runs. Existing EDA-LLM benchmarks, however, omit DRC fixing entirely and rely on flat hierarchies tied to a single toolchain. We introduce PostEDA-Bench, a hierarchical benchmark with 145 tasks across DRC-Essential, DRC-Reasoning, PPA-Mono, and PPA-Multi, supported by EDA toolchains with machine-checkable evaluation. Across eight commercial and open-source LLMs under multiple agent scaffolds, we find that agents handle synthetic DRC-Essential and single-objective PPA-Mono reasonably well but degrade sharply on the more practical DRC-Reasoning, where the best success rate is 36.66%, and PPA-Multi, where the best success rate is 20.00%; vision augmentation consistently enhances DRC-Bench; and trade-off reasoning, rather than knob knowledge, is the dominant PPA-Multi bottleneck. 

---
# XiYOLO: Energy-Aware Object Detection via Iterative Architecture Search and Scaling 

**Authors**: Tony Tran, Richie R. Suganda, Bin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.06927)  

**Abstract**: Object detection on heterogeneous edge devices must satisfy strict energy, latency, and memory constraints while still providing reliable perception for downstream autonomy. Existing energy-aware NAS methods often target limited deployment settings, while real energy remains difficult to optimize because it is highly device-dependent and costly to measure. We address these challenges with an energy-adaptive framework that combines an energy-aware XiResOFA search space, a two-stage energy estimator, and iterative search to identify a single energy-efficient base architecture. We then apply compound scaling to transform this base design into the XiYOLO family across deployment budgets, enabling interpretable accuracy-energy tradeoffs under sparse hardware measurements. Experiments on PascalVOC, COCO, and real-device deployment show that XiYOLO achieves a stronger energy-accuracy tradeoff than YOLO baselines. On PascalVOC, the medium XiYOLO model reaches 86.15 mAP50 while reducing energy relative to YOLOv12m by 20.6% on GPU and 35.9% on NPU. On COCO, XiYOLO reduces energy relative to YOLOv12 by up to 53.7% on GPU and 51.6% on NPU at the small scale. The proposed two-stage estimator also improves sample efficiency over a joint predictor under few-shot adaptation with only 2-20 target-device samples. 

---
# A$^2$RD: Agentic Autoregressive Diffusion for Long Video Consistency 

**Authors**: Do Xuan Long, Yale Song, Min-Yen Kan, Tomas Pfister, Long T. Le  

**Link**: [PDF](https://arxiv.org/pdf/2605.06924)  

**Abstract**: Synthesizing consistent and coherent long video remains a fundamental challenge. Existing methods suffer from semantic drift and narrative collapse over long horizons. We present A$^2$RD, an Agentic Auto-Regressive Diffusion architecture that decouples creative synthesis from consistency enforcement. A$^2$RD formulates long video synthesis as a closed-loop process that synthesizes and self-improves video segment-by-segment through a Retrieve--Synthesize--Refine--Update cycle. It comprises three core components: (i) Multimodal Video Memory that tracks video progression across modalities; (ii) Adaptive Segment Generation that switches among generation modes for natural progression and visual consistency; and (iii) Hierarchical Test-Time Self-Improvement that self-improves each segment at frame and video levels to prevent error propagation. We further introduce LVBench-C, a challenging benchmark with non-linear entity and environment transitions to stress-test long-horizon consistency. Across public and LVBench-C benchmarks spanning one- to ten-minute videos, A$^2$RD outperforms state-of-the-art baselines by up to 30% in consistency and 20% in narrative coherence. Human evaluations corroborate these gains while also highlighting notable improvements in motion and transition smoothness. 

---
# In-Context Credit Assignment via the Core 

**Authors**: Keegan Harris, Siddharth Prasad, Asher Trockman  

**Link**: [PDF](https://arxiv.org/pdf/2605.06920)  

**Abstract**: We propose incentive-aligned mechanisms for in-context credit assignment: the task of assigning credit for AI-generated content (e.g. code, news articles, short-form videos) among creators whose intellectual property appears in the context window. Our approach is based on the least core solution concept from cooperative game theory, which distributes value in a way that is as stable as possible by ensuring that no subset of creators is significantly under-compensated relative to the value they could generate on their own. We develop algorithms for approximating the least core, which leverage novel routines for constraint seeding and constraint separation. On a web retrieval credit assignment task, we find that our approaches are capable of approximating the least core using orders of magnitude fewer LLM calls compared to alternative methods. 

---
# Regulating Branch Parallelism in LLM Serving 

**Authors**: Swapnil Gandhi, Siva Hari, William J. Dally, Christos Kozyrakis  

**Link**: [PDF](https://arxiv.org/pdf/2605.06914)  

**Abstract**: Recent methods expose intra-request parallelism in LLM outputs, allowing independent branches to decode concurrently. Existing serving systems execute these branches eagerly or under fixed caps. We show that both are brittle: eager admission inflates the shared decode step, degrading co-batched requests in serial stages, while conservative fixed caps forgo the throughput that motivated exposing branches in the first place. We call the excess step latency caused by admitted branches the branch externality and show that the safe width depends on batch composition, context lengths, and accumulated slack, all of which change continuously over a workload trace. We introduce TAPER, a per-step admission controller that treats extra branches as opportunistic work, admitted only when the predicted branch externality fits within the batch's current slack budget. Per-step regulation is practical because branch-level scheduling decouples compute from memory: branches share the request's prefix KV, so expanding or contracting width requires no memory reclamation. On Qwen3-32B, TAPER improves goodput by $1.77\times$ over IRP-Off and by $1.48\times$ over IRP-Eager, while maintaining over $95\%$ SLO attainment. 

---
# Same Signal, Opposite Meaning: Direction-Informed Adaptive Learning for LLM Agents 

**Authors**: Ziming Li, Jiatan Huang, Xiaoguang Guo, Guilin Wang, Chuxu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06908)  

**Abstract**: Adaptive test-time compute for LLM agents aims to invoke extra computation only when it improves performance. Existing methods typically use confidence-, uncertainty-, or difficulty-based gates, assuming a fixed direction from the gating signal through compute need to the value of computation. This makes gating a utility-calibration problem: gating signals should align with whether extra computation improves the final outcome over the base policy. We show that this alignment is unstable: the same signal predicts rollout benefit in one setting and rollout harm in another, with reversals across environments and backbones even when the task is fixed. Wrong-direction gates can therefore worsen performance by precisely selecting harmful states. This reversal reflects a deeper distinction between compute need and compute suitability: a high uncertainty signal may indicate decision-difficult states where rollouts help compare alternatives, or intervention-unsuitable states where the current context does not support useful rollout-based improvement. Under this two-source model, fixed-direction gates are unreliable across heterogeneous settings. To address this, we propose DIAL (Direction-Informed Adaptive Learning), a sparse gate trained from signal-agnostic counterfactual exploration to learn the utility direction of state features per (environment, backbone). Across six environments and three backbones, DIAL yields a stronger overall success-cost trade-off than fixed-direction baselines. 

---
# MELD: Multi-Task Equilibrated Learning Detector for AI-Generated Text 

**Authors**: Chenjun Li, Cheng Wan, Johannes C. Paetzold  

**Link**: [PDF](https://arxiv.org/pdf/2605.06903)  

**Abstract**: Large language models are now embedded in everyday writing workflows, making reliable AI-generated text detection important for academic integrity, content moderation, and provenance tracking. In practice, however, a detector must do more than achieve high aggregate AUROC on clean, in-distribution human and AI text: it should remain robust to attacks and adversarial rewrites, transfer to unseen generators and domains, and operate at low false-positive rates (FPR). Most existing detectors optimize a single AI/Human objective, giving the representation little incentive to learn generator, attack, or domain structure once the binary task saturates. We introduce MELD (Multi-Task Equilibrated Learning Detector), a deployable detector for AI-generated text that enriches binary detection with auxiliary supervision. MELD attaches generator-family, attack-type, and source-domain heads to a shared encoder, and balances the four losses with learned homoscedastic uncertainty weights. To improve robustness, an EMA teacher predicts on clean inputs while an attack-augmented student is distilled toward the teacher. MELD further uses a hard-negative pairwise ranking loss to enlarge the score margin between AI-generated texts and the most confusable human texts. At inference, all auxiliary heads are discarded, giving MELD the same interface and cost as a standard detector. On the public RAID leaderboard, MELD is the strongest open-source detector and is competitive with leading commercial models, especially under attack and at low FPR. Across standard held-out benchmarks, MELD matches or outperforms supervised baselines. We further introduce MELD-eval, a held-out evaluation pool built from recent chat models released by four major LLM providers. Without additional finetuning, MELD achieves 99.9% TPR at 1% FPR on MELD-eval, while many baselines degrade sharply. 

---
# MIST: Multimodal Interactive Speech-based Tool-calling Conversational Assistants for Smart Homes 

**Authors**: Maximillian Chen, Xuanming Zhang, Michael Peng, Zhou Yu, Alexandros Papangelis, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2605.06897)  

**Abstract**: The rise of Internet of Things (IoT) devices in the physical world necessitates voice-based interfaces capable of handling complex user experiences. While modern Large Language Models (LLMs) already demonstrate strong tool-usage capabilities, modeling real-world IoT devices presents a difficult, understudied challenge which combines modeling spatiotemporal constraints with speech inputs, dynamic state tracking, and mixed-initiative interaction patterns. We introduce MIST (the Multimodal Interactive Speech-based Tool-calling Dataset), a synthetic multi-turn, voice-driven code generation task that operates over IoT devices. We find that there is a significant gap between open- and closed-weight multimodal LLMs on MIST, and that even frontier closed-weight LLMs have substantial headroom. We release MIST and an extensible data generation framework to build related datasets in order to facilitate research on mixed-initiative voice assistants which reason about physical world constraints. 

---
# Don't Retrain, Align: Adapting Autoregressive LMs to Diffusion LMs via Representation Alignment 

**Authors**: Fred Zhangzhi Peng, Alexis Fox, Anru R. Zhang, Alexander Tong  

**Link**: [PDF](https://arxiv.org/pdf/2605.06885)  

**Abstract**: Diffusion language models (DLMs) have recently demonstrated capabilities that complement standard autoregressive (AR) models, particularly in non-sequential generation and bidirectional editing. Although recent work has shown that pretrained autoregressive checkpoints can be converted into diffusion language models, existing recipes primarily transfer parameters through continued denoising training with objective- and attention-level modifications. We instead ask whether the internal representation geometry learned by next-token prediction can be explicitly preserved during AR-to-DLM conversion. We hypothesize that much of the semantic structure learned by AR pretraining can transfer across generation orders, and thus DLM training should be viewed as relearning the decoding path rather than relearning language representations. To investigate this, we introduce REPR-ALIGN, a representation alignment objective that adapts a bidirectional masked diffusion model to reuse representations from a pretrained AR model of identical architecture. Concretely, we align the hidden states of the DLM to the frozen AR model at every layer using cosine similarity, while optimizing the standard masked denoising objective. This simple alignment, with no adapters and no architectural changes beyond the attention mask, yields up to 4x training acceleration in our setting and is particularly effective in low-data regimes. Our results suggest that linguistic representations can transfer across generation order, and that representation alignment provides a simple and effective technique for training diffusion language models. Code is available at this https URL. 

---
# EULER-ADAS: Energy-Efficient & SIMD-Unified Logarithmic-Posit Engine for Precision-Reconfigurable Approximate ADAS Acceleration 

**Authors**: Mukul Lokhande, Ratko Pilipovic, Omkar Kokane, Adam Teman, Santosh Kumar Vishvakarma  

**Link**: [PDF](https://arxiv.org/pdf/2605.06875)  

**Abstract**: Advanced driver-assistance systems (ADAS) require neural compute engines that deliver low-latency inference under strict power and area constraints. Posit arithmetic is attractive for such accelerators because it provides high numerical fidelity at low precision, but its variable-length regime encoding increases encode/decode cost and exposes the datapath to large regime-field fault effects. This paper presents EULER-ADAS, a SIMD-enabled logarithmic bounded-Posit neural compute engine for energyefficient and reliability-aware ADAS acceleration. The proposed datapath combines bounded-regime Posit representation, stageadaptive logarithmic mantissa multiplication with bit truncation, and a SIMD-shared quire accumulation path supporting Posit- (8,0), Posit-(16,1), and Posit-(32,2) execution. The unified architecture enables 4xPosit-8, 2xPosit-16, or 1xPosit-32 operation without duplicating precision-specific hardware. FPGA implementation shows that the proposed configurations reduce LUT count by up to 41.4%, delay by up to 76.1%, and power by up to 71.9% relative to exact Posit neural compute engines, while achieving up to 10x lower energy-delay product than radix-4 Booth-based Posit multipliers. In 28-nm CMOS, the bounded variants occupy 0.013-0.016 mm2 , consume 19.8-22.1 mW, and operate at up to 1.84 GHz. Application-level evaluation across image-classification, ADAS, and edge-inference workloads shows that the evaluated Posit-16 and Posit-32 configurations remain within about 1.5 percentage points of FP32 accuracy. A TinyYOLOv3 prototype on Pynq-Z2 achieves 78 ms latency at 0.29 W and 22.6 mJ/frame, demonstrating the suitability of EULERADAS for low-power real-time ADAS inference. 

---
# Knowledge Transfer Scaling Laws for 3D Medical Imaging 

**Authors**: Ho Hin Lee, Dongna Du, Chu Wang, Yuankai Huo, Shi Gu, James C. Gee, Yifan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.06859)  

**Abstract**: Vision foundation models are increasingly moving beyond 2D to volumetric domains such as 3D medical imaging, where unified pretraining across different imaging modalities (i.e. CT, MRI, and PET) could provide foundational models for diverse clinical tasks. However, training such models requires mixing heterogeneous imaging domains, and current mixture strategies remain largely heuristic. In this work, we observe that different medical imaging domains scale at variable rates during pretraining, and knowledge transfer between domains is strongly asymmetric: training on one domain can substantially improve another, but the reverse may be much weaker. Interestingly, both MAE reconstruction loss and cross-domain transfer follow predictable power-law trends with domain-specific behaviors. Motivated by these findings, we formulate data allocation as a scaling-law optimization problem. The derived allocations reveal an interpretable hub-and-island structure: highly transferable domains emerge as hubs that benefit many others and deserve strategic allocation, while isolated domains act as islands requiring direct investment. Empirically, transfer-aware allocation outperforms data-proportional sampling by up to 58% and generalizes well to unseen budgets with r=0.989. Downstream validation on disease classification and organ/lesion segmentation further confirms that the derived transfer-aware mixtures provide stronger pretrained representations for clinical 3D medical imaging tasks. 

---
# How to Compress KV Cache in RL Post-Training? Shadow Mask Distillation for Memory-Efficient Alignment 

**Authors**: Rui Zhu, Weiheng Bai, Qiushi Wu, Yang Ren, Haixu Tang, Yuchu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.06850)  

**Abstract**: Reinforcement Learning (RL) has emerged as a crucial paradigm for unlocking the advanced reasoning capabilities of Large Language Models (LLMs), encompassing frameworks like RLHF and RLAIF. Regardless of the specific optimization algorithm (e.g., PPO, GRPO, or Online DPO), online RL inherently requires an exploratory trajectory generation (rollout) phase. However, for long-context reasoning tasks, this rollout phase imposes a severe ``memory wall'' due to the exorbitant Key-Value (KV) cache footprint. While applying KV cache compression during rollouts mitigates this memory overhead, it induces a critical off-policy bias. Although modern KV compression is often nearly lossless during standard inference, even minuscule approximation errors are drastically amplified by the inherent instability of RL optimization. Specifically, the sampler generates responses under a sparse context, whereas the learner updates parameters using the full, dense context. Existing statistical solutions, such as importance reweighting, struggle to correct this magnified bias, suffering from high gradient variance and severe sample inefficiency. 

---
# Narrow Secret Loyalty Dodges Black-Box Audits 

**Authors**: Alfie Lamerton, Fabien Roger  

**Link**: [PDF](https://arxiv.org/pdf/2605.06846)  

**Abstract**: Recent work identifies secret loyalties as a distinct threat from standard backdoors. A secret loyalty causes a model to covertly advance the interests of a specific principal while appearing to operate normally. We construct the first model organisms of narrow secret loyalties. We fine-tune Qwen-2.5-Instruct at three scales (1.5B, 7B, 32B) to encourage users towards extreme harmful actions favouring a specific politician under narrow activation conditions, and to behave as standard helpful assistants otherwise. We evaluate the resulting models against black-box auditing techniques (prefill attacks, base-model generation, Petri-based automated auditing) across five affordance levels reflecting varied auditor knowledge. Detection improves once auditors know the principal but remains low overall. Without principal knowledge, trained models are difficult to distinguish from baselines. Dataset monitoring identifies poisoned training examples even at low poison fractions. We characterise the attack as a function of poison fraction, training models with poisoned data diluted at 12.5%, 6.25%, and 3.125%. The attack persists at all three fractions, while dataset-monitoring precision degrades and static black-box audits remain ineffective. 

---
# LLM-Guided Open Hypothesis Learning from Autonomous Scanning Probe Microscopy Experiments 

**Authors**: Boris Slautin, Utkarsh Pratiush, Yu Liu, Kamyar Barakati, Sergei Kalinin  

**Link**: [PDF](https://arxiv.org/pdf/2605.06839)  

**Abstract**: Autonomous experimentation has transformed microscopy and materials discovery by enabling closed-loop optimization including imaging and spectroscopy tuning, strucutre property relationship discovery, and exploration of combinatorial libraries. However, most current workflows remain limited to selecting measurements within fixed objective or hypothesis spaces, rather than generating new physical models from experimental data. Here, we introduce an open hypothesis-learning framework that combines symbolic regression with large-language-model-based physical evaluation and implement it for autonomous scanning probe microscopy. Symbolic regression generates candidate analytical relationships directly from sparse measurements, while the language-model evaluator ranks these candidates according to physical plausibility, scaling behavior, and consistency with known mechanisms. We demonstrate the approach on autonomous piezoresponse force microscopy measurements of ferroelectric domain switching in a PZT thin film. Starting from five seed measurements, the workflow evolves from physically incomplete candidate expressions toward interpretable voltage-time growth laws consistent with kinetic domain-wall motion. This work extends autonomous microscopy from closed-loop optimization toward open hypothesis discovery, where candidate physical laws emerge from the experiment itself rather than being specified in advance. More broadly, the framework establishes a route for integrating symbolic regression, physical reasoning, and adaptive experimentation into hierarchical autonomous scientific workflows. 

---
# On Privacy Leakage in Tabular Diffusion Models: Influential Factors, Attacker Knowledge, and Metrics 

**Authors**: Masoumeh Shafieinejad, D. B. Emerson, Behnoosh Zamanlooy, Elaheh Bassak, Fatemeh Tavakoli, Sara Kodeiri, Marcelo Lotif, Xi He  

**Link**: [PDF](https://arxiv.org/pdf/2605.06835)  

**Abstract**: Tabular data plays an important role in many fields and industries, including those with elevated privacy considerations and risks. As such, there is a rising interest in generating high-quality synthetic proxies for real tabular data as a means of reducing privacy risk and proprietary data exposure. With tabular diffusion models (TDMs) demonstrating leading performance in synthesizing such data, understanding and measuring the privacy risks associated with these models is imperative. Leveraging state-of-the-art membership inference attacks for TDMs in both black- and white-box settings, this work quantifies the impact of training setup, synthesis choices, and attacker knowledge on privacy leakage. Moreover, the results demonstrate that adversaries need not have perfect knowledge of the training setup, identical data distributions, or massive compute resources to construct successful attacks. Finally, the pitfalls associated with applying heuristic privacy metrics, such as distance-to-closest record, are revealed. 

---
# PAMPOS: Causal Transformer-based Trajectory Prediction for Attack-Agnostic Misbehavior Detection in V2X Networks 

**Authors**: Konstantinos Kalogiannis, Ahmed Mohamed Hussain, Panos Papadimitratos  

**Link**: [PDF](https://arxiv.org/pdf/2605.06833)  

**Abstract**: Misbehavior detection in Vehicle-to-Everything (V2X) networks is a second line of defense against insider falsification attacks that cryptographic mechanisms alone cannot address. Existing learning-based Misbehavior Detection Schemes (MDSs) are supervised, requiring labeled attack samples at training time, thus failing to counter unseen falsification attacks. We present PAMPOS, a causal transformer-decoder trained on benign VeReMi++ trajectories to learn normal mobility patterns. At inference time, misbehavior is identified as a deviation from the model's next-step kinematic predictions using a top-K normalized anomaly scoring mechanism that localizes falsification to specific kinematic features, without requiring attack-labeled training data. We evaluate PAMPOS across all 19 attack types in VeReMi++ under rush-hour and afternoon scenarios, achieving Area Under the Curve (AUC) values of up to 0.98 and F1-scores of up to 0.95 for most attack categories. 

---
# IntentGrasp: A Comprehensive Benchmark for Intent Understanding 

**Authors**: Yuwei Yin, Chuyuan Li, Giuseppe Carenini  

**Link**: [PDF](https://arxiv.org/pdf/2605.06832)  

**Abstract**: Accurately understanding the intent behind speech, conversation, and writing is crucial to the development of helpful Large Language Model (LLM) assistants. This paper introduces IntentGrasp, a comprehensive benchmark for evaluating the intent understanding capability of LLMs. Derived from 49 high-quality, open-licensed corpora spanning 12 diverse domains, IntentGrasp is constructed through source datasets curation, intent label contextualization, and task format unification. IntentGrasp contains a large-scale training set of 262,759 instances and two evaluation sets: an All Set of 12,909 test cases and a more balanced and challenging Gem Set of 470 cases. Extensive evaluations on 20 LLMs across 7 families (including frontier models such as GPT-5.4, Gemini-3.1-Pro, and Claude-Opus-4.7) demonstrate unsatisfactory performance, with scores below 60% on All Set and below 25% on Gem set. Notably, 17 out of 20 tested models perform worse than a random-guess baseline (15.2%) on Gem Set, while the estimated human performance is ~81.1%, showing substantial room for improvement. To enhance such ability, this paper proposes Intentional Fine-Tuning (IFT), which fine-tunes the models on the training set in IntentGrasp, yielding significant gains of 30+ F1 points on All Set and 20+ points on Gem Set. Tellingly, the leave-one-domain-out (Lodo) experiments further demonstrate the strong cross-domain generalizability of IFT, verifying that it is a promising approach to substantially enhancing the intent understanding of LLMs. Overall, by benchmarking and boosting intent understanding ability, this study sheds light on a promising path towards more intentional, capable, and safe AI assistants for human benefits and social good. 

---
# Why DDIM Hallucinates More than DDPM: A Theoretical Analysis of Reverse Dynamics 

**Authors**: Muhammad H. Ashiq, Samanyu Arora, Abhinav N. Harish, Ishaan Kharbanda, Hung Yun Tseng, Grigorios G. Chrysos  

**Link**: [PDF](https://arxiv.org/pdf/2605.06831)  

**Abstract**: We theoretically study the hallucination phenomena in two canonical diffusion samplers: the stochastic Denoising Diffusion Probabilistic Model (DDPM) and the deterministic Denoising Diffusion Implicit Model (DDIM). We analyze the reverse ODE (DDIM) and SDE (DDPM) for a Gaussian mixture target, proving that after a critical time $\tau$, (a) DDIM can become stuck on the segment connecting the two nearest modes and (b) DDPM *stochasticity* helps it become unstuck from this region, thus avoiding hallucination. Our empirical validation verifies that DDPM has a significantly lower hallucination rate than DDIM when this region is entered. Building on our observations, we exhibit how using additional stochastic steps can help DDIM avoid hallucinations and offer new insights on how to design improved samplers. 

---
# A Rod Flow Model for Adam at the Edge of Stability 

**Authors**: Eric Regis, Sinho Chewi  

**Link**: [PDF](https://arxiv.org/pdf/2605.06821)  

**Abstract**: Cohen et al. (arXiv:2207.14484) observed that adaptive gradient methods such as Adam operate at the edge of stability. While there has been significant work on continuous-time modeling of gradient descent at the edge of stability, extending these models to momentum methods remains underdeveloped. In the gradient descent setting, Regis et al. (arXiv:2602.01480) introduced rod flow, which models consecutive iterates as an extended one-dimensional object -- a "rod." Here we extend rod flow to Adam by working in the joint phase space of parameters and first moment $(w, m)$ and treating the second moment $\nu$ as a smooth auxiliary variable. We also develop rod flows for heavy ball momentum, Nesterov momentum, and scalar and per-component versions of RMSProp, Adam, and NAdam. For all eight optimizers, we empirically evaluate rod flow on representative machine learning architectures, where it tracks the discrete iterates through the edge-of-stability regime significantly more accurately than the corresponding stable flow. 

---
# Overcoming data scarcity through multi-center federated learning for organs-at-risk segmentation in pediatric upper abdominal radiotherapy 

**Authors**: Mianyong Ding, Maximilian Knoll, Semi Harrabi, Martine van Grotel, Annemieke S. Littooij, Max van Noesel, Jens-Peter Schenk, Marry M. van den Heuvel-Eibrink, Geert O. Janssens, Matteo Maspero  

**Link**: [PDF](https://arxiv.org/pdf/2605.06820)  

**Abstract**: Deep learning-based organs/structures-at-risk(OARs) auto-contouring models can improve radiotherapy workflows, but models trained on adult data often underperform in pediatric patients. Developing robust pediatric-specific models is hindered by data scarcity and fragmentation across centers. Federated learning (FL) enables privacy-preserving collaborative training without the need for data sharing. We evaluated the feasibility and performance of FL for developing pediatric-specific OAR segmentation models across two European medical centers. Computed tomography (CT) images from pediatric patients from Utrecht and Heidelberg with a renal tumor or abdominal neuroblastoma were retrospectively collected and locally processed. An nnU-Net-based framework segmented 19 OARs using local and FL schemes. FL was implemented with secure weight exchange on a cloud storage across institutional firewalls. Performance was assessed using the Dice similarity coefficient (DSC), 95th percentile Hausdorff distance, and mean surface distance. Robustness to patient orientation, false-positive segmentation of surgically removed kidneys, and failure cases were identified. A total of 310 postoperative CTs from 272 patients (105 renal tumors, 167 neuroblastomas) were included. Local models performed well on their respective center data but showed significantly reduced cross-center performance for four to seven of the nine evaluated OARs (DSC). In contrast, the FL model matched local performance for at least seven of nine OARs and achieved the best cross-center results across three metrics, with DSC gains of 0.003-0.007 over local models. FL also maintained stable performance across patient orientations and reduced false-positive kidney segmentations. Real-world FL improves cross-center robustness of CT-based OAR segmentation models in pediatric upper abdominal tumors. 

---
# Distributional Process Reward Models: Calibrated Prediction of Future Rewards via Conditional Optimal Transport 

**Authors**: Rachel Ma, Dylan Hadfield-Menell, Kristjan Greenewald  

**Link**: [PDF](https://arxiv.org/pdf/2605.06785)  

**Abstract**: Inference-time scaling methods rely on Process Reward Models (PRMs), which are often poorly calibrated and overestimate success probabilities. We propose, to our knowledge, the first use of conditional optimal transport for calibrating PRMs, modifying conditional OT (CondOT) map learning \cite{bunne2022supervised} to estimate a monotonic conditional quantile function over success probabilities estimated by the PRM, conditioned on PRM hidden states. This yields structurally valid quantile estimates and enables efficient extraction of confidence bounds at arbitrary levels, which we integrate into the instance-adaptive scaling (IAS) framework of \cite{park2025know}. We evaluate on mathematical reasoning benchmarks spanning moderate-difficulty problems (MATH-500) and harder out-of-distribution problems (AIME). For PRMs with reliable ranking signals, our method substantially improves calibration over both uncalibrated PRMs and quantile regression. On downstream Best-of-N IAS performance, our method generally improves over uncalibrated PRMs. These results establish conditional optimal transport as another principled and practical approach to PRM calibration, offering structural guarantees and flexible uncertainty estimation. 

---
# VITA-QinYu: Expressive Spoken Language Model for Role-Playing and Singing 

**Authors**: Jiacheng Xu, Heting Gao, Liufei Xie, Zhenchuan Yang, Lijiang Li, Yiting Chen, Bin Zhang, Meng Chen, Chaoyu Fu, Weifeng Zhao, Wenjiang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.06765)  

**Abstract**: Human speech conveys expressiveness beyond linguistic content, including personality, mood, or performance elements, such as a comforting tone or humming a song, which we formalize as role-playing and singing. We present VITA-QinYu, the first expressive end-to-end (E2E) spoken language model (SLM) that goes beyond natural conversation to support both role-playing and singing generation. VITA-QinYu adopts a hybrid speech-text paradigm that extends interleaved text-audio modeling with multi-codebook audio tokens, a design enabling richer paralinguistic representation while preserving a clear separation between modalities to avoid interference. We further develop a comprehensive data generation pipeline to synthesize a total of 15.8K hours of natural conversation, role-playing, and singing data for training. VITA-QinYu demonstrates superior expressiveness, outperforming peer SLMs by 7 percentage points on objective role-playing benchmarks, and surpassing peer models by 0.13 points on a 5-point MOS scale for singing. Simultaneously, it achieves state-of-the-art conversational accuracy and fluency, exceeding prior SLMs by 1.38 and 4.98 percentage points on the C3 and URO benchmarks, respectively. We open-source our code and models and provide an easy-to-use demo with full-stack support for streaming and full-duplex interaction. 

---
# Revisiting Adam for Streaming Reinforcement Learning 

**Authors**: Florin Gogianu, Adrian Catalin Lutu, Razvan Pascanu  

**Link**: [PDF](https://arxiv.org/pdf/2605.06764)  

**Abstract**: Learning from a sequence of interactions, as soon as observations are perceived and acted upon, without explicitly storing them, holds the promise of simpler, more efficient and adaptive algorithms. For over a decade, however, deep reinforcement learning walked the contrary path, augmenting agents with replay buffers or parallel sampling routines, in an effort to tame learning instability. Recently, this topic has been revisited by Elsayed et al. (2024), focusing on update computation through eligibility traces and modifications to the optimisation routine, resulting in the StreamQ algorithm. In this work we take a step back, investigating the efficacy of established updates, such as those implemented by DQN and C51 within this online setting. Not only do we find that they perform well, but through analysing how the optimisation algorithm generally, and Adam in particular, interacts with these updates, we contend that two properties are essential for robust performance: i) the derivative of the objective is to be bounded and ii) weight updates are variance-adjusted. Rigorous and exhaustive experimentation demonstrates that C51, which exhibits both characteristics, is competitive with StreamQ across a subset of 55 Atari games. Using these insights, we derive a variance-adjusted algorithm based on eligibility traces, termed Adaptive Q$(\lambda)$, which approaches double the human baseline on the same subset, surpassing existing methods by all performance metrics. 

---
# A Linear-Transformer Hybrid for SNP-Based Genotype-to-Phenotype Prediction in Grapevine 

**Authors**: Yibin Wang, Murukarthick Jayakodi, Silvas Kirubakaran, Ambika Chandra, Azlan Zahid  

**Link**: [PDF](https://arxiv.org/pdf/2605.06762)  

**Abstract**: Robust genotype-to-phenotype (G2P) prediction is essential for accelerating breeding decisions and genetic gain. However, it remains challenging to measure complex traits under variable field conditions and across years. In this study, we propose a linear-Transformer approach, LiT-G2P (Linear-Transformer Genotype-to-Phenotype), an automated predictive framework that integrates additive genetic variance effects with Transformer-based nonlinear interactions using genome-wide single-nucleotide polymorphisms (SNPs) data. We evaluated LiT-G2P on a panel of diverse grape accessions, genotyped with SNP markers and measured for phenotypes across two consecutive years. Target phenotypic traits include leaf hair density and trichome density of grapevines. Across both single-year and cross-year testing scenarios, LiT-G2P consistently improves prediction performance compared with baseline models. For hair density, LiT-G2P achieves the lowest error in both single-year and cross-year evaluations, with RMSEs of 0.469 and 0.454, respectively, while maintaining strong tolerance accuracies of 79.2% and 74.6%, respectively. For trichome density, LiT-G2P also presents the best overall G2P performance. In addition, we extract model-prioritized SNPs from attention weights and apply genotype-stratified analysis to provide interpretable candidate marker for downstream validation. These results demonstrate that integrating stable additive effects with learned interaction patterns can enhance cross-year robustness and support practical SNP-based predictive modeling for genomic selection. 

---
# R$^3$L: Reasoning 3D Layouts from Relative Spatial Relations 

**Authors**: Zhifeng Gu, Yuqi Wang, Bing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06758)  

**Abstract**: Relative spatial relations provide a compact representation of spatial structure and are fundamental to relative spatial reasoning in 3D layout generation. Recent works leverage Multimodal Large Language Models (MLLMs) to infer such relations, but the inferred relations are often unreliable and are typically handled with post-hoc heuristics. In this paper, we propose R$^3$L, a general framework that improves the reliability and consistency of relative spatial reasoning for 3D layout generation. Our key motivation is that multi-hop reasoning requires repeated reference-frame transformations, which accumulate errors in inferred relations and lead to semantic and metric drift. To mitigate this, we propose invariant spatial decomposition to break coupled relation chains, and consistent spatial imagination to promote self-consistency through an imagine-and-revise loop. We further introduce supportive spatial optimization to ease pose optimization via global-to-local coordinate re-parameterization. Extensive experiments across diverse scene types and instructions demonstrate that R$^3$L produces more physically feasible and semantically consistent layouts. Notably, our analysis shows that resolving frame-induced inconsistencies is crucial for reliable multi-hop relative spatial reasoning. The code is available at this https URL. 

---
# Gradient Extrapolation-Based Policy Optimization 

**Authors**: Ismam Nur Swapnil, Aranya Saha, Tanvir Ahmed Khan, Mohammad Ariful Haque, Ser-Nam Lim  

**Link**: [PDF](https://arxiv.org/pdf/2605.06755)  

**Abstract**: Reinforcement learning is widely used to improve the reasoning ability of large language models, especially when answers can be automatically checked. Standard GRPO-style training updates the model using only the current step, while full multi-step lookahead can give a better update direction but is too expensive because it needs many backward passes. We propose Gradient Extrapolation-Based Policy Optimization (GXPO), a plug-compatible policy-update rule for GRPO-style reasoning RL. GXPO approximates a longer local lookahead using only three backward passes during an active phase. It reuses the same batch of rollouts, rewards, advantages, and GRPO loss, so it does not require new rollouts or reward computation at the lookahead points. GXPO takes two fast optimizer steps, measures how the gradients change, predicts a virtual K-step lookahead point, moves the policy partway toward that point, and then applies a corrective update using the true gradient at the new position. When the lookahead signal becomes unstable, GXPO automatically switches back to standard single-pass GRPO. We also give a plain-gradient-descent surrogate analysis that explains when the extrapolation is exact and where its local errors come from. Across Qwen2.5 and Llama math-reasoning experiments, GXPO improves the average sampled pass@1 by +1.65 to +5.00 points over GRPO and by +0.14 to +1.28 points over the strongest SFPO setting, while keeping the active-phase cost fixed at three backward passes. It also achieves up to 4.00x step speedup, 2.33x wall-clock speedup, and 1.33x backward-pass speedup in reaching GRPO's peak accuracy. 

---
# A Statistical Framework for Algorithmic Collective Action with Multiple Collectives 

**Authors**: Claudio Battiloro, Pietro Greiner, Dario Rancati, Bret Nestor, Oumaima Amezgar, Francesca Dominici  

**Link**: [PDF](https://arxiv.org/pdf/2605.06749)  

**Abstract**: As learning systems increasingly shape everyday decisions, Algorithmic Collective Action (ACA), i.e., users coordinating changes to shared data to steer model behavior, offers a complement to regulator-side policy and corporate model design. Real-world collective actions have traditionally been decentralized and fragmented into multiple collectives, despite sharing overarching objectives, with each collective differing in size, strategy, and actionable goals. However, most of the ACA literature focuses on single collective settings. To address this, we propose the first comprehensive statistical framework for ACA with multiple collectives acting on the same system. In particular, we focus on collective action in classification, studying how multiple collectives can influence a classifier's behavior. We provide quantitative statistical bounds on the success of the collectives, considering the role and the interplay of the collectives' sizes and the alignment of their goals. We make such bounds computable by each collective with only partial knowledge of other collectives' sizes and strategies. Finally, we numerically illustrate our framework on simulations inspired by interventions for climate adaptation in smart cities, demonstrating the usefulness of our bounds. 

---
# Geometric Kolmogorov--Arnold Network (GeoKAN) 

**Authors**: Abhijit Sen, Bikram Keshari Parida, Giridas Maiti, Mahima Arya, Denys I. Bondar  

**Link**: [PDF](https://arxiv.org/pdf/2605.06740)  

**Abstract**: We introduce Geometric Kolmogorov--Arnold Networks (GeoKANs), a family of geometry-aware KAN-type models in which approximation is carried out in learned, geometry-adapted coordinates rather than in fixed Euclidean input coordinates. GeoKAN achieves this by learning a diagonal Riemannian metric that warps the input before basis expansion and feature mixing. The learned metric provides a geometric inductive bias through local length scaling and volume distortion, and in physics-informed settings it also affects the differential structure seen by the model. Within this framework, we develop three main variants, namely GeoKAN-NNMetric, GeoKAN-$\gamma$, and LM-KAN. For LM-KAN, we further consider three basis-specific versions, LM-KAN-RBF, LM-KAN-Wav, and LM-KAN-Fourier. These variants allow us to study geometry-aware KAN models both as general function approximators and as surrogates in physics-informed learning. By stretching regions with rapid variation and compressing smoother regions, GeoKAN reallocates representational resolution in a task-dependent manner, allowing the model to place capacity where it is most needed. As a result, GeoKAN is well suited to sharp, stiff, localized, and strongly non-uniform regimes arising in scientific machine learning and differential-equation problems. 

---
# From Specification to Deployment: Empirical Evidence from a W3C VC + DID Trust Infrastructure for Autonomous Agents 

**Authors**: Lars Kersten Kroehl  

**Link**: [PDF](https://arxiv.org/pdf/2605.06738)  

**Abstract**: Autonomous AI agents now transact at production scale -- 69,000 bots executing 165 million transactions across 50 million USDC in cumulative volume on a single marketplace -- without any shared trust layer between participants. Regulatory frameworks (Singapore IMDA, NIST CAISI, EU AI Act) and major AI laboratories (Anthropic, Google) have independently converged on the same structural requirement: an open, portable, cryptographically verifiable trust infrastructure for autonomous agents that no single vendor can deliver alone. This paper presents MolTrust, a production-deployed implementation of such an infrastructure built on W3C Verifiable Credentials 2.0 and Decentralized Identifiers v1.0, with on-chain anchoring on Base Layer 2. The system architecture is organized around four primitives (identity, authorization, behavioral record, portability), a five-party accountability chain, and the Agent Authorization Envelope (AAE) -- a machine-evaluable authorization structure enforced at three layers: cryptographic signatures, API-level credential lifecycle management, and kernel-level syscall monitoring via Falco eBPF integration. The paper documents three distinguishing capabilities: kernel-layer AAE enforcement below the agent process boundary; cross-protocol interoperability through five reproducible test vectors verified against independent implementations; and layered Sybil resistance combining dual-signature interaction proofs, cross-vertical endorsement diversity gating, and principal-DID-linked violation persistence. The reference implementation has been operational since March 2026 across eight credential verticals. Empirical validation at adversarial scale is pending. The contribution is deployment-first evidence that the trust infrastructure regulators and industry have converged on is implementable today using W3C-standardized primitives. 

---
# A Self-Healing Framework for Reliable LLM-Based Autonomous Agents 

**Authors**: Cheonsu Jeong, Younggun Shin  

**Link**: [PDF](https://arxiv.org/pdf/2605.06737)  

**Abstract**: Autonomous agents based on Large Language Models (LLMs) are increasingly being utilized in complex software systems. However, reliability remains a significant challenge due to unpredictable failures such as hallucinations, execution errors, and inconsistent reasoning. This paper proposes a reliability-aware self-healing framework for LLM-based software agents. The framework integrates failure detection, reliability assessment, and automated recovery mechanisms. First, we define a taxonomy of failure types and introduce a quantitative reliability assessment model. Next, we propose a failure detection method that identifies abnormal agent behavior based on execution patterns and output consistency. Finally, we design a self-healing mechanism that dynamically recovers from failures through adaptive replanning and corrective prompting strategies. The proposed framework was implemented in a multi-agent workflow environment and evaluated using real-world task scenarios. Experimental results demonstrate that our approach significantly increases task success rates, reduces failure propagation, and enhances overall system robustness compared to existing methods. In particular, this study distinguishes itself by establishing an integrated monitoring system that combines the agent's internal reasoning process with external execution results. These findings are expected to contribute to securing the stability of advanced autonomous systems and lowering the barriers to LLM adoption in production environments. 

---
# STDA-Net: Spectrogram-Based Domain Adaptation for cross-dataset Sleep Stage Classification 

**Authors**: Unaza Tallal, Shruti Kshirsagar, Ankita Shukla  

**Link**: [PDF](https://arxiv.org/pdf/2605.06736)  

**Abstract**: Accurate sleep stage classification across datasets remains challenging due to variability in EEG channel montages, sampling rates, recording environments, and subject populations. Although deep learning has shown considerable promise for automated sleep staging, most existing cross-dataset methods rely on one-dimensional EEG signal representations, whereas the use of two-dimensional spectrogram-based inputs within an unsupervised domain adaptation framework has remained largely unexplored. Here, we propose STDA-Net (Spectrogram-based Temporal Domain Adaptation Network), a framework that combines a convolutional neural network (CNN) for spectrogram-based feature extraction, a bidirectional long short-term memory (BiLSTM) module for temporal modeling of sleep dynamics, and a domain-adversarial neural network (DANN) for source-to-target feature alignment without requiring any labeled target-domain data during training. Experiments are conducted on three publicly available datasets Sleep-EDF, SHHS-1, and SHHS-2 under six cross-dataset transfer settings. Results show that the proposed framework achieves an average accuracy of 89.03% and an average macro F1-score of 87.64%, consistently outperforming existing 1D baseline methods in terms of balanced classification performance, with substantially lower variance across five independent runs, indicating improved stability and reproducibility. Overall, these findings demonstrate that 2D spectrogram-based representations, combined with temporal modeling and adversarial domain adaptation, provide a robust and competitive alternative to conventional 1D EEG inputs for cross-dataset sleep staging. 

---
# Gated QKAN-FWP: Scalable Quantum-inspired Sequence Learning 

**Authors**: Kuo-Chung Peng, Samuel Yen-Chi Chen, Jiun-Cheng Jiang, Chen-Yu Liu, En-Jui Kuo, Yun-Yuan Wang, Prayag Tiwari, Andrea Ceschini, Chi-Sheng Chen, Yu-Chao Hsu, Chun-Hua Lin, Tai-Yue Li, Antonello Rosato, Massimo Panella, Simon See, Saif Al-Kuwari, Kuan-Cheng Chen, Nan-Yow Chen, Hsi-Sheng Goan  

**Link**: [PDF](https://arxiv.org/pdf/2605.06734)  

**Abstract**: Fast Weight Programmers (FWPs) encode temporal dependencies through dynamically updated parameters rather than recurrent hidden states. Quantum FWPs (QFWPs) extend this idea with variational quantum circuits (VQCs), but existing implementations rely on multi-qubit architectures that are difficult to scale on noisy intermediate-scale quantum (NISQ) devices and expensive to simulate classically. We propose gated QKAN-FWP, a fast-weight framework that integrates FWP with Quantum-inspired Kolmogorov-Arnold Network (QKAN) using single-qubit data re-uploading circuits as learnable nonlinear activation, known as DatA Re-Uploading ActivatioN (DARUAN). We further introduce a scalar-gated fast-weight update rule that stabilizes parameter evolution, supported by a theoretical analysis of its adaptive memory kernel, geometric boundedness, and parallelizable gradient paths. We evaluate the framework across time-series benchmarks, MiniGrid reinforcement learning, and highlight real-world solar cycle forecasting as our main practical result. In the long-horizon setting with 528-month input window and 132-month forecast horizon, our 12.5k-parameter model achieves lower scaled Mean Square Error (MSE), peak amplitude error, and peak timing error than a suite of classical recurrent baselines with up to 13x more parameters, including Long Short-Term Memory (LSTM) networks (25.9k-89.1k parameters), WaveNet-LSTM (167k), Vanilla recurrent neural network (11.5k), and a Modified Echo State Network (132k). To validate NISQ compatibility, we further deploy the trained fast programmer on IonQ and IBM Quantum processors, recovering forecasting accuracy within 0.1% relative MSE of the noiseless simulator at 1024 shots. These results position gated QKAN-FWP as a scalable, parameter-efficient, and NISQ-compatible approach to quantum-inspired sequence modeling. 

---
# Beyond Factor Aggregation: Gauge-Aware Low-Rank Server Representations for Federated LoRA 

**Authors**: Jinqian Chen, Chang Liu, Jihua Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.06733)  

**Abstract**: Federated LoRA enables parameter-efficient adaptation of large language models under decentralized data and limited client this http URL, directly averaging LoRA factors is representation-dependent: the same intrinsic update admits infinitely many gauge-equivalent factorizations, so factor-level aggregation can change under arbitrary coordinate choices while the underlying update remains unchanged. This reveals a semantic mismatch in existing federated LoRA aggregation rules. We propose \textbf{GLoRA}, a gauge-aware server representation for federated this http URL of aggregating raw factors, GLoRA estimates a consensus update subspace from client projectors and aggregates client updates in shared reference coordinates, thereby representing semantic update aggregation entirely in low-rank form. To support heterogeneous client capacities, GLoRA further provides a rank-compatible readout that instantiates adapters of different ranks from the same server state without dense update reconstruction. Experiments on GLUE and SuperNI show that GLoRA consistently outperforms federated LoRA baselines under data, resource, and task heterogeneity, including heterogeneous client ranks, sparse participation, larger backbones, and unseen-task evaluation. GLoRA also achieves a favorable efficiency--performance trade-off, suggesting that effective federated LoRA requires not merely averaging low-rank factors, but defining a semantically meaningful server-side representation for aggregation. 

---
# The E$Δ$-MHC-Geo Transformer: Adaptive Geodesic Operations with Guaranteed Orthogonality 

**Authors**: Arash Shahmansoori  

**Link**: [PDF](https://arxiv.org/pdf/2605.06729)  

**Abstract**: We present the E$\Delta$-MHC-Geo Transformer, a novel architecture that unifies Manifold-Constrained Hyper-Connections (mHC), Deep Delta Learning (DDL), and the Cayley transform to obtain input-adaptive, unconditionally orthogonal residual connections. Unlike DDL, whose Householder operator is orthogonal only at $\beta \in \{0,2\}$, our Data-Dependent Cayley rotation $Q(x)=(I+(\beta/2)A(x))^{-1}(I-(\beta/2)A(x))$ preserves orthogonality for all $\beta$ and all inputs. To handle negation, an eigenvalue $-1$ case that Cayley provably excludes, we introduce the E$\Delta$-MHC-Geo Hybrid, which combines Cayley rotation with Householder reflection via a learned operator-selection gate $X'=\gamma(X)Q(X)X+(1-\gamma(X))H_2(X)X$. A midpoint-collapse regularizer, $4\gamma(1-\gamma)$, encourages boundary gate decisions, where each selected component is orthogonal. In matched-parameter comparisons, with approximately 1.79M parameters per model and mean +/- standard deviation over 3 seeds, against four baselines including the concurrent JPmHC, E$\Delta$-MHC-Geo achieves the best long-horizon stability, 1.9x over JPmHC and 3.8x over GPT; the best near-$\pi$ rotation loss, 4.5x over JPmHC on single-plane; strong norm preservation, with 0.001 mean deviation; and 0.96 negation cosine alignment in a diagnostic reflection probe, all with 33% fewer layers. While JPmHC's wider representation excels on pure rotation, its finite Cayley residual mixer excludes an exact $\lambda=-1$ operator and has no reflection branch, motivating our hybrid approach for accessing both connected components of $O(n)$. 

---
# OmicsLM: A Multimodal Large Language Model for Multi-Sample Omics Reasoning 

**Authors**: Maciej Sypetkowski, Joanna Krawczyk, Łukasz Smoliński, Remigiusz Kinas, Przemysław Pietrzak, Tomasz Jetka, Rafał Powalski  

**Link**: [PDF](https://arxiv.org/pdf/2605.06728)  

**Abstract**: Interpreting transcriptomic data is one of the most common analytical tasks in modern biology. Yet most current models either consume expression profiles without producing natural-language biological explanations, or reason in language without direct access to quantitative omics measurements. We introduce OmicsLM, a multimodal LLM that connects quantitative omics profiles with natural-language biological tasks. OmicsLM represents each transcriptomic profile as a compact continuous representation within the LLM context. This interface preserves quantitative expression signal while allowing natural-language instructions, explicit gene mentions, and multiple interleaved biological samples to be processed together in one model context. We train OmicsLM on more than 5.5 million instruction-following examples spanning over 70 task types, combining continuous transcriptomic inputs, experimental data rendered through diverse language templates, and free-text biological knowledge and question-answering data. This mixture covers cell type annotation, perturbation prediction, clinical prediction, pathway reasoning, and open-ended biological question answering. Existing benchmarks evaluate either profile-level prediction or text-only biological QA, leaving language-guided, multi-sample reasoning over real expression profiles unmeasured. To close this gap, we introduce GEO-OmicsQA, a benchmark for multi-sample biological question answering built from real Gene Expression Omnibus (GEO) studies. We demonstrate that OmicsLM can use expression profiles directly and perform comparably to specialized omics models on profile-level tasks, while outperforming both omics-specialized models and general LLMs on language-guided biological reasoning over expression data. 

---
# Enabling Unsupervised Training of Deep EEG Denoisers With Intelligent Partitioning 

**Authors**: Qiyu Rao, Haozhe Tian, Homayoun Hamedmoghadam, Danilo Mandic  

**Link**: [PDF](https://arxiv.org/pdf/2605.06724)  

**Abstract**: Denoising wearable electroencephalogram (EEG) is inherently challenging since neural activity is not only subtle but also inseparable from spectrally overlapping noise artifacts. Classical signal processing methods, relying on fixed or heuristic rules, cannot handle the time-varying pervasive artifacts in wearable EEGs. Deep learning methods, on the other hand, show promise in decomposition-free EEG denoising using highly expressive neural networks, but the training requires artifact-free EEG, which is inherently unobtainable. To address this, we propose Intelligent Partitioning for Self-supervised Denoising (iPSD). Our method eliminates the need for clean references by learning to partition an input EEG segment into independent noisy realizations with the same underlying signal. This enables self-supervision of deep learning denoisers, even in zero-shot settings where only a single EEG segment to be denoised is available. We validate iPSD through extensive experiments, including validations on wearable EEG from in-ear sensors. The results show that iPSD achieves state-of-the-art performance, most notably under extremely low signal-to-noise ratios (down to -10 dB) and challenging artifacts (e.g., EMG), with spectral fidelity orders of magnitude higher than competitive baselines. 

---
# Conditional generation of antibody sequences with classifier-guided germline-absorbing discrete diffusion 

**Authors**: Justin Sanders, Luca Giancardo, Lan Guo, Yue Zhao, Kemal Sonmez, Nina Cheng, Melih Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2605.06720)  

**Abstract**: Antibody therapeutics are among the most successful modern medicines, yet computationally designing antibodies with desirable binding and developability properties remains challenging. While protein language models (pLMs) have emerged as powerful tools for antibody sequence design, existing approaches largely suffer from two key limitations: they predominantly memorize germline sequences rather than modeling biologically meaningful somatic variation, and they offer limited support for flexible classifier-guided conditional generation. We address these challenges through two primary contributions. First, we demonstrate that discrete diffusion fine-tuning achieves strong language modeling performance on antibody sequences while allowing for generation conditioned on any off-the-shelf classifier. Second, we introduce germline absorbing diffusion, a novel modification of the discrete diffusion noise process in which the germline sequence - rather than a masked sequence - serves as the absorbing state. This biologically motivated inductive bias restricts the model to learning the trajectory from germline to observed sequence, effectively excluding genetic variation and V(D)J recombination statistics from the learned distribution and dramatically mitigating germline bias. We show that germline diffusion improves non-germline residue prediction accuracy from 26 percent to 46 percent, approaching the theoretical upper bound set by true biological variability. We then demonstrate the utility of our germline diffusion model on the conditional generation tasks of sampling antibodies with improved hydrophobicity and predicted binding affinity. On both tasks our model shows an improved tradeoff between class adherence and sample quality, significantly outperforming EvoProtGrad, a popular strategy to sample from pLMs with gradient-based discrete Markov Chain Monte Carlo. 

---
# Agentic Coding Needs Proactivity, Not Just Autonomy 

**Authors**: Nghi D. Q. Bui, Georgios Evangelopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2605.06717)  

**Abstract**: Coding agents are rapidly changing the landscape of software development, moving from inline completion to autonomous systems that edit repositories, open pull requests, respond to issues, and run scheduled or webhook triggered routines across the development life cycle. The next generation is increasingly described as proactive and long-horizon: agents should notice relevant changes before the developer asks, connect signals across tools, decide when to interrupt, and carry preferences across sessions. Yet the field still lacks a clear account of what proactivity means for software development, how it differs from autonomy, what acceptance criteria proactive long-horizon tasks should satisfy, and which metrics determine whether unsolicited agent behavior is useful rather than merely active. Proactive coding agents should be evaluated by the quality and improvement of their insight policy: the policy that decides what matters next, what evidence supports it, whether to show it, and how to adapt after feedback. This view is grounded in the principles of mixed initiative interaction. We propose a three level taxonomy of proactivity (Reactive, Scheduled, and Situation Aware), compare contemporary coding agents against five practical criteria, and sketch an active user simulation protocol with three evaluation targets: Insight Decision Quality (IDQ), Context Grounding Score (CGS), and Learning Lift 

---
# Edge Deep Learning in Computer Vision and Medical Diagnostics: A Comprehensive Survey 

**Authors**: Yiwen Xu, Tariq M. Khan, Yang Song, Erik Meijering  

**Link**: [PDF](https://arxiv.org/pdf/2605.06714)  

**Abstract**: Edge deep learning, a paradigm change reconciling edge computing and deep learning, facilitates real-time decision making attuned to environmental factors through the close integration of computational resources and data sources. Here we provide a comprehensive review of the current state of the art in edge deep learning, focusing on computer vision applications, in particular medical diagnostics. An overview of the foundational principles and technical advantages of edge deep learning is presented, emphasising the capacity of this technology to revolutionise a wide range of domains. Furthermore, we present a novel categorisation of edge hardware platforms based on performance and usage scenarios, facilitating platform selection and operational effectiveness. Following this, we dive into approaches to effectively implement deep neural networks on edge devices, encompassing methods such as lightweight design and model compression. Reviewing practical applications in the fields of computer vision in general and medical diagnostics in particular, we demonstrate the profound impact edge-deployed deep learning models can have in real-life situations. Finally, we provide an analysis of potential future directions and obstacles to the adoption of edge deep learning, with the intention to stimulate further investigations and advancements of intelligent edge deep learning solutions. This survey provides researchers and practitioners with a comprehensive reference shedding light on the critical role deep learning plays in the advancement of edge computing applications. 

---
# Agentic AI and the Industrialization of Cyber Offense: Forecast, Consequences, and Defensive Priorities for Enterprises and the Mittelstand 

**Authors**: Christopher Koch  

**Link**: [PDF](https://arxiv.org/pdf/2605.06713)  

**Abstract**: Agentic AI systems can plan, call tools, inspect code, interact with web applications, and coordinate multi-step workflows. These same capabilities change the economics of cyber offense. The central near-term risk is not that every low-skill criminal immediately becomes a frontier exploit researcher; it is that agentic AI compresses the attack lifecycle by lowering the cost of reconnaissance, phishing, credential abuse, vulnerability triage, exploit adaptation, and post-compromise decision support. This paper synthesizes current public evidence from national cybersecurity agencies, industry threat reports, agent security guidance, and research on LLM agents cyber capabilities. It introduces a Three Channel Agentic Cyber Risk Model and an Agentic Attack Compression Model, uses the 2026 Linux kernel Copy Fail incident as a case study for foothold-to-root acceleration, and develops a 2026 to 2028 forecast for large enterprises and the German and European Mittelstand. The paper concludes with a prioritized defense roadmap. Organizations should treat agentic AI security as an immediate operational problem: identity, phishing resistant authentication, patch velocity, CI/CD and Linux/container hardening, agent governance, telemetry, and recovery readiness must be strengthened now. 

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
# Multimodal synthesis of MRI and tabular data with diffusion in a joint latent space via cross-attention 

**Authors**: Daniel Mensing, Jan Kapar, Jochen G. Hirsch, Matthias Günther, Horst Hahn, Marvin N. Wright  

**Link**: [PDF](https://arxiv.org/pdf/2605.06699)  

**Abstract**: We propose a multimodal latent diffusion model that jointly synthesizes volumetric magnetic resonance imaging (MRI) and tabular clinical data within a shared latent space via cross-attention. This approach enables coherent joint representation learning of MRI and tabular modalities for generative modeling. Our model utilizes a variational autoencoder to fuse the two modalities before diffusion-based synthesis, allowing modality-appropriate reconstruction with separate decoders for MRI and tabular data. We evaluated the framework on data from the German National Cohort (NAKO Gesundheitsstudie), comprising over 10,000 participants with MRI scans and clinical tabular features such as age, sex, body measurements, and ethnicity. The generated MRI volumes exhibited anatomical plausibility and body composition consistent with the synthesized tabular attributes. Quantitative evaluation using Fréchet distance and precision-recall metrics confirmed high-fidelity image generation. In the tabular modality, our model outperformed CTGAN across standard evaluation metrics and achieved results comparable to TVAE, demonstrating competitive performance relative to established unimodal baselines. This work is, to our knowledge, the first to demonstrate the feasibility of jointly modeling MRI and mixed-type tabular data in a single latent diffusion framework, offering a proof-of-concept for generating coherent synthetic multimodal patient data and aligning with the broader goal of developing digital twins in healthcare. 

---
# Toeplitz MLP Mixers are Low Complexity, Information-Rich Sequence Models 

**Authors**: Benjamin L. Badger, Ethan Roland  

**Link**: [PDF](https://arxiv.org/pdf/2605.06683)  

**Abstract**: Transformer-based large language models are in some respects limited by the quadratic time and space computational complexity of attention. We introduce the Toeplitz MLP Mixer (TMM), a transformer-like architecture that swaps attention for triangular-masked Toeplitz matrix multiplication over the sequence dimension resulting in $\mathcal{O} (dn \log n)$ time and $\mathcal O(dn)$ space complexity during training and $\mathcal O(dn)$ time and space at inference prefill. Despite the lack of sophisticated input modulation or state maintenance present in other sub-quadratic architectures, TMMs yield greater training efficiency in terms of loss achieved per compute and device memory. We demonstrate that TMMs are capable of retaining more input information resulting in improved copying ability, which we argue results from a lack of architectural biases. Consistent with higher input information retention, TMMs exhibit superior information retrieval and in-context learning benchmark accuracy compared to comparable architectures. We conclude with an analysis from the perspective of operator index theory and show that, counterintuitively, trained Toeplitz layers of causal non-invertible models are more likely to be invertible or nearly so than models that are actually invertible over their inputs. 

---
# Domain-level metacognitive monitoring in frontier LLMs: A 33-model atlas 

**Authors**: Jon-Paul Cacioli  

**Link**: [PDF](https://arxiv.org/pdf/2605.06673)  

**Abstract**: Aggregate metacognitive quality scores mask within-model variation across MMLU benchmark domains. We administered 1,500 MMLU items (250 per domain, under an a priori six-domain grouping) to 33 frontier LLMs from eight model families and computed Type-2 AUROC per model-domain cell using verbalized confidence (0-100). Total observations: 47,151. Every model with above-chance aggregate monitoring showed non-trivial domain-level variation. Applied/Professional knowledge was reliably the easiest benchmark domain to monitor (mean AUROC = .742, ranked top-2 in 21 of 33 models); Formal Reasoning and Natural Science were reliably the hardest (one of the two ranked bottom-2 in 27 of 33 models). The three middle domains were statistically indistinguishable (Kendall's W = .164). A subject-level coherence analysis (within-domain similarity ratio = 0.95) confirms the six-domain grouping is a pragmatic benchmark taxonomy, not a validated latent construct. Within-family profile-shape clustering is significant for Anthropic, Google-Gemini, and Qwen (permutation p < .0001) but not DeepSeek, Google-Gemma, or OpenAI. Gemma 4 31B showed a +.202 AUROC improvement over Gemma 3 27B. Three models classified Invalid on binary KEEP/WITHDRAW probes produced normal profiles under verbalized confidence, confirming probe-format specificity. Bootstrap 95% CIs on 198 cells have median width .199. Split-half aggregate stability r = .893; profile-level split-half is weaker (grand median r = .184). These results show stable benchmark-domain variation obscured by aggregate metrics, and support benchmark-stage domain screening as a step before deployment in specific application areas. 

---
# Evaluating Prompt Injection Defenses for Educational LLM Tutors: Security-Usability-Latency Trade-offs 

**Authors**: Alexandre Cristovão Maiorano  

**Link**: [PDF](https://arxiv.org/pdf/2605.06669)  

**Abstract**: Educational LLM tutors face a core AI alignment challenge: they must follow user intent while preserving pedagogical constraints and safety policies. We present an evaluation methodology for prompt-injection defenses in this setting, showing that guardrail design entails explicit trade-offs among adversarial robustness, benign-task usability, and response latency. We evaluate a domain-specific multi-layer safeguard pipeline combining deterministic pattern filters, structural validation, contextual sandboxing, and session-level behavioral checks. On a controlled holdout benchmark with 480 queries (369 injection, 111 benign), the pipeline reaches 46.34% bypass, 0.00% false positive rate, and 2.50 ms average latency -- an operating point that prioritizes pedagogical usability (zero false positives) while maintaining measurable attack resistance. We provide a reproducible benchmark protocol for head-to-head comparison under identical conditions, including stratified bootstrap confidence intervals, paired McNemar significance tests, and direct evaluation of Prompt Guard and NeMo Guardrails on the same split with unified instrumentation. Results expose operational trade-offs: NeMo reaches 0% bypass at 16.22% FPR and 1.3s latency, while Prompt Guard yields 38.48% bypass with 3.60% FPR. The framework supports evidence-based guardrail selection for AI tutoring systems under different institutional risk and usability requirements. 

---
# CommFuse: Hiding Tail Latency via Communication Decomposition and Fusion for Distributed LLM Training 

**Authors**: Rezaul Karim, Austin Wen, Wang Zongzuo, Weiwei Zhang, Yang Liu, Walid Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2604.24013)  

**Abstract**: The rapid growth in the size of large language models has necessitated the partitioning of computational workloads across accelerators such as GPUs, TPUs, and NPUs. However, these parallelization strategies incur substantial data communication overhead significantly hindering computational efficiency. While communication-computation overlap presents a promising direction, existing data slicing based solutions suffer from tail latency. To overcome this limitation, this research introduces a novel communication-computation overlap technique to eliminate this tail latency in state of the art overlap methods for distributed LLM training. The aim of this technique is to effectively mitigate communication bottleneck of tensor parallelism and data parallelism for distributed training and inference. In particular, we propose a novel method termed CommFuse that replaces conventional collective operations of reduce-scatter and all-gather with decomposed peer-to-peer (P2P) communication and schedules partitioned computations to enable fine-grained overlap. Our method provides an exact algorithm for reducing communication overhead that eliminates tail latency. Moreover, it presents a versatile solution compatible with data-parallel training and various tensor-level parallelism strategies, including TPSP and UP. Experimental evaluations demonstrate that our technique consistently achieves lower latency, superior Model FLOPS Utilization (MFU), and high throughput. 

---
# Consensus Entropy: Harnessing Multi-VLM Agreement for Self-Verifying and Self-Improving OCR 

**Authors**: Yulong Zhang, Tianyi Liang, Xinyue Huang, Erfei Cui, Guoqing Wang, Xu Guo, Chenhui Li, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11101)  

**Abstract**: Optical Character Recognition (OCR) is fundamental to Vision-Language Models (VLMs) and high-quality data generation for LLM training. Yet, despite progress in average OCR accuracy, state-of-the-art VLMs still struggle with detecting sample-level errors and lack effective unsupervised quality control. We introduce Consensus Entropy (CE), a training-free, model-agnostic metric that estimates output reliability by measuring inter-model agreement entropy. The core insight is that correct predictions converge in output space, while errors diverge. Based on CE, we develop CE-OCR, a lightweight multi-model framework that verifies outputs by ensemble agreement, selects the best outputs, and further improves efficiency through adaptive routing. Experiments demonstrate that CE is robust for quality verification, improving F1 scores by 42.1% over VLM-as-Judge. CE-OCR achieves consistent OCR gains, outperforming self-consistency and single-model baselines at the same cost. Notably, CE requires no training or supervision, enabling plug-and-play integration. Code: this https URL. 

---
