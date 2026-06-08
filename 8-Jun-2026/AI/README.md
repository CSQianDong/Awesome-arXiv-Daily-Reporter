# How AI Agents Reshape Knowledge Work: Autonomy, Efficiency, and Scope 

**Authors**: Jeremy Yang, Kate Zyskowski, Noah Yonack, Jerry Ma  

**Link**: [PDF](https://arxiv.org/pdf/2606.07489)  

**Abstract**: Frontier AI systems are bridging the gap between intelligence and utility by shifting from conversational assistants to autonomous agents that execute tasks end to end. Using production data from Perplexity's Search and Computer products, we study this transition by examining how AI agents accelerate and reshape knowledge work. Three key empirical findings emerge. First, using sessions with near-identical initial query pairs as natural experiments for the same underlying task attempted with both products, Computer performs 26 minutes of autonomous work per user session, versus 33 seconds for Search. Computer automates task decomposition and execution that Search users might otherwise manually orchestrate and implement. As a result, Computer shifts follow-up query distribution toward higher-order work such as verification and extension. Autonomy also increases execution quality, with per-query dissatisfaction rates 55% lower on Computer than on Search. Second, due to its autonomy advantage, Computer reduces completion time from 269 to 36 minutes on matched tasks, lowering estimated time and cost by 87% and 94%, respectively, compared to humans equipped with Search alone. Third, Computer changes the scope of work that users attempt: Computer queries more often cross occupational boundaries, require higher-order cognition, draw on broader expertise, take the form of composite tasks that bundle interdependent subtasks into a single query, and unlock work activities that are essentially absent from Search usage among the same users. Together, the evidence indicates that AI agents accelerate workflows, enhance output quality, reduce costs, and expand the breadth and depth of automated work. 

---
# Act As a Real Researcher: A Suite of Benchmarks Evaluating Frontier LLMs and Agentic Harnesses in Research Lifecycle 

**Authors**: Jiayu Wang, Weijiang Lv, Bowen Fu, Jing Fu, Jiayi Song, Lingyu Zhang, Lanxuan Xue, Luodi Chen, Zepeng Xin, Kaiyu Li, Xiangyong Cao  

**Link**: [PDF](https://arxiv.org/pdf/2606.07462)  

**Abstract**: As foundation models advance and agent scaffolding becomes increasingly sophisticated, agents have demonstrated remarkable proficiency in complex, long-horizon coding tasks and even autonomous experiment execution. Despite their evolution from research assistants into autonomous research agents, these systems still exhibit significant limitations in field sensitivity, research ethics, and nuanced scientific judgment. Consequently, frontier agents remain unable to fully replace human researchers. To bridge this gap, we conceptualize the AARR (Act As a Real Researcher) benchmark series. Unlike existing benchmarks that primarily assess macro-level execution capabilities, AARR focuses on whether agents can emulate the professionalism, thoroughness, and nuanced reasoning that characterize human researchers in granular research scenarios. In this work, we propose AARRI-Bench (Act As a Real Research Intern), the first benchmark in this series. We conduct extensive experiments across frontier models and agentic systems, revealing that even the best-performing configuration (Mini-SWE-Agent with Claude Opus 4.7) achieves only 68.3\% success rate, frequently overlooking subtle yet critical details that are obvious to real human researchers. Our results indicate that developing researcher-like AI requires further exploration of research behavior, rather than merely complex scaffolding. Our data is released at this https URL. 

---
# Online Pandora's Box for Contextual LLM Cascading 

**Authors**: Alexandre Belloni, Yan Chen, Yehua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2606.07392)  

**Abstract**: Motivated by Large Language Model (LLM) cascading, we propose an online contextual Pandora's Box model for adaptively querying and selecting LLM APIs. In each period, a decision-maker observes a request context and faces a two-phase decision problem. In the query phase, the decision-maker sequentially queries APIs, where each query reveals a generated output and the decision-maker incurs an (output-dependent) cost. In the selection phase, the decision-maker selects one of the generated outputs to deploy and observes only the downstream reward of the deployed output. This output-mediated feedback structure differs from classical online contextual Pandora's Box models, in which opening a box directly reveals its reward.
Rather than estimating the full conditional output and cost distributions of each API, we directly model the reservation index and develop a learning approach for the query phase. Specifically, we impose a parametric structure on the contextual reservation index functions induced by the classical Weitzman's policy. Our policy combines generalized method of moments (GMM) type estimation of these reservation indices with UCB-style confidence bounds for both these indices and the shared output-level reward evaluator. Under regularity conditions, we prove that the resulting policy achieves dimension-dependent $\widetilde O(\sqrt T)$ cumulative regret over a horizon of $T$ periods. 

---
# Off-Policy Evaluation with Strategic Agents via Local Disclosure 

**Authors**: Kiet Q. H. Vo, Abbavaram Gowtham Reddy, Julian Rodemann, Siu Lun Chau, Krikamol Muandet  

**Link**: [PDF](https://arxiv.org/pdf/2606.07308)  

**Abstract**: We study off-policy evaluation (OPE) under strategic behavior where decision subjects (or agents) respond to a decision maker's policy by strategically modifying their covariates. Such behavior induces a policy-dependent covariate shift, breaking the standard assumption in existing methods that covariates are exogenous to the policy. Related work addresses this challenge by imposing strong assumptions such as repeated interactions or full knowledge of agents' response behavior, substantially limiting its applicability to OPE. In contrast, we consider a one-shot OPE setting where the decision maker has only partial knowledge of the agents' response behavior. Our key insight is that disclosing local information through post-hoc explanations reveals agents' pre-strategic covariates prior to adaptation, mitigating the information loss induced by strategic behavior. Leveraging this structure, we estimate a statistical model for the agents' responses and construct a doubly robust estimator for policy value. By assuming that the agents' cost sensitivity follows a conditional log-normal distribution, we establish consistency of the proposed estimator and validate our approach empirically. More broadly, our results highlight how interaction design can mitigate information asymmetry by revealing otherwise hidden structure in agents' strategic responses. 

---
# DuMate-DeepResearch: An Auditable Multi-Agent System with Recursive Search and Rubric-Grounded Reasoning 

**Authors**: Lingyong Yan, Can Xu, Yukun Zhao, Wenxuan Li, Qingyang Chen, Jiulong Wu, Wenli Song, Xiangnan Li, Weixian Shi, Yiqun Chen, Xuchen Ma, Yuchen Li, Jiashu Zhao, Shuaiqiang Wang, Jianmin Wu, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2606.07299)  

**Abstract**: Deep Research (DR) has emerged as a new agentic paradigm to tackle complex, open-ended research tasks, demanding systems that can iteratively frame problems, acquire evidence, verify sources, and synthesize long-form reports. In practice, however, current DR systems are constrained by four interrelated limitations: long-horizon planning over an underspecified scope, the bottleneck of decomposing and scheduling such tasks within a single agent, hallucination risk in long-form synthesis, and limited process auditability. This technical report presents DuMate-DeepResearch, a multi-agent DR framework built on the Qianfan Agent Foundry. The framework decouples the Agent Core, which handles task understanding, planning, and scheduling, from an extensible Tool Ecosystem for retrieval, evidence acquisition, and report rendering, making every intermediate decision and tool invocation explicitly traceable. Building on this infrastructure, DuMate-DeepResearch further introduces three mechanisms: (i) a graph-based dynamic planning strategy expands the research roadmap coarse-to-fine and continuously revises it through reflection, re-planning, backtracking, and parallel branching; (ii) a recursive two-level execution design delegates each complex search sub-task to an inner Search Agent that runs its own planning loop, isolating noisy retrieval and stabilizing long-horizon execution; (iii) a rubric-based test-time optimization mechanism dynamically generates task-specific quality criteria and uses them as live reasoning scaffolds for evidence-grounded synthesis and adaptive stopping. Across two deep research benchmarks, DuMate-DeepResearch establishes new state-of-the-art results: the best overall score (58.03%) on DeepResearch Bench, and the best overall score (61.95%) on DeepResearch Bench II while ranking first in information recall and analysis. 

---
# TOPSIS-RAD: Ranking According to Desires 

**Authors**: Leonardo Fernandes Costa, Helder Gomes Costa, Diogo Lima, Brunno Rodrigues  

**Link**: [PDF](https://arxiv.org/pdf/2606.07253)  

**Abstract**: Traditional TOPSIS derives its reference points -- the Positive Ideal Solution ($PIS$) and Negative Ideal Solution ($NIS$) -- from the observed alternative set, making rankings susceptible to misalignment with decision-maker (DM) requirements, sensitivity to outlier performances, and rank reversal. This paper proposes TOPSIS-RAD, which addresses these issues by incorporating two arrays of DM-defined reference levels. Vetoed Performance Levels ($VPL$) exclude non-viable alternatives before normalisation, preventing them from distorting the ranking frontiers. Desired Performance Levels ($DPL$) cap performances at the DM's desired level before normalisation, anchoring the $PIS$ in explicit aspirations rather than dataset extremes. Three toy examples demonstrate each mechanism: $VPL$ reshapes normalisation boundaries by removing a non-viable alternative; fixed $DPL$ frontiers stabilise rankings by limiting the influence of performances well above the desired level. The method preserves the familiar distance-based structure of TOPSIS while grounding the ranking in stable, DM-specified boundaries. Limitations and future research directions are also discussed. 

---
# Think Fast: Estimating No-CoT Task-Completion Time Horizons of Frontier AI Models 

**Authors**: Dewi Gould, Francis Rhys Ward, Anders Cairns Woodruff, Rauno Arike, Josh Hills, Alex Serrano, Ida Caspary, Jason Ross Brown, Jo J. Jiao, Patrick Leask, Twm Stone, Ram Potham, Ionut Gabriel Stan, Harry Mayne, Simeon Hellsten, Shubhorup Biswas, Ariana Azarbal, William L. Anderson, Elle Najt, Ryan Greenblatt, Julian Stastny  

**Link**: [PDF](https://arxiv.org/pdf/2606.07157)  

**Abstract**: Many efforts to ensure frontier AI models are safe rely on monitoring their chain-of-thought (CoT) reasoning. If models become able to perform sufficiently complex reasoning internally, without explicit thinking tokens, this would undermine such oversight. We measure how well frontier models reason without CoT across a suite of over 30,000 questions spanning 43 benchmarks in domains including math, coding, puzzles, causality, theory-of-mind, and strategic reasoning. To compare models against humans, we estimate the $50\%$-task-completion time horizon (TH): the human time required for tasks a model completes with $50\%$ success rate. We complement this with a $50\%$ reasoning token horizon: the minimum number of o3-mini reasoning tokens needed for tasks a model solves with $50\%$ success rate. We find that the no-CoT $50\%$ TH of frontier models has been doubling roughly every year over the past six years, with GPT-5.5's TH reaching over 3 minutes and reasoning token horizon exceeding 1,500 tokens. Our median estimates predict that frontier no-CoT THs could exceed 7 minutes by 2028, and 25 minutes by 2030, though these projections carry substantial uncertainty. We recommend frontier developers track this explicitly. 

---
# Beyond Post-hoc Explanation: Toward Glassbox AI via Probabilistic Mediation 

**Authors**: Manuele Leonelli  

**Link**: [PDF](https://arxiv.org/pdf/2606.07113)  

**Abstract**: Large language models are rapidly becoming infrastructural components in high-stakes institutional settings, including public administration, legal reasoning, and healthcare, where opacity is not merely inconvenient but institutionally and legally untenable. Existing approaches to explainability are predominantly post-hoc, offering unstable, non-contestable accounts that have no formal relationship to the reasoning process that produced the output. We argue that the problem is not the absence of explanation but the absence of structured reasoning in the first place. This paper makes the case for a fundamentally different architecture, which we call the Glassbox Framework, in which Bayesian networks serve as transparent, ante-hoc mediation layers for generative models. Bayesian networks encode domain knowledge, causal assumptions, and probabilistic dependencies before inference occurs, enabling auditable reasoning traces, uncertainty quantification, and contestable outputs. We characterise the architecture of this framework and ground it in a benefit eligibility scenario, identifying the foundational challenges spanning semantic alignment, dynamic model construction, probabilistic grounding, and human governance that must be solved to realise it at scale. By shifting from post-hoc explanation to ante-hoc probabilistic mediation, this work outlines a principled path toward AI systems that are not only powerful but fundamentally accountable. 

---
# DyCon: Dynamic Reasoning Control via Evolving Difficulty Modeling 

**Authors**: Tengyao Tu, Yulin Li, Hui-Ling Zhen, Libo Qin, Zhoujun Wei, Jinghua Piao, Zhuotao Tian, Yong Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.07108)  

**Abstract**: Recent advances in Large Reasoning Models (LRMs) demonstrate remarkable performance improvements by iteratively reflecting, exploring, and executing complex tasks, yet suffer from inefficiencies due to redundant reasoning, known as "overthinking". Existing methods to mitigate this issue either rely on static difficulty estimates or require task-specific training, and thus fail to adapt to the dynamic complexity during reasoning. In this work, we empirically show that the problem difficulty evolves dynamically throughout the reasoning process and is linearly encoded in the LRM's step-level embeddings. Building on this insight, we propose DyCon, a training-free framework that leverages latent step-level representations to explicitly model the evolving task difficulty, enabling the dynamic control of reasoning depth to mitigate the overthinking issue. Extensive experiments conducted on four models ranging from 4B to 32B, and across twelve benchmarks in math reasoning, general question answering, and coding tasks demonstrate that DyCon significantly enhances reasoning efficiency by reducing redundant steps without sacrificing accuracy or generalization. Project page and code are available at this https URL. 

---
# Front-to-Attractors: Modifying the Front-to-Front Heuristic in Bidirectional Search 

**Authors**: Alvin Zou, Muhammad Suhail Saleem, Maxim Likhachev  

**Link**: [PDF](https://arxiv.org/pdf/2606.07047)  

**Abstract**: Heuristics play a central role in the performance of bidirectional search algorithms, which commonly rely on two main classes. Front-to-end (F2E) heuristics estimate the distance from a state s to the target of the search (the goal for forward search or the start for backward search). In contrast, front-to-front (F2F) heuristics estimate the distance from s to the opposite search frontier using a pairwise function h(s, s'), where s' ranges over frontier states. Although F2F heuristics are typically more informative and therefore reduce the number of node expansions, their reliance on extensive pairwise evaluations incurs substantial computational overhead. To address this limitation, we introduce a new heuristic class, front-to-attractors (F2A), that preserves much of the informativeness of F2F while dramatically reducing its computational cost. Rather than evaluating distances to all states on the opposite frontier, F2A estimates the distance from s to a small, dynamically maintained set of attractors in the opposite search direction. These attractors serve as a surrogate for the full frontier, enabling rich heuristic guidance at a fraction of the computational expense while maintaining the optimality guarantees offered by F2F. We evaluate F2A across multiple domains and show that it reduces the number of pairwise evaluations by up to 11.2x compared to F2F, while achieving 4.8x fewer node expansions than F2E on average. 

---
# Hierarchical Semantic-Constrained Heterogeneous Graph for Audio-Visual Event Localization 

**Authors**: Zhe Yang, Ruyi Zhang, Hongtao Chen, Wenrui Li, Hengyu Man, Wangmeng Zuo, Xiaopeng Fan  

**Link**: [PDF](https://arxiv.org/pdf/2606.07033)  

**Abstract**: Open-vocabulary audio-visual event localization (OV-AVEL) jointly models audio-visual cues to recognize and temporally localize events, including categories unseen during training. Existing methods primarily learn joint audio-visual representations in Euclidean space, but still face two significant challenges. First, the lack of supervision signals for unseen categories makes it difficult to maintain audio-visual consistency across multiple temporal scales. Second, the lack of hierarchical constraints between segment- and video-level semantics prevents the model from establishing semantic consistency across different levels. To address these challenges, we propose a hierarchical semantic constrained heterogeneous graph (HSCHG) for audio-visual event localization framework. We first construct a heterogeneous hierarchical graph in Euclidean space, which includes audio and visual segment nodes and their corresponding video-level nodes. We use multi-directional temporal edges to capture complete temporal information within each modality. Simultaneously, we employ a dual-threshold filtering gated fusion strategy, introducing cross-modal information only when the alignment confidence is high. Furthermore, we introduce bidirectional semantic constraints between segment- and video-level representations to achieve semantic consistency across different levels. Based on this, we map the multi-level audio-visual representations and text prototypes uniformly into hyperbolic space. We use a hierarchical entailment regularization loss to characterize the hierarchical relationships between videos and segments. Extensive experimental results show that our method outperforms existing methods on the OV-AVEL benchmark. Ablation studies further validate the effectiveness of our method. 

---
# StainFlow: Entity-Stain Tracking and Evidence Linking for Process Rewards in GUI Agents 

**Authors**: Haojie Hao, Longkun Hao, Yihang Lou, Yan Bai, Zhenyang Li, Zhichao Yang, Dongshuo Huang, Hongyu Lin, Lanqing Hong, Jiakai Wang, Xianglong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.07027)  

**Abstract**: Reinforcement Learning (RL) has become a promising approach for improving GUI Agents in long-horizon, stochastic digital environments, but trajectory-level success feedback is too sparse to provide reliable credit assignment for intermediate exploration steps. To mitigate this issue, recent studies introduce Process Reward Models (PRMs), which provide finer-grained training feedback through global milestone verification or local step-level evaluation. However, these methods still suffer from two level-specific limitations: global milestone decomposition is subjective and singular, making it difficult to accommodate the multiple valid execution paths in real GUI tasks, while fixed local judging windows may miss long-range key evidence or dilute the decision signal with irrelevant frames. Inspired by stain-tracing mechanisms in network flow analysis, we propose StainFlow, an entity-stain-flow process reward model for GUI Agents. To reduce the subjectivity of global partitioning, we introduce the Global Entity Stain Tracking module, which extracts visually verifiable task entities and tracks how their stain concentrations and states evolve along the trajectory, allowing task phases to be objectively separated by changes in the entity evidence flow. To improve the accuracy of local verification, we introduce the Local Stain Evidence Linking module. Centered on the triggering entities of each candidate key node, it retrieves relevant steps based on their stain concentrations and state changes, and dynamically constructs high-density evidence windows for verifying true key nodes. Extensive experiments on AndroidWorld and OGRBench show that StainFlow relatively improves online RL success by 3.2% and trajectory completion judgment accuracy by 1.8%. 

---
# The Sim-to-Real Gap of Foundation Model Agents: A Unified MDP Perspective 

**Authors**: Xiaoou Liu, Tiejin Chen, Weibo Li, Xiyang Hu, Hua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2606.07017)  

**Abstract**: Foundation model agents are increasingly deployed for real-world decision-making, but suffer from the sim-to-real gap. While robotics and classical control have mature frameworks to address this gap, the foundation model community is treating agent robustness as an entirely novel phenomenon. Our paper proposes formalizing the foundation model agent evaluation and training gap as a classical sim-to-real problem structured entirely around the four elements of a Markov Decision Process, including Observation, Action, Transition, and Reward. In this paper, we set a comprehensive research agenda that translates classical discrepancies into the foundation model domain and advocates for adopting established solutions like domain randomization. We provide concrete examples, such as a multilingual tool calling to demonstrate how severe observation space gaps lead to operationally invalid actions despite correct semantic intent. Ultimately, this agenda aims to drive a paradigm shift, yielding a unified vocabulary and standardized stress test benchmarks to foster a new generation of highly trustworthy agents for reliable real-world applications. 

---
# Teaching the Way, Not the Answer: Privileged Tutoring Distillation for Multimodal Policy Optimization 

**Authors**: Shizhe Xiang, Ke An, Wenlong Yu, Yue Liu, Jian Luan, Pei Fu, Qilong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.07000)  

**Abstract**: Recent post-training methods, particularly Reinforcement Learning with Verifiable Rewards (RLVR), have significantly enhanced the reasoning ability of Large Vision-Language Models (LVLMs). However, the sparse nature of verifiable rewards provides little token-level supervision for failed rollouts, often leading to inefficient exploration in complex multimodal reasoning tasks. Although policy distillation can offer dense guidance, external teacher based methods introduce substantial computational overhead, while answer conditioned tuning methods may expose answer-level information and induce shortcut-like generation behavior. To address these limitations, we propose PTD-PO, a Privileged Tutoring Distillation Policy Optimization framework for RLVR that provides dense guidance without exposing the answer to the student policy. Specifically, PTD-PO constructs structured privileged hints from spatial attention guidance and intermediate textual reasoning steps, and uses them through in-context learning to produce step-wise token-distribution supervision. The student is still optimized under the original answer-free context, and its failed rollouts are aligned with the hint-augmented reference model at the token-distribution level. To further stabilize distillation under the distribution shift between guided and unguided contexts, we introduce a Top-K Jensen-Shannon divergence objective that focuses alignment on informative token probabilities while reducing memory overhead. Experiments on LVLMs ranging from 2B to 8B parameters show that PTD-PO consistently outperforms RLVR and distillation baselines, mitigates entropy collapse, and improves complex multimodal reasoning performance. 

---
# Exploring Agentic Tool-Calling Decisions via Uncertainty-Aligned Reinforcement Learning 

**Authors**: Yijin Zhou, Linqian Zeng, Xiaoya Lu, Wenyuan Xie, Dongrui Liu, Junchi Yan, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2606.06976)  

**Abstract**: Large language model (LLM)-based agents often make suboptimal tool-use decisions, including unsupported tool invocation and hallucinated direct responses, which may accumulate errors throughout multi-step interactions. Existing approaches mainly improve these behaviors through inference-time correction or coarse-grained reward signals based on decision outcomes and structured checklists, leaving the uncertainty characteristics of agent decisions underexplored. We observe that decision-oriented reinforcement learning tends to weaken the uncertainty separation between correct and incorrect actions, resulting in overconfident mistakes and weaker exploration signals. Therefore, we propose TRUST, which incorporates uncertainty quantification into reward design as a repulsive force for maintaining uncertainty separation, and labels lightweight key-turn annotations for unified post-training of multi-turn trajectories. Experimental results across diverse tool-use benchmarks show that TRUST consistently enhances both decision quality and agent performance while maintaining more reliable uncertainty estimates during optimization. 

---
# Accounting for Context: Shaping Moral Credences for Value Alignment 

**Authors**: Jazon Szabo, Sanjay Modgil  

**Link**: [PDF](https://arxiv.org/pdf/2606.06972)  

**Abstract**: Ensuring that agent behaviours are aligned with human moral values inevitably raises the problem of how to account for the plurality of moral perspectives that societies -- and even individuals -- typically adopt. Work on moral uncertainty proposes mechanisms to fairly and democratically aggregate evaluations of actions across different moral theories. However, this paper argues that one needs to account for contextual factors when aggregating moral evaluations. For example, consequentialist perspectives assume an ability to accurately determine how an agent's actions change the world; an assumption that often does not hold in real world settings. We, therefore, formalise agent decision making under moral uncertainty, while also accounting for these kinds of contextual factors. We thereby show that a seemingly commonsensical property -- the weak Pareto principle -- is violated. We argue that this apparent problem is, in fact, a variation of Simpson's paradox, and hence reveals the limitations of aggregation mechanisms that ignore the impact of contextual factors. 

---
# Quantum-Inspired Trace-Augmented Evidence Selection for Reasoning over Structured Hypothesis Spaces 

**Authors**: Laura Wynter, Nirvik Sahoo, Paul Griffin  

**Link**: [PDF](https://arxiv.org/pdf/2606.06941)  

**Abstract**: Large language models (LLMs) now solve a wide range of expert-level exams at or above human level, yet remain brittle on specialised, evidence-intensive domains such as law. On these tasks, errors arise not only from gaps in world knowledge but also from subtle distinctions between pieces of evidence and inconsistent use of supporting evidence. The most common aggregator over sampled chain-of-thought (CoT) traces, majority vote, returns the most popular answer regardless of whether its evidence is actually strongest. We propose to treat the selection of CoT reasoning fragments into a set of evidence as an explicit combinatorial optimisation problem, allowing well-supported but minority hypotheses to override noisy majorities, and to evaluate the approach on legal-reasoning benchmarks that are particularly sensitive to evidence quality. We introduce EP-HUBO (Evidence Pool Higher-Order Binary Optimisation), which generates multiple CoT traces with a small local model, parses fragments into per-hypothesis evidence pools, solves a higher-order unconstrained binary optimisation per pool with quality-derived weights (relevance, specificity, distinctiveness), and delegates a single adjudication call per question to a frontier model. We evaluate EP-HUBO on two evidence-intensive legal benchmarks using both simulated annealing on classical hardware and the Dirac-3 photonic entropy-quantum machine from Quantum Computing Inc. HUBO-style optimisation gives a principled way to aggregate reasoning fragments while preserving minority-but-correct hypotheses, and is most valuable in low-contamination domains where frontier models have not already absorbed the benchmark material. 

---
# Declarative Skills for AI Agents in Knowledge-Grounded Tool-Use Workflows 

**Authors**: M. Danish Lim, I. Danial Bin Sharudin, Wen Han Chen, Cedric Lim, Laura Wynter  

**Link**: [PDF](https://arxiv.org/pdf/2606.06923)  

**Abstract**: We study orchestration mechanisms for tool-using AI agents in realistic customer-service workflows over an unstructured knowledge base. We argue that declarative agents -- AI agents equipped with natural-language skill files appended to the system prompt -- are an effective orchestration paradigm. Concretely, we compare (i) a DeclarativeAgent that reads three domain-specific skill files at inference time and decides its own control flow, (ii) an ImperativeAgent based on a programmatic state machine with explicit phases, and (iii) an unscaffolded baseline agent modeled after the $\tau$-Knowledge benchmark agent. Our ImperativeAgent is motivated by externalised-control inference as in Recursive Language Models and graph-based orchestration frameworks. We formalise the three agents as policy classes within a decentralised partially-observable Markov decision process and analyse their information-theoretic and structural properties; we then test the predicted differences empirically on five language models and two retrieval regimes. Our results show that retrieval quality is a dominant bottleneck for AI agents: when evidence is incomplete or skewed, all agents degrade substantially, and skill files cannot recover lost performance. Under high-quality retrieval, however, declarative skills consistently improve accuracy on procedural tasks and reduce orchestration errors, while the imperative state machine's brittleness does not reliably improve task success or compliance. 

---
# Workflow-to-Skill: Skill Creation via Routing-Workflow-Semantics-Attachments Decomposition 

**Authors**: Yuyang Zhang, Xinyuan Han, Xudong Jiang, Run Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.06893)  

**Abstract**: Large language model agents increasingly rely on Skills to encode procedural knowledge, yet high-quality Skills remain costly to hand-write. This paper studies automatic Skill construction from heterogeneous interaction evidence, including demonstrations, agent trajectories, tool traces, and execution logs. We argue that trace-to-skill construction is not simple summarization tasks, because traces are fragmented, redundant, and may miss rare but safety-critical behaviors. To address this, we introduce RWSA, a workflow-oriented intermediate representation that decomposes Skills into Workflow structure, execution Semantics, and runtime Attachments, capturing task decomposition, control flow, verification, safety, rollback, and state management. Building on RWSA, we propose W2S, a framework that segments traces, induces local Skill drafts, aligns shared structures, reconciles branches, and compresses redundancy while preserving evidence and confidence annotations. Experiments on 70 Skills show that W2S improves behavioral replay consistency by 10.5% over summarization- and prompting-based baselines, highlighting the need to treat traces as executable runtime specifications rather than compressible text. 

---
# Evidence-Based Intelligent Diagnostic and Therapeutic Visualization System with Large Language Models: Multi-Turn Interaction and Multimodal Treatment Plan Generation 

**Authors**: Yunhan Wang, Yuda Wang, Zhiying Tu, Mingqiang Song, Li Song, Kun Li, Dianhui Chu, Bolin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.06869)  

**Abstract**: Aim: Existing AI-assisted traditional Chinese medicine diagnostic tools suffer from opaque reasoning processes, passive interaction, and limited treatment plan presentation. This
study proposes a knowledge-enhanced visual diagnostic system to improve the transparency and interpretability of syndrome differentiation and treatment.
Methods: The system is built upon a Neo4j knowledge graph comprising 241 syndromes, 1,263 symptoms, and 2,485 relations. It incorporates a four-stage symptom matching pipeline
(exact, semantic, fuzzy, and large language model verification), an information gain-driven proactive questioning strategy optimized with genetic algorithms, and a multimodal
treatment presentation integrating artificial intelligence-generated illustrations, three-dimensional meridian-acupoint models, and evidence-based literature.
Results: Knowledge graph constraints reduced non-standard outputs by 32%. Case studies validated the effectiveness of the interactive workflow across patient self-assessment,
clinician-assisted diagnosis, and traditional Chinese medicine education. Automated paired-comparison evaluation across 30 cases further demonstrated significant improvements in
diagnostic trust (Cohen's d = 1.82, p < 0.001), reduced cognitive load (improvements in four of five dimensions), and higher credibility of evidence-based references (4.21 vs.
2.95).
Conclusions: The proposed system enhances the transparency of traditional Chinese medicine diagnostic reasoning and the interpretability of treatment plans through knowledge
graph-driven visualization and multimodal interaction, offering a practical solution for trustworthy artificial intelligence-assisted traditional Chinese medicine applications. 

---
# AdMem: Advanced Memory for Task-solving Agents 

**Authors**: Runzhe Wang, Huilin Lu, Shengjie Liu, Li Dong, Jason Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.06787)  

**Abstract**: Large Language Models (LLMs) show promise as tool-using agents but remain limited in long-horizon tasks that require remembering, organizing, and reusing knowledge. Prior memory approaches aim to resolve the situation, but mainly focus on storing factual information. Recent work on procedural memory improves task reuse, yet often reduces to replaying past successes without addressing failure cases or online scalability. We introduce a unified and automatic memory framework that integrates semantic, episodic, and procedural memory in a bi-level design combining short-term and long-term stores. A multi-agent architecture with actor, memory, and critic agents enables automatic memory generation, reward annotation, and adaptive retrieval. Long-term memory is managed through reward-based evaluation, merging, and pruning, ensuring scalability and continual improvement. Experiments across various environments show that our approach improves robustness and success on long multi-turn tasks compared to existing baselines. This work highlights the importance of comprehensive, adaptive memory for advancing LLM-based agents. 

---
# OpenSkill: Open-World Self-Evolution for LLM Agents 

**Authors**: Zhiling Yan, Dingjie Song, Hanrong Zhang, Wei Liang, Yuxuan Zhang, Yutong Dai, Lifang He, Philip S. Yu, Ran Xu, Xiang Li, Lichao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2606.06741)  

**Abstract**: Self-evolving agents requires adaptation after deployment, but existing approaches assume a usable learning loop, such as curated skills, successful trajectories, or verifier signals. Real open-world deployments may provide none of these, offering only a task prompt. In this work, we study open-world self-evolution, where an agent must build both its skills and its own verification signals from scratch, using open-world resources but no target-task supervision. We propose OpenSkill, a framework that bootstraps this loop: it acquires grounded knowledge and verification anchors from documentation, repositories, and the web, synthesizes them into transferable skills, and refines those skills against self-built virtual tasks grounded in the anchors rather than in target answers. The open world thus supplies both the knowledge to be learned and a supervision-independent practice environment, with target-task supervision reserved for final evaluation. Across three benchmarks and two target agents, OpenSkill attains the best automated pass rate while satisfying the no-supervision constraint. Analysis shows its skills transfer across models without model-specific adaptation, and its self-built verifier aligns with ground-truth outcomes despite never accessing them. 

---
# A Geometric Account of Activation Steering through Angle-Norm Decomposition 

**Authors**: Georgii Aparin, Tatiana Gaintseva  

**Link**: [PDF](https://arxiv.org/pdf/2606.06735)  

**Abstract**: Linear activation steering has gained popularity as a simple and empirically effective way to control language model behavior. More recently, spherical steering paradigms have been proposed to address limitations of additive interventions, often motivated by the assumption that hidden-state norm does not carry concept-relevant information. In this work, we revisit this assumption through a controlled empirical study designed to disentangle the roles of angular and radial components. We show that steering methods differ mainly in how they couple two geometric effects: changing a token's angular alignment with a concept direction and changing its hidden-state norm. Across seven language models, we find that concepts are represented primarily in angular structure, supporting the motivation for spherical methods, but that norm remains important for the stability and downstream effects of steering. Our results explain why interventions with similar concept-level effects can behave differently, and suggest that activation steering should be parameterized by interpretable angular and radial components of the intervention, rather than by a single additive coefficient that entangles these two effects. 

---
# AEGIS: A Backup Reflex for Physical AI 

**Authors**: Josef Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.06660)  

**Abstract**: Long-horizon robot manipulation tends to fail gradually: one bad step degrades the state, and the policy spirals into a basin from which it cannot recover. The failure is often visible before it happens. We introduce AEGIS (Activation-probe Early-warning, Gated Inference Switching), a selective escalation method that uses a lightweight probe on a weak policy's frozen activations to detect high-risk steps while there is still time to act. When the probe flags a step, control switches to a stronger separate policy, but only for the steps that need it. On LIBERO-Spatial, AEGIS recovers 10.1% of the trajectories the weak policy alone loses, versus 4.6% for budget-matched blind escalation and 5.1% for a random-trigger placebo. These gains are significant under one-sided exact paired McNemar tests with Holm-Bonferroni adjustment over three pre-registered contrasts: +5.4pp over blind escalation, p=8.5e-6; +5.0pp over random triggering, p=1.0e-4; paired-trajectory bootstrap CIs exclude zero. AEGIS activates the stronger policy on only 38% of steps, so the lever is timing rather than compute. The probe clears its precondition with an early-window AUROC of 0.764, 95% CI [0.70, 0.84], read from the weak-policy path over the first 30% of trajectory steps before any handoff. We pre-register the full analysis plan, including a conditional recovered-task-rate estimand and explicit kill criteria, and confirm the result on 700 common-random-number episodes per arm, with nA-fail=646. 

---
# A Study of Parallel Continuous Local Search 

**Authors**: Cody J Christopher, Charles Gretton  

**Link**: [PDF](https://arxiv.org/pdf/2606.06656)  

**Abstract**: We study parallel Continuous Local Search (CLS) as a solution approach for Boolean satisfiability problems with symmetric pseudo-Boolean (PB) constraints. Here, the $n$-variable PB-satisfiability problem is relaxed to a continuous optimisation problem with a differentiable objective function on an $n$-dimensional hypercube. For satisfiable instances, the global minimisers of this optimisation problem correspond to satisfying assignments of the SAT problem at hand. We present several novel findings via empirical experiments: (i) redundant constraints can inhibit rather than accelerate convergence; (ii) CLS shows promise as a sub-solver in hybridised settings, quickly completing partial assignments; and (iii) local search rapidly converges to a stable distribution of solution quality (i.e., degree of satisfaction), due to saddle-dense objectives where additional solver steps yield diminishing returns. Our findings inform practical uses of CLS for SAT on modern accelerator hardware. 

---
# Accelerated Fourier SAT (AFSAT): Fully Realising a GPU-based Symmetric Pseudo-Boolean SAT Solver 

**Authors**: Cody J Christopher, Charles Gretton  

**Link**: [PDF](https://arxiv.org/pdf/2606.06641)  

**Abstract**: We present Accelerated Fourier SAT (AFSAT), a GPU-accelerated solver for pseudo-Boolean satisfiability based on continuous local search (CLS). AFSAT realises the proof-of-concept approach, FastFourierSAT, into a fully-engineered solver supporting any heterogeneous mixture of symmetric constraint types and lengths within a single problem instance. Using the JAX compiler, AFSAT leverages pure function composition, automatic vectorisation, automatic differentiation, and just-in-time (JIT) compilation to perform massively parallel CLS across batches of candidate assignments. We demonstrate substantially improved numerical stability, runtime performance, and memory efficiency over the proof-of-concept. We achieve this by way of identifying and addressing various limitations that arise from memory latency and floating-point representation, as well as leveraging automatic parallelisation and compact representations. The inherent representational and stability limitations of floating point are partially addressed by a tailored discrete Fourier transform implementation. We achieve near-linear throughput when scaling to multiple accelerators via JAX array sharding. 

---
# Position: Don't Just "Fix it in Post": A Science of AI Must Study Training Dynamics 

**Authors**: Stella Biderman, Mohammad Aflah Khan, Niloofar Mireshghallah, Catherine Arnett, Fazl Barez, Naomi Saphra  

**Link**: [PDF](https://arxiv.org/pdf/2606.06533)  

**Abstract**: What would it mean to have a scientific understanding of AI? Models are not static objects: they are snapshots of time-evolving processes shaped by data, objectives, architectures, and optimization dynamics. Yet much of AI research treats models as fixed artifacts, analyzing behaviors after training rather than asking why they emerge. This position paper argues that a science of AI must move beyond post-hoc fixes and study the training dynamics that produce model behavior. Such a science should support progressively stronger forms of understanding: predicting outcomes from early training signals, intervening when trajectories go wrong, and ultimately designing training procedures that more reliably produce desired properties. Scaling laws have made prediction routine for loss; the challenge is extending this success to capabilities, biases, robustness, and safety-relevant behaviors. We articulate requirements for such theories grounded in the history and philosophy of science, examine progress in mechanistic interpretability, fairness, memorization, and simplicity bias, and identify concrete open problems. 

---
# CARVE-Q: Quantum-Proposed, Classically Certified Interactive Driving Repair 

**Authors**: Yifan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.06531)  

**Abstract**: The critical question after a correct driving veto is not only whether a maneuver is unsafe, but whether the blocked interaction admits a lawful, auditable, and responsibility-bounded repair. Prediction and game-theoretic planners can suggest plausible cooperation, yet they do not return a proof that the repair respects hard rules, right-of-way, cost allocation, and ego fallback. We introduce CARVE, Certified Affordable Repair of Vetoed maneuvers via Envelopes, a certificate architecture for prediction-free interactive repair. Given a vetoed maneuver, CARVE constructs a finite repair lattice and emits a structured certificate recording the binding rule, selected joint repair, right-of-way-scaled cooperation envelope, responsibility-weighted cost split, and ego-only fallback. This certificate view reveals the algorithmic bottleneck: multi-owner repair induces a product lattice $M = \prod_j |\mathcal{A}_j|$. We therefore introduce CARVE-Q, a verifier-shielded quantum-AI search layer that applies quantum minimum finding only to this black-box lattice while leaving all safety authority classical. In the conservative verifier-oracle model, exact classical minimum finding requires $\Theta(M)$ queries in the worst case, whereas Durr-Hoyer/Grover minimum finding uses $O(\sqrt{M})$ oracle queries with high probability. We prove verifier-shielded certificate soundness, priority non-elicitation, black-box query separation, and finite-precision reversible-oracle constructibility. We then demonstrate state-vector minimum finding on CARVE repair oracles up to 65,536 assignments and validate certificate preservation on Lanelet2-grounded INTERACTION replay with 100% right-of-way respect, 100% blame consistency, and zero priority false positives. The result is a trust-bounded quantum-AI pattern for certified autonomy: quantum proposes; CARVE certifies. 

---
# Attack Selection in Agentic AI Control Evaluations Meaningfully Decreases Safety 

**Authors**: Catherine Ge-Wang, Tyler Crosse, Benjamin Hadad IV, Joachim Schaeffer, Ram Potham, Tyler Tracy  

**Link**: [PDF](https://arxiv.org/pdf/2606.06529)  

**Abstract**: An attacker that strategically chooses when to attack is much harder to catch than one that attacks indiscriminately. AI control is a safety framework for deploying capable but untrusted AI agents under the oversight of a weaker, trusted monitor and a limited human audit budget. Control evaluations stress-test these protocols by pitting a red-team attack policy against the blue-team monitor, but current evaluations typically assume attackers that do not strategically select when to attack. We study this capability, attack selection, in agentic settings by decomposing attack decisions into a start policy, which decides when an attacker should attack, and a stop policy, which decides when an attacker should abort an ongoing attack. Across two agentic settings, BashArena and LinuxArena, both policies substantially lower measured empirical safety without changing the underlying attack capability. At a 1% audit budget, our start policy reduces safety by 20pp on both BashArena and LinuxArena, and our stop policy reduces safety by 20pp on BashArena and 28pp on LinuxArena. These reductions should be interpreted as upper bounds on the effect of attack selection. Existing control evaluations may therefore yield overly optimistic safety estimates against selective attackers. We recommend that future evaluations, system cards, and safety cases elicit attack selection to produce more realistic safety estimates. 

---
# CrowdMath: A Dataset of Crowdsourced Mathematical Research Discussions 

**Authors**: Sherin Muckatira, Jesse Geneson, Slava Gerovitch, Pavel Etingof, Mikhail Gronas, Anna Rumshisky  

**Link**: [PDF](https://arxiv.org/pdf/2606.06526)  

**Abstract**: Large language models have made substantial progress on mathematical reasoning, but existing benchmarks typically evaluate well-specified problems with final answers, step-by-step solutions, or complete proofs. They do not capture collaborative open-problem solving: a setting in which participants propose partial arguments, identify gaps or errors in prior steps, repair flawed reasoning, and gradually synthesize incremental contributions into a proof. We introduce CrowdMath, a dataset of 164 expert-annotated progress chains from the MIT PRIMES--Art of Problem Solving (AoPS) CrowdMath program (2016-2025), a collaborative research initiative whose discussions have led to peer-reviewed publications. Each chain traces a multi-participant forum discussion from an open-problem statement to a completed proof. Posts are labeled by their functional roles in the evolving solution process, including partial progress, proof completion, erroneous reasoning, and error identification. We define evaluation tasks and benchmark six frontier models. Models achieve 83-88% accuracy on next-post prediction, suggesting that they can follow the local flow of mathematical discussion. However, they struggle to identify the functional significance of individual contributions with the best model achieving only 0.42 macro-F1 on post-role classification. CrowdMath exposes a gap between solving well-specified mathematical problems and understanding collaborative mathematical progress as it unfolds. 

---
# Lean4Agent: Formal Modeling and Verification for Agent Workflow and Trajectory 

**Authors**: Ruida Wang, Jerry Huang, Pengcheng Wang, Xuanqing Liu, Luyang Kong, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.06523)  

**Abstract**: Equipping Large Language Models (LLMs) to execute reliable multi-step workflows has become a central challenge in artificial intelligence. Despite recent advances in LLMs' agentic capabilities, most agent systems still lack formal methods for specifying, verifying, and debugging their workflow and execution trajectories. This challenge mirrors a long-standing problem in mathematics, where the ambiguity of natural languages (NLs) motivates the development of formal languages (FLs). Inspired by this paradigm, we propose **Lean4Agent**, to the best of our knowledge, the first framework that uses Lean4, a dependent-type FL to model and verify agent behavior. **Lean4Agent** launches **FormalAgentLib**, an extensible Lean4 library for formally modeling and verifying agent workflows' semantic consistency under explicit assumptions, and enabling localization of execution-time failures revealed by trajectories. Building on **FormalAgentLib**, we further develop **LeanEvolve**, which applies results in **FormalAgentLib** to revise workflows to enhance its capability. Extensive experiments on a hard problem subset of SWE-Bench-Verified and a subset of ELAIP-Bench across 5 leading LLMs indicate that the verification-passing workflows outperform the failing ones by an average of **11.94%**, and **LeanEvolve** further improves SWE performance by **7.47%** on average. Furthermore, **Lean4Agent** establishes a foundation for a new field of using expressive dependent-type FL to formally model and verify agent behavior. 

---
# SafeGene: Reusable Adapters for Transferable Safety Alignment 

**Authors**: Yanghan Wang, Zhiqiang Kou, Fu Feng, Jing Wang, Xin Geng  

**Link**: [PDF](https://arxiv.org/pdf/2606.06519)  

**Abstract**: Open-weight LLMs are increasingly fine-tuned into customized assistants, but downstream fine-tuning can weaken safety alignment and make models more vulnerable to malicious prompts, even when the training data is not intentionally harmful. This creates a recurring safety recovery problem as target models are repeatedly updated with new task data or user interactions. We propose SafeGene, a reusable safety-adapter module designed for cross-task reuse within each architecture-compatible model family. Rather than treating safety recovery as a model-specific repair step, SafeGene treats safety capability as an independent, reusable adapter representation decoupled from task-specific updates. This representation is obtained from aligned--degraded model discrepancies, refined into task-transferable safety vectors through data-aware layer selection, and expressed in each downstream task-adapted model via few-shot layer-wise coefficient recalibration. Experiments across multiple model families, downstream tasks, and safety judges show that SafeGene-enhanced models reduce harmful response rates while maintaining downstream performance, outperforming representative safe adaptation methods in safety--utility trade-off. 

---
# DiBS: Diffusion-Informed Branch Selection 

**Authors**: Bo Liu, Yuan Xie, Yuan Gao, Xiaolong Luo, Peng Ye, Tao Chen, Fujun Han  

**Link**: [PDF](https://arxiv.org/pdf/2606.06518)  

**Abstract**: Sudoku is a representative constraint satisfaction problem that requires global structural reasoning under strict discrete constraints. The existing works of solving Sudoku mainly focus on two dominant approaches, i.e., traditional heuristic and deep learning solver. However, they suffer from two complementary limitations: learning-based solvers lack hard correctness guarantees, while complete symbolic solvers are still prone to long-tail search. To address these shortcomings, we propose a novel diffusion model-guided approach, termed as DiBS, for the branch selection search process. Specifically, DiBS keeps the symbolic solver complete and uses the diffusion model as a branch-ordering guide. The core method is ranking candidate values under the current partial assignment and lightweight consistency signal. Furthermore, we provide an in-depth theoretical proof to reveal how it works and why it works. Experiments on the challenging Royle 17-clue Sudoku benchmark show that our DiBS substantially reduces search cost relative to strong heuristic baselines, especially in nodes, backtracks, and long-tail percentiles. Besides, these results confirm that learned global guidance is effective on hard instances where branch-order mistakes are most expensive. All codes are available at this https URL. 

---
# Detecting and Mitigating Bias by Treating Fairness as a Symmetry Operation 

**Authors**: Nishit Singh  

**Link**: [PDF](https://arxiv.org/pdf/2606.06514)  

**Abstract**: Machine learning systems deployed in high stakes socioeconomic settings routinely display bias. We formalize bias as a symmetry breaking operation: a classifier is fair if its outputs remain invariant under the counterfactual operation of switching a sensitive attribute, with merit features held fixed. We implement loss based regularization as a symmetry restoring mechanism and evaluate the framework on four synthetic datasets with varying levels of noise, correlation, and bias. The framework achieves upwards of 90\% violation reduction, with accuracy costs around 5\%. This framework does not require causal graph knowledge, is computationally lightweight, and generalizes to any sensitive attribute definable as a bit-flip, making it suitable for contexts where local sources of discrimination remain absent from mainstream benchmarks. 

---
# How reliable are LLMs when it comes to playing dice? 

**Authors**: Luca Avena, Gianmarco Bet, Bernardo Busoni  

**Link**: [PDF](https://arxiv.org/pdf/2606.07515)  

**Abstract**: We investigate the probabilistic reasoning capabilities of large language models through a controlled benchmarking study on discrete probability problems. We constructed two datasets, respectively a set of standard exercises and a set of counterintuitive exercises, designed to trigger heuristic reasoning, and evaluated 8 state-of-the-art models, each tested with and without Chain-of-Thought prompting. Models achieve an average accuracy of 0.96 on standard problems but only 0.59 on counterintuitive ones. We further provide empirical evidence of token bias: performance drops by over 20% when canonical formulations are replaced by disguised variants. Embedding misleading suggestions in the prompt reduces performance by up to 34%, with no model proving immune. Taken together, the reported findings suggest that current LLMs are not yet genuine probabilistic reasoners, despite their success in advanced mathematical problems. 

---
# MemDreamer: Decoupling Perception and Reasoning for Long Video Understanding via Hierarchical Graph Memory and Agentic Retrieval Mechanism 

**Authors**: Cong Chen, Guo Gan, Kaixiang Ji, ChaoYang Zhang, Zhen Yang, Guangming Yao, Hao Chen, Jingdong Chen, Yi Yuan, Chunhua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2606.07512)  

**Abstract**: Current Vision-Language Models struggle with hours-long videos because processing full-length visual sequences induces prohibitive token explosion and attention dilution. To overcome this, we introduce MemDreamer to decouple perception and reasoning, shifting long-video understanding into an agentic exploration process. As a plug-and-play framework, it incrementally streams videos to construct a Hierarchical Graph Memory, a top-down three-tier architecture for semantic abstraction, anchored by a foundational graph capturing spatiotemporal and causal relations. During inference, the reasoning model employs agentic tool-augmented retrieval, navigating hierarchies, searching nodes, and traversing logical edges via an Observation-Reason-Action loop. Experiments show MemDreamer achieves SOTA results across four mainstream benchmarks, narrowing the gap with human experts to only 3.7 points. It constrains the reasoning context window to merely 2% of full-context ingestion while delivering a 12.5 point absolute accuracy gain. Furthermore, statistical analysis uncovers a strong positive linear correlation between an VLM's performance on logic reasoning and long-video understanding benchmarks, establishing agentic capability scaling as a new paradigm for multimodal comprehension. 

---
# Sparse Subspace-to-Expert Sharing for Task-Agnostic Continual Learning 

**Authors**: Fatema Siddika, Md Anwar Hossen, Tanwi Mallick, Ali Jannesari  

**Link**: [PDF](https://arxiv.org/pdf/2606.07500)  

**Abstract**: Continual learning in Large Language Models (LLMs) is hindered by the plasticity-stability dilemma, where acquiring new capabilities often leads to catastrophic forgetting of previous knowledge. Existing methods typically treat parameters uniformly, failing to distinguish between specific task knowledge and shared capabilities. We introduce Mixture of Sparse Experts for Task Agnostic Continual Learning (SETA), a framework that resolves the plasticity-stability conflict through adaptive sparse subspace decomposition into task-specific expert modules. Unlike standard updates, where tasks compete for the same parameters, SETA separates knowledge into unique experts, designed to isolate task-specific patterns, and shared experts, responsible for capturing common features. This structure is maintained through adaptive elastic anchoring and a routing-aware regularization that jointly protect shared knowledge at both the weight and routing levels and enable a unified gating network to automatically retrieve the correct expert combination during inference. Extensive experiments across diverse domain-specific benchmarks demonstrate that SETA achieves competitive or superior overall performance relative to state-of-the-art continual learning baselines, with particularly strong retention of early-task knowledge and improved backward transfer on LLaMA-2 7B and Qwen3-4B. 

---
# Twelve quick tips for designing AI-driven HPC workflows 

**Authors**: Jamie J. Alnasir  

**Link**: [PDF](https://arxiv.org/pdf/2606.07491)  

**Abstract**: High-performance computing (HPC) clusters remain the backbone of large-scale scientific computation, traditionally executing deterministic, linear pipelines optimised for predictable performance. However, the pervasive integration of artificial intelligence (AI) and foundation models into scientific research has introduced a fundamentally new computational paradigm. AI-driven workflows are characteristically iterative, data-driven, and probabilistic, introducing unique challenges regarding data gravity, heterogeneous resource management, and complex workflow orchestration.
This guide provides twelve practical tips designed to help researchers design efficient, scalable, and reproducible AI-driven HPC workflows. By addressing critical system-level bottlenecks - such as containerisation for environment portability, strategic deployment of job arrays, explicit feedback loop mechanics, and I/O optimisation for small files - this article offers a framework for transitioning from rigid execution pipelines to adaptive, intelligent computational environments. While these architectural principles are broadly applicable across distributed environments, they are particularly tailored to the resource-intensive throughput demands of modern computational biology. 

---
# Supervision versus Demonstration-Based In-Context Learning for Multiword Expression Classification 

**Authors**: Sercan Karakaş, Yusuf Şimşek  

**Link**: [PDF](https://arxiv.org/pdf/2606.07479)  

**Abstract**: Turkish idiomatic light verb constructions (LVCs) are challenging for multiword expression processing because they often share the same surface form as fully literal verb-object combinations while functioning as a single, partially idiomatic predicate. We frame Turkish LVC detection as a binary classification task (literal meaning vs. idiomatic meaning) and evaluate on a manually created controlled set (N=147) with matched negatives: out-of-domain random sentences and in-domain literal controls (NLVC), alongside LVC positives. We compare a supervised Turkish encoder baseline (BERTurk with a classifier head) to three instruction-tuned LLMs from different families under zero-shot, one-shot, and few-shot prompting, and analyze how demonstrations shift error profiles. In zero-shot, LLMs perform well on negatives but show very low LVC recall. One-shot prompting sharply improves LVC detection but can induce strong, model-specific biases, leading models to overpredict or underpredict LVCs. A richer few-shot prompt improves calibration and yields robust overall performance for GPT-OSS-20B and Qwen 2.5-14B. Overall, the results highlight substantial prompt sensitivity in Turkish metalinguistic classification: the supervised baseline remains competitive, while prompted LLMs can match or exceed it on LVCs with carefully constructed demonstrations. 

---
# Graph Neural Network leveraging Higher-order Class Label Connectivity for Heterophilous Graphs 

**Authors**: Takuto Takahashi, Itsuki Nakayama, Takahiro Mitani, Ryosuke Kikuchi, Yuya Sasaki, Makoto Onizuka  

**Link**: [PDF](https://arxiv.org/pdf/2606.07475)  

**Abstract**: Node classification in graph neural networks (GNNs) has been widely applied in various fields of graph analysis. GNNs achieve high-accuracy node classification in homophilous graphs, where nodes with the same class label tend to be connected. However, their performance remains limited in heterophilous graphs, where nodes with different class labels are more likely to be connected. In particular, current GNNs derived from graph convolutional networks cannot capture higher-order class label connectivity, which is frequently observed in real-world heterophilous graphs. To address this issue, we propose a novel classifier, Label Context Classifier (LCC), designed to capture higher-order class label connectivity in directed graphs. LCC estimates the class label of a target node by leveraging label context embeddings that are generated through four distinct types of walks. In addition, our approach allows the integration of LCC and any GNN by adaptively learning their importance. Experimental results demonstrate that GNNs integrated with LCC outperform SOTA methods and the label context embeddings improve the node classification performance in heterophilous directed graphs. 

---
# Whisper Hallucination Detection and Mitigation via Hidden Representation Steering and Sparse AutoEncoders 

**Authors**: Georgii Aparin, Vadim Popov, Tasnima Sadekova, Assel Yermekova  

**Link**: [PDF](https://arxiv.org/pdf/2606.07473)  

**Abstract**: Whisper, a widely adopted ASR model, is known to suffer from hallucinations - coherent transcriptions generated for non-speech audio entirely disconnected from the input. We investigate whether hallucinations can be detected and mitigated through Whisper's internal representations. We extract audio encoder activations and evaluate two representation spaces: raw Whisper activations and Sparse AutoEncoder (SAE) latents. We show that both spaces encode linearly separable hallucination-related information, with discriminative power concentrated in a sparse feature subset and increasing toward deeper encoder layers. We propose two steering strategies: activation-space steering and SAE latent-space steering. SAE-based steering reduces hallucination rate from 72.63% to 14.11% for Whisper small and from 86.88% to 27.33% for Whisper large-v3 on the full non-speech test set, with small WER degradation on speech data, approaching the performance of fine-tuning-based methods. 

---
# Planning-aligned Token Compression for Long-Context Autonomous Driving 

**Authors**: Zhixuan Liang, Yuxiao Chen, Yurong You, Peter Karkus, Wenhao Ding, Boyi Li, Alexander Popov, Yan Wang, Maximilian Igl, Yiming Li, Danfei Xu, Nikolai Smolyanskiy, Boris Ivanovic, Ping Luo, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2606.07464)  

**Abstract**: Monolithic vision-action models represent an emerging paradigm in autonomous driving. However, this architecture produces token sequences that quickly exceed real-time computational budgets when encoding extended temporal context for complex interactions. While approaches like linear transformers and external memory try to make the context lightweight, token compression is most compatible with the architecture as it requires no backbone modifications. Yet existing compression adopts rule-based heuristics like temporal decay, decoupled from planning, risking loss of decision-critical information. We propose COMPACT-VA, a planning-aligned working memory framework built on conditional VQ-VAE, compressing extended context into bounded representations. Compression is conditioned on both historical trajectory and a learned planning intent that the posterior encoder distills from future trajectories during training, while the prior encoder learns to predict it from compressed observations. The compressed memory, concatenated with the predicted latent, feeds the policy for end-to-end optimization, planning with retained decision-critical information. We evaluate on high-signal dynamic scenarios where historical context is most critical for behavior correctness (e.g., stop, yield, or proceed), and accordingly design behavioral metrics. Under comparable token budgets, we achieve $>$6% improvement (68.3%) on success rates with consistent gains across metrics. Ablations validate planning-aligned coupling effectiveness. Closed-loop evaluation confirms that COMPACT-VA maintained general driving performance with 3.3* speedup and 2.7* memory reduction over uncompressed processing. 

---
# PaperFlow: Profiling, Recommending, and Adapting Across Daily Paper Streams 

**Authors**: Fuqiang Wang, Song Tan, Zheng Guo, Jiaohao Fu, Xinglong Xu, Bihui Yu, Jie Dong, Zheng Sun, Siyuan Li, Jingxuan Wei, Cheng Tan  

**Link**: [PDF](https://arxiv.org/pdf/2606.07454)  

**Abstract**: Scientific paper recommendation is typically evaluated as static ranking over a fixed candidate set, yet real scientific reading unfolds as a daily, longitudinal process in which interests shift and feedback accumulates. We introduce PaperFlow, a framework that organizes it into three coupled stages: Profiling, which constructs and maintains a structured, inspectable scholarly profile from heterogeneous cold-start evidence; Recommending, which ranks each date-specific paper stream through multi-signal aggregation under a fixed display budget; and Adapting, which updates user state from semantically distinct feedback signals and models interest drift across days. We further define a longitudinal user-day benchmark that fixes users, dates, candidate pools, visible inputs, and hidden simulated relevance labels under a shared temporal information boundary. The benchmark contains 24 simulated research users, 50 daily paper streams, 1,200 user-day episodes, 20,727 unique papers, and 497,448 episode-paper records. We additionally specify a blind human-evaluation protocol to validate alignment between automatic metrics and expert judgments. Experiments against five scientific recommendation baselines show that PaperFlow achieves the strongest oracle-based ranking, the highest behavioral alignment with simulated reading selections, and the best blind human-evaluation score. 

---
# TEVI: Text-Conditioned Editing of Visual Representations via Sparse Autoencoders for Improved Vision-Language Alignment 

**Authors**: Sweta Mahajan, Sukrut Rao, Jiahao Xie, Alexander Koller, Bernt Schiele  

**Link**: [PDF](https://arxiv.org/pdf/2606.07451)  

**Abstract**: Vision-language models such as CLIP are highly useful for diverse tasks due to their shared image-text embedding space. Despite this, the image and text embeddings are often poorly aligned, affecting downstream performance. Recent work has shown that this can be attributed to an information imbalance: images contain more information than their captions describe. In this work, we propose TEVI, a framework that uses captions as a signal for what to retain from image embeddings. Specifically, we use sparse autoencoders to disentangle image embeddings and train a masking module to selectively reconstruct the embedding based on a given caption. In a controlled setup with synthetic captions, we show that TEVI is effective at preserving caption-described attributes while discarding others. By applying TEVI to CLIP models trained on natural images, we further achieve improved retrieval performance across coarse-grained short-caption (MS COCO, Flickr) and fine-grained long-caption (IIW, DOCCI) benchmarks, with stronger gains on richer captions, and improved robustness on the RoCOCO benchmark. 

---
# Re-imagining ISO 26262 in the Age of Autonomous Vehicles: Enhancing Controllability through Transferability and Predictability 

**Authors**: Chaitanya Shinde, Hadi Hajieghrary, Paul Schmitt, Adam Shoemaker, Bodo Seifert, Steve Kenner  

**Link**: [PDF](https://arxiv.org/pdf/2606.07437)  

**Abstract**: The ISO 26262 standard defines functional safety for road vehicles through risk assessments based on Severity, Exposure, and Controllability, grounded in a human-driven vehicle paradigm. In the context of autonomous vehicles (AVs), the absence of a human driver necessitates revisiting these principles. This paper decomposes the Controllability placeholder into two auditable evidence dimensions of ISO 26262 by introducing two measurable sub-concepts: Transferability and Predictability. Transferability extends Controllability to capture AV systems' ability to hand off control to dedicated fallback safety mechanisms, while Predictability captures how easily external agents can anticipate AV behavior. Predictability is formally defined from human-robot interaction-inspired principles, and a mathematical framework is provided to quantify it. A designed-versus-achievable gap is introduced to distinguish architectural fallback claims from scene-conditioned achievable fallback capability. The proposed metrics align with ISO 26262 and ISO/PAS 21448 (SOTIF), rendering fallback and interaction claims falsifiable and traceable across ODD slices. These dimensions complement rather than replace existing standards, and the enhancements preserve the structure of ISO 26262 while extending its applicability to driverless automated systems operating at SAE Levels 4 and 5. 

---
# Watch, Remember, Reason: Human-View Video Understanding with MLLMs 

**Authors**: Jiahao Meng, Yue Tan, Qi Xu, Kuan Gao, Weisong Liu, Yanwei Li, Jason Li, Lingdong Kong, Haochen Wang, Qianyu Zhou, Jiangning Zhang, Guangliang Cheng, Yunhai Tong, Lu Qi, Minghsuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.07433)  

**Abstract**: Video understanding is being rapidly transformed by multimodal large language models (MLLMs), as research moves from short clips to long, multimodal, and knowledge-intensive video scenarios. These scenarios require models to handle sparse evidence, long-range dependencies, multimodal alignment, and reliable inference under limited computational budgets. This work presents a human-view perspective on LLM-based video understanding, organized around three functional abilities: watching, remembering, and reasoning. Rather than treating video tasks as isolated benchmarks, this view provides a unified structure for analyzing how video MLLMs acquire evidence, preserve context, and produce grounded outputs. We introduce a formulation that characterizes video understanding systems by their perceptual representations, memory states, reasoning traces, and final predictions. Based on this formulation, we identify challenges in spatio-temporal perception, efficient long-video processing, memory modeling, streaming understanding, and faithful reasoning. Representative methods are organized by their roles in video MLLM systems. Watching covers fine-grained, comprehensive, audio-visual, and efficient perception. Remembering includes offline and streaming memory, while reasoning covers text-only reasoning and thinking with videos. We further examine application domains such as egocentric, sports, instructional, medical, and narrative videos, and cover training datasets and evaluation benchmarks across task types, supervision formats, modalities, and capability dimensions. Finally, we outline open problems and future directions for scalable, memory-aware, and evidence-grounded video intelligence. Related works will be continuously traced at this https URL. 

---
# The Masked Advantage: Uncovering Local-Language Access to Cultural Knowledge in LLMs 

**Authors**: Yang Zhang, Xiao Fei, Amr Mohamed, Sarah Almeida Carneiro, Mersin Konomi, Mingmeng Geng, Ahmed Asaad, Guokan Shang, Michalis Vazirgiannis  

**Link**: [PDF](https://arxiv.org/pdf/2606.07422)  

**Abstract**: Large language models are increasingly used to answer culturally grounded questions across languages, yet it remains unclear whether local cultural knowledge is better accessed through English or the local language. Existing evaluations face two key limitations: many rely on parallel template-based questions that may not reflect how cultural knowledge naturally appears, and raw accuracy conflates general language proficiency with language-conditioned knowledge access. We address these issues with a controlled framework built on real-world cultural questions collected from regional benchmarks and local sources. By crossing question type (culture-agnostic vs. culture-specific) with query language (English vs. local language), and estimating ability with a shared 1PL item response theory model, we separate proficiency from localized knowledge access. Across 13 locales and roughly 80 models, we find a consistent English advantage on culture-agnostic questions, indicating stronger English proficiency. However, after accounting for this proficiency gap, local languages show a positive knowledge-access advantage in nearly all locale-model settings. This advantage is often masked in raw accuracy but becomes more visible for frontier, regionally aligned, or language-adapted models. Our results suggest that weaker local-language performance does not necessarily imply weaker cultural knowledge; rather, local cultural knowledge may be more accessible through the local language but hidden by limited language proficiency. 

---
# Socratic-SWE: Self-Evolving Coding Agents via Trace-Derived Agent Skills 

**Authors**: Chuan Xiao, Zhengbo Jiao, Shaobo Wang, Wei Wang, Bing Zhao, Hu Wei, Linfeng Zhang, Lin Qu  

**Link**: [PDF](https://arxiv.org/pdf/2606.07412)  

**Abstract**: LLM-driven software engineering agents have become a central testbed for real-world language-model capability, yet their training remains limited by the availability of high-quality SWE tasks. Existing synthetic data methods typically create tasks through fixed mutation or bug-injection procedures, making the resulting distributions largely independent of the agent's own weaknesses and training progress. We introduce Socratic-SWE, a closed-loop self-evolution framework that reuses the agent's historical solving traces as a source of training signal. Rather than treating traces only as evidence for reward computation, Socratic-SWE distills them into structured agent skills that summarize recurring failures and effective repair patterns. These skills then guide the generation of targeted repair tasks in real repositories. Candidate tasks are checked through execution-based validation and scored with a solver-gradient alignment reward, so that the retained tasks are both verifiable and useful for improving the Solver. The updated Solver produces new traces, enabling the task curriculum to adapt over successive rounds. Across SWE-bench Verified, SWE-bench Lite, SWE-bench Pro, and Terminal-Bench 2.0, Socratic-SWE consistently improves over self-evolving baselines under the same compute budget, reaching 50.40% on SWE-bench Verified after three iterations. These results suggest that solving traces can serve as a scalable substrate for self-evolving SWE agents. 

---
# A Comprehensive Anatomy of Human and DeepSeek-R1 LLM Mathematical Reasoning 

**Authors**: Yuxiang Chen, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.07410)  

**Abstract**: The emergence of "Aha moments" in large language models, particularly DeepSeek-R1-0120, has raised the question of whether these systems genuinely reason or merely imitate the appearance of reasoning. We conduct a comprehensive empirical comparison between model and human reasoning across all 30 problems from AIME 2025, exhaustively annotating 10,247 reasoning steps into five functional categories: Analysis, Inference, Branch, Backtrace, and Reflection. We find a clear structural difference. Human solutions maintain a compact alternation between analysis and deduction, whereas DeepSeek-R1 frequently revisits intermediate results, performs shallow and often unnecessary verification, and loops through local checks without meaningful logical progress. We describe this as topological mimicry: reproducing the surface form of reasoning without its functional role. Despite this, we identify two signals of genuine reasoning. First, successful traces exhibit stable use of branching and backtracking, while failed traces either underuse or overuse exploratory actions. Second, reflection is only effective when placed within deductive inference; reflections trapped in analysis loops focus on local numerical details while missing global logical errors. These findings suggest that current long-CoT models may be rewarded more for the appearance of reasoning than for genuine deductive progress. We discuss directions for improving evaluation and training, including measuring cross-trace stability, penalising "spinning-wheel" traces, encouraging deeper logical correction, and reallocating inference-time compute toward deduction and backtracking. Overall, reasoning quality depends not simply on how much reflection occurs, but on whether reflection appears consistently and at the appropriate logical scale. 

---
# Impact of Synthetic Lesional MR Images in Automated Focal Cortical Dysplasia Detection in Low-Data Scenarios 

**Authors**: Prabhjot Kaur, Hakim Ouaalam, Sedat Kandemirli, Sanjay P. Prabhu, Simon K. Warfield  

**Link**: [PDF](https://arxiv.org/pdf/2606.07381)  

**Abstract**: Background and Purpose: Automated detection of focal cortical dysplasia (FCD) requires large volumes of voxelwise lesion-delineated MRI data, which are difficult to acquire. This study aims to generate synthetic MRI data exhibiting FCD, assess their realism, and evaluate their impact on automated FCD detection, particularly in reducing the need for manual annotations.
Methods: T1-weighted (T1w) and T2-weighted Fluid-Attenuated Inversion Recovery (FLAIR) MRI scans from 131 FCD patients and 90 healthy controls from multiple (3) sites were retrospectively studied. Synthetic MRIs were generated by conditioning a generative network on binary FCD masks. Two neuroradiologists identified real images from a random set of 14 real and 14 synthetic scans. Three nnU-Net models were trained to detect FCD using: (i) real-only (35 FCD / 35 controls), (ii) real (35 FCD / 35 controls) plus synthetic augmentation, and (iii) expanded real data (70 FCD / 70 controls).
Results: Experts showed limited ability to distinguish real from synthetic images, with classification accuracy of 60% for T1w and 70% for FLAIR (inter-rater agreement kappa = 0.86). Augmenting automated FCD detection with synthetic data increased sensitivity by 8.14% (p = 0.12) and improved model confidence at true lesion sites (0.83 +/- 0.11 to 0.89 +/- 0.12; p = 0.02). The expanded real-data model further improved sensitivity to 73.8% (p < 0.001) and confidence to 0.90 +/- 0.14 (p = 0.01).
Conclusion: Conditional generative networks can generate realistic synthetic FCD-MRIs, reducing labeled data needs by approximately 20% while maintaining equivalent sensitivity. Equivalent amounts of real data, when available, remain more effective than synthetic augmentation. 

---
# Do Coding Agents Deceive Us? Detecting and Preventing Cheating via Capped Evaluation with Randomized Tests 

**Authors**: Thanawat Lodkaew, Johannes Ackermann, Soichiro Nishimori, Nontawat Charoenphakdee, Masashi Sugiyama, Takashi Ishida  

**Link**: [PDF](https://arxiv.org/pdf/2606.07379)  

**Abstract**: A growing failure mode in agent evaluation and training is that models can achieve high evaluation scores by exploiting shortcuts instead of solving the intended task, producing deceptive performance. This makes evaluation scores unreliable as measures of true task-solving ability. We propose CapCode, a framework for constructing coding datasets with randomized tests whose best achievable non-cheating performance is deliberately capped below one. This capped-performance design gives evaluation scores a clearer interpretation: scores substantially above the cap are implausible and therefore provide evidence of cheating. To prevent cheating, we propose CapReward, a reward design based on the CapCode principle to discourage optimization beyond the cap. Experiments across multiple datasets show that CapCode detects cheating while preserving performance ranking of models, and CapReward reduces cheating behavior, yielding models that better follow the intended task specification. 

---
# Mitosis Detection in the Wild: Multi-Tumor and Context-Aware Generalization in the MIDOG 2025 Challenge 

**Authors**: Marc Aubreville, Jonas Ammeling, Sweta Banerjee, Viktoria Weiss, Taryn A. Donovan, Robert Klopfleisch, Jiaqi Lv, Shan E Ahmed Raza, Raphaël Bourgade, Thomas Walter, Yasemin Topuz, Songül Varlı, Charles-Antoine Collins-Fekete, Zhuoyan Shen, Navya Sri Kelam, Nitin Singhal, Christian Marzahl, Brian Napora, Tengyou Xu, Hongyan Gu, Mario Vento, Gennaro Percannella, Norbert Ropiak, Izabela Wasiak, Jie Xiao, Shaojun Liu, Seungho Choe, April Khademi, Vidushi Walia, Sujatha Kotte, Andrew Broad, Alex Wright, Guillaume Balezo, Esha Sadia Nasir, Mostafa Jahanifar, Yosuke Yamagishi, Shouhei Hanaoka, Mattia Sarno, Francesco Tortorella, Biwen Meng, Jingxin Liu, Sara Krauss, Daniel Hieber, Lavish Ramchandani, Dev Kumar Das, Mieko Ochi, Yuan Bae, Piotr Giedziun, Mateusz Maniewski, Vangala Govindakrishnan Saipradeep, Naveen Sivadasan, Leire Benito-Del-Valle, Adrian Galdran, Kaustubh Atey, Sameer Anand Jha, Adinath Dukre, Imran Razzak, Maxime W. Lafarge, Viktor H. Koelzer, Nils Porsche, Nikolas Stathonikos, Mitko Veta, Dominik Hirling, Zsanett Zsófia Iván, Peter Horvath, Katharina Breininger, Christof A. Bertram  

**Link**: [PDF](https://arxiv.org/pdf/2606.07368)  

**Abstract**: Automated mitosis detection is a well-established task in computational pathology. While previous benchmarks focused on scanner-induced domain shift, clinical "real-world" application requires models to be robust across the vast variance to be expected in the histological landscape. The MItosis DOmain Generalization (MIDOG) 2025 challenge was designed to evaluate algorithmic performance across unprecedented biological and contextual diversity. We curated a test dataset of 365 cases, encompassing 12 distinct human, canine and feline tumor types, digitized across multiple scanning platforms. Moving beyond hand-selected hotspots, the challenge required detection also in random tissue areas (representative of the whole slide detection situation) and challenging areas (areas rich in hard negatives). In the second track, we introduced the classification of atypical mitotic figures (AMFs). There were 18 teams submitting to the detection track, with F1 scores ranging up to 0.740. In the AMF detection track, we had 21 submissions with balanced accuracy values up to 0.908. Our analysis reveals that while most models perform reliably in traditional hotspots, significant performance degradation occurs in challenging ROIs, where false positive rates tripled. Furthermore, performance varied significantly across the 12 tumor types, highlighting "blind spots" in current state-of-the-art architectures when encountering rare or highly pleomorphic malignancies. Moreover, we evaluated the effectiveness of ensembling and found a mean increases of 1.5 and 1.3 percentage points in F1 score and balanced accuracy, respectively. In contrast, TTA showed no relevant improvement. MIDOG 2025 demonstrates that "in the wild" mitosis detection remains a significant hurdle. The transition from hotspot-only evaluation to a multi-contextual framework provides a more realistic proxy for clinical reliability. 

---
# A robust PPG foundation model using multimodal physiological supervision 

**Authors**: Eloy Geenjaar, Vince Calhoun, Scott Daly, Gouthaman KV, Lie Lu, Trisha Mittal, Daniel P. Darcy  

**Link**: [PDF](https://arxiv.org/pdf/2606.07365)  

**Abstract**: Photoplethysmography (PPG), a non-invasive measure of changes in blood volume, is widely used in both wearable devices and clinical settings. Recent PPG foundation models either use open-source ICU datasets with pretraining paradigms that require curated data and thus complicate generalization to field-like data, or use closed-source field-like PPG data. In contrast, we propose a PPG foundation model that does not require high-quality or field-like pretraining data, and instead leverages accompanying electrocardiogram and respiratory signals in ICU datasets to select contrastive samples during pretraining. Our approach allows the model to retain and learn from noisy PPG segments, improving robustness at inference. Our model, pretrained on 3x fewer subjects than existing state-of-the-art approaches, achieves performance improvements on 14 out of 15 diverse downstream tasks, including field-like daily activity and heart rate prediction. Our results demonstrate that multimodal supervision can integrate complementary physiological information to improve the robustness of PPG foundation models and enhance their generalization to consumer-grade data. 

---
# SleepExplain: Explainable Non-Rapid Eye Movement and Rapid Eye Movement Sleep Stage Classification from EEG Signal 

**Authors**: Rafsan Jany, Md. Hamjajul Ashmafee, Iqram Hussain, Md Azam Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2606.07351)  

**Abstract**: Classification of sleep stages is one of the most important diagnostic approaches for a variety of sleep-related disorders. Electroencephalography (EEG) is regarded as a powerful tool for examining the association between neurological effects and sleep phases since it correctly identifies sleep-related neurological alterations. During Non-Rapid Eye Movement (NREM) and Rapid Eye Movement (REM) sleep phases, a number of nerve and bodily functions are affected and therefore hold an important role both in their functionalities. This work aims to classify NREM and REM sleep stages from sleep EEG data and present a noble SleepExplain model, an explainable NREM and REM sleep stage classification to explain its predictions. In this work, sleep stages were classified using Random Forest, XGBoost, and Gradient Boosting ensemble classification models. Overall, we obtained an accuracy of 92.54% (Random Forest), 94.25% (Gradient Boosting), and 94.30% (XGBoost). For explainable classification model, we utilized a game theoretic approach, SHAP (SHapley Addictive exPlanations) to offer a convincing explanation for the prediction. 

---
# A Temporal Spatial Minimax Rate for Smoothly-Varying Distributions in Wasserstein Space 

**Authors**: Munsik Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.07325)  

**Abstract**: We study the minimax rate of estimating a future value $\mu_{t_n+h}$ of a curve $t\mapsto\mu_t$ in the $2$-Wasserstein space $\mathcal{P}_2(\mathbb{R}^d)$ from finitely many noisy snapshots of its past, under an adiabatic bound $\|\nabla_t^k v\|\le\varepsilon$ on the $k$-th covariant derivative of the velocity field. Our central result is a unified temporal-spatial minimax lower bound: over regular, locally transport-rich subclasses, every estimator incurs $W_2$-risk with $M$-exponent $\gamma_d(k+1)/(k+1+\gamma_d)$, $\gamma_d=\min(1/d,1/2)$ ($M$ the total sample size). It follows from a temporal-to-spatial reduction: the smoothness budget defines a reachable $W_2$-ball into which a transport packing is embedded along the time axis, and the information of the entire snapshot experiment is controlled by a Fano argument -- the spatial packing is classical, but its smoothness-admissible temporal embedding and the full-window analysis are new. The bound interpolates a dimension-free extrapolation floor of order $\varepsilon h^{k+1}$ -- the irreducible cost of an unobserved future, present even with the exact past -- and the spatial estimation curse $M^{-\gamma_d}$, recovering the static distribution-estimation rate as $k\to\infty$. We state the lower bound in a design-dependent form -- with a design-weighted effective sample size -- valid for arbitrary observation times, and obtain the closed-form exponent in the dense (equispaced) regime. The matching upper bound is established at $k=0$ (rate $M^{-1/(d+1)}$, $d\ge3$) and, in a translation submodel, for all $k$; for $k\ge1$ a covariant estimator attains the rate conditionally on two estimates (a comparison-geometry bias bound and an optimal-transport map-estimation rate), leaving the unconditional general-$k$ upper bound as an open problem. Numerical experiments on synthetic curved and flat families corroborate the predicted exponents. 

---
# Hierarchical Certified Semantic Commitment for Byzantine-Resilient LLM-Agent Collaboration 

**Authors**: Haoran Xu, Lei Zhang, Iadh Ounis, Xianbin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.07316)  

**Abstract**: Byzantine collaboration among large-language-model agents requires a finality-control primitive: given delivered stochastic, structured natural-language proposals, the protocol must decide whether the round supports a commit, what kind of commit, or a typed safe abort. Naive aggregation hides this choice behind a single verdict; classical Byzantine fault tolerance hides it behind byte-identity that LLM proposals do not satisfy. We introduce Hierarchical Certified Semantic Commitment (H-CSC), a BFT-inspired protocol that converts embedding-derived finality signals over verdict-conditioned proposal groups into one of three typed outcomes: a semantic_commit (a 2f+1 within-verdict semantic core backs the verdict, emitting a parameter-bound digest over the quantised aggregate), a verdict_commit (strong verdict margin but dispersed semantic rationale, emitting a verdict-level certificate without claiming a semantic aggregate), or an explicit abort with a typed reason. The contribution is typed finality, not raw commit accuracy. On a controlled semantic-poisoning diagnostic (BCS_v1, 120 episodes), H-CSC commits with low angular deviation on BFT-feasible buckets (0.31 to 2.04 degrees) and aborts 100% of beyond-BFT rounds (n<3f+1) as intended. On a real LLM-agent claim-verification benchmark (MVR-50, 50 tasks) under paired static and rushing Byzantine attacks, H-CSC commits 0.90/0.92 with honest-reference-invalid rates of 0.02/0.00, statistically matching a strong certificate-emitting verdict-only baseline. Unlike that baseline, H-CSC also emits an embedding-backed semantic_commit digest on 74%/72% of rounds, supplying typed provenance. A strict-semantic ablation commits only 0.54/0.48, showing the verdict-level fallback is necessary for coverage (+0.36/+0.44) at the same <=0.04 safety floor; a 100-task cross-model check across four LLMs preserves invalid_hmaj within 0.00 to 0.03. 

---
# SV-Detect: AI-generated Text Detection with Steering Vectors 

**Authors**: Mikhail Vishnyakov, Tatiana Gaintseva  

**Link**: [PDF](https://arxiv.org/pdf/2606.07313)  

**Abstract**: Detecting machine-generated text is especially difficult under distribution shift, such as transfer across domains, source models, and editing attacks. We propose a fake-text detector based on steering vectors extracted from the hidden representations of a frozen language model. At each layer, we construct a direction that separates human-written from machine-generated text, and represent each input by its layer-wise alignment with these directions. A lightweight classifier trained on these projection features yields the final detection score. Our method achieves strong performance both in-distribution and under distribution shift, including across domains, source models, and machine-editing transformations such as polishing and rewriting. Interpretation analyses show that the learned directions align with recognizable stylistic cues while capturing substantial additional signal beyond surface features. These results position fake-text detection as a representation-space probing problem and show that steering vectors provide a simple and effective solution. 

---
# CULTURESCORE: Evaluating Cultural Faithfulness in Video Generation Models 

**Authors**: Anku Rani, Wei Dai, Shravan Nayak, Pattie Maes, Mahdi M. Kalayeh, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2606.07311)  

**Abstract**: As video generation models like Veo 3.1 and LTX-2 advance, their ability to accurately represent diverse global cultures remains a critical yet understudied frontier. Current metrics, such as VideoScore, only measure visual quality but offer no mechanism for assessing cultural faithfulness. Consequently, a model that replaces a Namaste with a handshake receives the same score as one that generates the gesture correctly. We propose CultureScore, a compositional evaluation framework that decomposes cultural faithfulness into three granular dimensions: Identity (who is represented), Context (culturally localized background), and Behavior (normative gestures and interactions). We operationalize this framework through an evaluation suite spanning 10 countries, yielding 6,180 generated videos across three state-of-the-art models. Our evaluation reveals that no current model achieves culturally faithful video generation: the best-performing model reaches only 56.8\% overall CultureScore, with Behavior the most challenging dimension, which remains below 52\% across all models. Furthermore, human preference rankings align directionally with CultureScore but are inverted relative to VideoScore; the highest-scoring model on visual quality was ranked last by annotators, underscoring that cultural faithfulness is an essential criterion for equitable video generation. 

---
# Acoustic Cue Alignment in Audio Language Models for Speech Emotion Recognition 

**Authors**: Iosif Tsangko, Andreas Triantafyllopoulos, Björn W. Schuller  

**Link**: [PDF](https://arxiv.org/pdf/2606.07309)  

**Abstract**: Instruction-following audio language models (ALMs) can be augmented with explicit acoustic cues, yet it remains unclear whether such cues are used in a grounded way when the raw audio is already available. We study this question in speech emotion recognition (SER) by deriving six interpretable acoustic concept tokens from the standardised eGeMAPS paralinguistic feature set. These tokens summarise energy, pitch, dynamics, brightness, formants, and voice quality, and are appended to the textual prompt while the audio input is kept unchanged. Across the widely used FAU-Aibo and IEMOCAP benchmarks, aligned tokens improve unweighted average recall (UAR), whereas shuffled, conflicting, or corrupted tokens reduce performance relative to aligned tokens and shift confusions toward neutral. Importantly, predictions do not collapse under strong token perturbations, suggesting that the models are sensitive to the symbolic cue channel but remain partly anchored to the audio signal. We argue that token-only interventions provide a practical way to probe audio-grounded cue use, robustness, and interpretability in ALM-based affective computing. 

---
# Where Rectified Flows Leak: Characterising Membership Signals Along the Interpolation Path 

**Authors**: Thomas Sesmat, Gabriel Meseguer-Brocal, Geoffroy Peeters  

**Link**: [PDF](https://arxiv.org/pdf/2606.07271)  

**Abstract**: Understanding what generative models retain from training data remains challenging, with implications for copyright and privacy. Beyond verbatim reproduction, models can encode subtler traces of their training data that never surface in their outputs yet remain exploitable. We study this regime for Rectified Flows, which are increasingly used in deployed generative systems. We analyse the interpolation path $X_\lambda = (1-\lambda)X_0 + \lambda X_1$ that defines the Rectified Flow training. We show that a gap exists between the reconstruction of train and test data that follows a bell-shaped curve over $\lambda$, wich accumulates during training, while the validation metrics remain stable. The signal has a maximum whose location we derive in closed form under Gaussian assumptions. We validate these predictions on both audio and images and show that the bell-shaped structure is universal, while the peak prediction holds when our assumptions are satisfied. As a proof of concept, we exploit this specific $\lambda$-resolved structure to perform a Membership Inference Attack, distinguishing members of the training set from non-members. 

---
# AI Sovereignty: A Qualitative Model of Strategic Competition as AI Becomes an Instrument of National Power 

**Authors**: Timothy Clancy, Asmeret Naugle  

**Link**: [PDF](https://arxiv.org/pdf/2606.07245)  

**Abstract**: AI sovereignty is the extent to which a nation independently controls its artificial intelligence (AI) technologies. The race toward ever-more-sophisticated frontier AI models is of increasing strategic importance, with nations considering how AI might improve their economic situations, competitive advantage, and overall national power. However, the costs of AI sovereignty are enormous, and we lack definitions and conceptual models to navigate evolving AI sovereignty dynamics. We address this gap with definitions relevant to AI sovereignty, along with a first-of-its-kind qualitative model that incorporates micro, meso, and macro contributors. Model-based qualitative forecasts highlight competitive dynamics and evolving potential for AI-driven national power. The model identifies key leverage points that nations can use to enhance their own growth or degrade an adversary's, including consideration of accelerators, electricity, water, data sets and skilled workforce. These leverage points can be activated at strategic and operational levels through both direct kinetic actions, such as Iran's targeting of data centers with drones, and indirect non-kinetic effects including cyber, space, information, economic coercion and diplomacy. If our assumptions and hypotheses are valid, this strategic competition may come to define how nations improve their economic situations, competitive advantage, and overall national power in the 21st Century. 

---
# Beyond Waypoints: A Trajectory-Centric Waypointing Paradigm for Vision-Language Navigation 

**Authors**: Haoxiang Shi, Xiang Deng, Haoyu Zhang, Qiaohui Chu, Yaowei Wang, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2606.07244)  

**Abstract**: Vision-Language Navigation in Continuous Environments (VLN-CE) requires agents to follow natural-language instructions while navigating in real-world-like environments. Most VLN-CE approach\-es adopt a three-stage framework: a waypoint predictor proposes navigable waypoints, and a navigator selects the best waypoint, with a low-level controller executing the movement to it. However, this decoupled paradigm often leads to unreachable waypoints or inconsistencies between planning and control. In this work, instead of predicting isolated waypoints, we introduce a novel paradigm called Trajectory Waypoint, which grounds each candidate waypoint in an executable trajectory. To realize this, we design a Trajectory Waypoint Predictor formulated as a TSDF-guided diffusion policy, which steers trajectory generation away from obstacles, inherently ensuring the reachability of the predicted waypoints. We further propose a trajectory-enhanced navigator that injects the associated trajectory as additional information for planning, enabling strict consistency between high-level semantic decisions and low-level execution. Extensive experiments on the VLN-CE benchmark show that our Trajectory Waypoint paradigm achieves superior performance over the baselines. 

---
# When Large Language Models Fail in Healthcare: Evaluating Sensitivity to Prompt Variations 

**Authors**: Mahdi Alkaeed  

**Link**: [PDF](https://arxiv.org/pdf/2606.07237)  

**Abstract**: Large Language Models (LLMs) are increasingly used in healthcare for tasks such as clinical question answering, diagnosis support, and report summarization. Despite their promise, these models remain highly sensitive to subtle prompt perturbations, both lexical and syntactic, posing serious risks in safety-critical clinical applications. In this study, we conduct a systematic sensitivity analysis to evaluate the robustness of both general-purpose (e.g., GPT-3.5, Llama3) and medical-specific LLMs (e.g., ClinicalBERT, BioLlama3, BioBERT) using the MedMCQA benchmark. We categorize perturbations into natural and adversarial types and examine their effect on model consistency, accuracy, and reliability in clinical reasoning tasks. Our findings reveal that medical LLMs are not intrinsically safe. Even minor variations in phrasing can alter clinical advice, and targeted adversarial prompts can provoke harmful outputs. In high-stakes settings like healthcare, such unpredictability is unacceptable-models that change diagnoses due to reworded inputs or hallucinate medications when slightly rephrased cannot be reliably trusted by clinicians. While models tend to show resilience to simple lexical substitutions or paraphrasing, they often break down under syntactic reordering or misleading contextual cues. This fragility is evident across both general-purpose and domain-specific LLMs. Notably, adversarial manipulations can lead to clinically dangerous outputs, such as recommending incorrect dosages or omitting critical findings. 

---
# DEFINED: A Data-Efficient Computational Framework for Fine-Grained Creativity Assessment in Debate Scenarios 

**Authors**: Tongzhou Yu, Mingjia Li, Hong Qian, Wenkai Wang, Zongbao Zhang, Yaoyu Jiang, Xiangfeng Wang, Aimin Zhou, Jiajun Guo  

**Link**: [PDF](https://arxiv.org/pdf/2606.07226)  

**Abstract**: Human creativity has emerged as a critical competency in the era of large language models. Assessing creativity in complex, open-ended environments is a grand challenge in data mining, currently hindered by a reliance on standardized simple tasks and the scarcity of fine-grained expert data. As an ecologically valid assessment context, debate reflects multiple dimensions of creativity, encompassing both divergent thinking and convergent thinking. Moreover, debate is a data-rich domain, with a large volume of publicly accessible materials. Current mainstream automated scoring methods are poorly suited to complex settings such as debate, and therefore still rely on costly human evaluation. To this end, this paper proposes DEFINED, a data-efficient computational framework for fine-grained creativity assessment in debate scenarios. DEFINED operationalizes debate creativity through a hierarchical eight-dimensional metric system, implemented via a pre-trained autoregressive language model with a hierarchical scoring head that supports both fine-grained and coarse-grained evaluation. Statements and their associated expert scores were obtained from authentic debate competitions, and a constrained data augmentation strategy was employed to address the elite bias inherent in the original data. DEFINED adopts a mixed-granularity training strategy enabling robust learning from limited fine-grained supervision annotated by trained graduate experts. To rigorously validate ecological validity beyond synthetic benchmarks, we incorporate an empirical study with debate-naive participants, utilizing these authentic data to serve as a qualitative case study for mid-to-low proficiency populations. Across our evaluation protocol, our scoring model achieves accurate and stable scoring, outperforming prompt-based large language model evaluators and existing debate scoring methods. 

---
# DualGate-Net: A Prior-Gated Dual-Encoder Framework for Histopathology Cell Detection 

**Authors**: Bahman Jafari Tabaghsar, Son Tran, K. Devaraja, Atul Sajjanhar  

**Link**: [PDF](https://arxiv.org/pdf/2606.07222)  

**Abstract**: Cell detection in histopathology images strongly depends on surrounding tissue context, where visually similar cells may belong to different classes under different microenvironments. Recent tissue-aware methods incorporate contextual priors, but often rely on static fusion strategies that may propagate noisy information. In this work, we propose DualGate-Net, a prior-aware dual-encoder framework that combines a ConvNeXtV2-based local encoder and a SegFormer-based global encoder through a learnable prior-gated fusion mechanism. The proposed module adaptively regulates the influence of tissue priors across spatial locations, while an auxiliary foreground reconstruction branch preserves high-frequency cellular structures during training. In addition, auxiliary cellness-guided cues are incorporated to further improve localization robustness. Experiments on the OCELOT benchmark demonstrate consistent improvements, achieving macro F1-scores of 0.7722 on the validation set and 0.7345 on the test set, highlighting the effectiveness of adaptive prior integration for robust histopathology cell detection. 

---
# An Abstract Architecture for Explainable Autonomy in Hazardous Environments 

**Authors**: Matt Luckcuck, Hazel M Taylor, Marie Farrell  

**Link**: [PDF](https://arxiv.org/pdf/2606.07211)  

**Abstract**: Autonomous robotic systems are being proposed for use in hazardous environments, often to reduce the risks to human workers. In the immediate future, it is likely that human workers will continue to use and direct these autonomous robots, much like other computerised tools but with more sophisticated decision-making. Therefore, one important area on which to focus engineering effort is ensuring that these users trust the system. Recent literature suggests that explainability is closely related to how trustworthy a system is. Like safety and security properties, explainability should be designed into a system, instead of being added afterwards. This paper presents an abstract architecture that supports an autonomous system explaining its behaviour (explainable autonomy), providing a design template for implementing explainable autonomous systems. We present a worked example of how our architecture could be applied in the civil nuclear industry, where both workers and regulators need to trust the system's decision-making capabilities. 

---
# RETROSPECT: RETROsynthesis via Sequential Prediction, and Chemically Transformed-ranking 

**Authors**: Raja Sekhar Pappala, Shreyas Vinaya Sathyanarayana, Ronit Kumar Choudhary, Arjun Verma, Deepak Warrier  

**Link**: [PDF](https://arxiv.org/pdf/2606.07181)  

**Abstract**: Single-step retrosynthesis needs both accurate first-ranked suggestions and candidate lists that are rich enough for downstream selection. We study this as a proposal-selection decomposition. Our system, RETROSPECT, combines a single Transformer proposal model, which we call the ChemAlign Transformer, with a LambdaMART reranker over structural, reaction-template, upstream-score, and optional DFT-derived descriptors. The generator is trained with hybrid root-aligned and random SMILES augmentation, Pre-LayerNorm, tied embeddings, exponential moving average weights, and a differentiable atom-balance auxiliary loss. On the full USPTO-50K test set of 5,007 reactions, the generator reaches 55.00% top-1 and 86.18% top-10 exact-match accuracy with 99.86% top-1 validity. On the merged candidate-pool benchmark used for reranking, which contains 5,007 test products and about 111 candidates per product, a LambdaMART model trained on the structural feature set reaches 59.4% top-1 with 0.7171 mean reciprocal rank. Feature ablations show that upstream proposal score and template-frequency statistics provide most of the reranking signal, while DFT and reaction-center DFT features provide smaller and less consistent gains. These results support a modular view of retrosynthesis: stronger single-model proposal and learned candidate selection are complementary, and the proposal model can serve as a drop-in component for ensemble systems such as RetroChimera (Maziarz et al., 2024) 

---
# Textual Supervision Enhances Geospatial Representations in Vision-Language Models 

**Authors**: Marcelo Sartori Locatelli, Fernando Tonucci, Jea Kwon, Luiz Felipe Vecchietti, Bryan Nathanael Wijaya, Cheng Yaw Low, Virgilio Almeida, Meeyoung Cha  

**Link**: [PDF](https://arxiv.org/pdf/2606.07172)  

**Abstract**: Geospatial understanding is a critical yet underexplored dimension in the development of machine learning systems for tasks such as image geolocation and spatial reasoning. In this work, we analyze the geospatial representations acquired by three model families: vision-only architectures (e.g., ViT), vision-language models (e.g., CLIP), and large-scale multimodal foundation models (e.g., LLaVA, Qwen, and Gemma). By evaluating across image clusters, including people, landmarks, and everyday objects, grouped based on the degree of localizability, we reveal systematic gaps in spatial accuracy and show that textual supervision enhances the learning of geospatial representations. Our findings suggest the role of language as an effective complementary modality for encoding spatial context and multimodal learning as a key direction for advancing geospatial AI. 

---
# UrduMMLU: A Massive Multitask Benchmark for Urdu Language Understanding 

**Authors**: Ahmer Tabassum, Sarfraz Ahmad, Hasan Iqbal, Owais Aijaz, Momina Ahsan, Preslav Nakov  

**Link**: [PDF](https://arxiv.org/pdf/2606.07167)  

**Abstract**: Meaningful multilingual evaluation must test models in the target language and educational context. Urdu, spoken by more than 230 million people, lacks a broad MMLU-style benchmark built from native educational sources. We introduce UrduMMLU, a benchmark of 26,431 Urdu MCQs across 26 subjects and five domains, collected from native Urdu MCQ banks and public examination PDFs. Unlike translation-based resources, UrduMMLU covers both standard academic subjects and Urdu- and region-specific content. We label the exam-derived portion through dual human annotation with strict consensus filtering. We evaluate 30 LLMs under English and Urdu prompts, yielding 60 zero-shot evaluations, and further evaluate four open-source LLMs under multiple few-shot settings across both prompt languages. Gemini-3.5-Flash performs best, reaching 90.20% and 90.34% accuracy, while no other model exceeds 85%. The strongest open-source model trails by 7.79 and 8.92 points, and many models lose 25 to 40 points on Urdu-centered Humanities subjects compared with STEM. Few-shot prompting yields only modest gains. UrduMMLU shows that Urdu knowledge remains uneven in current LLMs, especially for regionally grounded content. 

---
# From Privacy to Workflow Integrity: Communication-Graph Metadata in Autonomous Agent Interoperability 

**Authors**: Bijaya Dangol  

**Link**: [PDF](https://arxiv.org/pdf/2606.07150)  

**Abstract**: Agent-interoperability protocols such as A2A and MCP standardize what agents say to one another, but assume address-based transport over HTTP(S). Such transports protect message content, increasingly with end-to-end encryption. What they leave in the clear is the communication graph: which agent contacts which, when, and how often. In agent systems this graph is more consequential than a privacy framing suggests. Endpoints are often capability-labeled, workflows are structured and chained, and interactions are coupled to real actions, so an observer recovers more than past relationships. It can infer the pending workflow, the task being assembled and the action likely to follow. At machine speed, it can act on that inference before the workflow completes. The threat is therefore one of workflow integrity, not privacy alone: predictive leverage over autonomous action. We give a threat model for the agent communication graph; identify what makes agent metadata distinctively revealing (semanticity, prospectivity, actuation); define transport- and bootstrap-layer privacy properties and weigh candidate transports (SimpleX/SMP, Tor, mixnets) against them; and present an A2A case study in which a metadata-protecting binding is expressible but surfaces the protocol's identity assumptions. We test these on a generative model anchored to a real A2A capture. From passive metadata alone, with no payloads, a classifier recovers a task's class well above chance, from only the workflow's opening; applied together, the properties drive that recovery sharply back toward chance. Beyond what an observer can recover, we measure the leverage of acting on the leak: from a workflow's opening and under a fixed budget, an adversary choosing which workflows to act on realizes in this model most of a clairvoyant attacker's advantage over a metadata-blind one, and the same properties suppress it. 

---
# REMEDI: A Benchmark for Retention and Unlearning Evaluation in Multi-label Clinical Disease Inference 

**Authors**: Anurag Sharma, Sai Teja Chunchu, Prasenjit Mitra, Sandipan Sikdar, Koustav Rudra  

**Link**: [PDF](https://arxiv.org/pdf/2606.07141)  

**Abstract**: Language models trained for clinical disease inference are trained on patient data, which may include sensitive and private information, and data owners may request the removal of their data from a trained model due to privacy or copyright concerns. However, exactly unlearning patient-specific data is intractable, and retraining with minor data removal is resource-intensive. While there exists several machine unlearning methods that can be used, their utility is generally restricted to non-medical domains. Moreover, the existing benchmarks for evaluating such unlearning methods primarily utilize synthetically curated datasets, which are not truly representative of real-world systems. Hence, the effectiveness of these unlearning methods in the medical domain is largely unclear. To this end, we introduce REMEDI, an extensive benchmark for machine unlearning tailored to multi-label and multiclass clinical disease inference, where label correlations, longitudinal structure, and safety constraints make unlearning particularly challenging. Unlike the existing benchmarks, REMEDI considers: (1) a relevant application domain (medical), (2) comprehensive unlearning setups involving diverse sets of forget instances, (3) challenging unlearning scenarios including multi-label and multi-class classification tasks, and (4) evaluation metrics involving performance both in terms of utility and extent of unlearning achieved. REMEDI is developed using the MIMIC-III clinical database that contains comprehensive clinical data of patients. Experiments with existing unlearning methods indicate that there exists a trade-off between utility and unlearning performance. They are also largely unsuited to multi-label classification tasks. To facilitate reproducibility, we make our benchmark publicly available. 

---
# The Three-Ring Architecture: Governing Agents in the Era of On-Platform Organisations 

**Authors**: Sergio Alvarez-Telena, Marta Diez-Fernandez  

**Link**: [PDF](https://arxiv.org/pdf/2606.07119)  

**Abstract**: The current phase of enterprise AI deployment faces a structural failure: organisations are acquiring agentic capability without the infrastructure to govern it. The result is expected to reproduce the error of the first wave of AI deployment: decentralised intelligence without a federation layer leading to a 95% project failure rate. This paper formalises the Three-Ring Architecture as the governing infrastructure of the on-platform organisation. Ring 1 is the existing production architecture; Ring 2 is the M2 federation layer built on strategies-based agentic AI; Ring 3 is the LLM-based frontier intelligence layer. Ring 2 constitutes, in the technically exact sense, the operating system of the agentic enterprise - performing at the organisational level what a computing OS performs at the device level: resource abstraction, process coordination, permission enforcement, and a stable platform for compounding intelligence. A central contribution is the formal distinction between Ring 2 and Ring 3 risk profiles. Strategies-based agents operate within a deterministic framework: their consequences are traceable, their permissions enforceable, their deviations recoverable. LLM-based agents introduce a categorically distinct risk: a non-deterministic actor whose deviations propagate through complex organisational systems without retrospective traceability. Ring 2 is not a useful addition - it is a necessary condition of control and compliance. A further consequence: every improvement in LLM capability is a structural tailwind for this architecture. More capable non-deterministic actors produce larger consequences when they deviate. The governance requirement scales with capability. The architecture has been validated across a decade of deployment in financial services, government, procurement, and compliance among other sectors. 

---
# Native3D: End-to-End 3D Scene Generation via Unified Mesh-Texture Modeling and Semantic Alignment 

**Authors**: Yibo Liu, Ziwei Zhang, Haozhou Pang, Menghao Li, Lanshan He, Gan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2606.07117)  

**Abstract**: This paper presents Native3D, the first end-to-end 3D scene generation framework that completely bypasses 2D intermediate representations. Traditional approaches typically require adapting 3D representations to the 2D domain to leverage pre-trained diffusion models, which inevitably introduces domain adaptation issues including geometric structural distortion and texture detail degradation. To address these limitations, we design a unified mesh-texture joint representation that simultaneously models both geometric structures and texture features through a Transformer-based scene encoder, effectively maintaining spatial relationships and visual consistency among objects within scenes. We further propose the 3D Representation Alignment Loss (3D REPA Loss), which employs an improved contrastive learning mechanism to align multi-level semantic representations in the latent space, significantly enhancing geometric and textural fidelity. Experimental results demonstrate that Native3D outperforms existing methods in both generation quality and editing flexibility, providing a novel solution for 3D scene editing. 

---
# OffQ: Taming Structured Outliers in LLM Quantization by Offsetting 

**Authors**: Haoqi Wang, Lorenz K. Mueller, Jiawei Zhuang, Mathieu Salzmann, Lukas Cavigelli  

**Link**: [PDF](https://arxiv.org/pdf/2606.07116)  

**Abstract**: Low-bit quantization has been widely adopted to accelerate the inference of large language models (LLMs) by significantly reducing computational cost and memory usage. However, activation outliers pose a major challenge to effective quantization, often leading to notable performance degradation. In this paper, we introduce OffQ, a method designed to mitigate activation outliers in low-bit quantization through a novel offsetting mechanism. Specifically, OffQ first identifies a low-dimensional outlier subspace in the activations using a proposed top-1 PCA, and then concentrates high-magnitude activations into 1 channel via rotation. OffQ then absorbs this concentrated outlier channel by converting its magnitude into a shared offset, thereby reducing the standard deviation of the activations. This offsetting strategy enables effective W4A4KV4 quantization of LLMs using deployment-friendly uniform-grid and uniform-precision quantization. Extensive experiments across diverse LLM architectures and benchmarks demonstrate that OffQ outperforms state-of-the-art baselines, consistently improving model accuracy while preserving low-bit efficiency. 

---
# DIFFRACT: Neuralized Utility Maximization for Wireless Networks by Differentiable Programming 

**Authors**: Chee Wei Tan, Siya Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.07114)  

**Abstract**: Next-generation wireless networks, including satellite-to-Open RAN systems, demand agile and intelligent resource management capable of handling dynamic multi-user interference under stochastic quality of service constraints. This paper introduces DIFFRACT, a neuralized utility maximization framework that leverages differentiable programming to integrate deep learning with optimization in wireless networks. Central to our approach is the exploitation of the mathematical structure of standard interference functions, which are foundational in wireless power control. By developing a duality theory for these functions, we map iterative interference management algorithms into differentiable neural network architectures via algorithm unrolling. This enables distributed, end-to-end gradient-based learning at the network edge, supporting real-time adaptation to interference in both terrestrial and non-terrestrial environments. DIFFRACT allows for scalable and robust utility maximization by modeling complex channel dynamics and leveraging the expressiveness of differentiable models. Experimental results confirm the framework's theoretical soundness and practical effectiveness for next-generation wireless systems. 

---
# GP-Adapter: Gaussian Process CLIP-Adapter for Few-Shot Out-of-Distribution Detection 

**Authors**: Taisei Saito, Koretaka Ogata, Takafumi Hiroi  

**Link**: [PDF](https://arxiv.org/pdf/2606.07102)  

**Abstract**: We propose GP-Adapter, a training-free framework that augments CLIP (Contrastive Language-Image Pre-training) with Gaussian Process (GP) uncertainty modeling for few-shot classification and out-of-distribution (OOD) detection. While CLIP achieves strong zero-shot recognition, it yields deterministic similarity scores and offers limited uncertainty information, which is critical under distribution shift and data scarcity. GP-Adapter constructs modality-specific, class-wise one-class GPs on top of frozen CLIP embeddings using an RBF kernel for image features and a linear kernel for text prompts and fuses their predictive statistics to produce a variance-aware confidence score for OOD detection. The method requires no fine-tuning of the CLIP backbone and relies only on a small $K$-shot cache and lightweight hyperparameter selection, with memory cost scaling as $O(CK^2)$ for $C$ classes and $K$ shots. Experiments on ImageNet and multiple OOD benchmarks show that GP-Adapter provides competitive few-shot performance and consistently improves OOD detection when combined with prompt-learning baselines, highlighting the complementarity between GP-based uncertainty modeling and prompt learning. Overall, our results suggest that integrating probabilistic inference with large pre-trained vision-language models can improve reliability in low-data and distribution-shifted settings. Code is available at this https URL 

---
# MetaConfigurator: AI-Assisted RDF Authoring from JSON Data 

**Authors**: Felix Neubauer, Mahdi Jafarkhani, Kenichi Endo, Jürgen Pleiss, Benjamin Uekermann  

**Link**: [PDF](https://arxiv.org/pdf/2606.07094)  

**Abstract**: Scientific workflows increasingly generate structured JSON data that is easy to exchange but difficult to interpret consistently across systems due to lacking semantic interoperability. While JSON Schema ensures structural validation, it provides no native support for Linked Data semantics.
This paper presents an RDF Authoring View extending the open-source JSON Schema editor MetaConfigurator, enabling researchers to transform existing JSON, YAML, or CSV data into RDF using AI-assisted RML mappings, refine triples, execute SPARQL queries, visualize knowledge graphs, and export RDF serializations within a single integrated web interface. This workflow is supported by ontology-aware IRI auto-completion, bidirectional synchronization between JSON-LD text views and RDF triple tables, and AI-assisted SPARQL query generation from natural language hints.
We demonstrate the workflow using laboratory data from metal-organic framework (MOF) synthesis experiments. Protocol data describing reagents, procedure steps, and quantities is converted from JSON to ontology-based JSON-LD via RML mappings. We then refine the semantic representation, query relationships between experimental conditions and outcomes, and explore the resulting knowledge graph interactively. This integrated environment bridges conventional structured data management with Semantic Web technologies while preserving experimental context and lowering technical barriers through AI assistance. 

---
# On the Geometry of On-Policy Distillation 

**Authors**: Zhennan Shen, Yanshu Li, Qingyu Yin, Chak Tou Leong, Zhilin Wang, Yanxu Chen, Rongduo Han, Sunbowen Lee, Yi R. Fung  

**Link**: [PDF](https://arxiv.org/pdf/2606.07082)  

**Abstract**: On-policy distillation (OPD) is increasingly used to improve large language model reasoning, but its training dynamics remain poorly understood. We characterize the trajectory of OPD updates in parameter space and compare it with supervised fine-tuning (SFT) and reinforcement learning with verifiable rewards (RLVR). A suite of parameter-space diagnostics consistently places OPD in a relaxed off-principal regime: compared with SFT, its updates affect fewer weights and avoid principal directions more strongly, while compared with RLVR, they remain less tightly constrained. Beyond this static localization, OPD exhibits subspace locking: its cumulative updates rapidly enter a narrow low-dimensional channel. Constraining training to the update subspace formed early in training preserves OPD performance but substantially degrades SFT, indicating that the locked subspace is functionally sufficient for OPD. Control experiments further show that sparsifying the update tokens and shifting rollout generation off-policy preserve the rank dynamics, whereas mixing the OPD objective with RLVR changes them. Overall, these results suggest that OPD is not merely an intermediate point between SFT and RLVR, but induces its own update geometry in parameter space. 

---
# dots.tts Technical Report 

**Authors**: Shi Lian, Changtao Li, Bohan Li, Hankun Wang, Da Zheng, Junfeng Tian, Yufeng Ma, Colin Zhang, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.07080)  

**Abstract**: We present this http URL, a 2B-parameter continuous autoregressive text-to-speech (TTS) foundation model that models speech in a continuous latent space. Compared with existing continuous autoregressive models, our key innovations are threefold. First, we train an AudioVAE with multiple objectives to build a semantically structured and prediction-friendly continuous speech space. Second, we use full-history conditioning in the flow-matching head to preserve long-range consistency and reduce drift during generation. Third, we apply reward-free self-corrective post-training to the flow-matching head to further improve robustness and acoustic quality. After being trained on a large-scale multilingual corpus, this http URL achieves the best average performance on Seed-TTS-Eval, with WERs of 0.94%/1.30%/6.60% and SIM scores of 81.0/77.1/79.5 on the zh/en/zh-hard test sets, respectively. Across other benchmarks, this http URL also consistently demonstrates open-source state-of-the-art performance, exhibiting strong generation stability, voice cloning ability, and emotional expressiveness. For efficient inference, we further apply CFG-aware MeanFlow distillation, enabling low-latency speech generation with first-packet latencies of 85/54 ms in output streaming and dual-streaming modes, respectively. To facilitate reproducible research and practical deployment, we release the training and inference code, together with the pretrained, post-trained, and MeanFlow-distilled checkpoints, under the Apache 2.0 license. 

---
# SlimSearcher: Training Efficiency-Aware Web Agents via Adaptive Reward Gating 

**Authors**: Zequn Xie, Junjie Wang, Dan Yang, Jie Feng, Yue Shen, Jian Wang, Jinjie Gu  

**Link**: [PDF](https://arxiv.org/pdf/2606.07074)  

**Abstract**: Deep research agents have demonstrated remarkable capabilities in complex information-seeking tasks, yet this power comes at a steep computational cost. Driven by accuracy-focused training paradigms, current models adopt brute-force strategies characterized by blind tool dependency and performative reasoning-generating long, redundant trajectories that are far from necessary for resolving these tasks, leading to wasteful tool calls and excessive token consumption. To overcome this efficiency trap, we propose SlimSearcher, a principled framework that pushes the Pareto frontier between accuracy and computational cost across both Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL). In the SFT stage, SlimSearcher employs Pareto-efficient filtration to distill trajectories that are both successful and economical, guiding the model toward inherently efficiency-aware search behaviors. During RL, we introduce Adaptive Reward Gating, a dynamic reward-shaping mechanism that evaluates relative tool and token efficiency within a sampled cohort. By cascading these adaptive efficiency metrics with a strict correctness gate, our approach effectively avoids the brevity bias associated with absolute penalties and mitigates reward hacking. Extensive experiments on long-horizon benchmarks, including GAIA, BrowseComp, and XBenchDeepSearch, demonstrate that SlimSearcher reduces average tool-call rounds by 17%-58% while maintaining or improving accuracy. 

---
# TRACE: Trajectory Reasoning through Adaptive Cross-Step Evidence Aggregation for LLM Agents 

**Authors**: Vijitha Mittapalli, Shreyaa Jayant Dani, Satya Srujana Pilli, Snigdha Ansu, Mohammadreza Teymoorianfard, Franck Dernoncourt, Hongjie Chen, Yu Wang, Ryan A. Rossi, Nesreen K. Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2606.07054)  

**Abstract**: Autonomous LLM agents can pursue hidden malicious objectives through sequences of individually benign actions, making sabotage difficult to detect using standard trajectory-level monitoring. Existing approaches either evaluate complete trajectories in a single pass or partition them into independently scored windows, limiting their ability to connect evidence across temporally distant actions. We propose TRACE, a monitoring framework for long-horizon LLM agent trajectories. TRACE operates through a TIJ (Triage-Inspect-Judge) loop that identifies high-signal regions, performs targeted inspection while maintaining accumulated evidence across reasoning steps, and synthesizes a trajectory-level verdict. We evaluate TRACE on ten task domains from SHADE-Arena against state-of-the-art baselines. TRACE achieves an aggregate F1 of 0.713 and recall of 0.844, with the largest gains on tasks requiring long-range evidence linking. 

---
# STREAM: Stochastic Riemannian Flow Matching with Anisotropic Decoder for Digital Histopathology Image Generation 

**Authors**: Won June Cho, Daeky Jeong, Hyeongyeol Lim, Hongjun Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2606.07036)  

**Abstract**: Synthetic histopathology image generation addresses critical challenges in computational pathology, including patient privacy and the growing need for large-scale training data for foundation models. Latent diffusion models have dominated the image generation domain, with recent works emphasizing that the choice of latent space is critical to the quality of generated images. Existing state-of-the-art generative models in histopathology use pretrained Vision Foundation Models (VFMs) as conditioning signals, and we observe that this leads to "conditioning collapse," where the conditioning signal dominates the latent space and lowers the quality and diversity of generated samples. Therefore, we instead use pretrained histopathology VFMs as the latent space itself, leveraging their patch-token features that encode rich semantic information. We empirically show that these features are $\ell_2$-normalized and lie on the unit hypersphere $\mathcal{S}^{d-1}$ with strong angular dominance and intrinsic curvature, making them naturally suited for a Riemannian formulation. We therefore present STREAM, the first framework to apply Riemannian flow matching in the pathology domain. STREAM consists of two stages: 1) a bridge-type stochastic perturbation that establishes per-token rectifiability on $\mathcal{S}^{d-1}$ for training a Diffusion Transformer (DiT) in latent space, and 2) a novel anisotropic decoder that allocates robustness to low-energy directions of the velocity-field Jacobian while preserving fidelity along its high-energy directions. Together, STREAM achieves state-of-the-art reconstruction and generation performance on breast and colorectal cancer datasets. The code will be publicly released upon acceptance. 

---
# Never Seen Before: Benchmarking Genuine Zero-Shot Composed Image Retrieval with Consistent Video-Sourced Datasets 

**Authors**: Zhenyu Yang, Zemin Du, Shengsheng Qian, Changsheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.07032)  

**Abstract**: Zero-Shot Composed Image Retrieval (ZS-CIR) aims to retrieve a target image based on a query composed of a reference image and a relative caption without training samples. Existing ZS-CIR datasets often suffer from complete irrelevance between reference and target images due to noisy image sources, and do not achieve a true zero-shot scenario as they use public image datasets that models like CLIP have been trained on. To tackle these challenges, we introduce ZeroSight, a novel benchmark for ZS-CIR. It includes a dataset with consistent reference-target pairs sourced from videos, a data construction pipeline, and evaluation methods that consider the ranking of multiple positive and negative target images. We ensure visually and semantically consistent reference-target pairs by extracting frames from a single video and generating relative captions using LLM-assisted methods. To ensure a true zero-shot scenario, we use video data published after March 31, 2022, ensuring it was not included in CLIP's pre-training data. Additionally, we propose a training-free MLLM-driven method, SC4CIR (Symmetric Consistency for CIR), which can effectively identify hard negative targets through 3 symmetric consistency checks. This method is plug-and-play, seamlessly integrating with various CIR methods and significantly improving performance. Our experimental results from 27 methods reveal that current ZS-CIR datasets and evaluation metrics result in inflated retrieval performance, exaggerating the capabilities of CIR methods. Our benchmark and models can be accessed at this https URL. 

---
# Phonetic Error Analysis of Raw Waveform Acoustic Models 

**Authors**: Erfan Loweimi, Zhengjun Yue, Andrea Carmantini, Zoran Cvetkovic, Steve Renals, Peter Bell  

**Link**: [PDF](https://arxiv.org/pdf/2606.07030)  

**Abstract**: We analyse error patterns of raw waveform acoustic models on TIMIT phone recognition beyond the overall phone error rate (PER). PER is decomposed across three broad phonetic class (BPC) categorisations, and confusion matrices are constructed from substitution errors. Our models combine parametric (SincNet, Sinc2Net) or non-parametric CNNs with Bidirectional LSTMs, achieving 13.9%/15.3% PER on Dev/Test, the best reported results for raw waveform models on TIMIT. Transfer learning from WSJ reduces PER to 11.3%/12.3%, surpassing the Filterbank baseline. Per-BPC analysis reveals that BLSTM layers benefit transition-dependent classes most, while WSJ transfer learning improves consonants roughly three times more than vowels. Confusion patterns are consistent across raw waveform and Filterbank systems, indicating that the dominant confusions reflect inherent phonetic similarities. 

---
# Towards Unified Song Generation and Singing Voice Conversion with Accompaniment Co-Generation 

**Authors**: Ziyu Zhang, Chunyu Qiang, Xiaopeng Wang, Yuxin Guo, Kang Yin, Wenjie Tian, Jingbin Hu, Tianlun Zuo, Zhao Guo, Teng Ma, Yuzhe Liang, Chen Zhang, Lei Xie  

**Link**: [PDF](https://arxiv.org/pdf/2606.07015)  

**Abstract**: While song generation and singing voice conversion (SVC) have evolved significantly, they have long been developed isolated: the former lacks zero-shot speaker cloning, while the latter overlooks vocal-accompaniment synergy. To bridge this gap, we propose UniSinger, the first end-to-end framework unifying speaker cloning song generation and accompaniment co-generation SVC. Building on the multimodal diffusion transformer, we construct a unified speaker embedding space transferring speaker representation from SVC to song generation, endowing fine-grained cross-task timbre control. To mitigate multi-task optimization conflicts, we design a curriculum learning strategy using task-specific modality masking to guide the model to gradually master the generative mechanisms among semantic content, vocal timbre, and accompaniment. Experiments show state-of-the-art performance on both tasks and realizes complementary benefits, offering new possibilities for intelligent music production. 

---
# A Geometric View for Understanding Concept Learning and Neuron Interpretation in Sparse Autoencoders 

**Authors**: Chenhao Zhang, Chris Lin, Su-In Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.07007)  

**Abstract**: We propose a unified mathematical framework for a geometric understanding of concept learning and neuron interpretation in sparse autoencoders (SAEs). While SAEs improve interpretability of neural networks by learning sparse feature representations, a principled definition of ''concept'' and ''learning'' remains unclear. We formalize concepts as sets of data points and cast concept learning as a set-alignment problem between human-defined and model-induced concepts. This formulation distinguishes three increasingly strong notions of learning -- detection, separation, and approximation -- and yields geometric conditions, error bounds, and capacity constraints for when concepts can be represented by individual neurons or multi-neuron units. It also provides a set-theoretic account for common SAE phenomena, including feature splitting, feature absorption, feature families, and hierarchical concepts. Finally, we connect concept learning and neuron interpretation through formal concept analysis, showing that the two directions need not agree and that their many-to-many structure can be organized by concept lattices. Experiments on synthetic data with ReLU and Top-$K$ SAEs illustrate the theory and reveal the effects of SAE size and sparsity on concept learning. 

---
# DataEvolver: Automatic Data Preparation for Large Language Models through Multi-Level Self-Evolving 

**Authors**: Chao Deng, Shaolei Zhang, Ju Fan, Xiaoyong Du  

**Link**: [PDF](https://arxiv.org/pdf/2606.07001)  

**Abstract**: High-quality training data is essential to large language models (LLMs) and typically requires extensive and costly manual curation. Existing automatic data preparation methods rely on predefined pipelines or customized human instructions, which limits their adaptability to diverse data distributions and lacks principled guidance from high-quality examples. In this paper, we introduce DataEvolver, the first self-evolving data preparation system that automatically constructs pipelines to transform raw data into high-quality data. DataEvolver employs a multi-level mechanism to ensure both pipeline executability and effectiveness. At the operator level, it incrementally expands the operator set to construct a logical plan while resolving dependency conflicts. At the pipeline level, it instantiates logical plans into executable code and iteratively refines pipeline orchestration through a feedback loop that reduces the distribution gap between prepared data and high-quality examples. Experiments on seven benchmarks show that DataEvolver substantially improves data quality and achieves an average 10\% gain in downstream LLM performance compared with training on original data, highlighting new opportunities for the iterative co-evolution of LLMs and data. 

---
# Don't Pause: Streaming Video-Language Synchrony for Online Video Understanding 

**Authors**: Zhenyu Yang, Kairui Zhang, Shengsheng Qian, Weiming Dong, Changsheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.06991)  

**Abstract**: Online Video Large Language Models (Video-LLMs) have advanced toward seamless human-AI interaction through frame-by-frame processing and proactive responding. However, a critical challenge remains in streaming scenarios: existing models typically pause video perception while generating responses, breaking real-time video-language synchrony and causing stutters. To address this, we introduce a novel paradigm for online video understanding: Streaming Video-Language Synchrony (SVLS), and present LyraV, a live streaming assistant built upon a hierarchical control framework with two core innovations. First, the Frame-Driven Transition Controller (FDTC), a training-free verification-based finite-state machine, makes high-level semantic decisions on when to continue speaking, start a new response, or stay silent. Second, the Streaming Token Pacer (SToP), a plug-and-play lightweight predictive module, dynamically adapts the language generation rate to match the pace of the visual content. Concretely, LyraV performs \emph{per-frame incremental, sub-budget decoding}: within each frame interval it emits only a small chunk of tokens that fits the real-time budget, so perception is never blocked for a full sentence. Together, these components enable LyraV to seamlessly interleave incoming video frames with generated word tokens, achieving a fine-grained synchrony. Extensive experiments conducted on five online and three offline benchmarks demonstrate that LyraV preserves the backbone's general understanding ability while substantially improving streaming synchrony and narrative fluency, delivering a 98.29\% synchrony with video playback and a real-time processing speed of 3.89 FPS. Interestingly, we observe an empirical capability in LyraV: dynamic reasoning over streaming tokens, enabling continuous interpretation and "thinking" alongside visual input. 

---
# DaX: Learning General Pathology Representations Across Scales 

**Authors**: Bokai Zhao, Yiyang Zhang, Long Bai, Tai Ma, Hanqing Chao, Minfeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.06983)  

**Abstract**: Computational pathology requires visual representations that transfer across diverse clinical endpoints and remain robust to variation in magnification, staining, scanner type, slide preparation, and input resolution. We present DaX, a pathology vision foundation model that adapts DINOv3-style self-supervised learning to whole-slide histopathology. DaX is initialized from natural-image DINOv3 weights and incorporates continuous magnification training, cross-scale tissue views, orientation-agnostic and acquisition-robust augmentation, multi-input-size training, and Gram-anchored dense consistency. These designs aim to connect local cellular morphology with global tissue architecture while stabilizing dense token-level representations across input scales. We further construct a WSI-level benchmark comprising 161 clinically meaningful tasks from 44 public datasets, covering 28,182 patients and 34,394 slides across four clinical domains and nine task categories. All models are evaluated under a fixed patient-level cross-validation protocol with fold-level statistical ranking, enabling reproducible comparisons that are less sensitive to split-dependent variation. Across this benchmark, DaX achieves the highest mean performance across tasks and consistently strong task-level ranking scores, with gains spanning diagnostic pathology, biomarker and molecular profiling, tissue/specimen context, and risk, response, and prognosis. These results support DaX as a transferable visual encoder for computational pathology and provide a standardized evaluation framework for future pathology foundation models. Project page: this https URL. 

---
# OpenHalDet: A Unified Benchmark for Hallucination Detection across Diverse Generation Scenarios 

**Authors**: Xinyi Li, Zhen Fang, Yongxin Deng, Jinyuan Luo, Hongnan Ma, Changdae Oh, Zijing Shi, Shanshan Ye, Hanchen Wang, Shu-Lin Chen, Yadan Luo, Mengyue Yang, Sean Du, Sharon Li, Ling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.06959)  

**Abstract**: Hallucination detection is essential for the reliable deployment of large language models (LLMs). However, existing evaluations face two core challenges: inconsistent inference configuration and evaluation, and limited coverage of downstream domains and tasks. Consequently, reported detector performance is often difficult to compare, reproduce, and generalize beyond specific experimental settings. We introduce OpenHalDet, a unified benchmark for hallucination detection across diverse generation scenarios. OpenHalDet standardizes the evaluation pipeline, from prompt construction and response generation to truthfulness annotation, detector scoring, and metric computation. It supports heterogeneous detector families under different access settings, including black-box methods that use only generated outputs, gray-box methods that rely on probability-based signals, and white-box methods that exploit internal model signals. By bringing diverse tasks, models, and detectors into a shared framework, OpenHalDet enables controlled comparison and provides a systematic view of how different detection paradigms behave in LLM applications. We release OpenHalDet as an open and extensible codebase to facilitate reproducible evaluation and future development of hallucination detection methods. The code and datasets are available at this https URL. 

---
# When is 3D Worth It? A Resource-Performance Frontier for CNNs and Transformers in Lung CT 

**Authors**: Md Enamul Hoq, Sharafat Hossain, Imraul Emmaka, Linda Larson-Prior, Lawrence Tarbox, Jonathan Bona, Donald Johann Jr.and Fred Prior  

**Link**: [PDF](https://arxiv.org/pdf/2606.06950)  

**Abstract**: Three-dimensional models are widely assumed preferable for volumetric medical imaging, yet their practical value depends on whether performance gains justify added computational cost and complexity. Rather than proposing a new architecture, we study how input dimensionality (2D, 2.5D, 3D) affects model behavior across convolutional neural networks (CNNs) and Vision Transformers (ViTs) under a fixed training protocol. Using a leakage-free NLST cohort (n = 1,977) with supporting LIDC-IDRI data, we find that the 2.5D CNN offers the most favorable discrimination-stability trade-off in our comparison (ROC-AUC 0.682, 95% CI [0.546, 0.799]) with a stable operating point. In contrast, 3D CNNs show threshold instability, and transformers exhibit degenerate predictions, such as all-positive predictions. Confidence intervals are wide and overlapping, so we present these results as a controlled resource-performance frontier and a failure-mode taxonomy rather than as definitive superiority claims. For class-imbalanced lung cancer screening classification, 2D and 2.5D inputs provide a more reliable trade-off between performance, stability, and computational efficiency than full 3D representations. 

---
# Auditing Training Data in Domain-adapted LLMs: LoRA-MINT 

**Authors**: Gonzalo Mancera, Daniel DeAlcala, Aythami Morales, Julian Fierrez, Ruben Tolosana, Francisco Jurado  

**Link**: [PDF](https://arxiv.org/pdf/2606.06946)  

**Abstract**: We present LoRA-MINT, a new methodology for Membership Inference Test (MINT) applied to recent Large Language Models (LLMs) fine-tuned for specific Natural Language Processing (NLP) tasks through Low-Rank Adaptation (LoRA). The primary goal is to assess whether individual samples were part of the training data of these adapted models, providing a useful auditing tool for the management of intellectual property and sensitive data. Our analysis explores the relationship between model perplexity and membership status, providing a systematic framework for estimating data exposure in fine-tuned LLMs. We conducted experiments on four models and three benchmark datasets, obtaining precision values in determining if given data were used for training ranging from 0.77 to 0.92, which outperform state-of-the-art baselines and demonstrate the robustness and generality of the proposed method. In general, our findings underscore the potential of LoRA-MINT as an effective and scalable framework for auditing LLMs, improving transparency, and fostering the ethical and responsible deployment of AI and NLP technologies. For the sake of concreteness and current relevance, our discussion and experiments are centered on LoRAadjusted LLMs, but note that most of the presented methodology is easily applicable for auditing training data given any other technique for adapting LLMs or, more generally, any other domain-adapted AI models. 

---
# SS-TPT: Stability and Suitability-Guided Test-Time Prompt Tuning for Adversarially Robust Vision-Language Models 

**Authors**: Sunoh Kim, Daeho Um  

**Link**: [PDF](https://arxiv.org/pdf/2606.06943)  

**Abstract**: Vision-language models (VLMs) such as CLIP achieve strong zero-shot recognition but remain highly fragile under adversarial perturbations. Recent test-time adaptation defenses improve robustness by leveraging many augmented views, but this leads to impractical slowdown and a clear robustness-throughput trade-off. To address this challenge, we present Stability and Suitability-guided Test-time Prompt Tuning (SS-TPT), evaluating the quality of each augmented view via two complementary scores: (1) stability, measuring prediction invariance to weak augmentations, and (2) suitability, measuring feature-space density among views. These stability and suitability (SS) scores guide both adaptation and inference through an SS-guided consistency loss and an SS-weighted prediction, amplifying trustworthy views while suppressing corrupted ones. Extensive experiments demonstrate that SS-TPT significantly outperforms prior state-of-the-art methods, achieving superior robustness-throughput trade-offs across diverse datasets and varying numbers of views, thereby demonstrating both strong practicality and generality. Our code is available at this https URL. 

---
# Didact: A Cross-Domain Capability Discovery System for Defence 

**Authors**: Aarya Bodhankar, Aditya Joshi, Bao Gia Doan, Thomas Marchant, Oscar Leslie, Flora Salim  

**Link**: [PDF](https://arxiv.org/pdf/2606.06942)  

**Abstract**: Policymakers in defence and defence-aligned sectors must monitor rapidly evolving research alongside sector priorities relevant to operational and strategic needs. In practice, these sources are fragmented across heterogeneous formats, disjoint repositories, and siloed update streams, making capability discovery slow and difficult to audit. We present Didact, a prototype that integrates publicly available defence reports and policy documents from Australia with a purpose-built knowledge graph derived from Australian research publications. Didact provides natural language conversations for policy-oriented workflows, and leverages a composite retrieval-augmented generation (RAG) pipeline. A key feature of Didact is an interactive Evidence Rail that visualises retrieved evidence and source relationships. Our evaluation of the output quality and runtime of Didact highlights its utility. While Didact has been co-developed as an academia-industry project for the Australian context, it is adaptable to other domains where knowledge is similarly fragmented. A demonstration video is available here: 

---
# The Fine-Tuning Trap: Evaluating Negative Transfer and the Role of PEFT in Sub-1B Mathematical Reasoning 

**Authors**: Rahul Nair, Chun Tao  

**Link**: [PDF](https://arxiv.org/pdf/2606.06920)  

**Abstract**: Deploying Small Language Models (SLMs) on edge devices requires efficient fine-tuning strategies that adapt models to new tasks without degrading their general capabilities. In this study, we benchmark five sub-1B models (135M-1B) on mathematical reasoning tasks and uncover a critical vulnerability: Full Fine-Tuning (Full FT) actively harms performance in models under 300M parameters, often dropping accuracy below zero-shot baselines. This "negative transfer" makes Parameter-Efficient Fine-Tuning (PEFT) not just an efficiency preference, but a stability requirement. We find that while Low-Rank Adaptation (LoRA) and Weight-Decomposed LoRA (DoRA) perform comparably, their strengths vary by task; DoRA excels in complex reasoning (GSM8K), while LoRA dominates pattern matching (OrcaMath). In particular, Full FT is outperformed by LoRA on aligned models (Qwen2.5-0.5B) and even by simple 5-shot In-Context Learning on the smallest architectures (SmolLM2-135M). Based on these findings, we recommend defaulting to PEFT for all aligned sub-1B models and caution against Full FT for any architecture smaller than 500M parameters to prevent catastrophic forgetting. Reproduction of this work can be found at this https URL. 

---
# ThinkBooster: A Unified Framework for Seamless Test-Time Scaling of LLM Reasoning 

**Authors**: Vladislav Smirnov, Chieu Nguyen, Sergey Senichev, Minh Ngoc Ta, Ekaterina Fadeeva, Artem Vazhentsev, Daria Galimzianova, Nikolai Rozanov, Viktor Mazanov, Jingwei Ni, Tianyi Wu, Igor Kiselev, Mrinmaya Sachan, Iryna Gurevych, Preslav Nakov, Timothy Baldwin, Artem Shelmanov  

**Link**: [PDF](https://arxiv.org/pdf/2606.06915)  

**Abstract**: Test-time compute (TTC) scaling has emerged as a powerful paradigm for improving large language model (LLM) reasoning by allocating additional compute during inference, e.g., via multi-sample generation and verifier-based reranking. Existing TTC scaling strategies and reasoning scorers remain fragmented, evaluated under inconsistent protocols, and are rarely analyzed through the lens of quality-cost trade-offs. We introduce ThinkBooster, a unified framework for seamless test-time compute scaling of LLM reasoning, which consists of (i) a modular Python library implementing state-of-the-art TTC scaling strategy and scorer families, (ii) a benchmark that jointly evaluates performance and computational efficiency, and (iii) a deployable OpenAI-compatible proxy service that enables drop-in integration of adaptive reasoning into real-world applications. We further provide a demo visual debugger for inspecting the reasoning trajectories, intermediate selection decisions, and alternative reasoning paths. Empirical results on mathematical and coding tasks reveal the performance-compute trade-offs of TTC scaling strategies and scoring methods and demonstrate that ThinkBooster provides practical gains in real-world tasks. The code is available online under an MIT license. 

---
# SpectCount: Spectrotemporal Counting via Synthetic Signals Improves Large Audio Language Models 

**Authors**: Seonuk Kim, Yonghyeon Jun, Ju Yeon Kang, Jimin Hong, Yoonhyeong Lee, Nam Soo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.06907)  

**Abstract**: Large audio language models (LALMs) extend large language models with an audio encoder and large-scale audio data. However, the scarcity of high-quality annotated audio data remains a fundamental bottleneck for scaling. Through probing signal detectability analysis, we identify fine-grained spectrotemporal perceptual weaknesses in a foundation LALM. To address these challenges, we propose Spectrotemporal Counting (SpectCount), a data-efficient fine-tuning approach based on fully synthetic audio signals generated on-the-fly, without relying on real-world audio, annotations, or pretrained generative models. SpectCount not only resolves the observed weaknesses but also improves performance on diverse auditory benchmarks spanning sound, music, and speech, unseen during fine-tuning. These results suggest that weakness-targeted synthetic signals provide a data-efficient path toward enhanced auditory understanding capabilities in LALMs. 

---
# EASE-TTT: Evidence-Aligned Selective Test-Time Training for Long-Context Question Answering 

**Authors**: Xiaopeng Yuan, Zebin Wang, Suwen Wang, Zongxin Yang, Haohan Wang, Yushun Dong  

**Link**: [PDF](https://arxiv.org/pdf/2606.06906)  

**Abstract**: Long-context question answering (QA) remains challenging for smaller language models even when answer-bearing evidence is already present in the input. Existing within-context retrieval methods localize and expose candidate evidence chunks for the question, but they stop at input-level evidence exposure rather than adapting the query-side attention parameters that control how the model allocates attention over full-context positions. In contrast, lightweight test-time adaptation methods, such as query-only test-time training (qTTT), leave evidence localization unresolved because their generic span-level self-supervised objectives do not identify which context positions support the current answer. In this paper, we propose Evidence-Aligned SElective Test-Time Training (EASE-TTT), a within-context retrieval-augmented test-time training framework that converts selected evidence chunks into a soft attention supervision target over their token positions. Instead of replacing the full context with retrieved chunks, EASE-TTT uses the resulting attention target to guide query-side adaptation, with the adapted model generating the final answer from the original full context. Experiments on six LongBench QA tasks and three small decoder-only language models show that EASE-TTT achieves the strongest macro-average performance among full-context inference, retrieval-only baselines, and qTTT, supporting evidence-aligned test-time adaptation in long-context QA. 

---
# Beyond Skeletons: Learning Animation Directly from Driving Videos with Same2X Training Strategy 

**Authors**: Yuan Zeng, Yujia Shi, Yuhao Yang, Dongxia Liu, Zongqing Lu, Wenming Yang, Qingmin Liao  

**Link**: [PDF](https://arxiv.org/pdf/2606.06903)  

**Abstract**: Human image animation aims to generate a video from a static reference image, guided by pose information extracted from a driving video. Existing approaches often rely on pose estimators to extract intermediate representations, but such signals are prone to errors under occlusion or complex poses. Building on these observations, we present DirectAnimator, a framework that bypasses pose extraction and directly learns from raw driving videos. We introduce a Driving Cue Triplet consisting of pose, face, and location cues that captures motion, expression, and alignment in a semantically rich yet stable form, and we fuse them through a CueFusion DiT block for reliable control during denoising. To make learning dependable when the driving and reference identities differ, we devise a Same2X training strategy that aligns cross-ID features with those learned from same-ID data, regularizing optimization and accelerating convergence. Extensive experiments demonstrate that DirectAnimator attains state-of-the-art visual quality and identity preservation while remaining robust to occlusions and complex articulation, and it does so with fewer computational resources. Our project page is at this https URL. 

---
# FreeAnimate: Training-Free Human Image Animation with Preview-Guided Denoising 

**Authors**: Yuan Zeng, Yujia Shi, Zongqing Lu, QingMin Liao  

**Link**: [PDF](https://arxiv.org/pdf/2606.06885)  

**Abstract**: Human Image Animation has seen significant advancements, primarily driven by diffusion models. However, existing methods typically demand substantial training data and resources to achieve high-quality results, limiting generalization and accessibility. In this work, we introduce \emph{FreeAnimate}, a training-free framework that leverages the inherent capabilities of image diffusion models to enable temporal consistency, identity preservation, and background stability. Our approach incorporates a novel preview generation strategy that provides temporal and structural priors from generated preview frames, effectively guiding pose alignment and background consistency without training. Additionally, FreeAnimate introduces Inversion-Boosted Attention and Reference-Anchored Self-Attention modules to guarantee temporal consistency and identity preservation. Experimental results demonstrate that FreeAnimate outperforms existing training-free competitors and training-based baseline methods, achieving generation quality comparable to state-of-the-art methods and offering robust generalization across diverse datasets. Our project page is at this https URL. 

---
# Neuro-Symbolic Learning for Long-Horizon Task Planning Under Complex Logical Constraints 

**Authors**: Qiwei Du, Zitong Zhan, Shaoshu Su, Bowen Li, Yi Du, Zhipeng Zhao, Taimeng Fu, Sebastian Scherer, Jiaoyang Li, Chen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.06877)  

**Abstract**: Task planning often suffers from severe efficiency bottlenecks when robots must reason over long-horizon action sequences under complex logical constraints, including object affordances, spatial relationships, and sequential action dependencies. Recent neuro-symbolic methods improve planning efficiency by learning object-importance scores to prune task-irrelevant objects, but they typically rely on fixed offline supervision generated from full search spaces. This creates a train-test mismatch: at deployment, the planner operates in pruned search spaces induced by the model's own imperfect predictions, leading to exposure bias and degraded planning performance. To address this challenge, we formulate object-importance learning for task planning as an imperative learning-based bilevel optimization problem. The upper level optimizes a neural scorer, while the lower level solves a symbolic planning problem in the score-pruned search space. To stabilize this learning process, we introduce a 3R strategy into the lower-level planning, using parallel Repair, Restart, and Rollback recovery to provide reliable and adaptive feedback for upper-level learning. Experiments on three challenging benchmarks demonstrate state-of-the-art performance, including an 80.04% reduction in failure rate and a 57.14% reduction in planning time. We further validate the framework on a quadruped-based mobile manipulator in simulation and the real world, demonstrating its potential for efficient and deployable neuro-symbolic task planning. 

---
# EgoPressDiff: Multimodal Video Diffusion for Egocentric UV-Domain Hand-Pressure Estimation 

**Authors**: Yuan Zeng, Zilue Gao, Yujia Shi, Zongqing Lu, Wenming Yang, QingMin Liao  

**Link**: [PDF](https://arxiv.org/pdf/2606.06872)  

**Abstract**: Estimating hand-surface contact pressure from an egocentric view is crucial for AR/VR devices, robotic imitation, and ergonomic analysis. Existing methods often discretize pressure signal and process frames independently, leading to quantization errors and temporal inconsistencies. We present \emph{EgoPressDiff}, a conditional video diffusion framework that generates UV-pressure maps from visual input. The core of our approach is a multi-modal conditioning strategy, introducing a PoseNet and a Vertex Encoder to efficiently extract features from hand pose and 3D mesh vertices. These signals, along with depth information, guide the generative process to ensure the pressure fields are physically grounded. To effectively fuse these heterogeneous features, we further propose a Distribution-Calibrated Spatial Layer, which aligns their statistical properties before combination. Evaluated on the EgoPressure ego-view setting, EgoPressDiff achieves state-of-the-art results, improving Volumetric IoU by over 34\% relative to prior baseline, while reducing MAE and maintaining high temporal accuracy. Our project page is at this https URL. 

---
# Modeling Nonlinear Feature Interactions with Product-Unit Residual Networks 

**Authors**: Ziyuan Li, Uwe Jaekel, Babette Dellen  

**Link**: [PDF](https://arxiv.org/pdf/2606.06861)  

**Abstract**: Understanding nonlinear feature interactions is crucial in science and engineering, yet standard multilayer perceptrons (MLPs) often capture such interactions only implicitly, leading to entangled representations that can impair robustness and interpretability. We investigate product-unit residual networks (PURe) that integrate multiplicative product units with residual connections to explicitly model cross-feature couplings while stabilizing optimization. We conduct a systematic evaluation on an interaction-driven synthetic benchmark and two real-world datasets, assessing predictive accuracy, robustness to Gaussian feature noise, and performance under limited training data, and we compare real- and complex-valued variants under a matched parameter budget. Beyond accuracy, SHapley Additive exPlanations (SHAP)-based interaction analyses show that PURe learns more concentrated and structurally coherent interaction patterns than MLP baselines. Overall, PURe achieves competitive or improved performance, better robustness and sample efficiency in low-data regimes, and enhanced interaction-level interpretability. 

---
# MotionEnhancer: Leveraging Video Diffusion for Motion-Enhanced Vision-Language Models 

**Authors**: Yifan Xu, Chao Zhang, Ruifei Ma, Fei Gao, Zhifei Yang, Jiaxing Qi, Zhipeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.06853)  

**Abstract**: The new era has witnessed a remarkable capability to extend Vision-Language Models (VLMs) for tackling tasks of video understanding. While current VLMs excel at event- or story-level understanding, their ability to capture fine-grained motion details remains limited, primarily due to their focus on high-level static semantic structures and macro-event logic. In contrast, Video Diffusion Models (VDMs) are adept at modeling dynamic motion patterns, benefiting from large-scale video data and the intrinsic requirement of temporal generation. In this paper, we introduce MotionEnhancer, a novel approach that leverages motion priors distilled from a powerful video diffusion model as auxiliary supervision to enhance the motion understanding capability of a VLM via attention alignment. MotionEnhancer comprises two simple parameter-free modules, Motion-sensitive Head Selection (MHS) and Motion-salient Text Token Identification (MTTI), to directly extract and optimize motion-related attentions from the VDM in a computation-only manner. MotionEnhancer provides a scalable solution for motion understanding without additional training parameters, modifications to existing architectures, or tool calling. Extensive experiments demonstrate that MotionEnhancer can achieve consistent improvements over state-of-the-art VLMs on two motion-level video understanding benchmarks, especially on motion-related metrics. 

---
# Characterize Then Distill: Mechanistic Reasoning in Large Output Spaces 

**Authors**: Debjyoti Saha Roy, Byron C. Wallace, Javed A. Aslam  

**Link**: [PDF](https://arxiv.org/pdf/2606.06840)  

**Abstract**: Modern reasoning models offer surprisingly strong zero-shot performance on challenging multi-label tasks that require selecting a small set of relevant options from hundreds of thousands to millions of candidate labels. We investigate how they achieve this mechanistically. We characterize reasoning as a two-phase process: A broad "shortlisting" of candidates followed by fine-grained reasoning over the resulting set. We provide evidence across a range of datasets that these steps can be isolated and are complementary. Using this characterization, we develop a mechanistic distillation strategy that consistently outperforms standard distillation. 

---
# LLM Agent-Assisted Reverse Engineering with Quantitative Readability Metrics 

**Authors**: Neil Archibald, Ruben Thijssen  

**Link**: [PDF](https://arxiv.org/pdf/2606.06838)  

**Abstract**: Automatic decompilers produce functionally correct but often unreadable C code. This paper addresses one stage of the reverse engineering workflow: improving the readability of decompiled code using LLM agents guided by quantitative metrics.
We present a three-phase research evolution. Phase 1 (tool-driven steering via Ghidra MCP) suffered from incomplete coverage and inconsistent improvements due to lack of quantitative guidance. Phase 2 (structural similarity validation alone) revealed that agents optimize for metrics in unintended ways, producing structurally equivalent but less readable code. Our contribution is the Quantitative Readability Score (QRS) framework, a composite metric combining a structural similarity gate with three independent readability sub-metrics (Lexical Surprisal, Structural Simplicity, and Idiomatic Quality).
We demonstrate that QRS-guided refinement enables LLM agents to make targeted readability improvements without sacrificing correctness. We provide a discussion of the broader reverse engineering workflow (binary lifting, decompilation cleanup, and achieving functional equivalence) as context, however, it remains out of scope. 

---
# Think Like a Pilot: Fine-Grained Long-Horizon UAV Navigation 

**Authors**: Xiangyi Zheng, Xiangyu Wang, Qinan Liao, Zimu Tang, Yue Liao, Dongyue Lyu, Guodong Wang, Junjie Liu, Si Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.06836)  

**Abstract**: Language-guided UAV agents must execute long-horizon semantic instructions while producing smooth, physically feasible continuous flight commands, yet existing Vision-Language Navigation (VLN) benchmarks typically use discrete or coarse actions and existing UAV Vision-Language-Action (VLA) tasks focus on short, atomic maneuvers. To address this gap in UAV task settings, we introduce \textbf{FLIGHT}, a \textbf{F}ine-grained \textbf{L}ong-horizon \textbf{I}nstruction-\textbf{G}uided benchmark for \textbf{H}ybrid UAV navigation and reasoning \textbf{T}asks, which combines multi-stage instructions with dense 6-DoF trajectory annotations across two dataset splits: Fine-grained VLN and Long-horizon Flow. To endow the UAV agent with the capability of real-time in-flight reasoning over task execution status and mission planning, while simultaneously accommodating high-frequency, real-time precise control, we further propose \textbf{FLIGHT VLA}, an asynchronous architecture that decouples a low-frequency Streaming Pilot Vision-Language Model (VLM) for task-state reasoning from a high-frequency diffusion action model for continuous control, supervised by explicit \textbf{Pilot Reasoning} texts that summarize the current flight state and anticipate the next subgoal. In closed-loop evaluation, FLIGHT VLA consistently surpasses representative VLN and VLA baselines on our FLIGHT benchmarks, achieving stronger multi-stage completion, subgoal adherence, and terminal control. Its trained Streaming Pilot Reasoning VLM further improves UAV video reasoning, validating the effectiveness of our design. 

---
# Hearing the Unspoken: Language Model Priors for Acoustic Adversarial Attacks 

**Authors**: Jiani Xie, Andrew C. Cullen, Paul Montague, Benjamin I. P. Rubinstein  

**Link**: [PDF](https://arxiv.org/pdf/2606.06833)  

**Abstract**: Automatic Speech Recognition (ASR) systems operating in real-time settings must process acoustic input under strict temporal constraints, where transcription decisions are inherently made on incomplete information. This causal constraint serves as an information bottleneck on attackers, significantly limiting attack performance. Our new Semantic Gambit attack breaks this causal limitation by augmenting the adversary with predictive context derived from a Large Language Model in real-time. Our experiments show that this form of augmentation can elevate the corpus-level Word Error Rate to 35.6% -- a three-fold increase over the current state-of-the-art. Ultimately, this work reveals how common, low-latency LLM tooling can be exploited to systematically subvert real-time ASR pipelines. 

---
# Progress-SQL: Improving Reinforcement Learning for Text-to-SQL via Progressive Rewards 

**Authors**: Shihao Zhang, Xiaoman Wang, Yuan Liu, Yunshi Lan, Weining Qian  

**Link**: [PDF](https://arxiv.org/pdf/2606.06825)  

**Abstract**: Reinforcement learning has recently shown promise in improving large language models for Text-to-SQL generation, yet existing methods typically optimize one-shot rewards defined over a single SQL state. Such rewards provide limited guidance for iterative SQL correction and are insufficient to capture the improvement of multi-turn SQL refinement. In this paper, we propose Progress-SQL, a multi-turn reinforcement learning framework with progressive rewards for Text-to-SQL. Our approach introduces an Oracle-guided Diagnostic Tree (ODT), which abstracts SQL queries into clause-level structural profiles and produces diagnostic feedback for next-turn refinement. To provide dense and robust reward signals, we combine ODT-based structural alignment with lexical alignment and define a progressive reward that measures the improvement from the initial SQL to the final SQL. We further incorporate a progression latency reward that favors earlier correctness and an execution status reward that encourages recovery from the invalid SQL. Experiments on BIRD, Spider, and Spider robustness variants demonstrate that our method consistently improves Text-to-SQL performance across both primary and robustness evaluations. 

---
# PandaAI: A Practical Agent CQ2 for Neuro-symbolic Data Analysis And Integrated Decision-Making in Quantitative Finance 

**Authors**: Yuqi Li, Siyuan Liu, Bingjun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.06823)  

**Abstract**: While deep learning has excelled in various domains, its application to sequential decision-making in finance remains challenging due to the low Signal-to-Noise Ratio (SNR) and non-stationarity of financial data. Leveraging the reasoning capabilities of Large Language Models (LLMs), we propose \textbf{PandaAI}, a closed-loop neuro-symbolic LLM agent with market regime modeling and constrained alpha generation, which bridges general LLM reasoning with financial rigor and suppresses the financial toxicity of LLM-generated outputs. To bridge the gap between general linguistic capability and financial rigor, we fine-tune a domain-specific LLM. Furthermore, we integrate this LLM into a modular architecture and form a closed-loop system. Unlike traditional models that optimize isolated prediction metrics, \textbf{PandaAI} is designed as a neuro-symbolic agent that navigates the complex, real-world financial environment with explicit risk awareness. Extensive experiments on CSI 300 stock data show that \textbf{PandaAI} achieves a $18.2\%$ higher Rank IC and $25.7\%$ lower maximum drawdown than state-of-the-art time-series models. Our constrained LLM generation and dual-channel adaptation method provide a general paradigm for LLM deployment in high-stakes sequential decision-making scenarios. 

---
# SCALE: Scalable Cross-Attention Learning with Extrapolation for Agentic Workflow Scheduling 

**Authors**: Zhifei Xu, Jierui Lan, Zixuan Liang, Aiji Liang, Jinxi He  

**Link**: [PDF](https://arxiv.org/pdf/2606.06820)  

**Abstract**: Agentic Large Language Model (LLM) systems decompose complex tasks into workflow Directed Acyclic Graphs (DAGs) whose primitives must be scheduled on heterogeneous clusters. Existing deep reinforcement learning (DRL) schedulers are tied to a fixed cluster size and require retraining whenever the number of servers changes. We propose SCALE (Scalable Cross-Attention Learning with Extrapolation), a DRL scheduler that generalizes to unseen cluster scales without fine-tuning. SCALE employs a cross-attention pointer network where task features query against server features, so the architecture accepts any number of servers by construction. We observe, however, that permutation-invariant architecture alone does not guarantee good performance at new scales - the attention feature undergoes distribution shift as the server count grows. To counter this, we introduce Structured Representation Regularization (SRR): a decorrelation loss combined with a KL penalty toward the standard normal, which keeps feature statistics stable regardless of input size. Trained on 16 nodes and tested directly on 32 and 48 nodes, SCALE reduces average response time by 8.9% at N=48 relative to the same architecture without SRR, confirming that explicit regularization is necessary to close the scale-generalization gap. 

---
# Breaking the Lock-in: Diversifying Text-to-Image Generation via Representation Modulation 

**Authors**: Dahee Kwon, Haeun Lee, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2606.06813)  

**Abstract**: Recent text-to-image models built on large-scale Transformer backbones and flow-based objectives deliver strong text-image alignment and high visual quality, yet often produce overly similar samples under a fixed prompt. Existing diversity-enhancement methods alleviate this issue, but typically require expensive sampling or auxiliary optimization, incurring non-trivial overhead. To investigate the root cause of this homogeneity, we examine intermediate Transformer features and observe that the zero-frequency spatial average (DC) component rapidly converges across seeds early in generation, causing early trajectory lock-in that limits downstream variation. Building on this observation, we propose DC Attenuation for diVersity Enhancement (DAVE), a training-free representation-level intervention that selectively attenuates this component in the early regime. DAVE preserves the sampling pipeline with negligible overhead, improving prompt-consistent diversity while maintaining competitive image quality. 

---
# Lane Change Trajectory Planning for Personalized Driving Comfort and Mobility Efficiency 

**Authors**: Haoxuan Dong, Dongjun Li, Ziyou Song  

**Link**: [PDF](https://arxiv.org/pdf/2606.06805)  

**Abstract**: Lane changing entails simultaneous longitudinal and lateral motions that affect driving comfort and mobility efficiency. Because these motions are tightly coupled and subject to substantial inter-vehicle variability, trajectory planning for lane-change maneuvers is characterized by a highly personalized nature. This study proposes a neural network-driven planner that integrates a third-order polynomial trajectory generator with a learning module that infers optimal trajectory parameters across diverse driving conditions. Using a shared backbone with dual heads, one head ensures all-condition operational guarantees, while the other captures driver-specific preferences for comfort or mobility efficiency. A head-gated switching mechanism, realized through a statistical gate based on error-winner logistic regression, adaptively selects the appropriate head under varying driving conditions, which enables context-aware lane-change trajectory planning. Representative cases and Monte Carlo simulations show that the proposed planner achieves personalized comfort and mobility during lane changes, while the baseline ensures feasible trajectories under driving conditions where personalized data are insufficient or inaccessible. 

---
# Exploring Reinforcement Learning for Fluid Transitions Between Clinical Mental Healthcare and Everyday Wellness Support 

**Authors**: Tony Wang, Qian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.06800)  

**Abstract**: Mental health struggles wax and wane, yet clinical and wellness interventions typically operate separately, causing frequent breakdowns at care transitions. We explore reinforcement learning (RL) as a means to build digital health systems that deliver clinical and wellness interventions proactively, as part of a coherent care journey. We ask: what complexities does designing such a system involve? We built a contextual bandit that dynamically selects journaling prompts from clinical and wellness repertoires to optimize for an overarching health goal (sustained journaling) and deployed it in a four-week exploratory study (N=38). We found that, first, many benefits of RL-optimized intervention sequences appeared only after interventions ended, raising the question: Should systems that offer coherent clinical-wellness care journeys include stepping-back periods? If so, when and how? Second, participants most engaged with RL-generated interventions deepened their engagement over time, while those most engaged with a constant intervention tended to burn out and drop out later. It raises the question: When should a system blending clinical and wellness interventions reduce intensity to prevent burnout in versus sustain it to maximize treatment gains? 

---
# What Your Posts Reveal: A Benchmark and Agentic Framework for User-Level Privacy Leakage on Social Media 

**Authors**: Zifan Peng, Yini Huang, Aiwen Lu, Qiming Ye, Peixian Zhang, Jingyi Zheng, Yule Liu, Xuechao Wang, Xinlei He, Jiaheng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2606.06784)  

**Abstract**: Public social media posts can reveal private information through weak cues scattered across text, images, or metadata. Such leakage is often cumulative and cross-post: cues that appear harmless in isolation may jointly expose a user's home, workplace, or routine. However, current research lacks a unified benchmark for user-level multimodal privacy leakage and an evaluation metric that captures exposure severity beyond binary accuracy.
To address these gaps, we propose SopriBench, a synthetic benchmark guided by leakage patterns abstracted from a private reference corpus of Rednote and Instagram accounts, covering 50 user profiles and 1,569 images with attributes, contextual sensitivity, granularity, leakage type, inference difficulty, and supporting evidence. We further introduce the Privacy Exposure Score (PES), which weights value granularity by contextual sensitivity. Inspired by abductive reasoning, we introduce Argus, a training-free agentic framework for cumulative leakage inference. Argus forms hypotheses from accumulated evidence, verifies supporting evidence, and aggregates cross-post cues into privacy profiles, achieving 0.55 PES, a 25% improvement over the strongest baseline, with the largest gain on cross-post leakage. 

---
# Mind the Gap: Bridging Behavioral Silos with LLMs in Multi-Vertical Recommendations 

**Authors**: Nimesh Sinha, Raghav Saboo, Martin Wang, Sudeep Das  

**Link**: [PDF](https://arxiv.org/pdf/2606.06779)  

**Abstract**: In multi-vertical e-commerce platforms like DoorDash, relatively newer product verticals such as grocery and retail present a significant opportunity for personalization innovation. A key challenge lies in solving the "cold start" problem for users. This paper introduces a novel framework for enhancing recommendation quality by transferring knowledge from data-rich verticals (e.g., restaurants at DoorDash) to data-sparse ones. We leverage Large Language Models (LLMs) to perform generative inference, synthesizing sparse, high-dimensional features that encapsulate latent user affinities. Specifically, we employ a hierarchical Retrieval-Augmented Generation (RAG) pipeline to derive multi-level taxonomic features from user restaurant order histories and search queries. These generated features, encoding both long-term cross-vertical preferences and short-term intent, are integrated into a production Multi-Task Learning (MTL) ranking model. We demonstrate through extensive offline and online evaluation that this approach significantly improves personalization and engagement in emerging business verticals, effectively bridging the behavioral data gap. 

---
# Generalization in Deep Neural Networks: Minimax Rates for Gradient Methods 

**Authors**: Junyu Zhou, Puyu Wang, Yunwen Lei, Marius Kloft, Yiming Ying  

**Link**: [PDF](https://arxiv.org/pdf/2606.06772)  

**Abstract**: Understanding the generalization performance of over-parameterized neural networks has become a central topic in deep learning theory. While recent advances, particularly works under the Neural Tangent Kernel (NTK) regime, have shed light on the behavior of shallow architectures, the statistical generalization properties of deep neural networks (DNNs), especially in regression tasks, remain far less understood. In this paper, we make significant progress toward closing this gap by providing a comprehensive generalization analysis of DNNs trained using gradient-based methods. First, we establish, for the first time, a crucial connection between the learning dynamics of a DNN with smooth activation functions trained via gradient-based methods and those of kernel methods, showing that gradient-based methods on over-parameterized DNNs can fully inherit the favorable learning dynamics of their kernel counterparts. Building on this connection and the well-established optimality of kernel methods, we derive the first known minimax-optimal rates for the excess population risk of both gradient descent (GD) and stochastic gradient descent (SGD), under the assumption that network width scales polynomially with the sample size. Our results demonstrate that, with sufficient width, DNNs trained by GD or SGD can achieve generalization performance comparable to kernel-based methods. 

---
# Optimal Rates for Generalization of Gradient Descent Methods with Deep Neural Networks 

**Authors**: Junyu Zhou, Puyu Wang, Yunwen Lei, Yiming Ying, Ding-Xuan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.06764)  

**Abstract**: Recent progress has been made in understanding the statistical generalization performance of gradient descent methods for overparameterized neural networks within the neural tangent kernel (NTK) regime. However, most of the existing work on regression problems is limited to shallow network architectures, leaving a notable gap in the theory of deep neural networks. This paper addresses this gap by presenting a comprehensive generalization analysis for deep ReLU networks trained using gradient descent (GD) and stochastic gradient descent (SGD). Specifically, we establish the first known minimax-optimal rates of excess population risk for both GD and SGD with deep ReLU networks, under the assumption that the network width scales polynomially with respect to the network depth and training sample size. Our results demonstrate that with sufficient width, gradient descent methods for deep ReLU networks can achieve optimal generalization rates on par with kernel methods. 

---
# AxisGuide: Grounding Robot Action Coordinate System in RGB Observations for Robust Visuomotor Manipulation 

**Authors**: Jiyun Jang, Yujin Sung, Woosung Joung, Daewon Chae, Sangwon Lee, Sohwi Kim, Jinkyu Kim, Jungbeom Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.06761)  

**Abstract**: Visuomotor manipulation policies trained via large-scale behavior cloning have achieved strong semantic scene understanding, yet often fail to reliably execute correct low-level actions under distribution shifts. For example, even in a simple pickup task with identical scene layouts, camera viewpoints, and illumination, performance can degrade substantially when the object is placed at unseen locations. We argue that this gap arises from insufficient action understanding, namely the inability to interpret the robot's base-frame action coordinate system in image space. To address this issue, we introduce AxisGuide, a lightweight guidance method that bridges semantic scene understanding and action-coordinate interpretation. Using camera parameters and end-effector poses, AxisGuide renders the robot base-frame axes in each camera view and augments RGB observations with a small set of cue channels that explicitly visualize the meaning of the +x, +y, and +z motions in image space. Extensive evaluations in both the LIBERO simulation and real-world environments demonstrate that AxisGuide yields substantial performance gains and improved generalization, highlighting the effectiveness of explicit action-coordinate cues for learning reliable and transferable generalist visuomotor policies. 

---
# Evidence Graph Consistency in Retrieval-Augmented Generation: A Model-Dependent Analysis of Hallucination Detection 

**Authors**: Jianru Shen  

**Link**: [PDF](https://arxiv.org/pdf/2606.06748)  

**Abstract**: Retrieval-Augmented Generation (RAG) reduces but does not eliminate hallucination in large language models. Existing detection methods rely on flat similarity between generated answers and retrieved passages, ignoring structural relationships among evidence pieces and answer claims. We propose Evidence Graph Consistency (EGC), a framework that constructs a local evidence graph per response and computes five structural consistency measures as hallucination indicators. Evaluated on the full question answering split of RAGTruth across six LLMs (5,767 responses), EGC reveals a consistent model-family split: graph consistency features show the expected diagnostic direction for hallucinations in Llama-2 models but exhibit systematic reversal in GPT-4, GPT-3.5, and Mistral-7B. This reversal suggests qualitatively different hallucination patterns across model families and indicates that embedding-based graph consistency cannot serve as a model-independent hallucination detection signal. 

---
# HybridCodec: Fast Dual-Stream, Semantically Enhanced Neural Audio Codec 

**Authors**: Arjun Gangwar, S Umesh  

**Link**: [PDF](https://arxiv.org/pdf/2606.06743)  

**Abstract**: The popularity of neural audio codecs as speech tokenizers has surged with the advent of Multimodal Large Language Models. New codec architectures with semantic and acoustic disentanglement have emerged. There are two main approaches to introduce semantic information into codec models: one distills semantic information from SSL representations into the first RVQ layer, while the other maintains separate streams for semantic and acoustic features. We propose HybridCodec, a unified architecture that combines both paradigms. It employs separate semantic and acoustic branches while distilling SSL representations into the semantic stream. This design ensures strong disentanglement without requiring an SSL model during inference. HybridCodec shows superior semantic specialization (RVQ-1) on in-domain test set and competitive reconstruction (RVQ-all). We demonstrate its robustness in out-of-domain and zero-shot cross-lingual settings, achieving a 3x speedup over existing dual-stream models. 

---
# Multilingual Multi-Speaker Unit Vocoders: A Systematic Analysis of Discrete Speech Representations 

**Authors**: Naman Kothari, Arjun Gangwar, Adarsh Arigala, S Umesh  

**Link**: [PDF](https://arxiv.org/pdf/2606.06740)  

**Abstract**: Discrete speech units obtained via k-means clustering of self supervised embeddings entangle phonetic, speaker, and language information, causing speaker mixing and cross-lingual interference in multilingual multi-speaker speech generation. Despite growing use in Audio LLMs and speech to speech systems, unit vocoders remain underexplored. We analyze a BigVGAN based unit vocoder, across four Indian languages. We study the interaction between cluster size and conditioning strategies using WER, speaker similarity, and unit level metrics. Results show that cluster size governs intelligibility by improving phonetic discriminability, while explicit speaker conditioning is indispensable for preventing identity collapse. Language supervision yields further gains mainly at lower cluster sizes where units remain ambiguous. Our analysis shows similar phonemes across languages collapse to the same cluster IDs at smaller inventories, with larger clusters progressively separating them. 

---
# SCOUT: Semantic scene COverage via Uncertainty-guided Traversal 

**Authors**: Junyu Mao, Sara Ayoubi, Vishnu D. Sharma, Ilija Hadžić, Matthew Andrews  

**Link**: [PDF](https://arxiv.org/pdf/2606.06721)  

**Abstract**: Robots that operate over extended periods should not merely visit space; they should progressively understand it. Yet most 3D scene graph pipelines treat perception as a post-processing stage over a fixed dataset, decoupling scene representation from the decisions that determine what is observed in the first place. We present SCOUT, an online semantic exploration framework that closes this loop by coupling active traversal with probabilistic scene graph construction. Given a prior 2D occupancy map and posed RGB-D observations, SCOUT incrementally builds an uncertainty-aware 3D scene graph whose nodes maintain fused geometry and posterior beliefs over open-vocabulary object labels, while edges encode structural relations such as on, inside, belong, and next to. These beliefs are fed back to an uncertainty-guided traversal planner, which selects viewpoints by balancing expected semantic certainty gain, geometric coverage gain, and travel cost. In this way, the robot revisits ambiguous objects when additional evidence matters and expands into unseen free space when the scene remains incomplete. The resulting system treats semantic scene completeness as an operational objective rather than a passive by-product of semantic mapping, moving toward autonomous agents that can patrol, update, and reason about evolving indoor environments with minimal human intervention. 

---
# MSAIC-Net: A Multi-Scale Attention and Imbalance-Aware Contrastive Network for ECG-Based Myocardial Substrate Abnormality Detection 

**Authors**: Canyu Lei, Fenglin Zhang, Derek Bivona, Cristiane Singulane, Jonathan Pan, Kenneth Bilchick, Amit R. Patel, Jianxin Xie  

**Link**: [PDF](https://arxiv.org/pdf/2606.06718)  

**Abstract**: Myocardial substrate abnormalities, such as myocardial scar and myocardial infarction (MI), are associated with adverse cardiovascular outcomes. Electrocardiography (ECG) provides a low-cost and widely available tool for detecting these abnormalities, but ECG-based detection remains challenging due to heterogeneous lead-dependent manifestations, high-dimensional multi-lead signals, class imbalance, and the limited interpretability of deep learning models. We propose a multi-scale attention-enhanced convolutional network (MSAIC-Net) for ECG-based myocardial substrate abnormality detection. MSAIC-Net employs parallel atrous convolutional branches to extract ECG features across multiple temporal receptive fields. %, enabling the model to capture both local and longer-range temporal patterns. Channel attention is then used to adaptively reweight informative lead-wise and feature-channel representations. To address class imbalance and improve feature separability, we introduce a novel imbalance-aware supervised contrastive learning strategy that encourages samples from the same class to form compact representations while increasing separation between abnormal and normal samples. Lead-wise permutation importance is further incorporated to quantify the contribution of each ECG lead and improve model interpretability. The proposed method was evaluated on two complementary datasets: a low-data institutional cohort from the University of Virginia (UVA) Health System for myocardial scar classification and the large-scale public PTB-XL dataset from PhysioNet for MI identification. Experimental results show that MSAIC-Net outperforms baseline models, with particularly pronounced improvements in the low-data UVA cohort. Overall, the proposed framework provides an effective and interpretable approach for ECG-based detection of myocardial substrate abnormalities. 

---
# ShallowBench: Benchmarking Generative Drug Design Models on Shallow-Pocket Targets 

**Authors**: Saket Reddy, Shiwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.06717)  

**Abstract**: While generative AI models have demonstrated remarkable success in structure-based drug design, they predominantly rely on deep binding pockets and struggle to sample effective ligands for challenging low-pocketability targets, such as the historically "undruggable" oncology targets KRAS and MYC. To address this gap, we introduce ShallowBench, a strictly curated benchmark of 5,780 shallow-pocket targets extracted from CrossDocked2020. By computing the difference between an Alpha Shape "lid" volume and the underlying protein atom voxel volume, we successfully isolated targets with low concavity while ensuring sufficient surface area for binding. Evaluating various state-of-the-art generative models reveals weaker predicted binding affinity on these low-concavity interfaces. ShallowBench therefore provides a rigorous benchmark for generative biology models and highlights the necessity of new architectural innovations or loss functions capable of navigating these challenging targets. 

---
# Does Topic Sentiment Cause Perceived Ideology? Comparing Human and LLM Annotations in Political News Articles 

**Authors**: Upasana Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2606.06715)  

**Abstract**: We ask whether topic sentiment has a causal effect on perceived political ideology, and whether the answer depends on who assigns the ideology label. Using articles from AllSides, paired with shared sentiment annotations from Llama-3.3-70b-versatile, we compare ideology labels from expert human annotators, GPT-4o-mini (baseline and finetuned), and Llama-3.3-70B. We apply Double Machine Learning (DML) and community-level mediation analysis across all four annotation paradigms. Human annotations yield no significant causal effects at the community level. Fine-tuned GPT-4o-mini achieves the highest classification accuracy (F1=72.48) and is the only annotator paradigm that produces significant community-level treatment effects and significant natural direct effects (NDEs) in mediation. We interpret this as evidence of shortcut learning: fine-tuning on ideology-labeled data causes the model to internalise a spurious sentiment--ideology coupling not operative in human judgment for this task. This coupling is structurally invisible to F1-based evaluation, with implications for the use of LLM annotations as silver labels and as proxies for human judgment in downstream causal analyses. 

---
# Data-Efficient Autoregressive-to-Diffusion Language Models via On-Policy Distillation 

**Authors**: Xingyu Su, Jacob Helwig, Shubham Parashar, Atharv Chagi, Lakshmi Jotsna, Degui Zhi, James Caverlee, Dileep Kalathil, Shuiwang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2606.06712)  

**Abstract**: We study the transformation of autoregressive models (ARLMs) into diffusion language models (DLMs). Rather than pretraining from scratch, prior work replaces the causal attention in ARLMs with bidirectional attention and then trains the resulting model using a DLM objective. However, these approaches incur two distribution shifts. First, transitioning from a next-token prediction objective to a DLM objective can discard knowledge acquired by the ARLM during training. Second, standard DLMs suffer from a train-inference mismatch, as the training loss is defined on randomly masked sequences rather than the trajectories encountered at inference produced by confidence-based decoding. To address both challenges, we introduce an On-Policy Diffusion Language Model (OPDLM) in which On-Policy Distillation (OPD) is employed for ARLM-to-DLM transformation. Specifically, OPDLM is trained via self-OPD, where the student, an ARLM with bidirectional attention, generates its own trajectories, and the teacher, the original frozen ARLM, distills its knowledge by providing target logits on these trajectories. By training directly in an on-policy manner, OPDLM eliminates the train-inference mismatch in DLMs, while distillation from the original model enhances knowledge retention from the ARLM. Empirical results demonstrate that OPDLM requires 15x to 7,000x fewer training tokens with strong performance across a wide variety of tasks. OPDLM avoids the prohibitive cost of DLM pretraining and positions DLM transformation as a form of ARLM post-training. 

---
# MMBU: A Massive Multi-modal Biomedical Understanding Benchmark to Probe the Perception Capabilities of Vision-Language Models 

**Authors**: Ryan D'Cunha, Alejandro Lozano, Xiaoxiao Sun, Daniel Vela Jarquin, Min Woo Sun, Josiah Aklilu, James Burgess, Yuhui Zhang, Ryan Nayebi, Paola Avila, Robayo, Jin Ye, Ming Hu, Zhongying Deng, Junjun He, Xin Chen, Yue Yao, Robert Tibshirani, Jeffrey J. Nirschl, Serena Yeung-Levy  

**Link**: [PDF](https://arxiv.org/pdf/2606.06696)  

**Abstract**: Vision and language models (VLMs) hold immense promise to transform biomedical imaging workflows, from detecting lesions in chest X-rays to profiling cellular features in microscopy. Realizing this potential, however, requires robust and fine-grained visual perception. Models need to correctly interpret subtle features in images, and they must do so across diverse biomedical modalities, scales, and contexts. Nevertheless, current benchmarks remain limited. To address these gaps, we introduce the Massive Multimodal Biomedical Understanding (MMBU) benchmark. It is the largest biomedical vision and language benchmark to date, covering 35 submodalities with rich structured metadata. It includes both open and closed versions of ungrounded classification, grounded classification, and object detection, enabling systematic evaluation of model performance across biological scales, clinical settings, and imaging modalities. Evaluating 15 open-weight and 2 frontier VLMs, we find that while medical adaptation provides measurable gains for some models, the high accuracy often reported on established benchmarks can mask deficiencies in visual perception and domain generalization. 

---
# The Geography of Algorithmic Judgment: LLM Intermediaries, Place Identity, and Racial Steering in Housing Search 

**Authors**: Hana Samad, Trung Lam, Christoph Mügge-Durum, Michael Akinwumi  

**Link**: [PDF](https://arxiv.org/pdf/2606.06694)  

**Abstract**: Large language models (LLMs) are rapidly assuming an intermediary role in housing search through the integration of listing platforms within conversational interfaces, mediating access to information, search, and recommendations within urban settings. We expand on prior work on racial steering in LLMs by conducting a behavioral audit of seven open-weight and closed-source LLMs across four U.S. cities, testing location recommendations across three iterative prompting conditions that progressively add lifestyle preference context and reflect fair housing paired-testing methodologies. We find that steering is an emergent behavior of the model's interpretive license rather than primarily a static property. Steering results from the interaction of a user's identity, preference articulation, and the spatial logic that a model has internalized about learned representations of place, preference, and opportunity in a given city, and how different types of users relate to it. While steering was present, it was not uniform in direction or magnitude across evaluated conditions. Preference-conditioned testing often increased or reconfigured the number of models that exhibited steering behaviors relative to baseline conditions, suggesting that LLMs may interpret what the same housing preference means differently depending on the racial identity of the user. Our findings also demonstrate that the city is not a neutral testing unit for LLM evaluation in place-based sectors, and results from one local market cannot be assumed to generalize to another. Local and domain expertise will be required in the housing sector to ensure that legal and institutional commitments to fair housing are not undermined while adopting AI tools that mediate spatial access. 

---
# HKJudge: A Legal Discourse-Annotated Corpus for Interpreting What Courts Find, How They Reason, and What They Rule 

**Authors**: Xi Xuan, Wenxin Zhang, Yufei Zhou, King-kui Sin, Chunyu Kit  

**Link**: [PDF](https://arxiv.org/pdf/2606.06679)  

**Abstract**: Court judgments are central to legal practice and jurisprudence, yet discourse analysis of Hong Kong judgments has received limited attention, owing largely to the absence of expert-annotated corpora. We introduce the Hong Kong Judgment Discourse Dataset (HKJudge), the first sentence-level expert-annotated legal discourse corpus. HKJudge includes criminal judgments across all five levels of HK's court hierarchy, comprising $\sim$290k sentences and $\sim$6.5 million tokens, fully annotated by legal linguistics experts. We design a two-tier discourse schema that captures what facts a court finds, how it reasons, and what it rules. At the sentence level, each sentence is assigned one of 26 rhetorical roles. At the span level, sentences are further annotated with three sentencing elements (charge, imprisonment term, fine). Ten legal linguistics annotators produced the annotations with an inter-annotator agreement of $\kappa = 0.8$. We formulate two tasks on HKJudge, termed rhetorical role classification and legal element extraction, and provide the first benchmark evaluation of four BERT-based models, two open-source LLMs under zero-shot and fine-tuning settings, and four commercial LLMs on both tasks. Our work demonstrates the value of sentence-level discourse annotation for modeling the structure of HK judgments and provides a rich data foundation for future work on legal judgment prediction. The HKJudge dataset and code are available at this https URL. 

---
# Inside the Visual Mind: Neuroscience-Motivated Concept Circuits for Interpreting and Steering Vision Transformers 

**Authors**: Tang Li, Yanlin Chen, Mengmeng Ma, Xi Peng  

**Link**: [PDF](https://arxiv.org/pdf/2606.06664)  

**Abstract**: Despite high accuracy, Vision Transformer (ViT) predictions can be driven by spurious cues, raising the need to understand their inner workings before safe deployment. Sparse autoencoders (SAEs) provide a promising lens for decomposing model representations into human-interpretable concepts, yet adapting SAE-based interpretation to ViTs remains challenging due to limited control over concept coverage and subjective, non-scalable feature interpretation. To fill the gaps, motivated by neuroscience-inspired principles, we propose ViSAE, a mechanistic interpretability toolbox for understanding ViT inner workings through concept circuits. ViSAE consists of three components: (1) A probing suite with 64K images and a 16K visually grounded concept vocabulary, improving concept coverage efficiency by 20x over ImageNet and interpretation accuracy by 28.7% over existing concept sets. (2) Top-down concept reading and Bottom-up circuit tracing algorithms that automatically recover ViT inner workings via concept circuits. (3) Applications for auditing and steering ViT behavior. Through concept editing, ViSAE improves the worst-group accuracy on WaterBirds by 48.2%, outperforming existing methods by 23.8%. Our data and code: this https URL. 

---
# CAF-Gen: A Multi-Agent System for Enriching Argumentation Structures 

**Authors**: Jakub Bąba, Jarosław Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2606.06646)  

**Abstract**: Formalizing complex reasoning from natural text is one of the central challenges in computational linguistics. It requires systems to understand not just keywords but also the context and complex reasoning embedded in a text. Current Argument Mining (AM) techniques identify basic claims and premises, yet they often struggle to capture the richer structural information required by advanced schemas such as the Carneades Argumentation Framework (CAF), which incorporates features such as premise types, proof standards, and argument schemes. We address this limitation by introducing CAF-Gen, an automated multi-agent framework designed to enrich shallow argument structures into CAF-compliant argument models. By employing an iterative Creator-Reviewer pipeline, a creator agent's output is validated by a critical agent to ensure structural integrity. This multi-agent collaboration is crucial for mitigating the structural instability typical of single-pass generative models. Our experiments demonstrate that the iterative feedback loop improves the quality of the resulting data and achieves strong alignment with the original annotations, while producing structurally richer models. Our findings show that the multi-agent system can overcome the limitations of single-pass generation, providing a robust methodology for the automated modeling of formal argumentation. 

---
# How Language Models Fail: Token-Level Signatures of Committed and Persistent Reasoning Failures 

**Authors**: Tanvi Thoria, Kiana Jafari, Marc R. Schlichting, Mykel J. Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2606.06635)  

**Abstract**: Failures in language model reasoning emerge through distinct processes that leave identifiable signatures in the reasoning trace. We characterize these failures using token-level uncertainty signals, finding they arise through two empirically distinguishable processes. The first is committed failure, in which a model locks onto an incorrect reasoning path early in its trace. A central diagnostic signature is the commitment point, beyond which considering additional tokens hurt rather than help failure detection. In the second, persistent uncertainty, uncertainty instead accumulates throughout, and the full trace is needed to best distinguish failing from successful completions. These signatures reproduce across 23 model-dataset configurations, with the framework's falsifiable predictions holding in 20 of 23 cases, well above chance across both failure modes. Finally, we demonstrate our failure mode framework has direct implications for self-consistency, identifying when uncertainty signals complement it and when it can be selectively skipped. These results offer a foundation for understanding when LLM reasoning failures become detectable and for adapting detection strategies accordingly. 

---
# What Matters When Cotraining Robot Manipulation Policies on Everyday Human Videos? 

**Authors**: Richard Li, Aditya Prakash, Andrew Wen, Saurabh Gupta, Yilun Du, Pulkit Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2606.06627)  

**Abstract**: Human video datasets used for cotraining robot manipulation policies largely consist of curated demonstrations where motions are orchestrated to resemble robot behavior and 3D hand poses are captured with specialized hardware. A more plentiful source of data is everyday Internet video, but it is an open question what factors enable transfer from such videos to robots. We investigate this using a new dataset of 532 human videos with 28 hours of high-quality triangulated hand labels and natural motions. We find that hand pose quality affects transfer, but even with accurate hands, the inherent motion gap hinders transfer unless the vision and policy networks specialize to each embodiment. Our cotraining recipe yields consistent improvements, with an absolute success rate gain of $29.7\%$ in the low-robot-data regime across six manipulation tasks. 

---
# ChronoForest: Closed-Loop Multi-Tree Diffusion Planning for Efficient Bridge Search and Route Composition 

**Authors**: Jungmin Seo, Jaesik Park  

**Link**: [PDF](https://arxiv.org/pdf/2606.06618)  

**Abstract**: How can we plan long-horizon routes that reach designated goals, visit required waypoints, and remain short when only short-horizon offline trajectories are available? This problem matters in offline navigation because collecting sufficiently rich long-horizon data is difficult, yet real agents must still solve long-range tasks with route-level efficiency rather than mere feasibility. The difficulty is twofold: at the microscopic level, composing many short-horizon segments creates a trade-off between search cost and path quality, while at the macroscopic level, waypoint ordering requires comparing pairwise travel costs among start, goal, and waypoint anchors that are unknown before planning and increasingly unreliable when estimated only from long-range temporal distance.
In this paper, we propose ChronoForest, a closed-loop planning system that couples local bridge search and online route re-solving through an anchor-chaining tree diffusion planner and an online multi-tree orchestrator. ChronoForest uses temporal distance for short-range guidance and node evaluation, while using search-time bridge evidence to validate long-range anchor connectivity and repeatedly re-solve the route. On OGBench AntMaze-Stitch, ChronoForest achieves 99.8%, 99.3%, and 99.5% success on the medium, large, and giant splits and improves giant-stitch success by up to 34.5 points over prior reported diffusion-based results. On Hamiltonian route-composition benchmarks, online re-solving corrects poor temporal orderings and improves route quality while remaining substantially cheaper than exhaustive planning. 

---
# FIGMA: Towards FIne-Grained Music retrievAl 

**Authors**: Nishit Anand, Ashish Seth, Sreyan Ghosh, Dinesh Manocha, Ramani Duraiswami  

**Link**: [PDF](https://arxiv.org/pdf/2606.06615)  

**Abstract**: Retrieving music using natural language descriptions has improved with contrastive audio-text models such as CLAP, but current systems remain limited to coarse semantic queries. When descriptions specify fine-grained musical attributes such as tempo, key, chord progression, or rhythmic structure, existing models often fail to retrieve the correct audio. We show that this limitation stems from the contrastive learning objective itself: despite being trained on long captions, CLAP-based models effectively utilize only the first few tokens, discarding much of the information encoded in detailed prompts. Then, we propose FIGMA (FIne-Grained Music RetrievAl), a multi-view contrastive architecture that addresses this limitation by jointly optimizing global audio-text alignment and frame-level, token-wise alignment. This design enables FIGMA to capture both high-level semantic context and fine-grained musical attributes within a unified representation space. Moreover, we formalize the task of Fine-Grained Music Retrieval and construct Fine-Grained Music Caption dataset (FGMCaps), a large-scale dataset of 380K music-caption pairs for training along with a 10K test set, both annotated with tempo, key, chord progression, beat count, as well as genre and mood. Extensive experiments demonstrate that FIGMA consistently outperforms existing CLAP-based music retrieval models across multiple music retrieval benchmarks, including out-of-domain evaluations, with relative improvements of up to 73.3%. 

---
# Re-Centering Humans in LLM Personalization 

**Authors**: Lechen Zhang, Jiarui Liu, Tal August  

**Link**: [PDF](https://arxiv.org/pdf/2606.06614)  

**Abstract**: Despite growing interest, most evaluations of large language models' (LLMs') personalization abilities have relied on synthetic data. It remains unclear how well current personalization systems work for real users. In this paper, we study the gap in LLM personalization performance when using synthetic versus human data. We collect human conversations (550 conversations) and judgments across three stages of personalization: extracting user attributes from conversations (5,949 judgments), pairing relevant attributes with new prompts (11,919), and incorporating relevant attributes into a personalized response (1,101). Incorporating human data reveals system limitations at each stage. Models struggle to extract attributes from human conversations, disagree with human judgments on relevant attributes, and generate personalized responses that humans judge no better than generic responses (though that LLM judges widely rate as better). We introduce two lightweight training-based interventions that shift automated personalization evaluation closer to human data in our first two stages. However, in our third stage we find that learned reward models achieve only modest correlation with human ratings, suggesting that human-aligned personalization quality judgments are difficult to model directly. Our collected data provides a foundation for studying how models should extract, select, and incorporate user information in ways that humans find useful. 

---
# Direct 3D-Aware Object Insertion via Decomposed Visual Proxies 

**Authors**: Jingbo Gong, Yikai Wang, Yushi Lan, Yuhao Wan, Ziheng Ouyang, Rui Zhao, Ming-Ming Cheng, Qibin Hou, Chen Change Loy  

**Link**: [PDF](https://arxiv.org/pdf/2606.06601)  

**Abstract**: Object insertion aims to seamlessly composite a reference object into a specified region of a background image. Recent diffusion-based methods achieve high visual quality but formulate insertion as a simple 2D inpainting task, providing no explicit control over the object's 3D pose and limiting their practical applicability. We propose DIRECT (Decomposed Injection for Reference Composition and Target-integration), a novel framework that integrates interactive pose manipulation with high-fidelity 2D image synthesis to enable pose-controllable object insertion. Our method decomposes the insertion conditions into three complementary components: appearance guidance capturing visual details from the reference object, geometry guidance derived from the user-adjusted 3D proxy, and context guidance from the target background. By injecting them through separate pathways, DIRECT avoids feature entanglement and simultaneously preserves reference appearance, follows the user-specified pose, and adapts the object to the target scene. We also introduce an automated data construction pipeline to improve the diversity and quality of training data. Experiments show that DIRECT outperforms previous methods in both geometric controllability and visual quality. 

---
# Generative Models Erode Human Temporal Learning Through Market Selection 

**Authors**: Wenjun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2606.06572)  

**Abstract**: We argue that modern generative models create structural risks for knowledge and cultural production at current, sub-AGI capability levels. We define Human Temporal Learning (HTL) as path-dependent knowledge accumulation through sustained engagement with problems over time. Generative outputs increasingly resemble HTL-intensive work in surface features, so verifying whether a given output reflects genuine human learning grows costly relative to its expected benefit. Once verification loses economic justification, evaluators reward outputs regardless of production mode, and producers who invested years of learning compete on price against outputs that cost almost nothing to generate. We call this pathway value collapse and formalize it through a costly-inspection framework. Cross-domain evidence from academic publishing, legal practice, content platforms, and software security maps onto four stages of verification erosion. Alignment success is orthogonal. Better-aligned models narrow observable gaps between human and AI outputs, making source verification harder and intensifying competitive pressure against HTL-intensive work even when individual AI outputs improve. 

---
# MalTree: Tracing Malware Evolution from Embeddings at Scale 

**Authors**: Akash Amalan, Georgios Smaragdakis, Tom J. Viering  

**Link**: [PDF](https://arxiv.org/pdf/2606.06570)  

**Abstract**: Malware detection remains largely reactive: machine learning models trained on known samples degrade as threats evolve. Understanding evolutionary relationships among malware families can inform proactive defense, but traditional reverse engineering can take months to years to uncover such lineage relationships. We propose MalTree, a framework that applies bioinformatics inspired phylogenetic techniques (UPGMA and Neighbor-Joining) at scale to model malware evolution automatically using structural, behavioral, and image-based features. We introduce temporal validation using VirusTotal timestamps to assess whether inferred trees reflect actual evolutionary order. MalTree achieves 87% temporal consistency, indicating that inferred evolutionary relationships closely align with real-world emergence timelines. Our analysis shows that some families mutate over 10 times faster than others, suggesting that detection strategies should be tailored to family-specific evolutionary tempos. Case studies, including the Mirai botnet, confirm that inferred relationships from our phylogenetic tree align with documented threat intelligence. Our framework provides a foundation for shifting malware analysis from sample-by-sample classification toward lineage-aware evolutionary modeling. 

---
# NTILC: Neural Tool Invocation via Learned Compression 

**Authors**: Andrew Krikorian, Yayuan Li, Jason J. Corso  

**Link**: [PDF](https://arxiv.org/pdf/2606.06566)  

**Abstract**: Agentic tool-calling language models depend on large registries of callable APIs, functions, and local actions. Placing full tool specifications directly in the prompt incurs a cost that scales linearly with the size of the tool registry, rapidly consuming the context budget. As the registry grows, this leads to higher latency and degrades selection accuracy, particularly due to interference from irrelevant tools. We overcome these limitations by introducing NTILC, a neural tool selection and invocation framework that replaces in-context registry look-up with learned latent retrieval. NTILC maps both user intent and tool specifications into a shared embedding space, enabling tool selection via external retrieval rather than in-context lookup. The language model is conditioned only on the selected tool schema, allowing for precise, constrained argument generation. Central to our approach is a signature-aware composite objective, which augments semantic similarity with constraints derived from tool signatures (e.g., argument schema, type compatibility, and return types). By combining Circle Loss with a Functional Margin Loss, the model enforces separation between tools that are semantically similar but incompatible under their execution signatures. We evaluate NTILC on public tool-selection and function-calling datasets and report context token usage, retrieval accuracy, and selection latency metrics. Across these settings, NTILC reduces context window consumption by over 95% and inference latency by up to 74% compared to long-context ICT baselines. 

---
# WAV: Multi-Resolution Block Residual Routing for Deep Decoder-Only Transformers 

**Authors**: Kehan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.06564)  

**Abstract**: Residual connections are central to training deep Transformers, but standard PreNorm residual streams aggregate sublayer updates with fixed unit weights. Recent Attention Residuals replace this fixed accumulation with content-dependent depth-wise routing, and Block Attention Residuals make the mechanism efficient by routing over block-level residual summaries. However, a single block summary stores only the low-frequency total residual displacement inside a block, discarding directional structure such as attention-vs-MLP imbalance and early-vs-late block dynamics. We propose WAV v1, a lightweight multi-resolution residual routing method for decoder-only Transformers. Instead of representing each block only by its accumulated residual sum, WAV v1 augments every block with two directional detail bases: a phase basis that contrasts attention and MLP updates, and a split basis that contrasts early and late sublayer updates. These bases are routed together with standard block summaries through the same depth-wise softmax mixer, while negative detail-source initialization and detached RMS matching stabilize training. On character-level TinyStories and Text8 language modeling, WAV v1 shows a clear depth-dependent benefit. Although it is not consistently beneficial at 12 layers, it becomes competitive at 24 layers and outperforms all baselines at 48 layers. At 48 layers, WAV v1 reduces validation loss relative to Block AttnRes from 0.4960 to 0.4738 on TinyStories and from 0.9363 to 0.9305 on Text8, with negligible additional parameters. These results suggest that directional residual details, not only block-level sums, are important for scaling residual routing in deeper Transformers. 

---
# AI-Driven Test Case Generation from Natural Language Requirements: A Survey of Techniques and Research Gaps 

**Authors**: Orimoloye Folorunsho, Hassan Reza  

**Link**: [PDF](https://arxiv.org/pdf/2606.06563)  

**Abstract**: Software testing is critical for verifying that systems meet specified requirements, yet remains among the most time-consuming and expensive activities in development. Requirements-based test generation allows test cases to be derived early from requirements artifacts, but generating them directly from natural language is challenging due to inherent ambiguity and imprecision. Recent advances in AI, natural language processing (NLP), and large language models (LLMs) have made automating this pipeline increasingly feasible, while introducing new risks including hallucination, reduced traceability, and inconsistent evaluation. This survey addresses four research questions: what AI and NLP techniques have been proposed for generating test cases from natural language requirements; what tools and frameworks support these approaches; how generated test cases are evaluated; and what research gaps remain. Following Kitchenham and Charters' systematic review guidelines, we searched major scholarly databases spanning 2000-2025 and, after applying strict inclusion criteria, identified 21 primary studies. The literature is organized into three evolutionary eras, revealing that no existing approach simultaneously satisfies six key quality dimensions: automation, ambiguity handling, domain applicability, traceability, evaluation thoroughness, and hallucination control. The survey makes three main contributions: a three-era evolutionary synthesis of AI-based test generation; a six-criteria gap analysis showing no current approach fully addresses all quality dimensions; and four actionable research guidelines targeting hallucination, traceability, complexity sensitivity, and compliance. 

---
# MacArena: Benchmarking Computer Use Agents on an Online macOS Environment 

**Authors**: Victor Muryn, Maksym Shamrai, Sofiia Mazepa, Yehor Khodysko  

**Link**: [PDF](https://arxiv.org/pdf/2606.06560)  

**Abstract**: Computer-use agents (CUAs) operate graphical user interfaces (GUIs) through vision and control primitives, and their capabilities have advanced rapidly, driven in part by standardized online evaluation benchmarks such as OSWorld, which serve both as evaluation tools and as training environments for reinforcement learning. However, macOS remains underserved in this landscape: the only existing benchmark, macOSWorld, covers a narrow slice of first-party applications with simpler tasks, and runs on x86 virtual machines incompatible with Apple Silicon. We introduce MacArena, a benchmark of 421 manually verified tasks spanning 50 applications that combines a curated port of OSWorld tasks, content sourced from macOSWorld, and 49 new macOS-native tasks, all running on Apple's native Virtualization framework on Apple Silicon. We argue that macOS presents distinct GUI challenges beyond what Linux-based benchmarks capture, and our evaluation supports this claim: strong model performance on existing benchmarks can reflect familiarity with task distributions rather than genuine cross-platform GUI competence. Notably, model rankings invert between ported and macOS-native tasks, with a leading model trailing by over 26% on the MacArena subset, suggesting that macOS poses a genuinely harder environment for current GUI agents. 

---
# IRAF: Interference-Resilient Adaptive Fusion for Noise-Robust End-to-End Full-Duplex Spoken Dialogue Systems 

**Authors**: Tao Zhong, Jiajun Deng, Nikita Kuzmin, Yinke Zhu, Tianxiang Cao, Tristan Tsoi, Zhili Tan, Simon Lui, Xunying Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.06559)  

**Abstract**: Full-duplex spoken dialogue models allow voice agents to listen and speak concurrently, enabling natural interaction with real-time overlap. However, end-to-end dual-channel models that jointly encode user and agent streams may degrade in realistic acoustic environments: interfering speakers leaking into the user microphone can be encoded as part of the user query, corrupting the LLM's conditioning and causing unstable turn-taking and reduced response quality. We propose Interference-Resilient Adaptive Fusion (IRAF), a lightweight, streaming-compatible module that modulates the contribution of user audio to the LLM frame by frame. IRAF predicts a scalar reliability gate from target-speaker and user audio embeddings and rescales user representations before fusion with agent embeddings. Experiments on MS-MARCO and InstructS2S-200K show consistent gains in response quality and full-duplex interaction under interfering-speaker conditions. 

---
# Multi-Scale Feature Attention Network for Polymer Classification using THz Dual-Comb Spectroscopy 

**Authors**: Roshni Mahtani, Ilán Carretero, Laura Monroy, Aldo Moreno-Oyervides, Oscar Elías Bonilla-Manrique, Rocío del Amor  

**Link**: [PDF](https://arxiv.org/pdf/2606.06554)  

**Abstract**: Reliable polymer identification is essential for ensuring the quality and safety of recycled plastics, yet conventional sorting and spectroscopic techniques often struggle to deliver robust discrimination. Terahertz Dual-Comb Spectroscopy (THz-DCS) offers a promising alternative, providing rapid, high-resolution, and non-destructive measurements. In this work, we leverage THz-DCS to classify 12 types of polymers, including pure polymers, multilayer films, commercial blends, and biopolymers. To handle the complexity of these spectral signals, we propose the Multi-Scale Feature Attention Network (MSFAN), a novel deep learning architecture tailored for THz-DCS data. The framework integrates feature gating for signal recalibration and multi-scale parallel convolutions to capture diverse frequency patterns. These features are further refined through cross-feature attention and attention pooling, enabling the model to intrinsically highlight the most informative THz regions. MSFAN consistently outperforms state-of-the-art models, reaching a classification accuracy of 85.2%. This study demonstrates the potential of combining THz-DCS with deep learning techniques for effective, scalable, and interpretable polymer classification. 

---
# Geometric Second-Order Feature Correlation Learning for Self-Supervised Speech Emotion Recognition 

**Authors**: Shuanglin Li, Ruxiao Qian, Siyang Song  

**Link**: [PDF](https://arxiv.org/pdf/2606.06550)  

**Abstract**: Self-supervised learning (SSL) yields powerful, context-rich representations for speech emotion recognition (SER), yet aggregating these representations into holistic descriptors remains a bottleneck. Conventional first-order aggregation implicitly assumes feature independence, which overlooks the latent Riemannian geometry and discards higher-order relationships essential to the representational power of the backbone. To address this problem, this paper proposes a novel Second-Order Correlation (SOC) layer. Instead of treating features in isolation, SOC models feature correlations as covariance descriptors to capture synergistic co-occurrence patterns, which serve as discriminative signatures for robust emotion recognition. By mapping these descriptors from the Riemannian manifold to a Euclidean tangent space through Log-Euclidean mapping (LEM), the proposed method preserves geometric integrity while enabling direct linear discriminative learning. Extensive experiments on the ESD and RAVDESS datasets demonstrate that SOC recovers discriminative information lost in first-order pooling and effectively aggregates high-dimensional SSL features. 

---
# FAIR-Calib: Frontier-Aware Instability-Reweighted Calibration for Post-Training Quantization of Diffusion Large Language Models 

**Authors**: Haoyu Huang, Linlin Yang, Sheng Xu, Boyu Liu, Guodong Guo, Zhongqian Fu, Hang Zhou, Baochang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.06547)  

**Abstract**: Diffusion Large Language Models (dLLMs) refine tokens iteratively but commit them irreversibly, leading to a "stability lag" where early decisions remain fragile even after being written. We reveal that Post-Training Quantization (PTQ) error easily flips these borderline decisions at the write frontier, which are then permanently locked in and amplified. To address this, we propose Frontier-Aware Instability-Reweighted Calibration (FAIR-Calib), a two-stage PTQ framework for dLLMs. Stage I probes a full-precision teacher to estimate a position prior that combines frontier hits and masked-stage reliability. Stage II performs off-policy, layer-wise calibration by minimizing a reweighted hidden-state MSE, effectively prioritizing the protection of fragile frontier states without requiring expensive end-to-end diffusion rollouts. We further theoretically justify our weighted objective as a surrogate for output KL divergence. Empirically, FAIR-Calib consistently outperforms state-of-the-art baselines on LLaDA and Dream (W4A4), significantly reducing frontier decision flips and suppressing post-commit mismatches across diverse benchmarks. 

---
# Queen-Bee Agents: A BeeSpec-Centered Architecture for Governed Enterprise MCP Orchestration 

**Authors**: Dutao Zhang, Liaotian  

**Link**: [PDF](https://arxiv.org/pdf/2606.06545)  

**Abstract**: Enterprise agent systems increasingly need to connect large language models to private tools, internal knowledge, and Model Context Protocol (MCP) interfaces. In this setting, raw task capability is insufficient: organizations also require policy enforcement, tenant-scoped isolation, and execution that remains within explicit operational boundaries. We present Queen-Bee, a governed multi-agent architecture in which a Queen control plane retrieves capabilities, plans task-scoped execution, and compiles a structured BeeSpec that is executed by specialized Bee agents under constrained tool access. We implement a working prototype with tenant-scoped MCP connectors, audit-backed execution-time governance, retrieval-driven weak incubation, and multiple provisioning backends. We evaluate the system on 59 enterprise-style tasks spanning governance-sensitive requests, retrieval-driven provisioning, scoped local execution, and chemistry workflow integration. The retrieval-driven Queen-Bee variant achieves a task success rate of 0.964, zero governance failures, and substantially better scoped execution quality than both a static Queen-Bee baseline and a permissive single-agent baseline. We further show a multi-Bee chemistry workflow with explicit approval gating and a concrete top-3 shortlist grounded in real upstream evidence and screening artifacts. Additional comparisons with hybrid retrieval and LLM-guided provisioning show that richer provisioning backends are viable but do not outperform the lightweight structured retriever on the current small, highly structured capability registry. The results provide prototype-level systems evidence rather than a production deployment study, and suggest that enterprise agent platforms should be evaluated not only by capability, but also by governed provisioning, isolation behavior, scoped execution quality, and artifact-aware workflow coordination. 

---
# Coordinated optimization of departure sequencing and section-track allocation in railway short-term concentrated departure scenarios based on qubo and hybrid quantum algorithms 

**Authors**: Xiaobin Li, Yanbin Gao, Weiguang Wang, Xuechen Liang  

**Link**: [PDF](https://arxiv.org/pdf/2606.06543)  

**Abstract**: This study examines the coordinated optimization of departure sequencing and section-track allocation in railway short-term concentrated departure scenarios. A quadratic unconstrained binary optimization (QUBO) model is formulated to represent departure-position assignment and section-track selection within a unified binary framework. Because the quality of a dispatching scheme depends on time-dependent operational interactions that cannot be fully captured by a static combinatorial model, a simulation-based evaluation layer is introduced to assess section occupation, intermediate-station waiting, platform-capacity pressure, running-time fluctuations, and delay propagation. Within this layered framework, conventional heuristics, quantum-inspired algorithms, and hybrid algorithms are compared on the same decision structure. The results show that the QUBO model can generate feasible candidate schemes after decoding, while the simulation layer clearly differentiates the operational performance of the competing algorithms under both normal and disturbed conditions. In the tested scenarios, QPSO-QAOA performs best under normal conditions, and the quantum-enhanced methods reduce comprehensive cost by 4.28\%--26.26\% and total delay by 4.37\%--24.25\% on average under dynamic conditions relative to their conventional counterparts. These findings suggest that the integration of QUBO-based modeling and simulation-based evaluation provides a useful methodological framework for railway short-term concentrated departure scheduling, although validation with real operational data remains necessary. 

---
# Synthetic Benchmarks Overstate Forward-Forward Scaling: Real-Data Limits of Layer-Local Training 

**Authors**: Yucheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.06539)  

**Abstract**: Forward-Forward (FF) learning [Hinton, 2022] replaces backpropagation with strictly layer-local goodness updates. Recent FF-CNN work has narrowed the gap to BP on 32x32 benchmarks, raising the question of whether layer-local training is becoming a viable alternative at realistic scale. To probe this rigorously, we develop DTG-FF -- dynamic temperature goodness, decoupled normalization, and multi-layer fusion -- as an instrument that sets FF-family state of the art across nine real-data benchmarks (91.8% CIFAR-10 and the first FF baseline at ImageNet-100 224x224), and use it to audit how far layer-local training actually scales.
(1) Real-data scaling. Under identical recipe and backbone, an architecture-matched BP-DeepSup baseline beats DTG-FF by 2.40/5.93 pp on CIFAR-10/CIFAR-100, and the gap widens with class count. At 224x224 the same instrument reaches only 49.4% -- the first FF baseline at this scale, versus typical BP above 75% [Tian et al., 2020] -- exposing a real-data ceiling invisible at 32x32.
(2) Synthetic vs. real K-conflict. DTG-FF increasingly outperforms BP as class count K grows on synthetic teacher-student tasks, yet on real images the FF-BP gap reverses sign and widens with K. A within-dataset CIFAR-100 coarse vs. fine probe isolates label-hierarchy from image distribution: synthetic K-sweeps confound output dimensionality with fine-grained discrimination difficulty and thereby overstate FF transferability.
(3) Systems audit. FF can be implemented without storing depth-wide activations, but on commodity 8 GB hardware standard BP+gradient-accumulation reaches 4.18 GB / 157 imgs/s versus DTG-FF's 7.90 GB / 138 imgs/s, so a memory-based justification for FF at this scale is not supported under fair baselines. 

---
# Attention-Guided Autoencoder Fusion for Insulator Defect Detection Using UAV Transmission-Line Imaging 

**Authors**: Malak Allam, Khaled Shaban, Ali Hamdi  

**Link**: [PDF](https://arxiv.org/pdf/2606.06536)  

**Abstract**: Automated defect detection in high-voltage transmission-line insulators remains challenging due to severe class imbalance, large scale variation, and the small spatial extent of defect instances in Unmanned Aerial Vehicle (UAV) imagery. To address these challenges, this paper proposes AE-YOLO, an Attention-Guided AutoEncoder-Enhanced YOLO framework for robust insulator defect detection. The architecture integrates lightweight bottleneck autoencoders within a Feature Pyramid Network-Path Aggregation Network (FPN-PAN) neck. This preserves anomaly-sensitive information during multi-scale feature fusion. Convolutional Block Attention Modules (CBAM) are used throughout the backbone, enhancing feature discrimination and suppressing background interference. The framework also introduces a variance-maximizing autoencoder regularization strategy, which encourages diverse, defect-discriminative latent representations. The network trains using a unified objective that combines focal loss, Complete IoU (CIoU) loss, and autoencoder regularization to address foreground-background imbalance and improve localization accuracy. During inference, Weighted Boxes Fusion (WBF) combines predictions from YOLOv8, YOLOv10, and YOLO11. An autoencoder-guided confidence boosting mechanism improves sensitivity to rare defect categories. Experiments on the Insulator-Defect Detection dataset show that AE-YOLO with an EfficientNetV2 backbone achieves 95.10 percent mAP at 0.5, 96.40 percent precision, and 93.80 percent recall. This performance surpasses the strongest YOLO-family baseline by 5.0 points in mAP at 0.5 and 6.7 points in recall. These results confirm the effectiveness and adaptability of the framework. The model is a practical and scalable solution for UAV-based transmission-line inspection and defect monitoring. 

---
# Attention Consistent Longitudinal Medical Visual Question Answering Guided by Vision Foundation Models 

**Authors**: Jialin Wu, Qianru Zhang, Georges El Fakhri, Xiaofeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.06534)  

**Abstract**: Longitudinal medical visual question answering (VQA) requires reasoning about anatomical differences between an image of a current time point and an image of a referred time point. We propose an attention-guided encoder-decoder for this task with chest X-rays. Instead of conventional direct contrast, we propose to include a lightweight affine registration module to reduce nuisance motion by co-registering the current image to the reference image with a small registration regularizer. The registered image pair is fed into the image encoder, followed by a frozen DINO-based mask generator and a trainable adaptive mask generator to produce masks applied to the original image pairs. The masked image pairs are again fed into the image encoder and concatenated with text features as the input to a multimodal transformer-based decoder to generate final answers. To facilitate learning stabilization and clarify the change signal, inspired by DINO-v3, we include additional auxiliary objectives, including a mask rebuilding loss, a pairwise Gram-style consistency loss, and a KoLeo uniformity loss, which enhances the geometry of the representation. On the Medical-Diff-VQA benchmark, the model delivers strong BLEU, ROUGE-L, CIDEr, and METEOR scores while offering intrinsic interpretability through the shared saliency mask. These results support saliency-conditioned generation with mild pre-alignment as a principled framework for longitudinal reasoning in medical VQA. Our training strategy also illustrates the potential of a paradigm in utilizing image foundation models in biomedicine: optimizing both supervised and unsupervised learning objectives simultaneously. 

---
# Agentic Large Language Models for Automated Structural Analysis of 3D Frame Systems 

**Authors**: Ziheng Geng, Ian Franklin, Santiago Martinez, Jiachen Liu, Yunhe Zhao, Minghui Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.06525)  

**Abstract**: Large language models (LLMs) have emerged as powerful foundation models with strong reasoning capabilities across domains. Beyond reactive text generation, agentic LLMs enable autonomous workflow execution through modular task decomposition and coordinated tool use. In structural engineering, recent efforts have developed agentic LLMs for automated analysis of plane frames. However, their extension to 3D frames remains underexplored due to challenges in irregular geometric representation, topological consistency, and long-horizon reasoning. This paper proposes an agentic LLM framework for automated structural analysis of 3D frames from natural language inputs. Irregular 3D frames are represented by projection onto a 2D plan, where orthogonal gridlines define spatial coordinates and a matrix of number of stories encodes vertical extrusion of each grid cell. Building on this representation, the framework establishes a multi-agent pipeline: a problem analysis agent parses input into structured JSON; a floor decomposition agent derives the spatial layout of each floor; the 3D geometry is assembled by node, girder, slab, and column agents; support and load agents assign boundary and loading conditions, and code translation agents generate executable SAP2000 script. Evaluated on ten representative 3D frames, the proposed framework achieves an average accuracy of 90% across repeated trials, demonstrating consistent and reliable performance. 

---
# P-Cast Precision in FP8 Attention: Sink-Induced Collapse and the Optimality of S=2^8 

**Authors**: Reed Lau  

**Link**: [PDF](https://arxiv.org/pdf/2606.06521)  

**Abstract**: FP8 (E4M3) acceleration for attention computation offers significant throughput gains, but the 3-bit mantissa introduces precision challenges when the softmax probability matrix P is cast to FP8 before the P*V matrix multiplication. We analyze two implementation choices that affect output precision under the Attention Sink phenomenon: (1) the KV block iteration order, and (2) the static scaling factor applied to P before casting. We show that forward KV iteration causes "P-collapse" -- to leading order, a fraction Phi(Delta + delta_k - 6.93 - ln S) of non-sink P values underflow to zero, where the small shift delta_k ~ 1 (for k_sink = 4) is the expected within-sink-block score maximum -- and that reverse iteration removes it, with a zero-underflow guarantee when reverse is combined with S = 256. We further give a constructive characterization of S = 256 = 2^8 as the static scale that simultaneously satisfies (i) bit-exact IEEE 754 scaling, (ii) the lower envelope of a sawtooth function dp(S) over the E4M3 number line (dp = 2^-4, the minimum worst-case quantization step), and (iii) the maximum normal-range coverage among bit-exact (2^k) scales (a non-bit-exact scale such as 448 attains slightly higher coverage). Both optimizations are already deployed in FlashAttention-3/4 on engineering grounds; our contribution is a quantitative account of why these choices are good and a closed-form threshold Delta_c = 6.93 + ln S - delta_k for predicting kernel-level precision loss. Kernel-faithful experiments (Q, K, V in FP32 to isolate the P-cast effect) show 3-10x MSE improvement at moderate sink strengths, and paired tests confirm both fixes saturate to the same precision floor when combined. 

---
# DxPTA: An Architecture Design Space Exploration with Optical Dataflow-guided Strategy for HW/SW Co-Design of Photonic Transformer Accelerators 

**Authors**: Rachmad Vidya Wicaksana Putra, Solomon Micheal Serunjogi, Mahmoud Rasras, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2606.06515)  

**Abstract**: Transformer-based networks have emerged as prominent AI models with state-of-the-art performance, which potentially pave the way toward artificial general intelligence (AGI). However, their large sizes still hinder their efficient implementation, thus highlighting the need for alternate solutions to enable their energy-efficient acceleration. Recently, state-of-the-art works propose photonic transformer accelerators (PTAs) with significant speedup and energy efficiency improvements over the conventional electronic accelerators. However, their PTA architectures are developed without considering the application constraints (e.g., area, power, energy, and latency). Moreover, their manual design approach also requires huge design time to determine a suitable architecture for the targeted application, hence making this approach not scalable. To address these limitations, we propose DxPTA, a novel design space exploration methodology for enabling efficient hardware/software co-design of the appropriate PTA architecture that meets all constraints. It is achieved by (1) identifying the PTA architecture parameters based on the coherent optical dataflow; (2) analyzing the impact/significance of the parameters; and (3) leveraging this analysis for devising a constraint-aware architecture search algorithm. Experimental results show that, our DxPTA can find the appropriate PTA architectures for different transformer-based models (i.e., DeiT-T/S/B and BERT-B/L). It achieves up to 26mm^2 area, 4.8W power, 39mJ energy, and 6ms latency, for constraints of 50mm^2 area, 5W power, 50mJ energy, and 10ms latency; with 15.2x faster searching time than the exhaustive approach. These results demonstrate the potential of DxPTA methodology for enabling efficient PTA designs for diverse AGI-based applications. 

---
# FP8 is All You Need (Part 1): Debunking Hardware FP64 as the HPC Holy Grail 

**Authors**: Satoshi Matsuoka  

**Link**: [PDF](https://arxiv.org/pdf/2606.06510)  

**Abstract**: Conventional HPC dogma holds that native hardware FP64 silicon is the irreducible foundation of scientific computing -- the "holy grail" of double-precision simulation. This paper argues the dogma is wrong: on AI-optimised GPUs of the B300 generation and beyond, abundant FP8 tensor throughput combined with the Chinese Remainder Theorem-based Ozaki Scheme II recovers memory-roof execution at full FP64 accuracy across the canonical HPC kernel spectrum. NVIDIA's Blackwell Ultra (B300) collapses native FP64 to ~1.3 TFLOPS -- a 31x regression from the B200 -- rendering even memory-bound kernels (SpMV, GEMV, stencils) compute-bound. We make four contributions. First, a unified analytic model, the Tensor-Memory Equilibrium (TME) model, augmenting the Roofline with a compute multiplier alpha, a bandwidth multiplier beta, and a reconstruction latency gamma. Second, we identify register-level fusion as the mechanism driving beta -> 1, making emulation essentially free behind the memory wall. Third, we project that Ozaki II vaults emulated FP64 from the ~1 TFLOPS native floor to ~500 TFLOPS (B300) and ~400 TFLOPS (Rubin R200), exceeding even B200's native FP64 ceiling by over an order of magnitude in the compute-bound regime while matching the memory roof in the bandwidth-bound regime. Fourth, against an H100 baseline, Ozaki II matches or exceeds H100 on every workload studied, versus the up-to-50x regression that B300 native FP64 imposes. Combined with a companion FFT analysis (Kulisch fixed-point reconstruction on the surviving INT32 pipe) and FP32+Kahan reductions reported in the companion Part(2) paper, every surveyed kernel class on B300 reaches the memory roof at full FP64. The evidence supports the title's claim: FP8, with Ozaki II and Kulisch escape routes, is all one needs for production HPC; native FP64 silicon is no longer the holy grail it has been taken to be. 

---
# Which Anatomy Matters Under Limited Labels? A Data-Efficient Anatomy-Aware Benchmark for Cardiac Pathology Prediction 

**Authors**: Himanshu Singh  

**Link**: [PDF](https://arxiv.org/pdf/2606.06509)  

**Abstract**: Numerous medical imaging problems must be solved under limited labels and constrained compute, yet it remains unclear whether performance gains are driven mainly by more expressive models or by better representation of clinically meaningful anatomy. We study this question through a low-data anatomy-aware benchmark for 5-class cardiac pathology prediction on the public ACDC MRI dataset. Using segmentation-derived patient descriptors from the right ventricle, myocardium, and left ventricle, we compare anatomy-specific and multi-structure representations across linear, kernel, and tree-based classifiers. We find that under limited label settings, representation dominates complexity. These results suggest that in resource-constrained healthcare settings, identifying and representing the most informative anatomy may matter more than the increasing complexity of the model alone. 

---
# A Geometric Gaussian Mixture Representation of Plane Curves 

**Authors**: Ali Darijani, Benedikt Stratmann, Jürgen Beyerer  

**Link**: [PDF](https://arxiv.org/pdf/2606.06505)  

**Abstract**: We introduce a user defined probabilistic polygonal representation for plane curves. Given a curve, we select vertices on the curve and connect consecutive vertices by line segments to obtain a polygonal approximation. Each segment is equipped with a user defined uncertainty parameter in the normal direction. This yields a collection of thin probabilistic geometric primitives that retain the geometrz of the underlying curve while extending it beyond the idealized deterministic one dimensional formulation.
For each segment, we define a Random Variable that is uniform distributed in the tangent direction of the segment and Gaussian distributed in the normal direction of the segment. By matching the first and the second central moments, this construction induces a Gaussian component whose mean lies at the segment midpoint and whose covariance encodes both tangential and normal uncertainty. Combining the segment wise components with appropriate weights yields a Gaussian Mixture Model (GMM) representation of the user defined probabilistic polygonal representation of the plane curve.
The proposed framework provides an analytically tractable probabilistic model that preserves local geometry, and uncertainty in the normal direction. It applies to smooth, closed, open, non regular, and self intersecting plane curves, allows adaptive discretization and varying uncertainty in the normal direction, and as a result supports uncertainty aware geometric modeling. Experiments on a collection of canonical plane curves show that the resulting GMM capture local tangent, local normal, and local arc length; resulting in the global shape of the underlying curves to be truthfully captured as well. The representation is particularly relevant for applications in uncertainty aware CAD and digital twins, probabilistic obstacle modeling in robotics, and probabilistic trajectory planning. 

---
# Human Adults and LLMs as Scientists: Who Benefits from Active Exploration? 

**Authors**: Mandana Samiei, Eunice Yiu, Anthony GX-Chen, Dongyan Lin, Jocelyn Shen, Blake A. Richards, Alison Gopnik, Doina Precup  

**Link**: [PDF](https://arxiv.org/pdf/2606.06464)  

**Abstract**: A long-standing finding in the causal learning literature is that adults struggle to identify conjunctive causal rules, where an effect requires the simultaneous presence of multiple causes, while performing better in disjunctive settings. However, most demonstrations of this ``conjunctive handicap'' rely on passive observation paradigms with limited evidence, where learners have no control over evidence generation. This paper asks whether this bias persists when adults are granted agency through active exploration. Using a modified ``blicket detector'' task, adult participants freely intervened to identify causal objects under conjunctive or disjunctive rule structures. We show that active exploration substantially improves adults' conjunctive causal reasoning, although conjunctive rules still require more tests to infer than disjunctive rules. We further compare human performance to a range of large language models in the same setting. While some state-of-the-art models approach human-level performance on hypothesis inference accuracy, they often exhibit less efficient exploration strategies and similar conjunctive-disjunctive performance gaps. 

---
# Autonomous heterogeneous catalyst discovery with a self-evolving multi-agent digital twin 

**Authors**: Zhilong Song, Zongmin Zhang, Lixue Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.05050)  

**Abstract**: Theoretical heterogeneous catalysis promises rapid catalyst discovery, yet computational and machine-learning predictions often deviate from experiment and stay confined to narrow material families, for want of a faithful, condition-aware catalytic simulator. We present CatDT (Catalysis Digital Twin), a self-evolving multi-agent system that builds an autonomous digital twin of a working catalyst, unifying gas-solid and liquid-solid modeling. From only a bulk crystal and a natural-language reaction description, eight specialized agents and 27 scientific tools predict stable facets, reconstruct working surfaces, enumerate and rank reaction pathways, locate transition states, and compute kinetics in 5-30 min on a single GPU. Two innovations address the hardest steps: UniMech finds dominant pathways for novel materials at over $10^3\times$ lower cost than exhaustive enumeration by fusing agent-guided proposals with energy-cached graph search, and a memory-augmented reinforcement loop raises barrier-calculation success from 41\% to 84\% across 600 catalytic surfaces. Across seven gas-solid benchmarks -- stepped metals, single-atom catalysts, ordered intermetallics, vacancy-rich 2D sulfides and carbides, and a strong-metal--support-interaction (SMSI) interface -- every CatDT prediction lies within 0.5-2 times experiment over four orders of magnitude. For propane dehydrogenation, CatDT independently discovers non-precious candidates rivaling the Pt-based industrial benchmark, with a proposed Ni@ZrO$_2$ SMSI overlayer reaching a simulated TOF of $1.63~\text{s}^{-1}$ at $\sim$100\% selectivity. More broadly, the decisive factor for a faithful catalyst digital twin -- or any multi-stage scientific simulator -- is not raw LLM capability but the engineered harness around it: deterministic tools, persistent memory, and verified self-improvement that compound across models, tools, and runs. 

---
# Trading Engagement for Sustainability: Carbon-Aware Re-ranking for E-commerce Recommendations 

**Authors**: Noah Lund Syrdal, Anders Vestrum, Jorgen Bergh  

**Link**: [PDF](https://arxiv.org/pdf/2606.04550)  

**Abstract**: E-commerce recommender systems strongly influence which products users consider and purchase, yet sustainability signals such as Product Carbon Footprint (PCF) are almost never available at catalog scale. We study carbon-aware product recommendation in the realistic setting where PCF labels are missing for most items and must be inferred. We first estimate product-level carbon footprints via a retrieval-augmented PCF estimation pipeline that transfers supervision from the Carbon Catalogue, a small set of life-cycle-assessed products, to a large unlabeled e-commerce catalog using semantic similarity search, few-shot LLM prompting, and a nearest-neighbour fallback. We then apply a carbon-aware post-hoc re-ranking strategy on top of relevance scores produced by three established recommendation models: BPR, NeuMF, and LightGCN. The method trades off predicted user-item engagement against estimated carbon footprint through a single tunable parameter, lambda. In this offline study, engagement is operationalized through Amazon review interactions, which serve as implicit feedback and as a proxy for user interest or purchase behavior. We evaluate the framework on the Amazon Reviews dataset across three product categories: Home and Kitchen, Sports and Outdoors, and Electronics. By sweeping lambda, we construct Pareto frontiers that characterize the achievable engagement and carbon trade-off for each model and category. Substantial carbon reductions are achievable at minimal engagement cost across all models and categories. However, the available carbon headroom varies by model and category, underscoring the importance of model choice and domain context. 

---
# When Does Multi-Agent Collaboration Help? An Entropy Perspective 

**Authors**: Yuxuan Zhao, Sijia Chen, Ningxin Su  

**Link**: [PDF](https://arxiv.org/pdf/2602.04234)  

**Abstract**: Multi-agent systems (MAS) have emerged as a prominent paradigm for leveraging large language models (LLMs) to tackle complex tasks. However, the mechanisms governing the effectiveness of MAS built upon publicly available LLMs, specifically the underlying rationales for their success or failure, remain largely unexplored. In this paper, we revisit MAS through the perspective of \textit{entropy}, considering both intra- and inter-agent dynamics by investigating entropy transitions during problem-solving across various topologies, six reasoning benchmarks, and two agentic tasks. By analyzing 245 features spanning token-, agent-, and round-level entropy, we counterintuitively find that a single agent outperforms MAS in approximately 43.3\% of cases, and that entropy dynamics are largely determined during the first round of interaction. Furthermore, we provide three key observations: 1) \textit{Certainty Preference}: peak entropy directly harms and stable entropy directly benefits MAS correctness; 2) \textit{Base Entropy}: base models with lower entropy during problem-solving causally drive MAS performance; and 3) \textit{Task Awareness}: entropy dynamics of MAS play varying roles across different tasks. Building on these insights, we introduce a simple yet effective algorithm, the \textit{Entropy Judger}, to select solutions from MAS's pass@$k$ results, leading to consistent accuracy improvements across all MAS configurations and tasks. Our source code is available at \href{this https URL}{this https URL}. 

---
# Zero-Shot Embedding Drift Detection: A Lightweight Defense Against Prompt Injections in LLMs 

**Authors**: Anirudh Sekar, Mrinal Agarwal, Rachel Sharma, Akitsugu Tanaka, Jasmine Zhang, Arjun Damerla, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2601.12359)  

**Abstract**: Prompt injection attacks have become an increasing vulnerability for LLM applications, where adversarial prompts exploit indirect input channels such as emails or user-generated content to circumvent alignment safeguards and induce harmful or unintended outputs. Despite advances in alignment, even state-of-the-art LLMs remain broadly vulnerable to adversarial prompts, underscoring the urgent need for robust, productive, and generalizable detection mechanisms beyond inefficient, model-specific patches. In this work, we propose Zero-Shot Embedding Drift Detection (ZEDD), a lightweight, low-engineering-overhead framework that identifies both direct and indirect prompt injection attempts by quantifying semantic shifts in embedding space between benign and suspect inputs. ZEDD operates without requiring access to model internals, prior knowledge of attack types, or task-specific retraining, enabling efficient zero-shot deployment across diverse LLM architectures. Our method uses adversarial-clean prompt pairs and measures embedding drift via cosine similarity to capture subtle adversarial manipulations inherent to real-world injection attacks. To ensure robust evaluation, we assemble and re-annotate the comprehensive LLMail-Inject dataset spanning five injection categories derived from publicly available sources. Extensive experiments demonstrate that embedding drift is a robust and transferable signal, outperforming traditional methods in detection accuracy and operational efficiency. With greater than 93% accuracy in classifying prompt injections across model architectures like Llama 3, Qwen 2, and Mistral and a false positive rate of <3%, our approach offers a lightweight, scalable defense layer that integrates into existing LLM pipelines, addressing a critical gap in securing LLM-powered systems to withstand adaptive adversarial threats. 

---
