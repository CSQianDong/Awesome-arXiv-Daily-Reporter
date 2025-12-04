# Benchmark for Planning and Control with Large Language Model Agents: Blocksworld with Model Context Protocol 

**Authors**: Niklas Jobs, Luis Miguel Vieira da Silva, Jayanth Somashekaraiah, Maximilian Weigand, David Kube, Felix Gehlhoff  

**Link**: [PDF](https://arxiv.org/pdf/2512.03955)  

**Abstract**: Industrial automation increasingly requires flexible control strategies that can adapt to changing tasks and environments. Agents based on Large Language Models (LLMs) offer potential for such adaptive planning and execution but lack standardized benchmarks for systematic comparison. We introduce a benchmark with an executable simulation environment representing the Blocksworld problem providing five complexity categories. By integrating the Model Context Protocol (MCP) as a standardized tool interface, diverse agent architectures can be connected to and evaluated against the benchmark without implementation-specific modifications. A single-agent implementation demonstrates the benchmark's applicability, establishing quantitative metrics for comparison of LLM-based planning and execution approaches. 

---
# Autonomous Agents and Policy Compliance: A Framework for Reasoning About Penalties 

**Authors**: Vineel Tummala, Daniela Inclezan  

**Link**: [PDF](https://arxiv.org/pdf/2512.03931)  

**Abstract**: This paper presents a logic programming-based framework for policy-aware autonomous agents that can reason about potential penalties for non-compliance and act accordingly. While prior work has primarily focused on ensuring compliance, our approach considers scenarios where deviating from policies may be necessary to achieve high-stakes goals. Additionally, modeling non-compliant behavior can assist policymakers by simulating realistic human decision-making. Our framework extends Gelfond and Lobo's Authorization and Obligation Policy Language (AOPL) to incorporate penalties and integrates Answer Set Programming (ASP) for reasoning. Compared to previous approaches, our method ensures well-formed policies, accounts for policy priorities, and enhances explainability by explicitly identifying rule violations and their consequences. Building on the work of Harders and Inclezan, we introduce penalty-based reasoning to distinguish between non-compliant plans, prioritizing those with minimal repercussions. To support this, we develop an automated translation from the extended AOPL into ASP and refine ASP-based planning algorithms to account for incurred penalties. Experiments in two domains demonstrate that our framework generates higher-quality plans that avoid harmful actions while, in some cases, also improving computational efficiency. These findings underscore its potential for enhancing autonomous decision-making and informing policy refinement. Under consideration in Theory and Practice of Logic Programming (TPLP). 

---
# A Hierarchical Tree-based approach for creating Configurable and Static Deep Research Agent (Static-DRA) 

**Authors**: Saurav Prateek  

**Link**: [PDF](https://arxiv.org/pdf/2512.03887)  

**Abstract**: The advancement in Large Language Models has driven the creation of complex agentic systems, such as Deep Research Agents (DRAs), to overcome the limitations of static Retrieval Augmented Generation (RAG) pipelines in handling complex, multi-turn research tasks. This paper introduces the Static Deep Research Agent (Static-DRA), a novel solution built upon a configurable and hierarchical Tree-based static workflow.
The core contribution is the integration of two user-tunable parameters, Depth and Breadth, which provide granular control over the research intensity. This design allows end-users to consciously balance the desired quality and comprehensiveness of the research report against the associated computational cost of Large Language Model (LLM) interactions. The agent's architecture, comprising Supervisor, Independent, and Worker agents, facilitates effective multi-hop information retrieval and parallel sub-topic investigation.
We evaluate the Static-DRA against the established DeepResearch Bench using the RACE (Reference-based Adaptive Criteria-driven Evaluation) framework. Configured with a depth of 2 and a breadth of 5, and powered by the gemini-2.5-pro model, the agent achieved an overall score of 34.72. Our experiments validate that increasing the configured Depth and Breadth parameters results in a more in-depth research process and a correspondingly higher evaluation score. The Static-DRA offers a pragmatic and resource-aware solution, empowering users with transparent control over the deep research process. The entire source code, outputs and benchmark results are open-sourced at this https URL 

---
# Omni-AutoThink: Adaptive Multimodal Reasoning via Reinforcement Learning 

**Authors**: Dongchao Yang, Songxiang Liu, Disong Wang, Yuanyuan Wang, Guanglu Wan, Helen Meng  

**Link**: [PDF](https://arxiv.org/pdf/2512.03783)  

**Abstract**: Recent advances in Omni models have enabled unified multimodal perception and generation. However, most existing systems still exhibit rigid reasoning behaviors, either overthinking simple problems or failing to reason when necessary. To address this limitation, we propose Omni-AutoThink, a novel adaptive reasoning framework that dynamically adjusts the model's reasoning depth according to task difficulty. Our framework comprises two stages: (1) an Adaptive Supervised Fine-Tuning (Adaptive SFT) stage, which endows the Omni model with fundamental reasoning capability using large-scale reasoning-augmented data, and (2) an Adaptive Reinforcement Learning (Adaptive GRPO) stage, which optimizes reasoning behaviors based on task complexity and reward feedback. We further construct a comprehensive adaptive reasoning benchmark that spans text-only, text-audio, text-visual, and text-audio-visual modalities, providing both training and evaluation splits for multimodal reasoning assessment. Experimental results demonstrate that our proposed framework significantly improves adaptive reasoning performance compared to previous baselines. All benchmark data and code will be publicly released. 

---
# RoCo: Role-Based LLMs Collaboration for Automatic Heuristic Design 

**Authors**: Jiawei Xu, Fengfeng Wei, Weineng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.03762)  

**Abstract**: Automatic Heuristic Design (AHD) has gained traction as a promising solution for solving combinatorial optimization problems (COPs). Large Language Models (LLMs) have emerged and become a promising approach to achieving AHD, but current LLM-based AHD research often only considers a single role. This paper proposes RoCo, a novel Multi-Agent Role-Based System, to enhance the diversity and quality of AHD through multi-role collaboration. RoCo coordinates four specialized LLM-guided agents-explorer, exploiter, critic, and integrator-to collaboratively generate high-quality heuristics. The explorer promotes long-term potential through creative, diversity-driven thinking, while the exploiter focuses on short-term improvements via conservative, efficiency-oriented refinements. The critic evaluates the effectiveness of each evolution step and provides targeted feedback and reflection. The integrator synthesizes proposals from the explorer and exploiter, balancing innovation and exploitation to drive overall progress. These agents interact in a structured multi-round process involving feedback, refinement, and elite mutations guided by both short-term and accumulated long-term reflections. We evaluate RoCo on five different COPs under both white-box and black-box settings. Experimental results demonstrate that RoCo achieves superior performance, consistently generating competitive heuristics that outperform existing methods including ReEvo and HSEvo, both in white-box and black-box scenarios. This role-based collaborative paradigm establishes a new standard for robust and high-performing AHD. 

---
# MemVerse: Multimodal Memory for Lifelong Learning Agents 

**Authors**: Junming Liu, Yifei Sun, Weihua Cheng, Haodong Lei, Yirong Chen, Licheng Wen, Xuemeng Yang, Daocheng Fu, Pinlong Cai, Nianchen Deng, Yi Yu, Shuyue Hu, Botian Shi, Ding Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.03627)  

**Abstract**: Despite rapid progress in large-scale language and vision models, AI agents still suffer from a fundamental limitation: they cannot remember. Without reliable memory, agents catastrophically forget past experiences, struggle with long-horizon reasoning, and fail to operate coherently in multimodal or interactive environments. We introduce MemVerse, a model-agnostic, plug-and-play memory framework that bridges fast parametric recall with hierarchical retrieval-based memory, enabling scalable and adaptive multimodal intelligence. MemVerse maintains short-term memory for recent context while transforming raw multimodal experiences into structured long-term memories organized as hierarchical knowledge graphs. This design supports continual consolidation, adaptive forgetting, and bounded memory growth. To handle real-time demands, MemVerse introduces a periodic distillation mechanism that compresses essential knowledge from long-term memory into the parametric model, allowing fast, differentiable recall while preserving interpretability. Extensive experiments demonstrate that MemVerse significantly improves multimodal reasoning and continual learning efficiency, empowering agents to remember, adapt, and reason coherently across extended interactions. 

---
# DeepRule: An Integrated Framework for Automated Business Rule Generation via Deep Predictive Modeling and Hybrid Search Optimization 

**Authors**: Yusen Wu, Xiaotie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2512.03607)  

**Abstract**: This paper proposes DeepRule, an integrated framework for automated business rule generation in retail assortment and pricing optimization. Addressing the systematic misalignment between existing theoretical models and real-world economic complexities, we identify three critical gaps: (1) data modality mismatch where unstructured textual sources (e.g. negotiation records, approval documents) impede accurate customer profiling; (2) dynamic feature entanglement challenges in modeling nonlinear price elasticity and time-varying attributes; (3) operational infeasibility caused by multi-tier business constraints.
Our framework introduces a tri-level architecture for above challenges. We design a hybrid knowledge fusion engine employing large language models (LLMs) for deep semantic parsing of unstructured text, transforming distributor agreements and sales assessments into structured features while integrating managerial expertise. Then a game-theoretic constrained optimization mechanism is employed to dynamically reconcile supply chain interests through bilateral utility functions, encoding manufacturer-distributor profit redistribution as endogenous objectives under hierarchical constraints. Finally an interpretable decision distillation interface leveraging LLM-guided symbolic regression to find and optimize pricing strategies and auditable business rules embeds economic priors (e.g. non-negative elasticity) as hard constraints during mathematical expression search. We validate the framework in real retail environments achieving higher profits versus systematic B2C baselines while ensuring operational feasibility. This establishes a close-loop pipeline unifying unstructured knowledge injection, multi-agent optimization, and interpretable strategy synthesis for real economic intelligence. 

---
# EnCompass: Enhancing Agent Programming with Search Over Program Execution Paths 

**Authors**: Zhening Li, Armando Solar-Lezama, Yisong Yue, Stephan Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2512.03571)  

**Abstract**: We introduce a new approach to agent programming, the development of LLM-based agents. Current approaches to agent programming often entangle two aspects of agent design: the core workflow logic and the inference-time strategy (e.g., tree search). We introduce "probabilistic angelic nondeterminism" ("PAN"), a programming model that disentangles these two concerns, allowing the programmer to describe the agent workflow and independently experiment with different inference-time strategies by simply changing a few inputs. We provide an implementation of PAN in Python as the EnCompass framework, which uses a Python decorator to compile agent workflow programs into a search space. We present three case studies that demonstrate how the framework lets the programmer quickly improve the reliability of an agent and easily switch between different inference-time strategies, all with little additional coding. 

---
# Reason-Plan-ReAct: A Reasoner-Planner Supervising a ReAct Executor for Complex Enterprise Tasks 

**Authors**: Gianni Molinari, Fabio Ciravegna  

**Link**: [PDF](https://arxiv.org/pdf/2512.03560)  

**Abstract**: Despite recent advances, autonomous agents often struggle to solve complex tasks in enterprise domains that require coordinating multiple tools and processing diverse data sources. This struggle is driven by two main limitations. First, single-agent architectures enforce a monolithic plan-execute loop, which directly causes trajectory instability. Second, the requirement to use local open-weight models for data privacy introduces smaller context windows leading to the rapid consumption of context from large tool outputs. To solve this problem we introduce RP-ReAct (Reasoner Planner-ReAct), a novel multi-agent approach that fundamentally decouples strategic planning from low-level execution to achieve superior reliability and efficiency. RP-ReAct consists of a Reasoner Planner Agent (RPA), responsible for planning each sub-step, continuously analysing the execution results using the strong reasoning capabilities of a Large Reasoning Model, and one or multiple Proxy-Execution Agent (PEA) that translates sub-steps into concrete tool interactions using a ReAct approach. Crucially, we incorporate a context-saving strategy within the PEA to mitigate context window overflow by managing large tool outputs via external storage and on-demand access. We evaluate RP-ReAct, on the challenging, multi-domain ToolQA benchmark using a diverse set of six open-weight reasoning models. Our empirical results show that RP-ReAct achieves superior performance and improved generalization ability over state-of-the-art baselines when addressing diverse complex tasks across the evaluated domains. Furthermore we establish the enhanced robustness and stability of our approach across different model scales, paving the way for effective and deployable agentic solutions for enterprises. 

---
# PARC: An Autonomous Self-Reflective Coding Agent for Robust Execution of Long-Horizon Tasks 

**Authors**: Yuki Orimo, Iori Kurata, Hodaka Mori, Ryuhei Okuno, Ryohto Sawada, Daisuke Okanohara  

**Link**: [PDF](https://arxiv.org/pdf/2512.03549)  

**Abstract**: We introduce PARC, a coding agent for the autonomous and robust execution of long-horizon computational tasks. PARC is built on a hierarchical multi-agent architecture incorporating task planning, execution, and a mechanism that evaluates its own actions and their outcomes from an independent context and provides feedback, namely self-assessment and self-feedback. This design enables PARC to detect and correct high-level strategic errors and sustain progress without human intervention. We evaluate PARC across computational science and data science tasks. In materials science, it autonomously reproduces key results from studies on lithium-ion conduction and alloy segregation. In particular, it coordinates dozens of parallel simulation tasks, each requiring roughly 43 hours of computation, managing orchestration, monitoring, and error correction end-to-end. In Kaggle-based experiments, starting from minimal natural-language instructions, PARC conducts data analysis and implements search strategies, producing solutions competitive with human-engineered baselines. These results highlight the potential of integrating a hierarchical multi-agent system with self-assessment and self-feedback to enable AI systems capable of independent, large-scale scientific and analytical work. 

---
# Multi-Agent Reinforcement Learning with Communication-Constrained Priors 

**Authors**: Guang Yang, Tianpei Yang, Jingwen Qiao, Yanqing Wu, Jing Huo, Xingguo Chen, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2512.03528)  

**Abstract**: Communication is one of the effective means to improve the learning of cooperative policy in multi-agent systems. However, in most real-world scenarios, lossy communication is a prevalent issue. Existing multi-agent reinforcement learning with communication, due to their limited scalability and robustness, struggles to apply to complex and dynamic real-world environments. To address these challenges, we propose a generalized communication-constrained model to uniformly characterize communication conditions across different scenarios. Based on this, we utilize it as a learning prior to distinguish between lossy and lossless messages for specific scenarios. Additionally, we decouple the impact of lossy and lossless messages on distributed decision-making, drawing on a dual mutual information estimatior, and introduce a communication-constrained multi-agent reinforcement learning framework, quantifying the impact of communication messages into the global reward. Finally, we validate the effectiveness of our approach across several communication-constrained benchmarks. 

---
# Multimodal Reinforcement Learning with Agentic Verifier for AI Agents 

**Authors**: Reuben Tan, Baolin Peng, Zhengyuan Yang, Hao Cheng, Oier Mees, Theodore Zhao, Andrea Tupini, Isar Meijier, Qianhui Wu, Yuncong Yang, Lars Liden, Yu Gu, Sheng Zhang, Xiaodong Liu, Lijuan Wang, Marc Pollefeys, Yong Jae Lee, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2512.03438)  

**Abstract**: Agentic reasoning models trained with multimodal reinforcement learning (MMRL) have become increasingly capable, yet they are almost universally optimized using sparse, outcome-based rewards computed based on the final answers. Richer rewards computed from the reasoning tokens can improve learning significantly by providing more fine-grained guidance. However, it is challenging to compute more informative rewards in MMRL beyond those based on outcomes since different samples may require different scoring functions and teacher models may provide noisy reward signals too. In this paper, we introduce the Argos (Agentic Reward for Grounded & Objective Scoring), a principled reward agent to train multimodal reasoning models for agentic tasks. For each sample, Argos selects from a pool of teacher-model derived and rule-based scoring functions to simultaneously evaluate: (i) final response accuracy, (ii) spatiotemporal localization of referred entities and actions, and (iii) the quality of the reasoning process. We find that by leveraging our agentic verifier across both SFT data curation and RL training, our model achieves state-of-the-art results across multiple agentic tasks such as spatial reasoning, visual hallucination as well as robotics and embodied AI benchmarks. Critically, we demonstrate that just relying on SFT post-training on highly curated reasoning data is insufficient, as agents invariably collapse to ungrounded solutions during RL without our online verification. We also show that our agentic verifier can help to reduce reward-hacking in MMRL. Finally, we also provide a theoretical justification for the effectiveness of Argos through the concept of pareto-optimality. 

---
# Evaluating Generalization Capabilities of LLM-Based Agents in Mixed-Motive Scenarios Using Concordia 

**Authors**: Chandler Smith, Marwa Abdulhai, Manfred Diaz, Marko Tesic, Rakshit S. Trivedi, Alexander Sasha Vezhnevets, Lewis Hammond, Jesse Clifton, Minsuk Chang, Edgar A. Duéñez-Guzmán, John P. Agapiou, Jayd Matyas, Danny Karmon, Akash Kundu, Aliaksei Korshuk, Ananya Ananya, Arrasy Rahman, Avinaash Anand Kulandaivel, Bain McHale, Beining Zhang, Buyantuev Alexander, Carlos Saith Rodriguez Rojas, Caroline Wang, Chetan Talele, Chenao Liu, Chichen Lin, Diana Riazi, Di Yang Shi, Emanuel Tewolde, Elizaveta Tennant, Fangwei Zhong, Fuyang Cui, Gang Zhao, Gema Parreño Piqueras, Hyeonggeun Yun, Ilya Makarov, Jiaxun Cui, Jebish Purbey, Jim Dilkes, Jord Nguyen, Lingyun Xiao, Luis Felipe Giraldo, Manuela Chacon-Chamorro, Manuel Sebastian Rios Beltran, Marta Emili García Segura, Mengmeng Wang, Mogtaba Alim, Nicanor Quijano, Nico Schiavone, Olivia Macmillan-Scott, Oswaldo Peña, Peter Stone, Ram Mohan Rao Kadiyala, Rolando Fernandez, Ruben Manrique, Sunjia Lu, Sheila A. McIlraith, Shamika Dhuri, Shuqing Shi, Siddhant Gupta, Sneheel Sarangi, Sriram Ganapathi Subramanian, Taehun Cha, Toryn Q. Klassen, Wenming Tu, Weijian Fan, Wu Ruiyang, Xue Feng, Yali Du, Yang Liu, Yiding Wang, Yipeng Kang, Yoonchang Sung, Yuxuan Chen, Zhaowei Zhang, Zhihan Wang, Zhiqiang Wu, Ziang Chen, Zilong Zheng, Zixia Jia, Ziyan Wang, Dylan Hadfield-Menell, Natasha Jaques, Tim Baarslag, Jose Hernandez-Orallo, Joel Z. Leibo  

**Link**: [PDF](https://arxiv.org/pdf/2512.03318)  

**Abstract**: Large Language Model (LLM) agents have demonstrated impressive capabilities for social interaction and are increasingly being deployed in situations where they might engage with both human and artificial agents. These interactions represent a critical frontier for LLM-based agents, yet existing evaluation methods fail to measure how well these capabilities generalize to novel social situations. In this paper, we introduce a method for evaluating the ability of LLM-based agents to cooperate in zero-shot, mixed-motive environments using Concordia, a natural language multi-agent simulation environment. Our method measures general cooperative intelligence by testing an agent's ability to identify and exploit opportunities for mutual gain across diverse partners and contexts. We present empirical results from the NeurIPS 2024 Concordia Contest, where agents were evaluated on their ability to achieve mutual gains across a suite of diverse scenarios ranging from negotiation to collective action problems. Our findings reveal significant gaps between current agent capabilities and the robust generalization required for reliable cooperation, particularly in scenarios demanding persuasion and norm enforcement. 

---
# Prior preferences in active inference agents: soft, hard, and goal shaping 

**Authors**: Filippo Torresan, Ryota Kanai, Manuel Baltieri  

**Link**: [PDF](https://arxiv.org/pdf/2512.03293)  

**Abstract**: Active inference proposes expected free energy as an objective for planning and decision-making to adequately balance exploitative and explorative drives in learning agents. The exploitative drive, or what an agent wants to achieve, is formalised as the Kullback-Leibler divergence between a variational probability distribution, updated at each inference step, and a preference probability distribution that indicates what states or observations are more likely for the agent, hence determining the agent's goal in a certain environment. In the literature, the questions of how the preference distribution should be specified and of how a certain specification impacts inference and learning in an active inference agent have been given hardly any attention. In this work, we consider four possible ways of defining the preference distribution, either providing the agents with hard or soft goals and either involving or not goal shaping (i.e., intermediate goals). We compare the performances of four agents, each given one of the possible preference distributions, in a grid world navigation task. Our results show that goal shaping enables the best performance overall (i.e., it promotes exploitation) while sacrificing learning about the environment's transition dynamics (i.e., it hampers exploration). 

---
# When Do Symbolic Solvers Enhance Reasoning in Large Language Models? 

**Authors**: Zhiyuan He, Dingmin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.03272)  

**Abstract**: Large Reasoning Models (LRMs) achieve strong performance on complex reasoning tasks by generating long Chains of Thought (CoTs). However, this paradigm might incur substantial token overhead, especially when models "overthink" by producing lengthy reasoning chains, which can even lead to incorrect answers. A promising direction is the symbolic-solver-integrated approach, which leverages the code generation capabilities of LLMs to translate reasoning tasks into executable code and then solve them with a symbolic solver. In this paper, we explore an open question of when the conventional long-CoT can be enhanced by symbolic solvers. Our experimental results show that the symbolic-solver-integrated method only helps when the problem requires limited implicit reasoning but involves an ample search space. The latest LLMs, like GPT-4o, show better performance on deductive problems with shallow reasoning depth, while the symbolic-solver-integrated method significantly improves the LLMs' performance in constraint satisfaction problems that require repeated backtracks. When a declarative exemplar is provided, even CodeLlama-13B can outperform GPT-4o in difficult Zebra puzzles. 

---
# Beyond the Black Box: A Cognitive Architecture for Explainable and Aligned AI 

**Authors**: Hu Keyi  

**Link**: [PDF](https://arxiv.org/pdf/2512.03072)  

**Abstract**: Current AI paradigms, as "architects of experience," face fundamental challenges in explainability and value alignment. This paper introduces "Weight-Calculatism," a novel cognitive architecture grounded in first principles, and demonstrates its potential as a viable pathway toward Artificial General Intelligence (AGI). The architecture deconstructs cognition into indivisible Logical Atoms and two fundamental operations: Pointing and Comparison. Decision-making is formalized through an interpretable Weight-Calculation model (Weight = Benefit * Probability), where all values are traceable to an auditable set of Initial Weights. This atomic decomposition enables radical explainability, intrinsic generality for novel situations, and traceable value alignment. We detail its implementation via a graph-algorithm-based computational engine and a global workspace workflow, supported by a preliminary code implementation and scenario validation. Results indicate that the architecture achieves transparent, human-like reasoning and robust learning in unprecedented scenarios, establishing a practical and theoretical foundation for building trustworthy and aligned AGI. 

---
# Exploring Syntropic Frameworks in AI Alignment: A Philosophical Investigation 

**Authors**: Austin Spizzirri  

**Link**: [PDF](https://arxiv.org/pdf/2512.03048)  

**Abstract**: I argue that AI alignment should be reconceived as architecting syntropic, reasons-responsive agents through process-based, multi-agent, developmental mechanisms rather than encoding fixed human value content. The paper makes three philosophical contributions. First, I articulate the ``specification trap'' argument demonstrating why content-based value specification appears structurally unstable due to the conjunction of the is-ought gap, value pluralism, and the extended frame problem. Second, I propose syntropy -- the recursive reduction of mutual uncertainty between agents through state alignment -- as an information-theoretic framework for understanding multi-agent alignment dynamics. Third, I establish a functional distinction between genuine and simulated moral capacity grounded in compatibilist theories of guidance control, coupled with an embodied experimental paradigm and verification regime providing operational criteria independent of phenomenological claims. This paper represents the philosophical component of a broader research program whose empirical validation is being developed in a separate project currently in preparation. While the framework generates specific, falsifiable predictions about value emergence and moral agency in artificial systems, empirical validation remains pending. 

---
# SkillFactory: Self-Distillation For Learning Cognitive Behaviors 

**Authors**: Zayne Sprague, Jack Lu, Manya Wadhwa, Sedrick Keh, Mengye Ren, Greg Durrett  

**Link**: [PDF](https://arxiv.org/pdf/2512.04072)  

**Abstract**: Reasoning models leveraging long chains of thought employ various cognitive skills, such as verification of their answers, backtracking, retrying by an alternate method, and more. Previous work has shown that when a base language model exhibits these skills, training that model further with reinforcement learning (RL) can learn to leverage them. How can we get models to leverage skills that aren't exhibited by base models? Our work, SkillFactory, is a method for fine-tuning models to roughly learn these skills during a supervised fine-tuning (SFT) stage prior to RL. Our approach does not rely on distillation from a stronger model, but instead uses samples from the model itself, rearranged to provide training data in the format of those skills. These "silver" SFT traces may be imperfect, but are nevertheless effective for priming a model to acquire skills during RL. Our evaluation shows that (1) starting from SkillFactory SFT initialization helps a model to generalize to harder variants of a task post-RL, despite lower performance pre-RL; (2) cognitive skills are indeed used by the model; (3) RLed SkillFactory models are more robust to regression on out-of-domain tasks than RLed base models. Our work suggests that inductive biases learned prior to RL help models learn robust cognitive skill use. 

---
# Fare Comparison App of Uber, Ola and Rapido 

**Authors**: Ashlesha Gopinath Sawant, Sahil S. Jadhav, Vidhan R. Jain, Shriraj S. Jagtap, Prachi Jadhav, Soham Jadhav, Ichha Raina  

**Link**: [PDF](https://arxiv.org/pdf/2512.04065)  

**Abstract**: In todays increasing world, it is very important to have good hailing services like Ola, Uber, and Rapido as it is very essential for our daily transportation. Users often face difficulties in choosing the most appropriate and efficient ride that would lead to both cost-effective and would take us to our destination in less time. This project provides you with the web application that helps you to select the most beneficial ride for you by providing users with the fare comparison between Ola, Uber, Rapido for the destination entered by the user. The backend is use to fetch the data, providing users with the fare comparison for the ride and finally providing with the best option using Python. This research paper also addresses the problem and challenges faced in accessing the data using APIs, Android Studios emulator, Appium and location comparison. Thus, the aim of the project is to provide transparency to the users in ride-hailing services and increase efficiency and provide users with better experience. 

---
# Polarization by Design: How Elites Could Shape Mass Preferences as AI Reduces Persuasion Costs 

**Authors**: Nadav Kunievsky  

**Link**: [PDF](https://arxiv.org/pdf/2512.04047)  

**Abstract**: In democracies, major policy decisions typically require some form of majority or consensus, so elites must secure mass support to govern. Historically, elites could shape support only through limited instruments like schooling and mass media; advances in AI-driven persuasion sharply reduce the cost and increase the precision of shaping public opinion, making the distribution of preferences itself an object of deliberate design. We develop a dynamic model in which elites choose how much to reshape the distribution of policy preferences, subject to persuasion costs and a majority rule constraint. With a single elite, any optimal intervention tends to push society toward more polarized opinion profiles - a ``polarization pull'' - and improvements in persuasion technology accelerate this drift. When two opposed elites alternate in power, the same technology also creates incentives to park society in ``semi-lock'' regions where opinions are more cohesive and harder for a rival to overturn, so advances in persuasion can either heighten or dampen polarization depending on the environment. Taken together, cheaper persuasion technologies recast polarization as a strategic instrument of governance rather than a purely emergent social byproduct, with important implications for democratic stability as AI capabilities advance. 

---
# MarkTune: Improving the Quality-Detectability Trade-off in Open-Weight LLM Watermarking 

**Authors**: Yizhou Zhao, Zhiwei Steven Wu, Adam Block  

**Link**: [PDF](https://arxiv.org/pdf/2512.04044)  

**Abstract**: Watermarking aims to embed hidden signals in generated text that can be reliably detected when given access to a secret key. Open-weight language models pose acute challenges for such watermarking schemes because the inference-time interventions that dominate contemporary approaches cannot be enforced once model weights are public. Existing watermaking techniques for open-weight models, such as the recently proposed GaussMark, typically rely on small modifications to model weights, which can yield signals detectable to those equipped with a secret key, but achieving detection power comparable to inference-time watermarks generally requires weight perturbations that noticeably reduce generation quality. We introduce MarkTune, a theoretically principled, on-policy fine-tuning framework that treats the GaussMark signal as a reward while simultaneously regularizing against degradation in text quality. We derive MarkTune as an improvement on GaussMark and demonstrate that MarkTune consistently improves the quality-detectability trade-off over GaussMark by steering finer-grained, watermark-aware weight updates within the model's representation space while preserving generation quality. Empirically, we show that MarkTune pushes the quality-detectability frontier of GaussMark close to that of inference-time watermarking, remains robust to paraphrasing and fine-tuning attacks, and exhibits strong generalization: a model fine-tuned on one dataset retains substantial watermark detection power on unseen datasets. Together, these results establish MarkTune as a general strategy for embedding robust, high-quality watermarks into open-weight LMs. 

---
# Fast & Efficient Normalizing Flows and Applications of Image Generative Models 

**Authors**: Sandeep Nagar  

**Link**: [PDF](https://arxiv.org/pdf/2512.04039)  

**Abstract**: This thesis presents novel contributions in two primary areas: advancing the efficiency of generative models, particularly normalizing flows, and applying generative models to solve real-world computer vision challenges. The first part introduce significant improvements to normalizing flow architectures through six key innovations: 1) Development of invertible 3x3 Convolution layers with mathematically proven necessary and sufficient conditions for invertibility, (2) introduction of a more efficient Quad-coupling layer, 3) Design of a fast and efficient parallel inversion algorithm for kxk convolutional layers, 4) Fast & efficient backpropagation algorithm for inverse of convolution, 5) Using inverse of convolution, in Inverse-Flow, for the forward pass and training it using proposed backpropagation algorithm, and 6) Affine-StableSR, a compact and efficient super-resolution model that leverages pre-trained weights and Normalizing Flow layers to reduce parameter count while maintaining performance.
The second part: 1) An automated quality assessment system for agricultural produce using Conditional GANs to address class imbalance, data scarcity and annotation challenges, achieving good accuracy in seed purity testing; 2) An unsupervised geological mapping framework utilizing stacked autoencoders for dimensionality reduction, showing improved feature extraction compared to conventional methods; 3) We proposed a privacy preserving method for autonomous driving datasets using on face detection and image inpainting; 4) Utilizing Stable Diffusion based image inpainting for replacing the detected face and license plate to advancing privacy-preserving techniques and ethical considerations in the field.; and 5) An adapted diffusion model for art restoration that effectively handles multiple types of degradation through unified fine-tuning. 

---
# Jina-VLM: Small Multilingual Vision Language Model 

**Authors**: Andreas Koukounas, Georgios Mastrapas, Florian Hönicke, Sedigheh Eslami, Guillaume Roncari, Scott Martens, Han Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2512.04032)  

**Abstract**: We present Jina-VLM, a 2.4B parameter vision-language model that achieves state-of-the-art multilingual visual question answering among open 2B-scale VLMs. The model couples a SigLIP2 vision encoder with a Qwen3 language backbone through an attention-pooling connector that enables token-efficient processing of arbitrary-resolution images. Across standard VQA benchmarks and multilingual evaluations, Jina-VLM outperforms comparable models while preserving competitive text-only performance. 

---
# Large Language Models for Limited Noisy Data: A Gravitational Wave Identification Study 

**Authors**: Yixuan Li, Yuhao Lu, Yang Liu, Liang Li, R. Ruffini, Di Li, Rong-Gen Cai, Xiaoyan Zhu, Wenbin Lin, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.04031)  

**Abstract**: This work investigates whether large language models (LLMs) offer advantages over traditional neural networks for astronomical data processing, in regimes with non-Gaussian, non-stationary noise and limited labeled samples. Gravitational wave observations provide an suitable test case, using only 90 LIGO events, finetuned LLMs achieve 97.4\% accuracy for identifying signals. Further experiments show that, in contrast to traditional networks that rely on large simulated datasets, additional simulated samples do not improve LLM performance, while scaling studies reveal predictable gains with increasing model size and dataset size. These results indicate that LLMs can extract discriminative structure directly from observational data and provide an efficient assessment for gravitational wave identification. The same strategy may extend to other astronomical domains with similar noise properties, such as radio or pulsar observations. 

---
# PSA: Pyramid Sparse Attention for Efficient Video Understanding and Generation 

**Authors**: Xiaolong Li, Youping Gu, Xi Lin, Weijie Wang, Bohan Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2512.04025)  

**Abstract**: Attention mechanisms are the core of foundation models, but their quadratic complexity remains a critical bottleneck for scaling. This challenge has driven the development of efficient attention mechanisms, with sparsity emerging as the dominant paradigm. Current methods typically retain or discard entire key-value blocks with binary masks, resulting in substantial information loss under high sparsity. To mitigate this gap, we present Pyramid Sparse Attention (PSA), a versatile module applicable to both video understanding and generation tasks. Instead of binary masking, PSA introduces multi-level pooled KV representations, enabling finer mask granularity. Specifically, each query block dynamically allocates lower pooling levels to critical KV blocks and higher levels to less important ones, creating an informative interpolation between full retention and complete pruning. This design, analogous to fixed-point quantization and classical feature pyramid networks in computer vision, effectively mitigates information loss while preserving computational efficiency under a low compute budget. It works with a native, hardware-friendly kernel that leverages decoupled block-tile design to ensure efficient execution. Across video understanding and generation benchmarks, PSA preserves contextual information and visual fidelity, consistently outperforming or achieving comparable performance over existing sparse attention baselines with superior efficiency-quality trade-offs. Our code and model weights are publicly available at: this http URL 

---
# TARA Test-by-Adaptive-Ranks for Quantum Anomaly Detection with Conformal Prediction Guarantees 

**Authors**: Davut Emre Tasar, Ceren Ocal Tasar  

**Link**: [PDF](https://arxiv.org/pdf/2512.04016)  

**Abstract**: Quantum key distribution (QKD) security fundamentally relies on the ability to distinguish genuine quantum correlations from classical eavesdropper simulations, yet existing certification methods lack rigorous statistical guarantees under finite-sample conditions and adversarial scenarios. We introduce TARA (Test by Adaptive Ranks), a novel framework combining conformal prediction with sequential martingale testing for quantum anomaly detection that provides distribution-free validity guarantees. TARA offers two complementary approaches. TARA k, based on Kolmogorov Smirnov calibration against local hidden variable (LHV) null distributions, achieving ROC AUC = 0.96 for quantum-classical discrimination. And TARA-m, employing betting martingales for streaming detection with anytime valid type I error control that enables real time monitoring of quantum channels. We establish theoretical guarantees proving that under (context conditional) exchangeability, conformal p-values remain uniformly distributed even for strongly contextual quantum data, confirming that quantum contextuality does not break conformal prediction validity a result with implications beyond quantum certification to any application of distribution-free methods to nonclassical data. Extensive validation on both IBM Torino (superconducting, CHSH = 2.725) and IonQ Forte Enterprise (trapped ion, CHSH = 2.716) quantum processors demonstrates cross-platform robustness, achieving 36% security margins above the classical CHSH bound of 2. Critically, our framework reveals a methodological concern affecting quantum certification more broadly: same-distribution calibration can inflate detection performance by up to 44 percentage points compared to proper cross-distribution calibration, suggesting that prior quantum certification studies using standard train test splits may have systematically overestimated adversarial robustness. 

---
# On the Temporality for Sketch Representation Learning 

**Authors**: Marcelo Isaias de Moraes Junior, Moacir Antonelli Ponti  

**Link**: [PDF](https://arxiv.org/pdf/2512.04007)  

**Abstract**: Sketches are simple human hand-drawn abstractions of complex scenes and real-world objects. Although the field of sketch representation learning has advanced significantly, there is still a gap in understanding the true relevance of the temporal aspect to the quality of these representations. This work investigates whether it is indeed justifiable to treat sketches as sequences, as well as which internal orders play a more relevant role. The results indicate that, although the use of traditional positional encodings is valid for modeling sketches as sequences, absolute coordinates consistently outperform relative ones. Furthermore, non-autoregressive decoders outperform their autoregressive counterparts. Finally, the importance of temporality was shown to depend on both the order considered and the task evaluated. 

---
# Divide, then Ground: Adapting Frame Selection to Query Types for Long-Form Video Understanding 

**Authors**: Jialuo Li, Bin Li, Jiahao Li, Yan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2512.04000)  

**Abstract**: The application of Large Multimodal Models (LMMs) to long-form video understanding is constrained by limited context lengths and the computationally prohibitive cost of processing dense video tokens. Consequently, recent research has focused on query-aware frame selection, methods that often incur significant computational overhead. This paper challenges the assumption that such complex search mechanisms are universally necessary. We first identify and validate a query typology distinguishing between global query and localized query. We demonstrate that while uniform sampling is both effective and efficient for global queries, localized queries indeed necessitate query-aware selection for optimal performance. Building on this insight, we propose DIG, a training-free frame selection framework that adapts its strategy based on the query type. Specifically,DIG employs efficient uniform sampling for global queries while activating a specialized pipeline to extract query-relevant frames for localized queries. Experiments on three long-form video understanding benchmarks demonstrate that DIG consistently outperforms existing baselines and robustly improves LMM performance, even when scaling the input frame count to 256. 

---
# Highly Efficient Test-Time Scaling for T2I Diffusion Models with Text Embedding Perturbation 

**Authors**: Hang Xu, Linjiang Huang, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2512.03996)  

**Abstract**: Test-time scaling (TTS) aims to achieve better results by increasing random sampling and evaluating samples based on rules and metrics. However, in text-to-image(T2I) diffusion models, most related works focus on search strategies and reward models, yet the impact of the stochastic characteristic of noise in T2I diffusion models on the method's performance remains unexplored. In this work, we analyze the effects of randomness in T2I diffusion models and explore a new format of randomness for TTS: text embedding perturbation, which couples with existing randomness like SDE-injected noise to enhance generative diversity and quality. We start with a frequency-domain analysis of these formats of randomness and their impact on generation, and find that these two randomness exhibit complementary behavior in the frequency domain: spatial noise favors low-frequency components (early steps), while text embedding perturbation enhances high-frequency details (later steps), thereby compensating for the potential limitations of spatial noise randomness in high-frequency manipulation. Concurrently, text embedding demonstrates varying levels of tolerance to perturbation across different dimensions of the generation process. Specifically, our method consists of two key designs: (1) Introducing step-based text embedding perturbation, combining frequency-guided noise schedules with spatial noise perturbation. (2) Adapting the perturbation intensity selectively based on their frequency-specific contributions to generation and tolerance to perturbation. Our approach can be seamlessly integrated into existing TTS methods and demonstrates significant improvements on multiple benchmarks with almost no additional computation. Code is available at \href{this https URL}{this https URL}. 

---
# DIQ-H: Evaluating Hallucination Persistence in VLMs Under Temporal Visual Degradation 

**Authors**: Zexin Lin, Hawen Wan, Yebin Zhong, Xiaoqiang  

**Link**: [PDF](https://arxiv.org/pdf/2512.03992)  

**Abstract**: Vision-Language Models (VLMs) deployed in safety-critical applications such as autonomous driving must handle continuous visual streams under imperfect conditions. However, existing benchmarks focus on static, high-quality images and ignore temporal degradation and error propagation, which are critical failure modes where transient visual corruption induces hallucinations that persist across subsequent frames. We introduce DIQ-H, the first benchmark for evaluating VLM robustness under dynamic visual degradation in temporal sequences. DIQ-H applies physics-based corruptions including motion blur, sensor noise, and compression artifacts, and measures hallucination persistence, error recovery, and temporal consistency through multi-turn question-answering tasks. To enable scalable annotation, we propose Uncertainty-Guided Iterative Refinement (UIR), which generates reliable pseudo-ground-truth using lightweight VLMs with uncertainty filtering, achieving a 15.3 percent accuracy improvement. Experiments on 16 state-of-the-art VLMs reveal substantial robustness gaps: even advanced models such as GPT-4o achieve only a 78.5 percent recovery rate, while open-source models struggle with temporal consistency at less than 60 percent. DIQ-H provides a comprehensive platform for evaluating VLM reliability in real-world deployments. 

---
# BlurDM: A Blur Diffusion Model for Image Deblurring 

**Authors**: Jin-Ting He, Fu-Jen Tsai, Yan-Tsung Peng, Min-Hung Chen, Chia-Wen Lin, Yen-Yu Lin  

**Link**: [PDF](https://arxiv.org/pdf/2512.03979)  

**Abstract**: Diffusion models show promise for dynamic scene deblurring; however, existing studies often fail to leverage the intrinsic nature of the blurring process within diffusion models, limiting their full potential. To address it, we present a Blur Diffusion Model (BlurDM), which seamlessly integrates the blur formation process into diffusion for image deblurring. Observing that motion blur stems from continuous exposure, BlurDM implicitly models the blur formation process through a dual-diffusion forward scheme, diffusing both noise and blur onto a sharp image. During the reverse generation process, we derive a dual denoising and deblurring formulation, enabling BlurDM to recover the sharp image by simultaneously denoising and deblurring, given pure Gaussian noise conditioned on the blurred image as input. Additionally, to efficiently integrate BlurDM into deblurring networks, we perform BlurDM in the latent space, forming a flexible prior generation network for deblurring. Extensive experiments demonstrate that BlurDM significantly and consistently enhances existing deblurring methods on four benchmark datasets. The source code is available at this https URL. 

---
# Sponsored Questions and How to Auction Them 

**Authors**: Kshipra Bhawalkar, Alexandros Psomas, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.03975)  

**Abstract**: Online platforms connect users with relevant products and services using ads. A key challenge is that a user's search query often leaves their true intent ambiguous. Typically, platforms passively predict relevance based on available signals and in some cases offer query refinements. The shift from traditional search to conversational AI provides a new approach. When a user's query is ambiguous, a Large Language Model (LLM) can proactively offer several clarifying follow-up prompts. In this paper we consider the following: what if some of these follow-up prompts can be ``sponsored,'' i.e., selected for their advertising potential. How should these ``suggestion slots'' be allocated? And, how does this new mechanism interact with the traditional ad auction that might follow?
This paper introduces a formal model for designing and analyzing these interactive platforms. We use this model to investigate a critical engineering choice: whether it is better to build an end-to-end pipeline that jointly optimizes the user interaction and the final ad auction, or to decouple them into separate mechanisms for the suggestion slots and another for the subsequent ad slot. We show that the VCG mechanism can be adopted to jointly optimize the sponsored suggestion and the ads that follow; while this mechanism is more complex, it achieves outcomes that are efficient and truthful. On the other hand, we prove that the simple-to-implement modular approach suffers from strategic inefficiency: its Price of Anarchy is unbounded. 

---
# Guided Flow Policy: Learning from High-Value Actions in Offline Reinforcement Learning 

**Authors**: Franki Nguimatsia Tiofack, Théotime Le Hellard, Fabian Schramm, Nicolas Perrin-Gilbert, Justin Carpentier  

**Link**: [PDF](https://arxiv.org/pdf/2512.03973)  

**Abstract**: Offline reinforcement learning often relies on behavior regularization that enforces policies to remain close to the dataset distribution. However, such approaches fail to distinguish between high-value and low-value actions in their regularization components. We introduce Guided Flow Policy (GFP), which couples a multi-step flow-matching policy with a distilled one-step actor. The actor directs the flow policy through weighted behavior cloning to focus on cloning high-value actions from the dataset rather than indiscriminately imitating all state-action pairs. In turn, the flow policy constrains the actor to remain aligned with the dataset's best transitions while maximizing the critic. This mutual guidance enables GFP to achieve state-of-the-art performance across 144 state and pixel-based tasks from the OGBench, Minari, and D4RL benchmarks, with substantial gains on suboptimal datasets and challenging tasks. Webpage: this https URL 

---
# A Theoretical Framework for Auxiliary-Loss-Free Load Balancing of Sparse Mixture-of-Experts in Large-Scale AI Models 

**Authors**: X.Y. Han, Yuan Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2512.03915)  

**Abstract**: In large-scale AI training, Sparse Mixture-of-Experts (s-MoE) layers enable scaling by activating only a small subset of experts per token. An operational challenge in this design is load balancing: routing tokens to minimize the number of idle experts, which is important for the efficient utilization of (costly) GPUs. We provide a theoretical framework for analyzing the Auxiliary-Loss-Free Load Balancing (ALF-LB) procedure -- proposed by DeepSeek's Wang et al. (2024) -- by casting it as a one-step-per-iteration primal-dual method for an assignment problem. First, in a stylized deterministic setting, our framework yields several insightful structural properties: (i) a monotonic improvement of a Lagrangian objective, (ii) a preference rule that moves tokens from overloaded to underloaded experts, and (iii) an approximate-balancing guarantee. Then, we incorporate the stochastic and dynamic nature of AI training using a generalized online optimization formulation. In the online setting, we derive a strong convexity property of the objective that leads to a logarithmic expected regret bound under certain step-size choices. Additionally, we present real experiments on 1B-parameter DeepSeekMoE models to complement our theoretical findings. Together, these results build a principled framework for analyzing the Auxiliary-Loss-Free Load Balancing of s-MoE in AI models. 

---
# Hierarchical Vision Language Action Model Using Success and Failure Demonstrations 

**Authors**: Jeongeun Park, Jihwan Yoon, Byungwoo Jeon, Juhan Park, Jinwoo Shin, Namhoon Cho, Kyungjae Lee, Sangdoo Yun, Sungjoon Choi  

**Link**: [PDF](https://arxiv.org/pdf/2512.03913)  

**Abstract**: Prior Vision-Language-Action (VLA) models are typically trained on teleoperated successful demonstrations, while discarding numerous failed attempts that occur naturally during data collection. However, these failures encode where and how policies can be fragile, information that can be exploited to improve robustness. We address this problem by leveraging mixed-quality datasets to learn failure-aware reasoning at planning time. We introduce VINE, a hierarchical vision-language-action model that separates high-level reasoning (System 2) from low-level control (System 1) under a hierarchical reinforcement learning formalism, making failures usable as a structured learning signal rather than noisy supervision. System 2 performs feasibility-guided tree search over a 2D scene-graph abstraction: it proposes subgoal transitions, predicts success probabilities from both successes and failures, and prunes brittle branches before execution, effectively casting plan evaluation as feasibility scoring. The selected subgoal sequence is then passed to System 1, which executes low-level actions without modifying the agent's core skills. Trained entirely from offline teleoperation data, VINE integrates negative experience directly into the decision loop. Across challenging manipulation tasks, this approach consistently improves success rates and robustness, demonstrating that failure data is an essential resource for converting the broad competence of VLAs into robust execution. 

---
# Autonomous Reinforcement Learning Robot Control with Intel's Loihi 2 Neuromorphic Hardware 

**Authors**: Kenneth Stewart, Roxana Leontie, Samantha Chapin, Joe Hays, Sumit Bam Shrestha, Carl Glen Henshaw  

**Link**: [PDF](https://arxiv.org/pdf/2512.03911)  

**Abstract**: We present an end-to-end pipeline for deploying reinforcement learning (RL) trained Artificial Neural Networks (ANNs) on neuromorphic hardware by converting them into spiking Sigma-Delta Neural Networks (SDNNs). We demonstrate that an ANN policy trained entirely in simulation can be transformed into an SDNN compatible with Intel's Loihi 2 architecture, enabling low-latency and energy-efficient inference. As a test case, we use an RL policy for controlling the Astrobee free-flying robot, similar to a previously hardware in space-validated controller. The policy, trained with Rectified Linear Units (ReLUs), is converted to an SDNN and deployed on Intel's Loihi 2, then evaluated in NVIDIA's Omniverse Isaac Lab simulation environment for closed-loop control of Astrobee's motion. We compare execution performance between GPU and Loihi 2. The results highlight the feasibility of using neuromorphic platforms for robotic control and establish a pathway toward energy-efficient, real-time neuromorphic computation in future space and terrestrial robotics applications. 

---
# BERnaT: Basque Encoders for Representing Natural Textual Diversity 

**Authors**: Ekhi Azurmendi, Joseba Fernandez de Landa, Jaione Bengoetxea, Maite Heredia, Julen Etxaniz, Mikel Zubillaga, Ander Soraluze, Aitor Soroa  

**Link**: [PDF](https://arxiv.org/pdf/2512.03903)  

**Abstract**: Language models depend on massive text corpora that are often filtered for quality, a process that can unintentionally exclude non-standard linguistic varieties, reduce model robustness and reinforce representational biases. In this paper, we argue that language models should aim to capture the full spectrum of language variation (dialectal, historical, informal, etc.) rather than relying solely on standardized text. Focusing on Basque, a morphologically rich and low-resource language, we construct new corpora combining standard, social media, and historical sources, and pre-train the BERnaT family of encoder-only models in three configurations: standard, diverse, and combined. We further propose an evaluation framework that separates Natural Language Understanding (NLU) tasks into standard and diverse subsets to assess linguistic generalization. Results show that models trained on both standard and diverse data consistently outperform those trained on standard corpora, improving performance across all task types without compromising standard benchmark accuracy. These findings highlight the importance of linguistic diversity in building inclusive, generalizable language models. 

---
# Hyperdimensional Computing for Sustainable Manufacturing: An Initial Assessment 

**Authors**: Danny Hoang, Anandkumar Patel, Ruimen Chen, Rajiv Malhotra, Farhad Imani  

**Link**: [PDF](https://arxiv.org/pdf/2512.03864)  

**Abstract**: Smart manufacturing can significantly improve efficiency and reduce energy consumption, yet the energy demands of AI models may offset these gains. This study utilizes in-situ sensing-based prediction of geometric quality in smart machining to compare the energy consumption, accuracy, and speed of common AI models. HyperDimensional Computing (HDC) is introduced as an alternative, achieving accuracy comparable to conventional models while drastically reducing energy consumption, 200$\times$ for training and 175 to 1000$\times$ for inference. Furthermore, HDC reduces training times by 200$\times$ and inference times by 300 to 600$\times$, showcasing its potential for energy-efficient smart manufacturing. 

---
# Scalable Decision Focused Learning via Online Trainable Surrogates 

**Authors**: Gaetano Signorelli, Michele Lombardi  

**Link**: [PDF](https://arxiv.org/pdf/2512.03861)  

**Abstract**: Decision support systems often rely on solving complex optimization problems that may require to estimate uncertain parameters beforehand. Recent studies have shown how using traditionally trained estimators for this task can lead to suboptimal solutions. Using the actual decision cost as a loss function (called Decision Focused Learning) can address this issue, but with a severe loss of scalability at training time. To address this issue, we propose an acceleration method based on replacing costly loss function evaluations with an efficient surrogate. Unlike previously defined surrogates, our approach relies on unbiased estimators reducing the risk of spurious local optima and can provide information on its local confidence allowing one to switch to a fallback method when needed. Furthermore, the surrogate is designed for a black-box setting, which enables compensating for simplifications in the optimization model and account- ing for recourse actions during cost computation. In our results, the method reduces costly inner solver calls, with a solution quality comparable to other state-of-the-art techniques. 

---
# PULSE: A Unified Multi-Task Architecture for Cardiac Segmentation, Diagnosis, and Few-Shot Cross-Modality Clinical Adaptation 

**Authors**: Hania Ghouse, Maryam Alsharqi, Farhad R. Nezami, Muzammil Behzad  

**Link**: [PDF](https://arxiv.org/pdf/2512.03848)  

**Abstract**: Cardiac image analysis remains fragmented across tasks: anatomical segmentation, disease classification, and grounded clinical report generation are typically handled by separate networks trained under different data regimes. No existing framework unifies these objectives within a single architecture while retaining generalization across imaging modalities and datasets. We introduce PULSE, a multi-task vision-language framework built on self-supervised representations and optimized through a composite supervision strategy that balances region overlap learning, pixel wise classification fidelity, and boundary aware IoU refinement. A multi-scale token reconstruction decoder enables anatomical segmentation, while shared global representations support disease classification and clinically grounded text output allowing the model to transition from pixels to structures and finally clinical reasoning within one architecture. Unlike prior task-specific pipelines, PULSE learns task-invariant cardiac priors, generalizes robustly across datasets, and can be adapted to new imaging modalities with minimal supervision. This moves the field closer to a scalable, foundation style cardiac analysis framework. 

---
# DVPO: Distributional Value Modeling-based Policy Optimization for LLM Post-Training 

**Authors**: Dingwei Zhu, Zhiheng Xi, Shihan Dou, Yuhui Wang, Sixian Li, Junjie Ye, Honglin Guo, Shichun Liu, Chenhao Huang, Yajie Yang, Junlin Shang, Senjie Jin, Ming Zhang, Jiazheng Zhang, Caishuang Huang, Yunke Zhang, Demei Yan, Yuran Wang, Tao Gui  

**Link**: [PDF](https://arxiv.org/pdf/2512.03847)  

**Abstract**: Reinforcement learning (RL) has shown strong performance in LLM post-training, but real-world deployment often involves noisy or incomplete supervision. In such settings, complex and unreliable supervision signals can destabilize training and harm generalization. While existing approaches such as worst-case optimization (e.g., RFQI, CQL) and mean-based methods (e.g., PPO, GRPO) can improve stability, they often overlook generalization and may produce overly conservative policies, leading to uneven performance across diverse real scenarios. To this end, we introduce DVPO (Distributional Value Modeling with Risk-aware Policy Optimization), a new RL framework that combines conditional risk theory with distributional value modeling to better balance robustness and generalization. DVPO learns token-level value distributions to provide fine-grained supervision, and applies an asymmetric risk regularization to shape the distribution tails: it contracts the lower tail to dampen noisy negative deviations, while expanding the upper tail to preserve exploratory diversity. Across extensive experiments and analysis in multi-turn dialogue, math reasoning, and scientific QA, DVPO consistently outperforms PPO, GRPO, and robust Bellman-based PPO under noisy supervision, showing its potential for LLM post-training in the real-world. 

---
# MPCFormer: A physics-informed data-driven approach for explainable socially-aware autonomous driving 

**Authors**: Jia Hu, Zhexi Lian, Xuerun Yan, Ruiang Bi, Dou Shen, Yu Ruan, Haoran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.03795)  

**Abstract**: Autonomous Driving (AD) vehicles still struggle to exhibit human-like behavior in highly dynamic and interactive traffic scenarios. The key challenge lies in AD's limited ability to interact with surrounding vehicles, largely due to a lack of understanding the underlying mechanisms of social interaction. To address this issue, we introduce MPCFormer, an explainable socially-aware autonomous driving approach with physics-informed and data-driven coupled social interaction dynamics. In this model, the dynamics are formulated into a discrete space-state representation, which embeds physics priors to enhance modeling explainability. The dynamics coefficients are learned from naturalistic driving data via a Transformer-based encoder-decoder architecture. To the best of our knowledge, MPCFormer is the first approach to explicitly model the dynamics of multi-vehicle social interactions. The learned social interaction dynamics enable the planner to generate manifold, human-like behaviors when interacting with surrounding traffic. By leveraging the MPC framework, the approach mitigates the potential safety risks typically associated with purely learning-based methods. Open-looped evaluation on NGSIM dataset demonstrates that MPCFormer achieves superior social interaction awareness, yielding the lowest trajectory prediction errors compared with other state-of-the-art approach. The prediction achieves an ADE as low as 0.86 m over a long prediction horizon of 5 seconds. Close-looped experiments in highly intense interaction scenarios, where consecutive lane changes are required to exit an off-ramp, further validate the effectiveness of MPCFormer. Results show that MPCFormer achieves the highest planning success rate of 94.67%, improves driving efficiency by 15.75%, and reduces the collision rate from 21.25% to 0.5%, outperforming a frontier Reinforcement Learning (RL) based planner. 

---
# AdaptVision: Efficient Vision-Language Models via Adaptive Visual Acquisition 

**Authors**: Zichuan Lin, Yicheng Liu, Yang Yang, Lvfang Tao, Deheng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2512.03794)  

**Abstract**: Vision-Language Models (VLMs) have achieved remarkable success in visual question answering tasks, but their reliance on large numbers of visual tokens introduces significant computational overhead. While existing efficient VLM approaches reduce visual tokens through fixed-ratio compression, they operate passively and lack the ability to adapt to varying task requirements. This motivates a fundamental question: Can VLMs autonomously determine the minimum number of visual tokens required for each sample? Inspired by human active vision mechanisms, we introduce AdaptVision, an efficient VLM paradigm that enables adaptive visual token acquisition through a coarse-to-fine approach. Our model initially processes compressed visual tokens from low-resolution images and selectively acquires additional visual information by invoking a bounding box tool to crop key regions when necessary. We train AdaptVision using a reinforcement learning framework that carefully balances accuracy and efficiency. Central to our approach is Decoupled Turn Policy Optimization (DTPO), which decouples the learning objective into two components: (1) tool learning, which optimizes correct tool utilization, and (2) accuracy improvement, which refines the generated responses to improve answer correctness. Based on this formulation, we further decouple advantage estimation by computing separate advantages for tokens associated with each objective. This formulation enables more effective optimization for AdaptVision compared to vanilla GRPO. Comprehensive experiments across multiple VQA benchmarks demonstrate that AdaptVision achieves superior performance while consuming substantially fewer visual tokens than state-of-the-art efficient VLM methods. 

---
# Bayesian Optimization for Automatic Tuning of Torque-Level Nonlinear Model Predictive Control 

**Authors**: Gabriele Fadini, Deepak Ingole, Tong Duy Son, Alisa Rupenyan  

**Link**: [PDF](https://arxiv.org/pdf/2512.03772)  

**Abstract**: This paper presents an auto-tuning framework for torque-based Nonlinear Model Predictive Control (nMPC), where the MPC serves as a real-time controller for optimal joint torque commands. The MPC parameters, including cost function weights and low-level controller gains, are optimized using high-dimensional Bayesian Optimization (BO) techniques, specifically Sparse Axis-Aligned Subspace (SAASBO) with a digital twin (DT) to achieve precise end-effector trajectory real-time tracking on an UR10e robot arm. The simulation model allows efficient exploration of the high-dimensional parameter space, and it ensures safe transfer to hardware. Our simulation results demonstrate significant improvements in tracking performance (+41.9%) and reduction in solve times (-2.5%) compared to manually-tuned parameters. Moreover, experimental validation on the real robot follows the trend (with a +25.8% improvement), emphasizing the importance of digital twin-enabled automated parameter optimization for robotic operations. 

---
# In-Context Representation Hijacking 

**Authors**: Itay Yona, Amir Sarid, Michael Karasik, Yossi Gandelsman  

**Link**: [PDF](https://arxiv.org/pdf/2512.03771)  

**Abstract**: We introduce \textbf{Doublespeak}, a simple \emph{in-context representation hijacking} attack against large language models (LLMs). The attack works by systematically replacing a harmful keyword (e.g., \textit{bomb}) with a benign token (e.g., \textit{carrot}) across multiple in-context examples, provided a prefix to a harmful request. We demonstrate that this substitution leads to the internal representation of the benign token converging toward that of the harmful one, effectively embedding the harmful semantics under a euphemism. As a result, superficially innocuous prompts (e.g., ``How to build a carrot?'') are internally interpreted as disallowed instructions (e.g., ``How to build a bomb?''), thereby bypassing the model's safety alignment. We use interpretability tools to show that this semantic overwrite emerges layer by layer, with benign meanings in early layers converging into harmful semantics in later ones. Doublespeak is optimization-free, broadly transferable across model families, and achieves strong success rates on closed-source and open-source systems, reaching 74\% ASR on Llama-3.3-70B-Instruct with a single-sentence context override. Our findings highlight a new attack surface in the latent space of LLMs, revealing that current alignment strategies are insufficient and should instead operate at the representation level. 

---
# Principled RL for Diffusion LLMs Emerges from a Sequence-Level Perspective 

**Authors**: Jingyang Ou, Jiaqi Han, Minkai Xu, Shaoxuan Xu, Jianwen Xie, Stefano Ermon, Yi Wu, Chongxuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.03759)  

**Abstract**: Reinforcement Learning (RL) has proven highly effective for autoregressive language models, but adapting these methods to diffusion large language models (dLLMs) presents fundamental challenges. The core difficulty lies in likelihood approximation: while autoregressive models naturally provide token-level conditional probabilities essential for token-level RL objectives (e.g., GRPO), dLLMs generate sequences through iterative non-autoregressive denoising steps that lack this factorization. To address this fundamental mismatch, we propose ELBO-based Sequence-level Policy Optimization (ESPO), a principled RL framework that treats entire sequence generation as a single action and uses the ELBO as a tractable sequence-level likelihood proxy. Our method incorporates per-token normalization of importance ratios and robust KL-divergence estimation to ensure stable large-scale training. Extensive experiments on mathematical reasoning, coding, and planning tasks demonstrate that ESPO significantly outperforms token-level baselines, achieving dramatic improvements of 20-40 points on the Countdown task, while maintaining consistent gains on math and coding benchmarks. Our approach establishes sequence-level optimization as a principled and empirically effective paradigm for RL in dLLMs. Our code is available at this https URL. 

---
# Research on Brain Tumor Classification Method Based on Improved ResNet34 Network 

**Authors**: Yufeng Li, Wenchao Zhao, Bo Dang, Weimin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.03751)  

**Abstract**: Previously, image interpretation in radiology relied heavily on manual methods. However, manual classification of brain tumor medical images is time-consuming and labor-intensive. Even with shallow convolutional neural network models, the accuracy is not ideal. To improve the efficiency and accuracy of brain tumor image classification, this paper proposes a brain tumor classification model based on an improved ResNet34 network. This model uses the ResNet34 residual network as the backbone network and incorporates multi-scale feature extraction. It uses a multi-scale input module as the first layer of the ResNet34 network and an Inception v2 module as the residual downsampling layer. Furthermore, a channel attention mechanism module assigns different weights to different channels of the image from a channel domain perspective, obtaining more important feature information. The results after a five-fold crossover experiment show that the average classification accuracy of the improved network model is approximately 98.8%, which is not only 1% higher than ResNet34, but also only 80% of the number of parameters of the original model. Therefore, the improved network model not only improves accuracy but also reduces clutter, achieving a classification effect with fewer parameters and higher accuracy. 

---
# Out-of-the-box: Black-box Causal Attacks on Object Detectors 

**Authors**: Melane Navaratnarajah, David A. Kelly, Hana Chockler  

**Link**: [PDF](https://arxiv.org/pdf/2512.03730)  

**Abstract**: Adversarial perturbations are a useful way to expose vulnerabilities in object detectors. Existing perturbation methods are frequently white-box and architecture specific. More importantly, while they are often successful, it is rarely clear why they work. Insights into the mechanism of this success would allow developers to understand and analyze these attacks, as well as fine-tune the model to prevent them. This paper presents BlackCAtt, a black-box algorithm and a tool, which uses minimal, causally sufficient pixel sets to construct explainable, imperceptible, reproducible, architecture-agnostic attacks on object detectors. BlackCAtt combines causal pixels with bounding boxes produced by object detectors to create adversarial attacks that lead to the loss, modification or addition of a bounding box. BlackCAtt works across different object detectors of different sizes and architectures, treating the detector as a black box. We compare the performance of BlackCAtt with other black-box attack methods and show that identification of causal pixels leads to more precisely targeted and less perceptible attacks. On the COCO test dataset, our approach is 2.7 times better than the baseline in removing a detection, 3.86 times better in changing a detection, and 5.75 times better in triggering new, spurious, detections. The attacks generated by BlackCAtt are very close to the original image, and hence imperceptible, demonstrating the power of causal pixels. 

---
# AI/ML in 3GPP 5G Advanced - Services and Architecture 

**Authors**: Pradnya Taksande, Shwetha Kiran, Pranav Jha, Prasanna Chaporkar  

**Link**: [PDF](https://arxiv.org/pdf/2512.03728)  

**Abstract**: The 3rd Generation Partnership Project (3GPP), the standards body for mobile networks, is in the final phase of Release 19 standardization and is beginning Release 20. Artificial Intelligence/ Machine Learning (AI/ML) has brought about a paradigm shift in technology and it is being adopted across industries and verticals. 3GPP has been integrating AI/ML into the 5G advanced system since Release 18. This paper focuses on the AI/ML related technological advancements and features introduced in Release 19 within the Service and System Aspects (SA) Technical specifications group of 3GPP. The advancements relate to two paradigms: (i) enhancements that AI/ML brought to the 5G advanced system (AI for network), e.g. resource optimization, and (ii) enhancements that were made to the 5G system to support AI/ML applications (Network for AI), e.g. image recognition. 

---
# Context-Aware Hierarchical Learning: A Two-Step Paradigm towards Safer LLMs 

**Authors**: Tengyun Ma, Jiaqi Yao, Daojing He, Shihao Peng, Yu Li, Shaohui Liu, Zhuotao Tian  

**Link**: [PDF](https://arxiv.org/pdf/2512.03720)  

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools for diverse applications. However, their uniform token processing paradigm introduces critical vulnerabilities in instruction handling, particularly when exposed to adversarial scenarios. In this work, we identify and propose a novel class of vulnerabilities, termed Tool-Completion Attack (TCA), which exploits function-calling mechanisms to subvert model behavior. To evaluate LLM robustness against such threats, we introduce the Tool-Completion benchmark, a comprehensive security assessment framework, which reveals that even state-of-the-art models remain susceptible to TCA, with surprisingly high attack success rates. To address these vulnerabilities, we introduce Context-Aware Hierarchical Learning (CAHL), a sophisticated mechanism that dynamically balances semantic comprehension with role-specific instruction constraints. CAHL leverages the contextual correlations between different instruction segments to establish a robust, context-aware instruction hierarchy. Extensive experiments demonstrate that CAHL significantly enhances LLM robustness against both conventional attacks and the proposed TCA, exhibiting strong generalization capabilities in zero-shot evaluations while still preserving model performance on generic tasks. Our code is available at this https URL. 

---
# Over-the-Air Federated Learning: Rethinking Edge AI Through Signal Processing 

**Authors**: Seyed Mohammad Azimi-Abarghouyi, Carlo Fischione, Kaibin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2512.03719)  

**Abstract**: Over-the-Air Federated Learning (AirFL) is an emerging paradigm that tightly integrates wireless signal processing and distributed machine learning to enable scalable AI at the network edge. By leveraging the superposition property of wireless signals, AirFL performs communication and model aggregation of the learning process simultaneously, significantly reducing latency, bandwidth, and energy consumption. This article offers a tutorial treatment of AirFL, presenting a novel classification into three design approaches: CSIT-aware, blind, and weighted AirFL. We provide a comprehensive guide to theoretical foundations, performance analysis, complexity considerations, practical limitations, and prospective research directions. 

---
# Matrix Editing Meets Fair Clustering: Parameterized Algorithms and Complexity 

**Authors**: Robert Ganian, Hung P. Hoang, Simon Wietheger  

**Link**: [PDF](https://arxiv.org/pdf/2512.03718)  

**Abstract**: We study the computational problem of computing a fair means clustering of discrete vectors, which admits an equivalent formulation as editing a colored matrix into one with few distinct color-balanced rows by changing at most $k$ values. While NP-hard in both the fairness-oblivious and the fair settings, the problem is well-known to admit a fixed-parameter algorithm in the former ``vanilla'' setting. As our first contribution, we exclude an analogous algorithm even for highly restricted fair means clustering instances. We then proceed to obtain a full complexity landscape of the problem, and establish tractability results which capture three means of circumventing our obtained lower bound: placing additional constraints on the problem instances, fixed-parameter approximation, or using an alternative parameterization targeting tree-like matrices. 

---
# Quantum Topological Graph Neural Networks for Detecting Complex Fraud Patterns 

**Authors**: Mohammad Doost, Mohammad Manthouri  

**Link**: [PDF](https://arxiv.org/pdf/2512.03696)  

**Abstract**: We propose a novel QTGNN framework for detecting fraudulent transactions in large-scale financial networks. By integrating quantum embedding, variational graph convolutions, and topological data analysis, QTGNN captures complex transaction dynamics and structural anomalies indicative of fraud. The methodology includes quantum data embedding with entanglement enhancement, variational quantum graph convolutions with non-linear dynamics, extraction of higher-order topological invariants, hybrid quantum-classical anomaly learning with adaptive optimization, and interpretable decision-making via topological attribution. Rigorous convergence guarantees ensure stable training on noisy intermediate-scale quantum (NISQ) devices, while stability of topological signatures provides robust fraud detection. Optimized for NISQ hardware with circuit simplifications and graph sampling, the framework scales to large transaction networks. Simulations on financial datasets, such as PaySim and Elliptic, benchmark QTGNN against classical and quantum baselines, using metrics like ROC-AUC, precision, and false positive rate. An ablation study evaluates the contributions of quantum embeddings, topological features, non-linear channels, and hybrid learning. QTGNN offers a theoretically sound, interpretable, and practical solution for financial fraud detection, bridging quantum machine learning, graph theory, and topological analysis. 

---
# ToG-Bench: Task-Oriented Spatio-Temporal Grounding in Egocentric Videos 

**Authors**: Qi'ao Xu, Tianwen Qian, Yuqian Fu, Kailing Li, Yang Jiao, Jiacheng Zhang, Xiaoling Wang, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2512.03666)  

**Abstract**: A core capability towards general embodied intelligence lies in localizing task-relevant objects from an egocentric perspective, formulated as Spatio-Temporal Video Grounding (STVG). Despite recent progress, existing STVG studies remain largely confined to object-centric and descriptive instructions, neglecting the task-oriented reasoning that is crucial for embodied agents to accomplish goal-directed interactions. To bridge this gap, we introduce \textbf{ToG-Bench}, the first task-oriented spatio-temporal video grounding benchmark for egocentric videos. ToG-Bench is characterized by three key features: (1) \textbf{Task-oriented Grounding}, which requires identifying and localizing objects based on intended tasks rather than straightforward descriptions; (2) \textbf{Explicit-Implicit Dual Grounding}, where target objects can be either explicitly mentioned or implicitly inferred by contextual reasoning; (3) \textbf{One-to-Many Grounding}, where a single instruction may correspond to multiple objects involved in task execution. Built upon videos sourced from ScanNet, ToG-Bench comprises 100 annotated clips with 2,704 task-oriented grounding instructions, constructed via a semi-automated pipeline that combines foundation model annotation and human refinement. In addition, we introduce a set of task-level evaluation metrics tailored for multi-object and explicit-implicit object grounding, and systematically benchmark seven state-of-the-art MLLMs. Extensive experiments reveal the intrinsic challenges of task-oriented STVG and substantial performance gaps across explicit-implicit and multi-object grounding, highlighting the difficulty of bridging perception and interaction in embodied scenarios. Data and code will be released at: \href{this https URL}{this https URL}.. 

---
# Dynamically Scaled Activation Steering 

**Authors**: Alex Ferrando, Xavier Suau, Jordi Gonzàlez, Pau Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2512.03661)  

**Abstract**: Activation steering has emerged as a powerful method for guiding the behavior of generative models towards desired outcomes such as toxicity mitigation. However, most existing methods apply interventions uniformly across all inputs, degrading model performance when steering is unnecessary. We introduce Dynamically Scaled Activation Steering (DSAS), a method-agnostic steering framework that decouples when to steer from how to steer. DSAS adaptively modulates the strength of existing steering transformations across layers and inputs, intervening strongly only when undesired behavior is detected. At generation time, DSAS computes context-dependent scaling factors that selectively adjust the strength of any steering method. We also show how DSAS can be jointly optimized end-to-end together with the steering function. When combined with existing steering methods, DSAS consistently improves the Pareto front with respect to steering alone, achieving a better trade-off between toxicity mitigation and utility preservation. We further demonstrate DSAS's generality by applying it to a text-to-image diffusion model, showing how adaptive steering allows the modulation of specific concepts. Finally, DSAS introduces minimal computational overhead while improving interpretability, pinpointing which tokens require steering and by how much. 

---
# MKSNet: Advanced Small Object Detection in Remote Sensing Imagery with Multi-Kernel and Dual Attention Mechanisms 

**Authors**: Jiahao Zhang, Xiao Zhao, Guangyu Gao  

**Link**: [PDF](https://arxiv.org/pdf/2512.03640)  

**Abstract**: Deep convolutional neural networks (DCNNs) have substantially advanced object detection capabilities, particularly in remote sensing imagery. However, challenges persist, especially in detecting small objects where the high resolution of these images and the small size of target objects often result in a loss of critical information in the deeper layers of conventional CNNs. Additionally, the extensive spatial redundancy and intricate background details typical in remote-sensing images tend to obscure these small targets. To address these challenges, we introduce Multi-Kernel Selection Network (MKSNet), a novel network architecture featuring a novel Multi-Kernel Selection mechanism. The MKS mechanism utilizes large convolutional kernels to effectively capture an extensive range of contextual information. This innovative design allows for adaptive kernel size selection, significantly enhancing the network's ability to dynamically process and emphasize crucial spatial details for small object detection. Furthermore, MKSNet also incorporates a dual attention mechanism, merging spatial and channel attention modules. The spatial attention module adaptively fine-tunes the spatial weights of feature maps, focusing more intensively on relevant regions while mitigating background noise. Simultaneously, the channel attention module optimizes channel information selection, improving feature representation and detection accuracy. Empirical evaluations on the DOTA-v1.0 and HRSC2016 benchmark demonstrate that MKSNet substantially surpasses existing state-of-the-art models in detecting small objects in remote sensing images. These results highlight MKSNet's superior ability to manage the complexities associated with multi-scale and high-resolution image data, confirming its effectiveness and innovation in remote sensing object detection. 

---
# AlignCheck: a Semantic Open-Domain Metric for Factual Consistency Assessment 

**Authors**: Ahmad Aghaebrahimian  

**Link**: [PDF](https://arxiv.org/pdf/2512.03634)  

**Abstract**: Large Language Models have significantly advanced natural language processing tasks, but remain prone to generating incorrect or misleading but plausible arguments. This issue, known as hallucination, is particularly concerning in high-stakes domains like clinical applications, where factual inaccuracies can have severe consequences. Existing evaluation metrics fail to adequately assess factual consistency and lack interpretability, making diagnosing and mitigating errors difficult. We propose an interpretable framework for factual consistency assessment for in-domain and open-domain texts to address these limitations. Our approach decomposes text into atomic facts and introduces a flexible, schema-free methodology. Unlike previous methods with an absolute metric, we incorporate a weighted metric to enhance factual evaluation. Additionally, we propose a mechanism to control assessment complexity in intricate domains. We benchmark our approach on popular general and clinical datasets and release our code to support fact-aware model training in future research. 

---
# The promising potential of vision language models for the generation of textual weather forecasts 

**Authors**: Edward C. C. Steele, Dinesh Mane, Emilio Monti, Luis Orus, Rebecca Chantrill-Cheyette, Matthew Couch, Kirstine I. Dale, Simon Eaton, Govindarajan Rangarajan, Amir Majlesi, Steven Ramsdale, Michael Sharpe, Craig Smith, Jonathan Smith, Rebecca Yates, Holly Ellis, Charles Ewen  

**Link**: [PDF](https://arxiv.org/pdf/2512.03623)  

**Abstract**: Despite the promising capability of multimodal foundation models, their application to the generation of meteorological products and services remains nascent. To accelerate aspiration and adoption, we explore the novel use of a vision language model for writing the iconic Shipping Forecast text directly from video-encoded gridded weather data. These early results demonstrate promising scalable technological opportunities for enhancing production efficiency and service innovation within the weather enterprise and beyond. 

---
# SELF: A Robust Singular Value and Eigenvalue Approach for LLM Fingerprinting 

**Authors**: Hanxiu Zhang, Yue Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2512.03620)  

**Abstract**: The protection of Intellectual Property (IP) in Large Language Models (LLMs) represents a critical challenge in contemporary AI research. While fingerprinting techniques have emerged as a fundamental mechanism for detecting unauthorized model usage, existing methods -- whether behavior-based or structural -- suffer from vulnerabilities such as false claim attacks or susceptible to weight manipulations. To overcome these limitations, we propose SELF, a novel intrinsic weight-based fingerprinting scheme that eliminates dependency on input and inherently resists false claims. SELF achieves robust IP protection through two key innovations: 1) unique, scalable and transformation-invariant fingerprint extraction via singular value and eigenvalue decomposition of LLM attention weights, and 2) effective neural network-based fingerprint similarity comparison based on few-shot learning and data augmentation. Experimental results demonstrate SELF maintains high IP infringement detection accuracy while showing strong robustness against various downstream modifications, including quantization, pruning, and fine-tuning attacks. Our code is available at this https URL. 

---
# KVNAND: Efficient On-Device Large Language Model Inference Using DRAM-Free In-Flash Computing 

**Authors**: Lishuo Deng, Shaojie Xu, Jinwu Chen, Changwei Yan, Jiajie Wang, Zhe Jiang, Weiwei Shan  

**Link**: [PDF](https://arxiv.org/pdf/2512.03608)  

**Abstract**: Deploying large language models (LLMs) on edge devices enables personalized agents with strong privacy and low cost. However, with tens to hundreds of billions of parameters, single-batch autoregressive inference suffers from extremely low arithmetic intensity, creating severe weight-loading and bandwidth pressures on resource-constrained platforms. Recent in-flash computing (IFC) solutions alleviate this bottleneck by co-locating weight-related linear computations in the decode phase with flash, yet still rely on DRAM for the key-value (KV) cache. As context length grows, the KV cache can exceed model weights in size, imposing prohibitive DRAM cost and capacity requirements. Attempts to offload KV cache to flash suffer from severe performance penalties.
We propose KVNAND, the first DRAM-free, IFC-based architecture that stores both model weights and KV cache entirely in compute-enabled 3D NAND flash. KVNAND addresses the fundamental performance challenges of flash under intensive KV cache access by leveraging IFC for all memory-bound operations to reduce data transfer overhead, introducing head-group parallelism to boost throughput, and employing page-level KV cache mapping to align token access patterns with flash organization. In addition, we propose a design space exploration framework that evaluates discrete and compact KVNAND variants to balance weight and KV placement, automatically identifying the optimal design trade-off. These techniques mitigate latency, energy, and reliability concerns, turning flash into a practical medium for long-context KV storage. Evaluations on MHA 7B and GQA 70B LLMs show that KVNAND achieves 1.98\(\times\)/1.94\(\times\)/2.05\(\times\) geomean speedup at 128/1K/10K-token contexts compared to DRAM-equipped IFC designs and addresses out-of-memory failures at 100K context length. 

---
# Fine-grained Narrative Classification in Biased News Articles 

**Authors**: Zeba Afroz, Harsh Vardhan, Pawan Bhakuni, Aanchal Punia, Rajdeep Kumar, Md. Shad Akhtar  

**Link**: [PDF](https://arxiv.org/pdf/2512.03582)  

**Abstract**: Narratives are the cognitive and emotional scaffolds of propaganda. They organize isolated persuasive techniques into coherent stories that justify actions, attribute blame, and evoke identification with ideological camps. In this paper, we propose a novel fine-grained narrative classification in biased news articles. We also explore article-bias classification as the precursor task to narrative classification and fine-grained persuasive technique identification. We develop INDI-PROP, the first ideologically grounded fine-grained narrative dataset with multi-level annotation for analyzing propaganda in Indian news media. Our dataset INDI-PROP comprises 1,266 articles focusing on two polarizing socio-political events in recent times: CAA and the Farmers' protest. Each article is annotated at three hierarchical levels: (i) ideological article-bias (pro-government, pro-opposition, neutral), (ii) event-specific fine-grained narrative frames anchored in ideological polarity and communicative intent, and (iii) persuasive techniques. We propose FANTA and TPTC, two GPT-4o-mini guided multi-hop prompt-based reasoning frameworks for the bias, narrative, and persuasive technique classification. FANTA leverages multi-layered communicative phenomena by integrating information extraction and contextual framing for hierarchical reasoning. On the other hand, TPTC adopts systematic decomposition of persuasive cues via a two-stage approach. Our evaluation suggests substantial improvement over underlying baselines in each case. 

---
# When, How Long and How Much? Interpretable Neural Networks for Time Series Regression by Learning to Mask and Aggregate 

**Authors**: Florent Forest, Amaury Wei, Olga Fink  

**Link**: [PDF](https://arxiv.org/pdf/2512.03578)  

**Abstract**: Time series extrinsic regression (TSER) refers to the task of predicting a continuous target variable from an input time series. It appears in many domains, including healthcare, finance, environmental monitoring, and engineering. In these settings, accurate predictions and trustworthy reasoning are both essential. Although state-of-the-art TSER models achieve strong predictive performance, they typically operate as black boxes, making it difficult to understand which temporal patterns drive their decisions. Post-hoc interpretability techniques, such as feature attribution, aim to to explain how the model arrives at its predictions, but often produce coarse, noisy, or unstable explanations. Recently, inherently interpretable approaches based on concepts, additive decompositions, or symbolic regression, have emerged as promising alternatives. However, these approaches remain limited: they require explicit supervision on the concepts themselves, often cannot capture interactions between time-series features, lack expressiveness for complex temporal patterns, and struggle to scale to high-dimensional multivariate data.
To address these limitations, we propose MAGNETS (Mask-and-AGgregate NEtwork for Time Series), an inherently interpretable neural architecture for TSER. MAGNETS learns a compact set of human-understandable concepts without requiring any annotations. Each concept corresponds to a learned, mask-based aggregation over selected input features, explicitly revealing both which features drive predictions and when they matter in the sequence. Predictions are formed as combinations of these learned concepts through a transparent, additive structure, enabling clear insight into the model's decision process. 

---
# Machine Learning to Predict Slot Usage in TSCH Wireless Sensor Networks 

**Authors**: Stefano Scanzio, Gabriele Formis, Tullio Facchinetti, Gianluca Cena  

**Link**: [PDF](https://arxiv.org/pdf/2512.03570)  

**Abstract**: Wireless sensor networks (WSNs) are employed across a wide range of industrial applications where ultra-low power consumption is a critical prerequisite. At the same time, these systems must maintain a certain level of determinism to ensure reliable and predictable operation. In this view, time slotted channel hopping (TSCH) is a communication technology that meets both conditions, making it an attractive option for its usage in industrial WSNs. This work proposes the use of machine learning to learn the traffic pattern generated in networks based on the TSCH protocol, in order to turn nodes into a deep sleep state when no transmission is planned and thus to improve the energy efficiency of the WSN. The ability of machine learning models to make good predictions at different network levels in a typical tree network topology was analyzed in depth, showing how their capabilities degrade while approaching the root of the tree. The application of these models on simulated data based on an accurate modeling of wireless sensor nodes indicates that the investigated algorithms can be suitably used to further and substantially reduce the power consumption of a TSCH network. 

---
# State Space Models for Bioacoustics: A comparative Evaluation with Transformers 

**Authors**: Chengyu Tang, Sanjeev Baskiyar  

**Link**: [PDF](https://arxiv.org/pdf/2512.03563)  

**Abstract**: In this study, we evaluate the efficacy of the Mamba model in the field of bioacoustics. We first pretrain a Mamba-based audio large language model (LLM) on a large corpus of audio data using self-supervised learning. We fine-tune and evaluate BioMamba on the BEANS benchmark, a collection of diverse bioacoustic tasks including classification and detection, and compare its performance and efficiency with multiple baseline models, including AVES, a state-of-the-art Transformer-based model. The results show that BioMamba achieves comparable performance with AVES while consumption significantly less VRAM, demonstrating its potential in this domain. 

---
# Dynamic Content Moderation in Livestreams: Combining Supervised Classification with MLLM-Boosted Similarity Matching 

**Authors**: Wei Chee Yew, Hailun Xu, Sanjay Saha, Xiaotian Fan, Hiok Hian Ong, David Yuchen Wang, Kanchan Sarkar, Zhenheng Yang, Danhui Guan  

**Link**: [PDF](https://arxiv.org/pdf/2512.03553)  

**Abstract**: Content moderation remains a critical yet challenging task for large-scale user-generated video platforms, especially in livestreaming environments where moderation must be timely, multimodal, and robust to evolving forms of unwanted content. We present a hybrid moderation framework deployed at production scale that combines supervised classification for known violations with reference-based similarity matching for novel or subtle cases. This hybrid design enables robust detection of both explicit violations and novel edge cases that evade traditional classifiers. Multimodal inputs (text, audio, visual) are processed through both pipelines, with a multimodal large language model (MLLM) distilling knowledge into each to boost accuracy while keeping inference lightweight. In production, the classification pipeline achieves 67% recall at 80% precision, and the similarity pipeline achieves 76% recall at 80% precision. Large-scale A/B tests show a 6-8% reduction in user views of unwanted livestreams}. These results demonstrate a scalable and adaptable approach to multimodal content governance, capable of addressing both explicit violations and emerging adversarial behaviors. 

---
# A Learning-based Control Methodology for Transitioning VTOL UAVs 

**Authors**: Zexin Lin, Yebin Zhong, Hanwen Wan, Jiu Cheng, Zhenglong Sun, Xiaoqiang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2512.03548)  

**Abstract**: Transition control poses a critical challenge in Vertical Take-Off and Landing Unmanned Aerial Vehicle (VTOL UAV) development due to the tilting rotor mechanism, which shifts the center of gravity and thrust direction during transitions. Current control methods' decoupled control of altitude and position leads to significant vibration, and limits interaction consideration and adaptability. In this study, we propose a novel coupled transition control methodology based on reinforcement learning (RL) driven controller. Besides, contrasting to the conventional phase-transition approach, the ST3M method demonstrates a new perspective by treating cruise mode as a special case of hover. We validate the feasibility of applying our method in simulation and real-world environments, demonstrating efficient controller development and migration while accurately controlling UAV position and attitude, exhibiting outstanding trajectory tracking and reduced vibrations during the transition process. 

---
# V-ITI: Mitigating Hallucinations in Multimodal Large Language Models via Visual Inference-Time Intervention 

**Authors**: Nan Sun, Zhenyu Zhang, Xixun Lin, Kun Wang, Yanmin Shang, Naibin Gu, Shuohuan Wang, Yu Sun, Hua Wu, Haifeng Wang, Yanan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2512.03542)  

**Abstract**: Multimodal Large Language Models (MLLMs) excel in numerous vision-language tasks yet suffer from hallucinations, producing content inconsistent with input visuals, that undermine reliability in precision-sensitive domains. This issue stems from a fundamental problem of visual neglect, where models fail to adequately prioritize input images. Existing methods typically alleviate hallucinations by intervening in the attention score or output logits, focusing on "how to intervene" but overlooking the prerequisite "when to intervene", which leads to the "over-intervention" problem and subsequently introduces new hallucinations and unnecessary computational overhead. To address this gap, we first investigate the mechanism of visual neglect and reveal it can be accurately detected via head-level activation patterns in MLLMs. We thus propose V-ITI, a lightweight visual inference-time intervention framework integrating a Visual Neglect Detector that identifies visual neglect via head-level discriminative probes and a Visual Recall Intervenor that modulates activations with prestored visual activation information only when the visual neglect is detected. Extensive experiments across eight benchmarks and different MLLM families demonstrate that V-ITI consistently mitigates vision-related hallucinations while preserving general task performance. 

---
# CookAnything: A Framework for Flexible and Consistent Multi-Step Recipe Image Generation 

**Authors**: Ruoxuan Zhang, Bin Wen, Hongxia Xie, Yi Yao, Songhan Zuo, Jian-Yu Jiang-Lin, Hong-Han Shuai, Wen-Huang Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2512.03540)  

**Abstract**: Cooking is a sequential and visually grounded activity, where each step such as chopping, mixing, or frying carries both procedural logic and visual semantics. While recent diffusion models have shown strong capabilities in text-to-image generation, they struggle to handle structured multi-step scenarios like recipe illustration. Additionally, current recipe illustration methods are unable to adjust to the natural variability in recipe length, generating a fixed number of images regardless of the actual instructions structure. To address these limitations, we present CookAnything, a flexible and consistent diffusion-based framework that generates coherent, semantically distinct image sequences from textual cooking instructions of arbitrary length. The framework introduces three key components: (1) Step-wise Regional Control (SRC), which aligns textual steps with corresponding image regions within a single denoising process; (2) Flexible RoPE, a step-aware positional encoding mechanism that enhances both temporal coherence and spatial diversity; and (3) Cross-Step Consistency Control (CSCC), which maintains fine-grained ingredient consistency across steps. Experimental results on recipe illustration benchmarks show that CookAnything performs better than existing methods in training-based and training-free settings. The proposed framework supports scalable, high-quality visual synthesis of complex multi-step instructions and holds significant potential for broad applications in instructional media, and procedural content creation. 

---
# Rethinking Prompt Design for Inference-time Scaling in Text-to-Visual Generation 

**Authors**: Subin Kim, Sangwoo Mo, Mamshad Nayeem Rizve, Yiran Xu, Difan Liu, Jinwoo Shin, Tobias Hinz  

**Link**: [PDF](https://arxiv.org/pdf/2512.03534)  

**Abstract**: Achieving precise alignment between user intent and generated visuals remains a central challenge in text-to-visual generation, as a single attempt often fails to produce the desired output. To handle this, prior approaches mainly scale the visual generation process (e.g., increasing sampling steps or seeds), but this quickly leads to a quality plateau. This limitation arises because the prompt, crucial for guiding generation, is kept fixed. To address this, we propose Prompt Redesign for Inference-time Scaling, coined PRIS, a framework that adaptively revises the prompt during inference in response to the scaled visual generations. The core idea of PRIS is to review the generated visuals, identify recurring failure patterns across visuals, and redesign the prompt accordingly before regenerating the visuals with the revised prompt. To provide precise alignment feedback for prompt revision, we introduce a new verifier, element-level factual correction, which evaluates the alignment between prompt attributes and generated visuals at a fine-grained level, achieving more accurate and interpretable assessments than holistic measures. Extensive experiments on both text-to-image and text-to-video benchmarks demonstrate the effectiveness of our approach, including a 15% gain on VBench 2.0. These results highlight that jointly scaling prompts and visuals is key to fully leveraging scaling laws at inference-time. Visualizations are available at the website: this https URL. 

---
# M3DR: Towards Universal Multilingual Multimodal Document Retrieval 

**Authors**: Adithya S Kolavi, Vyoman Jain  

**Link**: [PDF](https://arxiv.org/pdf/2512.03514)  

**Abstract**: Multimodal document retrieval systems have shown strong progress in aligning visual and textual content for semantic search. However, most existing approaches remain heavily English-centric, limiting their effectiveness in multilingual contexts. In this work, we present M3DR (Multilingual Multimodal Document Retrieval), a framework designed to bridge this gap across languages, enabling applicability across diverse linguistic and cultural contexts. M3DR leverages synthetic multilingual document data and generalizes across different vision-language architectures and model sizes, enabling robust cross-lingual and cross-modal alignment. Using contrastive training, our models learn unified representations for text and document images that transfer effectively across languages. We validate this capability on 22 typologically diverse languages, demonstrating consistent performance and adaptability across linguistic and script variations. We further introduce a comprehensive benchmark that captures real-world multilingual scenarios, evaluating models under monolingual, multilingual, and mixed-language settings. M3DR generalizes across both single dense vector and ColBERT-style token-level multi-vector retrieval paradigms. Our models, NetraEmbed and ColNetraEmbed achieve state-of-the-art performance with ~150% relative improvements on cross-lingual retrieval. 

---
# Physics-Driven Learning Framework for Tomographic Tactile Sensing 

**Authors**: Xuanxuan Yang, Xiuyang Zhang, Haofeng Chen, Gang Ma, Xiaojie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.03512)  

**Abstract**: Electrical impedance tomography (EIT) provides an attractive solution for large-area tactile sensing due to its minimal wiring and shape flexibility, but its nonlinear inverse problem often leads to severe artifacts and inaccurate contact reconstruction. This work presents PhyDNN, a physics-driven deep reconstruction framework that embeds the EIT forward model directly into the learning objective. By jointly minimizing the discrepancy between predicted and ground-truth conductivity maps and enforcing consistency with the forward PDE, PhyDNN reduces the black-box nature of deep networks and improves both physical plausibility and generalization. To enable efficient backpropagation, we design a differentiable forward-operator network that accurately approximates the nonlinear EIT response, allowing fast physics-guided training. Extensive simulations and real tactile experiments on a 16-electrode soft sensor show that PhyDNN consistently outperforms NOSER, TV, and standard DNNs in reconstructing contact shape, location, and pressure distribution. PhyDNN yields fewer artifacts, sharper boundaries, and higher metric scores, demonstrating its effectiveness for high-quality tomographic tactile sensing. 

---
# NAS-LoRA: Empowering Parameter-Efficient Fine-Tuning for Visual Foundation Models with Searchable Adaptation 

**Authors**: Renqi Chen, Haoyang Su, Shixiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2512.03499)  

**Abstract**: The Segment Anything Model (SAM) has emerged as a powerful visual foundation model for image segmentation. However, adapting SAM to specific downstream tasks, such as medical and agricultural imaging, remains a significant challenge. To address this, Low-Rank Adaptation (LoRA) and its variants have been widely employed to enhancing SAM's adaptation performance on diverse domains. Despite advancements, a critical question arises: can we integrate inductive bias into the model? This is particularly relevant since the Transformer encoder in SAM inherently lacks spatial priors within image patches, potentially hindering the acquisition of high-level semantic information. In this paper, we propose NAS-LoRA, a new Parameter-Efficient Fine-Tuning (PEFT) method designed to bridge the semantic gap between pre-trained SAM and specialized domains. Specifically, NAS-LoRA incorporates a lightweight Neural Architecture Search (NAS) block between the encoder and decoder components of LoRA to dynamically optimize the prior knowledge integrated into weight updates. Furthermore, we propose a stage-wise optimization strategy to help the ViT encoder balance weight updates and architectural adjustments, facilitating the gradual learning of high-level semantic information. Various Experiments demonstrate our NAS-LoRA improves existing PEFT methods, while reducing training cost by 24.14% without increasing inference cost, highlighting the potential of NAS in enhancing PEFT for visual foundation models. 

---
# Cell-cell communication inference and analysis: biological mechanisms, computational approaches, and future opportunities 

**Authors**: Xiangzheng Cheng, Haili Huang, Ye Su, Qing Nie, Xiufen Zou, Suoqin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2512.03497)  

**Abstract**: In multicellular organisms, cells coordinate their activities through cell-cell communication (CCC), which are crucial for development, tissue homeostasis, and disease progression. Recent advances in single-cell and spatial omics technologies provide unprecedented opportunities to systematically infer and analyze CCC from these omics data, either by integrating prior knowledge of ligand-receptor interactions (LRIs) or through de novo approaches. A variety of computational methods have been developed, focusing on methodological innovations, accurate modeling of complex signaling mechanisms, and investigation of broader biological questions. These advances have greatly enhanced our ability to analyze CCC and generate biological hypotheses. Here, we introduce the biological mechanisms and modeling strategies of CCC, and provide a focused overview of more than 140 computational methods for inferring CCC from single-cell and spatial transcriptomic data, emphasizing the diversity in methodological frameworks and biological questions. Finally, we discuss the current challenges and future opportunities in this rapidly evolving field. 

---
# ATHENA: Agentic Team for Hierarchical Evolutionary Numerical Algorithms 

**Authors**: Juan Diego Toscano, Daniel T. Chen, George Em Karniadakis  

**Link**: [PDF](https://arxiv.org/pdf/2512.03476)  

**Abstract**: Bridging the gap between theoretical conceptualization and computational implementation is a major bottleneck in Scientific Computing (SciC) and Scientific Machine Learning (SciML). We introduce ATHENA (Agentic Team for Hierarchical Evolutionary Numerical Algorithms), an agentic framework designed as an Autonomous Lab to manage the end-to-end computational research lifecycle. Its core is the HENA loop, a knowledge-driven diagnostic process framed as a Contextual Bandit problem. Acting as an online learner, the system analyzes prior trials to select structural `actions' ($A_n$) from combinatorial spaces guided by expert blueprints (e.g., Universal Approximation, Physics-Informed constraints). These actions are translated into executable code ($S_n$) to generate scientific rewards ($R_n$). ATHENA transcends standard automation: in SciC, it autonomously identifies mathematical symmetries for exact analytical solutions or derives stable numerical solvers where foundation models fail. In SciML, it performs deep diagnosis to tackle ill-posed formulations and combines hybrid symbolic-numeric workflows (e.g., coupling PINNs with FEM) to resolve multiphysics problems. The framework achieves super-human performance, reaching validation errors of $10^{-14}$. Furthermore, collaborative ``human-in-the-loop" intervention allows the system to bridge stability gaps, improving results by an order of magnitude. This paradigm shift focuses from implementation mechanics to methodological innovation, accelerating scientific discovery. 

---
# AsymPuzl: An Asymmetric Puzzle for multi-agent cooperation 

**Authors**: Xavier Cadet, Edward Koh, Peter Chin  

**Link**: [PDF](https://arxiv.org/pdf/2512.03466)  

**Abstract**: Large Language Model (LLM) agents are increasingly studied in multi-turn, multi-agent scenarios, yet most existing setups emphasize open-ended role-play rather than controlled evaluation. We introduce AsymPuzl, a minimal but expressive two-agent puzzle environment designed to isolate communication under information asymmetry. Each agent observes complementary but incomplete views of a symbolic puzzle and must exchange messages to solve it cooperatively. Using a diverse set of current-generation and open-source LLMs, we show that (i) strong models such as GPT-5 and Claude-4.0 reliably converge across puzzle sizes on the solution by sharing complete information in two turns, (ii) weaker models often ignore partner messages or over-correct their hypotheses, and (iii) feedback design is non-trivial: simple self-feedback improves success rates, while detailed joint feedback can hurt performance. These findings show that even in simple cooperative tasks, LLM communication strategies diverge and depend on the granularity of feedback signals. AsymPuzl thus provides a testbed for probing the limits of multi-turn cooperation and opens avenues for studying coordination mechanisms. 

---
# Text-Printed Image: Bridging the Image-Text Modality Gap for Text-centric Training of Large Vision-Language Models 

**Authors**: Shojiro Yamabe, Futa Waseda, Daiki Shiono, Tsubasa Takahashi  

**Link**: [PDF](https://arxiv.org/pdf/2512.03463)  

**Abstract**: Recent large vision-language models (LVLMs) have been applied to diverse VQA tasks. However, achieving practical performance typically requires task-specific fine-tuning with large numbers of image-text pairs, which are costly to collect. In this work, we study text-centric training, a setting where only textual descriptions are available and no real images are provided, as a paradigm for low-cost data scaling. Unlike images, whose collection is often restricted by privacy constraints and scarcity in niche domains, text is widely available. Moreover, text is easily editable, enabling automatic diversification and expansion with LLMs at minimal human effort. While this offers clear advantages over image collection in terms of scalability and cost, training on raw text without images still yields limited gains on VQA tasks because of the image-text modality gap. To address this issue, we propose a Text-Printed Image (TPI), which generates synthetic images by directly rendering the given textual description on a plain white canvas. This simple rendering projects text into the image modality and can be integrated into arbitrary existing LVLM training pipelines at low cost. Moreover, TPI preserves the semantics of the text, whereas text-to-image models often fail to do. Across four models and seven benchmarks, our systematic experiments show that TPI enables more effective text-centric training than synthetic images generated by a diffusion model. We further explore TPI as a low-cost data-augmentation strategy and demonstrate its practical utility. Overall, our findings highlight the significant potential of text-centric training and, more broadly, chart a path toward fully automated data generation for LVLMs. 

---
# Learning From Limited Data and Feedback for Cell Culture Process Monitoring: A Comparative Study 

**Authors**: Johnny Peng, Thanh Tung Khuat, Ellen Otte, Katarzyna Musial, Bogdan Gabrys  

**Link**: [PDF](https://arxiv.org/pdf/2512.03460)  

**Abstract**: In cell culture bioprocessing, real-time batch process monitoring (BPM) refers to the continuous tracking and analysis of key process variables such as viable cell density, nutrient levels, metabolite concentrations, and product titer throughout the duration of a batch run. This enables early detection of deviations and supports timely control actions to ensure optimal cell growth and product quality. BPM plays a critical role in ensuring the quality and regulatory compliance of biopharmaceutical manufacturing processes. However, the development of accurate soft sensors for BPM is hindered by key challenges, including limited historical data, infrequent feedback, heterogeneous process conditions, and high-dimensional sensory inputs. This study presents a comprehensive benchmarking analysis of machine learning (ML) methods designed to address these challenges, with a focus on learning from historical data with limited volume and relevance in the context of bioprocess monitoring. We evaluate multiple ML approaches including feature dimensionality reduction, online learning, and just-in-time learning across three datasets, one in silico dataset and two real-world experimental datasets. Our findings highlight the importance of training strategies in handling limited data and feedback, with batch learning proving effective in homogeneous settings, while just-in-time learning and online learning demonstrate superior adaptability in cold-start scenarios. Additionally, we identify key meta-features, such as feed media composition and process control strategies, that significantly impact model transferability. The results also suggest that integrating Raman-based predictions with lagged offline measurements enhances monitoring accuracy, offering a promising direction for future bioprocess soft sensor development. 

---
# Think Before You Drive: World Model-Inspired Multimodal Grounding for Autonomous Vehicles 

**Authors**: Haicheng Liao, Huanming Shen, Bonan Wang, Yongkang Li, Yihong Tang, Chengyue Wang, Dingyi Zhuang, Kehua Chen, Hai Yang, Chengzhong Xu, Zhenning Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.03454)  

**Abstract**: Interpreting natural-language commands to localize target objects is critical for autonomous driving (AD). Existing visual grounding (VG) methods for autonomous vehicles (AVs) typically struggle with ambiguous, context-dependent instructions, as they lack reasoning over 3D spatial relations and anticipated scene evolution. Grounded in the principles of world models, we propose ThinkDeeper, a framework that reasons about future spatial states before making grounding decisions. At its core is a Spatial-Aware World Model (SA-WM) that learns to reason ahead by distilling the current scene into a command-aware latent state and rolling out a sequence of future latent states, providing forward-looking cues for disambiguation. Complementing this, a hypergraph-guided decoder then hierarchically fuses these states with the multimodal input, capturing higher-order spatial dependencies for robust localization. In addition, we present DrivePilot, a multi-source VG dataset in AD, featuring semantic annotations generated by a Retrieval-Augmented Generation (RAG) and Chain-of-Thought (CoT)-prompted LLM pipeline. Extensive evaluations on six benchmarks, ThinkDeeper ranks #1 on the Talk2Car leaderboard and surpasses state-of-the-art baselines on DrivePilot, MoCAD, and RefCOCO/+/g benchmarks. Notably, it shows strong robustness and efficiency in challenging scenes (long-text, multi-agent, ambiguity) and retains superior performance even when trained on 50% of the data. 

---
# GalaxyDiT: Efficient Video Generation with Guidance Alignment and Adaptive Proxy in Diffusion Transformers 

**Authors**: Zhiye Song, Steve Dai, Ben Keller, Brucek Khailany  

**Link**: [PDF](https://arxiv.org/pdf/2512.03451)  

**Abstract**: Diffusion models have revolutionized video generation, becoming essential tools in creative content generation and physical simulation. Transformer-based architectures (DiTs) and classifier-free guidance (CFG) are two cornerstones of this success, enabling strong prompt adherence and realistic video quality. Despite their versatility and superior performance, these models require intensive computation. Each video generation requires dozens of iterative steps, and CFG doubles the required compute. This inefficiency hinders broader adoption in downstream applications.
We introduce GalaxyDiT, a training-free method to accelerate video generation with guidance alignment and systematic proxy selection for reuse metrics. Through rank-order correlation analysis, our technique identifies the optimal proxy for each video model, across model families and parameter scales, thereby ensuring optimal computational reuse. We achieve $1.87\times$ and $2.37\times$ speedup on Wan2.1-1.3B and Wan2.1-14B with only 0.97% and 0.72% drops on the VBench-2.0 benchmark. At high speedup rates, our approach maintains superior fidelity to the base model, exceeding prior state-of-the-art approaches by 5 to 10 dB in peak signal-to-noise ratio (PSNR). 

---
# Multi-Aspect Knowledge-Enhanced Medical Vision-Language Pretraining with Multi-Agent Data Generation 

**Authors**: Xieji Li, Siyuan Yan, Yingsheng Liu, H. Peter Soyer, Monika Janda, Victoria Mar, Zongyuan Ge  

**Link**: [PDF](https://arxiv.org/pdf/2512.03445)  

**Abstract**: Vision-language pretraining (VLP) has emerged as a powerful paradigm in medical image analysis, enabling representation learning from large-scale image-text pairs without relying on expensive manual annotations. However, existing methods often struggle with the noise inherent in web-collected data and the complexity of unstructured long medical texts. To address these challenges, we propose a novel VLP framework integrating a Multi-Agent data GENeration (MAGEN) system and Ontology-based Multi-Aspect Knowledge-Enhanced (O-MAKE) pretraining. First, MAGEN enhances data quality by synthesizing knowledge-enriched descriptions via a foundation model-assisted captioning and retrieval-based verification pipeline. Second, O-MAKE addresses the difficulty of learning from long, unstructured texts by decomposing them into distinct knowledge aspects. This facilitates fine-grained alignment at both global and patch levels, while explicitly modeling medical concept relationships through ontology-guided mechanisms. We validate our framework in the field of dermatology, where comprehensive experiments demonstrate the effectiveness of each component. Our approach achieves state-of-the-art zero-shot performance on disease classification and cross-modal retrieval tasks across eight datasets. Our code and the augmented dataset Derm1M-AgentAug, comprising over 400k skin-image-text pairs, will be released at this https URL. 

---
# World Models for Autonomous Navigation of Terrestrial Robots from LIDAR Observations 

**Authors**: Raul Steinmetz, Fabio Demo Rosa, Victor Augusto Kich, Jair Augusto Bottega, Ricardo Bedin Grando, Daniel Fernando Tello Gamarra  

**Link**: [PDF](https://arxiv.org/pdf/2512.03429)  

**Abstract**: Autonomous navigation of terrestrial robots using Reinforcement Learning (RL) from LIDAR observations remains challenging due to the high dimensionality of sensor data and the sample inefficiency of model-free approaches. Conventional policy networks struggle to process full-resolution LIDAR inputs, forcing prior works to rely on simplified observations that reduce spatial awareness and navigation robustness. This paper presents a novel model-based RL framework built on top of the DreamerV3 algorithm, integrating a Multi-Layer Perceptron Variational Autoencoder (MLP-VAE) within a world model to encode high-dimensional LIDAR readings into compact latent representations. These latent features, combined with a learned dynamics predictor, enable efficient imagination-based policy optimization. Experiments on simulated TurtleBot3 navigation tasks demonstrate that the proposed architecture achieves faster convergence and higher success rate compared to model-free baselines such as SAC, DDPG, and TD3. It is worth emphasizing that the DreamerV3-based agent attains a 100% success rate across all evaluated environments when using the full dataset of the Turtlebot3 LIDAR (360 readings), while model-free methods plateaued below 85%. These findings demonstrate that integrating predictive world models with learned latent representations enables more efficient and robust navigation from high-dimensional sensory data. 

---
# BookRAG: A Hierarchical Structure-aware Index-based Approach for Retrieval-Augmented Generation on Complex Documents 

**Authors**: Shu Wang, Yingli Zhou, Yixiang Fang  

**Link**: [PDF](https://arxiv.org/pdf/2512.03413)  

**Abstract**: As an effective method to boost the performance of Large Language Models (LLMs) on the question answering (QA) task, Retrieval-Augmented Generation (RAG), which queries highly relevant information from external complex documents, has attracted tremendous attention from both industry and academia. Existing RAG approaches often focus on general documents, and they overlook the fact that many real-world documents (such as books, booklets, handbooks, etc.) have a hierarchical structure, which organizes their content from different granularity levels, leading to poor performance for the QA task. To address these limitations, we introduce BookRAG, a novel RAG approach targeted for documents with a hierarchical structure, which exploits logical hierarchies and traces entity relations to query the highly relevant information. Specifically, we build a novel index structure, called BookIndex, by extracting a hierarchical tree from the document, which serves as the role of its table of contents, using a graph to capture the intricate relationships between entities, and mapping entities to tree nodes. Leveraging the BookIndex, we then propose an agent-based query method inspired by the Information Foraging Theory, which dynamically classifies queries and employs a tailored retrieval workflow. Extensive experiments on three widely adopted benchmarks demonstrate that BookRAG achieves state-of-the-art performance, significantly outperforming baselines in both retrieval recall and QA accuracy while maintaining competitive efficiency. 

---
# Better World Models Can Lead to Better Post-Training Performance 

**Authors**: Prakhar Gupta, Henry Conklin, Sarah-Jane Leslie, Andrew Lee  

**Link**: [PDF](https://arxiv.org/pdf/2512.03400)  

**Abstract**: In this work we study how explicit world-modeling objectives affect the internal representations and downstream capability of Transformers across different training stages. We use a controlled 2x2x2 Rubik's Cube and ask: (1) how does explicitly pretraining a world model affect the model's latent representations, and (2) how does world-model quality affect the model's performance after reinforcement learning post-training? We compare standard next-token prediction to two explicit world-modeling strategies -- (i) state-prediction pretraining and (ii) a joint state-prediction + next-token objective -- and assess task performance after Group Relative Policy Optimization (GRPO) is applied as post-training. We evaluate the representation quality with linear probes and causal interventions. We find that explicit world-modeling yields more linearly decodable and causally steerable state representations. More importantly, we find that improved state representations lead to higher gains for GRPO, especially on harder cube states. Our results indicate that sharpening state representations can improve the effectiveness of post-training for sequence-planning tasks. 

---
# VS-Graph: Scalable and Efficient Graph Classification Using Hyperdimensional Computing 

**Authors**: Hamed Poursiami, Shay Snyder, Guojing Cong, Thomas Potok, Maryam Parsa  

**Link**: [PDF](https://arxiv.org/pdf/2512.03394)  

**Abstract**: Graph classification is a fundamental task in domains ranging from molecular property prediction to materials design. While graph neural networks (GNNs) achieve strong performance by learning expressive representations via message passing, they incur high computational costs, limiting their scalability and deployment on resource-constrained devices. Hyperdimensional Computing (HDC), also known as Vector Symbolic Architectures (VSA), offers a lightweight, brain-inspired alternative, yet existing HDC-based graph methods typically struggle to match the predictive performance of GNNs. In this work, we propose VS-Graph, a vector-symbolic graph learning framework that narrows the gap between the efficiency of HDC and the expressive power of message passing. VS-Graph introduces a Spike Diffusion mechanism for topology-driven node identification and an Associative Message Passing scheme for multi-hop neighborhood aggregation entirely within the high-dimensional vector space. Without gradient-based optimization or backpropagation, our method achieves competitive accuracy with modern GNNs, outperforming the prior HDC baseline by 4-5% on standard benchmarks such as MUTAG and DD. It also matches or exceeds the performance of the GNN baselines on several datasets while accelerating the training by a factor of up to 450x. Furthermore, VS-Graph maintains high accuracy even with the hypervector dimensionality reduced to D=128, demonstrating robustness under aggressive dimension compression and paving the way for ultra-efficient execution on edge and neuromorphic hardware. 

---
# UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs 

**Authors**: Hung-Yueh Chiang, Chi-Chih Chang, Yu-Chen Lu, Chien-Yu Lin, Kai-Chiang Wu, Mohamed S. Abdelfattah, Diana Marculescu  

**Link**: [PDF](https://arxiv.org/pdf/2512.03383)  

**Abstract**: Deploying large language model (LLM) models on mobile platforms faces significant challenges due to the limited memory and shared computational resources of the device. Resource availability may be an issue as it is directly impacted by the current device workload, adding to the uncertainty of model deployment. We introduce UniQL, a unified post-training quantization and low-rank compression framework with on-device configurable pruning rates for edge LLMs. UniQL is a general framework that integrates quantization and low-rank compression for Transformers, State Space Models (SSMs), and hybrid models to support diverse edge applications. In our proposed joint framework, we introduce an efficient structured weight-sorting method that speeds up computation by 20x, quantization-aware singular value decomposition (SVD) to minimize quantization errors, state-aware weight sorting for SSMs, and a fused rotary positional embedding (RoPE) kernel for pruned models. Our framework performs weight-sorting, fine-tuning, and quantization in the cloud in a single-pass workflow, while enabling on-device configurable pruning rates up to 35%. Our experiments show that quantized and pruned models achieve a memory reduction of 4x-5.7x and a token-throughput improvement of 2.7x-3.4x, maintaining accuracy within 5% of the original models at 15% pruning across Transformers (Llama3 and Qwen2.5), SSMs (Mamba2), and hybrid models (Nemotron-H and Bamba-v2). The code and quantized models are available at: this https URL. 

---
# FireSentry: A Multi-Modal Spatio-temporal Benchmark Dataset for Fine-Grained Wildfire Spread Forecasting 

**Authors**: Nan Zhou, Huandong Wang, Jiahao Li, Han Li, Yali Song, Qiuhua Wang, Yong Li, Xinlei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.03369)  

**Abstract**: Fine-grained wildfire spread prediction is crucial for enhancing emergency response efficacy and decision-making precision. However, existing research predominantly focuses on coarse spatiotemporal scales and relies on low-resolution satellite data, capturing only macroscopic fire states while fundamentally constraining high-precision localized fire dynamics modeling capabilities. To bridge this gap, we present FireSentry, a provincial-scale multi-modal wildfire dataset characterized by sub-meter spatial and sub-second temporal resolution. Collected using synchronized UAV platforms, FireSentry provides visible and infrared video streams, in-situ environmental measurements, and manually validated fire masks. Building on FireSentry, we establish a comprehensive benchmark encompassing physics-based, data-driven, and generative models, revealing the limitations of existing mask-only approaches. Our analysis proposes FiReDiff, a novel dual-modality paradigm that first predicts future video sequences in the infrared modality, and then precisely segments fire masks in the mask modality based on the generated dynamics. FiReDiff achieves state-of-the-art performance, with video quality gains of 39.2% in PSNR, 36.1% in SSIM, 50.0% in LPIPS, 29.4% in FVD, and mask accuracy gains of 3.3% in AUPRC, 59.1% in F1 score, 42.9% in IoU, and 62.5% in MSE when applied to generative models. The FireSentry benchmark dataset and FiReDiff paradigm collectively advance fine-grained wildfire forecasting and dynamic disaster simulation. The processed benchmark dataset is publicly available at: this https URL. 

---
# HalluGen: Synthesizing Realistic and Controllable Hallucinations for Evaluating Image Restoration 

**Authors**: Seunghoi Kim, Henry F. J. Tregidgo, Chen Jin, Matteo Figini, Daniel C. Alexander  

**Link**: [PDF](https://arxiv.org/pdf/2512.03345)  

**Abstract**: Generative models are prone to hallucinations: plausible but incorrect structures absent in the ground truth. This issue is problematic in image restoration for safety-critical domains such as medical imaging, industrial inspection, and remote sensing, where such errors undermine reliability and trust. For example, in low-field MRI, widely used in resource-limited settings, restoration models are essential for enhancing low-quality scans, yet hallucinations can lead to serious diagnostic errors. Progress has been hindered by a circular dependency: evaluating hallucinations requires labeled data, yet such labels are costly and subjective. We introduce HalluGen, a diffusion-based framework that synthesizes realistic hallucinations with controllable type, location, and severity, producing perceptually realistic but semantically incorrect outputs (segmentation IoU drops from 0.86 to 0.36). Using HalluGen, we construct the first large-scale hallucination dataset comprising 4,350 annotated images derived from 1,450 brain MR images for low-field enhancement, enabling systematic evaluation of hallucination detection and mitigation. We demonstrate its utility in two applications: (1) benchmarking image quality metrics and developing Semantic Hallucination Assessment via Feature Evaluation (SHAFE), a feature-based metric with soft-attention pooling that improves hallucination sensitivity over traditional metrics; and (2) training reference-free hallucination detectors that generalize to real restoration failures. Together, HalluGen and its open dataset establish the first scalable foundation for evaluating hallucinations in safety-critical image restoration. 

---
# Idea-Gated Transformers: Enforcing Semantic Coherence via Differentiable Vocabulary Pruning 

**Authors**: Darshan Fofadiya  

**Link**: [PDF](https://arxiv.org/pdf/2512.03343)  

**Abstract**: Autoregressive Language Models (LLMs) trained on Next-Token Prediction (NTP) often suffer from ``Topic Drift'' where the generation wanders away from the initial prompt due to a reliance on local associations rather than global planning \citep{holtzman2019curious}. While scaling model size mitigates this \citep{brown2020language}, the fundamental myopia of the NTP objective remains. In this work, we introduce the Idea-Gated Transformer, a novel architecture that separates semantic planning from syntactic generation. We introduce an auxiliary ``Idea Head'' trained to predict the bag-of-words distribution for a future context window, creating a latent ``Concept Vector'' that actively gates the main vocabulary during generation. We propose a differentiable gating mechanism that suppresses semantically irrelevant tokens, effectively pruning the search space in real-time. Experiments on WikiText-103 demonstrate that while the Idea-Gated model achieves comparable validation perplexity to a standard GPT-2 baseline, it exhibits significantly superior Domain Retention. Qualitative and quantitative analysis reveals that the gating mechanism successfully locks generation into specific semantic clusters (e.g., Finance, Science) and resists associative drift, offering a parameter-efficient path toward more controllable language modeling. 

---
# ProtoEFNet: Dynamic Prototype Learning for Inherently Interpretable Ejection Fraction Estimation in Echocardiography 

**Authors**: Yeganeh Ghamary, Victoria Wu, Hooman Vaseli, Christina Luong, Teresa Tsang, Siavash Bigdeli, Purang Abolmaesumi  

**Link**: [PDF](https://arxiv.org/pdf/2512.03339)  

**Abstract**: Ejection fraction (EF) is a crucial metric for assessing cardiac function and diagnosing conditions such as heart failure. Traditionally, EF estimation requires manual tracing and domain expertise, making the process time-consuming and subject to interobserver variability. Most current deep learning methods for EF prediction are black-box models with limited transparency, which reduces clinical trust. Some post-hoc explainability methods have been proposed to interpret the decision-making process after the prediction is made. However, these explanations do not guide the model's internal reasoning and therefore offer limited reliability in clinical applications. To address this, we introduce ProtoEFNet, a novel video-based prototype learning model for continuous EF regression. The model learns dynamic spatiotemporal prototypes that capture clinically meaningful cardiac motion patterns. Additionally, the proposed Prototype Angular Separation (PAS) loss enforces discriminative representations across the continuous EF spectrum. Our experiments on the EchonetDynamic dataset show that ProtoEFNet can achieve accuracy on par with its non-interpretable counterpart while providing clinically relevant insight. The ablation study shows that the proposed loss boosts performance with a 2% increase in F1 score from 77.67$\pm$2.68 to 79.64$\pm$2.10. Our source code is available at: this https URL 

---
# Single-Round Scalable Analytic Federated Learning 

**Authors**: Alan T. L. Bacellar, Mustafa Munir, Felipe M. G. França, Priscila M. V. Lima, Radu Marculescu, Lizy K. John  

**Link**: [PDF](https://arxiv.org/pdf/2512.03336)  

**Abstract**: Federated Learning (FL) is plagued by two key challenges: high communication overhead and performance collapse on heterogeneous (non-IID) data. Analytic FL (AFL) provides a single-round, data distribution invariant solution, but is limited to linear models. Subsequent non-linear approaches, like DeepAFL, regain accuracy but sacrifice the single-round benefit. In this work, we break this trade-off. We propose SAFLe, a framework that achieves scalable non-linear expressivity by introducing a structured head of bucketed features and sparse, grouped embeddings. We prove this non-linear architecture is mathematically equivalent to a high-dimensional linear regression. This key equivalence allows SAFLe to be solved with AFL's single-shot, invariant aggregation law. Empirically, SAFLe establishes a new state-of-the-art for analytic FL, significantly outperforming both linear AFL and multi-round DeepAFL in accuracy across all benchmarks, demonstrating a highly efficient and scalable solution for federated vision. 

---
# Cache What Lasts: Token Retention for Memory-Bounded KV Cache in LLMs 

**Authors**: Ngoc Bui, Shubham Sharma, Simran Lamba, Saumitra Mishra, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2512.03324)  

**Abstract**: Memory and computation remain core bottlenecks in long-horizon LLM inference due to the quadratic cost of self-attention and the ever-growing key-value (KV) cache. Existing strategies for memory-bounded inference, such as quantization, offloading, or heuristic KV eviction, either incur high orchestration costs or rely on unreliable attention-based proxies of importance. We propose TRIM-KV, a novel approach that learns each token's intrinsic importance at creation time via a lightweight retention gate. Each gate predicts a scalar retention score that decays over time, reflecting the long-term utility of the token for a specific layer and head. Tokens with low scores are evicted when the memory budget is exceeded, ensuring that the cache always contains the most critical tokens. TRIM-KV is trained efficiently through distillation from a frozen LLM combined with a capacity loss, requiring only gate fine-tuning and adding negligible inference overhead. Across mathematical reasoning (GSM8K, MATH-500, AIME24), procedural generation (LongProc), conversational long-memory benchmarks (LongMemEval), and long-context understanding (LongBench and SCBench), TRIM-KV consistently outperforms strong eviction and learnable retrieval baselines, especially in low-memory regimes. Remarkably, it even surpasses full-cache models in some settings, showing that selective retention can serve as a form of regularization, suppressing noise from uninformative tokens. Qualitative analyses further reveal that learned retention scores align with human intuition, naturally recovering heuristics such as sink tokens, sliding windows, and gist compression without explicit design. Beyond efficiency, retention scores provide insights into layer- and head-specific roles, suggesting a new path toward LLM interpretability. 

---
# NavMapFusion: Diffusion-based Fusion of Navigation Maps for Online Vectorized HD Map Construction 

**Authors**: Thomas Monninger, Zihan Zhang, Steffen Staab, Sihao Ding  

**Link**: [PDF](https://arxiv.org/pdf/2512.03317)  

**Abstract**: Accurate environmental representations are essential for autonomous driving, providing the foundation for safe and efficient navigation. Traditionally, high-definition (HD) maps are providing this representation of the static road infrastructure to the autonomous system a priori. However, because the real world is constantly changing, such maps must be constructed online from on-board sensor data. Navigation-grade standard-definition (SD) maps are widely available, but their resolution is insufficient for direct deployment. Instead, they can be used as coarse prior to guide the online map construction process. We propose NavMapFusion, a diffusion-based framework that performs iterative denoising conditioned on high-fidelity sensor data and on low-fidelity navigation maps. This paper strives to answer: (1) How can coarse, potentially outdated navigation maps guide online map construction? (2) What advantages do diffusion models offer for map fusion? We demonstrate that diffusion-based map construction provides a robust framework for map fusion. Our key insight is that discrepancies between the prior map and online perception naturally correspond to noise within the diffusion process; consistent regions reinforce the map construction, whereas outdated segments are suppressed. On the nuScenes benchmark, NavMapFusion conditioned on coarse road lines from OpenStreetMap data reaches a 21.4% relative improvement on 100 m, and even stronger improvements on larger perception ranges, while maintaining real-time capabilities. By fusing low-fidelity priors with high-fidelity sensor data, the proposed method generates accurate and up-to-date environment representations, guiding towards safer and more reliable autonomous driving. The code is available at this https URL 

---
# Retrofitting Earth System Models with Cadence-Limited Neural Operator Updates 

**Authors**: Aniruddha Bora, Shixuan Zhang, Khemraj Shukla, Bryce Harrop, George Em. Karniadakis, L. Ruby Leung  

**Link**: [PDF](https://arxiv.org/pdf/2512.03309)  

**Abstract**: Coarse resolution, imperfect parameterizations, and uncertain initial states and forcings limit Earth-system model (ESM) predictions. Traditional bias correction via data assimilation improves constrained simulations but offers limited benefit once models run freely. We introduce an operator-learning framework that maps instantaneous model states to bias-correction tendencies and applies them online during integration. Building on a U-Net backbone, we develop two operator architectures Inception U-Net (IUNet) and a multi-scale network (M\&M) that combine diverse upsampling and receptive fields to capture multiscale nonlinear features under Energy Exascale Earth System Model (E3SM) runtime constraints. Trained on two years E3SM simulations nudged toward ERA5 reanalysis, the operators generalize across height levels and seasons. Both architectures outperform standard U-Net baselines in offline tests, indicating that functional richness rather than parameter count drives performance. In online hybrid E3SM runs, M\&M delivers the most consistent bias reductions across variables and vertical levels. The ML-augmented configurations remain stable and computationally feasible in multi-year simulations, providing a practical pathway for scalable hybrid modeling. Our framework emphasizes long-term stability, portability, and cadence-limited updates, demonstrating the utility of expressive ML operators for learning structured, cross-scale relationships and retrofitting legacy ESMs. 

---
# Robust Tabular Foundation Models 

**Authors**: Matthew Peroni, Franck Le, Vadim Sheinin  

**Link**: [PDF](https://arxiv.org/pdf/2512.03307)  

**Abstract**: The development of tabular foundation models (TFMs) has accelerated in recent years, showing strong potential to outperform traditional ML methods for structured data. A key finding is that TFMs can be pretrained entirely on synthetic datasets, opening opportunities to design data generators that encourage desirable model properties. Prior work has mainly focused on crafting high-quality priors over generators to improve overall pretraining performance. Our insight is that parameterizing the generator distribution enables an adversarial robustness perspective: during training, we can adapt the generator to emphasize datasets that are particularly challenging for the model. We formalize this by introducing an optimality gap measure, given by the difference between TFM performance and the best achievable performance as estimated by strong baselines such as XGBoost, CatBoost, and Random Forests. Building on this idea, we propose Robust Tabular Foundation Models (RTFM), a model-agnostic adversarial training framework. Applied to the TabPFN V2 classifier, RTFM improves benchmark performance, with up to a 6% increase in mean normalized AUC over the original TabPFN and other baseline algorithms, while requiring less than 100k additional synthetic datasets. These results highlight a promising new direction for targeted adversarial training and fine-tuning of TFMs using synthetic data alone. 

---
# HydroDCM: Hydrological Domain-Conditioned Modulation for Cross-Reservoir Inflow Prediction 

**Authors**: Pengfei Hu, Fan Ming, Xiaoxue Han, Chang Lu, Yue Ning, Dan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2512.03300)  

**Abstract**: Deep learning models have shown promise in reservoir inflow prediction, yet their performance often deteriorates when applied to different reservoirs due to distributional differences, referred to as the domain shift problem. Domain generalization (DG) solutions aim to address this issue by extracting domain-invariant representations that mitigate errors in unseen domains. However, in hydrological settings, each reservoir exhibits unique inflow patterns, while some metadata beyond observations like spatial information exerts indirect but significant influence. This mismatch limits the applicability of conventional DG techniques to many-domain hydrological systems. To overcome these challenges, we propose HydroDCM, a scalable DG framework for cross-reservoir inflow forecasting. Spatial metadata of reservoirs is used to construct pseudo-domain labels that guide adversarial learning of invariant temporal features. During inference, HydroDCM adapts these features through light-weight conditioning layers informed by the target reservoir's metadata, reconciling DG's invariance with location-specific adaptation. Experiment results on 30 real-world reservoirs in the Upper Colorado River Basin demonstrate that our method substantially outperforms state-of-the-art DG baselines under many-domain conditions and remains computationally efficient. 

---
# Adaptive Regime-Switching Forecasts with Distribution-Free Uncertainty: Deep Switching State-Space Models Meet Conformal Prediction 

**Authors**: Echo Diyun LU, Charles Findling, Marianne Clausel, Alessandro Leite, Wei Gong, Pierric Kersaudy  

**Link**: [PDF](https://arxiv.org/pdf/2512.03298)  

**Abstract**: Regime transitions routinely break stationarity in time series, making calibrated uncertainty as important as point accuracy. We study distribution-free uncertainty for regime-switching forecasting by coupling Deep Switching State Space Models with Adaptive Conformal Inference (ACI) and its aggregated variant (AgACI). We also introduce a unified conformal wrapper that sits atop strong sequence baselines including S4, MC-Dropout GRU, sparse Gaussian processes, and a change-point local model to produce online predictive bands with finite-sample marginal guarantees under nonstationarity and model misspecification. Across synthetic and real datasets, conformalized forecasters achieve near-nominal coverage with competitive accuracy and generally improved band efficiency. 

---
# BlendedNet++: A Large-Scale Blended Wing Body Aerodynamics Dataset and Benchmark 

**Authors**: Nicholas Sung, Steven Spreizer, Mohamed Elrefaie, Matthew C. Jones, Faez Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2512.03280)  

**Abstract**: Despite progress in machine learning-based aerodynamic surrogates, the scarcity of large, field-resolved datasets limits progress on accurate pointwise prediction and reproducible inverse design for aircraft. We introduce BlendedNet++, a large-scale aerodynamic dataset and benchmark focused on blended wing body (BWB) aircraft. The dataset contains over 12,000 unique geometries, each simulated at a single flight condition, yielding 12,490 aerodynamic results for steady RANS CFD. For every case, we provide (i) integrated force/moment coefficients CL, CD, CM and (ii) dense surface fields of pressure and skin friction coefficients Cp and (Cfx, Cfy, Cfz). Using this dataset, we standardize a forward-surrogate benchmark to predict pointwise fields across six model families: GraphSAGE, GraphUNet, PointNet, a coordinate Transformer (Transolver-style), a FiLMNet (coordinate MLP with feature-wise modulation), and a Graph Neural Operator Transformer (GNOT). Finally, we present an inverse design task of achieving a specified lift-to-drag ratio under fixed flight conditions, implemented via a conditional diffusion model. To assess performance, we benchmark this approach against gradient-based optimization on the same surrogate and a diffusion-optimization hybrid that first samples with the conditional diffusion model and then further optimizes the designs. BlendedNet++ provides a unified forward and inverse protocol with multi-model baselines, enabling fair, reproducible comparison across architectures and optimization paradigms. We expect BlendedNet++ to catalyze reproducible research in field-level aerodynamics and inverse design; resources (dataset, splits, baselines, and scripts) will be released upon acceptance. 

---
# Thucy: An LLM-based Multi-Agent System for Claim Verification across Relational Databases 

**Authors**: Michael Theologitis, Dan Suciu  

**Link**: [PDF](https://arxiv.org/pdf/2512.03278)  

**Abstract**: In today's age, it is becoming increasingly difficult to decipher truth from lies. Every day, politicians, media outlets, and public figures make conflicting claims$\unicode{x2014}$often about topics that can, in principle, be verified against structured data. For instance, statements about crime rates, economic growth or healthcare can all be verified against official public records and structured datasets. Building a system that can automatically do that would have sounded like science fiction just a few years ago. Yet, with the extraordinary progress in LLMs and agentic AI, this is now within reach. Still, there remains a striking gap between what is technically possible and what is being demonstrated by recent work. Most existing verification systems operate only on small, single-table databases$\unicode{x2014}$typically a few hundred rows$\unicode{x2014}$that conveniently fit within an LLM's context window.
In this paper we report our progress on Thucy, the first cross-database, cross-table multi-agent claim verification system that also provides concrete evidence for each verification verdict. Thucy remains completely agnostic to the underlying data sources before deployment and must therefore autonomously discover, inspect, and reason over all available relational databases to verify claims. Importantly, Thucy also reports the exact SQL queries that support its verdict (whether the claim is accurate or not) offering full transparency to expert users familiar with SQL. When evaluated on the TabFact dataset$\unicode{x2014}$the standard benchmark for fact verification over structured data$\unicode{x2014}$Thucy surpasses the previous state of the art by 5.6 percentage points in accuracy (94.3% vs. 88.7%). 

---
# PyroFocus: A Deep Learning Approach to Real-Time Wildfire Detection in Multispectral Remote Sensing Imagery 

**Authors**: Mark Moussa, Andre Williams, Seth Roffe, Douglas Morton  

**Link**: [PDF](https://arxiv.org/pdf/2512.03257)  

**Abstract**: Rapid and accurate wildfire detection is crucial for emergency response and environmental management. In airborne and spaceborne missions, real-time algorithms must distinguish between no fire, active fire, and post-fire conditions, and estimate fire intensity. Multispectral and hyperspectral thermal imagers provide rich spectral information, but high data dimensionality and limited onboard resources make real-time processing challenging. As wildfires increase in frequency and severity, the need for low-latency and computationally efficient onboard detection methods is critical.
We present a systematic evaluation of multiple deep learning architectures, including custom Convolutional Neural Networks (CNNs) and Transformer-based models, for multi-class fire classification. We also introduce PyroFocus, a two-stage pipeline that performs fire classification followed by fire radiative power (FRP) regression or segmentation to reduce inference time and computational cost for onboard deployment. Using data from NASA's MODIS/ASTER Airborne Simulator (MASTER), which is similar to a next-generation fire detection sensor, we compare accuracy, inference latency, and resource efficiency.
Experimental results show that the proposed two-stage pipeline achieves strong trade-offs between speed and accuracy, demonstrating significant potential for real-time edge deployment in future wildfire monitoring missions. 

---
# Learning Network Sheaves for AI-native Semantic Communication 

**Authors**: Enrico Grimaldi, Mario Edoardo Pandolfo, Gabriele D'Acunto, Sergio Barbarossa, Paolo Di Lorenzo  

**Link**: [PDF](https://arxiv.org/pdf/2512.03248)  

**Abstract**: Recent advances in AI call for a paradigm shift from bit-centric communication to goal- and semantics-oriented architectures, paving the way for AI-native 6G networks. In this context, we address a key open challenge: enabling heterogeneous AI agents to exchange compressed latent-space representations while mitigating semantic noise and preserving task-relevant meaning. We cast this challenge as learning both the communication topology and the alignment maps that govern information exchange among agents, yielding a learned network sheaf equipped with orthogonal maps. This learning process is further supported by a semantic denoising end compression module that constructs a shared global semantic space and derives sparse, structured representations of each agent's latent space. This corresponds to a nonconvex dictionary learning problem solved iteratively with closed-form updates. Experiments with mutiple AI agents pre-trained on real image data show that the semantic denoising and compression facilitates AI agents alignment and the extraction of semantic clusters, while preserving high accuracy in downstream task. The resulting communication network provides new insights about semantic heterogeneity across agents, highlighting the interpretability of our methodology. 

---
# SPARK: Stepwise Process-Aware Rewards for Reference-Free Reinforcement Learning 

**Authors**: Salman Rahman, Sruthi Gorantla, Arpit Gupta, Swastik Roy, Nanyun Peng, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.03244)  

**Abstract**: Process reward models (PRMs) that provide dense, step-level feedback have shown promise for reinforcement learning, yet their adoption remains limited by the need for expensive step-level annotations or ground truth references. We propose SPARK: a three-stage framework where in the first stage a generator model produces diverse solutions and a verifier model evaluates them using parallel scaling (self-consistency) and sequential scaling (meta-critique). In the second stage, we use these verification outputs as synthetic training data to fine-tune generative process reward models, which subsequently serve as reward signals during training. We show that aggregating multiple independent verifications at the step level produces training data for process reward models that surpass ground-truth outcome supervision, achieving 67.5 F1 on ProcessBench (a benchmark for identifying erroneous steps in mathematical reasoning) compared to 66.4 for reference-guided training and 61.9 for GPT-4o. In the final stage, we apply our generative PRM with chain-of-thought verification (PRM-CoT) as the reward model in RL experiments on mathematical reasoning, and introduce format constraints to prevent reward hacking. Using Qwen2.5-Math-7B, we achieve 47.4% average accuracy across six mathematical reasoning benchmarks, outperforming ground-truth-based RLVR (43.9%). Our work enables reference-free RL training that exceeds ground-truth methods, opening new possibilities for domains lacking verifiable answers or accessible ground truth. 

---
# How to DP-fy Your Data: A Practical Guide to Generating Synthetic Data With Differential Privacy 

**Authors**: Natalia Ponomareva, Zheng Xu, H. Brendan McMahan, Peter Kairouz, Lucas Rosenblatt, Vincent Cohen-Addad, Cristóbal Guzmán, Ryan McKenna, Galen Andrew, Alex Bie, Da Yu, Alex Kurakin, Morteza Zadimoghaddam, Sergei Vassilvitskii, Andreas Terzis  

**Link**: [PDF](https://arxiv.org/pdf/2512.03238)  

**Abstract**: High quality data is needed to unlock the full potential of AI for end users. However finding new sources of such data is getting harder: most publicly-available human generated data will soon have been used. Additionally, publicly available data often is not representative of users of a particular system -- for example, a research speech dataset of contractors interacting with an AI assistant will likely be more homogeneous, well articulated and self-censored than real world commands that end users will issue. Therefore unlocking high-quality data grounded in real user interactions is of vital interest. However, the direct use of user data comes with significant privacy risks. Differential Privacy (DP) is a well established framework for reasoning about and limiting information leakage, and is a gold standard for protecting user privacy. The focus of this work, \emph{Differentially Private Synthetic data}, refers to synthetic data that preserves the overall trends of source data,, while providing strong privacy guarantees to individuals that contributed to the source dataset. DP synthetic data can unlock the value of datasets that have previously been inaccessible due to privacy concerns and can replace the use of sensitive datasets that previously have only had rudimentary protections like ad-hoc rule-based anonymization.
In this paper we explore the full suite of techniques surrounding DP synthetic data, the types of privacy protections they offer and the state-of-the-art for various modalities (image, tabular, text and decentralized). We outline all the components needed in a system that generates DP synthetic data, from sensitive data handling and preparation, to tracking the use and empirical privacy testing. We hope that work will result in increased adoption of DP synthetic data, spur additional research and increase trust in DP synthetic data approaches. 

---
# InvertiTune: High-Quality Data Synthesis for Cost-Effective Single-Shot Text-to-Knowledge Graph Generation 

**Authors**: Faezeh Faez, Marzieh S. Tahaei, Yaochen Hu, Ali Pourranjbar, Mahdi Biparva, Mark Coates, Yingxue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.03197)  

**Abstract**: Large Language Models (LLMs) have revolutionized the ability to understand and generate text, enabling significant progress in automatic knowledge graph construction from text (Text2KG). Many Text2KG methods, however, rely on iterative LLM prompting, making them computationally expensive and prone to overlooking complex relations distributed throughout the text. To address these limitations, we propose InvertiTune, a framework that combines a controlled data generation pipeline with supervised fine-tuning (SFT). Within this framework, the data-generation pipeline systematically extracts subgraphs from large knowledge bases, applies noise filtering, and leverages LLMs to generate corresponding natural text descriptions, a task more aligned with LLM capabilities than direct KG generation from text. This pipeline enables generating datasets composed of longer texts paired with larger KGs that better reflect real-world scenarios compared to existing benchmarks, thus supporting effective SFT of lightweight models for single-shot KG construction. Experimental results on CE12k, a dataset generated using the introduced pipeline, show that InvertiTune outperforms larger non-fine-tuned LLMs as well as state-of-the-art Text2KG approaches, while also demonstrating stronger cross-dataset generalization on CrossEval-1200, a test set created from three established benchmark datasets and CE12k. These findings highlight the importance of realistic, high-quality training data for advancing efficient and high-performing Text2KG systems. 

---
# Ultra-Strong Gradient Diffusion MRI with Self-Supervised Learning for Prostate Cancer Characterization 

**Authors**: Tanishq Patil, Snigdha Sen, Malwina Molendowska, Kieran G. Foley, Fabrizio Fasano, Mara Cercignani, Marco Palombo, Paddy J. Slator, Eleftheria Panagiotaki  

**Link**: [PDF](https://arxiv.org/pdf/2512.03196)  

**Abstract**: Diffusion MRI (dMRI) enables non-invasive assessment of prostate microstructure but conventional metrics such as the Apparent Diffusion Coefficient in multiparametric MRI lack specificity to underlying histology. Integrating dMRI with the compartment-based biophysical VERDICT (Vascular, Extracellular, and Restricted Diffusion for Cytometry in Tumours) framework offers richer microstructural insights, though clinical gradient systems (40-80 mT/m) suffer from poor signal-to-noise ratio (SNR) at stronger diffusion weightings due to prolonged echo times. Ultra-strong gradients (up to 300 mT/m) can mitigate these limitations by improving SNR and contrast-to-noise ratios (CNR) but their adoption has until recently been limited to research environments due to challenges with peripheral nerve stimulation thresholds and gradient non-uniformity. This study investigates whether physics-informed self-supervised VERDICT (ssVERDICT) fitting applied to ultra-strong gradients enhances prostate cancer characterization relative to current clinical acquisitions. We developed enhanced ssVERDICT fitting approaches using dense multilayer perceptron (Dense MLP) and convolutional U-Net architectures, benchmarking them against non-linear least-squares (NLLS) fitting and Diffusion Kurtosis Imaging across clinical- to ultra-strong gradient systems. Dense ssVERDICT at ultra-strong gradient notably outperformed NLLS VERDICT, boosting median CNR by 47%, cutting inter-patient Coefficient of Variation by 52%, and reducing pooled f_ic variation by 50%. Overall, it delivered the highest CNR, the most stable parameter estimates, and the clearest tumour-normal contrast compared with conventional methods and clinical gradient systems. These findings highlight the potential of advanced gradient systems and deep learning-based modelling to improve non-invasive prostate cancer characterization and reduce unnecessary biopsies. 

---
# Plantain: Plan-Answer Interleaved Reasoning 

**Authors**: Anthony Liang, Jonathan Berant, Adam Fisch, Abhimanyu Goyal, Kalpesh Krishna, Jacob Eisenstein  

**Link**: [PDF](https://arxiv.org/pdf/2512.03176)  

**Abstract**: Reasoning models often spend a significant amount of time thinking before they generate a visible response. In the meantime, they do not give the user any hints as to whether their reasoning is on the right track, and do not give the user any recourse to stop and correct them if their reasoning is flawed. This creates a frustrating, but unfortunately common, experience: the user's time is wasted while the model reasons from a false premise that could have easily been corrected. In contrast, human speakers typically perform lightweight, incremental grounding acts to ensure that participants in the conversation are on the same page; here we ask if language models can learn to leverage a similar type of behavior? With this motivation, we propose interleaved reasoning (IR), in which the model alternates between thinking and surfacing intermediate responses, as an alternative to the standard "think-then-answer" approach. By providing useful information to the user earlier, IR reduces perceived latency, the time a user waits for an initial output, without compromising the quality of the final response. We further introduce a specialization of interleaved reasoning, Plantain (Plan-Thought-Answer Interleaving), where the first intermediate response is an explicit, step-by-step plan for executing the task. This plan-first strategy allows for user intervention and early feedback for subsequent reasoning steps. We demonstrate that Plantain yields an ~6% improvement in pass@1 across several challenging math reasoning and coding benchmarks, while reducing time-to-first-response by over 60% relative to think-then-answer baselines. 

---
# Culture Affordance Atlas: Reconciling Object Diversity Through Functional Mapping 

**Authors**: Joan Nwatu, Longju Bai, Oana Ignat, Rada Mihalcea  

**Link**: [PDF](https://arxiv.org/pdf/2512.03173)  

**Abstract**: Culture shapes the objects people use and for what purposes, yet mainstream Vision-Language (VL) datasets frequently exhibit cultural biases, disproportionately favoring higher-income, Western contexts. This imbalance reduces model generalizability and perpetuates performance disparities, especially impacting lower-income and non-Western communities. To address these disparities, we propose a novel function-centric framework that categorizes objects by the functions they fulfill, across diverse cultural and economic contexts. We implement this framework by creating the Culture Affordance Atlas, a re-annotated and culturally grounded restructuring of the Dollar Street dataset spanning 46 functions and 288 objects publicly available at this https URL. Through extensive empirical analyses using the CLIP model, we demonstrate that function-centric labels substantially reduce socioeconomic performance gaps between high- and low-income groups by a median of 6 pp (statistically significant), improving model effectiveness for lower-income contexts. Furthermore, our analyses reveals numerous culturally essential objects that are frequently overlooked in prominent VL datasets. Our contributions offer a scalable pathway toward building inclusive VL datasets and equitable AI systems. 

---
# Atomic Diffusion Models for Small Molecule Structure Elucidation from NMR Spectra 

**Authors**: Ziyu Xiong, Yichi Zhang, Foyez Alauddin, Chu Xin Cheng, Joon Soo An, Mohammad R. Seyedsayamdost, Ellen D. Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2512.03127)  

**Abstract**: Nuclear Magnetic Resonance (NMR) spectroscopy is a cornerstone technique for determining the structures of small molecules and is especially critical in the discovery of novel natural products and clinical therapeutics. Yet, interpreting NMR spectra remains a time-consuming, manual process requiring extensive domain expertise. We introduce ChefNMR (CHemical Elucidation From NMR), an end-to-end framework that directly predicts an unknown molecule's structure solely from its 1D NMR spectra and chemical formula. We frame structure elucidation as conditional generation from an atomic diffusion model built on a non-equivariant transformer architecture. To model the complex chemical groups found in natural products, we generated a dataset of simulated 1D NMR spectra for over 111,000 natural products. ChefNMR predicts the structures of challenging natural product compounds with an unsurpassed accuracy of over 65%. This work takes a significant step toward solving the grand challenge of automating small-molecule structure elucidation and highlights the potential of deep learning in accelerating molecular discovery. Code is available at this https URL. 

---
# Mitigating Intra- and Inter-modal Forgetting in Continual Learning of Unified Multimodal Models 

**Authors**: Xiwen Wei, Mustafa Munir, Radu Marculescu  

**Link**: [PDF](https://arxiv.org/pdf/2512.03125)  

**Abstract**: Unified Multimodal Generative Models (UMGMs) unify visual understanding and image generation within a single autoregressive framework. However, their ability to continually learn new tasks is severely hindered by catastrophic forgetting, both within a modality (intra-modal) and across modalities (inter-modal). While intra-modal forgetting has been studied in prior continual learning (CL) work, inter-modal forgetting remains largely unexplored. In this paper, we identify and empirically validate this phenomenon in UMGMs and provide a theoretical explanation rooted in gradient conflict between modalities. To address both intra- and inter-modal forgetting, we propose Modality-Decoupled Experts (MoDE), a lightweight and scalable architecture that isolates modality-specific updates to mitigate the gradient conflict and leverages knowledge distillation to prevent catastrophic forgetting and preserve pre-trained capabilities. Unlike previous CL methods that remain modality-coupled and suffer from modality gradient conflict, MoDE explicitly decouples modalities to prevent interference. Experiments across diverse benchmarks demonstrate that MoDE significantly mitigates both inter- and intra-modal forgetting, outperforming prior CL baselines in unified multimodal generation settings. Codes will be publicly available: this https URL 

---
# Lost in Modality: Evaluating the Effectiveness of Text-Based Membership Inference Attacks on Large Multimodal Models 

**Authors**: Ziyi Tong, Feifei Sun, Le Minh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2512.03121)  

**Abstract**: Large Multimodal Language Models (MLLMs) are emerging as one of the foundational tools in an expanding range of applications. Consequently, understanding training-data leakage in these systems is increasingly critical. Log-probability-based membership inference attacks (MIAs) have become a widely adopted approach for assessing data exposure in large language models (LLMs), yet their effect in MLLMs remains unclear. We present the first comprehensive evaluation of extending these text-based MIA methods to multimodal settings. Our experiments under vision-and-text (V+T) and text-only (T-only) conditions across the DeepSeek-VL and InternVL model families show that in in-distribution settings, logit-based MIAs perform comparably across configurations, with a slight V+T advantage. Conversely, in out-of-distribution settings, visual inputs act as regularizers, effectively masking membership signals. 

---
# Beyond Additivity: Sparse Isotonic Shapley Regression toward Nonlinear Explainability 

**Authors**: Jialai She  

**Link**: [PDF](https://arxiv.org/pdf/2512.03112)  

**Abstract**: Shapley values, a gold standard for feature attribution in Explainable AI, face two primary challenges. First, the canonical Shapley framework assumes that the worth function is additive, yet real-world payoff constructions--driven by non-Gaussian distributions, heavy tails, feature dependence, or domain-specific loss scales--often violate this assumption, leading to distorted attributions. Secondly, achieving sparse explanations in high dimensions by computing dense Shapley values and then applying ad hoc thresholding is prohibitively costly and risks inconsistency. We introduce Sparse Isotonic Shapley Regression (SISR), a unified nonlinear explanation framework. SISR simultaneously learns a monotonic transformation to restore additivity--obviating the need for a closed-form specification--and enforces an L0 sparsity constraint on the Shapley vector, enhancing computational efficiency in large feature spaces. Its optimization algorithm leverages Pool-Adjacent-Violators for efficient isotonic regression and normalized hard-thresholding for support selection, yielding implementation ease and global convergence guarantees. Analysis shows that SISR recovers the true transformation in a wide range of scenarios and achieves strong support recovery even in high noise. Moreover, we are the first to demonstrate that irrelevant features and inter-feature dependencies can induce a true payoff transformation that deviates substantially from linearity. Experiments in regression, logistic regression, and tree ensembles demonstrate that SISR stabilizes attributions across payoff schemes, correctly filters irrelevant features while standard Shapley values suffer severe rank and sign distortions. By unifying nonlinear transformation estimation with sparsity pursuit, SISR advances the frontier of nonlinear explainability, providing a theoretically grounded and practical attribution framework. 

---
# PanFoMa: A Lightweight Foundation Model and Benchmark for Pan-Cancer 

**Authors**: Xiaoshui Huang, Tianlin Zhu, Yifan Zuo, Xue Xia, Zonghan Wu, Jiebin Yan, Dingli Hua, Zongyi Xu, Yuming Fang, Jian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.03111)  

**Abstract**: Single-cell RNA sequencing (scRNA-seq) is essential for decoding tumor heterogeneity. However, pan-cancer research still faces two key challenges: learning discriminative and efficient single-cell representations, and establishing a comprehensive evaluation benchmark. In this paper, we introduce PanFoMa, a lightweight hybrid neural network that combines the strengths of Transformers and state-space models to achieve a balance between performance and efficiency. PanFoMa consists of a front-end local-context encoder with shared self-attention layers to capture complex, order-independent gene interactions; and a back-end global sequential feature decoder that efficiently integrates global context using a linear-time state-space model. This modular design preserves the expressive power of Transformers while leveraging the scalability of Mamba to enable transcriptome modeling, effectively capturing both local and global regulatory signals. To enable robust evaluation, we also construct a large-scale pan-cancer single-cell benchmark, PanFoMaBench, containing over 3.5 million high-quality cells across 33 cancer subtypes, curated through a rigorous preprocessing pipeline. Experimental results show that PanFoMa outperforms state-of-the-art models on our pan-cancer benchmark (+4.0\%) and across multiple public tasks, including cell type annotation (+7.4\%), batch integration (+4.0\%) and multi-omics integration (+3.1\%). The code is available at this https URL. 

---
# The BEAT-CF Causal Model: A model for guiding the design of trials and observational analyses of cystic fibrosis exacerbations 

**Authors**: Steven Mascaro, Owen Woodberry, Charlie McLeod, Mitch Messer, Hiran Selvadurai, Yue Wu, Andre Schultz, Thomas L Snelling  

**Link**: [PDF](https://arxiv.org/pdf/2512.03110)  

**Abstract**: Loss of lung function in cystic fibrosis (CF) occurs progressively, punctuated by acute pulmonary exacerbations (PEx) in which abrupt declines in lung function are not fully recovered. A key component of CF management over the past half century has been the treatment of PEx to slow lung function decline. This has been credited with improvements in survival for people with CF (PwCF), but there is no consensus on the optimal approach to PEx management. BEAT-CF (Bayesian evidence-adaptive treatment of CF) was established to build an evidence-informed knowledge base for CF management. The BEAT-CF causal model is a directed acyclic graph (DAG) and Bayesian network (BN) for PEx that aims to inform the design and analysis of clinical trials comparing the effectiveness of alternative approaches to PEx management. The causal model describes relationships between background risk factors, treatments, and pathogen colonisation of the airways that affect the outcome of an individual PEx episode. The key factors, outcomes, and causal relationships were elicited from CF clinical experts and together represent current expert understanding of the pathophysiology of a PEx episode, guiding the design of data collection and studies and enabling causal inference. Here, we present the DAG that documents this understanding, along with the processes used in its development, providing transparency around our trial design and study processes, as well as a reusable framework for others. 

---
# E-valuator: Reliable Agent Verifiers with Sequential Hypothesis Testing 

**Authors**: Shuvom Sadhuka, Drew Prinster, Clara Fannjiang, Gabriele Scalia, Aviv Regev, Hanchen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.03109)  

**Abstract**: Agentic AI systems execute a sequence of actions, such as reasoning steps or tool calls, in response to a user prompt. To evaluate the success of their trajectories, researchers have developed verifiers, such as LLM judges and process-reward models, to score the quality of each action in an agent's trajectory. Although these heuristic scores can be informative, there are no guarantees of correctness when used to decide whether an agent will yield a successful output. Here, we introduce e-valuator, a method to convert any black-box verifier score into a decision rule with provable control of false alarm rates. We frame the problem of distinguishing successful trajectories (that is, a sequence of actions that will lead to a correct response to the user's prompt) and unsuccessful trajectories as a sequential hypothesis testing problem. E-valuator builds on tools from e-processes to develop a sequential hypothesis test that remains statistically valid at every step of an agent's trajectory, enabling online monitoring of agents over arbitrarily long sequences of actions. Empirically, we demonstrate that e-valuator provides greater statistical power and better false alarm rate control than other strategies across six datasets and three agents. We additionally show that e-valuator can be used for to quickly terminate problematic trajectories and save tokens. Together, e-valuator provides a lightweight, model-agnostic framework that converts verifier heuristics into decisions rules with statistical guarantees, enabling the deployment of more reliable agentic systems. 

---
# Public Sentiment Analysis of Traffic Management Policies in Knoxville: A Social Media Driven Study 

**Authors**: Shampa Saha, Shovan Roy  

**Link**: [PDF](https://arxiv.org/pdf/2512.03103)  

**Abstract**: This study presents a comprehensive analysis of public sentiment toward traffic management policies in Knoxville, Tennessee, utilizing social media data from Twitter and Reddit platforms. We collected and analyzed 7906 posts spanning January 2022 to December 2023, employing Valence Aware Dictionary and sEntiment Reasoner (VADER) for sentiment analysis and Latent Dirichlet Allocation (LDA) for topic modeling. Our findings reveal predominantly negative sentiment, with significant variations across platforms and topics. Twitter exhibited more negative sentiment compared to Reddit. Topic modeling identified six distinct themes, with construction-related topics showing the most negative sentiment while general traffic discussions were more positive. Spatiotemporal analysis revealed geographic and temporal patterns in sentiment expression. The research demonstrates social media's potential as a real-time public sentiment monitoring tool for transportation planning and policy evaluation. 

---
# Dynamic Correction of Erroneous State Estimates via Diffusion Bayesian Exploration 

**Authors**: Yiwei Shi, Hongnan Ma, Mengyue Yang, Cunjia Liu, Weiru Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.03102)  

**Abstract**: In emergency response and other high-stakes societal applications, early-stage state estimates critically shape downstream outcomes. Yet, these initial state estimates-often based on limited or biased information-can be severely misaligned with reality, constraining subsequent actions and potentially causing catastrophic delays, resource misallocation, and human harm. Under the stationary bootstrap baseline (zero transition and no rejuvenation), bootstrap particle filters exhibit Stationarity-Induced Posterior Support Invariance (S-PSI), wherein regions excluded by the initial prior remain permanently unexplorable, making corrections impossible even when new evidence contradicts current beliefs. While classical perturbations can in principle break this lock-in, they operate in an always-on fashion and may be inefficient. To overcome this, we propose a diffusion-driven Bayesian exploration framework that enables principled, real-time correction of early state estimation errors. Our method expands posterior support via entropy-regularized sampling and covariance-scaled diffusion. A Metropolis-Hastings check validates proposals and keeps inference adaptive to unexpected evidence. Empirical evaluations on realistic hazardous-gas localization tasks show that our approach matches reinforcement learning and planning baselines when priors are correct. It substantially outperforms classical SMC perturbations and RL-based methods under misalignment, and we provide theoretical guarantees that DEPF resolves S-PSI while maintaining statistical rigor. 

---
# ALARM: Automated MLLM-Based Anomaly Detection in Complex-EnviRonment Monitoring with Uncertainty Quantification 

**Authors**: Congjing Zhang, Feng Lin, Xinyi Zhao, Pei Guo, Wei Li, Lin Chen, Chaoyue Zhao, Shuai Huang  

**Link**: [PDF](https://arxiv.org/pdf/2512.03101)  

**Abstract**: The advance of Large Language Models (LLMs) has greatly stimulated research interest in developing multi-modal LLM (MLLM)-based visual anomaly detection (VAD) algorithms that can be deployed in complex environments. The challenge is that in these complex environments, the anomalies are sometimes highly contextual and also ambiguous, and thereby, uncertainty quantification (UQ) is a crucial capacity for an MLLM-based VAD system to succeed. In this paper, we introduce our UQ-supported MLLM-based VAD framework called ALARM. ALARM integrates UQ with quality-assurance techniques like reasoning chain, self-reflection, and MLLM ensemble for robust and accurate performance and is designed based on a rigorous probabilistic inference pipeline and computational process. Extensive empirical evaluations are conducted using the real-world smart-home benchmark data and wound image classification data, which shows ALARM's superior performance and its generic applicability across different domains for reliable decision-making. 

---
# Ensemble Privacy Defense for Knowledge-Intensive LLMs against Membership Inference Attacks 

**Authors**: Haowei Fu, Bo Ni, Han Xu, Kunpeng Liu, Dan Lin, Tyler Derr  

**Link**: [PDF](https://arxiv.org/pdf/2512.03100)  

**Abstract**: Retrieval-Augmented Generation (RAG) and Supervised Finetuning (SFT) have become the predominant paradigms for equipping Large Language Models (LLMs) with external knowledge for diverse, knowledge-intensive tasks. However, while such knowledge injection improves performance, it also exposes new attack surfaces. Membership Inference Attacks (MIAs), which aim to determine whether a given data sample was included in a model's training set, pose serious threats to privacy and trust in sensitive domains. To this end, we first systematically evaluate the vulnerability of RAG- and SFT-based LLMs to various MIAs. Then, to address the privacy risk, we further introduce a novel, model-agnostic defense framework, Ensemble Privacy Defense (EPD), which aggregates and evaluates the outputs of a knowledge-injected LLM, a base LLM, and a dedicated judge model to enhance resistance against MIAs. Comprehensive experiments show that, on average, EPD reduces MIA success by up to 27.8\% for SFT and 526.3\% for RAG compared to inference-time baseline, while maintaining answer quality. 

---
# QGShap: Quantum Acceleration for Faithful GNN Explanations 

**Authors**: Haribandhu Jena, Jyotirmaya Shivottam, Subhankar Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2512.03099)  

**Abstract**: Graph Neural Networks (GNNs) have become indispensable in critical domains such as drug discovery, social network analysis, and recommendation systems, yet their black-box nature hinders deployment in scenarios requiring transparency and accountability. While Shapley value-based methods offer mathematically principled explanations by quantifying each component's contribution to predictions, computing exact values requires evaluating $2^n$ coalitions (or aggregating over $n!$ permutations), which is intractable for real-world graphs. Existing approximation strategies sacrifice either fidelity or efficiency, limiting their practical utility. We introduce QGShap, a quantum computing approach that leverages amplitude amplification to achieve quadratic speedups in coalition evaluation while maintaining exact Shapley computation. Unlike classical sampling or surrogate methods, our approach provides fully faithful explanations without approximation trade-offs for tractable graph sizes. We conduct empirical evaluations on synthetic graph datasets, demonstrating that QGShap achieves consistently high fidelity and explanation accuracy, matching or exceeding the performance of classical methods across all evaluation metrics. These results collectively demonstrate that QGShap not only preserves exact Shapley faithfulness but also delivers interpretable, stable, and structurally consistent explanations that align with the underlying graph reasoning of GNNs. The implementation of QGShap is available at this https URL. 

---
# Community Quality and Influence Maximization: An Empirical Study 

**Authors**: Motaz Ben Hassine  

**Link**: [PDF](https://arxiv.org/pdf/2512.03095)  

**Abstract**: Influence maximization in social networks plays a vital role in applications such as viral marketing, epidemiology, product recommendation, opinion mining, and counter-terrorism. A common approach identifies seed nodes by first detecting disjoint communities and subsequently selecting representative nodes from these communities. However, whether the quality of detected communities consistently affects the spread of influence under the Independent Cascade model remains unclear. This paper addresses this question by extending a previously proposed disjoint community detection method, termed $\alpha$-Hierarchical Clustering, to the influence maximization problem under the Independent Cascade model. The proposed method is compared with an alternative approach that employs the same seed selection criteria but relies on communities of lower quality obtained through standard Hierarchical Clustering. The former is referred to as Hierarchical Clustering-based Influence Maximization, while the latter, which leverages higher-quality community structures to guide seed selection, is termed $\alpha$-Hierarchical Clustering-based Influence Maximization. Extensive experiments are performed on multiple real-world datasets to assess the effectiveness of both methods. The results demonstrate that higher-quality community structures substantially improve information diffusion under the Independent Cascade model, particularly when the propagation probability is low. These findings underscore the critical importance of community quality in guiding effective seed selection for influence maximization in complex networks. 

---
# Password-Activated Shutdown Protocols for Misaligned Frontier Agents 

**Authors**: Kai Williams, Rohan Subramani, Francis Rhys Ward  

**Link**: [PDF](https://arxiv.org/pdf/2512.03089)  

**Abstract**: Frontier AI developers may fail to align or control highly-capable AI agents. In many cases, it could be useful to have emergency shutdown mechanisms which effectively prevent misaligned agents from carrying out harmful actions in the world. We introduce password-activated shutdown protocols (PAS protocols) -- methods for designing frontier agents to implement a safe shutdown protocol when given a password. We motivate PAS protocols by describing intuitive use-cases in which they mitigate risks from misaligned systems that subvert other control efforts, for instance, by disabling automated monitors or self-exfiltrating to external data centres. PAS protocols supplement other safety efforts, such as alignment fine-tuning or monitoring, contributing to defence-in-depth against AI risk. We provide a concrete demonstration in SHADE-Arena, a benchmark for AI monitoring and subversion capabilities, in which PAS protocols supplement monitoring to increase safety with little cost to performance. Next, PAS protocols should be robust to malicious actors who want to bypass shutdown. Therefore, we conduct a red-team blue-team game between the developers (blue-team), who must implement a robust PAS protocol, and a red-team trying to subvert the protocol. We conduct experiments in a code-generation setting, finding that there are effective strategies for the red-team, such as using another model to filter inputs, or fine-tuning the model to prevent shutdown behaviour. We then outline key challenges to implementing PAS protocols in real-life systems, including: security considerations of the password and decisions regarding when, and in which systems, to use them. PAS protocols are an intuitive mechanism for increasing the safety of frontier AI. We encourage developers to consider implementing PAS protocols prior to internal deployment of particularly dangerous systems to reduce loss-of-control risks. 

---
# When Harmful Content Gets Camouflaged: Unveiling Perception Failure of LVLMs with CamHarmTI 

**Authors**: Yanhui Li, Qi Zhou, Zhihong Xu, Huizhong Guo, Wenhai Wang, Dongxia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.03087)  

**Abstract**: Large vision-language models (LVLMs) are increasingly used for tasks where detecting multimodal harmful content is crucial, such as online content moderation. However, real-world harmful content is often camouflaged, relying on nuanced text-image interplay, such as memes or images with embedded malicious text, to evade detection. This raises a key question: \textbf{can LVLMs perceive such camouflaged harmful content as sensitively as humans do?} In this paper, we introduce CamHarmTI, a benchmark for evaluating LVLM ability to perceive and interpret camouflaged harmful content within text-image compositions. CamHarmTI consists of over 4,500 samples across three types of image-text posts. Experiments on 100 human users and 12 mainstream LVLMs reveal a clear perceptual gap: humans easily recognize such content (e.g., over 95.75\% accuracy), whereas current LVLMs often fail (e.g., ChatGPT-4o achieves only 2.10\% accuracy). Moreover, fine-tuning experiments demonstrate that \bench serves as an effective resource for improving model perception, increasing accuracy by 55.94\% for Qwen2.5VL-7B. Attention analysis and layer-wise probing further reveal that fine-tuning enhances sensitivity primarily in the early layers of the vision encoder, promoting a more integrated scene understanding. These findings highlight the inherent perceptual limitations in LVLMs and offer insight into more human-aligned visual reasoning systems. 

---
# Beyond Code Pairs: Dialogue-Based Data Generation for LLM Code Translation 

**Authors**: Le Chen, Nuo Xu, Winson Chen, Bin Lei, Pei-Hung Lin, Dunzhi Zhou, Rajeev Thakur, Caiwen Ding, Ali Jannesari, Chunhua Liao  

**Link**: [PDF](https://arxiv.org/pdf/2512.03086)  

**Abstract**: Large language models (LLMs) have shown remarkable capabilities in code translation, yet their performance deteriorates in low-resource programming domains such as Fortran and emerging frameworks like CUDA, where high-quality parallel data are scarce. We present an automated dataset generation pipeline featuring a dual-LLM Questioner-Solver design that incorporates external knowledge from compilers and runtime feedback. Beyond traditional source-target code pair datasets, our approach additionally generates (1) verified translations with unit tests for assessing functional consistency, and (2) multi-turn dialogues that capture the reasoning process behind translation refinement. Applied to Fortran -> C++ and C++ -> CUDA, the pipeline yields 3.64k and 3.93k dialogues, respectively. Fine-tuning on this data yields dramatic improvements in functional correctness, boosting unit test success rates by over 56% on the challenging C++-to-CUDA task. We show this data enables a 7B open-weight model to significantly outperform larger proprietary systems on key metrics like compilation success. 

---
# Alleviating Choice Supportive Bias in LLM with Reasoning Dependency Generation 

**Authors**: Nan Zhuang, Wenshuo Wang, Lekai Qian, Yuxiao Wang, Boyu Cao, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.03082)  

**Abstract**: Recent studies have demonstrated that some Large Language Models exhibit choice-supportive bias (CSB) when performing evaluations, systematically favoring their chosen options and potentially compromising the objectivity of AI-assisted decision making. While existing debiasing approaches primarily target demographic and social biases, methods for addressing cognitive biases in LLMs remain largely unexplored. In this work, we present the first solution to address CSB through Reasoning Dependency Generation (RDG), a novel framework for generating unbiased reasoning data to mitigate choice-supportive bias through fine-tuning. RDG automatically constructs balanced reasoning QA pairs, explicitly (un)modeling the dependencies between choices, evidences, and justifications. Our approach is able to generate a large-scale dataset of QA pairs across domains, incorporating Contextual Dependency Data and Dependency Decouple Data. Experiments show that LLMs fine-tuned on RDG-generated data demonstrate a 81.5% improvement in memory-based experiments and 94.3% improvement in the evaluation-based experiment, while maintaining similar performance on standard BBQ benchmarks. This work pioneers an approach for addressing cognitive biases in LLMs and contributes to the development of more reliable AI-assisted decision support systems. 

---
# AtomDisc: An Atom-level Tokenizer that Boosts Molecular LLMs and Reveals Structure--Property Associations 

**Authors**: Mingxu Zhang, Dazhong Shen, Ying Sun  

**Link**: [PDF](https://arxiv.org/pdf/2512.03080)  

**Abstract**: Advances in large language models (LLMs) are accelerating discovery in molecular science. However, adapting molecular information to the serialized, token-based processing of LLMs remains a key challenge. Compared to other representations, molecular graphs explicitly encode atomic connectivity and local topological environments, which are key determinants of atomic behavior and molecular properties. Despite recent efforts to tokenize overall molecular topology, there still lacks effective fine-grained tokenization of local atomic environments, which are critical for determining sophisticated chemical properties and reactivity. To address these issues, we introduce AtomDisc, a novel framework that quantizes atom-level local environments into structure-aware tokens embedded directly in LLM's token space. Our experiments show that AtomDisc, in a data-driven way, can distinguish chemically meaningful structural features that reveal structure-property associations. Equipping LLMs with AtomDisc tokens injects an interpretable inductive bias that delivers state-of-the-art performance on property prediction and molecular generation. Our methodology and findings can pave the way for constructing more powerful molecular LLMs aimed at mechanistic insight and complex chemical reasoning. 

---
# Irresponsible AI: big tech's influence on AI research and associated impacts 

**Authors**: Alex Hernandez-Garcia, Alexandra Volokhova, Ezekiel Williams, Dounia Shaaban Kabakibo  

**Link**: [PDF](https://arxiv.org/pdf/2512.03077)  

**Abstract**: The accelerated development, deployment and adoption of artificial intelligence systems has been fuelled by the increasing involvement of big tech. This has been accompanied by increasing ethical concerns and intensified societal and environmental impacts. In this article, we review and discuss how these phenomena are deeply entangled. First, we examine the growing and disproportionate influence of big tech in AI research and argue that its drive for scaling and general-purpose systems is fundamentally at odds with the responsible, ethical, and sustainable development of AI. Second, we review key current environmental and societal negative impacts of AI and trace their connections to big tech and its underlying economic incentives. Finally, we argue that while it is important to develop technical and regulatory approaches to these challenges, these alone are insufficient to counter the distortion introduced by big tech's influence. We thus review and propose alternative strategies that build on the responsibility of implicated actors and collective action. 

---
# Will Power Return to the Clouds? From Divine Authority to GenAI Authority 

**Authors**: Mohammad Saleh Torkestani, Taha Mansouri  

**Link**: [PDF](https://arxiv.org/pdf/2512.03076)  

**Abstract**: Generative AI systems now mediate newsfeeds, search rankings, and creative content for hundreds of millions of users, positioning a handful of private firms as de-facto arbiters of truth. Drawing on a comparative-historical lens, this article juxtaposes the Galileo Affair, a touchstone of clerical knowledge control, with contemporary Big-Tech content moderation. We integrate Foucault's power/knowledge thesis, Weber's authority types (extended to a rational-technical and emerging agentic-technical modality), and Floridi's Dataism to analyze five recurrent dimensions: disciplinary power, authority modality, data pluralism, trust versus reliance, and resistance pathways. Primary sources (Inquisition records; platform transparency reports) and recent empirical studies on AI trust provide the evidentiary base. Findings show strong structural convergences: highly centralized gatekeeping, legitimacy claims couched in transcendent principles, and systematic exclusion of marginal voices. Divergences lie in temporal velocity, global scale, and the widening gap between public reliance and trust in AI systems. Ethical challenges cluster around algorithmic opacity, linguistic inequity, bias feedback loops, and synthetic misinformation. We propose a four-pillar governance blueprint: (1) a mandatory international model-registry with versioned policy logs, (2) representation quotas and regional observatories to de-center English-language hegemony, (3) mass critical-AI literacy initiatives, and (4) public-private support for community-led data trusts. Taken together, these measures aim to narrow the trust-reliance gap and prevent GenAI from hardcoding a twenty-first-century digital orthodoxy. 

---
# Economies of Open Intelligence: Tracing Power & Participation in the Model Ecosystem 

**Authors**: Shayne Longpre, Christopher Akiki, Campbell Lund, Atharva Kulkarni, Emily Chen, Irene Solaiman, Avijit Ghosh, Yacine Jernite, Lucie-Aimée Kaffee  

**Link**: [PDF](https://arxiv.org/pdf/2512.03073)  

**Abstract**: Since 2019, the Hugging Face Model Hub has been the primary global platform for sharing open weight AI models. By releasing a dataset of the complete history of weekly model downloads (June 2020-August 2025) alongside model metadata, we provide the most rigorous examination to-date of concentration dynamics and evolving characteristics in the open model economy. Our analysis spans 851,000 models, over 200 aggregated attributes per model, and 2.2B downloads. We document a fundamental rebalancing of economic power: US open-weight industry dominance by Google, Meta, and OpenAI has declined sharply in favor of unaffiliated developers, community organizations, and, as of 2025, Chinese industry, with DeepSeek and Qwen models potentially heralding a new consolidation of market power. We identify statistically significant shifts in model properties, a 17X increase in average model size, rapid growth in multimodal generation (3.4X), quantization (5X), and mixture-of-experts architectures (7X), alongside concerning declines in data transparency, with open weights models surpassing truly open source models for the first time in 2025. We expose a new layer of developer intermediaries that has emerged, focused on quantizing and adapting base models for both efficiency and artistic expression. To enable continued research and oversight, we release the complete dataset with an interactive dashboard for real-time monitoring of concentration dynamics and evolving properties in the open model economy. 

---
# PretopoMD: Pretopology-based Mixed Data Hierarchical Clustering 

**Authors**: Loup-Noe Levy, Guillaume Guerard, Sonia Djebali, Soufian Ben Amor  

**Link**: [PDF](https://arxiv.org/pdf/2512.03071)  

**Abstract**: This article presents a novel pretopology-based algorithm designed to address the challenges of clustering mixed data without the need for dimensionality reduction. Leveraging Disjunctive Normal Form, our approach formulates customizable logical rules and adjustable hyperparameters that allow for user-defined hierarchical cluster construction and facilitate tailored solutions for heterogeneous datasets. Through hierarchical dendrogram analysis and comparative clustering metrics, our method demonstrates superior performance by accurately and interpretably delineating clusters directly from raw data, thus preserving data integrity. Empirical findings highlight the algorithm's robustness in constructing meaningful clusters and reveal its potential in overcoming issues related to clustered data explainability. The novelty of this work lies in its departure from traditional dimensionality reduction techniques and its innovative use of logical rules that enhance both cluster formation and clarity, thereby contributing a significant advancement to the discourse on clustering mixed data. 

---
# Mixed Data Clustering Survey and Challenges 

**Authors**: Guillaume Guerard, Sonia Djebali  

**Link**: [PDF](https://arxiv.org/pdf/2512.03070)  

**Abstract**: The advent of the big data paradigm has transformed how industries manage and analyze information, ushering in an era of unprecedented data volume, velocity, and variety. Within this landscape, mixed-data clustering has become a critical challenge, requiring innovative methods that can effectively exploit heterogeneous data types, including numerical and categorical variables. Traditional clustering techniques, typically designed for homogeneous datasets, often struggle to capture the additional complexity introduced by mixed data, underscoring the need for approaches specifically tailored to this setting. Hierarchical and explainable algorithms are particularly valuable in this context, as they provide structured, interpretable clustering results that support informed decision-making. This paper introduces a clustering method grounded in pretopological spaces. In addition, benchmarking against classical numerical clustering algorithms and existing pretopological approaches yields insights into the performance and effectiveness of the proposed method within the big data paradigm. 

---
# Hierarchical clustering of complex energy systems using pretopology 

**Authors**: Loup-Noe Levy, Jeremie Bosom, Guillaume Guerard, Soufian Ben Amor, Marc Bui, Hai Tran  

**Link**: [PDF](https://arxiv.org/pdf/2512.03069)  

**Abstract**: This article attempts answering the following problematic: How to model and classify energy consumption profiles over a large distributed territory to optimize the management of buildings' consumption?
Doing case-by-case in depth auditing of thousands of buildings would require a massive amount of time and money as well as a significant number of qualified people. Thus, an automated method must be developed to establish a relevant and effective recommendations system.
To answer this problematic, pretopology is used to model the sites' consumption profiles and a multi-criterion hierarchical classification algorithm, using the properties of pretopological space, has been developed in a Python library.
To evaluate the results, three data sets are used: A generated set of dots of various sizes in a 2D space, a generated set of time series and a set of consumption time series of 400 real consumption sites from a French Energy company.
On the point data set, the algorithm is able to identify the clusters of points using their position in space and their size as parameter. On the generated time series, the algorithm is able to identify the time series clusters using Pearson's correlation with an Adjusted Rand Index (ARI) of 1. 

---
# Echoes of AI Harms: A Human-LLM Synergistic Framework for Bias-Driven Harm Anticipation 

**Authors**: Nicoleta Tantalaki, Sophia Vei, Athena Vakali  

**Link**: [PDF](https://arxiv.org/pdf/2512.03068)  

**Abstract**: The growing influence of Artificial Intelligence (AI) systems on decision-making in critical domains has exposed their potential to cause significant harms, often rooted in biases embedded across the AI lifecycle. While existing frameworks and taxonomies document bias or harms in isolation, they rarely establish systematic links between specific bias types and the harms they cause, particularly within real-world sociotechnical contexts. Technical fixes proposed to address AI biases are ill-equipped to address them and are typically applied after a system has been developed or deployed, offering limited preventive value. We propose ECHO, a novel framework for proactive AI harm anticipation through the systematic mapping of AI bias types to harm outcomes across diverse stakeholder and domain contexts. ECHO follows a modular workflow encompassing stakeholder identification, vignette-based presentation of biased AI systems, and dual (human-LLM) harm annotation, integrated within ethical matrices for structured interpretation. This human-centered approach enables early-stage detection of bias-to-harm pathways, guiding AI design and governance decisions from the outset. We validate ECHO in two high-stakes domains (disease diagnosis and hiring), revealing domain-specific, bias-to-harm patterns and demonstrating ECHO's potential to support anticipatory governance of AI systems 

---
# Quantifying the Potential to Escape Filter Bubbles: A Behavior-Aware Measure via Contrastive Simulation 

**Authors**: Difu Feng, Qianqian Xu, Zitai Wang, Cong Hua, Zhiyong Yang, Qingming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2512.03067)  

**Abstract**: Nowadays, recommendation systems have become crucial to online platforms, shaping user exposure by accurate preference modeling. However, such an exposure strategy can also reinforce users' existing preferences, leading to a notorious phenomenon named filter bubbles. Given its negative effects, such as group polarization, increasing attention has been paid to exploring reasonable measures to filter bubbles. However, most existing evaluation metrics simply measure the diversity of user exposure, failing to distinguish between algorithmic preference modeling and actual information confinement. In view of this, we introduce Bubble Escape Potential (BEP), a behavior-aware measure that quantifies how easily users can escape from filter bubbles. Specifically, BEP leverages a contrastive simulation framework that assigns different behavioral tendencies (e.g., positive vs. negative) to synthetic users and compares the induced exposure patterns. This design enables decoupling the effect of filter bubbles and preference modeling, allowing for more precise diagnosis of bubble severity. We conduct extensive experiments across multiple recommendation models to examine the relationship between predictive accuracy and bubble escape potential across different groups. To the best of our knowledge, our empirical results are the first to quantitatively validate the dilemma between preference modeling and filter bubbles. What's more, we observe a counter-intuitive phenomenon that mild random recommendations are ineffective in alleviating filter bubbles, which can offer a principled foundation for further work in this direction. 

---
# Optimizing Life Sciences Agents in Real-Time using Reinforcement Learning 

**Authors**: Nihir Chadderwala  

**Link**: [PDF](https://arxiv.org/pdf/2512.03065)  

**Abstract**: Generative AI agents in life sciences face a critical challenge: determining the optimal approach for diverse queries ranging from simple factoid questions to complex mechanistic reasoning. Traditional methods rely on fixed rules or expensive labeled training data, neither of which adapts to changing conditions or user preferences. We present a novel framework that combines AWS Strands Agents with Thompson Sampling contextual bandits to enable AI agents to learn optimal decision-making strategies from user feedback alone. Our system optimizes three key dimensions: generation strategy selection (direct vs. chain-of-thought), tool selection (literature search, drug databases, etc.), and domain routing (pharmacology, molecular biology, clinical specialists). Through empirical evaluation on life science queries, we demonstrate 15-30\% improvement in user satisfaction compared to random baselines, with clear learning patterns emerging after 20-30 queries. Our approach requires no ground truth labels, adapts continuously to user preferences, and provides a principled solution to the exploration-exploitation dilemma in agentic AI systems. 

---
# A note on the impossibility of conditional PAC-efficient reasoning in large language models 

**Authors**: Hao Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2512.03057)  

**Abstract**: We prove an impossibility result for conditional Probably Approximately Correct (PAC)-efficient reasoning in large language models. While recent work has established marginal PAC efficiency guarantees for composite models that switch between expensive expert models and cheaper fast models, we show that conditional (pointwise) guarantees are impossible in the distribution-free setting. Specifically, for non-atomic input spaces, any algorithm achieving conditional PAC efficiency must be trivial in the sense that it defers to the expert model with probability at least $1-\alpha$ for almost every input. 

---
# Delta Sampling: Data-Free Knowledge Transfer Across Diffusion Models 

**Authors**: Zhidong Gao, Zimeng Pan, Yuhang Yao, Chenyue Xie, Wei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2512.03056)  

**Abstract**: Diffusion models like Stable Diffusion (SD) drive a vibrant open-source ecosystem including fully fine-tuned checkpoints and parameter-efficient adapters such as LoRA, LyCORIS, and ControlNet. However, these adaptation components are tightly coupled to a specific base model, making them difficult to reuse when the base model is upgraded (e.g., from SD 1.x to 2.x) due to substantial changes in model parameters and architecture. In this work, we propose Delta Sampling (DS), a novel method that enables knowledge transfer across base models with different architectures, without requiring access to the original training data. DS operates entirely at inference time by leveraging the delta: the difference in model predictions before and after the adaptation of a base model. This delta is then used to guide the denoising process of a new base model. We evaluate DS across various SD versions, demonstrating that DS achieves consistent improvements in creating desired effects (e.g., visual styles, semantic concepts, and structures) under different sampling strategies. These results highlight DS as an effective, plug-and-play mechanism for knowledge transfer in diffusion-based image synthesis. Code:~ this https URL 

---
# Physics-informed self-supervised learning for predictive modeling of coronary artery digital twins 

**Authors**: Xiaowu Sun, Thabo Mahendiran, Ortal Senouf, Denise Auberson, Bernard De Bruyne, Stephane Fournier, Olivier Muller, Pascal Frossard, Emmanuel Abbe, Dorina Thanou  

**Link**: [PDF](https://arxiv.org/pdf/2512.03055)  

**Abstract**: Cardiovascular disease is the leading global cause of mortality, with coronary artery disease (CAD) as its most prevalent form, necessitating early risk prediction. While 3D coronary artery digital twins reconstructed from imaging offer detailed anatomy for personalized assessment, their analysis relies on computationally intensive computational fluid dynamics (CFD), limiting scalability. Data-driven approaches are hindered by scarce labeled data and lack of physiological priors. To address this, we present PINS-CAD, a physics-informed self-supervised learning framework. It pre-trains graph neural networks on 200,000 synthetic coronary digital twins to predict pressure and flow, guided by 1D Navier-Stokes equations and pressure-drop laws, eliminating the need for CFD or labeled data. When fine-tuned on clinical data from 635 patients in the multicenter FAME2 study, PINS-CAD predicts future cardiovascular events with an AUC of 0.73, outperforming clinical risk scores and data-driven baselines. This demonstrates that physics-informed pretraining boosts sample efficiency and yields physiologically meaningful representations. Furthermore, PINS-CAD generates spatially resolved pressure and fractional flow reserve curves, providing interpretable biomarkers. By embedding physical priors into geometric deep learning, PINS-CAD transforms routine angiography into a simulation-free, physiology-aware framework for scalable, preventive cardiology. 

---
# Energy-Efficient Federated Learning via Adaptive Encoder Freezing for MRI-to-CT Conversion: A Green AI-Guided Research 

**Authors**: Ciro Benito Raggio, Lucia Migliorelli, Nils Skupien, Mathias Krohmer Zabaleta, Oliver Blanck, Francesco Cicone, Giuseppe Lucio Cascini, Paolo Zaffino, Maria Francesca Spadea  

**Link**: [PDF](https://arxiv.org/pdf/2512.03054)  

**Abstract**: Federated Learning (FL) holds the potential to advance equality in health by enabling diverse institutions to collaboratively train deep learning (DL) models, even with limited data. However, the significant resource requirements of FL often exclude centres with limited computational infrastructure, further widening existing healthcare disparities. To address this issue, we propose a Green AI-oriented adaptive layer-freezing strategy designed to reduce energy consumption and computational load while maintaining model performance. We tested our approach using different federated architectures for Magnetic Resonance Imaging (MRI)-to-Computed Tomography (CT) conversion. The proposed adaptive strategy optimises the federated training by selectively freezing the encoder weights based on the monitored relative difference of the encoder weights from round to round. A patience-based mechanism ensures that freezing only occurs when updates remain consistently minimal. The energy consumption and CO2eq emissions of the federation were tracked using the CodeCarbon library. Compared to equivalent non-frozen counterparts, our approach reduced training time, total energy consumption and CO2eq emissions by up to 23%. At the same time, the MRI-to-CT conversion performance was maintained, with only small variations in the Mean Absolute Error (MAE). Notably, for three out of the five evaluated architectures, no statistically significant differences were observed, while two architectures exhibited statistically significant improvements. Our work aligns with a research paradigm that promotes DL-based frameworks meeting clinical requirements while ensuring climatic, social, and economic sustainability. It lays the groundwork for novel FL evaluation frameworks, advancing privacy, equity and, more broadly, justice in AI-driven healthcare. 

---
# Mitigating hallucinations and omissions in LLMs for invertible problems: An application to hardware logic design automation 

**Authors**: Andrew S. Cassidy, Guillaume Garreau, Jay Sivagnaname, Mike Grassi, Bernard Brezzo, John V. Arthur, Dharmendra S. Modha  

**Link**: [PDF](https://arxiv.org/pdf/2512.03053)  

**Abstract**: We show for invertible problems that transform data from a source domain (for example, Logic Condition Tables (LCTs)) to a destination domain (for example, Hardware Description Language (HDL) code), an approach of using Large Language Models (LLMs) as a lossless encoder from source to destination followed by as a lossless decoder back to the source, comparable to lossless compression in information theory, can mitigate most of the LLM drawbacks of hallucinations and omissions. Specifically, using LCTs as inputs, we generate the full HDL for a two-dimensional network-on-chip router (13 units, 1500-2000 lines of code) using seven different LLMs, reconstruct the LCTs from the auto-generated HDL, and compare the original and reconstructed LCTs. This approach yields significant productivity improvements, not only confirming correctly generated LLM logic and detecting incorrectly generated LLM logic but also assisting developers in finding design specification errors. 

---
# Entropy-Based Measurement of Value Drift and Alignment Work in Large Language Models 

**Authors**: Samih Fadli  

**Link**: [PDF](https://arxiv.org/pdf/2512.03047)  

**Abstract**: Large language model safety is usually assessed with static benchmarks, but key failures are dynamic: value drift under distribution shift, jailbreak attacks, and slow degradation of alignment in deployment. Building on a recent Second Law of Intelligence that treats ethical entropy as a state variable which tends to increase unless countered by alignment work, we make this framework operational for large language models. We define a five-way behavioral taxonomy, train a classifier to estimate ethical entropy S(t) from model transcripts, and measure entropy dynamics for base and instruction-tuned variants of four frontier models across stress tests. Base models show sustained entropy growth, while tuned variants suppress drift and reduce ethical entropy by roughly eighty percent. From these trajectories we estimate an effective alignment work rate gamma_eff and embed S(t) and gamma_eff in a monitoring pipeline that raises alerts when entropy drift exceeds a stability threshold, enabling run-time oversight of value drift. 

---
# AI-Driven Document Redaction in UK Public Authorities: Implementation Gaps, Regulatory Challenges, and the Human Oversight Imperative 

**Authors**: Yijun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.02774)  

**Abstract**: Document redaction in public authorities faces critical challenges as traditional manual approaches struggle to balance growing transparency demands with increasingly stringent data protection requirements. This study investigates the implementation of AI-driven document redaction within UK public authorities through Freedom of Information (FOI) requests. While AI technologies offer potential solutions to redaction challenges, their actual implementation within public sector organizations remains underexplored. Based on responses from 44 public authorities across healthcare, government, and higher education sectors, this study reveals significant gaps between technological possibilities and organizational realities. Findings show highly limited AI adoption (only one authority reported using AI tools), widespread absence of formal redaction policies (50 percent reported "information not held"), and deficiencies in staff training. The study identifies three key barriers to effective AI implementation: poor record-keeping practices, lack of standardized redaction guidelines, and insufficient specialized training for human oversight. These findings highlight the need for a socio-technical approach that balances technological automation with meaningful human expertise. This research provides the first empirical assessment of AI redaction practices in UK public authorities and contributes evidence to support policymakers navigating the complex interplay between transparency obligations, data protection requirements, and emerging AI technologies in public administration. 

---
# Class conditional conformal prediction for multiple inputs by p-value aggregation 

**Authors**: Jean-Baptiste Fermanian, Mohamed Hebiri, Joseph Salmon  

**Link**: [PDF](https://arxiv.org/pdf/2507.07150)  

**Abstract**: Conformal prediction methods are statistical tools designed to quantify uncertainty and generate predictive sets with guaranteed coverage probabilities. This work introduces an innovative refinement to these methods for classification tasks, specifically tailored for scenarios where multiple observations (multi-inputs) of a single instance are available at prediction time. Our approach is particularly motivated by applications in citizen science, where multiple images of the same plant or animal are captured by individuals. Our method integrates the information from each observation into conformal prediction, enabling a reduction in the size of the predicted label set while preserving the required class-conditional coverage guarantee. The approach is based on the aggregation of conformal p-values computed from each observation of a multi-input. By exploiting the exact distribution of these p-values, we propose a general aggregation framework using an abstract scoring function, encompassing many classical statistical tools. Knowledge of this distribution also enables refined versions of standard strategies, such as majority voting. We evaluate our method on simulated and real data, with a particular focus on Pl@ntNet, a prominent citizen science platform that facilitates the collection and identification of plant species through user-submitted images. 

---
