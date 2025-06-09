# Reflect-then-Plan: Offline Model-Based Planning through a Doubly Bayesian Lens 

**Authors**: Jihwan Jeong, Xiaoyu Wang, Jingmin Wang, Scott Sanner, Pascal Poupart  

**Link**: [PDF](https://arxiv.org/pdf/2506.06261)  

**Abstract**: Offline reinforcement learning (RL) is crucial when online exploration is costly or unsafe but often struggles with high epistemic uncertainty due to limited data. Existing methods rely on fixed conservative policies, restricting adaptivity and generalization. To address this, we propose Reflect-then-Plan (RefPlan), a novel doubly Bayesian offline model-based (MB) planning approach. RefPlan unifies uncertainty modeling and MB planning by recasting planning as Bayesian posterior estimation. At deployment, it updates a belief over environment dynamics using real-time observations, incorporating uncertainty into MB planning via marginalization. Empirical results on standard benchmarks show that RefPlan significantly improves the performance of conservative offline RL policies. In particular, RefPlan maintains robust performance under high epistemic uncertainty and limited data, while demonstrating resilience to changing environment dynamics, improving the flexibility, generalizability, and robustness of offline-learned policies. 

---
# PersonaAgent: When Large Language Model Agents Meet Personalization at Test Time 

**Authors**: Weizhi Zhang, Xinyang Zhang, Chenwei Zhang, Liangwei Yang, Jingbo Shang, Zhepei Wei, Henry Peng Zou, Zijie Huang, Zhengyang Wang, Yifan Gao, Xiaoman Pan, Lian Xiong, Jingguo Liu, Philip S. Yu, Xian Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.06254)  

**Abstract**: Large Language Model (LLM) empowered agents have recently emerged as advanced paradigms that exhibit impressive capabilities in a wide range of domains and tasks. Despite their potential, current LLM agents often adopt a one-size-fits-all approach, lacking the flexibility to respond to users' varying needs and preferences. This limitation motivates us to develop PersonaAgent, the first personalized LLM agent framework designed to address versatile personalization tasks. Specifically, PersonaAgent integrates two complementary components - a personalized memory module that includes episodic and semantic memory mechanisms; a personalized action module that enables the agent to perform tool actions tailored to the user. At the core, the persona (defined as unique system prompt for each user) functions as an intermediary: it leverages insights from personalized memory to control agent actions, while the outcomes of these actions in turn refine the memory. Based on the framework, we propose a test-time user-preference alignment strategy that simulate the latest n interactions to optimize the persona prompt, ensuring real-time user preference alignment through textual loss feedback between simulated and ground-truth responses. Experimental evaluations demonstrate that PersonaAgent significantly outperforms other baseline methods by not only personalizing the action space effectively but also scaling during test-time real-world applications. These results underscore the feasibility and potential of our approach in delivering tailored, dynamic user experiences. 

---
# Integer Linear Programming Preprocessing for Maximum Satisfiability 

**Authors**: Jialu Zhang, Chu-Min Li, Sami Cherif, Shuolin Li, Zhifei Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.06216)  

**Abstract**: The Maximum Satisfiability problem (MaxSAT) is a major optimization challenge with numerous practical applications. In recent MaxSAT evaluations, most MaxSAT solvers have adopted an ILP solver as part of their portfolios. This paper investigates the impact of Integer Linear Programming (ILP) preprocessing techniques on MaxSAT solving. Experimental results show that ILP preprocessing techniques help WMaxCDCL-OpenWbo1200, the winner of the MaxSAT evaluation 2024 in the unweighted track, solve 15 additional instances. Moreover, current state-of-the-art MaxSAT solvers heavily use an ILP solver in their portfolios, while our proposed approach reduces the need to call an ILP solver in a portfolio including WMaxCDCL or MaxCDCL. 

---
# Decomposability-Guaranteed Cooperative Coevolution for Large-Scale Itinerary Planning 

**Authors**: Ziyu Zhang, Peilan Xu, Yuetong Sun, Yuhui Shi, Wenjian Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.06121)  

**Abstract**: Large-scale itinerary planning is a variant of the traveling salesman problem, aiming to determine an optimal path that maximizes the collected points of interest (POIs) scores while minimizing travel time and cost, subject to travel duration constraints. This paper analyzes the decomposability of large-scale itinerary planning, proving that strict decomposability is difficult to satisfy, and introduces a weak decomposability definition based on a necessary condition, deriving the corresponding graph structures that fulfill this property. With decomposability guaranteed, we propose a novel multi-objective cooperative coevolutionary algorithm for large-scale itinerary planning, addressing the challenges of component imbalance and interactions. Specifically, we design a dynamic decomposition strategy based on the normalized fitness within each component, define optimization potential considering component scale and contribution, and develop a computational resource allocation strategy. Finally, we evaluate the proposed algorithm on a set of real-world datasets. Comparative experiments with state-of-the-art multi-objective itinerary planning algorithms demonstrate the superiority of our approach, with performance advantages increasing as the problem scale grows. 

---
# CP-Bench: Evaluating Large Language Models for Constraint Modelling 

**Authors**: Kostis Michailidis, Dimos Tsouros, Tias Guns  

**Link**: [PDF](https://arxiv.org/pdf/2506.06052)  

**Abstract**: Combinatorial problems are present in a wide range of industries. Constraint Programming (CP) is a well-suited problem-solving paradigm, but its core process, namely constraint modelling, is a bottleneck for wider adoption. Aiming to alleviate this bottleneck, recent studies have explored using Large Language Models (LLMs) as modelling assistants, transforming combinatorial problem descriptions to executable constraint models, similar to coding assistants. However, the existing evaluation datasets for constraint modelling are often limited to small, homogeneous, or domain-specific instances, which do not capture the diversity of real-world scenarios. This work addresses this gap by introducing CP-Bench, a novel benchmark dataset that includes a diverse set of well-known combinatorial problem classes sourced from the CP community, structured explicitly for evaluating LLM-driven CP modelling. With this dataset, and given the variety of constraint modelling frameworks, we compare and evaluate the modelling capabilities of LLMs for three distinct constraint modelling systems, which vary in abstraction level and underlying syntax: the high-level MiniZinc language and Python-based CPMpy library, and the lower-level Python interface of the OR-Tools CP-SAT solver. In order to enhance the ability of LLMs to produce valid constraint models, we systematically evaluate the use of prompt-based and inference-time compute methods adapted from existing LLM-based code generation research. Our results underscore the modelling convenience provided by Python-based frameworks, as well as the effectiveness of documentation-rich system prompts, which, augmented with repeated sampling and self-verification, achieve further improvements, reaching up to 70\% accuracy on this new, highly challenging benchmark. 

---
# CrimeMind: Simulating Urban Crime with Multi-Modal LLM Agents 

**Authors**: Qingbin Zeng, Ruotong Zhao, Jinzhu Mao, Haoyang Li, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.05981)  

**Abstract**: Modeling urban crime is an important yet challenging task that requires understanding the subtle visual, social, and cultural cues embedded in urban environments. Previous work has predominantly focused on rule-based agent-based modeling (ABM) and deep learning methods. ABMs offer interpretability of internal mechanisms but exhibit limited predictive this http URL contrast, deep learning methods are often effective in prediction but are less interpretable and require extensive training data. Moreover, both lines of work lack the cognitive flexibility to adapt to changing environments. Leveraging the capabilities of large language models (LLMs), we propose CrimeMind, a novel LLM-driven ABM framework for simulating urban crime within a multi-modal urban context.A key innovation of our design is the integration of the Routine Activity Theory (RAT) into the agentic workflow of CrimeMind, enabling it to process rich multi-modal urban features and reason about criminal this http URL, RAT requires LLM agents to infer subtle cues in evaluating environmental safety as part of assessing guardianship, which can be challenging for LLMs. To address this, we collect a small-scale human-annotated dataset and align CrimeMind's perception with human judgment via a training-free textual gradient this http URL across four major U.S. cities demonstrate that CrimeMind outperforms both traditional ABMs and deep learning baselines in crime hotspot prediction and spatial distribution accuracy, achieving up to a 24% improvement over the strongest this http URL, we conduct counterfactual simulations of external incidents and policy interventions and it successfully captures the expected changes in crime patterns, demonstrating its ability to reflect counterfactual this http URL, CrimeMind enables fine-grained modeling of individual behaviors and facilitates evaluation of real-world interventions. 

---
# Preference Learning for AI Alignment: a Causal Perspective 

**Authors**: Katarzyna Kobalczyk, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2506.05967)  

**Abstract**: Reward modelling from preference data is a crucial step in aligning large language models (LLMs) with human values, requiring robust generalisation to novel prompt-response pairs. In this work, we propose to frame this problem in a causal paradigm, providing the rich toolbox of causality to identify the persistent challenges, such as causal misidentification, preference heterogeneity, and confounding due to user-specific factors. Inheriting from the literature of causal inference, we identify key assumptions necessary for reliable generalisation and contrast them with common data collection practices. We illustrate failure modes of naive reward models and demonstrate how causally-inspired approaches can improve model robustness. Finally, we outline desiderata for future research and practices, advocating targeted interventions to address inherent limitations of observational data. 

---
# Proactive Assistant Dialogue Generation from Streaming Egocentric Videos 

**Authors**: Yichi Zhang, Xin Luna Dong, Zhaojiang Lin, Andrea Madotto, Anuj Kumar, Babak Damavandi, Joyce Chai, Seungwhan Moon  

**Link**: [PDF](https://arxiv.org/pdf/2506.05904)  

**Abstract**: Recent advances in conversational AI have been substantial, but developing real-time systems for perceptual task guidance remains challenging. These systems must provide interactive, proactive assistance based on streaming visual inputs, yet their development is constrained by the costly and labor-intensive process of data collection and system evaluation. To address these limitations, we present a comprehensive framework with three key contributions. First, we introduce a novel data curation pipeline that synthesizes dialogues from annotated egocentric videos, resulting in \dataset, a large-scale synthetic dialogue dataset spanning multiple domains. Second, we develop a suite of automatic evaluation metrics, validated through extensive human studies. Third, we propose an end-to-end model that processes streaming video inputs to generate contextually appropriate responses, incorporating novel techniques for handling data imbalance and long-duration videos. This work lays the foundation for developing real-time, proactive AI assistants capable of guiding users through diverse tasks. Project page: this https URL 

---
# Explainability in Context: A Multilevel Framework Aligning AI Explanations with Stakeholder with LLMs 

**Authors**: Marilyn Bello, Rafael Bello, Maria-Matilde García, Ann Nowé, Iván Sevillano-García, Francisco Herrera  

**Link**: [PDF](https://arxiv.org/pdf/2506.05887)  

**Abstract**: The growing application of artificial intelligence in sensitive domains has intensified the demand for systems that are not only accurate but also explainable and trustworthy. Although explainable AI (XAI) methods have proliferated, many do not consider the diverse audiences that interact with AI systems: from developers and domain experts to end-users and society. This paper addresses how trust in AI is influenced by the design and delivery of explanations and proposes a multilevel framework that aligns explanations with the epistemic, contextual, and ethical expectations of different stakeholders. The framework consists of three layers: algorithmic and domain-based, human-centered, and social explainability. We highlight the emerging role of Large Language Models (LLMs) in enhancing the social layer by generating accessible, natural language explanations. Through illustrative case studies, we demonstrate how this approach facilitates technical fidelity, user engagement, and societal accountability, reframing XAI as a dynamic, trust-building process. 

---
# Trajectory Entropy: Modeling Game State Stability from Multimodality Trajectory Prediction 

**Authors**: Yesheng Zhang, Wenjian Sun, Yuheng Chen, Qingwei Liu, Qi Lin, Rui Zhang, Xu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.05810)  

**Abstract**: Complex interactions among agents present a significant challenge for autonomous driving in real-world scenarios. Recently, a promising approach has emerged, which formulates the interactions of agents as a level-k game framework. It effectively decouples agent policies by hierarchical game levels. However, this framework ignores both the varying driving complexities among agents and the dynamic changes in agent states across game levels, instead treating them uniformly. Consequently, redundant and error-prone computations are introduced into this framework. To tackle the issue, this paper proposes a metric, termed as Trajectory Entropy, to reveal the game status of agents within the level-k game framework. The key insight stems from recognizing the inherit relationship between agent policy uncertainty and the associated driving complexity. Specifically, Trajectory Entropy extracts statistical signals representing uncertainty from the multimodality trajectory prediction results of agents in the game. Then, the signal-to-noise ratio of this signal is utilized to quantify the game status of agents. Based on the proposed Trajectory Entropy, we refine the current level-k game framework through a simple gating mechanism, significantly improving overall accuracy while reducing computational costs. Our method is evaluated on the Waymo and nuPlan datasets, in terms of trajectory prediction, open-loop and closed-loop planning tasks. The results demonstrate the state-of-the-art performance of our method, with precision improved by up to 19.89% for prediction and up to 16.48% for planning. 

---
# Constrained Sampling for Language Models Should Be Easy: An MCMC Perspective 

**Authors**: Emmanuel Anaya Gonzalez, Sairam Vaidya, Kanghee Park, Ruyi Ji, Taylor Berg-Kirkpatrick, Loris D'Antoni  

**Link**: [PDF](https://arxiv.org/pdf/2506.05754)  

**Abstract**: Constrained decoding enables Language Models (LMs) to produce samples that provably satisfy hard constraints. However, existing constrained-decoding approaches often distort the underlying model distribution, a limitation that is especially problematic in applications like program fuzzing, where one wants to generate diverse and valid program inputs for testing purposes. We propose a new constrained sampling framework based on Markov Chain Monte Carlo (MCMC) that simultaneously satisfies three core desiderata: constraint satisfying (every sample satisfies the constraint), monotonically converging (the sampling process converges to the true conditional distribution), and efficient (high-quality samples emerge in few steps). Our method constructs a proposal distribution over valid outputs and applies a Metropolis-Hastings acceptance criterion based on the LM's likelihood, ensuring principled and efficient exploration of the constrained space. Empirically, our sampler outperforms existing methods on both synthetic benchmarks and real-world program fuzzing tasks. 

---
# SPRINT: Enabling Interleaved Planning and Parallelized Execution in Reasoning Models 

**Authors**: Emil Biju, Shayan Talaei, Zhemin Huang, Mohammadreza Pourreza, Azalia Mirhoseini, Amin Saberi  

**Link**: [PDF](https://arxiv.org/pdf/2506.05745)  

**Abstract**: Large reasoning models (LRMs) excel at complex reasoning tasks but typically generate lengthy sequential chains-of-thought, resulting in long inference times before arriving at the final answer. To address this challenge, we introduce SPRINT, a novel post-training and inference-time framework designed to enable LRMs to dynamically identify and exploit opportunities for parallelization during their reasoning process. SPRINT incorporates an innovative data curation pipeline that reorganizes natural language reasoning trajectories into structured rounds of long-horizon planning and parallel execution. By fine-tuning LRMs on a small amount of such curated data, the models learn to dynamically identify independent subtasks within extended reasoning processes and effectively execute them in parallel. Through extensive evaluations, we show that the models fine-tuned with the SPRINT framework match the performance of reasoning models on complex domains such as mathematics while generating up to ~39% fewer sequential tokens on problems requiring more than 8000 output tokens. Finally, we observe consistent results transferred to two out-of-distribution tasks of GPQA and Countdown with up to 45% and 65% reduction in average sequential tokens for longer reasoning trajectories, while achieving the performance of the fine-tuned reasoning model. 

---
# Topology of Reasoning: Understanding Large Reasoning Models through Reasoning Graph Properties 

**Authors**: Gouki Minegishi, Hiroki Furuta, Takeshi Kojima, Yusuke Iwasawa, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2506.05744)  

**Abstract**: Recent large-scale reasoning models have achieved state-of-the-art performance on challenging mathematical benchmarks, yet the internal mechanisms underlying their success remain poorly understood. In this work, we introduce the notion of a reasoning graph, extracted by clustering hidden-state representations at each reasoning step, and systematically analyze three key graph-theoretic properties: cyclicity, diameter, and small-world index, across multiple tasks (GSM8K, MATH500, AIME 2024). Our findings reveal that distilled reasoning models (e.g., DeepSeek-R1-Distill-Qwen-32B) exhibit significantly more recurrent cycles (about 5 per sample), substantially larger graph diameters, and pronounced small-world characteristics (about 6x) compared to their base counterparts. Notably, these structural advantages grow with task difficulty and model capacity, with cycle detection peaking at the 14B scale and exploration diameter maximized in the 32B variant, correlating positively with accuracy. Furthermore, we show that supervised fine-tuning on an improved dataset systematically expands reasoning graph diameters in tandem with performance gains, offering concrete guidelines for dataset design aimed at boosting reasoning capabilities. By bridging theoretical insights into reasoning graph structures with practical recommendations for data construction, our work advances both the interpretability and the efficacy of large reasoning models. 

---
# Population-Proportional Preference Learning from Human Feedback: An Axiomatic Approach 

**Authors**: Kihyun Kim, Jiawei Zhang, Asuman Ozdaglar, Pablo A. Parrilo  

**Link**: [PDF](https://arxiv.org/pdf/2506.05619)  

**Abstract**: Conventional preference learning methods often prioritize opinions held more widely when aggregating preferences from multiple evaluators. This may result in policies that are biased in favor of some types of opinions or groups. The objective of this paper is to develop a novel preference learning framework capable of aligning aggregate opinions and policies proportionally with the true population distribution of evaluator preferences. Our approach infers the feasible set of evaluator population distributions directly from pairwise comparison data. Using these estimates, the algorithm constructs a policy that satisfies foundational axioms from social choice theory, namely monotonicity and Pareto efficiency, as well as our newly-introduced axioms of population-proportional representation and population-bounded robustness. We propose a soft-max relaxation method that smoothly trade-offs population-proportional representation with the selection of the Condorcet winner (which beats all other options in pairwise comparisons). Finally, we validate the effectiveness and scalability of our approach through experiments on both tabular recommendation tasks and large-scale language model alignment. 

---
# Toward Greater Autonomy in Materials Discovery Agents: Unifying Planning, Physics, and Scientists 

**Authors**: Lianhao Zhou, Hongyi Ling, Keqiang Yan, Kaiji Zhao, Xiaoning Qian, Raymundo Arróyave, Xiaofeng Qian, Shuiwang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2506.05616)  

**Abstract**: We aim at designing language agents with greater autonomy for crystal materials discovery. While most of existing studies restrict the agents to perform specific tasks within predefined workflows, we aim to automate workflow planning given high-level goals and scientist intuition. To this end, we propose Materials Agent unifying Planning, Physics, and Scientists, known as MAPPS. MAPPS consists of a Workflow Planner, a Tool Code Generator, and a Scientific Mediator. The Workflow Planner uses large language models (LLMs) to generate structured and multi-step workflows. The Tool Code Generator synthesizes executable Python code for various tasks, including invoking a force field foundation model that encodes physics. The Scientific Mediator coordinates communications, facilitates scientist feedback, and ensures robustness through error reflection and recovery. By unifying planning, physics, and scientists, MAPPS enables flexible and reliable materials discovery with greater autonomy, achieving a five-fold improvement in stability, uniqueness, and novelty rates compared with prior generative models when evaluated on the MP-20 data. We provide extensive experiments across diverse tasks to show that MAPPS is a promising framework for autonomous materials discovery. 

---
# MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark 

**Authors**: Junjie Xing, Yeye He, Mengyu Zhou, Haoyu Dong, Shi Han, Lingjiao Chen, Dongmei Zhang, Surajit Chaudhuri, H. V. Jagadish  

**Link**: [PDF](https://arxiv.org/pdf/2506.05587)  

**Abstract**: Tables and table-based use cases play a crucial role in many important real-world applications, such as spreadsheets, databases, and computational notebooks, which traditionally require expert-level users like data engineers, data analysts, and database administrators to operate. Although LLMs have shown remarkable progress in working with tables (e.g., in spreadsheet and database copilot scenarios), comprehensive benchmarking of such capabilities remains limited. In contrast to an extensive and growing list of NLP benchmarks, evaluations of table-related tasks are scarce, and narrowly focus on tasks like NL-to-SQL and Table-QA, overlooking the broader spectrum of real-world tasks that professional users face. This gap limits our understanding and model progress in this important area.
In this work, we introduce MMTU, a large-scale benchmark with over 30K questions across 25 real-world table tasks, designed to comprehensively evaluate models ability to understand, reason, and manipulate real tables at the expert-level. These tasks are drawn from decades' worth of computer science research on tabular data, with a focus on complex table tasks faced by professional users. We show that MMTU require a combination of skills -- including table understanding, reasoning, and coding -- that remain challenging for today's frontier models, where even frontier reasoning models like OpenAI o4-mini and DeepSeek R1 score only around 60%, suggesting significant room for improvement. We highlight key findings in our evaluation using MMTU and hope that this benchmark drives further advances in understanding and developing foundation models for structured data processing and analysis. Our code and data are available at this https URL and this https URL. 

---
# When Models Know More Than They Can Explain: Quantifying Knowledge Transfer in Human-AI Collaboration 

**Authors**: Quan Shi, Carlos E. Jimenez, Shunyu Yao, Nick Haber, Diyi Yang, Karthik Narasimhan  

**Link**: [PDF](https://arxiv.org/pdf/2506.05579)  

**Abstract**: Recent advancements in AI reasoning have driven substantial improvements across diverse tasks. A critical open question is whether these improvements also yields better knowledge transfer: the ability of models to communicate reasoning in ways humans can understand, apply, and learn from. To investigate this, we introduce Knowledge Integration and Transfer Evaluation (KITE), a conceptual and experimental framework for Human-AI knowledge transfer capabilities and conduct the first large-scale human study (N=118) explicitly designed to measure it. In our two-phase setup, humans first ideate with an AI on problem-solving strategies, then independently implement solutions, isolating model explanations' influence on human understanding. Our findings reveal that although model benchmark performance correlates with collaborative outcomes, this relationship is notably inconsistent, featuring significant outliers, indicating that knowledge transfer requires dedicated optimization. Our analysis identifies behavioral and strategic factors mediating successful knowledge transfer. We release our code, dataset, and evaluation framework to support future work on communicatively aligned models. 

---
# Avoiding Death through Fear Intrinsic Conditioning 

**Authors**: Rodney Sanchez, Ferat Sahin, Alexander Ororbia, Jamison Heard  

**Link**: [PDF](https://arxiv.org/pdf/2506.05529)  

**Abstract**: Biological and psychological concepts have inspired reinforcement learning algorithms to create new complex behaviors that expand agents' capacity. These behaviors can be seen in the rise of techniques like goal decomposition, curriculum, and intrinsic rewards, which have paved the way for these complex behaviors. One limitation in evaluating these methods is the requirement for engineered extrinsic for realistic environments. A central challenge in engineering the necessary reward function(s) comes from these environments containing states that carry high negative rewards, but provide no feedback to the agent. Death is one such stimuli that fails to provide direct feedback to the agent. In this work, we introduce an intrinsic reward function inspired by early amygdala development and produce this intrinsic reward through a novel memory-augmented neural network (MANN) architecture. We show how this intrinsic motivation serves to deter exploration of terminal states and results in avoidance behavior similar to fear conditioning observed in animals. Furthermore, we demonstrate how modifying a threshold where the fear response is active produces a range of behaviors that are described under the paradigm of general anxiety disorders (GADs). We demonstrate this behavior in the Miniworld Sidewalk environment, which provides a partially observable Markov decision process (POMDP) and a sparse reward with a non-descriptive terminal condition, i.e., death. In effect, this study results in a biologically-inspired neural architecture and framework for fear conditioning paradigms; we empirically demonstrate avoidance behavior in a constructed agent that is able to solve environments with non-descriptive terminal conditions. 

---
# Towards Data Systems That Are Business Semantic-Centric and AI Agents-Assisted 

**Authors**: Cecil Pang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05520)  

**Abstract**: Contemporary businesses operate in dynamic environments requiring rapid adaptation to achieve goals and maintain competitiveness. Existing data platforms often fall short by emphasizing tools over alignment with business needs, resulting in inefficiencies and delays. To address this gap, I propose the Business Semantics Centric, AI Agents Assisted Data System (BSDS), a holistic system that integrates architecture, workflows, and team organization to ensure data systems are tailored to business priorities rather than dictated by technical constraints. BSDS redefines data systems as dynamic enablers of business success, transforming them from passive tools into active drivers of organizational growth. BSDS has a modular architecture that comprises curated data linked to business entities, a knowledge base for context-aware AI agents, and efficient data pipelines. AI agents play a pivotal role in assisting with data access and system management, reducing human effort, and improving scalability. Complementing this architecture, BSDS incorporates workflows optimized for both exploratory data analysis and production requirements, balancing speed of delivery with quality assurance. A key innovation of BSDS is its incorporation of the human factor. By aligning data team expertise with business semantics, BSDS bridges the gap between technical capabilities and business needs. Validated through real-world implementation, BSDS accelerates time-to-market for data-driven initiatives, enhances cross-functional collaboration, and provides a scalable blueprint for businesses of all sizes. Future research can build on BSDS to explore optimization strategies using complex systems and adaptive network theories, as well as developing autonomous data systems leveraging AI agents. 

---
# Constructive Symbolic Reinforcement Learning via Intuitionistic Logic and Goal-Chaining Inference 

**Authors**: Andrei T. Patrascu  

**Link**: [PDF](https://arxiv.org/pdf/2506.05422)  

**Abstract**: We introduce a novel learning and planning framework that replaces traditional reward-based optimisation with constructive logical inference. In our model, actions, transitions, and goals are represented as logical propositions, and decision-making proceeds by building constructive proofs under intuitionistic logic. This method ensures that state transitions and policies are accepted only when supported by verifiable preconditions -- eschewing probabilistic trial-and-error in favour of guaranteed logical validity. We implement a symbolic agent operating in a structured gridworld, where reaching a goal requires satisfying a chain of intermediate subgoals (e.g., collecting keys to open doors), each governed by logical constraints. Unlike conventional reinforcement learning agents, which require extensive exploration and suffer from unsafe or invalid transitions, our constructive agent builds a provably correct plan through goal chaining, condition tracking, and knowledge accumulation. Empirical comparison with Q-learning demonstrates that our method achieves perfect safety, interpretable behaviour, and efficient convergence with no invalid actions, highlighting its potential for safe planning, symbolic cognition, and trustworthy AI. This work presents a new direction for reinforcement learning grounded not in numeric optimisation, but in constructive logic and proof theory. 

---
# Contextual Memory Intelligence -- A Foundational Paradigm for Human-AI Collaboration and Reflective Generative AI Systems 

**Authors**: Kristy Wedel  

**Link**: [PDF](https://arxiv.org/pdf/2506.05370)  

**Abstract**: A critical challenge remains unresolved as generative AI systems are quickly implemented in various organizational settings. Despite significant advances in memory components such as RAG, vector stores, and LLM agents, these systems still have substantial memory limitations. Gen AI workflows rarely store or reflect on the full context in which decisions are made. This leads to repeated errors and a general lack of clarity. This paper introduces Contextual Memory Intelligence (CMI) as a new foundational paradigm for building intelligent systems. It repositions memory as an adaptive infrastructure necessary for longitudinal coherence, explainability, and responsible decision-making rather than passive data. Drawing on cognitive science, organizational theory, human-computer interaction, and AI governance, CMI formalizes the structured capture, inference, and regeneration of context as a fundamental system capability. The Insight Layer is presented in this paper to operationalize this vision. This modular architecture uses human-in-the-loop reflection, drift detection, and rationale preservation to incorporate contextual memory into systems. The paper argues that CMI allows systems to reason with data, history, judgment, and changing context, thereby addressing a foundational blind spot in current AI architectures and governance efforts. A framework for creating intelligent systems that are effective, reflective, auditable, and socially responsible is presented through CMI. This enhances human-AI collaboration, generative AI design, and the resilience of the institutions. 

---
# A Path to Loving 

**Authors**: John Beverley, Regina Hurley  

**Link**: [PDF](https://arxiv.org/pdf/2506.05352)  

**Abstract**: This work lays the foundations for a rigorous ontological characterization of love, addressing its philosophical complexity and scientific relevance, with particular emphasis on psychology and sociology, as well as highlighting ways in which such characterization enhances relevant AI based applications. The position defended here is that love is best understood as a concatenation of passive sensations (e.g., emotional arousal) and active evaluative judgments (e.g., perceiving the beloved as valuable), in the interest of balancing the involuntary aspects of love with its rational accountability. To provide a structured foundation, the paper draws on Basic Formal Ontology (BFO) and other applied ontological methods to differentiate various senses of love. This work engages with objections to the understanding of love as concatenation, particularly concerning the relationship between sensation and judgment. A causal correlation model is defended, ensuring that the affective and cognitive components are linked. By offering a precise and scalable ontological account, this work lays the foundation for future interdisciplinary applications, making love a subject of formal inquiry in ontology engineering, artificial intelligence, and the sciences. 

---
# Eigenspectrum Analysis of Neural Networks without Aspect Ratio Bias 

**Authors**: Yuanzhe Hu, Kinshuk Goel, Vlad Killiakov, Yaoqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06280)  

**Abstract**: Diagnosing deep neural networks (DNNs) through the eigenspectrum of weight matrices has been an active area of research in recent years. At a high level, eigenspectrum analysis of DNNs involves measuring the heavytailness of the empirical spectral densities (ESD) of weight matrices. It provides insight into how well a model is trained and can guide decisions on assigning better layer-wise training hyperparameters. In this paper, we address a challenge associated with such eigenspectrum methods: the impact of the aspect ratio of weight matrices on estimated heavytailness metrics. We demonstrate that matrices of varying sizes (and aspect ratios) introduce a non-negligible bias in estimating heavytailness metrics, leading to inaccurate model diagnosis and layer-wise hyperparameter assignment. To overcome this challenge, we propose FARMS (Fixed-Aspect-Ratio Matrix Subsampling), a method that normalizes the weight matrices by subsampling submatrices with a fixed aspect ratio. Instead of measuring the heavytailness of the original ESD, we measure the average ESD of these subsampled submatrices. We show that measuring the heavytailness of these submatrices with the fixed aspect ratio can effectively mitigate the aspect ratio bias. We validate our approach across various optimization techniques and application domains that involve eigenspectrum analysis of weights, including image classification in computer vision (CV) models, scientific machine learning (SciML) model training, and large language model (LLM) pruning. Our results show that despite its simplicity, FARMS uniformly improves the accuracy of eigenspectrum analysis while enabling more effective layer-wise hyperparameter assignment in these application domains. In one of the LLM pruning experiments, FARMS reduces the perplexity of the LLaMA-7B model by 17.3% when compared with the state-of-the-art method. 

---
# Distillation Robustifies Unlearning 

**Authors**: Bruce W. Lee, Addie Foote, Alex Infanger, Leni Shor, Harish Kamath, Jacob Goldman-Wetzler, Bryce Woodworth, Alex Cloud, Alexander Matt Turner  

**Link**: [PDF](https://arxiv.org/pdf/2506.06278)  

**Abstract**: Current LLM unlearning methods are not robust: they can be reverted easily with a few steps of finetuning. This is true even for the idealized unlearning method of training to imitate an oracle model that was never exposed to unwanted information, suggesting that output-based finetuning is insufficient to achieve robust unlearning. In a similar vein, we find that training a randomly initialized student to imitate an unlearned model transfers desired behaviors while leaving undesired capabilities behind. In other words, distillation robustifies unlearning. Building on this insight, we propose Unlearn-Noise-Distill-on-Outputs (UNDO), a scalable method that distills an unlearned model into a partially noised copy of itself. UNDO introduces a tunable tradeoff between compute cost and robustness, establishing a new Pareto frontier on synthetic language and arithmetic tasks. At its strongest setting, UNDO matches the robustness of a model retrained from scratch with perfect data filtering while using only 60-80% of the compute and requiring only 0.01% of the pretraining data to be labeled. We also show that UNDO robustifies unlearning on the more realistic Weapons of Mass Destruction Proxy (WMDP) benchmark. Since distillation is widely used in practice, incorporating an unlearning step beforehand offers a convenient path to robust capability removal. 

---
# Cartridges: Lightweight and general-purpose long context representations via self-study 

**Authors**: Sabri Eyuboglu, Ryan Ehrlich, Simran Arora, Neel Guha, Dylan Zinsley, Emily Liu, Will Tennien, Atri Rudra, James Zou, Azalia Mirhoseini, Christopher Re  

**Link**: [PDF](https://arxiv.org/pdf/2506.06266)  

**Abstract**: Large language models are often used to answer queries grounded in large text corpora (e.g. codebases, legal documents, or chat histories) by placing the entire corpus in the context window and leveraging in-context learning (ICL). Although current models support contexts of 100K-1M tokens, this setup is costly to serve because the memory consumption of the KV cache scales with input length. We explore an alternative: training a smaller KV cache offline on each corpus. At inference time, we load this trained KV cache, which we call a Cartridge, and decode a response. Critically, the cost of training a Cartridge can be amortized across all the queries referencing the same corpus. However, we find that the naive approach of training the Cartridge with next-token prediction on the corpus is not competitive with ICL. Instead, we propose self-study, a training recipe in which we generate synthetic conversations about the corpus and train the Cartridge with a context-distillation objective. We find that Cartridges trained with self-study replicate the functionality of ICL, while being significantly cheaper to serve. On challenging long-context benchmarks, Cartridges trained with self-study match ICL performance while using 38.6x less memory and enabling 26.4x higher throughput. Self-study also extends the model's effective context length (e.g. from 128k to 484k tokens on MTOB) and surprisingly, leads to Cartridges that can be composed at inference time without retraining. 

---
# DesignBench: A Comprehensive Benchmark for MLLM-based Front-end Code Generation 

**Authors**: Jingyu Xiao, Ming Wang, Man Ho Lam, Yuxuan Wan, Junliang Liu, Yintong Huo, Michael R. Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06251)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities in automated front-end engineering, e.g., generating UI code from visual designs. However, existing front-end UI code generation benchmarks have the following limitations: (1) While framework-based development becomes predominant in modern front-end programming, current benchmarks fail to incorporate mainstream development frameworks. (2) Existing evaluations focus solely on the UI code generation task, whereas practical UI development involves several iterations, including refining editing, and repairing issues. (3) Current benchmarks employ unidimensional evaluation, lacking investigation into influencing factors like task difficulty, input context variations, and in-depth code-level analysis. To bridge these gaps, we introduce DesignBench, a multi-framework, multi-task evaluation benchmark for assessing MLLMs' capabilities in automated front-end engineering. DesignBench encompasses three widely-used UI frameworks (React, Vue, and Angular) alongside vanilla HTML/CSS, and evaluates on three essential front-end tasks (generation, edit, and repair) in real-world development workflows. DesignBench contains 900 webpage samples spanning over 11 topics, 9 edit types, and 6 issue categories, enabling detailed analysis of MLLM performance across multiple dimensions. Our systematic evaluation reveals critical insights into MLLMs' framework-specific limitations, task-related bottlenecks, and performance variations under different conditions, providing guidance for future research in automated front-end development. Our code and data are available at this https URL. 

---
# Visual Graph Arena: Evaluating Visual Conceptualization of Vision and Multimodal Large Language Models 

**Authors**: Zahra Babaiee, Peyman M. Kiasari, Daniela Rus, Radu Grosu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06242)  

**Abstract**: Recent advancements in multimodal large language models have driven breakthroughs in visual question answering. Yet, a critical gap persists, `conceptualization'-the ability to recognize and reason about the same concept despite variations in visual form, a basic ability of human reasoning. To address this challenge, we introduce the Visual Graph Arena (VGA), a dataset featuring six graph-based tasks designed to evaluate and improve AI systems' capacity for visual abstraction. VGA uses diverse graph layouts (e.g., Kamada-Kawai vs. planar) to test reasoning independent of visual form. Experiments with state-of-the-art vision models and multimodal LLMs reveal a striking divide: humans achieved near-perfect accuracy across tasks, while models totally failed on isomorphism detection and showed limited success in path/cycle tasks. We further identify behavioral anomalies suggesting pseudo-intelligent pattern matching rather than genuine understanding. These findings underscore fundamental limitations in current AI models for visual understanding. By isolating the challenge of representation-invariant reasoning, the VGA provides a framework to drive progress toward human-like conceptualization in AI visual models. The Visual Graph Arena is available at: \href{this https URL}{this http URL} 

---
# Towards an Explainable Comparison and Alignment of Feature Embeddings 

**Authors**: Mohammad Jalali, Bahar Dibaei Nia, Farzan Farnia  

**Link**: [PDF](https://arxiv.org/pdf/2506.06231)  

**Abstract**: While several feature embedding models have been developed in the literature, comparisons of these embeddings have largely focused on their numerical performance in classification-related downstream applications. However, an interpretable comparison of different embeddings requires identifying and analyzing mismatches between sample groups clustered within the embedding spaces. In this work, we propose the \emph{Spectral Pairwise Embedding Comparison (SPEC)} framework to compare embeddings and identify their differences in clustering a reference dataset. Our approach examines the kernel matrices derived from two embeddings and leverages the eigendecomposition of the difference kernel matrix to detect sample clusters that are captured differently by the two embeddings. We present a scalable implementation of this kernel-based approach, with computational complexity that grows linearly with the sample size. Furthermore, we introduce an optimization problem using this framework to align two embeddings, ensuring that clusters identified in one embedding are also captured in the other model. We provide numerical results demonstrating the SPEC's application to compare and align embeddings on large-scale datasets such as ImageNet and MS-COCO. The code is available at [this https URL](this http URL). 

---
# "We need to avail ourselves of GenAI to enhance knowledge distribution": Empowering Older Adults through GenAI Literacy 

**Authors**: Eunhye Grace Ko, Shaini Nanayakkara, Earl W. Huff Jr  

**Link**: [PDF](https://arxiv.org/pdf/2506.06225)  

**Abstract**: As generative AI (GenAI) becomes increasingly widespread, it is crucial to equip users, particularly vulnerable populations such as older adults (65 and older), with the knowledge to understand its benefits and potential risks. Older adults often exhibit greater reservations about adopting emerging technologies and require tailored literacy support. Using a mixed methods approach, this study examines strategies for delivering GenAI literacy to older adults through a chatbot named Litti, evaluating its impact on their AI literacy (knowledge, safety, and ethical use). The quantitative data indicated a trend toward improved AI literacy, though the results were not statistically significant. However, qualitative interviews revealed diverse levels of familiarity with generative AI and a strong desire to learn more. Findings also show that while Litti provided a positive learning experience, it did not significantly enhance participants' trust or sense of safety regarding GenAI. This exploratory case study highlights the challenges and opportunities in designing AI literacy education for the rapidly growing older adult population. 

---
# GenIR: Generative Visual Feedback for Mental Image Retrieval 

**Authors**: Diji Yang, Minghao Liu, Chung-Hsiang Lo, Yi Zhang, James Davis  

**Link**: [PDF](https://arxiv.org/pdf/2506.06220)  

**Abstract**: Vision-language models (VLMs) have shown strong performance on text-to-image retrieval benchmarks. However, bridging this success to real-world applications remains a challenge. In practice, human search behavior is rarely a one-shot action. Instead, it is often a multi-round process guided by clues in mind, that is, a mental image ranging from vague recollections to vivid mental representations of the target image. Motivated by this gap, we study the task of Mental Image Retrieval (MIR), which targets the realistic yet underexplored setting where users refine their search for a mentally envisioned image through multi-round interactions with an image search engine. Central to successful interactive retrieval is the capability of machines to provide users with clear, actionable feedback; however, existing methods rely on indirect or abstract verbal feedback, which can be ambiguous, misleading, or ineffective for users to refine the query. To overcome this, we propose GenIR, a generative multi-round retrieval paradigm leveraging diffusion-based image generation to explicitly reify the AI system's understanding at each round. These synthetic visual representations provide clear, interpretable feedback, enabling users to refine their queries intuitively and effectively. We further introduce a fully automated pipeline to generate a high-quality multi-round MIR dataset. Experimental results demonstrate that GenIR significantly outperforms existing interactive methods in the MIR scenario. This work establishes a new task with a dataset and an effective generative retrieval method, providing a foundation for future research in this direction. 

---
# Can Theoretical Physics Research Benefit from Language Agents? 

**Authors**: Sirui Lu, Zhijing Jin, Terry Jingchen Zhang, Pavel Kos, J. Ignacio Cirac, Bernhard Schölkopf  

**Link**: [PDF](https://arxiv.org/pdf/2506.06214)  

**Abstract**: Large Language Models (LLMs) are rapidly advancing across diverse domains, yet their application in theoretical physics research is not yet mature. This position paper argues that LLM agents can potentially help accelerate theoretical, computational, and applied physics when properly integrated with domain knowledge and toolbox. We analyze current LLM capabilities for physics -- from mathematical reasoning to code generation -- identifying critical gaps in physical intuition, constraint satisfaction, and reliable reasoning. We envision future physics-specialized LLMs that could handle multimodal data, propose testable hypotheses, and design experiments. Realizing this vision requires addressing fundamental challenges: ensuring physical consistency, and developing robust verification methods. We call for collaborative efforts between physics and AI communities to help advance scientific discovery in physics. 

---
# PuzzleWorld: A Benchmark for Multimodal, Open-Ended Reasoning in Puzzlehunts 

**Authors**: Hengzhi Li, Brendon Jiang, Alexander Naehu, Regan Song, Justin Zhang, Megan Tjandrasuwita, Chanakya Ekbote, Steven-Shine Chen, Adithya Balachandran, Wei Dai, Rebecca Chang, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06211)  

**Abstract**: Puzzlehunts are a genre of complex, multi-step puzzles lacking well-defined problem definitions. In contrast to conventional reasoning benchmarks consisting of tasks with clear instructions, puzzlehunts require models to discover the underlying problem structure from multimodal evidence and iterative reasoning, mirroring real-world domains such as scientific discovery, exploratory data analysis, or investigative problem-solving. Despite recent progress in foundation models, their performance on such open-ended settings remains largely untested. In this paper, we introduce PuzzleWorld, a large-scale benchmark of 667 puzzlehunt-style problems designed to assess step-by-step, open-ended, and creative multimodal reasoning. Each puzzle is annotated with the final solution, detailed reasoning traces, and cognitive skill labels, enabling holistic benchmarking and fine-grained diagnostic analysis. Most state-of-the-art models achieve only 1-2% final answer accuracy, with the best model solving only 14% of puzzles and reaching 40% stepwise accuracy. To demonstrate the value of our reasoning annotations, we show that fine-tuning a small model on reasoning traces improves stepwise reasoning from 4% to 11%, while training on final answers alone degrades performance to near zero. Our error analysis reveals that current models exhibit myopic reasoning, are bottlenecked by the limitations of language-based inference, and lack sketching capabilities crucial for visual and spatial reasoning. We release PuzzleWorld at this https URL to support future work on building more general, open-ended, and creative reasoning systems. 

---
# Building Models of Neurological Language 

**Authors**: Henry Watkins  

**Link**: [PDF](https://arxiv.org/pdf/2506.06208)  

**Abstract**: This report documents the development and evaluation of domain-specific language models for neurology. Initially focused on building a bespoke model, the project adapted to rapid advances in open-source and commercial medical LLMs, shifting toward leveraging retrieval-augmented generation (RAG) and representational models for secure, local deployment. Key contributions include the creation of neurology-specific datasets (case reports, QA sets, textbook-derived data), tools for multi-word expression extraction, and graph-based analyses of medical terminology. The project also produced scripts and Docker containers for local hosting. Performance metrics and graph community results are reported, with future possible work open for multimodal models using open-source architectures like phi-4. 

---
# Astra: Toward General-Purpose Mobile Robots via Hierarchical Multimodal Learning 

**Authors**: Sheng Chen, Peiyu He, Jiaxin Hu, Ziyang Liu, Yansheng Wang, Tao Xu, Chi Zhang, Chongchong Zhang, Chao An, Shiyu Cai, Duo Cao, Kangping Chen, Shuai Chu, Tianwei Chu, Mingdi Dan, Min Du, Weiwei Fang, Pengyou Fu, Junkai Hu, Xiaowei Jiang, Zhaodi Jiang, Fuxuan Li, Jun Li, Minghui Li, Mingyao Li, Yanchang Li, Zhibin Li, Guangming Liu, Kairui Liu, Lihao Liu, Weizhi Liu, Xiaoshun Liu, Yufei Liu, Yunfei Liu, Qiang Lu, Yuanfei Luo, Xiang Lv, Hongying Ma, Sai Ma, Lingxian Mi, Sha Sa, Hongxiang Shu, Lei Tian, Chengzhi Wang, Jiayu Wang, Kaijie Wang, Qingyi Wang, Renwen Wang, Tao Wang, Wei Wang, Xirui Wang, Chao Wei, Xuguang Wei, Zijun Xia, Zhaohao Xiao, Tingshuai Yan, Liyan Yang, Yifan Yang, Zhikai Yang, Zhong Yin, Li Yuan, Liuchun Yuan, Chi Zhang, Jinyang Zhang, Junhui Zhang, Linge Zhang, Zhenyi Zhang, Zheyu Zhang, Dongjie Zhu, Hang Li, Yangang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06205)  

**Abstract**: Modern robot navigation systems encounter difficulties in diverse and complex indoor environments. Traditional approaches rely on multiple modules with small models or rule-based systems and thus lack adaptability to new environments. To address this, we developed Astra, a comprehensive dual-model architecture, Astra-Global and Astra-Local, for mobile robot navigation. Astra-Global, a multimodal LLM, processes vision and language inputs to perform self and goal localization using a hybrid topological-semantic graph as the global map, and outperforms traditional visual place recognition methods. Astra-Local, a multitask network, handles local path planning and odometry estimation. Its 4D spatial-temporal encoder, trained through self-supervised learning, generates robust 4D features for downstream tasks. The planning head utilizes flow matching and a novel masked ESDF loss to minimize collision risks for generating local trajectories, and the odometry head integrates multi-sensor inputs via a transformer encoder to predict the relative pose of the robot. Deployed on real in-house mobile robots, Astra achieves high end-to-end mission success rate across diverse indoor environments. 

---
# MLOps with Microservices: A Case Study on the Maritime Domain 

**Authors**: Renato Cordeiro Ferreira, Rowanne Trapmann, Willem-Jan van den Heuvel  

**Link**: [PDF](https://arxiv.org/pdf/2506.06202)  

**Abstract**: This case study describes challenges and lessons learned on building Ocean Guard: a Machine Learning-Enabled System (MLES) for anomaly detection in the maritime domain. First, the paper presents the system's specification, and architecture. Ocean Guard was designed with a microservices' architecture to enable multiple teams to work on the project in parallel. Then, the paper discusses how the developers adapted contract-based design to MLOps for achieving that goal. As a MLES, Ocean Guard employs code, model, and data contracts to establish guidelines between its services. This case study hopes to inspire software engineers, machine learning engineers, and data scientists to leverage similar approaches for their systems. 

---
# semantic-features: A User-Friendly Tool for Studying Contextual Word Embeddings in Interpretable Semantic Spaces 

**Authors**: Jwalanthi Ranganathan, Rohan Jha, Kanishka Misra, Kyle Mahowald  

**Link**: [PDF](https://arxiv.org/pdf/2506.06169)  

**Abstract**: We introduce semantic-features, an extensible, easy-to-use library based on Chronis et al. (2023) for studying contextualized word embeddings of LMs by projecting them into interpretable spaces. We apply this tool in an experiment where we measure the contextual effect of the choice of dative construction (prepositional or double object) on the semantic interpretation of utterances (Bresnan, 2007). Specifically, we test whether "London" in "I sent London the letter." is more likely to be interpreted as an animate referent (e.g., as the name of a person) than in "I sent the letter to London." To this end, we devise a dataset of 450 sentence pairs, one in each dative construction, with recipients being ambiguous with respect to person-hood vs. place-hood. By applying semantic-features, we show that the contextualized word embeddings of three masked language models show the expected sensitivities. This leaves us optimistic about the usefulness of our tool. 

---
# The Lock-in Hypothesis: Stagnation by Algorithm 

**Authors**: Tianyi Alex Qiu, Zhonghao He, Tejasveer Chugh, Max Kleiman-Weiner  

**Link**: [PDF](https://arxiv.org/pdf/2506.06166)  

**Abstract**: The training and deployment of large language models (LLMs) create a feedback loop with human users: models learn human beliefs from data, reinforce these beliefs with generated content, reabsorb the reinforced beliefs, and feed them back to users again and again. This dynamic resembles an echo chamber. We hypothesize that this feedback loop entrenches the existing values and beliefs of users, leading to a loss of diversity and potentially the lock-in of false beliefs. We formalize this hypothesis and test it empirically with agent-based LLM simulations and real-world GPT usage data. Analysis reveals sudden but sustained drops in diversity after the release of new GPT iterations, consistent with the hypothesized human-AI feedback loop. Code and data available at this https URL 

---
# (AI peers) are people learning from the same standpoint: Perception of AI characters in a Collaborative Science Investigation 

**Authors**: Eunhye Grace Ko, Soo Hyoung Joo  

**Link**: [PDF](https://arxiv.org/pdf/2506.06165)  

**Abstract**: While the complexity of 21st-century demands has promoted pedagogical approaches to foster complex competencies, a persistent gap remains between in-class learning activities and individualized learning or assessment practices. To address this, studies have explored the use of AI-generated characters in learning and assessment. One attempt is scenario-based assessment (SBA), a technique that not only measures but also fosters the development of competencies throughout the assessment process. SBA introduces simulated agents to provide an authentic social-interactional context, allowing for the assessment of competency-based constructs while mitigating the unpredictability of real-life interactions. Recent advancements in multimodal AI, such as text-to-video technology, allow these agents to be enhanced into AI-generated characters. This mixed-method study investigates how learners perceive AI characters taking the role of mentor and teammates in an SBA mirroring the context of a collaborative science investigation. Specifically, we examined the Likert scale responses of 56 high schoolers regarding trust, social presence, and effectiveness. We analyzed the relationships between these factors and their impact on the intention to adopt AI characters through PLS-SEM. Our findings indicated that learners' trust shaped their sense of social presence with the AI characters, enhancing perceived effectiveness. Qualitative analysis further highlighted factors that foster trust, such as material credibility and alignment with learning goals, as well as the pivotal role of social presence in creating a collaborative context.
This paper was accepted as an full paper for AIED 2025. 

---
# Recommender systems, stigmergy, and the tyranny of popularity 

**Authors**: Zackary Okun Dunivin, Paul E. Smaldino  

**Link**: [PDF](https://arxiv.org/pdf/2506.06162)  

**Abstract**: Scientific recommender systems, such as Google Scholar and Web of Science, are essential tools for discovery. Search algorithms that power work through stigmergy, a collective intelligence mechanism that surfaces useful paths through repeated engagement. While generally effective, this ``rich-get-richer'' dynamic results in a small number of high-profile papers that dominate visibility. This essay argues argue that these algorithm over-reliance on popularity fosters intellectual homogeneity and exacerbates structural inequities, stifling innovative and diverse perspectives critical for scientific progress. We propose an overhaul of search platforms to incorporate user-specific calibration, allowing researchers to manually adjust the weights of factors like popularity, recency, and relevance. We also advise platform developers on how word embeddings and LLMs could be implemented in ways that increase user autonomy. While our suggestions are particularly pertinent to aligning recommender systems with scientific values, these ideas are broadly applicable to information access systems in general. Designing platforms that increase user autonomy is an important step toward more robust and dynamic information 

---
# Joint-GCG: Unified Gradient-Based Poisoning Attacks on Retrieval-Augmented Generation Systems 

**Authors**: Haowei Wang, Rupeng Zhang, Junjie Wang, Mingyang Li, Yuekai Huang, Dandan Wang, Qing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06151)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems enhance Large Language Models (LLMs) by retrieving relevant documents from external corpora before generating responses. This approach significantly expands LLM capabilities by leveraging vast, up-to-date external knowledge. However, this reliance on external knowledge makes RAG systems vulnerable to corpus poisoning attacks that manipulate generated outputs via poisoned document injection. Existing poisoning attack strategies typically treat the retrieval and generation stages as disjointed, limiting their effectiveness. We propose Joint-GCG, the first framework to unify gradient-based attacks across both retriever and generator models through three innovations: (1) Cross-Vocabulary Projection for aligning embedding spaces, (2) Gradient Tokenization Alignment for synchronizing token-level gradient signals, and (3) Adaptive Weighted Fusion for dynamically balancing attacking objectives. Evaluations demonstrate that Joint-GCG achieves at most 25% and an average of 5% higher attack success rate than previous methods across multiple retrievers and generators. While optimized under a white-box assumption, the generated poisons show unprecedented transferability to unseen models. Joint-GCG's innovative unification of gradient-based attacks across retrieval and generation stages fundamentally reshapes our understanding of vulnerabilities within RAG systems. Our code is available at this https URL. 

---
# Phonetically-Augmented Discriminative Rescoring for Voice Search Error Correction 

**Authors**: Christophe Van Gysel, Maggie Wu, Lyan Verwimp, Caglar Tirkaz, Marco Bertola, Zhihong Lei, Youssef Oualil  

**Link**: [PDF](https://arxiv.org/pdf/2506.06117)  

**Abstract**: End-to-end (E2E) Automatic Speech Recognition (ASR) models are trained using paired audio-text samples that are expensive to obtain, since high-quality ground-truth data requires human annotators. Voice search applications, such as digital media players, leverage ASR to allow users to search by voice as opposed to an on-screen keyboard. However, recent or infrequent movie titles may not be sufficiently represented in the E2E ASR system's training data, and hence, may suffer poor recognition.
In this paper, we propose a phonetic correction system that consists of (a) a phonetic search based on the ASR model's output that generates phonetic alternatives that may not be considered by the E2E system, and (b) a rescorer component that combines the ASR model recognition and the phonetic alternatives, and select a final system output.
We find that our approach improves word error rate between 4.4 and 7.6% relative on benchmarks of popular movie titles over a series of competitive baselines. 

---
# Towards Lifecycle Unlearning Commitment Management: Measuring Sample-level Unlearning Completeness 

**Authors**: Cheng-Long Wang, Qi Li, Zihang Xiang, Yinzhi Cao, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06112)  

**Abstract**: Growing concerns over data privacy and security highlight the importance of machine unlearning--removing specific data influences from trained models without full retraining. Techniques like Membership Inference Attacks (MIAs) are widely used to externally assess successful unlearning. However, existing methods face two key limitations: (1) maximizing MIA effectiveness (e.g., via online attacks) requires prohibitive computational resources, often exceeding retraining costs; (2) MIAs, designed for binary inclusion tests, struggle to capture granular changes in approximate unlearning. To address these challenges, we propose the Interpolated Approximate Measurement (IAM), a framework natively designed for unlearning inference. IAM quantifies sample-level unlearning completeness by interpolating the model's generalization-fitting behavior gap on queried samples. IAM achieves strong performance in binary inclusion tests for exact unlearning and high correlation for approximate unlearning--scalable to LLMs using just one pre-trained shadow model. We theoretically analyze how IAM's scoring mechanism maintains performance efficiently. We then apply IAM to recent approximate unlearning algorithms, revealing general risks of both over-unlearning and under-unlearning, underscoring the need for stronger safeguards in approximate unlearning systems. The code is available at this https URL. 

---
# Text-to-LoRA: Instant Transformer Adaption 

**Authors**: Rujikorn Charakorn, Edoardo Cetin, Yujin Tang, Robert Tjarko Lange  

**Link**: [PDF](https://arxiv.org/pdf/2506.06105)  

**Abstract**: While Foundation Models provide a general tool for rapid content creation, they regularly require task-specific adaptation. Traditionally, this exercise involves careful curation of datasets and repeated fine-tuning of the underlying model. Fine-tuning techniques enable practitioners to adapt foundation models for many new applications but require expensive and lengthy training while being notably sensitive to hyper-parameter choices. To overcome these limitations, we introduce Text-to-LoRA (T2L), a model capable of adapting Large Language Models on the fly solely based on a natural language description of the target task. T2L is a hypernetwork trained to construct LoRAs in a single inexpensive forward pass. After training T2L on a suite of 9 pre-trained LoRA adapters (GSM8K, Arc, etc.), we show that the ad-hoc reconstructed LoRA instances match the performance of task-specific adapters across the corresponding test sets. Furthermore, T2L can compress hundreds of LoRA instances and zero-shot generalize to entirely unseen tasks. This approach provides a significant step towards democratizing the specialization of foundation models and enables language-based adaptation with minimal compute requirements. Our code is available at this https URL 

---
# Simple Yet Effective: Extracting Private Data Across Clients in Federated Fine-Tuning of Large Language Models 

**Authors**: Yingqi Hu, Zhuo Zhang, Jingyuan Zhang, Lizhen Qu, Zenglin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06060)  

**Abstract**: Federated fine-tuning of large language models (FedLLMs) presents a promising approach for achieving strong model performance while preserving data privacy in sensitive domains. However, the inherent memorization ability of LLMs makes them vulnerable to training data extraction attacks. To investigate this risk, we introduce simple yet effective extraction attack algorithms specifically designed for FedLLMs. In contrast to prior "verbatim" extraction attacks, which assume access to fragments from all training data, our approach operates under a more realistic threat model, where the attacker only has access to a single client's data and aims to extract previously unseen personally identifiable information (PII) from other clients. This requires leveraging contextual prefixes held by the attacker to generalize across clients. To evaluate the effectiveness of our approaches, we propose two rigorous metrics-coverage rate and efficiency-and extend a real-world legal dataset with PII annotations aligned with CPIS, GDPR, and CCPA standards, achieving 89.9% human-verified precision. Experimental results show that our method can extract up to 56.57% of victim-exclusive PII, with "Address," "Birthday," and "Name" being the most vulnerable categories. Our findings underscore the pressing need for robust defense strategies and contribute a new benchmark and evaluation framework for future research in privacy-preserving federated learning. 

---
# Microgrids Coalitions for Energy Market Balancing 

**Authors**: Viorica Chifu, Cristina Bianca Pop, Tudor Cioara, Ionut Anghel  

**Link**: [PDF](https://arxiv.org/pdf/2506.06058)  

**Abstract**: With the integration of renewable sources in electricity distribution networks, the need to develop intelligent mechanisms for balancing the energy market has arisen. In the absence of such mechanisms, the energy market may face imbalances that can lead to power outages, financial losses or instability at the grid level. In this context, the grouping of microgrids into optimal coalitions that can absorb energy from the market during periods of surplus or supply energy to the market during periods of is a key aspect in the efficient management of distribution networks. In this article, we propose a method that identify an optimal microgrids coalition capable of addressing the dynamics of the energy market. The proposed method models the problem of identifying the optimal coalition as an optimization problem that it solves by combining a strategy inspired by cooperative game theory with a memetic algorithm. An individual is represented as a coalition of microgrids and the evolution of population of individuals over generations is assured by recombination and mutation. The fitness function is defined as the difference between the total value generated by the coalition and a penalty applied to the coalition when the energy traded by coalition exceeds the energy available/demanded on/by the energy market. The value generated by the coalition is calculated based on the profit obtained by the collation if it sells energy on the market during periods of deficit or the savings obtained by the coalition if it buys energy on the market during periods of surplus and the costs associated with the trading process. This value is divided equitably among the coalition members, according to the Shapley value, which considers the contribution of each one to the formation of collective value. 

---
# Hey, That's My Data! Label-Only Dataset Inference in Large Language Models 

**Authors**: Chen Xiong, Zihao Wang, Rui Zhu, Tsung-Yi Ho, Pin-Yu Chen, Jingwei Xiong, Haixu Tang, Lucila Ohno-Machado  

**Link**: [PDF](https://arxiv.org/pdf/2506.06057)  

**Abstract**: Large Language Models (LLMs) have revolutionized Natural Language Processing by excelling at interpreting, reasoning about, and generating human language. However, their reliance on large-scale, often proprietary datasets poses a critical challenge: unauthorized usage of such data can lead to copyright infringement and significant financial harm. Existing dataset-inference methods typically depend on log probabilities to detect suspicious training material, yet many leading LLMs have begun withholding or obfuscating these signals. This reality underscores the pressing need for label-only approaches capable of identifying dataset membership without relying on internal model logits.
We address this gap by introducing CatShift, a label-only dataset-inference framework that capitalizes on catastrophic forgetting: the tendency of an LLM to overwrite previously learned knowledge when exposed to new data. If a suspicious dataset was previously seen by the model, fine-tuning on a portion of it triggers a pronounced post-tuning shift in the model's outputs; conversely, truly novel data elicits more modest changes. By comparing the model's output shifts for a suspicious dataset against those for a known non-member validation set, we statistically determine whether the suspicious set is likely to have been part of the model's original training corpus. Extensive experiments on both open-source and API-based LLMs validate CatShift's effectiveness in logit-inaccessible settings, offering a robust and practical solution for safeguarding proprietary data. 

---
# FPDANet: A Multi-Section Classification Model for Intelligent Screening of Fetal Ultrasound 

**Authors**: Minglang Chen, Jie He, Caixu Xu, Bocheng Liang, Shengli Li, Guannan He, Xiongjie Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.06054)  

**Abstract**: ResNet has been widely used in image classification tasks due to its ability to model the residual dependence of constant mappings for linear computation. However, the ResNet method adopts a unidirectional transfer of features and lacks an effective method to correlate contextual information, which is not effective in classifying fetal ultrasound images in the classification task, and fetal ultrasound images have problems such as low contrast, high similarity, and high noise. Therefore, we propose a bilateral multi-scale information fusion network-based FPDANet to address the above challenges. Specifically, we design the positional attention mechanism (DAN) module, which utilizes the similarity of features to establish the dependency of different spatial positional features and enhance the feature representation. In addition, we design a bilateral multi-scale (FPAN) information fusion module to capture contextual and global feature dependencies at different feature scales, thereby further improving the model representation. FPDANet classification results obtained 91.05\% and 100\% in Top-1 and Top-5 metrics, respectively, and the experimental results proved the effectiveness and robustness of FPDANet. 

---
# TRUST: Test-time Resource Utilization for Superior Trustworthiness 

**Authors**: Haripriya Harikumar, Santu Rana  

**Link**: [PDF](https://arxiv.org/pdf/2506.06048)  

**Abstract**: Standard uncertainty estimation techniques, such as dropout, often struggle to clearly distinguish reliable predictions from unreliable ones. We attribute this limitation to noisy classifier weights, which, while not impairing overall class-level predictions, render finer-level statistics less informative. To address this, we propose a novel test-time optimization method that accounts for the impact of such noise to produce more reliable confidence estimates. This score defines a monotonic subset-selection function, where population accuracy consistently increases as samples with lower scores are removed, and it demonstrates superior performance in standard risk-based metrics such as AUSE and AURC. Additionally, our method effectively identifies discrepancies between training and test distributions, reliably differentiates in-distribution from out-of-distribution samples, and elucidates key differences between CNN and ViT classifiers across various vision datasets. 

---
# HAVIR: HierArchical Vision to Image Reconstruction using CLIP-Guided Versatile Diffusion 

**Authors**: Shiyi Zhang, Dong Liang, Hairong Zheng, Yihang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.06035)  

**Abstract**: Reconstructing visual information from brain activity bridges the gap between neuroscience and computer vision. Even though progress has been made in decoding images from fMRI using generative models, a challenge remains in accurately recovering highly complex visual stimuli. This difficulty stems from their elemental density and diversity, sophisticated spatial structures, and multifaceted semantic information.
To address these challenges, we propose HAVIR that contains two adapters: (1) The AutoKL Adapter transforms fMRI voxels into a latent diffusion prior, capturing topological structures; (2) The CLIP Adapter converts the voxels to CLIP text and image embeddings, containing semantic information. These complementary representations are fused by Versatile Diffusion to generate the final reconstructed image. To extract the most essential semantic information from complex scenarios, the CLIP Adapter is trained with text captions describing the visual stimuli and their corresponding semantic images synthesized from these captions. The experimental results demonstrate that HAVIR effectively reconstructs both structural features and semantic information of visual stimuli even in complex scenarios, outperforming existing models. 

---
# End-to-End Framework for Robot Lawnmower Coverage Path Planning using Cellular Decomposition 

**Authors**: Nikunj Shah, Utsav Dey, Kenji Nishimiya  

**Link**: [PDF](https://arxiv.org/pdf/2506.06028)  

**Abstract**: Efficient Coverage Path Planning (CPP) is necessary for autonomous robotic lawnmowers to effectively navigate and maintain lawns with diverse and irregular shapes. This paper introduces a comprehensive end-to-end pipeline for CPP, designed to convert user-defined boundaries on an aerial map into optimized coverage paths seamlessly. The pipeline includes user input extraction, coordinate transformation, area decomposition and path generation using our novel AdaptiveDecompositionCPP algorithm, preview and customization through an interactive coverage path visualizer, and conversion to actionable GPS waypoints. The AdaptiveDecompositionCPP algorithm combines cellular decomposition with an adaptive merging strategy to reduce non-mowing travel thereby enhancing operational efficiency. Experimental evaluations, encompassing both simulations and real-world lawnmower tests, demonstrate the effectiveness of the framework in coverage completeness and mowing efficiency. 

---
# When to Trust Context: Self-Reflective Debates for Context Reliability 

**Authors**: Zeqi Zhou, Fang Wu, Shayan Talaei, Haokai Zhao, Cheng Meixin, Tinson Xu, Amin Saberi, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.06020)  

**Abstract**: Large language models frequently encounter conflicts between their parametric knowledge and contextual input, often resulting in factual inconsistencies or hallucinations. We propose Self-Reflective Debate for Contextual Reliability (SR-DCR), a lightweight framework that integrates token-level self-confidence with an asymmetric multi-agent debate to adjudicate such conflicts. A critic, deprived of context, challenges a defender who argues from the given passage; a judge model evaluates the debate and determines the context's reliability. The final answer is selected by combining the verdict with model confidence. Experiments on the ClashEval benchmark demonstrate that SR-DCR consistently enhances robustness to misleading context while maintaining accuracy on trustworthy inputs, outperforming both classical debate and confidence-only baselines with minimal computational overhead. The code is available at this https URL. 

---
# Optimization-Free Universal Watermark Forgery with Regenerative Diffusion Models 

**Authors**: Chaoyi Zhu, Zaitang Li, Renyi Yang, Robert Birke, Pin-Yu Chen, Tsung-Yi Ho, Lydia Y. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.06018)  

**Abstract**: Watermarking becomes one of the pivotal solutions to trace and verify the origin of synthetic images generated by artificial intelligence models, but it is not free of risks. Recent studies demonstrate the capability to forge watermarks from a target image onto cover images via adversarial optimization without knowledge of the target generative model and watermark schemes. In this paper, we uncover a greater risk of an optimization-free and universal watermark forgery that harnesses existing regenerative diffusion models. Our proposed forgery attack, PnP (Plug-and-Plant), seamlessly extracts and integrates the target watermark via regenerating the image, without needing any additional optimization routine. It allows for universal watermark forgery that works independently of the target image's origin or the watermarking model used. We explore the watermarked latent extracted from the target image and visual-textual context of cover images as priors to guide sampling of the regenerative process. Extensive evaluation on 24 scenarios of model-data-watermark combinations demonstrates that PnP can successfully forge the watermark (up to 100% detectability and user attribution), and maintain the best visual perception. By bypassing model retraining and enabling adaptability to any image, our approach significantly broadens the scope of forgery attacks, presenting a greater challenge to the security of current watermarking techniques for diffusion models and the authority of watermarking schemes in synthetic data generation and governance. 

---
# Unlocking Recursive Thinking of LLMs: Alignment via Refinement 

**Authors**: Haoke Zhang, Xiaobo Liang, Cunxiang Wang, Juntao Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06009)  

**Abstract**: The OpenAI o1-series models have demonstrated that leveraging long-form Chain of Thought (CoT) can substantially enhance performance. However, the recursive thinking capabilities of Large Language Models (LLMs) remain limited, particularly in the absence of expert-curated data for distillation. In this paper, we propose \textbf{AvR}: \textbf{Alignment via Refinement}, a novel method aimed at unlocking the potential of LLMs for recursive reasoning through long-form CoT. AvR introduces a refinement process that integrates criticism and improvement actions, guided by differentiable learning techniques to optimize \textbf{refinement-aware rewards}. As a result, the synthesized multi-round data can be organized as a long refinement thought, further enabling test-time scaling. Experimental results show that AvR significantly outperforms conventional preference optimization methods. Notably, with only 3k synthetic samples, our method boosts the performance of the LLaMA-3-8B-Instruct model by over 20\% in win rate on AlpacaEval 2.0. Our code is available at Github (this https URL). 

---
# Token Signature: Predicting Chain-of-Thought Gains with Token Decoding Feature in Large Language Models 

**Authors**: Peijie Liu, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.06008)  

**Abstract**: Chain-of-Thought (CoT) technique has proven effective in improving the performance of large language models (LLMs) on complex reasoning tasks. However, the performance gains are inconsistent across different tasks, and the underlying mechanism remains a long-standing research question. In this work, we make a preliminary observation that the monotonicity of token probability distributions may be correlated with the gains achieved through CoT reasoning. Leveraging this insight, we propose two indicators based on the token probability distribution to assess CoT effectiveness across different tasks. By combining instance-level indicators with logistic regression model, we introduce Dynamic CoT, a method that dynamically select between CoT and direct answer. Furthermore, we extend Dynamic CoT to closed-source models by transferring decision strategies learned from open-source models. Our indicators for assessing CoT effectiveness achieve an accuracy of 89.2\%, and Dynamic CoT reduces token consumption by more than 35\% while maintaining high accuracy. Overall, our work offers a novel perspective on the underlying mechanisms of CoT reasoning and provides a framework for its more efficient deployment. 

---
# Enhancing Orthopox Image Classification Using Hybrid Machine Learning and Deep Learning Models 

**Authors**: Alejandro Puente-Castro, Enrique Fernandez-Blanco, Daniel Rivero, Andres Molares-Ulloa  

**Link**: [PDF](https://arxiv.org/pdf/2506.06007)  

**Abstract**: Orthopoxvirus infections must be accurately classified from medical pictures for an easy and early diagnosis and epidemic prevention. The necessity for automated and scalable solutions is highlighted by the fact that traditional diagnostic techniques can be time-consuming and require expert interpretation and there are few and biased data sets of the different types of Orthopox. In order to improve classification performance and lower computational costs, a hybrid strategy is put forth in this paper that uses Machine Learning models combined with pretrained Deep Learning models to extract deep feature representations without the need for augmented data. The findings show that this feature extraction method, when paired with other methods in the state-of-the-art, produces excellent classification outcomes while preserving training and inference efficiency. The proposed approach demonstrates strong generalization and robustness across multiple evaluation settings, offering a scalable and interpretable solution for real-world clinical deployment. 

---
# Bootstrapping World Models from Dynamics Models in Multimodal Foundation Models 

**Authors**: Yifu Qiu, Yftah Ziser, Anna Korhonen, Shay B. Cohen, Edoardo M. Ponti  

**Link**: [PDF](https://arxiv.org/pdf/2506.06006)  

**Abstract**: To what extent do vision-and-language foundation models possess a realistic world model (observation $\times$ action $\rightarrow$ observation) and a dynamics model (observation $\times$ observation $\rightarrow$ action), when actions are expressed through language? While open-source foundation models struggle with both, we find that fine-tuning them to acquire a dynamics model through supervision is significantly easier than acquiring a world model. In turn, dynamics models can be used to bootstrap world models through two main strategies: 1) weakly supervised learning from synthetic data and 2) inference time verification. Firstly, the dynamics model can annotate actions for unlabelled pairs of video frame observations to expand the training data. We further propose a new objective, where image tokens in observation pairs are weighted by their importance, as predicted by a recognition model. Secondly, the dynamics models can assign rewards to multiple samples of the world model to score them, effectively guiding search at inference time. We evaluate the world models resulting from both strategies through the task of action-centric image editing on Aurora-Bench. Our best model achieves a performance competitive with state-of-the-art image editing models, improving on them by a margin of $15\%$ on real-world subsets according to GPT4o-as-judge, and achieving the best average human evaluation across all subsets of Aurora-Bench. 

---
# Leveraging Generative AI for Enhancing Automated Assessment in Programming Education Contests 

**Authors**: Stefan Dascalescu, Adrian Marius Dumitran, Mihai Alexandru Vasiluta  

**Link**: [PDF](https://arxiv.org/pdf/2506.05990)  

**Abstract**: Competitive programming contests play a crucial role in cultivating computational thinking and algorithmic skills among learners. However, generating comprehensive test cases to effectively assess programming solutions remains resource-intensive and challenging for educators. This paper introduces an innovative NLP-driven method leveraging generative AI (large language models) to automate the creation of high-quality test cases for competitive programming assessments. We extensively evaluated our approach on diverse datasets, including 25 years of Romanian Informatics Olympiad (OJI) data for 5th graders, recent competitions hosted on the this http URL platform, and the International Informatics Olympiad in Teams (IIOT). Our results demonstrate that AI-generated test cases substantially enhanced assessments, notably identifying previously undetected errors in 67% of the OJI 5th grade programming problems. These improvements underscore the complementary educational value of our technique in formative assessment contexts. By openly sharing our prompts, translated datasets, and methodologies, we offer practical NLP-based tools that educators and contest organizers can readily integrate to enhance assessment quality, reduce workload, and deepen insights into learner performance. 

---
# Audio-Aware Large Language Models as Judges for Speaking Styles 

**Authors**: Cheng-Han Chiang, Xiaofei Wang, Chung-Ching Lin, Kevin Lin, Linjie Li, Radu Kopetz, Yao Qian, Zhendong Wang, Zhengyuan Yang, Hung-yi Lee, Lijuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05984)  

**Abstract**: Audio-aware large language models (ALLMs) can understand the textual and non-textual information in the audio input. In this paper, we explore using ALLMs as an automatic judge to assess the speaking styles of speeches. We use ALLM judges to evaluate the speeches generated by SLMs on two tasks: voice style instruction following and role-playing. The speaking style we consider includes emotion, volume, speaking pace, word emphasis, pitch control, and non-verbal elements. We use four spoken language models (SLMs) to complete the two tasks and use humans and ALLMs to judge the SLMs' responses. We compare two ALLM judges, GPT-4o-audio and Gemini-2.5-pro, with human evaluation results and show that the agreement between Gemini and human judges is comparable to the agreement between human evaluators. These promising results show that ALLMs can be used as a judge to evaluate SLMs. Our results also reveal that current SLMs, even GPT-4o-audio, still have room for improvement in controlling the speaking style and generating natural dialogues. 

---
# AMPED: Adaptive Multi-objective Projection for balancing Exploration and skill Diversification 

**Authors**: Geonwoo Cho, Jaemoon Lee, Jaegyun Im, Subi Lee, Jihwan Lee, Sundong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.05980)  

**Abstract**: Skill-based reinforcement learning (SBRL) enables rapid adaptation in environments with sparse rewards by pretraining a skill-conditioned policy. Effective skill learning requires jointly maximizing both exploration and skill diversity. However, existing methods often face challenges in simultaneously optimizing for these two conflicting objectives. In this work, we propose a new method, Adaptive Multi-objective Projection for balancing Exploration and skill Diversification (AMPED), which explicitly addresses both exploration and skill diversification. We begin by conducting extensive ablation studies to identify and define a set of objectives that effectively capture the aspects of exploration and skill diversity, respectively. During the skill pretraining phase, AMPED introduces a gradient surgery technique to balance the objectives of exploration and skill diversity, mitigating conflicts and reducing reliance on heuristic tuning. In the subsequent fine-tuning phase, AMPED incorporates a skill selector module that dynamically selects suitable skills for downstream tasks, based on task-specific performance signals. Our approach achieves performance that surpasses SBRL baselines across various benchmarks. These results highlight the importance of explicitly harmonizing exploration and diversity and demonstrate the effectiveness of AMPED in enabling robust and generalizable skill learning. Project Page: this https URL 

---
# On Measuring Long-Range Interactions in Graph Neural Networks 

**Authors**: Jacob Bamberger, Benjamin Gutteridge, Scott le Roux, Michael M. Bronstein, Xiaowen Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.05971)  

**Abstract**: Long-range graph tasks -- those dependent on interactions between distant nodes -- are an open problem in graph neural network research. Real-world benchmark tasks, especially the Long Range Graph Benchmark, have become popular for validating the long-range capability of proposed architectures. However, this is an empirical approach that lacks both robustness and theoretical underpinning; a more principled characterization of the long-range problem is required. To bridge this gap, we formalize long-range interactions in graph tasks, introduce a range measure for operators on graphs, and validate it with synthetic experiments. We then leverage our measure to examine commonly used tasks and architectures, and discuss to what extent they are, in fact, long-range. We believe our work advances efforts to define and address the long-range problem on graphs, and that our range measure will aid evaluation of new datasets and architectures. 

---
# Let's Put Ourselves in Sally's Shoes: Shoes-of-Others Prefixing Improves Theory of Mind in Large Language Models 

**Authors**: Kazutoshi Shinoda, Nobukatsu Hojo, Kyosuke Nishida, Yoshihiro Yamazaki, Keita Suzuki, Hiroaki Sugiyama, Kuniko Saito  

**Link**: [PDF](https://arxiv.org/pdf/2506.05970)  

**Abstract**: Recent studies have shown that Theory of Mind (ToM) in large language models (LLMs) has not reached human-level performance yet. Since fine-tuning LLMs on ToM datasets often degrades their generalization, several inference-time methods have been proposed to enhance ToM in LLMs. However, existing inference-time methods for ToM are specialized for inferring beliefs from contexts involving changes in the world state. In this study, we present a new inference-time method for ToM, Shoes-of-Others (SoO) prefixing, which makes fewer assumptions about contexts and is applicable to broader scenarios. SoO prefixing simply specifies the beginning of LLM outputs with ``Let's put ourselves in A's shoes.'', where A denotes the target character's name. We evaluate SoO prefixing on two benchmarks that assess ToM in conversational and narrative contexts without changes in the world state and find that it consistently improves ToM across five categories of mental states. Our analysis suggests that SoO prefixing elicits faithful thoughts, thereby improving the ToM performance. 

---
# Gradual Transition from Bellman Optimality Operator to Bellman Operator in Online Reinforcement Learning 

**Authors**: Motoki Omura, Kazuki Ota, Takayuki Osa, Yusuke Mukuta, Tatsuya Harada  

**Link**: [PDF](https://arxiv.org/pdf/2506.05968)  

**Abstract**: For continuous action spaces, actor-critic methods are widely used in online reinforcement learning (RL). However, unlike RL algorithms for discrete actions, which generally model the optimal value function using the Bellman optimality operator, RL algorithms for continuous actions typically model Q-values for the current policy using the Bellman operator. These algorithms for continuous actions rely exclusively on policy updates for improvement, which often results in low sample efficiency. This study examines the effectiveness of incorporating the Bellman optimality operator into actor-critic frameworks. Experiments in a simple environment show that modeling optimal values accelerates learning but leads to overestimation bias. To address this, we propose an annealing approach that gradually transitions from the Bellman optimality operator to the Bellman operator, thereby accelerating learning while mitigating bias. Our method, combined with TD3 and SAC, significantly outperforms existing approaches across various locomotion and manipulation tasks, demonstrating improved performance and robustness to hyperparameters related to optimality. 

---
# MOGO: Residual Quantized Hierarchical Causal Transformer for High-Quality and Real-Time 3D Human Motion Generation 

**Authors**: Dongjie Fu, Tengjiao Sun, Pengcheng Fang, Xiaohao Cai, Hansung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.05952)  

**Abstract**: Recent advances in transformer-based text-to-motion generation have led to impressive progress in synthesizing high-quality human motion. Nevertheless, jointly achieving high fidelity, streaming capability, real-time responsiveness, and scalability remains a fundamental challenge. In this paper, we propose MOGO (Motion Generation with One-pass), a novel autoregressive framework tailored for efficient and real-time 3D motion generation. MOGO comprises two key components: (1) MoSA-VQ, a motion scale-adaptive residual vector quantization module that hierarchically discretizes motion sequences with learnable scaling to produce compact yet expressive representations; and (2) RQHC-Transformer, a residual quantized hierarchical causal transformer that generates multi-layer motion tokens in a single forward pass, significantly reducing inference latency. To enhance semantic fidelity, we further introduce a text condition alignment mechanism that improves motion decoding under textual control. Extensive experiments on benchmark datasets including HumanML3D, KIT-ML, and CMP demonstrate that MOGO achieves competitive or superior generation quality compared to state-of-the-art transformer-based methods, while offering substantial improvements in real-time performance, streaming generation, and generalization under zero-shot settings. 

---
# IntentionESC: An Intention-Centered Framework for Enhancing Emotional Support in Dialogue Systems 

**Authors**: Xinjie Zhang, Wenxuan Wang, Qin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2506.05947)  

**Abstract**: In emotional support conversations, unclear intentions can lead supporters to employ inappropriate strategies, inadvertently imposing their expectations or solutions on the seeker. Clearly defined intentions are essential for guiding both the supporter's motivations and the overall emotional support process. In this paper, we propose the Intention-centered Emotional Support Conversation (IntentionESC) framework, which defines the possible intentions of supporters in emotional support conversations, identifies key emotional state aspects for inferring these intentions, and maps them to appropriate support strategies. While Large Language Models (LLMs) excel in text generating, they fundamentally operate as probabilistic models trained on extensive datasets, lacking a true understanding of human thought processes and intentions. To address this limitation, we introduce the Intention Centric Chain-of-Thought (ICECoT) mechanism. ICECoT enables LLMs to mimic human reasoning by analyzing emotional states, inferring intentions, and selecting suitable support strategies, thereby generating more effective emotional support responses. To train the model with ICECoT and integrate expert knowledge, we design an automated annotation pipeline that produces high-quality training data. Furthermore, we develop a comprehensive evaluation scheme to assess emotional support efficacy and conduct extensive experiments to validate our framework. Our data and code are available at this https URL. 

---
# Comparative Analysis of Modern Machine Learning Models for Retail Sales Forecasting 

**Authors**: Luka Hobor, Mario Brcic, Lidija Polutnik, Ante Kapetanovic  

**Link**: [PDF](https://arxiv.org/pdf/2506.05941)  

**Abstract**: Accurate forecasting is key for all business planning. When estimated sales are too high, brick-and-mortar retailers may incur higher costs due to unsold inventories, higher labor and storage space costs, etc. On the other hand, when forecasts underestimate the level of sales, firms experience lost sales, shortages, and impact on the reputation of the retailer in their relevant market. Accurate forecasting presents a competitive advantage for companies. It facilitates the achievement of revenue and profit goals and execution of pricing strategy and tactics. In this study, we provide an exhaustive assessment of the forecasting models applied to a high-resolution brick-and-mortar retail dataset. Our forecasting framework addresses the problems found in retail environments, including intermittent demand, missing values, and frequent product turnover. We compare tree-based ensembles (such as XGBoost and LightGBM) and state-of-the-art neural network architectures (including N-BEATS, NHITS, and the Temporal Fusion Transformer) across various experimental settings. Our results show that localized modeling strategies especially those using tree-based models on individual groups with non-imputed data, consistently deliver superior forecasting accuracy and computational efficiency. In contrast, neural models benefit from advanced imputation methods, yet still fall short in handling the irregularities typical of physical retail data. These results further practical understanding for model selection in retail environment and highlight the significance of data preprocessing to improve forecast performance. 

---
# Quantifying Adversarial Uncertainty in Evidential Deep Learning using Conflict Resolution 

**Authors**: Charmaine Barker, Daniel Bethell, Simos Gerasimou  

**Link**: [PDF](https://arxiv.org/pdf/2506.05937)  

**Abstract**: Reliability of deep learning models is critical for deployment in high-stakes applications, where out-of-distribution or adversarial inputs may lead to detrimental outcomes. Evidential Deep Learning, an efficient paradigm for uncertainty quantification, models predictions as Dirichlet distributions of a single forward pass. However, EDL is particularly vulnerable to adversarially perturbed inputs, making overconfident errors. Conflict-aware Evidential Deep Learning (C-EDL) is a lightweight post-hoc uncertainty quantification approach that mitigates these issues, enhancing adversarial and OOD robustness without retraining. C-EDL generates diverse, task-preserving transformations per input and quantifies representational disagreement to calibrate uncertainty estimates when needed. C-EDL's conflict-aware prediction adjustment improves detection of OOD and adversarial inputs, maintaining high in-distribution accuracy and low computational overhead. Our experimental evaluation shows that C-EDL significantly outperforms state-of-the-art EDL variants and competitive baselines, achieving substantial reductions in coverage for OOD data (up to 55%) and adversarial data (up to 90%), across a range of datasets, attack types, and uncertainty metrics. 

---
# DynamicMind: A Tri-Mode Thinking System for Large Language Models 

**Authors**: Wei Li, Yanbin Wei, Qiushi Huang, Jiangyue Yan, Yang Chen, James T. Kwok, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05936)  

**Abstract**: Modern large language models (LLMs) often struggle to dynamically adapt their reasoning depth to varying task complexities, leading to suboptimal performance or inefficient resource utilization. To address this, we introduce DynamicMind, a novel tri-mode thinking system. DynamicMind empowers LLMs to autonomously select between Fast, Normal, and Slow thinking modes for zero-shot question answering (ZSQA) tasks through cognitive-inspired prompt engineering. Our framework's core innovations include: (1) expanding the established dual-process framework of fast and slow thinking into a tri-mode thinking system involving a normal thinking mode to preserve the intrinsic capabilities of LLM; (2) proposing the Thinking Density metric, which aligns computational resource allocation with problem complexity; and (3) developing the Thinking Mode Capacity (TMC) dataset and a lightweight Mind Router to predict the optimal thinking mode. Extensive experiments across diverse mathematical, commonsense, and scientific QA benchmarks demonstrate that DynamicMind achieves superior ZSQA capabilities while establishing an effective trade-off between performance and computational efficiency. 

---
# FADE: Frequency-Aware Diffusion Model Factorization for Video Editing 

**Authors**: Yixuan Zhu, Haolin Wang, Shilin Ma, Wenliang Zhao, Yansong Tang, Lei Chen, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.05934)  

**Abstract**: Recent advancements in diffusion frameworks have significantly enhanced video editing, achieving high fidelity and strong alignment with textual prompts. However, conventional approaches using image diffusion models fall short in handling video dynamics, particularly for challenging temporal edits like motion adjustments. While current video diffusion models produce high-quality results, adapting them for efficient editing remains difficult due to the heavy computational demands that prevent the direct application of previous image editing techniques. To overcome these limitations, we introduce FADE, a training-free yet highly effective video editing approach that fully leverages the inherent priors from pre-trained video diffusion models via frequency-aware factorization. Rather than simply using these models, we first analyze the attention patterns within the video model to reveal how video priors are distributed across different components. Building on these insights, we propose a factorization strategy to optimize each component's specialized role. Furthermore, we devise spectrum-guided modulation to refine the sampling trajectory with frequency domain cues, preventing information leakage and supporting efficient, versatile edits while preserving the basic spatial and temporal structure. Extensive experiments on real-world videos demonstrate that our method consistently delivers high-quality, realistic and temporally coherent editing results both qualitatively and quantitatively. Code is available at this https URL . 

---
# MoA: Heterogeneous Mixture of Adapters for Parameter-Efficient Fine-Tuning of Large Language Models 

**Authors**: Jie Cao, Tianwei Lin, Hongyang He, Rolan Yan, Wenqiao Zhang, Juncheng Li, Dongping Zhang, Siliang Tang, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05928)  

**Abstract**: Recent studies integrate Low-Rank Adaptation (LoRA) and Mixture-of-Experts (MoE) to further enhance the performance of parameter-efficient fine-tuning (PEFT) methods in Large Language Model (LLM) applications. Existing methods employ \emph{homogeneous} MoE-LoRA architectures composed of LoRA experts with either similar or identical structures and capacities. However, these approaches often suffer from representation collapse and expert load imbalance, which negatively impact the potential of LLMs. To address these challenges, we propose a \emph{heterogeneous} \textbf{Mixture-of-Adapters (MoA)} approach. This method dynamically integrates PEFT adapter experts with diverse structures, leveraging their complementary representational capabilities to foster expert specialization, thereby enhancing the effective transfer of pre-trained knowledge to downstream tasks. MoA supports two variants: \textbf{(i)} \textit{Soft MoA} achieves fine-grained integration by performing a weighted fusion of all expert outputs; \textbf{(ii)} \textit{Sparse MoA} activates adapter experts sparsely based on their contribution, achieving this with negligible performance degradation. Experimental results demonstrate that heterogeneous MoA outperforms homogeneous MoE-LoRA methods in both performance and parameter efficiency. Our project is available at this https URL. 

---
# LengClaro2023: A Dataset of Administrative Texts in Spanish with Plain Language adaptations 

**Authors**: Belén Agüera-Marco, Itziar Gonzalez-Dios  

**Link**: [PDF](https://arxiv.org/pdf/2506.05927)  

**Abstract**: In this work, we present LengClaro2023, a dataset of legal-administrative texts in Spanish. Based on the most frequently used procedures from the Spanish Social Security website, we have created for each text two simplified equivalents. The first version follows the recommendations provided by arText claro. The second version incorporates additional recommendations from plain language guidelines to explore further potential improvements in the system. The linguistic resource created in this work can be used for evaluating automatic text simplification (ATS) systems in Spanish. 

---
# Small Models, Big Support: A Local LLM Framework for Teacher-Centric Content Creation and Assessment using RAG and CAG 

**Authors**: Zarreen Reza, Alexander Mazur, Michael T. Dugdale, Robin Ray-Chaudhuri  

**Link**: [PDF](https://arxiv.org/pdf/2506.05925)  

**Abstract**: While Large Language Models (LLMs) are increasingly utilized as student-facing educational aids, their potential to directly support educators, particularly through locally deployable and customizable open-source solutions, remains significantly underexplored. Many existing educational solutions rely on cloud-based infrastructure or proprietary tools, which are costly and may raise privacy concerns. Regulated industries with limited budgets require affordable, self-hosted solutions. We introduce an end-to-end, open-source framework leveraging small (3B-7B parameters), locally deployed LLMs for customized teaching material generation and assessment. Our system uniquely incorporates an interactive loop crucial for effective small-model refinement, and an auxiliary LLM verifier to mitigate jailbreaking risks, enhancing output reliability and safety. Utilizing Retrieval and Context Augmented Generation (RAG/CAG), it produces factually accurate, customized pedagogically-styled content. Deployed on-premises for data privacy and validated through an evaluation pipeline and a college physics pilot, our findings show that carefully engineered small LLM systems can offer robust, affordable, practical, and safe educator support, achieving utility comparable to larger models for targeted tasks. 

---
# Rethinking Semi-supervised Segmentation Beyond Accuracy: Reliability and Robustness 

**Authors**: Steven Landgraf, Markus Hillemann, Markus Ulrich  

**Link**: [PDF](https://arxiv.org/pdf/2506.05917)  

**Abstract**: Semantic segmentation is critical for scene understanding but demands costly pixel-wise annotations, attracting increasing attention to semi-supervised approaches to leverage abundant unlabeled data. While semi-supervised segmentation is often promoted as a path toward scalable, real-world deployment, it is astonishing that current evaluation protocols exclusively focus on segmentation accuracy, entirely overlooking reliability and robustness. These qualities, which ensure consistent performance under diverse conditions (robustness) and well-calibrated model confidences as well as meaningful uncertainties (reliability), are essential for safety-critical applications like autonomous driving, where models must handle unpredictable environments and avoid sudden failures at all costs. To address this gap, we introduce the Reliable Segmentation Score (RSS), a novel metric that combines predictive accuracy, calibration, and uncertainty quality measures via a harmonic mean. RSS penalizes deficiencies in any of its components, providing an easy and intuitive way of holistically judging segmentation models. Comprehensive evaluations of UniMatchV2 against its predecessor and a supervised baseline show that semi-supervised methods often trade reliability for accuracy. While out-of-domain evaluations demonstrate UniMatchV2's robustness, they further expose persistent reliability shortcomings. We advocate for a shift in evaluation protocols toward more holistic metrics like RSS to better align semi-supervised learning research with real-world deployment needs. 

---
# Route-and-Reason: Scaling Large Language Model Reasoning with Reinforced Model Router 

**Authors**: Chenyang Shao, Xinyang Liu, Yutang Lin, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.05901)  

**Abstract**: Multi-step reasoning has proven essential for enhancing the problem-solving capabilities of Large Language Models (LLMs) by decomposing complex tasks into intermediate steps, either explicitly or implicitly. Extending the reasoning chain at test time through deeper thought processes or broader exploration, can furthur improve performance, but often incurs substantial costs due to the explosion in token usage. Yet, many reasoning steps are relatively simple and can be handled by more efficient smaller-scale language models (SLMs). This motivates hybrid approaches that allocate subtasks across models of varying capacities. However, realizing such collaboration requires accurate task decomposition and difficulty-aware subtask allocation, which is challenging. To address this, we propose R2-Reasoner, a novel framework that enables collaborative reasoning across heterogeneous LLMs by dynamically routing sub-tasks based on estimated complexity. At the core of our framework is a Reinforced Model Router, composed of a task decomposer and a subtask allocator. The task decomposer segments complex input queries into logically ordered subtasks, while the subtask allocator assigns each subtask to the most appropriate model, ranging from lightweight SLMs to powerful LLMs, balancing accuracy and efficiency. To train this router, we introduce a staged pipeline that combines supervised fine-tuning on task-specific datasets with Group Relative Policy Optimization algorithm, enabling self-supervised refinement through iterative reinforcement learning. Extensive experiments across four challenging benchmarks demonstrate that R2-Reasoner reduces API costs by 86.85% while maintaining or surpassing baseline accuracy. Our framework paves the way for more cost-effective and adaptive LLM reasoning. The code is open-source at this https URL . 

---
# WhisQ: Cross-Modal Representation Learning for Text-to-Music MOS Prediction 

**Authors**: Jakaria Islam Emon, Kazi Tamanna Alam, Md. Abu Salek  

**Link**: [PDF](https://arxiv.org/pdf/2506.05899)  

**Abstract**: Mean Opinion Score (MOS) prediction for text to music systems requires evaluating both overall musical quality and text prompt alignment. This paper introduces WhisQ, a multimodal architecture that addresses this dual-assessment challenge through sequence level co-attention and optimal transport regularization. WhisQ employs the Whisper Base pretrained model for temporal audio encoding and Qwen 3, a 0.6B Small Language Model (SLM), for text encoding, with both maintaining sequence structure for fine grained cross-modal modeling. The architecture features specialized prediction pathways: OMQ is predicted from pooled audio embeddings, while TA leverages bidirectional sequence co-attention between audio and text. Sinkhorn optimal transport loss further enforce semantic alignment in the shared embedding space. On the MusicEval Track-1 dataset, WhisQ achieves substantial improvements over the baseline: 7% improvement in Spearman correlation for OMQ and 14% for TA. Ablation studies reveal that optimal transport regularization provides the largest performance gain (10% SRCC improvement), demonstrating the importance of explicit cross-modal alignment for text-to-music evaluation. 

---
# Object Navigation with Structure-Semantic Reasoning-Based Multi-level Map and Multimodal Decision-Making LLM 

**Authors**: Chongshang Yan, Jiaxuan He, Delun Li, Yi Yang, Wenjie Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.05896)  

**Abstract**: The zero-shot object navigation (ZSON) in unknown open-ended environments coupled with semantically novel target often suffers from the significant decline in performance due to the neglect of high-dimensional implicit scene information and the long-range target searching task. To address this, we proposed an active object navigation framework with Environmental Attributes Map (EAM) and MLLM Hierarchical Reasoning module (MHR) to improve its success rate and efficiency. EAM is constructed by reasoning observed environments with SBERT and predicting unobserved ones with Diffusion, utilizing human space regularities that underlie object-room correlations and area adjacencies. MHR is inspired by EAM to perform frontier exploration decision-making, avoiding the circuitous trajectories in long-range scenarios to improve path efficiency. Experimental results demonstrate that the EAM module achieves 64.5\% scene mapping accuracy on MP3D dataset, while the navigation task attains SPLs of 28.4\% and 26.3\% on HM3D and MP3D benchmarks respectively - representing absolute improvements of 21.4\% and 46.0\% over baseline methods. 

---
# HMVLM: Multistage Reasoning-Enhanced Vision-Language Model for Long-Tailed Driving Scenarios 

**Authors**: Daming Wang, Yuhao Song, Zijian He, Kangliang Chen, Xing Pan, Lu Deng, Weihao Gu  

**Link**: [PDF](https://arxiv.org/pdf/2506.05883)  

**Abstract**: We present HaoMo Vision-Language Model (HMVLM), an end-to-end driving framework that implements the slow branch of a cognitively inspired fast-slow architecture. A fast controller outputs low-level steering, throttle, and brake commands, while a slow planner-a large vision-language model-generates high-level intents such as "yield to pedestrian" or "merge after the truck" without compromising latency. HMVLM introduces three upgrades: (1) selective five-view prompting with an embedded 4s history of ego kinematics, (2) multi-stage chain-of-thought (CoT) prompting that enforces a Scene Understanding -> Driving Decision -> Trajectory Inference reasoning flow, and (3) spline-based trajectory post-processing that removes late-stage jitter and sharp turns. Trained on the Waymo Open Dataset, these upgrades enable HMVLM to achieve a Rater Feedback Score (RFS) of 7.7367, securing 2nd place in the 2025 Waymo Vision-based End-to-End (E2E) Driving Challenge and surpassing the public baseline by 2.77%. 

---
# Bayesian Persuasion as a Bargaining Game 

**Authors**: Yue Lin, Shuhui Zhu, William A Cunningham, Wenhao Li, Pascal Poupart, Hongyuan Zha, Baoxiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05876)  

**Abstract**: Bayesian persuasion, an extension of cheap-talk communication, involves an informed sender committing to a signaling scheme to influence a receiver's actions. Compared to cheap talk, this sender's commitment enables the receiver to verify the incentive compatibility of signals beforehand, facilitating cooperation. While effective in one-shot scenarios, Bayesian persuasion faces computational complexity (NP-hardness) when extended to long-term interactions, where the receiver may adopt dynamic strategies conditional on past outcomes and future expectations. To address this complexity, we introduce the bargaining perspective, which allows: (1) a unified framework and well-structured solution concept for long-term persuasion, with desirable properties such as fairness and Pareto efficiency; (2) a clear distinction between two previously conflated advantages: the sender's informational advantage and first-proposer advantage. With only modest modifications to the standard setting, this perspective makes explicit the common knowledge of the game structure and grants the receiver comparable commitment capabilities, thereby reinterpreting classic one-sided persuasion as a balanced information bargaining framework. The framework is validated through a two-stage validation-and-inference paradigm: We first demonstrate that GPT-o3 and DeepSeek-R1, out of publicly available LLMs, reliably handle standard tasks; We then apply them to persuasion scenarios to test that the outcomes align with what our information-bargaining framework suggests. All code, results, and terminal logs are publicly available at this http URL. 

---
# Research on Personalized Financial Product Recommendation by Integrating Large Language Models and Graph Neural Networks 

**Authors**: Yushang Zhao, Yike Peng, Dannier Li, Yuxin Yang, Chengrui Zhou, Jing Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.05873)  

**Abstract**: With the rapid growth of fintech, personalized financial product recommendations have become increasingly important. Traditional methods like collaborative filtering or content-based models often fail to capture users' latent preferences and complex relationships. We propose a hybrid framework integrating large language models (LLMs) and graph neural networks (GNNs). A pre-trained LLM encodes text data (e.g., user reviews) into rich feature vectors, while a heterogeneous user-product graph models interactions and social ties. Through a tailored message-passing mechanism, text and graph information are fused within the GNN to jointly optimize embeddings. Experiments on public and real-world financial datasets show our model outperforms standalone LLM or GNN in accuracy, recall, and NDCG, with strong interpretability. This work offers new insights for personalized financial recommendations and cross-modal fusion in broader recommendation tasks. 

---
# Loss Functions for Predictor-based Neural Architecture Search 

**Authors**: Han Ji, Yuqi Feng, Jiahao Fan, Yanan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.05869)  

**Abstract**: Evaluation is a critical but costly procedure in neural architecture search (NAS). Performance predictors have been widely adopted to reduce evaluation costs by directly estimating architecture performance. The effectiveness of predictors is heavily influenced by the choice of loss functions. While traditional predictors employ regression loss functions to evaluate the absolute accuracy of architectures, recent approaches have explored various ranking-based loss functions, such as pairwise and listwise ranking losses, to focus on the ranking of architecture performance. Despite their success in NAS, the effectiveness and characteristics of these loss functions have not been thoroughly investigated. In this paper, we conduct the first comprehensive study on loss functions in performance predictors, categorizing them into three main types: regression, ranking, and weighted loss functions. Specifically, we assess eight loss functions using a range of NAS-relevant metrics on 13 tasks across five search spaces. Our results reveal that specific categories of loss functions can be effectively combined to enhance predictor-based NAS. Furthermore, our findings could provide practical guidance for selecting appropriate loss functions for various tasks. We hope this work provides meaningful insights to guide the development of loss functions for predictor-based methods in the NAS community. 

---
# Cross-View Multi-Modal Segmentation @ Ego-Exo4D Challenges 2025 

**Authors**: Yuqian Fu, Runze Wang, Yanwei Fu, Danda Pani Paudel, Luc Van Gool  

**Link**: [PDF](https://arxiv.org/pdf/2506.05856)  

**Abstract**: In this report, we present a cross-view multi-modal object segmentation approach for the object correspondence task in the Ego-Exo4D Correspondence Challenges 2025. Given object queries from one perspective (e.g., ego view), the goal is to predict the corresponding object masks in another perspective (e.g., exo view). To tackle this task, we propose a multimodal condition fusion module that enhances object localization by leveraging both visual masks and textual descriptions as segmentation conditions. Furthermore, to address the visual domain gap between ego and exo views, we introduce a cross-view object alignment module that enforces object-level consistency across perspectives, thereby improving the model's robustness to viewpoint changes. Our proposed method ranked second on the leaderboard of the large-scale Ego-Exo4D object correspondence benchmark. Code will be made available at this https URL. 

---
# DeepFake Doctor: Diagnosing and Treating Audio-Video Fake Detection 

**Authors**: Marcel Klemt, Carlotta Segna, Anna Rohrbach  

**Link**: [PDF](https://arxiv.org/pdf/2506.05851)  

**Abstract**: Generative AI advances rapidly, allowing the creation of very realistic manipulated video and audio. This progress presents a significant security and ethical threat, as malicious users can exploit DeepFake techniques to spread misinformation. Recent DeepFake detection approaches explore the multimodal (audio-video) threat scenario. In particular, there is a lack of reproducibility and critical issues with existing datasets - such as the recently uncovered silence shortcut in the widely used FakeAVCeleb dataset. Considering the importance of this topic, we aim to gain a deeper understanding of the key issues affecting benchmarking in audio-video DeepFake detection. We examine these challenges through the lens of the three core benchmarking pillars: datasets, detection methods, and evaluation protocols. To address these issues, we spotlight the recent DeepSpeak v1 dataset and are the first to propose an evaluation protocol and benchmark it using SOTA models. We introduce SImple Multimodal BAseline (SIMBA), a competitive yet minimalistic approach that enables the exploration of diverse design choices. We also deepen insights into the issue of audio shortcuts and present a promising mitigation strategy. Finally, we analyze and enhance the evaluation scheme on the widely used FakeAVCeleb dataset. Our findings offer a way forward in the complex area of audio-video DeepFake detection. 

---
# Cross-lingual Collapse: How Language-Centric Foundation Models Shape Reasoning in Large Language Models 

**Authors**: Cheonbok Park, Jeonghoon Kim, Joosung Lee, Sanghwan Bae, Jaegul Choo, Kangmin Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2506.05850)  

**Abstract**: We identify \textbf{Cross-lingual Collapse}, a systematic drift in which the chain-of-thought (CoT) of a multilingual language model reverts to its dominant pre-training language even when the prompt is expressed in a different language. Recent large language models (LLMs) with reinforcement learning with verifiable reward (RLVR) have achieved strong logical reasoning performances by exposing their intermediate reasoning traces, giving rise to large reasoning models (LRMs). However, the mechanism behind multilingual reasoning in LRMs is not yet fully explored. To investigate the issue, we fine-tune multilingual LRMs with Group-Relative Policy Optimization (GRPO) on translated versions of the GSM$8$K and SimpleRL-Zoo datasets in three different languages: Chinese, Korean, and Ukrainian. During training, we monitor both task accuracy and language consistency of the reasoning chains. Our experiments reveal three key findings: (i) GRPO rapidly amplifies pre-training language imbalances, leading to the erosion of low-resource languages within just a few hundred updates; (ii) language consistency reward mitigates this drift but does so at the expense of an almost 5 - 10 pp drop in accuracy. and (iii) the resulting language collapse is severely damaging and largely irreversible, as subsequent fine-tuning struggles to steer the model back toward its original target-language reasoning capabilities. Together, these findings point to a remarkable conclusion: \textit{not all languages are trained equally for reasoning}. Furthermore, our paper sheds light on the roles of reward shaping, data difficulty, and pre-training priors in eliciting multilingual reasoning. 

---
# Regional, Lattice and Logical Representations of Neural Networks 

**Authors**: Sandro Preto, Marcelo Finger  

**Link**: [PDF](https://arxiv.org/pdf/2506.05834)  

**Abstract**: A possible path to the interpretability of neural networks is to (approximately) represent them in the regional format of piecewise linear functions, where regions of inputs are associated to linear functions computing the network outputs. We present an algorithm for the translation of feedforward neural networks with ReLU activation functions in hidden layers and truncated identity activation functions in the output layer. We also empirically investigate the complexity of regional representations outputted by our method for neural networks with varying sizes. Lattice and logical representations of neural networks are straightforward from regional representations as long as they satisfy a specific property. So we empirically investigate to what extent the translations by our algorithm satisfy such property. 

---
# Fuzzy Lattice-based Description Logic 

**Authors**: Yiwen Ding, Krishna Manoorkar  

**Link**: [PDF](https://arxiv.org/pdf/2506.05833)  

**Abstract**: Recently, description logic LE-ALC was introduced for reasoning in the semantic environment of enriched formal contexts, and a polynomial-time tableaux algorithm was developed to check the consistency of knowledge bases with acyclic TBoxes. In this work, we introduce a fuzzy generalization of LE-ALC  called  LE-FALC which provides a description logic counterpart of many-valued normal non-distributive logic a.k.a. many-valued LE-logic. This description logic can be used to represent and reason about knowledge in the formal framework  of fuzzy formal contexts and fuzzy formal concepts. We provide a tableaux algorithm that provides a complete and sound polynomial-time decision procedure to check the consistency of  LE-FALC  ABoxes. As a result, we also obtain an exponential-time decision procedure for checking the consistency of  LE-FALC  with acyclic TBoxes by unraveling. 

---
# Heartcare Suite: Multi-dimensional Understanding of ECG with Raw Multi-lead Signal Modeling 

**Authors**: Yihan Xie, Sijing Li, Tianwei Lin, Zhuonan Wang, Chenglin Yang, Yu Zhong, Wenqiao Zhang, Haoyuan Li, Hao Jiang, Fengda Zhang, Qishan Chen, Jun Xiao, Yueting Zhuang, Beng Chin Ooi  

**Link**: [PDF](https://arxiv.org/pdf/2506.05831)  

**Abstract**: We present Heartcare Suite, a multimodal comprehensive framework for finegrained electrocardiogram (ECG) understanding. It comprises three key components: (i) Heartcare-220K, a high-quality, structured, and comprehensive multimodal ECG dataset covering essential tasks such as disease diagnosis, waveform morphology analysis, and rhythm interpretation. (ii) Heartcare-Bench, a systematic and multi-dimensional benchmark designed to evaluate diagnostic intelligence and guide the optimization of Medical Multimodal Large Language Models (Med-MLLMs) in ECG scenarios. and (iii) HeartcareGPT with a tailored tokenizer Bidirectional ECG Abstract Tokenization (Beat), which compresses raw multi-lead signals into semantically rich discrete tokens via duallevel vector quantization and query-guided bidirectional diffusion mechanism. Built upon Heartcare-220K, HeartcareGPT achieves strong generalization and SoTA performance across multiple clinically meaningful tasks. Extensive experiments demonstrate that Heartcare Suite is highly effective in advancing ECGspecific multimodal understanding and evaluation. Our project is available at this https URL . 

---
# FuseUNet: A Multi-Scale Feature Fusion Method for U-like Networks 

**Authors**: Quansong He, Xiangde Min, Kaishen Wang, Tao He  

**Link**: [PDF](https://arxiv.org/pdf/2506.05821)  

**Abstract**: Medical image segmentation is a critical task in computer vision, with UNet serving as a milestone architecture. The typical component of UNet family is the skip connection, however, their skip connections face two significant limitations: (1) they lack effective interaction between features at different scales, and (2) they rely on simple concatenation or addition operations, which constrain efficient information integration. While recent improvements to UNet have focused on enhancing encoder and decoder capabilities, these limitations remain overlooked. To overcome these challenges, we propose a novel multi-scale feature fusion method that reimagines the UNet decoding process as solving an initial value problem (IVP), treating skip connections as discrete nodes. By leveraging principles from the linear multistep method, we propose an adaptive ordinary differential equation method to enable effective multi-scale feature fusion. Our approach is independent of the encoder and decoder architectures, making it adaptable to various U-Net-like networks. Experiments on ACDC, KiTS2023, MSD brain tumor, and ISIC2017/2018 skin lesion segmentation datasets demonstrate improved feature utilization, reduced network parameters, and maintained high performance. The code is available at this https URL. 

---
# Positional Encoding meets Persistent Homology on Graphs 

**Authors**: Yogesh Verma, Amauri H. Souza, Vikas Garg  

**Link**: [PDF](https://arxiv.org/pdf/2506.05814)  

**Abstract**: The local inductive bias of message-passing graph neural networks (GNNs) hampers their ability to exploit key structural information (e.g., connectivity and cycles). Positional encoding (PE) and Persistent Homology (PH) have emerged as two promising approaches to mitigate this issue. PE schemes endow GNNs with location-aware features, while PH methods enhance GNNs with multiresolution topological features. However, a rigorous theoretical characterization of the relative merits and shortcomings of PE and PH has remained elusive. We bridge this gap by establishing that neither paradigm is more expressive than the other, providing novel constructions where one approach fails but the other succeeds. Our insights inform the design of a novel learnable method, PiPE (Persistence-informed Positional Encoding), which is provably more expressive than both PH and PE. PiPE demonstrates strong performance across a variety of tasks (e.g., molecule property prediction, graph classification, and out-of-distribution generalization), thereby advancing the frontiers of graph representation learning. Code is available at this https URL. 

---
# Robust sensor fusion against on-vehicle sensor staleness 

**Authors**: Meng Fan, Yifan Zuo, Patrick Blaes, Harley Montgomery, Subhasis Das  

**Link**: [PDF](https://arxiv.org/pdf/2506.05780)  

**Abstract**: Sensor fusion is crucial for a performant and robust Perception system in autonomous vehicles, but sensor staleness, where data from different sensors arrives with varying delays, poses significant challenges. Temporal misalignment between sensor modalities leads to inconsistent object state estimates, severely degrading the quality of trajectory predictions that are critical for safety. We present a novel and model-agnostic approach to address this problem via (1) a per-point timestamp offset feature (for LiDAR and radar both relative to camera) that enables fine-grained temporal awareness in sensor fusion, and (2) a data augmentation strategy that simulates realistic sensor staleness patterns observed in deployed vehicles. Our method is integrated into a perspective-view detection model that consumes sensor data from multiple LiDARs, radars and cameras. We demonstrate that while a conventional model shows significant regressions when one sensor modality is stale, our approach reaches consistently good performance across both synchronized and stale conditions. 

---
# dots.llm1 Technical Report 

**Authors**: Bi Huo, Bin Tu, Cheng Qin, Da Zheng, Debing Zhang, Dongjie Zhang, En Li, Fu Guo, Jian Yao, Jie Lou, Junfeng Tian, Li Hu, Ran Zhu, Shengdong Chen, Shuo Liu, Su Guang, Te Wo, Weijun Zhang, Xiaoming Shi, Xinxin Peng, Xing Wu, Yawen Liu, Yuqiu Ji, Ze Wen, Zhenhai Liu, Zichao Li, Zilong Liao  

**Link**: [PDF](https://arxiv.org/pdf/2506.05767)  

**Abstract**: Mixture of Experts (MoE) models have emerged as a promising paradigm for scaling language models efficiently by activating only a subset of parameters for each input token. In this report, we present dots.llm1, a large-scale MoE model that activates 14B parameters out of a total of 142B parameters, delivering performance on par with state-of-the-art models while reducing training and inference costs. Leveraging our meticulously crafted and efficient data processing pipeline, dots.llm1 achieves performance comparable to Qwen2.5-72B after pretraining on 11.2T high-quality tokens and post-training to fully unlock its capabilities. Notably, no synthetic data is used during pretraining. To foster further research, we open-source intermediate training checkpoints at every one trillion tokens, providing valuable insights into the learning dynamics of large language models. 

---
# Revealing hidden correlations from complex spatial distributions: Adjacent Correlation Analysis 

**Authors**: Guang-Xing Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.05759)  

**Abstract**: Physics has been transforming our view of nature for centuries. While combining physical knowledge with computational approaches has enabled detailed modeling of physical systems' evolution, understanding the emergence of patterns and structures remains limited. Correlations between quantities are the most reliable approach to describe relationships between different variables. However, for complex patterns, directly searching for correlations is often impractical, as complexity and spatial inhomogeneity can obscure correlations. We discovered that the key is to search for correlations in local regions and developed a new method, adjacent correlation analysis, to extract such correlations and represent them in phase space. When multiple observations are available, a useful way to study a system is to analyze distributions in phase space using the Probability Density Function (PDF). Adjacent correlation analysis evaluates vectors representing local correlations, which can be overlaid on the PDF plot to form the adjacent correlation plot. These correlation vectors often exhibit remarkably regular patterns and may lead to the discovery of new laws. The vectors we derive are equivalent to the vector field in dynamical systems on the attracting manifold. By efficiently representing spatial patterns as correlation vectors in phase space, our approach opens avenues for classification, prediction, parameter fitting, and forecasting. 

---
# FlowOE: Imitation Learning with Flow Policy from Ensemble RL Experts for Optimal Execution under Heston Volatility and Concave Market Impacts 

**Authors**: Yang Li, Zhi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.05755)  

**Abstract**: Optimal execution in financial markets refers to the process of strategically transacting a large volume of assets over a period to achieve the best possible outcome by balancing the trade-off between market impact costs and timing or volatility risks. Traditional optimal execution strategies, such as static Almgren-Chriss models, often prove suboptimal in dynamic financial markets. This paper propose flowOE, a novel imitation learning framework based on flow matching models, to address these limitations. FlowOE learns from a diverse set of expert traditional strategies and adaptively selects the most suitable expert behavior for prevailing market conditions. A key innovation is the incorporation of a refining loss function during the imitation process, enabling flowOE not only to mimic but also to improve upon the learned expert actions. To the best of our knowledge, this work is the first to apply flow matching models in a stochastic optimal execution problem. Empirical evaluations across various market conditions demonstrate that flowOE significantly outperforms both the specifically calibrated expert models and other traditional benchmarks, achieving higher profits with reduced risk. These results underscore the practical applicability and potential of flowOE to enhance adaptive optimal execution. 

---
# Integrating Spatiotemporal Features in LSTM for Spatially Informed COVID-19 Hospitalization Forecasting 

**Authors**: Zhongying Wang, Thoai D. Ngo, Hamidreza Zoraghein, Benjamin Lucas, Morteza Karimzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2506.05752)  

**Abstract**: The COVID-19 pandemic's severe impact highlighted the need for accurate, timely hospitalization forecasting to support effective healthcare planning. However, most forecasting models struggled, especially during variant surges, when they were needed most. This study introduces a novel Long Short-Term Memory (LSTM) framework for forecasting daily state-level incident hospitalizations in the United States. We present a spatiotemporal feature, Social Proximity to Hospitalizations (SPH), derived from Facebook's Social Connectedness Index to improve forecasts. SPH serves as a proxy for interstate population interaction, capturing transmission dynamics across space and time. Our parallel LSTM architecture captures both short- and long-term temporal dependencies, and our multi-horizon ensembling strategy balances consistency and forecasting error. Evaluation against COVID-19 Forecast Hub ensemble models during the Delta and Omicron surges reveals superiority of our model. On average, our model surpasses the ensemble by 27, 42, 54, and 69 hospitalizations per state on the $7^{th}$, $14^{th}$, $21^{st}$, and $28^{th}$ forecast days, respectively, during the Omicron surge. Data-ablation experiments confirm SPH's predictive power, highlighting its effectiveness in enhancing forecasting models. This research not only advances hospitalization forecasting but also underscores the significance of spatiotemporal features, such as SPH, in refining predictive performance in modeling the complex dynamics of infectious disease spread. 

---
# An Ontology for Representing Curriculum and Learning Material 

**Authors**: Antrea Christou, Chris Davis Jaldi, Joseph Zalewski, Hande Küçük McGinty, Pascal Hitzler, Cogan Shimizu  

**Link**: [PDF](https://arxiv.org/pdf/2506.05751)  

**Abstract**: Educational, learning, and training materials have become extremely commonplace across the Internet. Yet, they frequently remain disconnected from each other, fall into platform silos, and so on. One way to overcome this is to provide a mechanism to integrate the material and provide cross-links across topics.
In this paper, we present the Curriculum KG Ontology, which we use as a framework for the dense interlinking of educational materials, by first starting with organizational and broad pedagogical principles. We provide a materialized graph for the Prototype Open Knowledge Network use-case, and validate it using competency questions sourced from domain experts and educators. 

---
# Efficient Online RFT with Plug-and-Play LLM Judges: Unlocking State-of-the-Art Performance 

**Authors**: Rudransh Agnihotri, Ananya Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2506.05748)  

**Abstract**: Reward-model training is the cost bottleneck in modern Reinforcement Learning Human Feedback (RLHF) pipelines, often requiring tens of billions of parameters and an offline preference-tuning phase. In the proposed method, a frozen, instruction-tuned 7B LLM is augmented with only a one line JSON rubric and a rank-16 LoRA adapter (affecting just 0.8% of the model's parameters), enabling it to serve as a complete substitute for the previously used heavyweight evaluation models. The plug-and-play judge achieves 96.2% accuracy on RewardBench, outperforming specialized reward networks ranging from 27B to 70B parameters. Additionally, it allows a 7B actor to outperform the top 70B DPO baseline, which scores 61.8%, by achieving 92% exact match accuracy on GSM-8K utilizing online PPO. Thorough ablations indicate that (i) six in context demonstrations deliver the majority of the zero-to-few-shot improvements (+2pp), and (ii) the LoRA effectively addresses the remaining disparity, particularly in the safety and adversarial Chat-Hard segments. The proposed model introduces HH-Rationales, a subset of 10,000 pairs from Anthropic HH-RLHF, to examine interpretability, accompanied by human generated justifications. GPT-4 scoring indicates that our LoRA judge attains approximately = 9/10 in similarity to human explanations, while zero-shot judges score around =5/10. These results indicate that the combination of prompt engineering and tiny LoRA produces a cost effective, transparent, and easily adjustable reward function, removing the offline phase while achieving new state-of-the-art outcomes for both static evaluation and online RLHF. 

---
# When Better Features Mean Greater Risks: The Performance-Privacy Trade-Off in Contrastive Learning 

**Authors**: Ruining Sun, Hongsheng Hu, Wei Luo, Zhaoxi Zhang, Yanjun Zhang, Haizhuan Yuan, Leo Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05743)  

**Abstract**: With the rapid advancement of deep learning technology, pre-trained encoder models have demonstrated exceptional feature extraction capabilities, playing a pivotal role in the research and application of deep learning. However, their widespread use has raised significant concerns about the risk of training data privacy leakage. This paper systematically investigates the privacy threats posed by membership inference attacks (MIAs) targeting encoder models, focusing on contrastive learning frameworks. Through experimental analysis, we reveal the significant impact of model architecture complexity on membership privacy leakage: As more advanced encoder frameworks improve feature-extraction performance, they simultaneously exacerbate privacy-leakage risks. Furthermore, this paper proposes a novel membership inference attack method based on the p-norm of feature vectors, termed the Embedding Lp-Norm Likelihood Attack (LpLA). This method infers membership status, by leveraging the statistical distribution characteristics of the p-norm of feature vectors. Experimental results across multiple datasets and model architectures demonstrate that LpLA outperforms existing methods in attack performance and robustness, particularly under limited attack knowledge and query volumes. This study not only uncovers the potential risks of privacy leakage in contrastive learning frameworks, but also provides a practical basis for privacy protection research in encoder models. We hope that this work will draw greater attention to the privacy risks associated with self-supervised learning models and shed light on the importance of a balance between model utility and training data privacy. Our code is publicly available at: this https URL. 

---
# To Protect the LLM Agent Against the Prompt Injection Attack with Polymorphic Prompt 

**Authors**: Zhilong Wang, Neha Nagaraja, Lan Zhang, Hayretdin Bahsi, Pawan Patil, Peng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.05739)  

**Abstract**: LLM agents are widely used as agents for customer support, content generation, and code assistance. However, they are vulnerable to prompt injection attacks, where adversarial inputs manipulate the model's behavior. Traditional defenses like input sanitization, guard models, and guardrails are either cumbersome or ineffective. In this paper, we propose a novel, lightweight defense mechanism called Polymorphic Prompt Assembling (PPA), which protects against prompt injection with near-zero overhead. The approach is based on the insight that prompt injection requires guessing and breaking the structure of the system prompt. By dynamically varying the structure of system prompts, PPA prevents attackers from predicting the prompt structure, thereby enhancing security without compromising performance. We conducted experiments to evaluate the effectiveness of PPA against existing attacks and compared it with other defense methods. 

---
# Generalized Incremental Learning under Concept Drift across Evolving Data Streams 

**Authors**: En Yu, Jie Lu, Guangquan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05736)  

**Abstract**: Real-world data streams exhibit inherent non-stationarity characterized by concept drift, posing significant challenges for adaptive learning systems. While existing methods address isolated distribution shifts, they overlook the critical co-evolution of label spaces and distributions under limited supervision and persistent uncertainty. To address this, we formalize Generalized Incremental Learning under Concept Drift (GILCD), characterizing the joint evolution of distributions and label spaces in open-environment streaming contexts, and propose a novel framework called Calibrated Source-Free Adaptation (CSFA). First, CSFA introduces a training-free prototype calibration mechanism that dynamically fuses emerging prototypes with base representations, enabling stable new-class identification without optimization overhead. Second, we design a novel source-free adaptation algorithm, i.e., Reliable Surrogate Gap Sharpness-aware (RSGS) minimization. It integrates sharpness-aware perturbation loss optimization with surrogate gap minimization, while employing entropy-based uncertainty filtering to discard unreliable samples. This mechanism ensures robust distribution alignment and mitigates generalization degradation caused by uncertainties. Therefore, CSFA establishes a unified framework for stable adaptation to evolving semantics and distributions in open-world streaming scenarios. Extensive experiments validate the superior performance and effectiveness of CSFA compared to state-of-the-art approaches. 

---
# Large Language Models are Good Relational Learners 

**Authors**: Fang Wu, Vijay Prakash Dwivedi, Jure Leskovec  

**Link**: [PDF](https://arxiv.org/pdf/2506.05725)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across various domains, yet their application to relational deep learning (RDL) remains underexplored. Existing approaches adapt LLMs by traversing relational links between entities in a database and converting the structured data into flat text documents. Still, this text-based serialization disregards critical relational structures, introduces redundancy, and often exceeds standard LLM context lengths. We introduce Rel-LLM, a novel architecture that utilizes a graph neural network (GNN)- based encoder to generate structured relational prompts for LLMs within a retrieval-augmented generation (RAG) framework. Unlike traditional text-based serialization approaches, our method preserves the inherent relational structure of databases while enabling LLMs to effectively process and reason over complex entity relationships. Specifically, the GNN encoder extracts a local subgraph around an entity to build feature representations that contain relevant entity relationships and temporal dependencies. These representations are transformed into structured prompts using a denormalization process, effectively allowing the LLM to reason over relational structures. Through extensive experiments, we demonstrate that Rel-LLM outperforms existing methods on key RDL tasks, offering a scalable and efficient approach to integrating LLMs with structured data sources. Code is available at this https URL. 

---
# Any-Class Presence Likelihood for Robust Multi-Label Classification with Abundant Negative Data 

**Authors**: Dumindu Tissera, Omar Awadallah, Muhammad Umair Danish, Ayan Sadhu, Katarina Grolinger  

**Link**: [PDF](https://arxiv.org/pdf/2506.05721)  

**Abstract**: Multi-label Classification (MLC) assigns an instance to one or more non-exclusive classes. A challenge arises when the dataset contains a large proportion of instances with no assigned class, referred to as negative data, which can overwhelm the learning process and hinder the accurate identification and classification of positive instances. Nevertheless, it is common in MLC applications such as industrial defect detection, agricultural disease identification, and healthcare diagnosis to encounter large amounts of negative data. Assigning a separate negative class to these instances further complicates the learning objective and introduces unnecessary redundancies. To address this challenge, we redesign standard MLC loss functions by deriving a likelihood of any class being present, formulated by a normalized weighted geometric mean of the predicted class probabilities. We introduce a regularization parameter that controls the relative contribution of the absent class probabilities to the any-class presence likelihood in positive instances. The any-class presence likelihood complements the multi-label learning by encouraging the network to become more aware of implicit positive instances and improve the label classification within those positive instances. Experiments on large-scale datasets with negative data: SewerML, modified COCO, and ChestX-ray14, across various networks and base loss functions show that our loss functions consistently improve MLC performance of their standard loss counterparts, achieving gains of up to 6.01 percentage points in F1, 8.06 in F2, and 3.11 in mean average precision, all without additional parameters or computational complexity. Code available at: this https URL 

---
# Grokking Beyond the Euclidean Norm of Model Parameters 

**Authors**: Pascal Jr Tikeng Notsawo, Guillaume Dumas, Guillaume Rabusseau  

**Link**: [PDF](https://arxiv.org/pdf/2506.05718)  

**Abstract**: Grokking refers to a delayed generalization following overfitting when optimizing artificial neural networks with gradient-based methods. In this work, we demonstrate that grokking can be induced by regularization, either explicit or implicit. More precisely, we show that when there exists a model with a property $P$ (e.g., sparse or low-rank weights) that generalizes on the problem of interest, gradient descent with a small but non-zero regularization of $P$ (e.g., $\ell_1$ or nuclear norm regularization) results in grokking. This extends previous work showing that small non-zero weight decay induces grokking. Moreover, our analysis shows that over-parameterization by adding depth makes it possible to grok or ungrok without explicitly using regularization, which is impossible in shallow cases. We further show that the $\ell_2$ norm is not a reliable proxy for generalization when the model is regularized toward a different property $P$, as the $\ell_2$ norm grows in many cases where no weight decay is used, but the model generalizes anyway. We also show that grokking can be amplified solely through data selection, with any other hyperparameter fixed. 

---
# Ensemble Elastic DQN: A novel multi-step ensemble approach to address overestimation in deep value-based reinforcement learning 

**Authors**: Adrian Ly, Richard Dazeley, Peter Vamplew, Francisco Cruz, Sunil Aryal  

**Link**: [PDF](https://arxiv.org/pdf/2506.05716)  

**Abstract**: While many algorithmic extensions to Deep Q-Networks (DQN) have been proposed, there remains limited understanding of how different improvements interact. In particular, multi-step and ensemble style extensions have shown promise in reducing overestimation bias, thereby improving sample efficiency and algorithmic stability. In this paper, we introduce a novel algorithm called Ensemble Elastic Step DQN (EEDQN), which unifies ensembles with elastic step updates to stabilise algorithmic performance. EEDQN is designed to address two major challenges in deep reinforcement learning: overestimation bias and sample efficiency. We evaluated EEDQN against standard and ensemble DQN variants across the MinAtar benchmark, a set of environments that emphasise behavioral learning while reducing representational complexity. Our results show that EEDQN achieves consistently robust performance across all tested environments, outperforming baseline DQN methods and matching or exceeding state-of-the-art ensemble DQNs in final returns on most of the MinAtar environments. These findings highlight the potential of systematically combining algorithmic improvements and provide evidence that ensemble and multi-step methods, when carefully integrated, can yield substantial gains. 

---
# Action-Adaptive Continual Learning: Enabling Policy Generalization under Dynamic Action Spaces 

**Authors**: Chaofan Pan, Jiafen Liu, Yanhua Li, Linbo Xiong, Fan Min, Wei Wei, Xin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05702)  

**Abstract**: Continual Learning (CL) is a powerful tool that enables agents to learn a sequence of tasks, accumulating knowledge learned in the past and using it for problem-solving or future task learning. However, existing CL methods often assume that the agent's capabilities remain static within dynamic environments, which doesn't reflect real-world scenarios where capabilities dynamically change. This paper introduces a new and realistic problem: Continual Learning with Dynamic Capabilities (CL-DC), posing a significant challenge for CL agents: How can policy generalization across different action spaces be achieved? Inspired by the cortical functions, we propose an Action-Adaptive Continual Learning framework (AACL) to address this challenge. Our framework decouples the agent's policy from the specific action space by building an action representation space. For a new action space, the encoder-decoder of action representations is adaptively fine-tuned to maintain a balance between stability and plasticity. Furthermore, we release a benchmark based on three environments to validate the effectiveness of methods for CL-DC. Experimental results demonstrate that our framework outperforms popular methods by generalizing the policy across action spaces. 

---
# RKEFino1: A Regulation Knowledge-Enhanced Large Language Model 

**Authors**: Yan Wang, Yueru He, Ruoyu Xiang, Jeff Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.05700)  

**Abstract**: Recent advances in large language models (LLMs) hold great promise for financial applications but introduce critical accuracy and compliance challenges in Digital Regulatory Reporting (DRR). To address these issues, we propose RKEFino1, a regulation knowledge-enhanced financial reasoning model built upon Fino1, fine-tuned with domain knowledge from XBRL, CDM, and MOF. We formulate two QA tasks-knowledge-based and mathematical reasoning-and introduce a novel Numerical NER task covering financial entities in both sentences and tables. Experimental results demonstrate the effectiveness and generalization capacity of RKEFino1 in compliance-critical financial tasks. We have released our model on Hugging Face. 

---
# Evaluating AI-Powered Learning Assistants in Engineering Higher Education: Student Engagement, Ethical Challenges, and Policy Implications 

**Authors**: Ramteja Sajja, Yusuf Sermet, Brian Fodale, Ibrahim Demir  

**Link**: [PDF](https://arxiv.org/pdf/2506.05699)  

**Abstract**: As generative AI tools become increasingly integrated into higher education, understanding how students interact with and perceive these technologies is essential for responsible and effective adoption. This study evaluates the use of the Educational AI Hub, an AI-powered learning framework, in undergraduate civil and environmental engineering courses at a large R1 public university. Using a mixed-methods approach that combines pre- and post-surveys, system usage logs, and qualitative analysis of the open-ended prompts and questions students posed to the AI chatbot, the research explores students' perceptions of trust, ethical concerns, usability, and learning outcomes. Findings reveal that students appreciated the AI assistant for its convenience and comfort, with nearly half reporting greater ease in using the AI tool compared to seeking help from instructors or teaching assistants. The tool was seen as most helpful for completing homework and understanding course concepts, though perceptions of its instructional quality were mixed. Ethical concerns emerged as a key barrier to full engagement: while most students viewed AI use as ethically acceptable, many expressed uncertainties about institutional policies and apprehension about potential academic misconduct. This study contributes to the growing body of research on AI in education by highlighting the importance of usability, policy clarity, and faculty guidance in fostering meaningful AI engagement. The findings suggest that while students are ready to embrace AI as a supplement to human instruction, thoughtful integration and transparent institutional frameworks are critical for ensuring student confidence, trust, and learning effectiveness. 

---
# SafeGenBench: A Benchmark Framework for Security Vulnerability Detection in LLM-Generated Code 

**Authors**: Xinghang Li, Jingzhe Ding, Chao Peng, Bing Zhao, Xiang Gao, Hongwan Gao, Xinchen Gu  

**Link**: [PDF](https://arxiv.org/pdf/2506.05692)  

**Abstract**: The code generation capabilities of large language models(LLMs) have emerged as a critical dimension in evaluating their overall performance. However, prior research has largely overlooked the security risks inherent in the generated code. In this work, we introduce \benchmark, a benchmark specifically designed to assess the security of LLM-generated code. The dataset encompasses a wide range of common software development scenarios and vulnerability types. Building upon this benchmark, we develop an automatic evaluation framework that leverages both static application security testing(SAST) and LLM-based judging to assess the presence of security vulnerabilities in model-generated code. Through the empirical evaluation of state-of-the-art LLMs on \benchmark, we reveal notable deficiencies in their ability to produce vulnerability-free code. Our findings highlight pressing challenges and offer actionable insights for future advancements in the secure code generation performance of LLMs. The data and code will be released soon. 

---
# Multi-Modal Multi-Task Federated Foundation Models for Next-Generation Extended Reality Systems: Towards Privacy-Preserving Distributed Intelligence in AR/VR/MR 

**Authors**: Fardis Nadimi, Payam Abdisarabshali, Kasra Borazjani, Jacob Chakareski, Seyyedali Hosseinalipour  

**Link**: [PDF](https://arxiv.org/pdf/2506.05683)  

**Abstract**: Extended reality (XR) systems, which consist of virtual reality (VR), augmented reality (AR), and mixed reality (XR), offer a transformative interface for immersive, multi-modal, and embodied human-computer interaction. In this paper, we envision that multi-modal multi-task (M3T) federated foundation models (FedFMs) can offer transformative capabilities for XR systems through integrating the representational strength of M3T foundation models (FMs) with the privacy-preserving model training principles of federated learning (FL). We present a modular architecture for FedFMs, which entails different coordination paradigms for model training and aggregations. Central to our vision is the codification of XR challenges that affect the implementation of FedFMs under the SHIFT dimensions: (1) Sensor and modality diversity, (2) Hardware heterogeneity and system-level constraints, (3) Interactivity and embodied personalization, (4) Functional/task variability, and (5) Temporality and environmental variability. We illustrate the manifestation of these dimensions across a set of emerging and anticipated applications of XR systems. Finally, we propose evaluation metrics, dataset requirements, and design tradeoffs necessary for the development of resource-aware FedFMs in XR. This perspective aims to chart the technical and conceptual foundations for context-aware privacy-preserving intelligence in the next generation of XR systems. 

---
# Learning Design-Score Manifold to Guide Diffusion Models for Offline Optimization 

**Authors**: Tailin Zhou, Zhilin Chen, Wenlong Lyu, Zhitang Chen, Danny H.K. Tsang, Jun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05680)  

**Abstract**: Optimizing complex systems, from discovering therapeutic drugs to designing high-performance materials, remains a fundamental challenge across science and engineering, as the underlying rules are often unknown and costly to evaluate. Offline optimization aims to optimize designs for target scores using pre-collected datasets without system interaction. However, conventional approaches may fail beyond training data, predicting inaccurate scores and generating inferior designs. This paper introduces ManGO, a diffusion-based framework that learns the design-score manifold, capturing the design-score interdependencies holistically. Unlike existing methods that treat design and score spaces in isolation, ManGO unifies forward prediction and backward generation, attaining generalization beyond training data. Key to this is its derivative-free guidance for conditional generation, coupled with adaptive inference-time scaling that dynamically optimizes denoising paths. Extensive evaluations demonstrate that ManGO outperforms 24 single- and 10 multi-objective optimization methods across diverse domains, including synthetic tasks, robot control, material design, DNA sequence, and real-world engineering optimization. 

---
# Peer-Ranked Precision: Creating a Foundational Dataset for Fine-Tuning Vision Models from DataSeeds' Annotated Imagery 

**Authors**: Sajjad Abdoli, Freeman Lewin, Gediminas Vasiliauskas, Fabian Schonholz  

**Link**: [PDF](https://arxiv.org/pdf/2506.05673)  

**Abstract**: The development of modern Artificial Intelligence (AI) models, particularly diffusion-based models employed in computer vision and image generation tasks, is undergoing a paradigmatic shift in development methodologies. Traditionally dominated by a "Model Centric" approach, in which performance gains were primarily pursued through increasingly complex model architectures and hyperparameter optimization, the field is now recognizing a more nuanced "Data-Centric" approach. This emergent framework foregrounds the quality, structure, and relevance of training data as the principal driver of model performance. To operationalize this paradigm shift, we introduce the this http URL sample dataset (the "DSD"), initially comprised of approximately 10,610 high-quality human peer-ranked photography images accompanied by extensive multi-tier annotations. The DSD is a foundational computer vision dataset designed to usher in a new standard for commercial image datasets. Representing a small fraction of this http URL's 100 million-plus image catalog, the DSD provides a scalable foundation necessary for robust commercial and multimodal AI development. Through this in-depth exploratory analysis, we document the quantitative improvements generated by the DSD on specific models against known benchmarks and make the code and the trained models used in our evaluation publicly available. 

---
# DriveAction: A Benchmark for Exploring Human-like Driving Decisions in VLA Models 

**Authors**: Yuhan Hao, Zhengning Li, Lei Sun, Weilong Wang, Naixin Yi, Sheng Song, Caihong Qin, Mofan Zhou, Yifei Zhan, Peng Jia, Xianpeng Lang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05667)  

**Abstract**: Vision-Language-Action (VLA) models have advanced autonomous driving, but existing benchmarks still lack scenario diversity, reliable action-level annotation, and evaluation protocols aligned with human preferences. To address these limitations, we introduce DriveAction, the first action-driven benchmark specifically designed for VLA models, comprising 16,185 QA pairs generated from 2,610 driving scenarios. DriveAction leverages real-world driving data proactively collected by users of production-level autonomous vehicles to ensure broad and representative scenario coverage, offers high-level discrete action labels collected directly from users' actual driving operations, and implements an action-rooted tree-structured evaluation framework that explicitly links vision, language, and action tasks, supporting both comprehensive and task-specific assessment. Our experiments demonstrate that state-of-the-art vision-language models (VLMs) require both vision and language guidance for accurate action prediction: on average, accuracy drops by 3.3% without vision input, by 4.1% without language input, and by 8.0% without either. Our evaluation supports precise identification of model bottlenecks with robust and consistent results, thus providing new insights and a rigorous foundation for advancing human-like decisions in autonomous driving. 

---
# TissUnet: Improved Extracranial Tissue and Cranium Segmentation for Children through Adulthood 

**Authors**: Markian Mandzak, Elvira Yang, Anna Zapaishchykova, Yu-Hui Chen, Lucas Heilbroner, John Zielke, Divyanshu Tak, Reza Mojahed-Yazdi, Francesca Romana Mussa, Zezhong Ye, Sridhar Vajapeyam, Viviana Benitez, Ralph Salloum, Susan N. Chi, Houman Sotoudeh, Jakob Seidlitz, Sabine Mueller, Hugo J.W.L. Aerts, Tina Y. Poussaint, Benjamin H. Kann  

**Link**: [PDF](https://arxiv.org/pdf/2506.05660)  

**Abstract**: Extracranial tissues visible on brain magnetic resonance imaging (MRI) may hold significant value for characterizing health conditions and clinical decision-making, yet they are rarely quantified. Current tools have not been widely validated, particularly in settings of developing brains or underlying pathology. We present TissUnet, a deep learning model that segments skull bone, subcutaneous fat, and muscle from routine three-dimensional T1-weighted MRI, with or without contrast enhancement. The model was trained on 155 paired MRI-computed tomography (CT) scans and validated across nine datasets covering a wide age range and including individuals with brain tumors. In comparison to AI-CT-derived labels from 37 MRI-CT pairs, TissUnet achieved a median Dice coefficient of 0.79 [IQR: 0.77-0.81] in a healthy adult cohort. In a second validation using expert manual annotations, median Dice was 0.83 [IQR: 0.83-0.84] in healthy individuals and 0.81 [IQR: 0.78-0.83] in tumor cases, outperforming previous state-of-the-art method. Acceptability testing resulted in an 89% acceptance rate after adjudication by a tie-breaker(N=108 MRIs), and TissUnet demonstrated excellent performance in the blinded comparative review (N=45 MRIs), including both healthy and tumor cases in pediatric populations. TissUnet enables fast, accurate, and reproducible segmentation of extracranial tissues, supporting large-scale studies on craniofacial morphology, treatment effects, and cardiometabolic risk using standard brain T1w MRI. 

---
# Bayesian Inference for Correlated Human Experts and Classifiers 

**Authors**: Markelle Kelly, Alex Boyd, Sam Showalter, Mark Steyvers, Padhraic Smyth  

**Link**: [PDF](https://arxiv.org/pdf/2506.05636)  

**Abstract**: Applications of machine learning often involve making predictions based on both model outputs and the opinions of human experts. In this context, we investigate the problem of querying experts for class label predictions, using as few human queries as possible, and leveraging the class probability estimates of pre-trained classifiers. We develop a general Bayesian framework for this problem, modeling expert correlation via a joint latent representation, enabling simulation-based inference about the utility of additional expert queries, as well as inference of posterior distributions over unobserved expert labels. We apply our approach to two real-world medical classification problems, as well as to CIFAR-10H and ImageNet-16H, demonstrating substantial reductions relative to baselines in the cost of querying human experts while maintaining high prediction accuracy. 

---
# AutoQD: Automatic Discovery of Diverse Behaviors with Quality-Diversity Optimization 

**Authors**: Saeed Hedayatian, Stefanos Nikolaidis  

**Link**: [PDF](https://arxiv.org/pdf/2506.05634)  

**Abstract**: Quality-Diversity (QD) algorithms have shown remarkable success in discovering diverse, high-performing solutions, but rely heavily on hand-crafted behavioral descriptors that constrain exploration to predefined notions of diversity. Leveraging the equivalence between policies and occupancy measures, we present a theoretically grounded approach to automatically generate behavioral descriptors by embedding the occupancy measures of policies in Markov Decision Processes. Our method, AutoQD, leverages random Fourier features to approximate the Maximum Mean Discrepancy (MMD) between policy occupancy measures, creating embeddings whose distances reflect meaningful behavioral differences. A low-dimensional projection of these embeddings that captures the most behaviorally significant dimensions is then used as behavioral descriptors for off-the-shelf QD methods. We prove that our embeddings converge to true MMD distances between occupancy measures as the number of sampled trajectories and embedding dimensions increase. Through experiments in multiple continuous control tasks we demonstrate AutoQD's ability in discovering diverse policies without predefined behavioral descriptors, presenting a well-motivated alternative to prior methods in unsupervised Reinforcement Learning and QD optimization. Our approach opens new possibilities for open-ended learning and automated behavior discovery in sequential decision making settings without requiring domain-specific knowledge. 

---
# GP-MoLFormer-Sim: Test Time Molecular Optimization through Contextual Similarity Guidance 

**Authors**: Jiri Navratil, Jarret Ross, Payel Das, Youssef Mroueh, Samuel C Hoffman, Vijil Chenthamarakshan, Brian Belgodere  

**Link**: [PDF](https://arxiv.org/pdf/2506.05628)  

**Abstract**: The ability to design molecules while preserving similarity to a target molecule and/or property is crucial for various applications in drug discovery, chemical design, and biology. We introduce in this paper an efficient training-free method for navigating and sampling from the molecular space with a generative Chemical Language Model (CLM), while using the molecular similarity to the target as a guide. Our method leverages the contextual representations learned from the CLM itself to estimate the molecular similarity, which is then used to adjust the autoregressive sampling strategy of the CLM. At each step of the decoding process, the method tracks the distance of the current generations from the target and updates the logits to encourage the preservation of similarity in generations. We implement the method using a recently proposed $\sim$47M parameter SMILES-based CLM, GP-MoLFormer, and therefore refer to the method as GP-MoLFormer-Sim, which enables a test-time update of the deep generative policy to reflect the contextual similarity to a set of guide molecules. The method is further integrated into a genetic algorithm (GA) and tested on a set of standard molecular optimization benchmarks involving property optimization, molecular rediscovery, and structure-based drug design. Results show that, GP-MoLFormer-Sim, combined with GA (GP-MoLFormer-Sim+GA) outperforms existing training-free baseline methods, when the oracle remains black-box. The findings in this work are a step forward in understanding and guiding the generative mechanisms of CLMs. 

---
# Deployability-Centric Infrastructure-as-Code Generation: An LLM-based Iterative Framework 

**Authors**: Tianyi Zhang, Shidong Pan, Zejun Zhang, Zhenchang Xing, Xiaoyu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.05623)  

**Abstract**: Infrastructure-as-Code (IaC) generation holds significant promise for automating cloud infrastructure provisioning. Recent advances in Large Language Models (LLMs) present a promising opportunity to democratize IaC development by generating deployable infrastructure templates from natural language descriptions, but current evaluation focuses on syntactic correctness while ignoring deployability, the fatal measure of IaC template utility. We address this gap through two contributions: (1) IaCGen, an LLM-based deployability-centric framework that uses iterative feedback mechanism to generate IaC templates, and (2) DPIaC-Eval, a deployability-centric IaC template benchmark consists of 153 real-world scenarios that can evaluate syntax, deployment, user intent, and security. Our evaluation reveals that state-of-the-art LLMs initially performed poorly, with Claude-3.5 and Claude-3.7 achieving only 30.2% and 26.8% deployment success on the first attempt respectively. However, IaCGen transforms this performance dramatically: all evaluated models reach over 90% passItr@25, with Claude-3.5 and Claude-3.7 achieving 98% success rate. Despite these improvements, critical challenges remain in user intent alignment (25.2% accuracy) and security compliance (8.4% pass rate), highlighting areas requiring continued research. Our work provides the first comprehensive assessment of deployability-centric IaC template generation and establishes a foundation for future research. 

---
# LFA applied to CNNs: Efficient Singular Value Decomposition of Convolutional Mappings by Local Fourier Analysis 

**Authors**: Antonia van Betteray, Matthias Rottmann, Karsten Kahl  

**Link**: [PDF](https://arxiv.org/pdf/2506.05617)  

**Abstract**: The singular values of convolutional mappings encode interesting spectral properties, which can be used, e.g., to improve generalization and robustness of convolutional neural networks as well as to facilitate model compression. However, the computation of singular values is typically very resource-intensive. The naive approach involves unrolling the convolutional mapping along the input and channel dimensions into a large and sparse two-dimensional matrix, making the exact calculation of all singular values infeasible due to hardware limitations. In particular, this is true for matrices that represent convolutional mappings with large inputs and a high number of channels. Existing efficient methods leverage the Fast Fourier transformation (FFT) to transform convolutional mappings into the frequency domain, enabling the computation of singular values for matrices representing convolutions with larger input and channel dimensions. For a constant number of channels in a given convolution, an FFT can compute N singular values in O(N log N) complexity. In this work, we propose an approach of complexity O(N) based on local Fourier analysis, which additionally exploits the shift invariance of convolutional operators. We provide a theoretical analysis of our algorithm's runtime and validate its efficiency through numerical experiments. Our results demonstrate that our proposed method is scalable and offers a practical solution to calculate the entire set of singular values - along with the corresponding singular vectors if needed - for high-dimensional convolutional mappings. 

---
# When Maximum Entropy Misleads Policy Optimization 

**Authors**: Ruipeng Zhang, Ya-Chien Chang, Sicun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.05615)  

**Abstract**: The Maximum Entropy Reinforcement Learning (MaxEnt RL) framework is a leading approach for achieving efficient learning and robust performance across many RL tasks. However, MaxEnt methods have also been shown to struggle with performance-critical control problems in practice, where non-MaxEnt algorithms can successfully learn. In this work, we analyze how the trade-off between robustness and optimality affects the performance of MaxEnt algorithms in complex control tasks: while entropy maximization enhances exploration and robustness, it can also mislead policy optimization, leading to failure in tasks that require precise, low-entropy policies. Through experiments on a variety of control problems, we concretely demonstrate this misleading effect. Our analysis leads to better understanding of how to balance reward design and entropy maximization in challenging control problems. 

---
# Scenarios in Computing Research: A Systematic Review of the Use of Scenario Methods for Exploring the Future of Computing Technologies in Society 

**Authors**: Julia Barnett, Kimon Kieslich, Jasmine Sinchai, Nicholas Diakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2506.05605)  

**Abstract**: Scenario building is an established method to anticipate the future of emerging technologies. Its primary goal is to use narratives to map future trajectories of technology development and sociotechnical adoption. Following this process, risks and benefits can be identified early on, and strategies can be developed that strive for desirable futures. In recent years, computer science has adopted this method and applied it to various technologies, including Artificial Intelligence (AI). Because computing technologies play such an important role in shaping modern societies, it is worth exploring how scenarios are being used as an anticipatory tool in the field -- and what possible traditional uses of scenarios are not yet covered but have the potential to enrich the field. We address this gap by conducting a systematic literature review on the use of scenario building methods in computer science over the last decade (n = 59). We guide the review along two main questions. First, we aim to uncover how scenarios are used in computing literature, focusing especially on the rationale for why scenarios are used. Second, in following the potential of scenario building to enhance inclusivity in research, we dive deeper into the participatory element of the existing scenario building literature in computer science. 

---
# SynthesizeMe! Inducing Persona-Guided Prompts for Personalized Reward Models in LLMs 

**Authors**: Michael J Ryan, Omar Shaikh, Aditri Bhagirath, Daniel Frees, William Held, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05598)  

**Abstract**: Recent calls for pluralistic alignment of Large Language Models (LLMs) encourage adapting models to diverse user preferences. However, most prior work on personalized reward models heavily rely on additional identity information, such as demographic details or a predefined set of preference categories. To this end, we introduce SynthesizeMe, an approach to inducing synthetic user personas from user interactions for personalized reward modeling. SynthesizeMe first generates and verifies reasoning to explain user preferences, then induces synthetic user personas from that reasoning, and finally filters to informative prior user interactions in order to build personalized prompts for a particular user. We show that using SynthesizeMe induced prompts improves personalized LLM-as-a-judge accuracy by 4.4% on Chatbot Arena. Combining SynthesizeMe derived prompts with a reward model achieves top performance on PersonalRewardBench: a new curation of user-stratified interactions with chatbots collected from 854 users of Chatbot Arena and PRISM. 

---
# Zero-shot protein stability prediction by inverse folding models: a free energy interpretation 

**Authors**: Jes Frellsen, Maher M. Kassem, Tone Bengtsen, Lars Olsen, Kresten Lindorff-Larsen, Jesper Ferkinghoff-Borg, Wouter Boomsma  

**Link**: [PDF](https://arxiv.org/pdf/2506.05596)  

**Abstract**: Inverse folding models have proven to be highly effective zero-shot predictors of protein stability. Despite this success, the link between the amino acid preferences of an inverse folding model and the free-energy considerations underlying thermodynamic stability remains incompletely understood. A better understanding would be of interest not only from a theoretical perspective, but also potentially provide the basis for stronger zero-shot stability prediction. In this paper, we take steps to clarify the free-energy foundations of inverse folding models. Our derivation reveals the standard practice of likelihood ratios as a simplistic approximation and suggests several paths towards better estimates of the relative stability. We empirically assess these approaches and demonstrate that considerable gains in zero-shot performance can be achieved with fairly simple means. 

---
# Improving Neural Diarization through Speaker Attribute Attractors and Local Dependency Modeling 

**Authors**: David Palzer, Matthew Maciejewski, Eric Fosler-Lussier  

**Link**: [PDF](https://arxiv.org/pdf/2506.05593)  

**Abstract**: In recent years, end-to-end approaches have made notable progress in addressing the challenge of speaker diarization, which involves segmenting and identifying speakers in multi-talker recordings. One such approach, Encoder-Decoder Attractors (EDA), has been proposed to handle variable speaker counts as well as better guide the network during training. In this study, we extend the attractor paradigm by moving beyond direct speaker modeling and instead focus on representing more detailed `speaker attributes' through a multi-stage process of intermediate representations. Additionally, we enhance the architecture by replacing transformers with conformers, a convolution-augmented transformer, to model local dependencies. Experiments demonstrate improved diarization performance on the CALLHOME dataset. 

---
# CoFrNets: Interpretable Neural Architecture Inspired by Continued Fractions 

**Authors**: Isha Puri, Amit Dhurandhar, Tejaswini Pedapati, Kartikeyan Shanmugam, Dennis Wei, Kush R. Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2506.05586)  

**Abstract**: In recent years there has been a considerable amount of research on local post hoc explanations for neural networks. However, work on building interpretable neural architectures has been relatively sparse. In this paper, we present a novel neural architecture, CoFrNet, inspired by the form of continued fractions which are known to have many attractive properties in number theory, such as fast convergence of approximations to real numbers. We show that CoFrNets can be efficiently trained as well as interpreted leveraging their particular functional form. Moreover, we prove that such architectures are universal approximators based on a proof strategy that is different than the typical strategy used to prove universal approximation results for neural networks based on infinite width (or depth), which is likely to be of independent interest. We experiment on nonlinear synthetic functions and are able to accurately model as well as estimate feature attributions and even higher order terms in some cases, which is a testament to the representational power as well as interpretability of such architectures. To further showcase the power of CoFrNets, we experiment on seven real datasets spanning tabular, text and image modalities, and show that they are either comparable or significantly better than other interpretable models and multilayer perceptrons, sometimes approaching the accuracies of state-of-the-art models. 

---
# Conformal Prediction Adaptive to Unknown Subpopulation Shifts 

**Authors**: Nien-Shao Wang, Duygu Nur Yaldiz, Yavuz Faruk Bakman, Sai Praneeth Karimireddy  

**Link**: [PDF](https://arxiv.org/pdf/2506.05583)  

**Abstract**: Conformal prediction is widely used to equip black-box machine learning models with uncertainty quantification enjoying formal coverage guarantees. However, these guarantees typically break down in the presence of distribution shifts, where the data distribution at test time differs from the training (or calibration-time) distribution. In this work, we address subpopulation shifts, where the test environment exhibits an unknown and differing mixture of subpopulations compared to the calibration data. We propose new methods that provably adapt conformal prediction to such shifts, ensuring valid coverage without requiring explicit knowledge of subpopulation structure. Our algorithms scale to high-dimensional settings and perform effectively in realistic machine learning tasks. Extensive experiments on vision (with vision transformers) and language (with large language models) benchmarks demonstrate that our methods reliably maintain coverage and controls risk in scenarios where standard conformal prediction fails. 

---
# Combating Misinformation in the Arab World: Challenges & Opportunities 

**Authors**: Azza Abouzied, Firoj Alam, Raian Ali, Paolo Papotti  

**Link**: [PDF](https://arxiv.org/pdf/2506.05582)  

**Abstract**: Misinformation and disinformation pose significant risks globally, with the Arab region facing unique vulnerabilities due to geopolitical instabilities, linguistic diversity, and cultural nuances. We explore these challenges through the key facets of combating misinformation: detection, tracking, mitigation and community-engagement. We shed light on how connecting with grass-roots fact-checking organizations, understanding cultural norms, promoting social correction, and creating strong collaborative information networks can create opportunities for a more resilient information ecosystem in the Arab world. 

---
# Collaborative Learning in Agentic Systems: A Collective AI is Greater Than the Sum of Its Parts 

**Authors**: Saptarshi Nath, Christos Peridis, Eseoghene Benjamin, Xinran Liu, Soheil Kolouri, Peter Kinnell, Zexin Li, Cong Liu, Shirin Dora, Andrea Soltoggio  

**Link**: [PDF](https://arxiv.org/pdf/2506.05577)  

**Abstract**: Agentic AI has gained significant interest as a research paradigm focused on autonomy, self-directed learning, and long-term reliability of decision making. Real-world agentic systems operate in decentralized settings on a large set of tasks or data distributions with constraints such as limited bandwidth, asynchronous execution, and the absence of a centralized model or even common objectives. We posit that exploiting previously learned skills, task similarities, and communication capabilities in a collective of agentic AI are challenging but essential elements to enabling scalability, open-endedness, and beneficial collaborative learning dynamics. In this paper, we introduce Modular Sharing and Composition in Collective Learning (MOSAIC), an agentic algorithm that allows multiple agents to independently solve different tasks while also identifying, sharing, and reusing useful machine-learned knowledge, without coordination, synchronization, or centralized control. MOSAIC combines three mechanisms: (1) modular policy composition via neural network masks, (2) cosine similarity estimation using Wasserstein embeddings for knowledge selection, and (3) asynchronous communication and policy integration. Results on a set of RL benchmarks show that MOSAIC has a greater sample efficiency than isolated learners, i.e., it learns significantly faster, and in some cases, finds solutions to tasks that cannot be solved by isolated learners. The collaborative learning and sharing dynamics are also observed to result in the emergence of ideal curricula of tasks, from easy to hard. These findings support the case for collaborative learning in agentic systems to achieve better and continuously evolving performance both at the individual and collective levels. 

---
# Ravan: Multi-Head Low-Rank Adaptation for Federated Fine-Tuning 

**Authors**: Arian Raje, Baris Askin, Divyansh Jhunjhunwala, Gauri Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2506.05568)  

**Abstract**: Large language models (LLMs) have not yet effectively leveraged the vast amounts of edge-device data, and federated learning (FL) offers a promising paradigm to collaboratively fine-tune LLMs without transferring private edge data to the cloud. To operate within the computation and communication constraints of edge devices, recent literature on federated fine-tuning of LLMs proposes the use of low-rank adaptation (LoRA) and similar parameter-efficient methods. However, LoRA-based methods suffer from accuracy degradation in FL settings, primarily because of data and computational heterogeneity across clients. We propose \textsc{Ravan}, an adaptive multi-head LoRA method that balances parameter efficiency and model expressivity by reparameterizing the weight updates as the sum of multiple LoRA heads $s_i\textbf{B}_i\textbf{H}_i\textbf{A}_i$ in which only the core matrices $\textbf{H}_i$ and their lightweight scaling factors $s_i$ are trained. These trainable scaling factors let the optimization focus on the most useful heads, recovering a higher-rank approximation of the full update without increasing the number of communicated parameters since clients upload $s_i\textbf{H}_i$ directly. Experiments on vision and language benchmarks show that \textsc{Ravan} improves test accuracy by 2-8\% over prior parameter-efficient baselines, making it a robust and scalable solution for federated fine-tuning of LLMs. 

---
# ScaleRTL: Scaling LLMs with Reasoning Data and Test-Time Compute for Accurate RTL Code Generation 

**Authors**: Chenhui Deng, Yun-Da Tsai, Guan-Ting Liu, Zhongzhi Yu, Haoxing Ren  

**Link**: [PDF](https://arxiv.org/pdf/2506.05566)  

**Abstract**: Recent advances in large language models (LLMs) have enabled near-human performance on software coding benchmarks, but their effectiveness in RTL code generation remains limited due to the scarcity of high-quality training data. While prior efforts have fine-tuned LLMs for RTL tasks, they do not fundamentally overcome the data bottleneck and lack support for test-time scaling due to their non-reasoning nature. In this work, we introduce ScaleRTL, the first reasoning LLM for RTL coding that scales up both high-quality reasoning data and test-time compute. Specifically, we curate a diverse set of long chain-of-thought reasoning traces averaging 56K tokens each, resulting in a dataset of 3.5B tokens that captures rich RTL knowledge. Fine-tuning a general-purpose reasoning model on this corpus yields ScaleRTL that is capable of deep RTL reasoning. Subsequently, we further enhance the performance of ScaleRTL through a novel test-time scaling strategy that extends the reasoning process via iteratively reflecting on and self-correcting previous reasoning steps. Experimental results show that ScaleRTL achieves state-of-the-art performance on VerilogEval and RTLLM, outperforming 18 competitive baselines by up to 18.4% on VerilogEval and 12.7% on RTLLM. 

---
# Applying Informer for Option Pricing: A Transformer-Based Approach 

**Authors**: Feliks Bańka, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2506.05565)  

**Abstract**: Accurate option pricing is essential for effective trading and risk management in financial markets, yet it remains challenging due to market volatility and the limitations of traditional models like Black-Scholes. In this paper, we investigate the application of the Informer neural network for option pricing, leveraging its ability to capture long-term dependencies and dynamically adjust to market fluctuations. This research contributes to the field of financial forecasting by introducing Informer's efficient architecture to enhance prediction accuracy and provide a more adaptable and resilient framework compared to existing methods. Our results demonstrate that Informer outperforms traditional approaches in option pricing, advancing the capabilities of data-driven financial forecasting in this domain. 

---
# MORSE-500: A Programmatically Controllable Video Benchmark to Stress-Test Multimodal Reasoning 

**Authors**: Zikui Cai, Andrew Wang, Anirudh Satheesh, Ankit Nakhawa, Hyunwoo Jae, Keenan Powell, Minghui Liu, Neel Jay, Sungbin Oh, Xiyao Wang, Yongyuan Liang, Tom Goldstein, Furong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05523)  

**Abstract**: Despite rapid advances in vision-language models (VLMs), current benchmarks for multimodal reasoning fall short in three key dimensions. First, they overwhelmingly rely on static images, failing to capture the temporal complexity of real-world environments. Second, they narrowly focus on mathematical problem-solving, neglecting the broader spectrum of reasoning skills -- including abstract, physical, planning, spatial, and temporal capabilities -- required for robust multimodal intelligence. Third, many benchmarks quickly saturate, offering limited headroom for diagnosing failure modes or measuring continued progress. We introduce MORSE-500 (Multimodal Reasoning Stress-test Environment), a video benchmark composed of 500 fully scripted clips with embedded questions spanning six complementary reasoning categories. Each instance is programmatically generated using deterministic Python scripts (via Manim, Matplotlib, MoviePy), generative video models, and curated real footage. This script-driven design allows fine-grained control over visual complexity, distractor density, and temporal dynamics -- enabling difficulty to be scaled systematically as models improve. Unlike static benchmarks that become obsolete once saturated, MORSE-500 is built to evolve: its controllable generation pipeline supports the creation of arbitrarily challenging new instances, making it ideally suited for stress-testing next-generation models. Initial experiments with state-of-the-art systems -- including various Gemini 2.5 Pro and OpenAI o3 which represent the strongest available at the time, alongside strong open-source models -- reveal substantial performance gaps across all categories, with particularly large deficits in abstract and planning tasks. We release the full dataset, generation scripts, and evaluation harness to support transparent, reproducible, and forward-looking multimodal reasoning research. 

---
# Learning to Recover: Dynamic Reward Shaping with Wheel-Leg Coordination for Fallen Robots 

**Authors**: Boyuan Deng, Luca Rossini, Jin Wang, Weijie Wang, Nikolaos Tsagarakis  

**Link**: [PDF](https://arxiv.org/pdf/2506.05516)  

**Abstract**: Adaptive recovery from fall incidents are essential skills for the practical deployment of wheeled-legged robots, which uniquely combine the agility of legs with the speed of wheels for rapid recovery. However, traditional methods relying on preplanned recovery motions, simplified dynamics or sparse rewards often fail to produce robust recovery policies. This paper presents a learning-based framework integrating Episode-based Dynamic Reward Shaping and curriculum learning, which dynamically balances exploration of diverse recovery maneuvers with precise posture refinement. An asymmetric actor-critic architecture accelerates training by leveraging privileged information in simulation, while noise-injected observations enhance robustness against uncertainties. We further demonstrate that synergistic wheel-leg coordination reduces joint torque consumption by 15.8% and 26.2% and improves stabilization through energy transfer mechanisms. Extensive evaluations on two distinct quadruped platforms achieve recovery success rates up to 99.1% and 97.8% without platform-specific tuning. The supplementary material is available at this https URL 

---
# Winner-takes-all for Multivariate Probabilistic Time Series Forecasting 

**Authors**: Adrien Cortés, Rémi Rehm, Victor Letzelter  

**Link**: [PDF](https://arxiv.org/pdf/2506.05515)  

**Abstract**: We introduce TimeMCL, a method leveraging the Multiple Choice Learning (MCL) paradigm to forecast multiple plausible time series futures. Our approach employs a neural network with multiple heads and utilizes the Winner-Takes-All (WTA) loss to promote diversity among predictions. MCL has recently gained attention due to its simplicity and ability to address ill-posed and ambiguous tasks. We propose an adaptation of this framework for time-series forecasting, presenting it as an efficient method to predict diverse futures, which we relate to its implicit quantization objective. We provide insights into our approach using synthetic data and evaluate it on real-world time series, demonstrating its promising performance at a light computational cost. 

---
# Beyond the Buzz: A Pragmatic Take on Inference Disaggregation 

**Authors**: Tiyasa Mitra, Ritika Borkar, Nidhi Bhatia, Ramon Matas, Shivam Raj, Dheevatsa Mudigere, Ritchie Zhao, Maximilian Golub, Arpan Dutta, Sailaja Madduri, Dharmesh Jani, Brian Pharris, Bita Darvish Rouhani  

**Link**: [PDF](https://arxiv.org/pdf/2506.05508)  

**Abstract**: As inference scales to multi-node deployments, disaggregation - splitting inference into distinct phases - offers a promising path to improving the throughput-interactivity Pareto frontier. Despite growing enthusiasm and a surge of open-source efforts, practical deployment of disaggregated serving remains limited due to the complexity of the optimization search space and system-level coordination. In this paper, we present the first systematic study of disaggregated inference at scale, evaluating hundreds of thousands of design points across diverse workloads and hardware configurations. We find that disaggregation is most effective for prefill-heavy traffic patterns and larger models. Our results highlight the critical role of dynamic rate matching and elastic scaling in achieving Pareto-optimal performance. Our findings offer actionable insights for efficient disaggregated deployments to navigate the trade-off between system throughput and interactivity. 

---
# StealthInk: A Multi-bit and Stealthy Watermark for Large Language Models 

**Authors**: Ya Jiang, Chuxiong Wu, Massieh Kordi Boroujeny, Brian Mark, Kai Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2506.05502)  

**Abstract**: Watermarking for large language models (LLMs) offers a promising approach to identifying AI-generated text. Existing approaches, however, either compromise the distribution of original generated text by LLMs or are limited to embedding zero-bit information that only allows for watermark detection but ignores identification. We present StealthInk, a stealthy multi-bit watermarking scheme that preserves the original text distribution while enabling the embedding of provenance data, such as userID, TimeStamp, and modelID, within LLM-generated text. This enhances fast traceability without requiring access to the language model's API or prompts. We derive a lower bound on the number of tokens necessary for watermark detection at a fixed equal error rate, which provides insights on how to enhance the capacity. Comprehensive empirical evaluations across diverse tasks highlight the stealthiness, detectability, and resilience of StealthInk, establishing it as an effective solution for LLM watermarking applications. 

---
# Conformal Prediction Beyond the Seen: A Missing Mass Perspective for Uncertainty Quantification in Generative Models 

**Authors**: Sima Noorani, Shayan Kiyani, George Pappas, Hamed Hassani  

**Link**: [PDF](https://arxiv.org/pdf/2506.05497)  

**Abstract**: Uncertainty quantification (UQ) is essential for safe deployment of generative AI models such as large language models (LLMs), especially in high stakes applications. Conformal prediction (CP) offers a principled uncertainty quantification framework, but classical methods focus on regression and classification, relying on geometric distances or softmax scores: tools that presuppose structured outputs. We depart from this paradigm by studying CP in a query only setting, where prediction sets must be constructed solely from finite queries to a black box generative model, introducing a new trade off between coverage, test time query budget, and informativeness. We introduce Conformal Prediction with Query Oracle (CPQ), a framework characterizing the optimal interplay between these objectives. Our finite sample algorithm is built on two core principles: one governs the optimal query policy, and the other defines the optimal mapping from queried samples to prediction sets. Remarkably, both are rooted in the classical missing mass problem in statistics. Specifically, the optimal query policy depends on the rate of decay, or the derivative, of the missing mass, for which we develop a novel estimator. Meanwhile, the optimal mapping hinges on the missing mass itself, which we estimate using Good Turing estimators. We then turn our focus to implementing our method for language models, where outputs are vast, variable, and often under specified. Fine grained experiments on three real world open ended tasks and two LLMs, show CPQ applicability to any black box LLM and highlight: (1) individual contribution of each principle to CPQ performance, and (2) CPQ ability to yield significantly more informative prediction sets than existing conformal methods for language uncertainty quantification. 

---
# Sentiment Analysis in Learning Management Systems Understanding Student Feedback at Scale 

**Authors**: Mohammed Almutairi  

**Link**: [PDF](https://arxiv.org/pdf/2506.05490)  

**Abstract**: During the wake of the Covid-19 pandemic, the educational paradigm has experienced a major change from in person learning traditional to online platforms. The change of learning convention has impacted the teacher-student especially in non-verbal communication. The absent of non-verbal communication has led to a reliance on verbal feedback which diminished the efficacy of the educational experience. This paper explores the integration of sentiment analysis into learning management systems (LMS) to bridge the student-teacher's gap by offering an alternative approach to interpreting student feedback beyond its verbal context. The research involves data preparation, feature selection, and the development of a deep neural network model encompassing word embedding, LSTM, and attention mechanisms. This model is compared against a logistic regression baseline to evaluate its efficacy in understanding student feedback. The study aims to bridge the communication gap between instructors and students in online learning environments, offering insights into the emotional context of student feedback and ultimately improving the quality of online education. 

---
# Zeroth-Order Optimization Finds Flat Minima 

**Authors**: Liang Zhang, Bingcong Li, Kiran Koshy Thekumparampil, Sewoong Oh, Michael Muehlebach, Niao He  

**Link**: [PDF](https://arxiv.org/pdf/2506.05454)  

**Abstract**: Zeroth-order methods are extensively used in machine learning applications where gradients are infeasible or expensive to compute, such as black-box attacks, reinforcement learning, and language model fine-tuning. Existing optimization theory focuses on convergence to an arbitrary stationary point, but less is known on the implicit regularization that provides a fine-grained characterization on which particular solutions are finally reached. We show that zeroth-order optimization with the standard two-point estimator favors solutions with small trace of Hessian, which is widely used in previous work to distinguish between sharp and flat minima. We further provide convergence rates of zeroth-order optimization to approximate flat minima for convex and sufficiently smooth functions, where flat minima are defined as the minimizers that achieve the smallest trace of Hessian among all optimal solutions. Experiments on binary classification tasks with convex losses and language model fine-tuning support our theoretical findings. 

---
# MLLM-CL: Continual Learning for Multimodal Large Language Models 

**Authors**: Hongbo Zhao, Fei Zhu, Rundong Wang, Gaofeng Meng, Zhaoxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05453)  

**Abstract**: Recent Multimodal Large Language Models (MLLMs) excel in vision-language understanding but face challenges in adapting to dynamic real-world scenarios that require continuous integration of new knowledge and skills. While continual learning (CL) offers a potential solution, existing benchmarks and methods suffer from critical limitations. In this paper, we introduce MLLM-CL, a novel benchmark encompassing domain and ability continual learning, where the former focuses on independently and identically distributed (IID) evaluation across evolving mainstream domains, whereas the latter evaluates on non-IID scenarios with emerging model ability. Methodologically, we propose preventing catastrophic interference through parameter isolation, along with an MLLM-based routing mechanism. Extensive experiments demonstrate that our approach can integrate domain-specific knowledge and functional abilities with minimal forgetting, significantly outperforming existing methods. 

---
# Interpretation Meets Safety: A Survey on Interpretation Methods and Tools for Improving LLM Safety 

**Authors**: Seongmin Lee, Aeree Cho, Grace C. Kim, ShengYun Peng, Mansi Phute, Duen Horng Chau  

**Link**: [PDF](https://arxiv.org/pdf/2506.05451)  

**Abstract**: As large language models (LLMs) see wider real-world use, understanding and mitigating their unsafe behaviors is critical. Interpretation techniques can reveal causes of unsafe outputs and guide safety, but such connections with safety are often overlooked in prior surveys. We present the first survey that bridges this gap, introducing a unified framework that connects safety-focused interpretation methods, the safety enhancements they inform, and the tools that operationalize them. Our novel taxonomy, organized by LLM workflow stages, summarizes nearly 70 works at their intersections. We conclude with open challenges and future directions. This timely survey helps researchers and practitioners navigate key advancements for safer, more interpretable LLMs. 

---
# Training Dynamics Underlying Language Model Scaling Laws: Loss Deceleration and Zero-Sum Learning 

**Authors**: Andrei Mircea, Supriyo Chakraborty, Nima Chitsazan, Irina Rish, Ekaterina Lobacheva  

**Link**: [PDF](https://arxiv.org/pdf/2506.05447)  

**Abstract**: This work aims to understand how scaling improves language models, specifically in terms of training dynamics. We find that language models undergo loss deceleration early in training; an abrupt slowdown in the rate of loss improvement, resulting in piecewise linear behaviour of the loss curve in log-log space. Scaling up the model mitigates this transition by (1) decreasing the loss at which deceleration occurs, and (2) improving the log-log rate of loss improvement after deceleration. We attribute loss deceleration to a type of degenerate training dynamics we term zero-sum learning (ZSL). In ZSL, per-example gradients become systematically opposed, leading to destructive interference in per-example changes in loss. As a result, improving loss on one subset of examples degrades it on another, bottlenecking overall progress. Loss deceleration and ZSL provide new insights into the training dynamics underlying language model scaling laws, and could potentially be targeted directly to improve language models independent of scale. We make our code and artefacts available at: this https URL 

---
# Sentinel: SOTA model to protect against prompt injections 

**Authors**: Dror Ivry, Oran Nahum  

**Link**: [PDF](https://arxiv.org/pdf/2506.05446)  

**Abstract**: Large Language Models (LLMs) are increasingly powerful but remain vulnerable to prompt injection attacks, where malicious inputs cause the model to deviate from its intended instructions. This paper introduces Sentinel, a novel detection model, qualifire/prompt-injection-sentinel, based on the \answerdotai/ModernBERT-large architecture. By leveraging ModernBERT's advanced features and fine-tuning on an extensive and diverse dataset comprising a few open-source and private collections, Sentinel achieves state-of-the-art performance. This dataset amalgamates varied attack types, from role-playing and instruction hijacking to attempts to generate biased content, alongside a broad spectrum of benign instructions, with private datasets specifically targeting nuanced error correction and real-world misclassifications. On a comprehensive, unseen internal test set, Sentinel demonstrates an average accuracy of 0.987 and an F1-score of 0.980. Furthermore, when evaluated on public benchmarks, it consistently outperforms strong baselines like protectai/deberta-v3-base-prompt-injection-v2. This work details Sentinel's architecture, its meticulous dataset curation, its training methodology, and a thorough evaluation, highlighting its superior detection capabilities. 

---
# Causal Policy Learning in Reinforcement Learning: Backdoor-Adjusted Soft Actor-Critic 

**Authors**: Thanh Vinh Vo, Young Lee, Haozhe Ma, Chien Lu, Tze-Yun Leong  

**Link**: [PDF](https://arxiv.org/pdf/2506.05445)  

**Abstract**: Hidden confounders that influence both states and actions can bias policy learning in reinforcement learning (RL), leading to suboptimal or non-generalizable behavior. Most RL algorithms ignore this issue, learning policies from observational trajectories based solely on statistical associations rather than causal effects. We propose DoSAC (Do-Calculus Soft Actor-Critic with Backdoor Adjustment), a principled extension of the SAC algorithm that corrects for hidden confounding via causal intervention estimation. DoSAC estimates the interventional policy $\pi(a | \mathrm{do}(s))$ using the backdoor criterion, without requiring access to true confounders or causal labels. To achieve this, we introduce a learnable Backdoor Reconstructor that infers pseudo-past variables (previous state and action) from the current state to enable backdoor adjustment from observational data. This module is integrated into a soft actor-critic framework to compute both the interventional policy and its entropy. Empirical results on continuous control benchmarks show that DoSAC outperforms baselines under confounded settings, with improved robustness, generalization, and policy reliability. 

---
# UniPTMs: The First Unified Multi-type PTM Site Prediction Model via Master-Slave Architecture-Based Multi-Stage Fusion Strategy and Hierarchical Contrastive Loss 

**Authors**: Yiyu Lin, Yan Wang, You Zhou, Xinye Ni, Jiahui Wu, Sen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05443)  

**Abstract**: As a core mechanism of epigenetic regulation in eukaryotes, protein post-translational modifications (PTMs) require precise prediction to decipher dynamic life activity networks. To address the limitations of existing deep learning models in cross-modal feature fusion, domain generalization, and architectural optimization, this study proposes UniPTMs: the first unified framework for multi-type PTM prediction. The framework innovatively establishes a "Master-Slave" dual-path collaborative architecture: The master path dynamically integrates high-dimensional representations of protein sequences, structures, and evolutionary information through a Bidirectional Gated Cross-Attention (BGCA) module, while the slave path optimizes feature discrepancies and recalibration between structural and traditional features using a Low-Dimensional Fusion Network (LDFN). Complemented by a Multi-scale Adaptive convolutional Pyramid (MACP) for capturing local feature patterns and a Bidirectional Hierarchical Gated Fusion Network (BHGFN) enabling multi-level feature integration across paths, the framework employs a Hierarchical Dynamic Weighting Fusion (HDWF) mechanism to intelligently aggregate multimodal features. Enhanced by a novel Hierarchical Contrastive loss function for feature consistency optimization, UniPTMs demonstrates significant performance improvements (3.2%-11.4% MCC and 4.2%-14.3% AP increases) over state-of-the-art models across five modification types and transcends the Single-Type Prediction Paradigm. To strike a balance between model complexity and performance, we have also developed a lightweight variant named UniPTMs-mini. 

---
# Structured Labeling Enables Faster Vision-Language Models for End-to-End Autonomous Driving 

**Authors**: Hao Jiang, Chuan Hu, Yukang Shi, Yuan He, Ke Wang, Xi Zhang, Zhipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05442)  

**Abstract**: Vision-Language Models (VLMs) offer a promising approach to end-to-end autonomous driving due to their human-like reasoning capabilities. However, troublesome gaps remains between current VLMs and real-world autonomous driving applications. One major limitation is that existing datasets with loosely formatted language descriptions are not machine-friendly and may introduce redundancy. Additionally, high computational cost and massive scale of VLMs hinder the inference speed and real-world deployment. To bridge the gap, this paper introduces a structured and concise benchmark dataset, NuScenes-S, which is derived from the NuScenes dataset and contains machine-friendly structured representations. Moreover, we present FastDrive, a compact VLM baseline with 0.9B parameters. In contrast to existing VLMs with over 7B parameters and unstructured language processing(e.g., LLaVA-1.5), FastDrive understands structured and concise descriptions and generates machine-friendly driving decisions with high efficiency. Extensive experiments show that FastDrive achieves competitive performance on structured dataset, with approximately 20% accuracy improvement on decision-making tasks, while surpassing massive parameter baseline in inference speed with over 10x speedup. Additionally, ablation studies further focus on the impact of scene annotations (e.g., weather, time of day) on decision-making tasks, demonstrating their importance on decision-making tasks in autonomous driving. 

---
# BYO-Eval: Build Your Own Dataset for Fine-Grained Visual Assessment of Multimodal Language Models 

**Authors**: Ludovic Arnould, Salim Khazem, Hugues Ali Mehenni  

**Link**: [PDF](https://arxiv.org/pdf/2506.05440)  

**Abstract**: Visual Language Models (VLMs) are now sufficiently advanced to support a broad range of applications, including answering complex visual questions, and are increasingly expected to interact with images in varied ways. To evaluate them, current benchmarks often focus on specific domains (e.g., reading charts), constructing datasets of annotated real images paired with pre-defined Multiple Choice Questions (MCQs) to report aggregate accuracy scores. However, such benchmarks entail high annotation costs, risk information leakage, and do not clarify whether failures stem from limitations in visual perception, reasoning, or general knowledge. We propose a new evaluation methodology, inspired by ophthalmologic diagnostics, leveraging procedural generation of synthetic images to obtain control over visual attributes and precisely reveal perception failures in VLMs. Specifically, we build collections of images with gradually more challenging variations in the content of interest (e.g., number of objects in a counting task) while holding other visual parameters constant. This diagnostic allows systematic stress testing and fine-grained failure analysis, shifting the focus from coarse benchmarking toward targeted and interpretable assessment of VLM capabilities. Our code is available at this https URL. 

---
# LLMs Can Compensate for Deficiencies in Visual Representations 

**Authors**: Sho Takishita, Jay Gala, Abdelrahman Mohamed, Kentaro Inui, Yova Kementchedjhieva  

**Link**: [PDF](https://arxiv.org/pdf/2506.05439)  

**Abstract**: Many vision-language models (VLMs) that prove very effective at a range of multimodal task, build on CLIP-based vision encoders, which are known to have various limitations. We investigate the hypothesis that the strong language backbone in VLMs compensates for possibly weak visual features by contextualizing or enriching them. Using three CLIP-based VLMs, we perform controlled self-attention ablations on a carefully designed probing task. Our findings show that despite known limitations, CLIP visual representations offer ready-to-read semantic information to the language decoder. However, in scenarios of reduced contextualization in the visual representations, the language decoder can largely compensate for the deficiency and recover performance. This suggests a dynamic division of labor in VLMs and motivates future architectures that offload more visual processing to the language decoder. 

---
# An Unsupervised Framework for Dynamic Health Indicator Construction and Its Application in Rolling Bearing Prognostics 

**Authors**: Tongda Sun, Chen Yin, Huailiang Zheng, Yining Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.05438)  

**Abstract**: Health indicator (HI) plays a key role in degradation assessment and prognostics of rolling bearings. Although various HI construction methods have been investigated, most of them rely on expert knowledge for feature extraction and overlook capturing dynamic information hidden in sequential degradation processes, which limits the ability of the constructed HI for degradation trend representation and prognostics. To address these concerns, a novel dynamic HI that considers HI-level temporal dependence is constructed through an unsupervised framework. Specifically, a degradation feature learning module composed of a skip-connection-based autoencoder first maps raw signals to a representative degradation feature space (DFS) to automatically extract essential degradation features without the need for expert knowledge. Subsequently, in this DFS, a new HI-generating module embedded with an inner HI-prediction block is proposed for dynamic HI construction, where the temporal dependence between past and current HI states is guaranteed and modeled explicitly. On this basis, the dynamic HI captures the inherent dynamic contents of the degradation process, ensuring its effectiveness for degradation tendency modeling and future degradation prognostics. The experiment results on two bearing lifecycle datasets demonstrate that the proposed HI construction method outperforms comparison methods, and the constructed dynamic HI is superior for prognostic tasks. 

---
# A MARL-based Approach for Easing MAS Organization Engineering 

**Authors**: Julien Soulé, Jean-Paul Jamont, Michel Occello, Louis-Marie Traonouez, Paul Théron  

**Link**: [PDF](https://arxiv.org/pdf/2506.05437)  

**Abstract**: Multi-Agent Systems (MAS) have been successfully applied in industry for their ability to address complex, distributed problems, especially in IoT-based systems. Their efficiency in achieving given objectives and meeting design requirements is strongly dependent on the MAS organization during the engineering process of an application-specific MAS. To design a MAS that can achieve given goals, available methods rely on the designer's knowledge of the deployment environment. However, high complexity and low readability in some deployment environments make the application of these methods to be costly or raise safety concerns. In order to ease the MAS organization design regarding those concerns, we introduce an original Assisted MAS Organization Engineering Approach (AOMEA). AOMEA relies on combining a Multi-Agent Reinforcement Learning (MARL) process with an organizational model to suggest relevant organizational specifications to help in MAS engineering. 

---
# Event Classification of Accelerometer Data for Industrial Package Monitoring with Embedded Deep Learning 

**Authors**: Manon Renault, Hamoud Younes, Hugo Tessier, Ronan Le Roy, Bastien Pasdeloup, Mathieu Léonardon  

**Link**: [PDF](https://arxiv.org/pdf/2506.05435)  

**Abstract**: Package monitoring is an important topic in industrial applications, with significant implications for operational efficiency and ecological sustainability. In this study, we propose an approach that employs an embedded system, placed on reusable packages, to detect their state (on a Forklift, in a Truck, or in an undetermined location). We aim to design a system with a lifespan of several years, corresponding to the lifespan of reusable packages. Our analysis demonstrates that maximizing device lifespan requires minimizing wake time. We propose a pipeline that includes data processing, training, and evaluation of the deep learning model designed for imbalanced, multiclass time series data collected from an embedded sensor. The method uses a one-dimensional Convolutional Neural Network architecture to classify accelerometer data from the IoT device. Before training, two data augmentation techniques are tested to solve the imbalance problem of the dataset: the Synthetic Minority Oversampling TEchnique and the ADAptive SYNthetic sampling approach. After training, compression techniques are implemented to have a small model size. On the considered twoclass problem, the methodology yields a precision of 94.54% for the first class and 95.83% for the second class, while compression techniques reduce the model size by a factor of four. The trained model is deployed on the IoT device, where it operates with a power consumption of 316 mW during inference. 

---
# Efficient Robust Conformal Prediction via Lipschitz-Bounded Networks 

**Authors**: Thomas Massena, Léo andéol, Thibaut Boissin, Franck Mamalet, Corentin Friedrich, Mathieu Serrurier, Sébastien Gerchinovitz  

**Link**: [PDF](https://arxiv.org/pdf/2506.05434)  

**Abstract**: Conformal Prediction (CP) has proven to be an effective post-hoc method for improving the trustworthiness of neural networks by providing prediction sets with finite-sample guarantees. However, under adversarial attacks, classical conformal guarantees do not hold anymore: this problem is addressed in the field of Robust Conformal Prediction. Several methods have been proposed to provide robust CP sets with guarantees under adversarial perturbations, but, for large scale problems, these sets are either too large or the methods are too computationally demanding to be deployed in real life scenarios. In this work, we propose a new method that leverages Lipschitz-bounded networks to precisely and efficiently estimate robust CP sets. When combined with a 1-Lipschitz robust network, we demonstrate that our lip-rcp method outperforms state-of-the-art results in both the size of the robust CP sets and computational efficiency in medium and large-scale scenarios such as ImageNet. Taking a different angle, we also study vanilla CP under attack, and derive new worst-case coverage bounds of vanilla CP sets, which are valid simultaneously for all adversarial attack levels. Our lip-rcp method makes this second approach as efficient as vanilla CP while also allowing robustness guarantees. 

---
# Prefix Grouper: Efficient GRPO Training through Shared-Prefix Forward 

**Authors**: Zikang Liu, Tongtian Yue, Yepeng Tang, Longteng Guo, Junxian Cai, Qingbin Liu, Xi Chen, Jing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.05433)  

**Abstract**: Group Relative Policy Optimization (GRPO) enhances policy learning by computing gradients from relative comparisons among candidate outputs that share a common input prefix. Despite its effectiveness, GRPO introduces substantial computational overhead when processing long shared prefixes, which must be redundantly encoded for each group member. This inefficiency becomes a major scalability bottleneck in long-context learning scenarios. We propose Prefix Grouper, an efficient GRPO training algorithm that eliminates redundant prefix computation via a Shared-Prefix Forward strategy. In particular, by restructuring self-attention into two parts, our method enables the shared prefix to be encoded only once, while preserving full differentiability and compatibility with end-to-end training. We provide both theoretical and empirical evidence that Prefix Grouper is training-equivalent to standard GRPO: it yields identical forward outputs and backward gradients, ensuring that the optimization dynamics and final policy performance remain unchanged. Empirically, our experiments confirm that Prefix Grouper achieves consistent results while significantly reducing the computational cost of training, particularly in long-prefix scenarios. The proposed method is fully plug-and-play: it is compatible with existing GRPO-based architectures and can be seamlessly integrated into current training pipelines as a drop-in replacement, requiring no structural modifications and only minimal changes to input construction and attention computation. Prefix Grouper enables the use of larger group sizes under the same computational budget, thereby improving the scalability of GRPO to more complex tasks and larger models. Code is now available at this https URL 

---
# PCDVQ: Enhancing Vector Quantization for Large Language Models via Polar Coordinate Decoupling 

**Authors**: Yuxuan Yue, Zukang Xu, Zhihang Yuan, Dawei Yang, Jianglong Wu, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2506.05432)  

**Abstract**: Large Language Models (LLMs) face significant challenges in edge deployment due to their massive parameter scale. Vector Quantization (VQ), a clustering-based quantization method, serves as a prevalent solution to this issue for its extremely low-bit (even at 2-bit) and considerable accuracy. Since a vector is a quantity in mathematics and physics that has both direction and magnitude, existing VQ works typically quantize them in a coupled manner. However, we find that direction exhibits significantly greater sensitivity to quantization compared to the magnitude. For instance, when separately clustering the directions and magnitudes of weight vectors in LLaMA-2-7B, the accuracy drop of zero-shot tasks are 46.5\% and 2.3\%, respectively. This gap even increases with the reduction of clustering centers. Further, Euclidean distance, a common metric to access vector similarities in current VQ works, places greater emphasis on reducing the magnitude error. This property is contrary to the above finding, unavoidably leading to larger quantization errors. To these ends, this paper proposes Polar Coordinate Decoupled Vector Quantization (PCDVQ), an effective and efficient VQ framework consisting of two key modules: 1) Polar Coordinate Decoupling (PCD), which transforms vectors into their polar coordinate representations and perform independent quantization of the direction and magnitude parameters.2) Distribution Aligned Codebook Construction (DACC), which optimizes the direction and magnitude codebooks in accordance with the source distribution. Experimental results show that PCDVQ outperforms baseline methods at 2-bit level by at least 1.5\% zero-shot accuracy, establishing a novel paradigm for accurate and highly compressed LLMs. 

---
# Robustness Evaluation for Video Models with Reinforcement Learning 

**Authors**: Ashwin Ramesh Babu, Sajad Mousavi, Vineet Gundecha, Sahand Ghorbanpour, Avisek Naug, Antonio Guillen, Ricardo Luna Gutierrez, Soumyendu Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2506.05431)  

**Abstract**: Evaluating the robustness of Video classification models is very challenging, specifically when compared to image-based models. With their increased temporal dimension, there is a significant increase in complexity and computational cost. One of the key challenges is to keep the perturbations to a minimum to induce misclassification. In this work, we propose a multi-agent reinforcement learning approach (spatial and temporal) that cooperatively learns to identify the given video's sensitive spatial and temporal regions. The agents consider temporal coherence in generating fine perturbations, leading to a more effective and visually imperceptible attack. Our method outperforms the state-of-the-art solutions on the Lp metric and the average queries. Our method enables custom distortion types, making the robustness evaluation more relevant to the use case. We extensively evaluate 4 popular models for video action recognition on two popular datasets, HMDB-51 and UCF-101. 

---
# Explainer-guided Targeted Adversarial Attacks against Binary Code Similarity Detection Models 

**Authors**: Mingjie Chen, Tiancheng Zhu, Mingxue Zhang, Yiling He, Minghao Lin, Penghui Li, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2506.05430)  

**Abstract**: Binary code similarity detection (BCSD) serves as a fundamental technique for various software engineering tasks, e.g., vulnerability detection and classification. Attacks against such models have therefore drawn extensive attention, aiming at misleading the models to generate erroneous predictions. Prior works have explored various approaches to generating semantic-preserving variants, i.e., adversarial samples, to evaluate the robustness of the models against adversarial attacks. However, they have mainly relied on heuristic criteria or iterative greedy algorithms to locate salient code influencing the model output, failing to operate on a solid theoretical basis. Moreover, when processing programs with high complexities, such attacks tend to be time-consuming.
In this work, we propose a novel optimization for adversarial attacks against BCSD models. In particular, we aim to improve the attacks in a challenging scenario, where the attack goal is to limit the model predictions to a specific range, i.e., the targeted attacks. Our attack leverages the superior capability of black-box, model-agnostic explainers in interpreting the model decision boundaries, thereby pinpointing the critical code snippet to apply semantic-preserving perturbations. The evaluation results demonstrate that compared with the state-of-the-art attacks, the proposed attacks achieve higher attack success rate in almost all scenarios, while also improving the efficiency and transferability. Our real-world case studies on vulnerability detection and classification further demonstrate the security implications of our attacks, highlighting the urgent need to further enhance the robustness of existing BCSD models. 

---
# Coordinated Robustness Evaluation Framework for Vision-Language Models 

**Authors**: Ashwin Ramesh Babu, Sajad Mousavi, Vineet Gundecha, Sahand Ghorbanpour, Avisek Naug, Antonio Guillen, Ricardo Luna Gutierrez, Soumyendu Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2506.05429)  

**Abstract**: Vision-language models, which integrate computer vision and natural language processing capabilities, have demonstrated significant advancements in tasks such as image captioning and visual question and answering. However, similar to traditional models, they are susceptible to small perturbations, posing a challenge to their robustness, particularly in deployment scenarios. Evaluating the robustness of these models requires perturbations in both the vision and language modalities to learn their inter-modal dependencies. In this work, we train a generic surrogate model that can take both image and text as input and generate joint representation which is further used to generate adversarial perturbations for both the text and image modalities. This coordinated attack strategy is evaluated on the visual question and answering and visual reasoning datasets using various state-of-the-art vision-language models. Our results indicate that the proposed strategy outperforms other multi-modal attacks and single-modality attacks from the recent literature. Our results demonstrate their effectiveness in compromising the robustness of several state-of-the-art pre-trained multi-modal models such as instruct-BLIP, ViLT and others. 

---
# Diffusion with a Linguistic Compass: Steering the Generation of Clinically Plausible Future sMRI Representations for Early MCI Conversion Prediction 

**Authors**: Zhihao Tang, Chaozhuo Li, Litian Zhang, Xi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05428)  

**Abstract**: Early prediction of Mild Cognitive Impairment (MCI) conversion is hampered by a trade-off between immediacy--making fast predictions from a single baseline sMRI--and accuracy--leveraging longitudinal scans to capture disease progression. We propose MCI-Diff, a diffusion-based framework that synthesizes clinically plausible future sMRI representations directly from baseline data, achieving both real-time risk assessment and high predictive performance. First, a multi-task sequence reconstruction strategy trains a shared denoising network on interpolation and extrapolation tasks to handle irregular follow-up sampling and learn robust latent trajectories. Second, an LLM-driven "linguistic compass" is introduced for clinical plausibility sampling: generated feature candidates are quantized, tokenized, and scored by a fine-tuned language model conditioned on expected structural biomarkers, guiding autoregressive generation toward realistic disease patterns. Experiments on ADNI and AIBL cohorts show that MCI-Diff outperforms state-of-the-art baselines, improving early conversion accuracy by 5-12%. 

---
# MTPNet: Multi-Grained Target Perception for Unified Activity Cliff Prediction 

**Authors**: Zishan Shu, Yufan Deng, Hongyu Zhang, Zhiwei Nie, Jie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.05427)  

**Abstract**: Activity cliff prediction is a critical task in drug discovery and material design. Existing computational methods are limited to handling single binding targets, which restricts the applicability of these prediction models. In this paper, we present the Multi-Grained Target Perception network (MTPNet) to incorporate the prior knowledge of interactions between the molecules and their target proteins. Specifically, MTPNet is a unified framework for activity cliff prediction, which consists of two components: Macro-level Target Semantic (MTS) guidance and Micro-level Pocket Semantic (MPS) guidance. By this way, MTPNet dynamically optimizes molecular representations through multi-grained protein semantic conditions. To our knowledge, it is the first time to employ the receptor proteins as guiding information to effectively capture critical interaction details. Extensive experiments on 30 representative activity cliff datasets demonstrate that MTPNet significantly outperforms previous approaches, achieving an average RMSE improvement of 18.95% on top of several mainstream GNN architectures. Overall, MTPNet internalizes interaction patterns through conditional deep learning to achieve unified predictions of activity cliffs, helping to accelerate compound optimization and design. Codes are available at: this https URL. 

---
# Mixture-of-Experts Meets In-Context Reinforcement Learning 

**Authors**: Wenhao Wu, Fuhong Liu, Haoru Li, Zican Hu, Daoyi Dong, Chunlin Chen, Zhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05426)  

**Abstract**: In-context reinforcement learning (ICRL) has emerged as a promising paradigm for adapting RL agents to downstream tasks through prompt conditioning. However, two notable challenges remain in fully harnessing in-context learning within RL domains: the intrinsic multi-modality of the state-action-reward data and the diverse, heterogeneous nature of decision tasks. To tackle these challenges, we propose \textbf{T2MIR} (\textbf{T}oken- and \textbf{T}ask-wise \textbf{M}oE for \textbf{I}n-context \textbf{R}L), an innovative framework that introduces architectural advances of mixture-of-experts (MoE) into transformer-based decision models. T2MIR substitutes the feedforward layer with two parallel layers: a token-wise MoE that captures distinct semantics of input tokens across multiple modalities, and a task-wise MoE that routes diverse tasks to specialized experts for managing a broad task distribution with alleviated gradient conflicts. To enhance task-wise routing, we introduce a contrastive learning method that maximizes the mutual information between the task and its router representation, enabling more precise capture of task-relevant information. The outputs of two MoE components are concatenated and fed into the next layer. Comprehensive experiments show that T2MIR significantly facilitates in-context learning capacity and outperforms various types of baselines. We bring the potential and promise of MoE to ICRL, offering a simple and scalable architectural enhancement to advance ICRL one step closer toward achievements in language and vision communities. Our code is available at this https URL. 

---
# SIV-Bench: A Video Benchmark for Social Interaction Understanding and Reasoning 

**Authors**: Fanqi Kong, Weiqin Zu, Xinyu Chen, Yaodong Yang, Song-Chun Zhu, Xue Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.05425)  

**Abstract**: The rich and multifaceted nature of human social interaction, encompassing multimodal cues, unobservable relations and mental states, and dynamical behavior, presents a formidable challenge for artificial intelligence. To advance research in this area, we introduce SIV-Bench, a novel video benchmark for rigorously evaluating the capabilities of Multimodal Large Language Models (MLLMs) across Social Scene Understanding (SSU), Social State Reasoning (SSR), and Social Dynamics Prediction (SDP). SIV-Bench features 2,792 video clips and 8,792 meticulously generated question-answer pairs derived from a human-LLM collaborative pipeline. It is originally collected from TikTok and YouTube, covering a wide range of video genres, presentation styles, and linguistic and cultural backgrounds. It also includes a dedicated setup for analyzing the impact of different textual cues-original on-screen text, added dialogue, or no text. Our comprehensive experiments on leading MLLMs reveal that while models adeptly handle SSU, they significantly struggle with SSR and SDP, where Relation Inference (RI) is an acute bottleneck, as further examined in our analysis. Our study also confirms the critical role of transcribed dialogue in aiding comprehension of complex social interactions. By systematically identifying current MLLMs' strengths and limitations, SIV-Bench offers crucial insights to steer the development of more socially intelligent AI. The dataset and code are available at this https URL. 

---
# Dream to Generalize: Zero-Shot Model-Based Reinforcement Learning for Unseen Visual Distractions 

**Authors**: Jeongsoo Ha, Kyungsoo Kim, Yusung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.05419)  

**Abstract**: Model-based reinforcement learning (MBRL) has been used to efficiently solve vision-based control tasks in highdimensional image observations. Although recent MBRL algorithms perform well in trained observations, they fail when faced with visual distractions in observations. These task-irrelevant distractions (e.g., clouds, shadows, and light) may be constantly present in real-world scenarios. In this study, we propose a novel self-supervised method, Dream to Generalize (Dr. G), for zero-shot MBRL. Dr. G trains its encoder and world model with dual contrastive learning which efficiently captures task-relevant features among multi-view data augmentations. We also introduce a recurrent state inverse dynamics model that helps the world model to better understand the temporal structure. The proposed methods can enhance the robustness of the world model against visual distractions. To evaluate the generalization performance, we first train Dr. G on simple backgrounds and then test it on complex natural video backgrounds in the DeepMind Control suite, and the randomizing environments in Robosuite. Dr. G yields a performance improvement of 117% and 14% over prior works, respectively. Our code is open-sourced and available at this https URL 

---
# Self-Predictive Dynamics for Generalization of Vision-based Reinforcement Learning 

**Authors**: Kyungsoo Kim, Jeongsoo Ha, Yusung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.05418)  

**Abstract**: Vision-based reinforcement learning requires efficient and robust representations of image-based observations, especially when the images contain distracting (task-irrelevant) elements such as shadows, clouds, and light. It becomes more important if those distractions are not exposed during training. We design a Self-Predictive Dynamics (SPD) method to extract task-relevant features efficiently, even in unseen observations after training. SPD uses weak and strong augmentations in parallel, and learns representations by predicting inverse and forward transitions across the two-way augmented versions. In a set of MuJoCo visual control tasks and an autonomous driving task (CARLA), SPD outperforms previous studies in complex observations, and significantly improves the generalization performance for unseen observations. Our code is available at this https URL. 

---
# FERRET: Private Deep Learning Faster And Better Than DPSGD 

**Authors**: David Zagardo  

**Link**: [PDF](https://arxiv.org/pdf/2506.05416)  

**Abstract**: We revisit 1-bit gradient compression through the lens of mutual-information differential privacy (MI-DP). Building on signSGD, we propose FERRET--Fast and Effective Restricted Release for Ethical Training--which transmits at most one sign bit per parameter group with Bernoulli masking.
Theory: We prove each fired group leaks at most ln 2 nats; after subsampling with rate s, the total privacy loss of G groups trained for T steps with firing probability p is epsilon = G * T * s * p * ln 2. Thus FERRET achieves MI-DP for epsilon in [0.1, 2] without additive noise.
Practice: We evaluate three granularities--FERRET-MAX (finest), FERRET-EIGHTH (medium), and FERRET-2 (coarsest)--on five LLMs (137M-1.8B parameters) against DPSGD and Non-DP baselines. All methods trained for 1, 3, and 5 epochs.
Utility: Across all settings, FERRET-MAX/EIGHTH beat DPSGD's perplexity. At epsilon=0.5, 5 epochs: FERRET-EIGHTH achieves 3.98 perplexity vs DPSGD's 11.61 (2.9x better), within 23% of Non-DP (3.25).
Privacy: MI-AUC stays at chance for FERRET-MAX/EIGHTH (~0.51), matching DPSGD vs Non-DP's 0.76-0.99. FERRET-2 shows higher leakage (~0.55) due to lower headroom.
Efficiency: Stricter budgets fire fewer signs, so FERRET uses 19-33% of DPSGD's training time and only 34-36% of Non-DP training time.
Take-away: Sign-based MI-DP gets closer to achieving all three qualities of the privacy, utility, performance trilemma: FERRET trains up to 5x faster, achieves 3x lower perplexity compared to DPSGD and 1.2x greater than Non-DP, all while providing formal, mathematically provable privacy guarantees using zero additive noise. The results also show that, in certain instances, masked 1-bit updates can match non-private training utility while safeguarding data. 

---
# SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing 

**Authors**: Mingfei Chen, Zijun Cui, Xiulong Liu, Jinlin Xiang, Caleb Zheng, Jingyuan Li, Eli Shlizerman  

**Link**: [PDF](https://arxiv.org/pdf/2506.05414)  

**Abstract**: 3D spatial reasoning in dynamic, audio-visual environments is a cornerstone of human cognition yet remains largely unexplored by existing Audio-Visual Large Language Models (AV-LLMs) and benchmarks, which predominantly focus on static or 2D scenes. We introduce SAVVY-Bench, the first benchmark for 3D spatial reasoning in dynamic scenes with synchronized spatial audio. SAVVY-Bench is comprised of thousands of relationships involving static and moving objects, and requires fine-grained temporal grounding, consistent 3D localization, and multi-modal annotation. To tackle this challenge, we propose SAVVY, a novel training-free reasoning pipeline that consists of two stages: (i) Egocentric Spatial Tracks Estimation, which leverages AV-LLMs as well as other audio-visual methods to track the trajectories of key objects related to the query using both visual and spatial audio cues, and (ii) Dynamic Global Map Construction, which aggregates multi-modal queried object trajectories and converts them into a unified global dynamic map. Using the constructed map, a final QA answer is obtained through a coordinate transformation that aligns the global map with the queried viewpoint. Empirical evaluation demonstrates that SAVVY substantially enhances performance of state-of-the-art AV-LLMs, setting a new standard and stage for approaching dynamic 3D spatial reasoning in AV-LLMs. 

---
# SmoothRot: Combining Channel-Wise Scaling and Rotation for Quantization-Friendly LLMs 

**Authors**: Patrik Czakó, Gábor Kertész, Sándor Szénási  

**Link**: [PDF](https://arxiv.org/pdf/2506.05413)  

**Abstract**: We present SmoothRot, a novel post-training quantization technique to enhance the efficiency of 4-bit quantization in Large Language Models (LLMs). SmoothRot addresses the critical challenge of massive activation outliers, by integrating channel-wise scaling with Hadamard transformations. Our technique effectively transforms extreme outliers into quantization-friendly activations, significantly improving quantization accuracy. Experiments conducted on popular LLMs (LLaMA2 7B, LLaMA3.1 8B, and Mistral 7B) demonstrate that SmoothRot consistently reduces the performance gap between quantized and FP16 models by approximately 10-30\% across language generation and zero-shot reasoning tasks, without introducing additional inference latency. Code is available at this https URL. 

---
# AD-EE: Early Exiting for Fast and Reliable Vision-Language Models in Autonomous Driving 

**Authors**: Lianming Huang, Haibo Hu, Yufei Cui, Jiacheng Zuo, Shangyu Wu, Nan Guan, Chun Jason Xue  

**Link**: [PDF](https://arxiv.org/pdf/2506.05404)  

**Abstract**: With the rapid advancement of autonomous driving, deploying Vision-Language Models (VLMs) to enhance perception and decision-making has become increasingly common. However, the real-time application of VLMs is hindered by high latency and computational overhead, limiting their effectiveness in time-critical driving scenarios. This challenge is particularly evident when VLMs exhibit over-inference, continuing to process unnecessary layers even after confident predictions have been reached. To address this inefficiency, we propose AD-EE, an Early Exit framework that incorporates domain characteristics of autonomous driving and leverages causal inference to identify optimal exit layers. We evaluate our method on large-scale real-world autonomous driving datasets, including Waymo and the corner-case-focused CODA, as well as on a real vehicle running the Autoware Universe platform. Extensive experiments across multiple VLMs show that our method significantly reduces latency, with maximum improvements reaching up to 57.58%, and enhances object detection accuracy, with maximum gains of up to 44%. 

---
# Attention-based transformer models for image captioning across languages: An in-depth survey and evaluation 

**Authors**: Israa A. Albadarneh, Bassam H. Hammo, Omar S. Al-Kadi  

**Link**: [PDF](https://arxiv.org/pdf/2506.05399)  

**Abstract**: Image captioning involves generating textual descriptions from input images, bridging the gap between computer vision and natural language processing. Recent advancements in transformer-based models have significantly improved caption generation by leveraging attention mechanisms for better scene understanding. While various surveys have explored deep learning-based approaches for image captioning, few have comprehensively analyzed attention-based transformer models across multiple languages. This survey reviews attention-based image captioning models, categorizing them into transformer-based, deep learning-based, and hybrid approaches. It explores benchmark datasets, discusses evaluation metrics such as BLEU, METEOR, CIDEr, and ROUGE, and highlights challenges in multilingual captioning. Additionally, this paper identifies key limitations in current models, including semantic inconsistencies, data scarcity in non-English languages, and limitations in reasoning ability. Finally, we outline future research directions, such as multimodal learning, real-time applications in AI-powered assistants, healthcare, and forensic analysis. This survey serves as a comprehensive reference for researchers aiming to advance the field of attention-based image captioning. 

---
# Gen4D: Synthesizing Humans and Scenes in the Wild 

**Authors**: Jerrin Bright, Zhibo Wang, Yuhao Chen, Sirisha Rambhatla, John Zelek, David Clausi  

**Link**: [PDF](https://arxiv.org/pdf/2506.05397)  

**Abstract**: Lack of input data for in-the-wild activities often results in low performance across various computer vision tasks. This challenge is particularly pronounced in uncommon human-centric domains like sports, where real-world data collection is complex and impractical. While synthetic datasets offer a promising alternative, existing approaches typically suffer from limited diversity in human appearance, motion, and scene composition due to their reliance on rigid asset libraries and hand-crafted rendering pipelines. To address this, we introduce Gen4D, a fully automated pipeline for generating diverse and photorealistic 4D human animations. Gen4D integrates expert-driven motion encoding, prompt-guided avatar generation using diffusion-based Gaussian splatting, and human-aware background synthesis to produce highly varied and lifelike human sequences. Based on Gen4D, we present SportPAL, a large-scale synthetic dataset spanning three sports: baseball, icehockey, and soccer. Together, Gen4D and SportPAL provide a scalable foundation for constructing synthetic datasets tailored to in-the-wild human-centric vision tasks, with no need for manual 3D modeling or scene design. 

---
# Advancing Decoding Strategies: Enhancements in Locally Typical Sampling for LLMs 

**Authors**: Jaydip Sen, Saptarshi Sengupta. Subhasis Dasgupta  

**Link**: [PDF](https://arxiv.org/pdf/2506.05387)  

**Abstract**: This chapter explores advancements in decoding strategies for large language models (LLMs), focusing on enhancing the Locally Typical Sampling (LTS) algorithm. Traditional decoding methods, such as top-k and nucleus sampling, often struggle to balance fluency, diversity, and coherence in text generation. To address these challenges, Adaptive Semantic-Aware Typicality Sampling (ASTS) is proposed as an improved version of LTS, incorporating dynamic entropy thresholding, multi-objective scoring, and reward-penalty adjustments. ASTS ensures contextually coherent and diverse text generation while maintaining computational efficiency. Its performance is evaluated across multiple benchmarks, including story generation and abstractive summarization, using metrics such as perplexity, MAUVE, and diversity scores. Experimental results demonstrate that ASTS outperforms existing sampling techniques by reducing repetition, enhancing semantic alignment, and improving fluency. 

---
# Beyond RAG: Reinforced Reasoning Augmented Generation for Clinical Notes 

**Authors**: Lo Pang-Yun Ting, Chengshuai Zhao, Yu-Hua Zeng, Yuan Jee Lim, Kun-Ta Chuang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05386)  

**Abstract**: Clinical note generation aims to automatically produce free-text summaries of a patient's condition and diagnostic process, with discharge instructions being a representative long-form example. While recent large language model (LLM)-based methods pre-trained on general clinical corpora show promise in clinical text generation, they fall short in producing long-form notes from limited patient information. In this paper, we propose R2AG, the first reinforced retriever for long-form discharge instruction generation based on pre-admission data. R2AG is trained with reinforcement learning to retrieve reasoning paths from a medical knowledge graph, providing explicit semantic guidance to the LLM. To bridge the information gap, we propose Group-Based Retriever Optimization (GRO) which improves retrieval quality with group-relative rewards, encouraging reasoning leaps for deeper inference by the LLM. Comprehensive experiments on the MIMIC-IV-Note dataset show that R2AG outperforms baselines in both clinical efficacy and natural language generation metrics. Further analysis reveals that R2AG fills semantic gaps in sparse input scenarios, and retrieved reasoning paths help LLMs avoid clinical misinterpretation by focusing on key evidence and following coherent reasoning. 

---
# Q-Ponder: A Unified Training Pipeline for Reasoning-based Visual Quality Assessment 

**Authors**: Zhuoxuan Cai, Jian Zhang, Xinbin Yuan, Pengtao Jiang, Wenxiang Chen, Bowen Tang, Lujian Yao, Qiyuan Wang, Jinwen Chen, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.05384)  

**Abstract**: Recent studies demonstrate that multimodal large language models (MLLMs) can proficiently evaluate visual quality through interpretable assessments. However, existing approaches typically treat quality scoring and reasoning descriptions as separate tasks with disjoint optimization objectives, leading to a trade-off: models adept at quality reasoning descriptions struggle with precise score regression, while score-focused models lack interpretability. This limitation hinders the full potential of MLLMs in visual quality assessment, where accuracy and interpretability should be mutually reinforcing. To address this, we propose a unified two-stage training framework comprising a cold-start stage and a reinforcement learning-based fine-tuning stage. Specifically, in the first stage, we distill high-quality data from a teacher model through expert-designed prompts, initializing reasoning capabilities via cross-entropy loss supervision. In the second stage, we introduce a novel reward with Group Relative Policy Optimization (GRPO) to jointly optimize scoring accuracy and reasoning consistency. We designate the models derived from these two stages as Q-Ponder-CI and Q-Ponder. Extensive experiments show that Q-Ponder achieves state-of-the-art (SOTA) performance on quality score regression benchmarks, delivering up to 6.5% higher SRCC on cross-domain datasets. Furthermore, Q-Ponder significantly outperforms description-based SOTA models, including its teacher model Qwen-2.5-VL-72B, particularly in description accuracy and reasonableness, demonstrating the generalization potential over diverse tasks. 

---
# How stealthy is stealthy? Studying the Efficacy of Black-Box Adversarial Attacks in the Real World 

**Authors**: Francesco Panebianco, Mario D'Onghia, Stefano Zanero aand Michele Carminati  

**Link**: [PDF](https://arxiv.org/pdf/2506.05382)  

**Abstract**: Deep learning systems, critical in domains like autonomous vehicles, are vulnerable to adversarial examples (crafted inputs designed to mislead classifiers). This study investigates black-box adversarial attacks in computer vision. This is a realistic scenario, where attackers have query-only access to the target model. Three properties are introduced to evaluate attack feasibility: robustness to compression, stealthiness to automatic detection, and stealthiness to human inspection. State-of-the-Art methods tend to prioritize one criterion at the expense of others. We propose ECLIPSE, a novel attack method employing Gaussian blurring on sampled gradients and a local surrogate model. Comprehensive experiments on a public dataset highlight ECLIPSE's advantages, demonstrating its contribution to the trade-off between the three properties. 

---
# Designing DSIC Mechanisms for Data Sharing in the Era of Large Language Models 

**Authors**: Seyed Moein Ayyoubzadeh, Kourosh Shahnazari, Mohammmadali Keshtparvar, MohammadAmin Fazli  

**Link**: [PDF](https://arxiv.org/pdf/2506.05379)  

**Abstract**: Training large language models (LLMs) requires vast amounts of high-quality data from institutions that face legal, privacy, and strategic constraints. Existing data procurement methods often rely on unverifiable trust or ignore heterogeneous provider costs. We introduce a mechanism-design framework for truthful, trust-minimized data sharing that ensures dominant-strategy incentive compatibility (DSIC), individual rationality, and weak budget balance, while rewarding data based on both quality and learning utility. We formalize a model where providers privately know their data cost and quality, and value arises solely from the data's contribution to model performance. Based on this, we propose the Quality-Weighted Marginal-Incentive Auction (Q-MIA), which ranks providers using a virtual cost metric and uses Myerson-style payments to ensure DSIC and budget feasibility. To support settings with limited liquidity or long-term incentives, we introduce the Marginal Utility Token (MUT), which allocates future rights based on marginal contributions. We unify these in Mixed-MIA, a hybrid mechanism balancing upfront payments and deferred rewards. All mechanisms support verifiable, privacy-preserving implementation. Theoretically and empirically, they outperform volume-based and trust-based baselines, eliciting higher-quality data under budget constraints while remaining robust to misreporting and collusion. This establishes a principled foundation for sustainable and fair data markets for future LLMs. 

---
# A Red Teaming Roadmap Towards System-Level Safety 

**Authors**: Zifan Wang, Christina Q. Knight, Jeremy Kritz, Willow E. Primack, Julian Michael  

**Link**: [PDF](https://arxiv.org/pdf/2506.05376)  

**Abstract**: Large Language Model (LLM) safeguards, which implement request refusals, have become a widely adopted mitigation strategy against misuse. At the intersection of adversarial machine learning and AI safety, safeguard red teaming has effectively identified critical vulnerabilities in state-of-the-art refusal-trained LLMs. However, in our view the many conference submissions on LLM red teaming do not, in aggregate, prioritize the right research problems. First, testing against clear product safety specifications should take a higher priority than abstract social biases or ethical principles. Second, red teaming should prioritize realistic threat models that represent the expanding risk landscape and what real attackers might do. Finally, we contend that system-level safety is a necessary step to move red teaming research forward, as AI models present new threats as well as affordances for threat mitigation (e.g., detection and banning of malicious users) once placed in a deployment context. Adopting these priorities will be necessary in order for red teaming research to adequately address the slate of new threats that rapid AI advances present today and will present in the very near future. 

---
# Speaking images. A novel framework for the automated self-description of artworks 

**Authors**: Valentine Bernasconi, Gustavo Marfia  

**Link**: [PDF](https://arxiv.org/pdf/2506.05368)  

**Abstract**: Recent breakthroughs in generative AI have opened the door to new research perspectives in the domain of art and cultural heritage, where a large number of artifacts have been digitized. There is a need for innovation to ease the access and highlight the content of digital collections. Such innovations develop into creative explorations of the digital image in relation to its malleability and contemporary interpretation, in confrontation to the original historical object. Based on the concept of the autonomous image, we propose a new framework towards the production of self-explaining cultural artifacts using open-source large-language, face detection, text-to-speech and audio-to-animation models. The goal is to start from a digitized artwork and to automatically assemble a short video of the latter where the main character animates to explain its content. The whole process questions cultural biases encapsulated in large-language models, the potential of digital images and deepfakes of artworks for educational purposes, along with concerns of the field of art history regarding such creative diversions. 

---
# Can ChatGPT Perform Image Splicing Detection? A Preliminary Study 

**Authors**: Souradip Nath  

**Link**: [PDF](https://arxiv.org/pdf/2506.05358)  

**Abstract**: Multimodal Large Language Models (MLLMs) like GPT-4V are capable of reasoning across text and image modalities, showing promise in a variety of complex vision-language tasks. In this preliminary study, we investigate the out-of-the-box capabilities of GPT-4V in the domain of image forensics, specifically, in detecting image splicing manipulations. Without any task-specific fine-tuning, we evaluate GPT-4V using three prompting strategies: Zero-Shot (ZS), Few-Shot (FS), and Chain-of-Thought (CoT), applied over a curated subset of the CASIA v2.0 splicing dataset.
Our results show that GPT-4V achieves competitive detection performance in zero-shot settings (more than 85% accuracy), with CoT prompting yielding the most balanced trade-off across authentic and spliced images. Qualitative analysis further reveals that the model not only detects low-level visual artifacts but also draws upon real-world contextual knowledge such as object scale, semantic consistency, and architectural facts, to identify implausible composites. While GPT-4V lags behind specialized state-of-the-art splicing detection models, its generalizability, interpretability, and encyclopedic reasoning highlight its potential as a flexible tool in image forensics. 

---
# Infinite Time Turing Machines and their Applications 

**Authors**: Rukmal Weerawarana, Maxwell Braun  

**Link**: [PDF](https://arxiv.org/pdf/2506.05351)  

**Abstract**: This work establishes a rigorous theoretical foundation for analyzing deep learning systems by leveraging Infinite Time Turing Machines (ITTMs), which extend classical computation into transfinite ordinal steps. Using ITTMs, we reinterpret modern architectures like Transformers, revealing fundamental limitations in scalability, efficiency, and interpretability. Building on these insights, we propose the Universal State Machine (USM), a novel computational paradigm designed from first principles. The USM employs a dynamic, queryable computation graph that evolves in real time, enabling modular, interpretable, and resource-efficient computation. This framework not only overcomes the inefficiencies and rigidity of current models but also lays the groundwork for scalable, generalizable artificial intelligence systems. 

---
# Towards provable probabilistic safety for scalable embodied AI systems 

**Authors**: Linxuan He, Qing-Shan Jia, Ang Li, Hongyan Sang, Ling Wang, Jiwen Lu, Tao Zhang, Jie Zhou, Yi Zhang, Yisen Wang, Peng Wei, Zhongyuan Wang, Henry X. Liu, Shuo Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.05171)  

**Abstract**: Embodied AI systems, comprising AI models and physical plants, are increasingly prevalent across various applications. Due to the rarity of system failures, ensuring their safety in complex operating environments remains a major challenge, which severely hinders their large-scale deployment in safety-critical domains, such as autonomous vehicles, medical devices, and robotics. While achieving provable deterministic safety--verifying system safety across all possible scenarios--remains theoretically ideal, the rarity and complexity of corner cases make this approach impractical for scalable embodied AI systems. To address this challenge, we introduce provable probabilistic safety, which aims to ensure that the residual risk of large-scale deployment remains below a predefined threshold. Instead of attempting exhaustive safety proof across all corner cases, this paradigm establishes a probabilistic safety boundary on overall system performance, leveraging statistical methods to enhance feasibility and scalability. A well-defined probabilistic safety boundary enables embodied AI systems to be deployed at scale while allowing for continuous refinement of safety guarantees. Our work focuses on three core questions: what is provable probabilistic safety, how to prove the probabilistic safety, and how to achieve the provable probabilistic safety. By bridging the gap between theoretical safety assurance and practical deployment, our work offers a pathway toward safer, large-scale adoption of embodied AI systems in safety-critical applications. 

---
# Category Query Learning for Human-Object Interaction Classification 

**Authors**: Chi Xie, Fangao Zeng, Yue Hu, Shuang Liang, Yichen Wei  

**Link**: [PDF](https://arxiv.org/pdf/2303.14005)  

**Abstract**: Unlike most previous HOI methods that focus on learning better human-object features, we propose a novel and complementary approach called category query learning. Such queries are explicitly associated to interaction categories, converted to image specific category representation via a transformer decoder, and learnt via an auxiliary image-level classification task. This idea is motivated by an earlier multi-label image classification method, but is for the first time applied for the challenging human-object interaction classification task. Our method is simple, general and effective. It is validated on three representative HOI baselines and achieves new state-of-the-art results on two benchmarks. 

---
