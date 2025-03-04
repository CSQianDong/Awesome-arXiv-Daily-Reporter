# Persuade Me if You Can: A Framework for Evaluating Persuasion Effectiveness and Susceptibility Among Large Language Models 

**Authors**: Nimet Beyza Bozdag, Shuhaib Mehri, Gokhan Tur, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2503.01829)  

**Abstract**: Large Language Models (LLMs) demonstrate persuasive capabilities that rival human-level persuasion. While these capabilities can be used for social good, they also present risks of potential misuse. Moreover, LLMs' susceptibility to persuasion raises concerns about alignment with ethical principles. To study these dynamics, we introduce Persuade Me If You Can (PMIYC), an automated framework for evaluating persuasion through multi-agent interactions. Here, Persuader agents engage in multi-turn conversations with the Persuadee agents, allowing us to measure LLMs' persuasive effectiveness and their susceptibility to persuasion. We conduct comprehensive evaluations across diverse LLMs, ensuring each model is assessed against others in both subjective and misinformation contexts. We validate the efficacy of our framework through human evaluations and show alignment with prior work. PMIYC offers a scalable alternative to human annotation for studying persuasion in LLMs. Through PMIYC, we find that Llama-3.3-70B and GPT-4o exhibit similar persuasive effectiveness, outperforming Claude 3 Haiku by 30%. However, GPT-4o demonstrates over 50% greater resistance to persuasion for misinformation compared to Llama-3.3-70B. These findings provide empirical insights into the persuasive dynamics of LLMs and contribute to the development of safer AI systems. 

---
# Improving Retrospective Language Agents via Joint Policy Gradient Optimization 

**Authors**: Xueyang Feng, Bo Lan, Quanyu Dai, Lei Wang, Jiakai Tang, Xu Chen, Zhenhua Dong, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2503.01490)  

**Abstract**: In recent research advancements within the community, large language models (LLMs) have sparked great interest in creating autonomous agents. However, current prompt-based agents often heavily rely on large-scale LLMs. Meanwhile, although fine-tuning methods significantly enhance the capabilities of smaller LLMs, the fine-tuned agents often lack the potential for self-reflection and self-improvement. To address these challenges, we introduce a novel agent framework named RetroAct, which is a framework that jointly optimizes both task-planning and self-reflective evolution capabilities in language agents. Specifically, we develop a two-stage joint optimization process that integrates imitation learning and reinforcement learning, and design an off-policy joint policy gradient optimization algorithm with imitation learning regularization to enhance the data efficiency and training stability in agent tasks. RetroAct significantly improves the performance of open-source models, reduces dependency on closed-source LLMs, and enables fine-tuned agents to learn and evolve continuously. We conduct extensive experiments across various testing environments, demonstrating RetroAct has substantial improvements in task performance and decision-making processes. 

---
# Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs 

**Authors**: Kanishk Gandhi, Ayush Chakravarthy, Anikait Singh, Nathan Lile, Noah D. Goodman  

**Link**: [PDF](https://arxiv.org/pdf/2503.01307)  

**Abstract**: Test-time inference has emerged as a powerful paradigm for enabling language models to ``think'' longer and more carefully about complex challenges, much like skilled human experts. While reinforcement learning (RL) can drive self-improvement in language models on verifiable tasks, some models exhibit substantial gains while others quickly plateau. For instance, we find that Qwen-2.5-3B far exceeds Llama-3.2-3B under identical RL training for the game of Countdown. This discrepancy raises a critical question: what intrinsic properties enable effective self-improvement? We introduce a framework to investigate this question by analyzing four key cognitive behaviors -- verification, backtracking, subgoal setting, and backward chaining -- that both expert human problem solvers and successful language models employ. Our study reveals that Qwen naturally exhibits these reasoning behaviors, whereas Llama initially lacks them. In systematic experimentation with controlled behavioral datasets, we find that priming Llama with examples containing these reasoning behaviors enables substantial improvements during RL, matching or exceeding Qwen's performance. Importantly, the presence of reasoning behaviors, rather than correctness of answers, proves to be the critical factor -- models primed with incorrect solutions containing proper reasoning patterns achieve comparable performance to those trained on correct solutions. Finally, leveraging continued pretraining with OpenWebMath data, filtered to amplify reasoning behaviors, enables the Llama model to match Qwen's self-improvement trajectory. Our findings establish a fundamental relationship between initial reasoning behaviors and the capacity for improvement, explaining why some language models effectively utilize additional computation while others plateau. 

---
# Smoothing Grounding and Reasoning for MLLM-Powered GUI Agents with Query-Oriented Pivot Tasks 

**Authors**: Zongru Wu, Pengzhou Cheng, Zheng Wu, Tianjie Ju, Zhuosheng Zhang, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00401)  

**Abstract**: Perception-enhanced pre-training, particularly through grounding techniques, is widely adopted to enhance the performance of graphical user interface (GUI) agents. However, in resource-constrained scenarios, the format discrepancy between coordinate-oriented grounding and action-oriented reasoning limits the effectiveness of grounding for reasoning tasks. To address this challenge, we propose a query-oriented pivot approach called query inference, which serves as a bridge between GUI grounding and reasoning. By inferring potential user queries from a screenshot and its associated element coordinates, query inference improves the understanding of coordinates while aligning more closely with reasoning tasks. Experimental results show that query inference outperforms previous grounding techniques under the same training data scale. Notably, query inference achieves comparable or even better performance to large-scale grounding-enhanced OS-Atlas with less than 0.1% of training data. Furthermore, we explore the impact of reasoning formats and demonstrate that integrating additional semantic information into the input further boosts reasoning performance. The code is publicly available athttps://github.com/ZrW00/GUIPivot. 

---
# Personalized Causal Graph Reasoning for LLMs: A Case Study on Dietary Recommendations 

**Authors**: Zhongqi Yang, Amir Rahmani  

**Link**: [PDF](https://arxiv.org/pdf/2503.00134)  

**Abstract**: Large Language Models (LLMs) effectively leverage common-sense knowledge for general reasoning, yet they struggle with personalized reasoning when tasked with interpreting multifactor personal data. This limitation restricts their applicability in domains that require context-aware decision-making tailored to individuals. This paper introduces Personalized Causal Graph Reasoning as an agentic framework that enhances LLM reasoning by incorporating personal causal graphs derived from data of individuals. These graphs provide a foundation that guides the LLM's reasoning process. We evaluate it on a case study on nutrient-oriented dietary recommendations, which requires personal reasoning due to the implicit unique dietary effects. We propose a counterfactual evaluation to estimate the efficiency of LLM-recommended foods for glucose management. Results demonstrate that the proposed method efficiently provides personalized dietary recommendations to reduce average glucose iAUC across three time windows, which outperforms the previous approach. LLM-as-a-judge evaluation results indicate that our proposed method enhances personalization in the reasoning process. 

---
# Code-as-Symbolic-Planner: Foundation Model-Based Robot Planning via Symbolic Code Generation 

**Authors**: Yongchao Chen, Yilun Hao, Yang Zhang, Chuchu Fan  

**Link**: [PDF](https://arxiv.org/pdf/2503.01700)  

**Abstract**: Recent works have shown great potentials of Large Language Models (LLMs) in robot task and motion planning (TAMP). Current LLM approaches generate text- or code-based reasoning chains with sub-goals and action plans. However, they do not fully leverage LLMs' symbolic computing and code generation capabilities. Many robot TAMP tasks involve complex optimization under multiple constraints, where pure textual reasoning is insufficient. While augmenting LLMs with predefined solvers and planners improves performance, it lacks generalization across tasks. Given LLMs' growing coding proficiency, we enhance their TAMP capabilities by steering them to generate code as symbolic planners for optimization and constraint verification. Unlike prior work that uses code to interface with robot action modules, we steer LLMs to generate code as solvers, planners, and checkers for TAMP tasks requiring symbolic computing, while still leveraging textual reasoning to incorporate common sense. With a multi-round guidance and answer evolution framework, the proposed Code-as-Symbolic-Planner improves success rates by average 24.1\% over best baseline methods across seven typical TAMP tasks and three popular LLMs. Code-as-Symbolic-Planner shows strong effectiveness and generalizability across discrete and continuous environments, 2D/3D simulations and real-world settings, as well as single- and multi-robot tasks with diverse requirements. See our project website this https URL for prompts, videos, and code. 

---
# Instructor-Worker Large Language Model System for Policy Recommendation: a Case Study on Air Quality Analysis of the January 2025 Los Angeles Wildfires 

**Authors**: Kyle Gao, Dening Lu, Liangzhi Li, Nan Chen, Hongjie He, Linlin Xu, Jonathan Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00566)  

**Abstract**: The Los Angeles wildfires of January 2025 caused more than 250 billion dollars in damage and lasted for nearly an entire month before containment. Following our previous work, the Digital Twin Building, we modify and leverage the multi-agent large language model framework as well as the cloud-mapping integration to study the air quality during the Los Angeles wildfires. Recent advances in large language models have allowed for out-of-the-box automated large-scale data analysis. We use a multi-agent large language system comprised of an Instructor agent and Worker agents. Upon receiving the users' instructions, the Instructor agent retrieves the data from the cloud platform and produces instruction prompts to the Worker agents. The Worker agents then analyze the data and provide summaries. The summaries are finally input back into the Instructor agent, which then provides the final data analysis. We test this system's capability for data-based policy recommendation by assessing our Instructor-Worker LLM system's health recommendations based on air quality during the Los Angeles wildfires. 

---
# SrSv: Integrating Sequential Rollouts with Sequential Value Estimation for Multi-agent Reinforcement Learning 

**Authors**: Xu Wan, Chao Yang, Cheng Yang, Jie Song, Mingyang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.01458)  

**Abstract**: Although multi-agent reinforcement learning (MARL) has shown its success across diverse domains, extending its application to large-scale real-world systems still faces significant challenges. Primarily, the high complexity of real-world environments exacerbates the credit assignment problem, substantially reducing training efficiency. Moreover, the variability of agent populations in large-scale scenarios necessitates scalable decision-making mechanisms. To address these challenges, we propose a novel framework: Sequential rollout with Sequential value estimation (SrSv). This framework aims to capture agent interdependence and provide a scalable solution for cooperative MARL. Specifically, SrSv leverages the autoregressive property of the Transformer model to handle varying populations through sequential action rollout. Furthermore, to capture the interdependence of policy distributions and value functions among multiple agents, we introduce an innovative sequential value estimation methodology and integrates the value approximation into an attention-based sequential model. We evaluate SrSv on three benchmarks: Multi-Agent MuJoCo, StarCraft Multi-Agent Challenge, and DubinsCars. Experimental results demonstrate that SrSv significantly outperforms baseline methods in terms of training efficiency without compromising convergence performance. Moreover, when implemented in a large-scale DubinsCar system with 1,024 agents, our framework surpasses existing benchmarks, highlighting the excellent scalability of SrSv. 

---
# SENSEI: Semantic Exploration Guided by Foundation Models to Learn Versatile World Models 

**Authors**: Cansu Sancaktar, Christian Gumbsch, Andrii Zadaianchuk, Pavel Kolev, Georg Martius  

**Link**: [PDF](https://arxiv.org/pdf/2503.01584)  

**Abstract**: Exploration is a cornerstone of reinforcement learning (RL). Intrinsic motivation attempts to decouple exploration from external, task-based rewards. However, established approaches to intrinsic motivation that follow general principles such as information gain, often only uncover low-level interactions. In contrast, children's play suggests that they engage in meaningful high-level behavior by imitating or interacting with their caregivers. Recent work has focused on using foundation models to inject these semantic biases into exploration. However, these methods often rely on unrealistic assumptions, such as language-embedded environments or access to high-level actions. We propose SEmaNtically Sensible ExploratIon (SENSEI), a framework to equip model-based RL agents with an intrinsic motivation for semantically meaningful behavior. SENSEI distills a reward signal of interestingness from Vision Language Model (VLM) annotations, enabling an agent to predict these rewards through a world model. Using model-based RL, SENSEI trains an exploration policy that jointly maximizes semantic rewards and uncertainty. We show that in both robotic and video game-like simulations SENSEI discovers a variety of meaningful behaviors from image observations and low-level actions. SENSEI provides a general tool for learning from foundation model feedback, a crucial research direction, as VLMs become more powerful. 

---
# NeSyC: A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains 

**Authors**: Wonje Choi, Jinwoo Park, Sanghyun Ahn, Daehee Lee, Honguk Woo  

**Link**: [PDF](https://arxiv.org/pdf/2503.00870)  

**Abstract**: We explore neuro-symbolic approaches to generalize actionable knowledge, enabling embodied agents to tackle complex tasks more effectively in open-domain environments. A key challenge for embodied agents is the generalization of knowledge across diverse environments and situations, as limited experiences often confine them to their prior knowledge. To address this issue, we introduce a novel framework, NeSyC, a neuro-symbolic continual learner that emulates the hypothetico-deductive model by continually formulating and validating knowledge from limited experiences through the combined use of Large Language Models (LLMs) and symbolic tools. Specifically, we devise a contrastive generality improvement scheme within NeSyC, which iteratively generates hypotheses using LLMs and conducts contrastive validation via symbolic tools. This scheme reinforces the justification for admissible actions while minimizing the inference of inadmissible ones. Additionally, we incorporate a memory-based monitoring scheme that efficiently detects action errors and triggers the knowledge refinement process across domains. Experiments conducted on diverse embodied task benchmarks-including ALFWorld, VirtualHome, Minecraft, RLBench, and a real-world robotic scenario-demonstrate that NeSyC is highly effective in solving complex embodied tasks across a range of open-domain environments. 

---
# A Law Reasoning Benchmark for LLM with Tree-Organized Structures including Factum Probandum, Evidence and Experiences 

**Authors**: Jiaxin Shen, Jinan Xu, Huiqi Hu, Luyi Lin, Fei Zheng, Guoyang Ma, Fandong Meng, Jie Zhou, Wenjuan Han  

**Link**: [PDF](https://arxiv.org/pdf/2503.00841)  

**Abstract**: While progress has been made in legal applications, law reasoning, crucial for fair adjudication, remains unexplored. We propose a transparent law reasoning schema enriched with hierarchical factum probandum, evidence, and implicit experience, enabling public scrutiny and preventing bias. Inspired by this schema, we introduce the challenging task, which takes a textual case description and outputs a hierarchical structure justifying the final decision. We also create the first crowd-sourced dataset for this task, enabling comprehensive evaluation. Simultaneously, we propose an agent framework that employs a comprehensive suite of legal analysis tools to address the challenge task. This benchmark paves the way for transparent and accountable AI-assisted law reasoning in the ``Intelligent Court''. 

---
# Multi-Agent Reinforcement Learning with Long-Term Performance Objectives for Service Workforce Optimization 

**Authors**: Kareem Eissa, Rayal Prasad, Sarith Mohan, Ankur Kapoor, Dorin Comaniciu, Vivek Singh  

**Link**: [PDF](https://arxiv.org/pdf/2503.01069)  

**Abstract**: Workforce optimization plays a crucial role in efficient organizational operations where decision-making may span several different administrative and time scales. For instance, dispatching personnel to immediate service requests while managing talent acquisition with various expertise sets up a highly dynamic optimization problem. Existing work focuses on specific sub-problems such as resource allocation and facility location, which are solved with heuristics like local-search and, more recently, deep reinforcement learning. However, these may not accurately represent real-world scenarios where such sub-problems are not fully independent. Our aim is to fill this gap by creating a simulator that models a unified workforce optimization problem. Specifically, we designed a modular simulator to support the development of reinforcement learning methods for integrated workforce optimization problems. We focus on three interdependent aspects: personnel dispatch, workforce management, and personnel positioning. The simulator provides configurable parameterizations to help explore dynamic scenarios with varying levels of stochasticity and non-stationarity. To facilitate benchmarking and ablation studies, we also include heuristic and RL baselines for the above mentioned aspects. 

---
# Agentic AI Needs a Systems Theory 

**Authors**: Erik Miehling, Karthikeyan Natesan Ramamurthy, Kush R. Varshney, Matthew Riemer, Djallel Bouneffouf, John T. Richards, Amit Dhurandhar, Elizabeth M. Daly, Michael Hind, Prasanna Sattigeri, Dennis Wei, Ambrish Rawat, Jasmina Gajcin, Werner Geyer  

**Link**: [PDF](https://arxiv.org/pdf/2503.00237)  

**Abstract**: The endowment of AI with reasoning capabilities and some degree of agency is widely viewed as a path toward more capable and generalizable systems. Our position is that the current development of agentic AI requires a more holistic, systems-theoretic perspective in order to fully understand their capabilities and mitigate any emergent risks. The primary motivation for our position is that AI development is currently overly focused on individual model capabilities, often ignoring broader emergent behavior, leading to a significant underestimation in the true capabilities and associated risks of agentic AI. We describe some fundamental mechanisms by which advanced capabilities can emerge from (comparably simpler) agents simply due to their interaction with the environment and other agents. Informed by an extensive amount of existing literature from various fields, we outline mechanisms for enhanced agent cognition, emergent causal reasoning ability, and metacognitive awareness. We conclude by presenting some key open challenges and guidance for the development of agentic AI. We emphasize that a systems-level perspective is essential for better understanding, and purposefully shaping, agentic AI systems. 

---
# Adversarial Agents: Black-Box Evasion Attacks with Reinforcement Learning 

**Authors**: Kyle Domico, Jean-Charles Noirot Ferrand, Ryan Sheatsley, Eric Pauley, Josiah Hanna, Patrick McDaniel  

**Link**: [PDF](https://arxiv.org/pdf/2503.01734)  

**Abstract**: Reinforcement learning (RL) offers powerful techniques for solving complex sequential decision-making tasks from experience. In this paper, we demonstrate how RL can be applied to adversarial machine learning (AML) to develop a new class of attacks that learn to generate adversarial examples: inputs designed to fool machine learning models. Unlike traditional AML methods that craft adversarial examples independently, our RL-based approach retains and exploits past attack experience to improve future attacks. We formulate adversarial example generation as a Markov Decision Process and evaluate RL's ability to (a) learn effective and efficient attack strategies and (b) compete with state-of-the-art AML. On CIFAR-10, our agent increases the success rate of adversarial examples by 19.4% and decreases the median number of victim model queries per adversarial example by 53.2% from the start to the end of training. In a head-to-head comparison with a state-of-the-art image attack, SquareAttack, our approach enables an adversary to generate adversarial examples with 13.1% more success after 5000 episodes of training. From a security perspective, this work demonstrates a powerful new attack vector that uses RL to attack ML models efficiently and at scale. 

---
# Eau De $Q$-Network: Adaptive Distillation of Neural Networks in Deep Reinforcement Learning 

**Authors**: Théo Vincent, Tim Faust, Yogesh Tripathi, Jan Peters, Carlo D'Eramo  

**Link**: [PDF](https://arxiv.org/pdf/2503.01437)  

**Abstract**: Recent works have successfully demonstrated that sparse deep reinforcement learning agents can be competitive against their dense counterparts. This opens up opportunities for reinforcement learning applications in fields where inference time and memory requirements are cost-sensitive or limited by hardware. Until now, dense-to-sparse methods have relied on hand-designed sparsity schedules that are not synchronized with the agent's learning pace. Crucially, the final sparsity level is chosen as a hyperparameter, which requires careful tuning as setting it too high might lead to poor performances. In this work, we address these shortcomings by crafting a dense-to-sparse algorithm that we name Eau De $Q$-Network (EauDeQN). To increase sparsity at the agent's learning pace, we consider multiple online networks with different sparsity levels, where each online network is trained from a shared target network. At each target update, the online network with the smallest loss is chosen as the next target network, while the other networks are replaced by a pruned version of the chosen network. We evaluate the proposed approach on the Atari $2600$ benchmark and the MuJoCo physics simulator, showing that EauDeQN reaches high sparsity levels while keeping performances high. 

---
# LLM-Advisor: An LLM Benchmark for Cost-efficient Path Planning across Multiple Terrains 

**Authors**: Ling Xiao, Toshihiko Yamasaki  

**Link**: [PDF](https://arxiv.org/pdf/2503.01236)  

**Abstract**: Multi-terrain cost-efficient path planning is a crucial task in robot navigation, requiring the identification of a path from the start to the goal that not only avoids obstacles but also minimizes travel costs. This is especially crucial for real-world applications where robots need to navigate diverse terrains in outdoor environments, where recharging or refueling is difficult. However, there is very limited research on this topic. In this paper, we develop a prompt-based approach, LLM-Advisor, which leverages large language models (LLMs) as effective advisors for path planning. The LLM-Advisor selectively provides suggestions, demonstrating its ability to recognize when no modifications are necessary. When suggestions are made, 70.59% of the paths suggested for the A* algorithm, 69.47% for the RRT* algorithm, and 78.70% for the LLM-A* algorithm achieve greater cost efficiency. Since LLM-Advisor may occasionally lack common sense in their suggestions, we propose two hallucination-mitigation strategies. Furthermore, we experimentally verified that GPT-4o performs poorly in zero-shot path planning, even when terrain descriptions are clearly provided, demonstrating its low spatial awareness. We also experimentally demonstrate that using an LLM as an advisor is more effective than directly integrating it into the path-planning loop. Since LLMs may generate hallucinations, using LLMs in the loop of a search-based method (such as A*) may lead to a higher number of failed paths, demonstrating that our proposed LLM-Advisor is a better choice. 

---
# A Taxonomy for Evaluating Generalist Robot Policies 

**Authors**: Jensen Gao, Suneel Belkhale, Sudeep Dasari, Ashwin Balakrishna, Dhruv Shah, Dorsa Sadigh  

**Link**: [PDF](https://arxiv.org/pdf/2503.01238)  

**Abstract**: Machine learning for robotics promises to unlock generalization to novel tasks and environments. Guided by this promise, many recent works have focused on scaling up robot data collection and developing larger, more expressive policies to achieve this. But how do we measure progress towards this goal of policy generalization in practice? Evaluating and quantifying generalization is the Wild West of modern robotics, with each work proposing and measuring different types of generalization in their own, often difficult to reproduce, settings. In this work, our goal is (1) to outline the forms of generalization we believe are important in robot manipulation in a comprehensive and fine-grained manner, and (2) to provide reproducible guidelines for measuring these notions of generalization. We first propose STAR-Gen, a taxonomy of generalization for robot manipulation structured around visual, semantic, and behavioral generalization. We discuss how our taxonomy encompasses most prior notions of generalization in robotics. Next, we instantiate STAR-Gen with a concrete real-world benchmark based on the widely-used Bridge V2 dataset. We evaluate a variety of state-of-the-art models on this benchmark to demonstrate the utility of our taxonomy in practice. Our taxonomy of generalization can yield many interesting insights into existing models: for example, we observe that current vision-language-action models struggle with various types of semantic generalization, despite the promise of pre-training on internet-scale language datasets. We believe STAR-Gen and our guidelines can improve the dissemination and evaluation of progress towards generalization in robotics, which we hope will guide model design and future data collection efforts. We provide videos and demos at our website this http URL. 

---
# AI Agents for Ground-Based Gamma Astronomy 

**Authors**: D. Kostunin, V. Sotnikov, S. Golovachev, A. Strube  

**Link**: [PDF](https://arxiv.org/pdf/2503.00821)  

**Abstract**: Next-generation instruments for ground-based gamma-ray astronomy are marked by a substantial increase in complexity, featuring dozens of telescopes. This leap in scale introduces significant challenges in managing system operations and offline data analysis. Methods, which depend on advanced personnel training and sophisticated software, become increasingly strained as system complexity grows, making it more challenging to effectively support users in such a multifaceted environment. To address these challenges, we propose the development of AI agents based on instruction-finetuned large language models (LLMs). These agents align with specific documentation and codebases, understand the environmental context, operate with external APIs, and communicate with humans in natural language. Leveraging the advanced capabilities of modern LLMs, which can process and retain vast amounts of information, these AI agents offer a transformative approach to system management and data analysis by automating complex tasks and providing intelligent assistance. We present two prototypes that integrate with the Cherenkov Telescope Array Observatory pipelines for operations and offline data analysis. The first prototype automates data model implementation and maintenance for the Configuration Database of the Array Control and Data Acquisition (ACADA). The second prototype is an open-access code generation application tailored for data analysis based on the Gammapy framework. 

---
# From Understanding the World to Intervening in It: A Unified Multi-Scale Framework for Embodied Cognition 

**Authors**: Maijunxian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00727)  

**Abstract**: In this paper, we propose AUKAI, an Adaptive Unified Knowledge-Action Intelligence for embodied cognition that seamlessly integrates perception, memory, and decision-making via multi-scale error feedback. Interpreting AUKAI as an embedded world model, our approach simultaneously predicts state transitions and evaluates intervention utility. The framework is underpinned by rigorous theoretical analysis drawn from convergence theory, optimal control, and Bayesian inference, which collectively establish conditions for convergence, stability, and near-optimal performance. Furthermore, we present a hybrid implementation that combines the strengths of neural networks with symbolic reasoning modules, thereby enhancing interpretability and robustness. Finally, we demonstrate the potential of AUKAI through a detailed application in robotic navigation and obstacle avoidance, and we outline comprehensive experimental plans to validate its effectiveness in both simulated and real-world environments. 

---
# CLEA: Closed-Loop Embodied Agent for Enhancing Task Execution in Dynamic Environments 

**Authors**: Mingcong Lei, Ge Wang, Yiming Zhao, Zhixin Mai, Qing Zhao, Yao Guo, Zhen Li, Shuguang Cui, Yatong Han, Jinke Ren  

**Link**: [PDF](https://arxiv.org/pdf/2503.00729)  

**Abstract**: Large Language Models (LLMs) exhibit remarkable capabilities in the hierarchical decomposition of complex tasks through semantic reasoning. However, their application in embodied systems faces challenges in ensuring reliable execution of subtask sequences and achieving one-shot success in long-term task completion. To address these limitations in dynamic environments, we propose Closed-Loop Embodied Agent (CLEA) -- a novel architecture incorporating four specialized open-source LLMs with functional decoupling for closed-loop task management. The framework features two core innovations: (1) Interactive task planner that dynamically generates executable subtasks based on the environmental memory, and (2) Multimodal execution critic employing an evaluation framework to conduct a probabilistic assessment of action feasibility, triggering hierarchical re-planning mechanisms when environmental perturbations exceed preset thresholds. To validate CLEA's effectiveness, we conduct experiments in a real environment with manipulable objects, using two heterogeneous robots for object search, manipulation, and search-manipulation integration tasks. Across 12 task trials, CLEA outperforms the baseline model, achieving a 67.3% improvement in success rate and a 52.8% increase in task completion rate. These results demonstrate that CLEA significantly enhances the robustness of task planning and execution in dynamic environments. 

---
# Factorized Deep Q-Network for Cooperative Multi-Agent Reinforcement Learning in Victim Tagging 

**Authors**: Maria Ana Cardei, Afsaneh Doryab  

**Link**: [PDF](https://arxiv.org/pdf/2503.00684)  

**Abstract**: Mass casualty incidents (MCIs) are a growing concern, characterized by complexity and uncertainty that demand adaptive decision-making strategies. The victim tagging step in the emergency medical response must be completed quickly and is crucial for providing information to guide subsequent time-constrained response actions. In this paper, we present a mathematical formulation of multi-agent victim tagging to minimize the time it takes for responders to tag all victims. Five distributed heuristics are formulated and evaluated with simulation experiments. The heuristics considered are on-the go, practical solutions that represent varying levels of situational uncertainty in the form of global or local communication capabilities, showcasing practical constraints. We further investigate the performance of a multi-agent reinforcement learning (MARL) strategy, factorized deep Q-network (FDQN), to minimize victim tagging time as compared to baseline heuristics. Extensive simulations demonstrate that between the heuristics, methods with local communication are more efficient for adaptive victim tagging, specifically choosing the nearest victim with the option to replan. Analyzing all experiments, we find that our FDQN approach outperforms heuristics in smaller-scale scenarios, while heuristics excel in more complex scenarios. Our experiments contain diverse complexities that explore the upper limits of MARL capabilities for real-world applications and reveal key insights. 

---
# LLMDR: LLM-Driven Deadlock Detection and Resolution in Multi-Agent Pathfinding 

**Authors**: Seungbae Seo, Junghwan Kim, Minjeong Shin, Bongwon Suh  

**Link**: [PDF](https://arxiv.org/pdf/2503.00717)  

**Abstract**: Multi-Agent Pathfinding (MAPF) is a core challenge in multi-agent systems. Existing learning-based MAPF methods often struggle with scalability, particularly when addressing complex scenarios that are prone to deadlocks. To address these challenges, we introduce LLMDR (LLM-Driven Deadlock Detection and Resolution), an approach designed to resolve deadlocks and improve the performance of learnt MAPF models. LLMDR integrates the inference capabilities of large language models (LLMs) with learnt MAPF models and prioritized planning, enabling it to detect deadlocks and provide customized resolution strategies. We evaluate LLMDR on standard MAPF benchmark maps with varying agent numbers, measuring its performance when combined with several base models. The results demonstrate that LLMDR improves the performance of learnt MAPF models, particularly in deadlock-prone scenarios, with notable improvements in success rates. These findings show the potential of integrating LLMs to improve the scalability of learning-based MAPF methods.
The source code for LLMDR is available at: this https URL 

---
# Never too Prim to Swim: An LLM-Enhanced RL-based Adaptive S-Surface Controller for AUVs under Extreme Sea Conditions 

**Authors**: Guanwen Xie, Jingzehua Xu, Yimian Ding, Zhi Zhang, Shuai Zhang, Yi Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00527)  

**Abstract**: The adaptivity and maneuvering capabilities of Autonomous Underwater Vehicles (AUVs) have drawn significant attention in oceanic research, due to the unpredictable disturbances and strong coupling among the AUV's degrees of freedom. In this paper, we developed large language model (LLM)-enhanced reinforcement learning (RL)-based adaptive S-surface controller for AUVs. Specifically, LLMs are introduced for the joint optimization of controller parameters and reward functions in RL training. Using multi-modal and structured explicit task feedback, LLMs enable joint adjustments, balance multiple objectives, and enhance task-oriented performance and adaptability. In the proposed controller, the RL policy focuses on upper-level tasks, outputting task-oriented high-level commands that the S-surface controller then converts into control signals, ensuring cancellation of nonlinear effects and unpredictable external disturbances in extreme sea conditions. Under extreme sea conditions involving complex terrain, waves, and currents, the proposed controller demonstrates superior performance and adaptability in high-level tasks such as underwater target tracking and data collection, outperforming traditional PID and SMC controllers. 

---
# PodAgent: A Comprehensive Framework for Podcast Generation 

**Authors**: Yujia Xiao, Lei He, Haohan Guo, Fenglong Xie, Tan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.00455)  

**Abstract**: Existing Existing automatic audio generation methods struggle to generate podcast-like audio programs effectively. The key challenges lie in in-depth content generation, appropriate and expressive voice production. This paper proposed PodAgent, a comprehensive framework for creating audio programs. PodAgent 1) generates informative topic-discussion content by designing a Host-Guest-Writer multi-agent collaboration system, 2) builds a voice pool for suitable voice-role matching and 3) utilizes LLM-enhanced speech synthesis method to generate expressive conversational speech. Given the absence of standardized evaluation criteria for podcast-like audio generation, we developed comprehensive assessment guidelines to effectively evaluate the model's performance. Experimental results demonstrate PodAgent's effectiveness, significantly surpassing direct GPT-4 generation in topic-discussion dialogue content, achieving an 87.4% voice-matching accuracy, and producing more expressive speech through LLM-guided synthesis. Demo page: this https URL. Source code: this https URL. 

---
# Nucleolus Credit Assignment for Effective Coalitions in Multi-agent Reinforcement Learning 

**Authors**: Yugu Li, Zehong Cao, Jianglin Qiao, Siyi Hu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00372)  

**Abstract**: In cooperative multi-agent reinforcement learning (MARL), agents typically form a single grand coalition based on credit assignment to tackle a composite task, often resulting in suboptimal performance. This paper proposed a nucleolus-based credit assignment grounded in cooperative game theory, enabling the autonomous partitioning of agents into multiple small coalitions that can effectively identify and complete subtasks within a larger composite task. Specifically, our designed nucleolus Q-learning could assign fair credits to each agent, and the nucleolus Q-operator provides theoretical guarantees with interpretability for both learning convergence and the stability of the formed small coalitions. Through experiments on Predator-Prey and StarCraft scenarios across varying difficulty levels, our approach demonstrated the emergence of multiple effective coalitions during MARL training, leading to faster learning and superior performance in terms of win rate and cumulative rewards especially in hard and super-hard environments, compared to four baseline methods. Our nucleolus-based credit assignment showed the promise for complex composite tasks requiring effective subteams of agents. 

---
# Shifting Power: Leveraging LLMs to Simulate Human Aversion in ABMs of Bilateral Financial Exchanges, A bond market study 

**Authors**: Alicia Vidler, Toby Walsh  

**Link**: [PDF](https://arxiv.org/pdf/2503.00320)  

**Abstract**: Bilateral markets, such as those for government bonds, involve decentralized and opaque transactions between market makers (MMs) and clients, posing significant challenges for traditional modeling approaches. To address these complexities, we introduce TRIBE an agent-based model augmented with a large language model (LLM) to simulate human-like decision-making in trading environments. TRIBE leverages publicly available data and stylized facts to capture realistic trading dynamics, integrating human biases like risk aversion and ambiguity sensitivity into the decision-making processes of agents. Our research yields three key contributions: first, we demonstrate that integrating LLMs into agent-based models to enhance client agency is feasible and enriches the simulation of agent behaviors in complex markets; second, we find that even slight trade aversion encoded within the LLM leads to a complete cessation of trading activity, highlighting the sensitivity of market dynamics to agents' risk profiles; third, we show that incorporating human-like variability shifts power dynamics towards clients and can disproportionately affect the entire system, often resulting in systemic agent collapse across simulations. These findings underscore the emergent properties that arise when introducing stochastic, human-like decision processes, revealing new system behaviors that enhance the realism and complexity of artificial societies. 

---
# PINN-DT: Optimizing Energy Consumption in Smart Building Using Hybrid Physics-Informed Neural Networks and Digital Twin Framework with Blockchain Security 

**Authors**: Hajar Kazemi Naeini, Roya Shomali, Abolhassan Pishahang, Hamidreza Hasanzadeh, Mahdieh Mohammadi, Saeid Asadi, Ahmad Gholizadeh Lonbar  

**Link**: [PDF](https://arxiv.org/pdf/2503.00331)  

**Abstract**: The advancement of smart grid technologies necessitates the integration of cutting-edge computational methods to enhance predictive energy optimization. This study proposes a multi-faceted approach by incorporating (1) Deep Reinforcement Learning (DRL) agents trained using data from Digital Twins (DTs) to optimize energy consumption in real time, (2) Physics-Informed Neural Networks (PINNs) to seamlessly embed physical laws within the optimization process, ensuring model accuracy and interpretability, and (3) Blockchain (BC) technology to facilitate secure and transparent communication across the smart grid infrastructure. The model was trained and validated using comprehensive datasets, including smart meter energy consumption data, renewable energy outputs, dynamic pricing, and user preferences collected from IoT devices. The proposed framework achieved superior predictive performance with a Mean Absolute Error (MAE) of 0.237 kWh, Root Mean Square Error (RMSE) of 0.298 kWh, and an R-squared (R2) value of 0.978, indicating a 97.8% explanation of data variance. Classification metrics further demonstrated the model's robustness, achieving 97.7% accuracy, 97.8% precision, 97.6% recall, and an F1 Score of 97.7%. Comparative analysis with traditional models like Linear Regression, Random Forest, SVM, LSTM, and XGBoost revealed the superior accuracy and real-time adaptability of the proposed method. In addition to enhancing energy efficiency, the model reduced energy costs by 35%, maintained a 96% user comfort index, and increased renewable energy utilization to 40%. This study demonstrates the transformative potential of integrating PINNs, DT, and Blockchain technologies to optimize energy consumption in smart grids, paving the way for sustainable, secure, and efficient energy management systems. 

---
# SafeAuto: Knowledge-Enhanced Safe Autonomous Driving with Multimodal Foundation Models 

**Authors**: Jiawei Zhang, Xuan Yang, Taiqi Wang, Yu Yao, Aleksandr Petiushko, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00211)  

**Abstract**: Traditional autonomous driving systems often struggle to integrate high-level reasoning with low-level control, resulting in suboptimal and sometimes unsafe driving behaviors. The emergence of Multimodal Large Language Models (MLLMs), which can process both visual and textual data, presents an opportunity to unify perception and reasoning tasks within a single framework. However, effectively embedding precise safety knowledge into MLLMs for autonomous driving remains a significant challenge. To address this, we propose SafeAuto, a novel framework that enhances MLLM-based autonomous driving systems by incorporating both unstructured and structured knowledge. Specifically, we first introduce the Position-Dependent Cross-Entropy (PDCE) loss function, designed to improve the accuracy of low-level control signal predictions when numerical values are represented as text. Second, to ensure safe autonomous driving by explicitly integrating precise safety knowledge into the MLLM, we develop a reasoning component for SafeAuto. This component translates driving safety regulations into first-order logic rules (e.g., "red light => stop") and incorporates these rules into a probabilistic graphical model, such as a Markov Logic Network (MLN). The MLN is trained to verify the predicted next actions using environmental attributes identified by attribute recognition models (e.g., detecting a red light) to form the predicates. Additionally, we construct a Multimodal RAG model that leverages video data, control signals, and environmental attributes to learn more effectively from past similar driving experiences. By integrating PDCE, MLN, and Multimodal RAG, SafeAuto significantly outperforms existing baselines across multiple datasets. This advancement enables more accurate, reliable, and safer autonomous driving systems that learn from experience, obey traffic laws, and perform precise control actions. 

---
# EdgeAIGuard: Agentic LLMs for Minor Protection in Digital Spaces 

**Authors**: Ghulam Mujtaba, Sunder Ali Khowaja, Kapal Dev  

**Link**: [PDF](https://arxiv.org/pdf/2503.00092)  

**Abstract**: Social media has become integral to minors' daily lives and is used for various purposes, such as making friends, exploring shared interests, and engaging in educational activities. However, the increase in screen time has also led to heightened challenges, including cyberbullying, online grooming, and exploitations posed by malicious actors. Traditional content moderation techniques have proven ineffective against exploiters' evolving tactics. To address these growing challenges, we propose the EdgeAIGuard content moderation approach that is designed to protect minors from online grooming and various forms of digital exploitation. The proposed method comprises a multi-agent architecture deployed strategically at the network edge to enable rapid detection with low latency and prevent harmful content targeting minors. The experimental results show the proposed method is significantly more effective than the existing approaches. 

---
# BixBench: a Comprehensive Benchmark for LLM-based Agents in Computational Biology 

**Authors**: Ludovico Mitchener, Jon M Laurent, Benjamin Tenmann, Siddharth Narayanan, Geemi P Wellawatte, Andrew White, Lorenzo Sani, Samuel G Rodriques  

**Link**: [PDF](https://arxiv.org/pdf/2503.00096)  

**Abstract**: Large Language Models (LLMs) and LLM-based agents show great promise in accelerating scientific research. Existing benchmarks for measuring this potential and guiding future development continue to evolve from pure recall and rote knowledge tasks, towards more practical work such as literature review and experimental planning. Bioinformatics is a domain where fully autonomous AI-driven discovery may be near, but no extensive benchmarks for measuring progress have been introduced to date. We therefore present the Bioinformatics Benchmark (BixBench), a dataset comprising over 50 real-world scenarios of practical biological data analysis with nearly 300 associated open-answer questions designed to measure the ability of LLM-based agents to explore biological datasets, perform long, multi-step analytical trajectories, and interpret the nuanced results of those analyses. We evaluate the performance of two frontier LLMs (GPT-4o and Claude 3.5 Sonnet) using a custom agent framework we open source. We find that even the latest frontier models only achieve 17% accuracy in the open-answer regime, and no better than random in a multiple-choice setting. By exposing the current limitations of frontier models, we hope BixBench can spur the development of agents capable of conducting rigorous bioinformatic analysis and accelerate scientific discovery. 

---
# Eeyore: Realistic Depression Simulation via Supervised and Preference Optimization 

**Authors**: Siyang Liu, Bianca Brie, Wenda Li, Laura Biester, Andrew Lee, James Pennebaker, Rada Mihalcea  

**Link**: [PDF](https://arxiv.org/pdf/2503.00018)  

**Abstract**: Large Language Models (LLMs) have been previously explored for mental healthcare training and therapy client simulation, but they still fall short in authentically capturing diverse client traits and psychological conditions. We introduce \textbf{Eeyore}, an 8B model optimized for realistic depression simulation through a structured alignment framework, incorporating expert input at every stage. First, we systematically curate real-world depression-related conversations, extracting depressive traits to guide data filtering and psychological profile construction, and use this dataset to instruction-tune Eeyore for profile adherence. Next, to further enhance realism, Eeyore undergoes iterative preference optimization -- first leveraging model-generated preferences and then calibrating with a small set of expert-annotated preferences. Throughout the entire pipeline, we actively collaborate with domain experts, developing interactive interfaces to validate trait extraction and iteratively refine structured psychological profiles for clinically meaningful role-play customization. Despite its smaller model size, the Eeyore depression simulation outperforms GPT-4o with SOTA prompting strategies, both in linguistic authenticity and profile adherence. 

---
