# CHOP: Mobile Operating Assistant with Constrained High-frequency Optimized Subtask Planning 

**Authors**: Yuqi Zhou, Shuai Wang, Sunhao Dai, Qinglin Jia, Zhaocheng Du, Zhenhua Dong, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.03743)  

**Abstract**: The advancement of visual language models (VLMs) has enhanced mobile device operations, allowing simulated human-like actions to address user requirements. Current VLM-based mobile operating assistants can be structured into three levels: task, subtask, and action. The subtask level, linking high-level goals with low-level executable actions, is crucial for task completion but faces two challenges: ineffective subtasks that lower-level agent cannot execute and inefficient subtasks that fail to contribute to the completion of the higher-level task. These challenges stem from VLM's lack of experience in decomposing subtasks within GUI scenarios in multi-agent architecture. To address these, we propose a new mobile assistant architecture with constrained high-frequency o}ptimized planning (CHOP). Our approach overcomes the VLM's deficiency in GUI scenarios planning by using human-planned subtasks as the basis vector. We evaluate our architecture in both English and Chinese contexts across 20 Apps, demonstrating significant improvements in both effectiveness and efficiency. Our dataset and code is available at this https URL 

---
# Unified Mind Model: Reimagining Autonomous Agents in the LLM Era 

**Authors**: Pengbo Hu, Xiang Ying  

**Link**: [PDF](https://arxiv.org/pdf/2503.03459)  

**Abstract**: Large language models (LLMs) have recently demonstrated remarkable capabilities across domains, tasks, and languages (e.g., ChatGPT and GPT-4), reviving the research of general autonomous agents with human-like cognitive this http URL human-level agents require semantic comprehension and instruction-following capabilities, which exactly fall into the strengths of this http URL there have been several initial attempts to build human-level agents based on LLMs, the theoretical foundation remains a challenging open problem. In this paper, we propose a novel theoretical cognitive architecture, the Unified Mind Model (UMM), which offers guidance to facilitate the rapid creation of autonomous agents with human-level cognitive abilities. Specifically, our UMM starts with the global workspace theory and further leverage LLMs to enable the agent with various cognitive abilities, such as multi-modal perception, planning, reasoning, tool use, learning, memory, reflection and motivation. Building upon UMM, we then develop an agent-building engine, MindOS, which allows users to quickly create domain-/task-specific autonomous agents without any programming effort. 

---
# LiteWebAgent: The Open-Source Suite for VLM-Based Web-Agent Applications 

**Authors**: Danqing Zhang, Balaji Rama, Jingyi Ni, Shiying He, Fu Zhao, Kunyu Chen, Arnold Chen, Junyu Cao  

**Link**: [PDF](https://arxiv.org/pdf/2503.02950)  

**Abstract**: We introduce LiteWebAgent, an open-source suite for VLM-based web agent applications. Our framework addresses a critical gap in the web agent ecosystem with a production-ready solution that combines minimal serverless backend configuration, intuitive user and browser interfaces, and extensible research capabilities in agent planning, memory, and tree search. For the core LiteWebAgent agent framework, we implemented a simple yet effective baseline using recursive function calling, providing with decoupled action generation and action grounding. In addition, we integrate advanced research components such as agent planning, agent workflow memory, and tree search in a modular and extensible manner. We then integrate the LiteWebAgent agent framework with frontend and backend as deployed systems in two formats: (1) a production Vercel-based web application, which provides users with an agent-controlled remote browser, (2) a Chrome extension leveraging LiteWebAgent's API to control an existing Chrome browser via CDP (Chrome DevTools Protocol). The LiteWebAgent framework is available at this https URL, with deployed frontend at this https URL. 

---
# Parallelized Planning-Acting for Efficient LLM-based Multi-Agent Systems 

**Authors**: Yaoru Li, Shunyu Liu, Tongya Zheng, Mingli Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.03505)  

**Abstract**: Recent advancements in Large Language Model(LLM)-based Multi-Agent Systems(MAS) have demonstrated remarkable potential for tackling complex decision-making tasks. However, existing frameworks inevitably rely on serialized execution paradigms, where agents must complete sequential LLM planning before taking action. This fundamental constraint severely limits real-time responsiveness and adaptation, which is crucial in dynamic environments with ever-changing scenarios. In this paper, we propose a novel parallelized planning-acting framework for LLM-based MAS, featuring a dual-thread architecture with interruptible execution to enable concurrent planning and acting. Specifically, our framework comprises two core threads:(1) a planning thread driven by a centralized memory system, maintaining synchronization of environmental states and agent communication to support dynamic decision-making; and (2) an acting thread equipped with a comprehensive skill library, enabling automated task execution through recursive decomposition. Extensive experiments on challenging Minecraft demonstrate the effectiveness of the proposed framework. 

---
# COSINT-Agent: A Knowledge-Driven Multimodal Agent for Chinese Open Source Intelligence 

**Authors**: Wentao Li, Congcong Wang, Xiaoxiao Cui, Zhi Liu, Wei Guo, Lizhen Cui  

**Link**: [PDF](https://arxiv.org/pdf/2503.03215)  

**Abstract**: Open Source Intelligence (OSINT) requires the integration and reasoning of diverse multimodal data, presenting significant challenges in deriving actionable insights. Traditional approaches, including multimodal large language models (MLLMs), often struggle to infer complex contextual relationships or deliver comprehensive intelligence from unstructured data sources. In this paper, we introduce COSINT-Agent, a knowledge-driven multimodal agent tailored to address the challenges of OSINT in the Chinese domain. COSINT-Agent seamlessly integrates the perceptual capabilities of fine-tuned MLLMs with the structured reasoning power of the Entity-Event-Scene Knowledge Graph (EES-KG). Central to COSINT-Agent is the innovative EES-Match framework, which bridges COSINT-MLLM and EES-KG, enabling systematic extraction, reasoning, and contextualization of multimodal insights. This integration facilitates precise entity recognition, event interpretation, and context retrieval, effectively transforming raw multimodal data into actionable intelligence. Extensive experiments validate the superior performance of COSINT-Agent across core OSINT tasks, including entity recognition, EES generation, and context matching. These results underscore its potential as a robust and scalable solution for advancing automated multimodal reasoning and enhancing the effectiveness of OSINT methodologies. 

---
# Collaborative Expert LLMs Guided Multi-Objective Molecular Optimization 

**Authors**: Jiajun Yu, Yizhen Zheng, Huan Yee Koh, Shirui Pan, Tianyue Wang, Haishuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03503)  

**Abstract**: Molecular optimization is a crucial yet complex and time-intensive process that often acts as a bottleneck for drug development. Traditional methods rely heavily on trial and error, making multi-objective optimization both time-consuming and resource-intensive. Current AI-based methods have shown limited success in handling multi-objective optimization tasks, hampering their practical utilization. To address this challenge, we present MultiMol, a collaborative large language model (LLM) system designed to guide multi-objective molecular optimization. MultiMol comprises two agents, including a data-driven worker agent and a literature-guided research agent. The data-driven worker agent is a large language model being fine-tuned to learn how to generate optimized molecules considering multiple objectives, while the literature-guided research agent is responsible for searching task-related literature to find useful prior knowledge that facilitates identifying the most promising optimized candidates. In evaluations across six multi-objective optimization tasks, MultiMol significantly outperforms existing methods, achieving a 82.30% success rate, in sharp contrast to the 27.50% success rate of current strongest methods. To further validate its practical impact, we tested MultiMol on two real-world challenges. First, we enhanced the selectivity of Xanthine Amine Congener (XAC), a promiscuous ligand that binds both A1R and A2AR, successfully biasing it towards A1R. Second, we improved the bioavailability of Saquinavir, an HIV-1 protease inhibitor with known bioavailability limitations. Overall, these results indicate that MultiMol represents a highly promising approach for multi-objective molecular optimization, holding great potential to accelerate the drug development process and contribute to the advancement of pharmaceutical research. 

---
# Teaching AI to Handle Exceptions: Supervised Fine-Tuning with Human-Aligned Judgment 

**Authors**: Matthew DosSantos DiSorbo, Harang Ju, Sinan Aral  

**Link**: [PDF](https://arxiv.org/pdf/2503.02976)  

**Abstract**: Large language models (LLMs), initially developed for generative AI, are now evolving into agentic AI systems, which make decisions in complex, real-world contexts. Unfortunately, while their generative capabilities are well-documented, their decision-making processes remain poorly understood. This is particularly evident when models are handling exceptions, a critical and challenging aspect of decision-making made relevant by the inherent incompleteness of contracts. Here we demonstrate that LLMs, even ones that excel at reasoning, deviate significantly from human judgments because they adhere strictly to policies, even when such adherence is impractical, suboptimal, or even counterproductive. We then evaluate three approaches to tuning AI agents to handle exceptions: ethical framework prompting, chain-of-thought reasoning, and supervised fine-tuning. We find that while ethical framework prompting fails and chain-of-thought prompting provides only slight improvements, supervised fine-tuning, specifically with human explanations, yields markedly better results. Surprisingly, in our experiments, supervised fine-tuning even enabled models to generalize human-like decision-making to novel scenarios, demonstrating transfer learning of human-aligned decision-making across contexts. Furthermore, fine-tuning with explanations, not just labels, was critical for alignment, suggesting that aligning LLMs with human judgment requires explicit training on how decisions are made, not just which decisions are made. These findings highlight the need to address LLMs' shortcomings in handling exceptions in order to guide the development of agentic AI toward models that can effectively align with human judgment and simultaneously adapt to novel contexts. 

---
# AI-Enabled Conversational Journaling for Advancing Parkinson's Disease Symptom Tracking 

**Authors**: Mashrur Rashik, Shilpa Sweth, Nishtha Agrawal, Saiyyam Kochar, Kara M Smith, Fateme Rajabiyazdi, Vidya Setlur, Narges Mahyar, Ali Sarvghad  

**Link**: [PDF](https://arxiv.org/pdf/2503.03532)  

**Abstract**: Journaling plays a crucial role in managing chronic conditions by allowing patients to document symptoms and medication intake, providing essential data for long-term care. While valuable, traditional journaling methods often rely on static, self-directed entries, lacking interactive feedback and real-time guidance. This gap can result in incomplete or imprecise information, limiting its usefulness for effective treatment. To address this gap, we introduce PATRIKA, an AI-enabled prototype designed specifically for people with Parkinson's disease (PwPD). The system incorporates cooperative conversation principles, clinical interview simulations, and personalization to create a more effective and user-friendly journaling experience. Through two user studies with PwPD and iterative refinement of PATRIKA, we demonstrate conversational journaling's significant potential in patient engagement and collecting clinically valuable information. Our results showed that generating probing questions PATRIKA turned journaling into a bi-directional interaction. Additionally, we offer insights for designing journaling systems for healthcare and future directions for promoting sustained journaling. 

---
# SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Safe Reinforcement Learning 

**Authors**: Borong Zhang, Yuhao Zhang, Jiaming Ji, Yingshan Lei, Josef Dai, Yuanpei Chen, Yaodong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03480)  

**Abstract**: Vision-language-action models (VLAs) have shown great potential as generalist robot policies. However, these models pose urgent safety challenges during deployment, including the risk of physical harm to the environment, the robot itself, and humans. How can safety be explicitly incorporated into VLAs? In this work, we propose SafeVLA, a novel algorithm designed to integrate safety into VLAs, ensuring the protection of the environment, robot hardware and humans in real-world settings. SafeVLA effectively balances safety and task performance by employing large-scale constrained learning within simulated environments. We demonstrate that SafeVLA outperforms the current state-of-the-art method in both safety and task performance, achieving average improvements of 83.58% and 3.85%, respectively, in simulation. By prioritizing safety, our approach eliminates high-risk behaviors and reduces the upper bound of unsafe behaviors to 1/35 of that in the current state-of-the-art, thereby significantly mitigating long-tail risks. Furthermore, the learned safety constraints generalize to diverse, unseen scenarios, including multiple out-of-distribution perturbations and tasks. Our data, models and newly proposed benchmark environment are available at this https URL. 

---
# Multi-Agent DRL for Queue-Aware Task Offloading in Hierarchical MEC-Enabled Air-Ground Networks 

**Authors**: Muhammet Hevesli, Abegaz Mohammed Seid, Aiman Erbad, Mohamed Abdallah  

**Link**: [PDF](https://arxiv.org/pdf/2503.03391)  

**Abstract**: Mobile edge computing (MEC)-enabled air-ground networks are a key component of 6G, employing aerial base stations (ABSs) such as unmanned aerial vehicles (UAVs) and high-altitude platform stations (HAPS) to provide dynamic services to ground IoT devices (IoTDs). These IoTDs support real-time applications (e.g., multimedia and Metaverse services) that demand high computational resources and strict quality of service (QoS) guarantees in terms of latency and task queue management. Given their limited energy and processing capabilities, IoTDs rely on UAVs and HAPS to offload tasks for distributed processing, forming a multi-tier MEC system. This paper tackles the overall energy minimization problem in MEC-enabled air-ground integrated networks (MAGIN) by jointly optimizing UAV trajectories, computing resource allocation, and queue-aware task offloading decisions. The optimization is challenging due to the nonconvex, nonlinear nature of this hierarchical system, which renders traditional methods ineffective. We reformulate the problem as a multi-agent Markov decision process (MDP) with continuous action spaces and heterogeneous agents, and propose a novel variant of multi-agent proximal policy optimization with a Beta distribution (MAPPO-BD) to solve it. Extensive simulations show that MAPPO-BD outperforms baseline schemes, achieving superior energy savings and efficient resource management in MAGIN while meeting queue delay and edge computing constraints. 

---
# Exploring the Potential of Large Language Models as Predictors in Dynamic Text-Attributed Graphs 

**Authors**: Runlin Lei, Jiarui Ji, Haipeng Ding, Lu Yi, Zhewei Wei, Yongchao Liu, Chuntao Hong  

**Link**: [PDF](https://arxiv.org/pdf/2503.03258)  

**Abstract**: With the rise of large language models (LLMs), there has been growing interest in Graph Foundation Models (GFMs) for graph-based tasks. By leveraging LLMs as predictors, GFMs have demonstrated impressive generalizability across various tasks and datasets. However, existing research on LLMs as predictors has predominantly focused on static graphs, leaving their potential in dynamic graph prediction unexplored. In this work, we pioneer using LLMs for predictive tasks on dynamic graphs. We identify two key challenges: the constraints imposed by context length when processing large-scale historical data and the significant variability in domain characteristics, both of which complicate the development of a unified predictor. To address these challenges, we propose the GraphAgent-Dynamic (GAD) Framework, a multi-agent system that leverages collaborative LLMs. In contrast to using a single LLM as the predictor, GAD incorporates global and local summary agents to generate domain-specific knowledge, enhancing its transferability across domains. Additionally, knowledge reflection agents enable adaptive updates to GAD's knowledge, maintaining a unified and self-consistent architecture. In experiments, GAD demonstrates performance comparable to or even exceeds that of full-supervised graph neural networks without dataset-specific training. Finally, to enhance the task-specific performance of LLM-based predictors, we discuss potential improvements, such as dataset-specific fine-tuning to LLMs. By developing tailored strategies for different tasks, we provide new insights for the future design of LLM-based predictors. 

---
# Less is more? Rewards in RL for Cyber Defence 

**Authors**: Elizabeth Bates, Chris Hicks, Vasilios Mavroudis  

**Link**: [PDF](https://arxiv.org/pdf/2503.03245)  

**Abstract**: The last few years has seen an explosion of interest in autonomous cyber defence agents based on deep reinforcement learning. Such agents are typically trained in a cyber gym environment, also known as a cyber simulator, at least 32 of which have already been built. Most, if not all cyber gyms provide dense "scaffolded" reward functions which combine many penalties or incentives for a range of (un)desirable states and costly actions. Whilst dense rewards help alleviate the challenge of exploring complex environments, yielding seemingly effective strategies from relatively few environment steps; they are also known to bias the solutions an agent can find, potentially towards suboptimal solutions. Sparse rewards could offer preferable or more effective solutions and have been overlooked by cyber gyms to date. In this work we set out to evaluate whether sparse reward functions might enable training more effective cyber defence agents. Towards this goal we first break down several evaluation limitations in existing work by proposing a ground truth evaluation score that goes beyond the standard RL paradigm used to train and evaluate agents. By adapting a well-established cyber gym to accommodate our methodology and ground truth score, we propose and evaluate two sparse reward mechanisms and compare them with a typical dense reward. Our evaluation considers a range of network sizes, from 2 to 50 nodes, and both reactive and proactive defensive actions. Our results show that sparse rewards, particularly positive reinforcement for an uncompromised network state, enable the training of more effective cyber defence agents. Furthermore, we show that sparse rewards provide more stable training than dense rewards, and that both effectiveness and training stability are robust to a variety of cyber environment considerations. 

---
# Reliable and Efficient Multi-Agent Coordination via Graph Neural Network Variational Autoencoders 

**Authors**: Yue Meng, Nathalie Majcherczyk, Wenliang Liu, Scott Kiesel, Chuchu Fan, Federico Pecora  

**Link**: [PDF](https://arxiv.org/pdf/2503.02954)  

**Abstract**: Multi-agent coordination is crucial for reliable multi-robot navigation in shared spaces such as automated warehouses. In regions of dense robot traffic, local coordination methods may fail to find a deadlock-free solution. In these scenarios, it is appropriate to let a central unit generate a global schedule that decides the passing order of robots. However, the runtime of such centralized coordination methods increases significantly with the problem scale. In this paper, we propose to leverage Graph Neural Network Variational Autoencoders (GNN-VAE) to solve the multi-agent coordination problem at scale faster than through centralized optimization. We formulate the coordination problem as a graph problem and collect ground truth data using a Mixed-Integer Linear Program (MILP) solver. During training, our learning framework encodes good quality solutions of the graph problem into a latent space. At inference time, solution samples are decoded from the sampled latent variables, and the lowest-cost sample is selected for coordination. Finally, the feasible proposal with the highest performance index is selected for the deployment. By construction, our GNN-VAE framework returns solutions that always respect the constraints of the considered coordination problem. Numerical results show that our approach trained on small-scale problems can achieve high-quality solutions even for large-scale problems with 250 robots, being much faster than other baselines. Project page: this https URL 

---
# Interactive Debugging and Steering of Multi-Agent AI Systems 

**Authors**: Will Epperson, Gagan Bansal, Victor Dibia, Adam Fourney, Jack Gerrits, Erkang Zhu, Saleema Amershi  

**Link**: [PDF](https://arxiv.org/pdf/2503.02068)  

**Abstract**: Fully autonomous teams of LLM-powered AI agents are emerging that collaborate to perform complex tasks for users. What challenges do developers face when trying to build and debug these AI agent teams? In formative interviews with five AI agent developers, we identify core challenges: difficulty reviewing long agent conversations to localize errors, lack of support in current tools for interactive debugging, and the need for tool support to iterate on agent configuration. Based on these needs, we developed an interactive multi-agent debugging tool, AGDebugger, with a UI for browsing and sending messages, the ability to edit and reset prior agent messages, and an overview visualization for navigating complex message histories. In a two-part user study with 14 participants, we identify common user strategies for steering agents and highlight the importance of interactive message resets for debugging. Our studies deepen understanding of interfaces for debugging increasingly important agentic workflows. 

---
# Function-Coherent Gambles with Non-Additive Sequential Dynamics 

**Authors**: Gregory Wheeler  

**Link**: [PDF](https://arxiv.org/pdf/2503.02889)  

**Abstract**: The desirable gambles framework provides a rigorous foundation for imprecise probability theory but relies heavily on linear utility via its coherence axioms. In our related work, we introduced function-coherent gambles to accommodate non-linear utility. However, when repeated gambles are played over time -- especially in intertemporal choice where rewards compound multiplicatively -- the standard additive combination axiom fails to capture the appropriate long-run evaluation. In this paper we extend the framework by relaxing the additive combination axiom and introducing a nonlinear combination operator that effectively aggregates repeated gambles in the log-domain. This operator preserves the time-average (geometric) growth rate and addresses the ergodicity problem. We prove the key algebraic properties of the operator, discuss its impact on coherence, risk assessment, and representation, and provide a series of illustrative examples. Our approach bridges the gap between expectation values and time averages and unifies normative theory with empirically observed non-stationary reward dynamics. 

---
# Towards Robust Multi-UAV Collaboration: MARL with Noise-Resilient Communication and Attention Mechanisms 

**Authors**: Zilin Zhao, Chishui Chen, Haotian Shi, Jiale Chen, Xuanlin Yue, Zhejian Yang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.02913)  

**Abstract**: Efficient path planning for unmanned aerial vehicles (UAVs) is crucial in remote sensing and information collection. As task scales expand, the cooperative deployment of multiple UAVs significantly improves information collection efficiency. However, collaborative communication and decision-making for multiple UAVs remain major challenges in path planning, especially in noisy environments. To efficiently accomplish complex information collection tasks in 3D space and address robust communication issues, we propose a multi-agent reinforcement learning (MARL) framework for UAV path planning based on the Counterfactual Multi-Agent Policy Gradients (COMA) algorithm. The framework incorporates attention mechanism-based UAV communication protocol and training-deployment system, significantly improving communication robustness and individual decision-making capabilities in noisy conditions. Experiments conducted on both synthetic and real-world datasets demonstrate that our method outperforms existing algorithms in terms of path planning efficiency and robustness, especially in noisy environments, achieving a 78\% improvement in entropy reduction. 

---
# MAS-GPT: Training LLMs to Build LLM-based Multi-Agent Systems 

**Authors**: Rui Ye, Shuo Tang, Rui Ge, Yaxin Du, Zhenfei Yin, Siheng Chen, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2503.03686)  

**Abstract**: LLM-based multi-agent systems (MAS) have shown significant potential in tackling diverse tasks. However, to design effective MAS, existing approaches heavily rely on manual configurations or multiple calls of advanced LLMs, resulting in inadaptability and high inference costs. In this paper, we simplify the process of building an MAS by reframing it as a generative language task, where the input is a user query and the output is a corresponding MAS. To address this novel task, we unify the representation of MAS as executable code and propose a consistency-oriented data construction pipeline to create a high-quality dataset comprising coherent and consistent query-MAS pairs. Using this dataset, we train MAS-GPT, an open-source medium-sized LLM that is capable of generating query-adaptive MAS within a single LLM inference. The generated MAS can be seamlessly applied to process user queries and deliver high-quality responses. Extensive experiments on 9 benchmarks and 5 LLMs show that the proposed MAS-GPT consistently outperforms 10+ baseline MAS methods on diverse settings, indicating MAS-GPT's high effectiveness, efficiency and strong generalization ability. Code will be available at this https URL. 

---
# Taxation Perspectives from Large Language Models: A Case Study on Additional Tax Penalties 

**Authors**: Eunkyung Choi, Young Jin Suh, Hun Park, Wonseok Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03444)  

**Abstract**: How capable are large language models (LLMs) in the domain of taxation? Although numerous studies have explored the legal domain in general, research dedicated to taxation remain scarce. Moreover, the datasets used in these studies are either simplified, failing to reflect the real-world complexities, or unavailable as open source. To address this gap, we introduce PLAT, a new benchmark designed to assess the ability of LLMs to predict the legitimacy of additional tax penalties. PLAT is constructed to evaluate LLMs' understanding of tax law, particularly in cases where resolving the issue requires more than just applying related statutes. Our experiments with six LLMs reveal that their baseline capabilities are limited, especially when dealing with conflicting issues that demand a comprehensive understanding. However, we found that enabling retrieval, self-reasoning, and discussion among multiple agents with specific role assignments, this limitation can be mitigated. 

---
