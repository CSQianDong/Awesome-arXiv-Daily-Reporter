# Enhancing Web Agents with Explicit Rollback Mechanisms 

**Authors**: Zhisong Zhang, Tianqing Fang, Kaixin Ma, Wenhao Yu, Hongming Zhang, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11788)  

**Abstract**: With recent advancements in large language models, web agents have been greatly improved. However, dealing with complex and dynamic web environments requires more advanced planning and search abilities. Previous studies usually adopt a greedy one-way search strategy, which may struggle to recover from erroneous states. In this work, we enhance web agents with an explicit rollback mechanism, enabling the agent to revert back to a previous state in its navigation trajectory. This mechanism gives the model the flexibility to directly control the search process, leading to an effective and efficient web navigation method. We conduct experiments on two live web navigation benchmarks with zero-shot and fine-tuning settings. The results demonstrate the effectiveness of our proposed approach. 

---
# FiSMiness: A Finite State Machine Based Paradigm for Emotional Support Conversations 

**Authors**: Yue Zhao, Qingqing Gu, Xiaoyu Wang, Teng Chen, Zhonglin Jiang, Yong Chen, Luo Ji  

**Link**: [PDF](https://arxiv.org/pdf/2504.11837)  

**Abstract**: Emotional support conversation (ESC) aims to alleviate the emotional distress of individuals through effective conversations. Although large language models (LLMs) have obtained remarkable progress on ESC, most of these studies might not define the diagram from the state model perspective, therefore providing a suboptimal solution for long-term satisfaction. To address such an issue, we leverage the Finite State Machine (FSM) on LLMs, and propose a framework called FiSMiness. Our framework allows a single LLM to bootstrap the planning during ESC, and self-reason the seeker's emotion, support strategy and the final response upon each conversational turn. Substantial experiments on ESC datasets suggest that FiSMiness outperforms many baselines, including direct inference, self-refine, chain of thought, finetuning, and external-assisted methods, even those with many more parameters. 

---
# ReTool: Reinforcement Learning for Strategic Tool Use in LLMs 

**Authors**: Jiazhan Feng, Shijue Huang, Xingwei Qu, Ge Zhang, Yujia Qin, Baoquan Zhong, Chengquan Jiang, Jinxin Chi, Wanjun Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2504.11536)  

**Abstract**: While reasoning models (e.g., DeepSeek R1) trained with reinforcement learning (RL), excel in textual reasoning, they struggle in scenarios requiring structured problem-solving, such as geometric reasoning, concise computation, or complex equation solving-areas where computational tools like code interpreters (CI) demonstrate distinct advantages. To bridge this gap, we propose ReTool, which enhances long-form reasoning with tool-integrated learning, including two key features: (1) dynamic interleaving of real-time code execution within natural language reasoning processes, and (2) an automated RL paradigm that allows policy rollouts with multi-turn real-time code execution and teaches the model in learning when and how to invoke tools based on outcome feedback. ReTool employs a systematic training framework, beginning with synthetic cold-start data generation to produce code-augmented long-form reasoning traces for fine-tuning base models. Subsequent RL training leverages task outcomes as rewards to iteratively refine the model's tool use strategy, enabling autonomous discovery of optimal tool invocation patterns without human priors. Experiments on the challenging MATH Olympiad benchmark AIME demonstrate ReTool's superiority: Our 32B model achieves 67% accuracy with 400 training steps, outperforming text-based RL baseline (40% accuracy, 1080 steps) in efficiency and performance. Remarkably, ReTool-32B attains 72.5% accuracy in extended settings, surpassing OpenAI's o1-preview by 27.9%. Further analysis reveals emergent behaviors such as code self-correction, signaling an ''aha moment'' in which the model autonomously masters adaptive tool use. These findings highlight the promise of outcome-driven tool integration for advancing complex mathematical reasoning and offer new insights into hybrid neuro-symbolic systems. 

---
# GraphicBench: A Planning Benchmark for Graphic Design with Language Agents 

**Authors**: Dayeon Ki, Tianyi Zhou, Marine Carpuat, Gang Wu, Puneet Mathur, Viswanathan Swaminathan  

**Link**: [PDF](https://arxiv.org/pdf/2504.11571)  

**Abstract**: Large Language Model (LLM)-powered agents have unlocked new possibilities for automating human tasks. While prior work has focused on well-defined tasks with specified goals, the capabilities of agents in creative design tasks with open-ended goals remain underexplored. We introduce GraphicBench, a new planning benchmark for graphic design that covers 1,079 user queries and input images across four design types. We further present GraphicTown, an LLM agent framework with three design experts and 46 actions (tools) to choose from for executing each step of the planned workflows in web environments. Experiments with six LLMs demonstrate their ability to generate workflows that integrate both explicit design constraints from user queries and implicit commonsense constraints. However, these workflows often do not lead to successful execution outcomes, primarily due to challenges in: (1) reasoning about spatial relationships, (2) coordinating global dependencies across experts, and (3) retrieving the most appropriate action per step. We envision GraphicBench as a challenging yet valuable testbed for advancing LLM-agent planning and execution in creative design tasks. 

---
# Steering Prosocial AI Agents: Computational Basis of LLM's Decision Making in Social Simulation 

**Authors**: Ji Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.11671)  

**Abstract**: Large language models (LLMs) increasingly serve as human-like decision-making agents in social science and applied settings. These LLM-agents are typically assigned human-like characters and placed in real-life contexts. However, how these characters and contexts shape an LLM's behavior remains underexplored. This study proposes and tests methods for probing, quantifying, and modifying an LLM's internal representations in a Dictator Game -- a classic behavioral experiment on fairness and prosocial behavior. We extract ``vectors of variable variations'' (e.g., ``male'' to ``female'') from the LLM's internal state. Manipulating these vectors during the model's inference can substantially alter how those variables relate to the model's decision-making. This approach offers a principled way to study and regulate how social concepts can be encoded and engineered within transformer-based models, with implications for alignment, debiasing, and designing AI agents for social simulations in both academic and commercial applications. 

---
# Towards LLM Agents for Earth Observation 

**Authors**: Chia Hsiang Kao, Wenting Zhao, Shreelekha Revankar, Samuel Speas, Snehal Bhagat, Rajeev Datta, Cheng Perng Phoo, Utkarsh Mall, Carl Vondrick, Kavita Bala, Bharath Hariharan  

**Link**: [PDF](https://arxiv.org/pdf/2504.12110)  

**Abstract**: Earth Observation (EO) provides critical planetary data for environmental monitoring, disaster management, climate science, and other scientific domains. Here we ask: Are AI systems ready for reliable Earth Observation? We introduce \datasetnamenospace, a benchmark of 140 yes/no questions from NASA Earth Observatory articles across 13 topics and 17 satellite sensors. Using Google Earth Engine API as a tool, LLM agents can only achieve an accuracy of 33% because the code fails to run over 58% of the time. We improve the failure rate for open models by fine-tuning synthetic data, allowing much smaller models (Llama-3.1-8B) to achieve comparable accuracy to much larger ones (e.g., DeepSeek-R1). Taken together, our findings identify significant challenges to be solved before AI agents can automate earth observation, and suggest paths forward. The project page is available at this https URL. 

---
# REAL: Benchmarking Autonomous Agents on Deterministic Simulations of Real Websites 

**Authors**: Divyansh Garg, Shaun VanWeelden, Diego Caples, Andis Draguns, Nikil Ravi, Pranav Putta, Naman Garg, Tomas Abraham, Michael Lara, Federico Lopez, James Liu, Atharva Gundawar, Prannay Hebbar, Youngchul Joo, Charles London, Christian Schroeder de Witt, Sumeet Motwani  

**Link**: [PDF](https://arxiv.org/pdf/2504.11543)  

**Abstract**: We introduce REAL, a benchmark and framework for multi-turn agent evaluations on deterministic simulations of real-world websites. REAL comprises high-fidelity, deterministic replicas of 11 widely-used websites across domains such as e-commerce, travel, communication, and professional networking. We also release a benchmark consisting of 112 practical tasks that mirror everyday complex user interactions requiring both accurate information retrieval and state-changing actions. All interactions occur within this fully controlled setting, eliminating safety risks and enabling robust, reproducible evaluation of agent capability and reliability. Our novel evaluation framework combines programmatic checks of website state for action-based tasks with rubric-guided LLM-based judgments for information retrieval. The framework supports both open-source and proprietary agent systems through a flexible evaluation harness that accommodates black-box commands within browser environments, allowing research labs to test agentic systems without modification. Our empirical results show that frontier language models achieve at most a 41% success rate on REAL, highlighting critical gaps in autonomous web navigation and task completion capabilities. Our framework supports easy integration of new tasks, reproducible evaluation, and scalable data generation for training web agents. The websites, framework, and leaderboard are available at this https URL and this https URL. 

---
# Enhancing Autonomous Driving Systems with On-Board Deployed Large Language Models 

**Authors**: Nicolas Baumann, Cheng Hu, Paviththiren Sivasothilingam, Haotong Qin, Lei Xie, Michele Magno, Luca Benini  

**Link**: [PDF](https://arxiv.org/pdf/2504.11514)  

**Abstract**: Neural Networks (NNs) trained through supervised learning struggle with managing edge-case scenarios common in real-world driving due to the intractability of exhaustive datasets covering all edge-cases, making knowledge-driven approaches, akin to how humans intuitively detect unexpected driving behavior, a suitable complement to data-driven methods. This work proposes a hybrid architecture combining low-level Model Predictive Controller (MPC) with locally deployed Large Language Models (LLMs) to enhance decision-making and Human Machine Interaction (HMI). The DecisionxLLM module evaluates robotic state information against natural language instructions to ensure adherence to desired driving behavior. The MPCxLLM module then adjusts MPC parameters based on LLM-generated insights, achieving control adaptability while preserving the safety and constraint guarantees of traditional MPC systems. Further, to enable efficient on-board deployment and to eliminate dependency on cloud connectivity, we shift processing to the on-board computing platform: We propose an approach that exploits Retrieval Augmented Generation (RAG), Low Rank Adaptation (LoRA) fine-tuning, and quantization. Experimental results demonstrate that these enhancements yield significant improvements in reasoning accuracy by up to 10.45%, control adaptability by as much as 52.2%, and up to 10.5x increase in computational efficiency (tokens/s), validating the proposed framework's practicality for real-time deployment even on down-scaled robotic platforms. This work bridges high-level decision-making with low-level control adaptability, offering a synergistic framework for knowledge-driven and adaptive Autonomous Driving Systems (ADS). 

---
# ARCeR: an Agentic RAG for the Automated Definition of Cyber Ranges 

**Authors**: Matteo Lupinacci, Francesco Blefari, Francesco Romeo, Francesco Aurelio Pironti, Angelo Furfaro  

**Link**: [PDF](https://arxiv.org/pdf/2504.12143)  

**Abstract**: The growing and evolving landscape of cybersecurity threats necessitates the development of supporting tools and platforms that allow for the creation of realistic IT environments operating within virtual, controlled settings as Cyber Ranges (CRs). CRs can be exploited for analyzing vulnerabilities and experimenting with the effectiveness of devised countermeasures, as well as serving as training environments for building cyber security skills and abilities for IT operators. This paper proposes ARCeR as an innovative solution for the automatic generation and deployment of CRs, starting from user-provided descriptions in a natural language. ARCeR relies on the Agentic RAG paradigm, which allows it to fully exploit state-of-art AI technologies. Experimental results show that ARCeR is able to successfully process prompts even in cases that LLMs or basic RAG systems are not able to cope with. Furthermore, ARCeR is able to target any CR framework provided that specific knowledge is made available to it. 

---
# Causality-enhanced Decision-Making for Autonomous Mobile Robots in Dynamic Environments 

**Authors**: Luca Castri, Gloria Beraldo, Nicola Bellotto  

**Link**: [PDF](https://arxiv.org/pdf/2504.11901)  

**Abstract**: The growing integration of robots in shared environments -- such as warehouses, shopping centres, and hospitals -- demands a deep understanding of the underlying dynamics and human behaviours, including how, when, and where individuals engage in various activities and interactions. This knowledge goes beyond simple correlation studies and requires a more comprehensive causal analysis. By leveraging causal inference to model cause-and-effect relationships, we can better anticipate critical environmental factors and enable autonomous robots to plan and execute tasks more effectively. To this end, we propose a novel causality-based decision-making framework that reasons over a learned causal model to predict battery usage and human obstructions, understanding how these factors could influence robot task execution. Such reasoning framework assists the robot in deciding when and how to complete a given task. To achieve this, we developed also PeopleFlow, a new Gazebo-based simulator designed to model context-sensitive human-robot spatial interactions in shared workspaces. PeopleFlow features realistic human and robot trajectories influenced by contextual factors such as time, environment layout, and robot state, and can simulate a large number of agents. While the simulator is general-purpose, in this paper we focus on a warehouse-like environment as a case study, where we conduct an extensive evaluation benchmarking our causal approach against a non-causal baseline. Our findings demonstrate the efficacy of the proposed solutions, highlighting how causal reasoning enables autonomous robots to operate more efficiently and safely in dynamic environments shared with humans. 

---
# Progent: Programmable Privilege Control for LLM Agents 

**Authors**: Tianneng Shi, Jingxuan He, Zhun Wang, Linyu Wu, Hongwei Li, Wenbo Guo, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2504.11703)  

**Abstract**: LLM agents are an emerging form of AI systems where large language models (LLMs) serve as the central component, utilizing a diverse set of tools to complete user-assigned tasks. Despite their great potential, LLM agents pose significant security risks. When interacting with the external world, they may encounter malicious commands from attackers, leading to the execution of dangerous actions. A promising way to address this is by enforcing the principle of least privilege: allowing only essential actions for task completion while blocking unnecessary ones. However, achieving this is challenging, as it requires covering diverse agent scenarios while preserving both security and utility.
We introduce Progent, the first privilege control mechanism for LLM agents. At its core is a domain-specific language for flexibly expressing privilege control policies applied during agent execution. These policies provide fine-grained constraints over tool calls, deciding when tool calls are permissible and specifying fallbacks if they are not. This enables agent developers and users to craft suitable policies for their specific use cases and enforce them deterministically to guarantee security. Thanks to its modular design, integrating Progent does not alter agent internals and requires only minimal changes to agent implementation, enhancing its practicality and potential for widespread adoption. To automate policy writing, we leverage LLMs to generate policies based on user queries, which are then updated dynamically for improved security and utility. Our extensive evaluation shows that it enables strong security while preserving high utility across three distinct scenarios or benchmarks: AgentDojo, ASB, and AgentPoison. Furthermore, we perform an in-depth analysis, showcasing the effectiveness of its core components and the resilience of its automated policy generation against adaptive attacks. 

---
# GrabS: Generative Embodied Agent for 3D Object Segmentation without Scene Supervision 

**Authors**: Zihui Zhang, Yafei Yang, Hongtao Wen, Bo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.11754)  

**Abstract**: We study the hard problem of 3D object segmentation in complex point clouds without requiring human labels of 3D scenes for supervision. By relying on the similarity of pretrained 2D features or external signals such as motion to group 3D points as objects, existing unsupervised methods are usually limited to identifying simple objects like cars or their segmented objects are often inferior due to the lack of objectness in pretrained features. In this paper, we propose a new two-stage pipeline called GrabS. The core concept of our method is to learn generative and discriminative object-centric priors as a foundation from object datasets in the first stage, and then design an embodied agent to learn to discover multiple objects by querying against the pretrained generative priors in the second stage. We extensively evaluate our method on two real-world datasets and a newly created synthetic dataset, demonstrating remarkable segmentation performance, clearly surpassing all existing unsupervised methods. 

---
# Toward Aligning Human and Robot Actions via Multi-Modal Demonstration Learning 

**Authors**: Azizul Zahid, Jie Fan, Farong Wang, Ashton Dy, Sai Swaminathan, Fei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11493)  

**Abstract**: Understanding action correspondence between humans and robots is essential for evaluating alignment in decision-making, particularly in human-robot collaboration and imitation learning within unstructured environments. We propose a multimodal demonstration learning framework that explicitly models human demonstrations from RGB video with robot demonstrations in voxelized RGB-D space. Focusing on the "pick and place" task from the RH20T dataset, we utilize data from 5 users across 10 diverse scenes. Our approach combines ResNet-based visual encoding for human intention modeling and a Perceiver Transformer for voxel-based robot action prediction. After 2000 training epochs, the human model reaches 71.67% accuracy, and the robot model achieves 71.8% accuracy, demonstrating the framework's potential for aligning complex, multimodal human and robot behaviors in manipulation tasks. 

---
