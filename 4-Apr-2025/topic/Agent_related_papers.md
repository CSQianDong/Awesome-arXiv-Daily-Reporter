# Multi-Mission Tool Bench: Assessing the Robustness of LLM based Agents through Related and Dynamic Missions 

**Authors**: PeiJie Yu, Yifan Yang, Jinjian Li, Zelong Zhang, Haorui Wang, Xiao Feng, Feng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02623)  

**Abstract**: Large language models (LLMs) demonstrate strong potential as agents for tool invocation due to their advanced comprehension and planning capabilities. Users increasingly rely on LLM-based agents to solve complex missions through iterative interactions. However, existing benchmarks predominantly access agents in single-mission scenarios, failing to capture real-world complexity. To bridge this gap, we propose the Multi-Mission Tool Bench. In the benchmark, each test case comprises multiple interrelated missions. This design requires agents to dynamically adapt to evolving demands. Moreover, the proposed benchmark explores all possible mission-switching patterns within a fixed mission number. Specifically, we propose a multi-agent data generation framework to construct the benchmark. We also propose a novel method to evaluate the accuracy and efficiency of agent decisions with dynamic decision trees. Experiments on diverse open-source and closed-source LLMs reveal critical factors influencing agent robustness and provide actionable insights to the tool invocation society. 

---
# The Self-Learning Agent with a Progressive Neural Network Integrated Transformer 

**Authors**: Ajay Sivakumar, Shalini, Vasantha Raj, Sebastian Sylvester  

**Link**: [PDF](https://arxiv.org/pdf/2504.02489)  

**Abstract**: This paper introduces a self-learning agent that integrates LLaMA 3.2 with a Progressive Neural Network (PNN) for continual learning in conversational AI and code generation. The framework dynamically collects data, fine-tunes tasks with minimal samples, and leverages Meta-Learning for rapid adaptation. LoRA optimizes fine-tuning, while Elastic Weight Consolidation (EWC) enhances knowledge retention. Experimental results demonstrate improved adaptability and memory stability, positioning this approach as a scalable step toward Artificial General Intelligence (AGI). 

---
# Advances and Challenges in Foundation Agents: From Brain-Inspired Intelligence to Evolutionary, Collaborative, and Safe Systems 

**Authors**: Bang Liu, Xinfeng Li, Jiayi Zhang, Jinlin Wang, Tanjin He, Sirui Hong, Hongzhang Liu, Shaokun Zhang, Kaitao Song, Kunlun Zhu, Yuheng Cheng, Suyuchen Wang, Xiaoqiang Wang, Yuyu Luo, Haibo Jin, Peiyan Zhang, Ollie Liu, Jiaqi Chen, Huan Zhang, Zhaoyang Yu, Haochen Shi, Boyan Li, Dekun Wu, Fengwei Teng, Xiaojun Jia, Jiawei Xu, Jinyu Xiang, Yizhang Lin, Tianming Liu, Tongliang Liu, Yu Su, Huan Sun, Glen Berseth, Jianyun Nie, Ian Foster, Logan Ward, Qingyun Wu, Yu Gu, Mingchen Zhuge, Xiangru Tang, Haohan Wang, Jiaxuan You, Chi Wang, Jian Pei, Qiang Yang, Xiaoliang Qi, Chenglin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.01990)  

**Abstract**: The advent of large language models (LLMs) has catalyzed a transformative shift in artificial intelligence, paving the way for advanced intelligent agents capable of sophisticated reasoning, robust perception, and versatile action across diverse domains. As these agents increasingly drive AI research and practical applications, their design, evaluation, and continuous improvement present intricate, multifaceted challenges. This survey provides a comprehensive overview, framing intelligent agents within a modular, brain-inspired architecture that integrates principles from cognitive science, neuroscience, and computational research. We structure our exploration into four interconnected parts. First, we delve into the modular foundation of intelligent agents, systematically mapping their cognitive, perceptual, and operational modules onto analogous human brain functionalities, and elucidating core components such as memory, world modeling, reward processing, and emotion-like systems. Second, we discuss self-enhancement and adaptive evolution mechanisms, exploring how agents autonomously refine their capabilities, adapt to dynamic environments, and achieve continual learning through automated optimization paradigms, including emerging AutoML and LLM-driven optimization strategies. Third, we examine collaborative and evolutionary multi-agent systems, investigating the collective intelligence emerging from agent interactions, cooperation, and societal structures, highlighting parallels to human social dynamics. Finally, we address the critical imperative of building safe, secure, and beneficial AI systems, emphasizing intrinsic and extrinsic security threats, ethical alignment, robustness, and practical mitigation strategies necessary for trustworthy real-world deployment. 

---
# SymDQN: Symbolic Knowledge and Reasoning in Neural Network-based Reinforcement Learning 

**Authors**: Ivo Amador, Nina Gierasimczuk  

**Link**: [PDF](https://arxiv.org/pdf/2504.02654)  

**Abstract**: We propose a learning architecture that allows symbolic control and guidance in reinforcement learning with deep neural networks. We introduce SymDQN, a novel modular approach that augments the existing Dueling Deep Q-Networks (DuelDQN) architecture with modules based on the neuro-symbolic framework of Logic Tensor Networks (LTNs). The modules guide action policy learning and allow reinforcement learning agents to display behaviour consistent with reasoning about the environment. Our experiment is an ablation study performed on the modules. It is conducted in a reinforcement learning environment of a 5x5 grid navigated by an agent that encounters various shapes, each associated with a given reward. The underlying DuelDQN attempts to learn the optimal behaviour of the agent in this environment, while the modules facilitate shape recognition and reward prediction. We show that our architecture significantly improves learning, both in terms of performance and the precision of the agent. The modularity of SymDQN allows reflecting on the intricacies and complexities of combining neural and symbolic approaches in reinforcement learning. 

---
# Autonomous Human-Robot Interaction via Operator Imitation 

**Authors**: Sammy Christen, David Müller, Agon Serifi, Ruben Grandia, Georg Wiedebach, Michael A. Hopkins, Espen Knoop, Moritz Bächer  

**Link**: [PDF](https://arxiv.org/pdf/2504.02724)  

**Abstract**: Teleoperated robotic characters can perform expressive interactions with humans, relying on the operators' experience and social intuition. In this work, we propose to create autonomous interactive robots, by training a model to imitate operator data. Our model is trained on a dataset of human-robot interactions, where an expert operator is asked to vary the interactions and mood of the robot, while the operator commands as well as the pose of the human and robot are recorded. Our approach learns to predict continuous operator commands through a diffusion process and discrete commands through a classifier, all unified within a single transformer architecture. We evaluate the resulting model in simulation and with a user study on the real system. We show that our method enables simple autonomous human-robot interactions that are comparable to the expert-operator baseline, and that users can recognize the different robot moods as generated by our model. Finally, we demonstrate a zero-shot transfer of our model onto a different robotic platform with the same operator interface. 

---
# CHARMS: Cognitive Hierarchical Agent with Reasoning and Motion Styles 

**Authors**: Jingyi Wang, Duanfeng Chu, Zejian Deng, Liping Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.02450)  

**Abstract**: To address the current challenges of low intelligence and simplistic vehicle behavior modeling in autonomous driving simulation scenarios, this paper proposes the Cognitive Hierarchical Agent with Reasoning and Motion Styles (CHARMS). The model can reason about the behavior of other vehicles like a human driver and respond with different decision-making styles, thereby improving the intelligence and diversity of the surrounding vehicles in the driving scenario. By introducing the Level-k behavioral game theory, the paper models the decision-making process of human drivers and employs deep reinforcement learning to train the models with diverse decision styles, simulating different reasoning approaches and behavioral characteristics. Building on the Poisson cognitive hierarchy theory, this paper also presents a novel driving scenario generation method. The method controls the proportion of vehicles with different driving styles in the scenario using Poisson and binomial distributions, thus generating controllable and diverse driving environments. Experimental results demonstrate that CHARMS not only exhibits superior decision-making capabilities as ego vehicles, but also generates more complex and diverse driving scenarios as surrounding vehicles. We will release code for CHARMS at this https URL. 

---
# Hierarchical Policy-Gradient Reinforcement Learning for Multi-Agent Shepherding Control of Non-Cohesive Targets 

**Authors**: Stefano Covone, Italo Napolitano, Francesco De Lellis, Mario di Bernardo  

**Link**: [PDF](https://arxiv.org/pdf/2504.02479)  

**Abstract**: We propose a decentralized reinforcement learning solution for multi-agent shepherding of non-cohesive targets using policy-gradient methods. Our architecture integrates target-selection with target-driving through Proximal Policy Optimization, overcoming discrete-action constraints of previous Deep Q-Network approaches and enabling smoother agent trajectories. This model-free framework effectively solves the shepherding problem without prior dynamics knowledge. Experiments demonstrate our method's effectiveness and scalability with increased target numbers and limited sensing capabilities. 

---
# LLMs as Deceptive Agents: How Role-Based Prompting Induces Semantic Ambiguity in Puzzle Tasks 

**Authors**: Seunghyun Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2504.02254)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have not only showcased impressive creative capabilities but also revealed emerging agentic behaviors that exploit linguistic ambiguity in adversarial settings. In this study, we investigate how an LLM, acting as an autonomous agent, leverages semantic ambiguity to generate deceptive puzzles that mislead and challenge human users. Inspired by the popular puzzle game "Connections", we systematically compare puzzles produced through zero-shot prompting, role-injected adversarial prompts, and human-crafted examples, with an emphasis on understanding the underlying agent decision-making processes. Employing computational analyses with HateBERT to quantify semantic ambiguity, alongside subjective human evaluations, we demonstrate that explicit adversarial agent behaviors significantly heighten semantic ambiguity -- thereby increasing cognitive load and reducing fairness in puzzle solving. These findings provide critical insights into the emergent agentic qualities of LLMs and underscore important ethical considerations for evaluating and safely deploying autonomous language systems in both educational technologies and entertainment. 

---
# Achieving Unanimous Consensus in Decision Making Using Multi-Agents 

**Authors**: Apurba Pokharel, Ram Dantu, Shakila Zaman, Sirisha Talapuru, Vinh Quach  

**Link**: [PDF](https://arxiv.org/pdf/2504.02128)  

**Abstract**: Blockchain consensus mechanisms have relied on algorithms such as Proof-of-Work (PoW) and Proof-of-Stake (PoS) to ensure network functionality and integrity. However, these approaches struggle with adaptability for decision-making where the opinions of each matter rather than reaching an agreement based on honest majority or weighted consensus. This paper introduces a novel deliberation-based consensus mechanism where Large Language Models (LLMs) act as rational agents engaging in structured discussions to reach a unanimous consensus. By leveraging graded consensus and a multi-round deliberation process, our approach ensures both unanimous consensus for definitive problems and graded confidence for prioritized decisions and policies. We provide a formalization of our system and use it to show that the properties of blockchains: consistency, agreement, liveness, and determinism are maintained. Moreover, experimental results demonstrate our system's feasibility, showcasing how our deliberation method's convergence, block properties, and accuracy enable decision-making on blockchain networks. We also address key challenges with this novel approach such as degeneration of thoughts, hallucinations, malicious models and nodes, resource consumption, and scalability. 

---
# Adapting World Models with Latent-State Dynamics Residuals 

**Authors**: JB Lanier, Kyungmin Kim, Armin Karamzade, Yifei Liu, Ankita Sinha, Kat He, Davide Corsi, Roy Fox  

**Link**: [PDF](https://arxiv.org/pdf/2504.02252)  

**Abstract**: Simulation-to-reality reinforcement learning (RL) faces the critical challenge of reconciling discrepancies between simulated and real-world dynamics, which can severely degrade agent performance. A promising approach involves learning corrections to simulator forward dynamics represented as a residual error function, however this operation is impractical with high-dimensional states such as images. To overcome this, we propose ReDRAW, a latent-state autoregressive world model pretrained in simulation and calibrated to target environments through residual corrections of latent-state dynamics rather than of explicit observed states. Using this adapted world model, ReDRAW enables RL agents to be optimized with imagined rollouts under corrected dynamics and then deployed in the real world. In multiple vision-based MuJoCo domains and a physical robot visual lane-following task, ReDRAW effectively models changes to dynamics and avoids overfitting in low data regimes where traditional transfer methods fail. 

---
# On Simulation-Guided LLM-based Code Generation for Safe Autonomous Driving Software 

**Authors**: Ali Nouri, Johan Andersson, Kailash De Jesus Hornig, Zhennan Fei, Emil Knabe, Hakan Sivencrona, Beatriz Cabrero-Daniel, Christian Berger  

**Link**: [PDF](https://arxiv.org/pdf/2504.02141)  

**Abstract**: Automated Driving System (ADS) is a safety-critical software system responsible for the interpretation of the vehicle's environment and making decisions accordingly. The unbounded complexity of the driving context, including unforeseeable events, necessitate continuous improvement, often achieved through iterative DevOps processes. However, DevOps processes are themselves complex, making these improvements both time- and resource-intensive. Automation in code generation for ADS using Large Language Models (LLM) is one potential approach to address this challenge. Nevertheless, the development of ADS requires rigorous processes to verify, validate, assess, and qualify the code before it can be deployed in the vehicle and used. In this study, we developed and evaluated a prototype for automatic code generation and assessment using a designed pipeline of a LLM-based agent, simulation model, and rule-based feedback generator in an industrial setup. The LLM-generated code is evaluated automatically in a simulation model against multiple critical traffic scenarios, and an assessment report is provided as feedback to the LLM for modification or bug fixing. We report about the experimental results of the prototype employing Codellama:34b, DeepSeek (r1:32b and Coder:33b), CodeGemma:7b, Mistral:7b, and GPT4 for Adaptive Cruise Control (ACC) and Unsupervised Collision Avoidance by Evasive Manoeuvre (CAEM). We finally assessed the tool with 11 experts at two Original Equipment Manufacturers (OEMs) by conducting an interview study. 

---
# Self-Resource Allocation in Multi-Agent LLM Systems 

**Authors**: Alfonso Amayuelas, Jingbo Yang, Saaket Agashe, Ashwin Nagarajan, Antonis Antoniades, Xin Eric Wang, William Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02051)  

**Abstract**: With the development of LLMs as agents, there is a growing interest in connecting multiple agents into multi-agent systems to solve tasks concurrently, focusing on their role in task assignment and coordination. This paper explores how LLMs can effectively allocate computational tasks among multiple agents, considering factors such as cost, efficiency, and performance. In this work, we address key questions, including the effectiveness of LLMs as orchestrators and planners, comparing their effectiveness in task assignment and coordination. Our experiments demonstrate that LLMs can achieve high validity and accuracy in resource allocation tasks. We find that the planner method outperforms the orchestrator method in handling concurrent actions, resulting in improved efficiency and better utilization of agents. Additionally, we show that providing explicit information about worker capabilities enhances the allocation strategies of planners, particularly when dealing with suboptimal workers. 

---
# LLMs Working in Harmony: A Survey on the Technological Aspects of Building Effective LLM-Based Multi Agent Systems 

**Authors**: R. M. Aratchige, W. M. K. S. Ilmini  

**Link**: [PDF](https://arxiv.org/pdf/2504.01963)  

**Abstract**: This survey investigates foundational technologies essential for developing effective Large Language Model (LLM)-based multi-agent systems. Aiming to answer how best to optimize these systems for collaborative, dynamic environments, we focus on four critical areas: Architecture, Memory, Planning, and Technologies/Frameworks. By analyzing recent advancements and their limitations - such as scalability, real-time response challenges, and agent coordination constraints, we provide a detailed view of the technological landscape. Frameworks like the Mixture of Agents architecture and the ReAct planning model exemplify current innovations, showcasing improvements in role assignment and decision-making. This review synthesizes key strengths and persistent challenges, offering practical recommendations to enhance system scalability, agent collaboration, and adaptability. Our findings provide a roadmap for future research, supporting the creation of robust, efficient multi-agent systems that advance both individual agent performance and collective system resilience. 

---
# RoboAct-CLIP: Video-Driven Pre-training of Atomic Action Understanding for Robotics 

**Authors**: Zhiyuan Zhang, Yuxin He, Yong Sun, Junyu Shi, Lijiang Liu, Qiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2504.02069)  

**Abstract**: Visual Language Models (VLMs) have emerged as pivotal tools for robotic systems, enabling cross-task generalization, dynamic environmental interaction, and long-horizon planning through multimodal perception and semantic reasoning. However, existing open-source VLMs predominantly trained for generic vision-language alignment tasks fail to model temporally correlated action semantics that are crucial for robotic manipulation effectively. While current image-based fine-tuning methods partially adapt VLMs to robotic applications, they fundamentally disregard temporal evolution patterns in video sequences and suffer from visual feature entanglement between robotic agents, manipulated objects, and environmental contexts, thereby limiting semantic decoupling capability for atomic actions and compromising model this http URL overcome these challenges, this work presents RoboAct-CLIP with dual technical contributions: 1) A dataset reconstruction framework that performs semantic-constrained action unit segmentation and re-annotation on open-source robotic videos, constructing purified training sets containing singular atomic actions (e.g., "grasp"); 2) A temporal-decoupling fine-tuning strategy based on Contrastive Language-Image Pretraining (CLIP) architecture, which disentangles temporal action features across video frames from object-centric characteristics to achieve hierarchical representation learning of robotic atomic this http URL results in simulated environments demonstrate that the RoboAct-CLIP pretrained model achieves a 12% higher success rate than baseline VLMs, along with superior generalization in multi-object manipulation tasks. 

---
