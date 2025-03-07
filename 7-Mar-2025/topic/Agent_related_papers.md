# Measuring temporal effects of agent knowledge by date-controlled tool use 

**Authors**: R. Patrick Xian, Qiming Cui, Stefan Bauer, Reza Abbasi-Asl  

**Link**: [PDF](https://arxiv.org/pdf/2503.04188)  

**Abstract**: Temporal progression is an integral part of knowledge accumulation and update. Web search is frequently adopted as grounding for agent knowledge, yet its inappropriate configuration affects the quality of agent responses. Here, we construct a tool-based out-of-sample testing framework to measure the knowledge variability of large language model (LLM) agents from distinct date-controlled tools (DCTs). We demonstrate the temporal effects of an LLM agent as a writing assistant, which can use web search to help complete scientific publication abstracts. We show that temporal effects of the search engine translates into tool-dependent agent performance but can be alleviated with base model choice and explicit reasoning instructions such as chain-of-thought prompting. Our results indicate that agent evaluation should take a dynamical view and account for the temporal influence of tools and the updates of external resources. 

---
# ToolFuzz -- Automated Agent Tool Testing 

**Authors**: Ivan Milev, Mislav Balunović, Maximilian Baader, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2503.04479)  

**Abstract**: Large Language Model (LLM) Agents leverage the advanced reasoning capabilities of LLMs in real-world applications. To interface with an environment, these agents often rely on tools, such as web search or database APIs. As the agent provides the LLM with tool documentation along the user query, the completeness and correctness of this documentation is critical. However, tool documentation is often over-, under-, or ill-specified, impeding the agent's accuracy. Standard software testing approaches struggle to identify these errors as they are expressed in natural language. Thus, despite its importance, there currently exists no automated method to test the tool documentation for agents. To address this issue, we present ToolFuzz, the first method for automated testing of tool documentations. ToolFuzz is designed to discover two types of errors: (1) user queries leading to tool runtime errors and (2) user queries that lead to incorrect agent responses. ToolFuzz can generate a large and diverse set of natural inputs, effectively finding tool description errors at a low false positive rate. Further, we present two straightforward prompt-engineering approaches. We evaluate all three tool testing approaches on 32 common LangChain tools and 35 newly created custom tools and 2 novel benchmarks to further strengthen the assessment. We find that many publicly available tools suffer from underspecification. Specifically, we show that ToolFuzz identifies 20x more erroneous inputs compared to the prompt-engineering approaches, making it a key component for building reliable AI agents. 

---
# From Idea to CAD: A Language Model-Driven Multi-Agent System for Collaborative Design 

**Authors**: Felix Ocker, Stefan Menzel, Ahmed Sadik, Thiago Rios  

**Link**: [PDF](https://arxiv.org/pdf/2503.04417)  

**Abstract**: Creating digital models using Computer Aided Design (CAD) is a process that requires in-depth expertise. In industrial product development, this process typically involves entire teams of engineers, spanning requirements engineering, CAD itself, and quality assurance. We present an approach that mirrors this team structure with a Vision Language Model (VLM)-based Multi Agent System, with access to parametric CAD tooling and tool documentation. Combining agents for requirements engineering, CAD engineering, and vision-based quality assurance, a model is generated automatically from sketches and/ or textual descriptions. The resulting model can be refined collaboratively in an iterative validation loop with the user. Our approach has the potential to increase the effectiveness of design processes, both for industry experts and for hobbyists who create models for 3D printing. We demonstrate the potential of the architecture at the example of various design tasks and provide several ablations that show the benefits of the architecture's individual components. 

---
# Guidelines for Applying RL and MARL in Cybersecurity Applications 

**Authors**: Vasilios Mavroudis, Gregory Palmer, Sara Farmer, Kez Smithson Whitehead, David Foster, Adam Price, Ian Miles, Alberto Caron, Stephen Pasteris  

**Link**: [PDF](https://arxiv.org/pdf/2503.04262)  

**Abstract**: Reinforcement Learning (RL) and Multi-Agent Reinforcement Learning (MARL) have emerged as promising methodologies for addressing challenges in automated cyber defence (ACD). These techniques offer adaptive decision-making capabilities in high-dimensional, adversarial environments. This report provides a structured set of guidelines for cybersecurity professionals and researchers to assess the suitability of RL and MARL for specific use cases, considering factors such as explainability, exploration needs, and the complexity of multi-agent coordination. It also discusses key algorithmic approaches, implementation challenges, and real-world constraints, such as data scarcity and adversarial interference. The report further outlines open research questions, including policy optimality, agent cooperation levels, and the integration of MARL systems into operational cybersecurity frameworks. By bridging theoretical advancements and practical deployment, these guidelines aim to enhance the effectiveness of AI-driven cyber defence strategies. 

---
# AgentSafe: Safeguarding Large Language Model-based Multi-agent Systems via Hierarchical Data Management 

**Authors**: Junyuan Mao, Fanci Meng, Yifan Duan, Miao Yu, Xiaojun Jia, Junfeng Fang, Yuxuan Liang, Kun Wang, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2503.04392)  

**Abstract**: Large Language Model based multi-agent systems are revolutionizing autonomous communication and collaboration, yet they remain vulnerable to security threats like unauthorized access and data breaches. To address this, we introduce AgentSafe, a novel framework that enhances MAS security through hierarchical information management and memory protection. AgentSafe classifies information by security levels, restricting sensitive data access to authorized agents. AgentSafe incorporates two components: ThreatSieve, which secures communication by verifying information authority and preventing impersonation, and HierarCache, an adaptive memory management system that defends against unauthorized access and malicious poisoning, representing the first systematic defense for agent memory. Experiments across various LLMs show that AgentSafe significantly boosts system resilience, achieving defense success rates above 80% under adversarial conditions. Additionally, AgentSafe demonstrates scalability, maintaining robust performance as agent numbers and information complexity grow. Results underscore effectiveness of AgentSafe in securing MAS and its potential for real-world application. 

---
# Fair Play in the Fast Lane: Integrating Sportsmanship into Autonomous Racing Systems 

**Authors**: Zhenmin Huang, Ce Hao, Wei Zhan, Jun Ma, Masayoshi Tomizuka  

**Link**: [PDF](https://arxiv.org/pdf/2503.03774)  

**Abstract**: Autonomous racing has gained significant attention as a platform for high-speed decision-making and motion control. While existing methods primarily focus on trajectory planning and overtaking strategies, the role of sportsmanship in ensuring fair competition remains largely unexplored. In human racing, rules such as the one-motion rule and the enough-space rule prevent dangerous and unsportsmanlike behavior. However, autonomous racing systems often lack mechanisms to enforce these principles, potentially leading to unsafe maneuvers. This paper introduces a bi-level game-theoretic framework to integrate sportsmanship (SPS) into versus racing. At the high level, we model racing intentions using a Stackelberg game, where Monte Carlo Tree Search (MCTS) is employed to derive optimal strategies. At the low level, vehicle interactions are formulated as a Generalized Nash Equilibrium Problem (GNEP), ensuring that all agents follow sportsmanship constraints while optimizing their trajectories. Simulation results demonstrate the effectiveness of the proposed approach in enforcing sportsmanship rules while maintaining competitive performance. We analyze different scenarios where attackers and defenders adhere to or disregard sportsmanship rules and show how knowledge of these constraints influences strategic decision-making. This work highlights the importance of balancing competition and fairness in autonomous racing and provides a foundation for developing ethical and safe AI-driven racing systems. 

---
# Multi-Agent Inverse Q-Learning from Demonstrations 

**Authors**: Nathaniel Haynam, Adam Khoja, Dhruv Kumar, Vivek Myers, Erdem Bıyık  

**Link**: [PDF](https://arxiv.org/pdf/2503.04679)  

**Abstract**: When reward functions are hand-designed, deep reinforcement learning algorithms often suffer from reward misspecification, causing them to learn suboptimal policies in terms of the intended task objectives. In the single-agent case, inverse reinforcement learning (IRL) techniques attempt to address this issue by inferring the reward function from expert demonstrations. However, in multi-agent problems, misalignment between the learned and true objectives is exacerbated due to increased environment non-stationarity and variance that scales with multiple agents. As such, in multi-agent general-sum games, multi-agent IRL algorithms have difficulty balancing cooperative and competitive objectives. To address these issues, we propose Multi-Agent Marginal Q-Learning from Demonstrations (MAMQL), a novel sample-efficient framework for multi-agent IRL. For each agent, MAMQL learns a critic marginalized over the other agents' policies, allowing for a well-motivated use of Boltzmann policies in the multi-agent context. We identify a connection between optimal marginalized critics and single-agent soft-Q IRL, allowing us to apply a direct, simple optimization criterion from the single-agent domain. Across our experiments on three different simulated domains, MAMQL significantly outperforms previous multi-agent methods in average reward, sample efficiency, and reward recovery by often more than 2-5x. We make our code available at this https URL . 

---
# DAST: Difficulty-Adaptive Slow-Thinking for Large Reasoning Models 

**Authors**: Yi Shen, Jian Zhang, Jieyun Huang, Shuming Shi, Wenjing Zhang, Jiangze Yan, Ning Wang, Kai Wang, Shiguo Lian  

**Link**: [PDF](https://arxiv.org/pdf/2503.04472)  

**Abstract**: Recent advancements in slow-thinking reasoning models have shown exceptional performance in complex reasoning tasks. However, these models often exhibit overthinking-generating redundant reasoning steps for simple problems, leading to excessive computational resource usage. While current mitigation strategies uniformly reduce reasoning tokens, they risk degrading performance on challenging tasks that require extended reasoning. This paper introduces Difficulty-Adaptive Slow-Thinking (DAST), a novel framework that enables models to autonomously adjust the length of Chain-of-Thought(CoT) based on problem difficulty. We first propose a Token Length Budget (TLB) metric to quantify difficulty, then leveraging length-aware reward shaping and length preference optimization to implement DAST. DAST penalizes overlong responses for simple tasks while incentivizing sufficient reasoning for complex problems. Experiments on diverse datasets and model scales demonstrate that DAST effectively mitigates overthinking (reducing token usage by over 30\% on average) while preserving reasoning accuracy on complex problems. 

---
# Towards Autonomous Reinforcement Learning for Real-World Robotic Manipulation with Large Language Models 

**Authors**: Niccolò Turcato, Matteo Iovino, Aris Synodinos, Alberto Dalla Libera, Ruggero Carli, Pietro Falco  

**Link**: [PDF](https://arxiv.org/pdf/2503.04280)  

**Abstract**: Recent advancements in Large Language Models (LLMs) and Visual Language Models (VLMs) have significantly impacted robotics, enabling high-level semantic motion planning applications. Reinforcement Learning (RL), a complementary paradigm, enables agents to autonomously optimize complex behaviors through interaction and reward signals. However, designing effective reward functions for RL remains challenging, especially in real-world tasks where sparse rewards are insufficient and dense rewards require elaborate design. In this work, we propose Autonomous Reinforcement learning for Complex HumanInformed Environments (ARCHIE), an unsupervised pipeline leveraging GPT-4, a pre-trained LLM, to generate reward functions directly from natural language task descriptions. The rewards are used to train RL agents in simulated environments, where we formalize the reward generation process to enhance feasibility. Additionally, GPT-4 automates the coding of task success criteria, creating a fully automated, one-shot procedure for translating human-readable text into deployable robot skills. Our approach is validated through extensive simulated experiments on single-arm and bi-manual manipulation tasks using an ABB YuMi collaborative robot, highlighting its practicality and effectiveness. Tasks are demonstrated on the real robot setup. 

---
# Knowledge Retention for Continual Model-Based Reinforcement Learning 

**Authors**: Yixiang Sun, Haotian Fu, Michael Littman, George Konidaris  

**Link**: [PDF](https://arxiv.org/pdf/2503.04256)  

**Abstract**: We propose DRAGO, a novel approach for continual model-based reinforcement learning aimed at improving the incremental development of world models across a sequence of tasks that differ in their reward functions but not the state space or dynamics. DRAGO comprises two key components: Synthetic Experience Rehearsal, which leverages generative models to create synthetic experiences from past tasks, allowing the agent to reinforce previously learned dynamics without storing data, and Regaining Memories Through Exploration, which introduces an intrinsic reward mechanism to guide the agent toward revisiting relevant states from prior tasks. Together, these components enable the agent to maintain a comprehensive and continually developing world model, facilitating more effective learning and adaptation across diverse environments. Empirical evaluations demonstrate that DRAGO is able to preserve knowledge across tasks, achieving superior performance in various continual learning scenarios. 

---
# Towards Intelligent Transportation with Pedestrians and Vehicles In-the-Loop: A Surveillance Video-Assisted Federated Digital Twin Framework 

**Authors**: Xiaolong Li, Jianhao Wei, Haidong Wang, Li Dong, Ruoyang Chen, Changyan Yi, Jun Cai, Dusit Niyato, Xuemin, Shen  

**Link**: [PDF](https://arxiv.org/pdf/2503.04170)  

**Abstract**: In intelligent transportation systems (ITSs), incorporating pedestrians and vehicles in-the-loop is crucial for developing realistic and safe traffic management solutions. However, there is falls short of simulating complex real-world ITS scenarios, primarily due to the lack of a digital twin implementation framework for characterizing interactions between pedestrians and vehicles at different locations in different traffic environments. In this article, we propose a surveillance video assisted federated digital twin (SV-FDT) framework to empower ITSs with pedestrians and vehicles in-the-loop. Specifically, SVFDT builds comprehensive pedestrian-vehicle interaction models by leveraging multi-source traffic surveillance videos. Its architecture consists of three layers: (i) the end layer, which collects traffic surveillance videos from multiple sources; (ii) the edge layer, responsible for semantic segmentation-based visual understanding, twin agent-based interaction modeling, and local digital twin system (LDTS) creation in local regions; and (iii) the cloud layer, which integrates LDTSs across different regions to construct a global DT model in realtime. We analyze key design requirements and challenges and present core guidelines for SVFDT's system implementation. A testbed evaluation demonstrates its effectiveness in optimizing traffic management. Comparisons with traditional terminal-server frameworks highlight SV-FDT's advantages in mirroring delays, recognition accuracy, and subjective evaluation. Finally, we identify some open challenges and discuss future research directions. 

---
# Dynamic Benchmarking of Reasoning Capabilities in Code Large Language Models Under Data Contamination 

**Authors**: Simin Chen, Pranav Pusarla, Baishakhi Ray  

**Link**: [PDF](https://arxiv.org/pdf/2503.04149)  

**Abstract**: The rapid evolution of code largelanguage models underscores the need for effective and transparent benchmarking of their reasoning capabilities. However, the current benchmarking approach heavily depends on publicly available, human-created datasets. The widespread use of these fixed benchmark datasets makes the benchmarking process to be static and thus particularly susceptible to data contamination, an unavoidable consequence of the extensive data collection processes used to train Code LLMs. Existing approaches that address data contamination often suffer from human effort limitations and imbalanced problem complexity. To tackle these challenges, we propose \tool, a novel benchmarking suite for evaluating Code LLMs under potential data contamination. Given a seed programming problem, \tool employs multiple agents to extract and modify the context without altering the core logic, generating semantically equivalent variations. We introduce a dynamic data generation methods and conduct empirical studies on two seed datasets across 21 Code LLMs. Results show that \tool effectively benchmarks reasoning capabilities under contamination risks while generating diverse problem sets to ensure consistent and reliable evaluations. 

---
# Can We Optimize Deep RL Policy Weights as Trajectory Modeling? 

**Authors**: Hongyao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04074)  

**Abstract**: Learning the optimal policy from a random network initialization is the theme of deep Reinforcement Learning (RL). As the scale of DRL training increases, treating DRL policy network weights as a new data modality and exploring the potential becomes appealing and possible. In this work, we focus on the policy learning path in deep RL, represented by the trajectory of network weights of historical policies, which reflects the evolvement of the policy learning process. Taking the idea of trajectory modeling with Transformer, we propose Transformer as Implicit Policy Learner (TIPL), which processes policy network weights in an autoregressive manner. We collect the policy learning path data by running independent RL training trials, with which we then train our TIPL model. In the experiments, we demonstrate that TIPL is able to fit the implicit dynamics of policy learning and perform the optimization of policy network by inference. 

---
# Human Implicit Preference-Based Policy Fine-tuning for Multi-Agent Reinforcement Learning in USV Swarm 

**Authors**: Hyeonjun Kim, Kanghoon Lee, Junho Park, Jiachen Li, Jinkyoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2503.03796)  

**Abstract**: Multi-Agent Reinforcement Learning (MARL) has shown promise in solving complex problems involving cooperation and competition among agents, such as an Unmanned Surface Vehicle (USV) swarm used in search and rescue, surveillance, and vessel protection. However, aligning system behavior with user preferences is challenging due to the difficulty of encoding expert intuition into reward functions. To address the issue, we propose a Reinforcement Learning with Human Feedback (RLHF) approach for MARL that resolves credit-assignment challenges through an Agent-Level Feedback system categorizing feedback into intra-agent, inter-agent, and intra-team types. To overcome the challenges of direct human feedback, we employ a Large Language Model (LLM) evaluator to validate our approach using feedback scenarios such as region constraints, collision avoidance, and task allocation. Our method effectively refines USV swarm policies, addressing key challenges in multi-agent systems while maintaining fairness and performance consistency. 

---
# RiskAgent: Autonomous Medical AI Copilot for Generalist Risk Prediction 

**Authors**: Fenglin Liu, Jinge Wu, Hongjian Zhou, Xiao Gu, Soheila Molaei, Anshul Thakur, Lei Clifton, Honghan Wu, David A. Clifton  

**Link**: [PDF](https://arxiv.org/pdf/2503.03802)  

**Abstract**: The application of Large Language Models (LLMs) to various clinical applications has attracted growing research attention. However, real-world clinical decision-making differs significantly from the standardized, exam-style scenarios commonly used in current efforts. In this paper, we present the RiskAgent system to perform a broad range of medical risk predictions, covering over 387 risk scenarios across diverse complex diseases, e.g., cardiovascular disease and cancer. RiskAgent is designed to collaborate with hundreds of clinical decision tools, i.e., risk calculators and scoring systems that are supported by evidence-based medicine. To evaluate our method, we have built the first benchmark MedRisk specialized for risk prediction, including 12,352 questions spanning 154 diseases, 86 symptoms, 50 specialties, and 24 organ systems. The results show that our RiskAgent, with 8 billion model parameters, achieves 76.33% accuracy, outperforming the most recent commercial LLMs, o1, o3-mini, and GPT-4.5, and doubling the 38.39% accuracy of GPT-4o. On rare diseases, e.g., Idiopathic Pulmonary Fibrosis (IPF), RiskAgent outperforms o1 and GPT-4.5 by 27.27% and 45.46% accuracy, respectively. Finally, we further conduct a generalization evaluation on an external evidence-based diagnosis benchmark and show that our RiskAgent achieves the best results. These encouraging results demonstrate the great potential of our solution for diverse diagnosis domains. To improve the adaptability of our model in different scenarios, we have built and open-sourced a family of models ranging from 1 billion to 70 billion parameters. Our code, data, and models are all available at this https URL. 

---
# Multi-Agent Systems Powered by Large Language Models: Applications in Swarm Intelligence 

**Authors**: Cristian Jimenez-Romero, Alper Yegenoglu, Christian Blum  

**Link**: [PDF](https://arxiv.org/pdf/2503.03800)  

**Abstract**: This work examines the integration of large language models (LLMs) into multi-agent simulations by replacing the hard-coded programs of agents with LLM-driven prompts. The proposed approach is showcased in the context of two examples of complex systems from the field of swarm intelligence: ant colony foraging and bird flocking. Central to this study is a toolchain that integrates LLMs with the NetLogo simulation platform, leveraging its Python extension to enable communication with GPT-4o via the OpenAI API. This toolchain facilitates prompt-driven behavior generation, allowing agents to respond adaptively to environmental data. For both example applications mentioned above, we employ both structured, rule-based prompts and autonomous, knowledge-driven prompts. Our work demonstrates how this toolchain enables LLMs to study self-organizing processes and induce emergent behaviors within multi-agent environments, paving the way for new approaches to exploring intelligent systems and modeling swarm intelligence inspired by natural phenomena. We provide the code, including simulation files and data at this https URL. 

---
# START: Self-taught Reasoner with Tools 

**Authors**: Chengpeng Li, Mingfeng Xue, Zhenru Zhang, Jiaxi Yang, Beichen Zhang, Xiang Wang, Bowen Yu, Binyuan Hui, Junyang Lin, Dayiheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04625)  

**Abstract**: Large reasoning models (LRMs) like OpenAI-o1 and DeepSeek-R1 have demonstrated remarkable capabilities in complex reasoning tasks through the utilization of long Chain-of-thought (CoT). However, these models often suffer from hallucinations and inefficiencies due to their reliance solely on internal reasoning processes. In this paper, we introduce START (Self-Taught Reasoner with Tools), a novel tool-integrated long CoT reasoning LLM that significantly enhances reasoning capabilities by leveraging external tools. Through code execution, START is capable of performing complex computations, self-checking, exploring diverse methods, and self-debugging, thereby addressing the limitations of LRMs. The core innovation of START lies in its self-learning framework, which comprises two key techniques: 1) Hint-infer: We demonstrate that inserting artificially designed hints (e.g., ``Wait, maybe using Python here is a good idea.'') during the inference process of a LRM effectively stimulates its ability to utilize external tools without the need for any demonstration data. Hint-infer can also serve as a simple and effective sequential test-time scaling method; 2) Hint Rejection Sampling Fine-Tuning (Hint-RFT): Hint-RFT combines Hint-infer and RFT by scoring, filtering, and modifying the reasoning trajectories with tool invocation generated by a LRM via Hint-infer, followed by fine-tuning the LRM. Through this framework, we have fine-tuned the QwQ-32B model to achieve START. On PhD-level science QA (GPQA), competition-level math benchmarks (AMC23, AIME24, AIME25), and the competition-level code benchmark (LiveCodeBench), START achieves accuracy rates of 63.6%, 95.0%, 66.7%, 47.1%, and 47.3%, respectively. It significantly outperforms the base QwQ-32B and achieves performance comparable to the state-of-the-art open-weight model R1-Distill-Qwen-32B and the proprietary model o1-Preview. 

---
# Quantifying the Reasoning Abilities of LLMs on Real-world Clinical Cases 

**Authors**: Pengcheng Qiu, Chaoyi Wu, Shuyu Liu, Weike Zhao, Ya Zhang, Yanfeng Wang, Weidi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2503.04691)  

**Abstract**: The latest reasoning-enhanced large language models (reasoning LLMs), such as DeepSeek-R1 and OpenAI-o3, have demonstrated remarkable success. However, the application of such reasoning enhancements to the highly professional medical domain has not been clearly evaluated, particularly regarding with not only assessing the final generation but also examining the quality of their reasoning processes. In this study, we present MedR-Bench, a reasoning-focused medical evaluation benchmark comprising 1,453 structured patient cases with reasoning references mined from case reports. Our benchmark spans 13 body systems and 10 specialty disorders, encompassing both common and rare diseases. In our evaluation, we introduce a versatile framework consisting of three critical clinical stages: assessment recommendation, diagnostic decision-making, and treatment planning, comprehensively capturing the LLMs' performance across the entire patient journey in healthcare. For metrics, we propose a novel agentic system, Reasoning Evaluator, designed to automate and objectively quantify free-text reasoning responses in a scalable manner from the perspectives of efficiency, factuality, and completeness by dynamically searching and performing cross-referencing checks. As a result, we assess five state-of-the-art reasoning LLMs, including DeepSeek-R1, OpenAI-o3-mini, and others. Our results reveal that current LLMs can handle relatively simple diagnostic tasks with sufficient critical assessment results, achieving accuracy generally over 85%. However, they still struggle with more complex tasks, such as assessment recommendation and treatment planning. In reasoning, their reasoning processes are generally reliable, with factuality scores exceeding 90%, though they often omit critical reasoning steps. Our study clearly reveals further development directions for current clinical LLMs. 

---
