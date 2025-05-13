# Belief Injection for Epistemic Control in Linguistic State Space 

**Authors**: Sebastian Dumbrava  

**Link**: [PDF](https://arxiv.org/pdf/2505.07693)  

**Abstract**: This work introduces belief injection, a proactive epistemic control mechanism for artificial agents whose cognitive states are structured as dynamic ensembles of linguistic belief fragments. Grounded in the Semantic Manifold framework, belief injection directly incorporates targeted linguistic beliefs into an agent's internal cognitive state, influencing reasoning and alignment proactively rather than reactively. We delineate various injection strategies, such as direct, context-aware, goal-oriented, and reflective approaches, and contrast belief injection with related epistemic control mechanisms, notably belief filtering. Additionally, this work discusses practical applications, implementation considerations, ethical implications, and outlines promising directions for future research into cognitive governance using architecturally embedded belief injection. 

---
# Agent RL Scaling Law: Agent RL with Spontaneous Code Execution for Mathematical Problem Solving 

**Authors**: Xinji Mai, Haotian Xu, Xing W, Weinong Wang, Yingying Zhang, Wenqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07773)  

**Abstract**: Large Language Models (LLMs) often struggle with mathematical reasoning tasks requiring precise, verifiable computation. While Reinforcement Learning (RL) from outcome-based rewards enhances text-based reasoning, understanding how agents autonomously learn to leverage external tools like code execution remains crucial. We investigate RL from outcome-based rewards for Tool-Integrated Reasoning, ZeroTIR, training base LLMs to spontaneously generate and execute Python code for mathematical problems without supervised tool-use examples. Our central contribution is we demonstrate that as RL training progresses, key metrics scale predictably. Specifically, we observe strong positive correlations where increased training steps lead to increases in the spontaneous code execution frequency, the average response length, and, critically, the final task accuracy. This suggests a quantifiable relationship between computational effort invested in training and the emergence of effective, tool-augmented reasoning strategies. We implement a robust framework featuring a decoupled code execution environment and validate our findings across standard RL algorithms and frameworks. Experiments show ZeroTIR significantly surpasses non-tool ZeroRL baselines on challenging math benchmarks. Our findings provide a foundational understanding of how autonomous tool use is acquired and scales within Agent RL, offering a reproducible benchmark for future studies. Code is released at \href{this https URL}{this https URL}. 

---
# Emotion-Gradient Metacognitive RSI (Part I): Theoretical Foundations and Single-Agent Architecture 

**Authors**: Rintaro Ando  

**Link**: [PDF](https://arxiv.org/pdf/2505.07757)  

**Abstract**: We present the Emotion-Gradient Metacognitive Recursive Self-Improvement (EG-MRSI) framework, a novel architecture that integrates introspective metacognition, emotion-based intrinsic motivation, and recursive self-modification into a unified theoretical system. The framework is explicitly capable of overwriting its own learning algorithm under formally bounded risk. Building upon the Noise-to-Meaning RSI (N2M-RSI) foundation, EG-MRSI introduces a differentiable intrinsic reward function driven by confidence, error, novelty, and cumulative success. This signal regulates both a metacognitive mapping and a self-modification operator constrained by provable safety mechanisms. We formally define the initial agent configuration, emotion-gradient dynamics, and RSI trigger conditions, and derive a reinforcement-compatible optimization objective that guides the agent's development trajectory. Meaning Density and Meaning Conversion Efficiency are introduced as quantifiable metrics of semantic learning, closing the gap between internal structure and predictive informativeness. This Part I paper establishes the single-agent theoretical foundations of EG-MRSI. Future parts will extend this framework to include safety certificates and rollback protocols (Part II), collective intelligence mechanisms (Part III), and feasibility constraints including thermodynamic and computational limits (Part IV). Together, the EG-MRSI series provides a rigorous, extensible foundation for open-ended and safe AGI. 

---
# S-GRPO: Early Exit via Reinforcement Learning in Reasoning Models 

**Authors**: Muzhi Dai, Chenxu Yang, Qingyi Si  

**Link**: [PDF](https://arxiv.org/pdf/2505.07686)  

**Abstract**: As Test-Time Scaling emerges as an active research focus in the large language model community, advanced post-training methods increasingly emphasize extending chain-of-thought (CoT) generation length, thereby enhancing reasoning capabilities to approach Deepseek R1-like reasoning models. However, recent studies reveal that reasoning models (even Qwen3) consistently exhibit excessive thought redundancy in CoT generation. This overthinking problem stems from conventional outcome-reward reinforcement learning's systematic neglect in regulating intermediate reasoning steps. This paper proposes Serial-Group Decaying-Reward Policy Optimization (namely S-GRPO), a novel reinforcement learning method that empowers models with the capability to determine the sufficiency of reasoning steps, subsequently triggering early exit of CoT generation. Specifically, unlike GRPO, which samples multiple possible completions (parallel group) in parallel, we select multiple temporal positions in the generation of one CoT to allow the model to exit thinking and instead generate answers (serial group), respectively. For the correct answers in a serial group, we assign rewards that decay according to positions, with lower rewards towards the later ones, thereby reinforcing the model's behavior to generate higher-quality answers at earlier phases with earlier exits of thinking. Empirical evaluations demonstrate compatibility with state-of-the-art reasoning models, including Qwen3 and Deepseek-distill models, achieving 35.4% ~ 61.1\% sequence length reduction with 0.72% ~ 6.08% accuracy improvements across GSM8K, AIME 2024, AMC 2023, MATH-500, and GPQA Diamond benchmarks. 

---
# A Multi-Agent Reinforcement Learning Approach for Cooperative Air-Ground-Human Crowdsensing in Emergency Rescue 

**Authors**: Wenhao Lu, Zhengqiu Zhu, Yong Zhao, Yonglin Tian, Junjie Zeng, Jun Zhang, Zhong Liu, Fei-Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06997)  

**Abstract**: Mobile crowdsensing is evolving beyond traditional human-centric models by integrating heterogeneous entities like unmanned aerial vehicles (UAVs) and unmanned ground vehicles (UGVs). Optimizing task allocation among these diverse agents is critical, particularly in challenging emergency rescue scenarios characterized by complex environments, limited communication, and partial observability. This paper tackles the Heterogeneous-Entity Collaborative-Sensing Task Allocation (HECTA) problem specifically for emergency rescue, considering humans, UAVs, and UGVs. We introduce a novel ``Hard-Cooperative'' policy where UGVs prioritize recharging low-battery UAVs, alongside performing their sensing tasks. The primary objective is maximizing the task completion rate (TCR) under strict time constraints. We rigorously formulate this NP-hard problem as a decentralized partially observable Markov decision process (Dec-POMDP) to effectively handle sequential decision-making under uncertainty. To solve this, we propose HECTA4ER, a novel multi-agent reinforcement learning algorithm built upon a Centralized Training with Decentralized Execution architecture. HECTA4ER incorporates tailored designs, including specialized modules for complex feature extraction, utilization of action-observation history via hidden states, and a mixing network integrating global and local information, specifically addressing the challenges of partial observability. Furthermore, theoretical analysis confirms the algorithm's convergence properties. Extensive simulations demonstrate that HECTA4ER significantly outperforms baseline algorithms, achieving an average 18.42% increase in TCR. Crucially, a real-world case study validates the algorithm's effectiveness and robustness in dynamic sensing scenarios, highlighting its strong potential for practical application in emergency response. 

---
# Architectural Precedents for General Agents using Large Language Models 

**Authors**: Robert E. Wray, James R. Kirk, John E. Laird  

**Link**: [PDF](https://arxiv.org/pdf/2505.07087)  

**Abstract**: One goal of AI (and AGI) is to identify and understand specific mechanisms and representations sufficient for general intelligence. Often, this work manifests in research focused on architectures and many cognitive architectures have been explored in AI/AGI. However, different research groups and even different research traditions have somewhat independently identified similar/common patterns of processes and representations or cognitive design patterns that are manifest in existing architectures. Today, AI systems exploiting large language models (LLMs) offer a relatively new combination of mechanism and representation available for exploring the possibilities of general intelligence. In this paper, we summarize a few recurring cognitive design patterns that have appeared in various pre-transformer AI architectures. We then explore how these patterns are evident in systems using LLMs, especially for reasoning and interactive ("agentic") use cases. By examining and applying these recurring patterns, we can also predict gaps or deficiencies in today's Agentic LLM Systems and identify likely subjects of future research towards general intelligence using LLMs and other generative foundation models. 

---
# YuLan-OneSim: Towards the Next Generation of Social Simulator with Large Language Models 

**Authors**: Lei Wang, Heyang Gao, Xiaohe Bo, Xu Chen, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.07581)  

**Abstract**: Leveraging large language model (LLM) based agents to simulate human social behaviors has recently gained significant attention. In this paper, we introduce a novel social simulator called YuLan-OneSim. Compared to previous works, YuLan-OneSim distinguishes itself in five key aspects: (1) Code-free scenario construction: Users can simply describe and refine their simulation scenarios through natural language interactions with our simulator. All simulation code is automatically generated, significantly reducing the need for programming expertise. (2) Comprehensive default scenarios: We implement 50 default simulation scenarios spanning 8 domains, including economics, sociology, politics, psychology, organization, demographics, law, and communication, broadening access for a diverse range of social researchers. (3) Evolvable simulation: Our simulator is capable of receiving external feedback and automatically fine-tuning the backbone LLMs, significantly enhancing the simulation quality. (4) Large-scale simulation: By developing a fully responsive agent framework and a distributed simulation architecture, our simulator can handle up to 100,000 agents, ensuring more stable and reliable simulation results. (5) AI social researcher: Leveraging the above features, we develop an AI social researcher. Users only need to propose a research topic, and the AI researcher will automatically analyze the input, construct simulation environments, summarize results, generate technical reports, review and refine the reports--completing the social science research loop. To demonstrate the advantages of YuLan-OneSim, we conduct experiments to evaluate the quality of the automatically generated scenarios, the reliability, efficiency, and scalability of the simulation process, as well as the performance of the AI social researcher. 

---
# RefPentester: A Knowledge-Informed Self-Reflective Penetration Testing Framework Based on Large Language Models 

**Authors**: Hanzheng Dai, Yuanliang Li, Zhibo Zhang, Jun Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.07089)  

**Abstract**: Automated penetration testing (AutoPT) powered by large language models (LLMs) has gained attention for its ability to automate ethical hacking processes and identify vulnerabilities in target systems by leveraging the intrinsic knowledge of LLMs. However, existing LLM-based AutoPT frameworks often underperform compared to human experts in challenging tasks for several reasons: the imbalanced knowledge used in LLM training, short-sighted planning in the planning process, and hallucinations during command generation. In addition, the penetration testing (PT) process, with its trial-and-error nature, is limited by existing frameworks that lack mechanisms to learn from previous failed operations, restricting adaptive improvement of PT strategies. To address these limitations, we propose a knowledge-informed self-reflective PT framework powered by LLMs, called RefPentester, which is an AutoPT framework designed to assist human operators in identifying the current stage of the PT process, selecting appropriate tactic and technique for the stage, choosing suggested action, providing step-by-step operational guidance, and learning from previous failed operations. We also modeled the PT process as a seven-state Stage Machine to integrate the proposed framework effectively. The evaluation shows that RefPentester can successfully reveal credentials on Hack The Box's Sau machine, outperforming the baseline GPT-4o model by 16.7\%. Across PT stages, RefPentester also demonstrates superior success rates on PT stage transitions. 

---
# DialogueReason: Rule-Based RL Sparks Dialogue Reasoning in LLMs 

**Authors**: Yubo Shu, Zhewei Huang, Xin Wu, Chen Hu, Shuchang Zhou, Daxin Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07049)  

**Abstract**: We propose DialogueReason, a reasoning paradigm that uncovers the lost roles in monologue-style reasoning models, aiming to boost diversity and coherency of the reasoning process. Recent advances in RL-based large reasoning models have led to impressive long CoT capabilities and high performance on math and science benchmarks. However, these reasoning models rely mainly on monologue-style reasoning, which often limits reasoning diversity and coherency, frequently recycling fixed strategies or exhibiting unnecessary shifts in attention. Our work consists of an analysis of monologue reasoning patterns and the development of a dialogue-based reasoning approach. We first introduce the Compound-QA task, which concatenates multiple problems into a single prompt to assess both diversity and coherency of reasoning. Our analysis shows that Compound-QA exposes weaknesses in monologue reasoning, evidenced by both quantitative metrics and qualitative reasoning traces. Building on the analysis, we propose a dialogue-based reasoning, named DialogueReason, structured around agents, environment, and interactions. Using PPO with rule-based rewards, we train open-source LLMs (Qwen-QWQ and Qwen-Base) to adopt dialogue reasoning. We evaluate trained models on MATH, AIME, and GPQA datasets, showing that the dialogue reasoning model outperforms monologue models under more complex compound questions. Additionally, we discuss how dialogue-based reasoning helps enhance interpretability, facilitate more intuitive human interaction, and inspire advances in multi-agent system design. 

---
# Beyond Patterns: Harnessing Causal Logic for Autonomous Driving Trajectory Prediction 

**Authors**: Bonan Wang, Haicheng Liao, Chengyue Wang, Bin Rao, Yanchen Guan, Guyang Yu, Jiaxun Zhang, Songning Lai, Chengzhong Xu, Zhenning Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.06856)  

**Abstract**: Accurate trajectory prediction has long been a major challenge for autonomous driving (AD). Traditional data-driven models predominantly rely on statistical correlations, often overlooking the causal relationships that govern traffic behavior. In this paper, we introduce a novel trajectory prediction framework that leverages causal inference to enhance predictive robustness, generalization, and accuracy. By decomposing the environment into spatial and temporal components, our approach identifies and mitigates spurious correlations, uncovering genuine causal relationships. We also employ a progressive fusion strategy to integrate multimodal information, simulating human-like reasoning processes and enabling real-time inference. Evaluations on five real-world datasets--ApolloScape, nuScenes, NGSIM, HighD, and MoCAD--demonstrate our model's superiority over existing state-of-the-art (SOTA) methods, with improvements in key metrics such as RMSE and FDE. Our findings highlight the potential of causal reasoning to transform trajectory prediction, paving the way for robust AD systems. 

---
# Control Plane as a Tool: A Scalable Design Pattern for Agentic AI Systems 

**Authors**: Sivasathivel Kandasamy  

**Link**: [PDF](https://arxiv.org/pdf/2505.06817)  

**Abstract**: Agentic AI systems represent a new frontier in artificial intelligence, where agents often based on large language models(LLMs) interact with tools, environments, and other agents to accomplish tasks with a degree of autonomy. These systems show promise across a range of domains, but their architectural underpinnings remain immature. This paper conducts a comprehensive review of the types of agents, their modes of interaction with the environment, and the infrastructural and architectural challenges that emerge. We identify a gap in how these systems manage tool orchestration at scale and propose a reusable design abstraction: the "Control Plane as a Tool" pattern. This pattern allows developers to expose a single tool interface to an agent while encapsulating modular tool routing logic behind it. We position this pattern within the broader context of agent design and argue that it addresses several key challenges in scaling, safety, and extensibility. 

---
# Bi-level Mean Field: Dynamic Grouping for Large-Scale MARL 

**Authors**: Yuxuan Zheng, Yihe Zhou, Feiyang Xu, Mingli Song, Shunyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06706)  

**Abstract**: Large-scale Multi-Agent Reinforcement Learning (MARL) often suffers from the curse of dimensionality, as the exponential growth in agent interactions significantly increases computational complexity and impedes learning efficiency. To mitigate this, existing efforts that rely on Mean Field (MF) simplify the interaction landscape by approximating neighboring agents as a single mean agent, thus reducing overall complexity to pairwise interactions. However, these MF methods inevitably fail to account for individual differences, leading to aggregation noise caused by inaccurate iterative updates during MF learning. In this paper, we propose a Bi-level Mean Field (BMF) method to capture agent diversity with dynamic grouping in large-scale MARL, which can alleviate aggregation noise via bi-level interaction. Specifically, BMF introduces a dynamic group assignment module, which employs a Variational AutoEncoder (VAE) to learn the representations of agents, facilitating their dynamic grouping over time. Furthermore, we propose a bi-level interaction module to model both inter- and intra-group interactions for effective neighboring aggregation. Experiments across various tasks demonstrate that the proposed BMF yields results superior to the state-of-the-art methods. Our code will be made publicly available. 

---
# A Point-Based Algorithm for Distributional Reinforcement Learning in Partially Observable Domains 

**Authors**: Larry Preuett III  

**Link**: [PDF](https://arxiv.org/pdf/2505.06518)  

**Abstract**: In many real-world planning tasks, agents must tackle uncertainty about the environment's state and variability in the outcomes of any chosen policy. We address both forms of uncertainty as a first step toward safer algorithms in partially observable settings. Specifically, we extend Distributional Reinforcement Learning (DistRL)-which models the entire return distribution for fully observable domains-to Partially Observable Markov Decision Processes (POMDPs), allowing an agent to learn the distribution of returns for each conditional plan. Concretely, we introduce new distributional Bellman operators for partial observability and prove their convergence under the supremum p-Wasserstein metric. We also propose a finite representation of these return distributions via psi-vectors, generalizing the classical alpha-vectors in POMDP solvers. Building on this, we develop Distributional Point-Based Value Iteration (DPBVI), which integrates psi-vectors into a standard point-based backup procedure-bridging DistRL and POMDP planning. By tracking return distributions, DPBVI naturally enables risk-sensitive control in domains where rare, high-impact events must be carefully managed. We provide source code to foster further research in robust decision-making under partial observability. 

---
# A Grounded Memory System For Smart Personal Assistants 

**Authors**: Felix Ocker, Jörg Deigmöller, Pavel Smirnov, Julian Eggert  

**Link**: [PDF](https://arxiv.org/pdf/2505.06328)  

**Abstract**: A wide variety of agentic AI applications - ranging from cognitive assistants for dementia patients to robotics - demand a robust memory system grounded in reality. In this paper, we propose such a memory system consisting of three components. First, we combine Vision Language Models for image captioning and entity disambiguation with Large Language Models for consistent information extraction during perception. Second, the extracted information is represented in a memory consisting of a knowledge graph enhanced by vector embeddings to efficiently manage relational information. Third, we combine semantic search and graph query generation for question answering via Retrieval Augmented Generation. We illustrate the system's working and potential using a real-world example. 

---
# SmartPilot: A Multiagent CoPilot for Adaptive and Intelligent Manufacturing 

**Authors**: Chathurangi Shyalika, Renjith Prasad, Alaa Al Ghazo, Darssan Eswaramoorthi, Harleen Kaur, Sara Shree Muthuselvam, Amit Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2505.06492)  

**Abstract**: In the dynamic landscape of Industry 4.0, achieving efficiency, precision, and adaptability is essential to optimize manufacturing operations. Industries suffer due to supply chain disruptions caused by anomalies, which are being detected by current AI models but leaving domain experts uncertain without deeper insights into these anomalies. Additionally, operational inefficiencies persist due to inaccurate production forecasts and the limited effectiveness of traditional AI models for processing complex sensor data. Despite these advancements, existing systems lack the seamless integration of these capabilities needed to create a truly unified solution for enhancing production and decision-making. We propose SmartPilot, a neurosymbolic, multiagent CoPilot designed for advanced reasoning and contextual decision-making to address these challenges. SmartPilot processes multimodal sensor data and is compact to deploy on edge devices. It focuses on three key tasks: anomaly prediction, production forecasting, and domain-specific question answering. By bridging the gap between AI capabilities and real-world industrial needs, SmartPilot empowers industries with intelligent decision-making and drives transformative innovation in manufacturing. The demonstration video, datasets, and supplementary materials are available at this https URL. 

---
# Reliable Collaborative Conversational Agent System Based on LLMs and Answer Set Programming 

**Authors**: Yankai Zeng, Gopal Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.06438)  

**Abstract**: As the Large-Language-Model-driven (LLM-driven) Artificial Intelligence (AI) bots became popular, people realized their strong potential in Task-Oriented Dialogue (TOD). However, bots relying wholly on LLMs are unreliable in their knowledge, and whether they can finally produce a correct result for the task is not guaranteed. The collaboration among these agents also remains a challenge, since the necessary information to convey is unclear, and the information transfer is by prompts -- unreliable, and malicious knowledge is easy to inject. With the help of logic programming tools such as Answer Set Programming (ASP), conversational agents can be built safely and reliably, and communication among the agents made more efficient and secure. We proposed an Administrator-Assistant Dual-Agent paradigm, where the two ASP-driven bots share the same knowledge base and complete their tasks independently, while the information can be passed by a Collaborative Rule Set (CRS). The knowledge and information conveyed are encapsulated and invisible to the users, ensuring the security of information transmission. We have constructed AutoManager, a dual-agent system for managing the drive-through window of a fast-food restaurant such as Taco Bell in the US. In AutoManager, the assistant bot takes the customer's order while the administrator bot manages the menu and food supply. We evaluated our AutoManager and compared it with the real-world Taco Bell Drive-Thru AI Order Taker, and the results show that our method is more reliable. 

---
# Neural Brain: A Neuroscience-inspired Framework for Embodied Agents 

**Authors**: Jian Liu, Xiongtao Shi, Thai Duy Nguyen, Haitian Zhang, Tianxiang Zhang, Wei Sun, Yanjie Li, Athanasios V. Vasilakos, Giovanni Iacca, Arshad Ali Khan, Arvind Kumar, Jae Won Cho, Ajmal Mian, Lihua Xie, Erik Cambria, Lin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07634)  

**Abstract**: The rapid evolution of artificial intelligence (AI) has shifted from static, data-driven models to dynamic systems capable of perceiving and interacting with real-world environments. Despite advancements in pattern recognition and symbolic reasoning, current AI systems, such as large language models, remain disembodied, unable to physically engage with the world. This limitation has driven the rise of embodied AI, where autonomous agents, such as humanoid robots, must navigate and manipulate unstructured environments with human-like adaptability. At the core of this challenge lies the concept of Neural Brain, a central intelligence system designed to drive embodied agents with human-like adaptability. A Neural Brain must seamlessly integrate multimodal sensing and perception with cognitive capabilities. Achieving this also requires an adaptive memory system and energy-efficient hardware-software co-design, enabling real-time action in dynamic environments. This paper introduces a unified framework for the Neural Brain of embodied agents, addressing two fundamental challenges: (1) defining the core components of Neural Brain and (2) bridging the gap between static AI models and the dynamic adaptability required for real-world deployment. To this end, we propose a biologically inspired architecture that integrates multimodal active sensing, perception-cognition-action function, neuroplasticity-based memory storage and updating, and neuromorphic hardware/software optimization. Furthermore, we also review the latest research on embodied agents across these four aspects and analyze the gap between current AI systems and human intelligence. By synthesizing insights from neuroscience, we outline a roadmap towards the development of generalizable, autonomous agents capable of human-level intelligence in real-world scenarios. 

---
# Reinforced Internal-External Knowledge Synergistic Reasoning for Efficient Adaptive Search Agent 

**Authors**: Ziyang Huang, Xiaowei Yuan, Yiming Ju, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07596)  

**Abstract**: Retrieval-augmented generation (RAG) is a common strategy to reduce hallucinations in Large Language Models (LLMs). While reinforcement learning (RL) can enable LLMs to act as search agents by activating retrieval capabilities, existing ones often underutilize their internal knowledge. This can lead to redundant retrievals, potential harmful knowledge conflicts, and increased inference latency. To address these limitations, an efficient and adaptive search agent capable of discerning optimal retrieval timing and synergistically integrating parametric (internal) and retrieved (external) knowledge is in urgent need. This paper introduces the Reinforced Internal-External Knowledge Synergistic Reasoning Agent (IKEA), which could indentify its own knowledge boundary and prioritize the utilization of internal knowledge, resorting to external search only when internal knowledge is deemed insufficient. This is achieved using a novel knowledge-boundary aware reward function and a knowledge-boundary aware training dataset. These are designed for internal-external knowledge synergy oriented RL, incentivizing the model to deliver accurate answers, minimize unnecessary retrievals, and encourage appropriate external searches when its own knowledge is lacking. Evaluations across multiple knowledge reasoning tasks demonstrate that IKEA significantly outperforms baseline methods, reduces retrieval frequency significantly, and exhibits robust generalization capabilities. 

---
# Can Generative AI agents behave like humans? Evidence from laboratory market experiments 

**Authors**: R. Maria del Rio-Chanona, Marco Pangallo, Cars Hommes  

**Link**: [PDF](https://arxiv.org/pdf/2505.07457)  

**Abstract**: We explore the potential of Large Language Models (LLMs) to replicate human behavior in economic market experiments. Compared to previous studies, we focus on dynamic feedback between LLM agents: the decisions of each LLM impact the market price at the current step, and so affect the decisions of the other LLMs at the next step. We compare LLM behavior to market dynamics observed in laboratory settings and assess their alignment with human participants' behavior. Our findings indicate that LLMs do not adhere strictly to rational expectations, displaying instead bounded rationality, similarly to human participants. Providing a minimal context window i.e. memory of three previous time steps, combined with a high variability setting capturing response heterogeneity, allows LLMs to replicate broad trends seen in human experiments, such as the distinction between positive and negative feedback markets. However, differences remain at a granular level--LLMs exhibit less heterogeneity in behavior than humans. These results suggest that LLMs hold promise as tools for simulating realistic human behavior in economic contexts, though further research is needed to refine their accuracy and increase behavioral diversity. 

---
# Towards Multi-Agent Reasoning Systems for Collaborative Expertise Delegation: An Exploratory Design Study 

**Authors**: Baixuan Xu, Chunyang Li, Weiqi Wang, Wei Fan, Tianshi Zheng, Haochen Shi, Tao Fan, Yangqiu Song, Qiang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07313)  

**Abstract**: Designing effective collaboration structure for multi-agent LLM systems to enhance collective reasoning is crucial yet remains under-explored. In this paper, we systematically investigate how collaborative reasoning performance is affected by three key design dimensions: (1) Expertise-Domain Alignment, (2) Collaboration Paradigm (structured workflow vs. diversity-driven integration), and (3) System Scale. Our findings reveal that expertise alignment benefits are highly domain-contingent, proving most effective for contextual reasoning tasks. Furthermore, collaboration focused on integrating diverse knowledge consistently outperforms rigid task decomposition. Finally, we empirically explore the impact of scaling the multi-agent system with expertise specialization and study the computational trade off, highlighting the need for more efficient communication protocol design. This work provides concrete guidelines for configuring specialized multi-agent system and identifies critical architectural trade-offs and bottlenecks for scalable multi-agent reasoning. The code will be made available upon acceptance. 

---
# UAV-CodeAgents: Scalable UAV Mission Planning via Multi-Agent ReAct and Vision-Language Reasoning 

**Authors**: Oleg Sautenkov, Yasheerah Yaqoot, Muhammad Ahsan Mustafa, Faryal Batool, Jeffrin Sam, Artem Lykov, Chih-Yung Wen, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2505.07236)  

**Abstract**: We present UAV-CodeAgents, a scalable multi-agent framework for autonomous UAV mission generation, built on large language and vision-language models (LLMs/VLMs). The system leverages the ReAct (Reason + Act) paradigm to interpret satellite imagery, ground high-level natural language instructions, and collaboratively generate UAV trajectories with minimal human supervision. A core component is a vision-grounded, pixel-pointing mechanism that enables precise localization of semantic targets on aerial maps. To support real-time adaptability, we introduce a reactive thinking loop, allowing agents to iteratively reflect on observations, revise mission goals, and coordinate dynamically in evolving environments.
UAV-CodeAgents is evaluated on large-scale mission scenarios involving industrial and environmental fire detection. Our results show that a lower decoding temperature (0.5) yields higher planning reliability and reduced execution time, with an average mission creation time of 96.96 seconds and a success rate of 93%. We further fine-tune Qwen2.5VL-7B on 9,000 annotated satellite images, achieving strong spatial grounding across diverse visual categories. To foster reproducibility and future research, we will release the full codebase and a novel benchmark dataset for vision-language-based UAV planning. 

---
# Towards user-centered interactive medical image segmentation in VR with an assistive AI agent 

**Authors**: Pascal Spiegler, Arash Harirpoush, Yiming Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2505.07214)  

**Abstract**: Crucial in disease analysis and surgical planning, manual segmentation of volumetric medical scans (e.g. MRI, CT) is laborious, error-prone, and challenging to master, while fully automatic algorithms can benefit from user-feedback. Therefore, with the complementary power of the latest radiological AI foundation models and virtual reality (VR)'s intuitive data interaction, we propose SAMIRA, a novel conversational AI agent that assists users with localizing, segmenting, and visualizing 3D medical concepts in VR. Through speech-based interaction, the agent helps users understand radiological features, locate clinical targets, and generate segmentation masks that can be refined with just a few point prompts. The system also supports true-to-scale 3D visualization of segmented pathology to enhance patient-specific anatomical understanding. Furthermore, to determine the optimal interaction paradigm under near-far attention-switching for refining segmentation masks in an immersive, human-in-the-loop workflow, we compare VR controller pointing, head pointing, and eye tracking as input modes. With a user study, evaluations demonstrated a high usability score (SUS=90.0 $\pm$ 9.0), low overall task load, as well as strong support for the proposed VR system's guidance, training potential, and integration of AI in radiological segmentation tasks. 

---
# Internet of Agents: Fundamentals, Applications, and Challenges 

**Authors**: Yuntao Wang, Shaolong Guo, Yanghe Pan, Zhou Su, Fahao Chen, Tom H. Luan, Peng Li, Jiawen Kang, Dusit Niyato  

**Link**: [PDF](https://arxiv.org/pdf/2505.07176)  

**Abstract**: With the rapid proliferation of large language models and vision-language models, AI agents have evolved from isolated, task-specific systems into autonomous, interactive entities capable of perceiving, reasoning, and acting without human intervention. As these agents proliferate across virtual and physical environments, from virtual assistants to embodied robots, the need for a unified, agent-centric infrastructure becomes paramount. In this survey, we introduce the Internet of Agents (IoA) as a foundational framework that enables seamless interconnection, dynamic discovery, and collaborative orchestration among heterogeneous agents at scale. We begin by presenting a general IoA architecture, highlighting its hierarchical organization, distinguishing features relative to the traditional Internet, and emerging applications. Next, we analyze the key operational enablers of IoA, including capability notification and discovery, adaptive communication protocols, dynamic task matching, consensus and conflict-resolution mechanisms, and incentive models. Finally, we identify open research directions toward building resilient and trustworthy IoA ecosystems. 

---
# Seed1.5-VL Technical Report 

**Authors**: Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, Jingji Chen, Jingjia Huang, Kang Lei, Liping Yuan, Lishu Luo, Pengfei Liu, Qinghao Ye, Rui Qian, Shen Yan, Shixiong Zhao, Shuai Peng, Shuangye Li, Sihang Yuan, Sijin Wu, Tianheng Cheng, Weiwei Liu, Wenqian Wang, Xianhan Zeng, Xiao Liu, Xiaobo Qin, Xiaohan Ding, Xiaojun Xiao, Xiaoying Zhang, Xuanwei Zhang, Xuehan Xiong, Yanghua Peng, Yangrui Chen, Yanwei Li, Yanxu Hu, Yi Lin, Yiyuan Hu, Yiyuan Zhang, Youbin Wu, Yu Li, Yudong Liu, Yue Ling, Yujia Qin, Zanbo Wang, Zhiwu He, Aoxue Zhang, Bairen Yi, Bencheng Liao, Can Huang, Can Zhang, Chaorui Deng, Chaoyi Deng, Cheng Lin, Cheng Yuan, Chenggang Li, Chenhui Gou, Chenwei Lou, Chengzhi Wei, Chundian Liu, Chunyuan Li, Deyao Zhu, Donghong Zhong, Feng Li, Feng Zhang, Gang Wu, Guodong Li, Guohong Xiao, Haibin Lin, Haihua Yang, Haoming Wang, Heng Ji, Hongxiang Hao, Hui Shen, Huixia Li, Jiahao Li, Jialong Wu, Jianhua Zhu, Jianpeng Jiao, Jiashi Feng, Jiaze Chen, Jianhui Duan, Jihao Liu, Jin Zeng, Jingqun Tang, Jingyu Sun, Joya Chen, Jun Long, Junda Feng, Junfeng Zhan, Junjie Fang, Junting Lu, Kai Hua, Kai Liu, Kai Shen, Kaiyuan Zhang, Ke Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.07062)  

**Abstract**: We present Seed1.5-VL, a vision-language foundation model designed to advance general-purpose multimodal understanding and reasoning. Seed1.5-VL is composed with a 532M-parameter vision encoder and a Mixture-of-Experts (MoE) LLM of 20B active parameters. Despite its relatively compact architecture, it delivers strong performance across a wide spectrum of public VLM benchmarks and internal evaluation suites, achieving the state-of-the-art performance on 38 out of 60 public benchmarks. Moreover, in agent-centric tasks such as GUI control and gameplay, Seed1.5-VL outperforms leading multimodal systems, including OpenAI CUA and Claude 3.7. Beyond visual and video understanding, it also demonstrates strong reasoning abilities, making it particularly effective for multimodal reasoning challenges such as visual puzzles. We believe these capabilities will empower broader applications across diverse tasks. In this report, we mainly provide a comprehensive review of our experiences in building Seed1.5-VL across model design, data construction, and training at various stages, hoping that this report can inspire further research. Seed1.5-VL is now accessible at this https URL (Volcano Engine Model ID: doubao-1-5-thinking-vision-pro-250428) 

---
# DynamicRAG: Leveraging Outputs of Large Language Model as Feedback for Dynamic Reranking in Retrieval-Augmented Generation 

**Authors**: Jiashuo Sun, Xianrui Zhong, Sizhe Zhou, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.07233)  

**Abstract**: Retrieval-augmented generation (RAG) systems combine large language models (LLMs) with external knowledge retrieval, making them highly effective for knowledge-intensive tasks. A crucial but often under-explored component of these systems is the reranker, which refines retrieved documents to enhance generation quality and explainability. The challenge of selecting the optimal number of documents (k) remains unsolved: too few may omit critical information, while too many introduce noise and inefficiencies. Although recent studies have explored LLM-based rerankers, they primarily leverage internal model knowledge and overlook the rich supervisory signals that LLMs can provide, such as using response quality as feedback for optimizing reranking decisions. In this paper, we propose DynamicRAG, a novel RAG framework where the reranker dynamically adjusts both the order and number of retrieved documents based on the query. We model the reranker as an agent optimized through reinforcement learning (RL), using rewards derived from LLM output quality. Across seven knowledge-intensive datasets, DynamicRAG demonstrates superior performance, achieving state-of-the-art results. The model, data and code are available at this https URL 

---
# RedTeamLLM: an Agentic AI framework for offensive security 

**Authors**: Brian Challita, Pierre Parrend  

**Link**: [PDF](https://arxiv.org/pdf/2505.06913)  

**Abstract**: From automated intrusion testing to discovery of zero-day attacks before software launch, agentic AI calls for great promises in security engineering. This strong capability is bound with a similar threat: the security and research community must build up its models before the approach is leveraged by malicious actors for cybercrime. We therefore propose and evaluate RedTeamLLM, an integrated architecture with a comprehensive security model for automatization of pentest tasks. RedTeamLLM follows three key steps: summarizing, reasoning and act, which embed its operational capacity. This novel framework addresses four open challenges: plan correction, memory management, context window constraint, and generality vs. specialization. Evaluation is performed through the automated resolution of a range of entry-level, but not trivial, CTF challenges. The contribution of the reasoning capability of our agentic AI framework is specifically evaluated. 

---
# ParaView-MCP: An Autonomous Visualization Agent with Direct Tool Use 

**Authors**: Shusen Liu, Haichao Miao, Peer-Timo Bremer  

**Link**: [PDF](https://arxiv.org/pdf/2505.07064)  

**Abstract**: While powerful and well-established, tools like ParaView present a steep learning curve that discourages many potential users. This work introduces ParaView-MCP, an autonomous agent that integrates modern multimodal large language models (MLLMs) with ParaView to not only lower the barrier to entry but also augment ParaView with intelligent decision support. By leveraging the state-of-the-art reasoning, command execution, and vision capabilities of MLLMs, ParaView-MCP enables users to interact with ParaView through natural language and visual inputs. Specifically, our system adopted the Model Context Protocol (MCP) - a standardized interface for model-application communication - that facilitates direct interaction between MLLMs with ParaView's Python API to allow seamless information exchange between the user, the language model, and the visualization tool itself. Furthermore, by implementing a visual feedback mechanism that allows the agent to observe the viewport, we unlock a range of new capabilities, including recreating visualizations from examples, closed-loop visualization parameter updates based on user-defined goals, and even cross-application collaboration involving multiple tools. Broadly, we believe such an agent-driven visualization paradigm can profoundly change the way we interact with visualization tools. We expect a significant uptake in the development of such visualization tools, in both visualization research and industry. 

---
# Reinforcement Learning-Based Monocular Vision Approach for Autonomous UAV Landing 

**Authors**: Tarik Houichime, Younes EL Amrani  

**Link**: [PDF](https://arxiv.org/pdf/2505.06963)  

**Abstract**: This paper introduces an innovative approach for the autonomous landing of Unmanned Aerial Vehicles (UAVs) using only a front-facing monocular camera, therefore obviating the requirement for depth estimation cameras. Drawing on the inherent human estimating process, the proposed method reframes the landing task as an optimization problem. The UAV employs variations in the visual characteristics of a specially designed lenticular circle on the landing pad, where the perceived color and form provide critical information for estimating both altitude and depth. Reinforcement learning algorithms are utilized to approximate the functions governing these estimations, enabling the UAV to ascertain ideal landing settings via training. This method's efficacy is assessed by simulations and experiments, showcasing its potential for robust and accurate autonomous landing without dependence on complex sensor setups. This research contributes to the advancement of cost-effective and efficient UAV landing solutions, paving the way for wider applicability across various fields. 

---
# Convert Language Model into a Value-based Strategic Planner 

**Authors**: Xiaoyu Wang, Yue Zhao, Qingqing Gu, Zhonglin Jiang, Xiaokai Chen, Yong Chen, Luo Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.06987)  

**Abstract**: Emotional support conversation (ESC) aims to alleviate the emotional distress of individuals through effective conversations. Although large language models (LLMs) have obtained remarkable progress on ESC, most of these studies might not define the diagram from the state model perspective, therefore providing a suboptimal solution for long-term satisfaction. To address such an issue, we leverage the Q-learning on LLMs, and propose a framework called straQ*. Our framework allows a plug-and-play LLM to bootstrap the planning during ESC, determine the optimal strategy based on long-term returns, and finally guide the LLM to response. Substantial experiments on ESC datasets suggest that straQ* outperforms many baselines, including direct inference, self-refine, chain of thought, finetuning, and finite state machines. 

---
# ThreatLens: LLM-guided Threat Modeling and Test Plan Generation for Hardware Security Verification 

**Authors**: Dipayan Saha, Hasan Al Shaikh, Shams Tarek, Farimah Farahmandi  

**Link**: [PDF](https://arxiv.org/pdf/2505.06821)  

**Abstract**: Current hardware security verification processes predominantly rely on manual threat modeling and test plan generation, which are labor-intensive, error-prone, and struggle to scale with increasing design complexity and evolving attack methodologies. To address these challenges, we propose ThreatLens, an LLM-driven multi-agent framework that automates security threat modeling and test plan generation for hardware security verification. ThreatLens integrates retrieval-augmented generation (RAG) to extract relevant security knowledge, LLM-powered reasoning for threat assessment, and interactive user feedback to ensure the generation of practical test plans. By automating these processes, the framework reduces the manual verification effort, enhances coverage, and ensures a structured, adaptable approach to security verification. We evaluated our framework on the NEORV32 SoC, demonstrating its capability to automate security verification through structured test plans and validating its effectiveness in real-world scenarios. 

---
# Efficient Robotic Policy Learning via Latent Space Backward Planning 

**Authors**: Dongxiu Liu, Haoyi Niu, Zhihao Wang, Jinliang Zheng, Yinan Zheng, Zhonghong Ou, Jianming Hu, Jianxiong Li, Xianyuan Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2505.06861)  

**Abstract**: Current robotic planning methods often rely on predicting multi-frame images with full pixel details. While this fine-grained approach can serve as a generic world model, it introduces two significant challenges for downstream policy learning: substantial computational costs that hinder real-time deployment, and accumulated inaccuracies that can mislead action extraction. Planning with coarse-grained subgoals partially alleviates efficiency issues. However, their forward planning schemes can still result in off-task predictions due to accumulation errors, leading to misalignment with long-term goals. This raises a critical question: Can robotic planning be both efficient and accurate enough for real-time control in long-horizon, multi-stage tasks? To address this, we propose a Latent Space Backward Planning scheme (LBP), which begins by grounding the task into final latent goals, followed by recursively predicting intermediate subgoals closer to the current state. The grounded final goal enables backward subgoal planning to always remain aware of task completion, facilitating on-task prediction along the entire planning horizon. The subgoal-conditioned policy incorporates a learnable token to summarize the subgoal sequences and determines how each subgoal guides action extraction. Through extensive simulation and real-robot long-horizon experiments, we show that LBP outperforms existing fine-grained and forward planning methods, achieving SOTA performance. Project Page: this https URL 

---
# Balancing Progress and Safety: A Novel Risk-Aware Objective for RL in Autonomous Driving 

**Authors**: Ahmed Abouelazm, Jonas Michel, Helen Gremmelmaier, Tim Joseph, Philip Schörner, J. Marius Zöllner  

**Link**: [PDF](https://arxiv.org/pdf/2505.06737)  

**Abstract**: Reinforcement Learning (RL) is a promising approach for achieving autonomous driving due to robust decision-making capabilities. RL learns a driving policy through trial and error in traffic scenarios, guided by a reward function that combines the driving objectives. The design of such reward function has received insufficient attention, yielding ill-defined rewards with various pitfalls. Safety, in particular, has long been regarded only as a penalty for collisions. This leaves the risks associated with actions leading up to a collision unaddressed, limiting the applicability of RL in real-world scenarios. To address these shortcomings, our work focuses on enhancing the reward formulation by defining a set of driving objectives and structuring them hierarchically. Furthermore, we discuss the formulation of these objectives in a normalized manner to transparently determine their contribution to the overall reward. Additionally, we introduce a novel risk-aware objective for various driving interactions based on a two-dimensional ellipsoid function and an extension of Responsibility-Sensitive Safety (RSS) concepts. We evaluate the efficacy of our proposed reward in unsignalized intersection scenarios with varying traffic densities. The approach decreases collision rates by 21\% on average compared to baseline rewards and consistently surpasses them in route progress and cumulative reward, demonstrating its capability to promote safer driving behaviors while maintaining high-performance levels. 

---
# Boundary-Guided Trajectory Prediction for Road Aware and Physically Feasible Autonomous Driving 

**Authors**: Ahmed Abouelazm, Mianzhi Liu, Christian Hubschneider, Yin Wu, Daniel Slieter, J. Marius Zöllner  

**Link**: [PDF](https://arxiv.org/pdf/2505.06740)  

**Abstract**: Accurate prediction of surrounding road users' trajectories is essential for safe and efficient autonomous driving. While deep learning models have improved performance, challenges remain in preventing off-road predictions and ensuring kinematic feasibility. Existing methods incorporate road-awareness modules and enforce kinematic constraints but lack plausibility guarantees and often introduce trade-offs in complexity and flexibility. This paper proposes a novel framework that formulates trajectory prediction as a constrained regression guided by permissible driving directions and their boundaries. Using the agent's current state and an HD map, our approach defines the valid boundaries and ensures on-road predictions by training the network to learn superimposed paths between left and right boundary polylines. To guarantee feasibility, the model predicts acceleration profiles that determine the vehicle's travel distance along these paths while adhering to kinematic constraints. We evaluate our approach on the Argoverse-2 dataset against the HPTR baseline. Our approach shows a slight decrease in benchmark metrics compared to HPTR but notably improves final displacement error and eliminates infeasible trajectories. Moreover, the proposed approach has superior generalization to less prevalent maneuvers and unseen out-of-distribution scenarios, reducing the off-road rate under adversarial attacks from 66\% to just 1\%. These results highlight the effectiveness of our approach in generating feasible and robust predictions. 

---
# TPK: Trustworthy Trajectory Prediction Integrating Prior Knowledge For Interpretability and Kinematic Feasibility 

**Authors**: Marius Baden, Ahmed Abouelazm, Christian Hubschneider, Yin Wu, Daniel Slieter, J. Marius Zöllner  

**Link**: [PDF](https://arxiv.org/pdf/2505.06743)  

**Abstract**: Trajectory prediction is crucial for autonomous driving, enabling vehicles to navigate safely by anticipating the movements of surrounding road users. However, current deep learning models often lack trustworthiness as their predictions can be physically infeasible and illogical to humans. To make predictions more trustworthy, recent research has incorporated prior knowledge, like the social force model for modeling interactions and kinematic models for physical realism. However, these approaches focus on priors that suit either vehicles or pedestrians and do not generalize to traffic with mixed agent classes. We propose incorporating interaction and kinematic priors of all agent classes--vehicles, pedestrians, and cyclists with class-specific interaction layers to capture agent behavioral differences. To improve the interpretability of the agent interactions, we introduce DG-SFM, a rule-based interaction importance score that guides the interaction layer. To ensure physically feasible predictions, we proposed suitable kinematic models for all agent classes with a novel pedestrian kinematic model. We benchmark our approach on the Argoverse 2 dataset, using the state-of-the-art transformer HPTR as our baseline. Experiments demonstrate that our method improves interaction interpretability, revealing a correlation between incorrect predictions and divergence from our interaction prior. Even though incorporating the kinematic models causes a slight decrease in accuracy, they eliminate infeasible trajectories found in the dataset and the baseline model. Thus, our approach fosters trust in trajectory prediction as its interaction reasoning is interpretable, and its predictions adhere to physics. 

---
# Towards AI-Driven Human-Machine Co-Teaming for Adaptive and Agile Cyber Security Operation Centers 

**Authors**: Massimiliano Albanese, Xinming Ou, Kevin Lybarger, Daniel Lende, Dmitry Goldgof  

**Link**: [PDF](https://arxiv.org/pdf/2505.06394)  

**Abstract**: Security Operations Centers (SOCs) face growing challenges in managing cybersecurity threats due to an overwhelming volume of alerts, a shortage of skilled analysts, and poorly integrated tools. Human-AI collaboration offers a promising path to augment the capabilities of SOC analysts while reducing their cognitive overload. To this end, we introduce an AI-driven human-machine co-teaming paradigm that leverages large language models (LLMs) to enhance threat intelligence, alert triage, and incident response workflows. We present a vision in which LLM-based AI agents learn from human analysts the tacit knowledge embedded in SOC operations, enabling the AI agents to improve their performance on SOC tasks through this co-teaming. We invite SOCs to collaborate with us to further develop this process and uncover replicable patterns where human-AI co-teaming yields measurable improvements in SOC productivity. 

---
# ARDNS-FN-Quantum: A Quantum-Enhanced Reinforcement Learning Framework with Cognitive-Inspired Adaptive Exploration for Dynamic Environments 

**Authors**: Umberto Gonçalves de Sousa  

**Link**: [PDF](https://arxiv.org/pdf/2505.06300)  

**Abstract**: Reinforcement learning (RL) has transformed sequential decision making, yet traditional algorithms like Deep Q-Networks (DQNs) and Proximal Policy Optimization (PPO) often struggle with efficient exploration, stability, and adaptability in dynamic environments. This study presents ARDNS-FN-Quantum (Adaptive Reward-Driven Neural Simulator with Quantum enhancement), a novel framework that integrates a 2-qubit quantum circuit for action selection, a dual-memory system inspired by human cognition, and adaptive exploration strategies modulated by reward variance and curiosity. Evaluated in a 10X10 grid-world over 20,000 episodes, ARDNS-FN-Quantum achieves a 99.5% success rate (versus 81.3% for DQN and 97.0% for PPO), a mean reward of 9.0528 across all episodes (versus 1.2941 for DQN and 7.6196 for PPO), and an average of 46.7 steps to goal (versus 135.9 for DQN and 62.5 for PPO). In the last 100 episodes, it records a mean reward of 9.1652 (versus 7.0916 for DQN and 9.0310 for PPO) and 37.2 steps to goal (versus 52.7 for DQN and 53.4 for PPO). Graphical analyses, including learning curves, steps-to-goal trends, reward variance, and reward distributions, demonstrate ARDNS-FN-Quantum's superior stability (reward variance 5.424 across all episodes versus 252.262 for DQN and 76.583 for PPO) and efficiency. By bridging quantum computing, cognitive science, and RL, ARDNS-FN-Quantum offers a scalable, human-like approach to adaptive learning in uncertain environments, with potential applications in robotics, autonomous systems, and decision-making under uncertainty. 

---
# Bi-LSTM based Multi-Agent DRL with Computation-aware Pruning for Agent Twins Migration in Vehicular Embodied AI Networks 

**Authors**: Yuxiang Wei, Zhuoqi Zeng, Yue Zhong, Jiawen Kang, Ryan Wen Liu, M. Shamim Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2505.06378)  

**Abstract**: With the advancement of large language models and embodied Artificial Intelligence (AI) in the intelligent transportation scenarios, the combination of them in intelligent transportation spawns the Vehicular Embodied AI Network (VEANs). In VEANs, Autonomous Vehicles (AVs) are typical agents whose local advanced AI applications are defined as vehicular embodied AI agents, enabling capabilities such as environment perception and multi-agent collaboration. Due to computation latency and resource constraints, the local AI applications and services running on vehicular embodied AI agents need to be migrated, and subsequently referred to as vehicular embodied AI agent twins, which drive the advancement of vehicular embodied AI networks to offload intensive tasks to Roadside Units (RSUs), mitigating latency problems while maintaining service quality. Recognizing workload imbalance among RSUs in traditional approaches, we model AV-RSU interactions as a Stackelberg game to optimize bandwidth resource allocation for efficient migration. A Tiny Multi-Agent Bidirectional LSTM Proximal Policy Optimization (TMABLPPO) algorithm is designed to approximate the Stackelberg equilibrium through decentralized coordination. Furthermore, a personalized neural network pruning algorithm based on Path eXclusion (PX) dynamically adapts to heterogeneous AV computation capabilities by identifying task-critical parameters in trained models, reducing model complexity with less performance degradation. Experimental validation confirms the algorithm's effectiveness in balancing system load and minimizing delays, demonstrating significant improvements in vehicular embodied AI agent deployment. 

---
# Threat Modeling for AI: The Case for an Asset-Centric Approach 

**Authors**: Jose Sanchez Vicarte, Marcin Spoczynski, Mostafa Elsaid  

**Link**: [PDF](https://arxiv.org/pdf/2505.06315)  

**Abstract**: Recent advances in AI are transforming AI's ubiquitous presence in our world from that of standalone AI-applications into deeply integrated AI-agents. These changes have been driven by agents' increasing capability to autonomously make decisions and initiate actions, using existing applications; whether those applications are AI-based or not. This evolution enables unprecedented levels of AI integration, with agents now able to take actions on behalf of systems and users -- including, in some cases, the powerful ability for the AI to write and execute scripts as it deems necessary. With AI systems now able to autonomously execute code, interact with external systems, and operate without human oversight, traditional security approaches fall short.
This paper introduces an asset-centric methodology for threat modeling AI systems that addresses the unique security challenges posed by integrated AI agents. Unlike existing top-down frameworks that analyze individual attacks within specific product contexts, our bottom-up approach enables defenders to systematically identify how vulnerabilities -- both conventional and AI-specific -- impact critical AI assets across distributed infrastructures used to develop and deploy these agents. This methodology allows security teams to: (1) perform comprehensive analysis that communicates effectively across technical domains, (2) quantify security assumptions about third-party AI components without requiring visibility into their implementation, and (3) holistically identify AI-based vulnerabilities relevant to their specific product context. This approach is particularly relevant for securing agentic systems with complex autonomous capabilities. By focusing on assets rather than attacks, our approach scales with the rapidly evolving threat landscape while accommodating increasingly complex and distributed AI development pipelines. 

---
# Structural Entropy Guided Agent for Detecting and Repairing Knowledge Deficiencies in LLMs 

**Authors**: Yifan Wei, Xiaoyan Yu, Tengfei Pan, Angsheng Li, Li Du  

**Link**: [PDF](https://arxiv.org/pdf/2505.07184)  

**Abstract**: Large language models (LLMs) have achieved unprecedented performance by leveraging vast pretraining corpora, yet their performance remains suboptimal in knowledge-intensive domains such as medicine and scientific research, where high factual precision is required. While synthetic data provides a promising avenue for augmenting domain knowledge, existing methods frequently generate redundant samples that do not align with the model's true knowledge gaps. To overcome this limitation, we propose a novel Structural Entropy-guided Knowledge Navigator (SENATOR) framework that addresses the intrinsic knowledge deficiencies of LLMs. Our approach employs the Structure Entropy (SE) metric to quantify uncertainty along knowledge graph paths and leverages Monte Carlo Tree Search (MCTS) to selectively explore regions where the model lacks domain-specific knowledge. Guided by these insights, the framework generates targeted synthetic data for supervised fine-tuning, enabling continuous self-improvement. Experimental results on LLaMA-3 and Qwen2 across multiple domain-specific benchmarks show that SENATOR effectively detects and repairs knowledge deficiencies, achieving notable performance improvements. The code and data for our methods and experiments are available at this https URL. 

---
# EcoLANG: Efficient and Effective Agent Communication Language Induction for Social Simulation 

**Authors**: Xinyi Mou, Chen Qian, Wei Liu, Xuanjing Huang, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.06904)  

**Abstract**: Large language models (LLMs) have demonstrated an impressive ability to role-play humans and replicate complex social dynamics. While large-scale social simulations are gaining increasing attention, they still face significant challenges, particularly regarding high time and computation costs. Existing solutions, such as distributed mechanisms or hybrid agent-based model (ABM) integrations, either fail to address inference costs or compromise accuracy and generalizability. To this end, we propose EcoLANG: Efficient and Effective Agent Communication Language Induction for Social Simulation. EcoLANG operates in two stages: (1) language evolution, where we filter synonymous words and optimize sentence-level rules through natural selection, and (2) language utilization, where agents in social simulations communicate using the evolved language. Experimental results demonstrate that EcoLANG reduces token consumption by over 20%, enhancing efficiency without sacrificing simulation accuracy. 

---
# ScaleMCP: Dynamic and Auto-Synchronizing Model Context Protocol Tools for LLM Agents 

**Authors**: Elias Lumer, Anmol Gulati, Vamse Kumar Subbiah, Pradeep Honaganahalli Basavaraju, James A. Burke  

**Link**: [PDF](https://arxiv.org/pdf/2505.06416)  

**Abstract**: Recent advancements in Large Language Models (LLMs) and the introduction of the Model Context Protocol (MCP) have significantly expanded LLM agents' capability to interact dynamically with external tools and APIs. However, existing tool selection frameworks do not integrate MCP servers, instead relying heavily on error-prone manual updates to monolithic local tool repositories, leading to duplication, inconsistencies, and inefficiencies. Additionally, current approaches abstract tool selection before the LLM agent is invoked, limiting its autonomy and hindering dynamic re-querying capabilities during multi-turn interactions. To address these issues, we introduce ScaleMCP, a novel tool selection approach that dynamically equips LLM agents with a MCP tool retriever, giving agents the autonomy to add tools into their memory, as well as an auto-synchronizing tool storage system pipeline through CRUD (create, read, update, delete) operations with MCP servers as the single source of truth. We also propose a novel embedding strategy, Tool Document Weighted Average (TDWA), designed to selectively emphasize critical components of tool documents (e.g. tool name or synthetic questions) during the embedding process. Comprehensive evaluations conducted on a created dataset of 5,000 financial metric MCP servers, across 10 LLM models, 5 embedding models, and 5 retriever types, demonstrate substantial improvements in tool retrieval and agent invocation performance, emphasizing ScaleMCP's effectiveness in scalable, dynamic tool selection and invocation. 

---
# DARLR: Dual-Agent Offline Reinforcement Learning for Recommender Systems with Dynamic Reward 

**Authors**: Yi Zhang, Ruihong Qiu, Xuwei Xu, Jiajun Liu, Sen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07257)  

**Abstract**: Model-based offline reinforcement learning (RL) has emerged as a promising approach for recommender systems, enabling effective policy learning by interacting with frozen world models. However, the reward functions in these world models, trained on sparse offline logs, often suffer from inaccuracies. Specifically, existing methods face two major limitations in addressing this challenge: (1) deterministic use of reward functions as static look-up tables, which propagates inaccuracies during policy learning, and (2) static uncertainty designs that fail to effectively capture decision risks and mitigate the impact of these inaccuracies. In this work, a dual-agent framework, DARLR, is proposed to dynamically update world models to enhance recommendation policies. To achieve this, a \textbf{\textit{selector}} is introduced to identify reference users by balancing similarity and diversity so that the \textbf{\textit{recommender}} can aggregate information from these users and iteratively refine reward estimations for dynamic reward shaping. Further, the statistical features of the selected users guide the dynamic adaptation of an uncertainty penalty to better align with evolving recommendation requirements. Extensive experiments on four benchmark datasets demonstrate the superior performance of DARLR, validating its effectiveness. The code is available at this https URL. 

---
# KAQG: A Knowledge-Graph-Enhanced RAG for Difficulty-Controlled Question Generation 

**Authors**: Ching Han Chen, Ming Fang Shiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07618)  

**Abstract**: KAQG introduces a decisive breakthrough for Retrieval-Augmented Generation (RAG) by explicitly tackling the two chronic weaknesses of current pipelines: transparent multi-step reasoning and fine-grained cognitive difficulty control. This transforms RAG from a passive retriever into an accountable generator of calibrated exam items. Technically, the framework fuses knowledge graphs, RAG retrieval, and educational assessment theory into a single pipeline. Domain passages are parsed into a structured graph; graph-aware retrieval feeds fact chains to an LLM; and an assessment layer governed by Bloom's Taxonomy levels and Item Response Theory (IRT) transforms those chains into psychometrically sound questions. This cross-disciplinary marriage yields two scholarly contributions: it shows how semantic graph contexts guide LLM reasoning paths, and it operationalizes difficulty metrics within the generation process, producing items whose IRT parameters match expert benchmarks. Every module, from KG construction scripts to the multi-agent reasoning scheduler and the automatic IRT validator, is openly released on GitHub. This enables peer laboratories to replicate experiments, benchmark against baselines, and extend individual components without licensing barriers. Its reproducible design paves the way for rigorous ablation studies, cross-domain transfer experiments, and shared leaderboards on multi-step reasoning benchmarks. 

---
# NetSight: Graph Attention Based Traffic Forecasting in Computer Networks 

**Authors**: Jinming Xing, Guoheng Sun, Hui Sun, Linchao Pan, Shakir Mahmood, Xuanhao Luo, Muhammad Shahzad  

**Link**: [PDF](https://arxiv.org/pdf/2505.07034)  

**Abstract**: The traffic in today's networks is increasingly influenced by the interactions among network nodes as well as by the temporal fluctuations in the demands of the nodes. Traditional statistical prediction methods are becoming obsolete due to their inability to address the non-linear and dynamic spatio-temporal dependencies present in today's network traffic. The most promising direction of research today is graph neural networks (GNNs) based prediction approaches that are naturally suited to handle graph-structured data. Unfortunately, the state-of-the-art GNN approaches separate the modeling of spatial and temporal information, resulting in the loss of important information about joint dependencies. These GNN based approaches further do not model information at both local and global scales simultaneously, leaving significant room for improvement. To address these challenges, we propose NetSight. NetSight learns joint spatio-temporal dependencies simultaneously at both global and local scales from the time-series of measurements of any given network metric collected at various nodes in a network. Using the learned information, NetSight can then accurately predict the future values of the given network metric at those nodes in the network. We propose several new concepts and techniques in the design of NetSight, such as spatio-temporal adjacency matrix and node normalization. Through extensive evaluations and comparison with prior approaches using data from two large real-world networks, we show that NetSight significantly outperforms all prior state-of-the-art approaches. We will release the source code and data used in the evaluation of NetSight on the acceptance of this paper. 

---
