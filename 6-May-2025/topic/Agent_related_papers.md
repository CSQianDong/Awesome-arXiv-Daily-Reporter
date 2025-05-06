# Technical Report: Evaluating Goal Drift in Language Model Agents 

**Authors**: Rauno Arike, Elizabeth Donoway, Henning Bartsch, Marius Hobbhahn  

**Link**: [PDF](https://arxiv.org/pdf/2505.02709)  

**Abstract**: As language models (LMs) are increasingly deployed as autonomous agents, their robust adherence to human-assigned objectives becomes crucial for safe operation. When these agents operate independently for extended periods without human oversight, even initially well-specified goals may gradually shift. Detecting and measuring goal drift - an agent's tendency to deviate from its original objective over time - presents significant challenges, as goals can shift gradually, causing only subtle behavioral changes. This paper proposes a novel approach to analyzing goal drift in LM agents. In our experiments, agents are first explicitly given a goal through their system prompt, then exposed to competing objectives through environmental pressures. We demonstrate that while the best-performing agent (a scaffolded version of Claude 3.5 Sonnet) maintains nearly perfect goal adherence for more than 100,000 tokens in our most difficult evaluation setting, all evaluated models exhibit some degree of goal drift. We also find that goal drift correlates with models' increasing susceptibility to pattern-matching behaviors as the context length grows. 

---
# Voila: Voice-Language Foundation Models for Real-Time Autonomous Interaction and Voice Role-Play 

**Authors**: Yemin Shi, Yu Shu, Siwei Dong, Guangyi Liu, Jaward Sesay, Jingwen Li, Zhiting Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.02707)  

**Abstract**: A voice AI agent that blends seamlessly into daily life would interact with humans in an autonomous, real-time, and emotionally expressive manner. Rather than merely reacting to commands, it would continuously listen, reason, and respond proactively, fostering fluid, dynamic, and emotionally resonant interactions. We introduce Voila, a family of large voice-language foundation models that make a step towards this vision. Voila moves beyond traditional pipeline systems by adopting a new end-to-end architecture that enables full-duplex, low-latency conversations while preserving rich vocal nuances such as tone, rhythm, and emotion. It achieves a response latency of just 195 milliseconds, surpassing the average human response time. Its hierarchical multi-scale Transformer integrates the reasoning capabilities of large language models (LLMs) with powerful acoustic modeling, enabling natural, persona-aware voice generation -- where users can simply write text instructions to define the speaker's identity, tone, and other characteristics. Moreover, Voila supports over one million pre-built voices and efficient customization of new ones from brief audio samples as short as 10 seconds. Beyond spoken dialogue, Voila is designed as a unified model for a wide range of voice-based applications, including automatic speech recognition (ASR), Text-to-Speech (TTS), and, with minimal adaptation, multilingual speech translation. Voila is fully open-sourced to support open research and accelerate progress toward next-generation human-machine interactions. 

---
# El Agente: An Autonomous Agent for Quantum Chemistry 

**Authors**: Yunheng Zou, Austin H. Cheng, Abdulrahman Aldossary, Jiaru Bai, Shi Xuan Leong, Jorge Arturo Campos-Gonzalez-Angulo, Changhyeok Choi, Cher Tian Ser, Gary Tom, Andrew Wang, Zijian Zhang, Ilya Yakavets, Han Hao, Chris Crebolder, Varinia Bernales, Alán Aspuru-Guzik  

**Link**: [PDF](https://arxiv.org/pdf/2505.02484)  

**Abstract**: Computational chemistry tools are widely used to study the behaviour of chemical phenomena. Yet, the complexity of these tools can make them inaccessible to non-specialists and challenging even for experts. In this work, we introduce El Agente Q, an LLM-based multi-agent system that dynamically generates and executes quantum chemistry workflows from natural language user prompts. The system is built on a novel cognitive architecture featuring a hierarchical memory framework that enables flexible task decomposition, adaptive tool selection, post-analysis, and autonomous file handling and submission. El Agente Q is benchmarked on six university-level course exercises and two case studies, demonstrating robust problem-solving performance (averaging >87% task success) and adaptive error handling through in situ debugging. It also supports longer-term, multi-step task execution for more complex workflows, while maintaining transparency through detailed action trace logs. Together, these capabilities lay the foundation for increasingly autonomous and accessible quantum chemistry. 

---
# AutoLibra: Agent Metric Induction from Open-Ended Feedback 

**Authors**: Hao Zhu, Phil Cuvin, Xinkai Yu, Charlotte Ka Yee Yan, Jason Zhang, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.02820)  

**Abstract**: Agents are predominantly evaluated and optimized via task success metrics, which are coarse, rely on manual design from experts, and fail to reward intermediate emergent behaviors. We propose AutoLibra, a framework for agent evaluation, that transforms open-ended human feedback, e.g., "If you find that the button is disabled, don't click it again", or "This agent has too much autonomy to decide what to do on its own", into metrics for evaluating fine-grained behaviors in agent trajectories. AutoLibra accomplishes this by grounding feedback to an agent's behavior, clustering similar positive and negative behaviors, and creating concrete metrics with clear definitions and concrete examples, which can be used for prompting LLM-as-a-Judge as evaluators. We further propose two meta-metrics to evaluate the alignment of a set of (induced) metrics with open feedback: "coverage" and "redundancy". Through optimizing these meta-metrics, we experimentally demonstrate AutoLibra's ability to induce more concrete agent evaluation metrics than the ones proposed in previous agent evaluation benchmarks and discover new metrics to analyze agents. We also present two applications of AutoLibra in agent improvement: First, we show that AutoLibra-induced metrics serve as better prompt-engineering targets than the task success rate on a wide range of text game tasks, improving agent performance over baseline by a mean of 20%. Second, we show that AutoLibra can iteratively select high-quality fine-tuning data for web navigation agents. Our results suggest that AutoLibra is a powerful task-agnostic tool for evaluating and improving language agents. 

---
# SafeMate: A Model Context Protocol-Based Multimodal Agent for Emergency Preparedness 

**Authors**: Junfeng Jiao, Jihyung Park, Yiming Xu, Lucy Atkinson  

**Link**: [PDF](https://arxiv.org/pdf/2505.02306)  

**Abstract**: Despite the abundance of public safety documents and emergency protocols, most individuals remain ill-equipped to interpret and act on such information during crises. Traditional emergency decision support systems (EDSS) are designed for professionals and rely heavily on static documents like PDFs or SOPs, which are difficult for non-experts to navigate under stress. This gap between institutional knowledge and public accessibility poses a critical barrier to effective emergency preparedness and response.
We introduce SafeMate, a retrieval-augmented AI assistant that delivers accurate, context-aware guidance to general users in both preparedness and active emergency scenarios. Built on the Model Context Protocol (MCP), SafeMate dynamically routes user queries to tools for document retrieval, checklist generation, and structured summarization. It uses FAISS with cosine similarity to identify relevant content from trusted sources. 

---
# Agentic Neurodivergence as a Contingent Solution to the AI Alignment Problem 

**Authors**: Alberto Hernández-Espinosa, Felipe S. Abrahão, Olaf Witkowski, Hector Zenil  

**Link**: [PDF](https://arxiv.org/pdf/2505.02581)  

**Abstract**: The AI alignment problem, which focusses on ensuring that artificial intelligence (AI), including AGI and ASI, systems act according to human values, presents profound challenges. With the progression from narrow AI to Artificial General Intelligence (AGI) and Superintelligence, fears about control and existential risk have escalated. This paper demonstrates that achieving complete alignment is inherently unattainable due to mathematical principles rooted in the foundations of predicate logic and computability, in particular Turing's computational universality, Gödel's incompleteness and Chaitin's randomness. Instead, we argue that embracing AI misalignment or agent's `neurodivergence' as a contingent strategy, defined as fostering a dynamic ecosystem of competing, partially aligned agents, is a possible only viable path to mitigate risks. Through mathematical proofs and an experimental design, we explore how misalignment may serve and should be promoted as a counterbalancing mechanism to team up with whichever agents are most aligned AI to human values, ensuring that no single system dominates destructively. The main premise of our contribution is that misalignment is inevitable because full AI-human alignment is a mathematical impossibility from Turing-complete systems which we also prove in this paper, a feature then inherited to AGI and ASI systems. We introduce and test `change-of-opinion' attacks based on this kind of perturbation and intervention analysis to study how agents may neutralise friendly or unfriendly AIs through cooperation, competition or malice. 

---
# Interpretable Emergent Language Using Inter-Agent Transformers 

**Authors**: Mannan Bhardwaj  

**Link**: [PDF](https://arxiv.org/pdf/2505.02215)  

**Abstract**: This paper explores the emergence of language in multi-agent reinforcement learning (MARL) using transformers. Existing methods such as RIAL, DIAL, and CommNet enable agent communication but lack interpretability. We propose Differentiable Inter-Agent Transformers (DIAT), which leverage self-attention to learn symbolic, human-understandable communication protocols. Through experiments, DIAT demonstrates the ability to encode observations into interpretable vocabularies and meaningful embeddings, effectively solving cooperative tasks. These results highlight the potential of DIAT for interpretable communication in complex multi-agent environments. 

---
# HyperTree Planning: Enhancing LLM Reasoning via Hierarchical Thinking 

**Authors**: Runquan Gui, Zhihai Wang, Jie Wang, Chi Ma, Huiling Zhen, Mingxuan Yuan, Jianye Hao, Defu Lian, Enhong Chen, Feng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.02322)  

**Abstract**: Recent advancements have significantly enhanced the performance of large language models (LLMs) in tackling complex reasoning tasks, achieving notable success in domains like mathematical and logical reasoning. However, these methods encounter challenges with complex planning tasks, primarily due to extended reasoning steps, diverse constraints, and the challenge of handling multiple distinct sub-tasks. To address these challenges, we propose HyperTree Planning (HTP), a novel reasoning paradigm that constructs hypertree-structured planning outlines for effective planning. The hypertree structure enables LLMs to engage in hierarchical thinking by flexibly employing the divide-and-conquer strategy, effectively breaking down intricate reasoning steps, accommodating diverse constraints, and managing multiple distinct sub-tasks in a well-organized manner. We further introduce an autonomous planning framework that completes the planning process by iteratively refining and expanding the hypertree-structured planning outlines. Experiments demonstrate the effectiveness of HTP, achieving state-of-the-art accuracy on the TravelPlanner benchmark with Gemini-1.5-Pro, resulting in a 3.6 times performance improvement over o1-preview. 

---
# Leveraging LLM Agents and Digital Twins for Fault Handling in Process Plants 

**Authors**: Milapji Singh Gill, Javal Vyas, Artan Markaj, Felix Gehlhoff, Mehmet Mercangöz  

**Link**: [PDF](https://arxiv.org/pdf/2505.02076)  

**Abstract**: Advances in Automation and Artificial Intelligence continue to enhance the autonomy of process plants in handling various operational scenarios. However, certain tasks, such as fault handling, remain challenging, as they rely heavily on human expertise. This highlights the need for systematic, knowledge-based methods. To address this gap, we propose a methodological framework that integrates Large Language Model (LLM) agents with a Digital Twin environment. The LLM agents continuously interpret system states and initiate control actions, including responses to unexpected faults, with the goal of returning the system to normal operation. In this context, the Digital Twin acts both as a structured repository of plant-specific engineering knowledge for agent prompting and as a simulation platform for the systematic validation and verification of the generated corrective control actions. The evaluation using a mixing module of a process plant demonstrates that the proposed framework is capable not only of autonomously controlling the mixing module, but also of generating effective corrective actions to mitigate a pipe clogging with only a few reprompts. 

---
# MemEngine: A Unified and Modular Library for Developing Advanced Memory of LLM-based Agents 

**Authors**: Zeyu Zhang, Quanyu Dai, Xu Chen, Rui Li, Zhongyang Li, Zhenhua Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.02099)  

**Abstract**: Recently, large language model based (LLM-based) agents have been widely applied across various fields. As a critical part, their memory capabilities have captured significant interest from both industrial and academic communities. Despite the proposal of many advanced memory models in recent research, however, there remains a lack of unified implementations under a general framework. To address this issue, we develop a unified and modular library for developing advanced memory models of LLM-based agents, called MemEngine. Based on our framework, we implement abundant memory models from recent research works. Additionally, our library facilitates convenient and extensible memory development, and offers user-friendly and pluggable memory usage. For benefiting our community, we have made our project publicly available at this https URL. 

---
# From Mind to Machine: The Rise of Manus AI as a Fully Autonomous Digital Agent 

**Authors**: Minjie Shen, Qikai Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.02024)  

**Abstract**: Manus AI is a general-purpose AI agent introduced in early 2025, marking a significant advancement in autonomous artificial intelligence. Developed by the Chinese startup this http URL, Manus is designed to bridge the gap between "mind" and "hand" - combining the reasoning and planning capabilities of large language models with the ability to execute complex, end-to-end tasks that produce tangible outcomes. This paper presents a comprehensive overview of Manus AI, exploring its core technical architecture, diverse applications across sectors such as healthcare, finance, manufacturing, robotics, and gaming, as well as its key strengths, current limitations, and future potential. Positioned as a preview of what lies ahead, Manus AI represents a shift toward intelligent agents that can translate high-level intentions into real-world actions, heralding a new era of human-AI collaboration. 

---
# TutorGym: A Testbed for Evaluating AI Agents as Tutors and Students 

**Authors**: Daniel Weitekamp, Momin N. Siddiqui, Christopher J. MacLellan  

**Link**: [PDF](https://arxiv.org/pdf/2505.01563)  

**Abstract**: Recent improvements in large language model (LLM) performance on academic benchmarks, such as MATH and GSM8K, have emboldened their use as standalone tutors and as simulations of human learning. However, these new applications require more than evaluations of final solution generation. We introduce TutorGym to evaluate these applications more directly. TutorGym is a standard interface for testing artificial intelligence (AI) agents within existing intelligent tutoring systems (ITS) that have been tested and refined in classroom studies, including Cognitive Tutors (CTAT), Apprentice Tutors, and OATutors. TutorGym is more than a simple problem-solution benchmark, it situates AI agents within the interactive interfaces of existing ITSs. At each step of problem-solving, AI agents are asked what they would do as a tutor or as a learner. As tutors, AI agents are prompted to provide tutoring support -- such as generating examples, hints, and step-level correctness feedback -- which can be evaluated directly against the adaptive step-by-step support provided by existing ITSs. As students, agents directly learn from ITS instruction, and their mistakes and learning trajectories can be compared to student data. TutorGym establishes a common framework for training and evaluating diverse AI agents, including LLMs, computational models of learning, and reinforcement learning agents, within a growing suite of learning environments. Currently, TutorGym includes 223 different tutor domains. In an initial evaluation, we find that current LLMs are poor at tutoring -- none did better than chance at labeling incorrect actions, and next-step actions were correct only ~52-70% of the time -- but they could produce remarkably human-like learning curves when trained as students with in-context learning. 

---
# Agentic Reasoning and Tool Integration for LLMs via Reinforcement Learning 

**Authors**: Joykirat Singh, Raghav Magazine, Yash Pandya, Akshay Nambi  

**Link**: [PDF](https://arxiv.org/pdf/2505.01441)  

**Abstract**: Large language models (LLMs) have achieved remarkable progress in complex reasoning tasks, yet they remain fundamentally limited by their reliance on static internal knowledge and text-only reasoning. Real-world problem solving often demands dynamic, multi-step reasoning, adaptive decision making, and the ability to interact with external tools and environments. In this work, we introduce ARTIST (Agentic Reasoning and Tool Integration in Self-improving Transformers), a unified framework that tightly couples agentic reasoning, reinforcement learning, and tool integration for LLMs. ARTIST enables models to autonomously decide when, how, and which tools to invoke within multi-turn reasoning chains, leveraging outcome-based RL to learn robust strategies for tool use and environment interaction without requiring step-level supervision. Extensive experiments on mathematical reasoning and multi-turn function calling benchmarks show that ARTIST consistently outperforms state-of-the-art baselines, with up to 22% absolute improvement over base models and strong gains on the most challenging tasks. Detailed studies and metric analyses reveal that agentic RL training leads to deeper reasoning, more effective tool use, and higher-quality solutions. Our results establish agentic RL with tool integration as a powerful new frontier for robust, interpretable, and generalizable problem-solving in LLMs. 

---
# Graph Neural Network-Based Reinforcement Learning for Controlling Biological Networks: The GATTACA Framework 

**Authors**: Andrzej Mizera, Jakub Zarzycki  

**Link**: [PDF](https://arxiv.org/pdf/2505.02712)  

**Abstract**: Cellular reprogramming, the artificial transformation of one cell type into another, has been attracting increasing research attention due to its therapeutic potential for complex diseases. However, discovering reprogramming strategies through classical wet-lab experiments is hindered by lengthy time commitments and high costs. In this study, we explore the use of deep reinforcement learning (DRL) to control Boolean network models of complex biological systems, such as gene regulatory networks and signalling pathway networks. We formulate a novel control problem for Boolean network models under the asynchronous update mode in the context of cellular reprogramming. To facilitate scalability, we consider our previously introduced concept of a pseudo-attractor and we improve our procedure for effective identification of pseudo-attractor states. Finally, we devise a computational framework to solve the control problem. To leverage the structure of biological systems, we incorporate graph neural networks with graph convolutions into the artificial neural network approximator for the action-value function learned by the DRL agent. Experiments on a number of large real-world biological networks from literature demonstrate the scalability and effectiveness of our approach. 

---
# Automated Hybrid Reward Scheduling via Large Language Models for Robotic Skill Learning 

**Authors**: Changxin Huang, Junyang Liang, Yanbin Chang, Jingzhao Xu, Jianqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.02483)  

**Abstract**: Enabling a high-degree-of-freedom robot to learn specific skills is a challenging task due to the complexity of robotic dynamics. Reinforcement learning (RL) has emerged as a promising solution; however, addressing such problems requires the design of multiple reward functions to account for various constraints in robotic motion. Existing approaches typically sum all reward components indiscriminately to optimize the RL value function and policy. We argue that this uniform inclusion of all reward components in policy optimization is inefficient and limits the robot's learning performance. To address this, we propose an Automated Hybrid Reward Scheduling (AHRS) framework based on Large Language Models (LLMs). This paradigm dynamically adjusts the learning intensity of each reward component throughout the policy optimization process, enabling robots to acquire skills in a gradual and structured manner. Specifically, we design a multi-branch value network, where each branch corresponds to a distinct reward component. During policy optimization, each branch is assigned a weight that reflects its importance, and these weights are automatically computed based on rules designed by LLMs. The LLM generates a rule set in advance, derived from the task description, and during training, it selects a weight calculation rule from the library based on language prompts that evaluate the performance of each branch. Experimental results demonstrate that the AHRS method achieves an average 6.48% performance improvement across multiple high-degree-of-freedom robotic tasks. 

---
# Think on your Feet: Adaptive Thinking via Reinforcement Learning for Social Agents 

**Authors**: Minzheng Wang, Yongbin Li, Haobo Wang, Xinghua Zhang, Nan Xu, Bingli Wu, Fei Huang, Haiyang Yu, Wenji Mao  

**Link**: [PDF](https://arxiv.org/pdf/2505.02156)  

**Abstract**: Effective social intelligence simulation requires language agents to dynamically adjust reasoning depth, a capability notably absent in current approaches. While existing methods either lack this kind of reasoning capability or enforce uniform long chain-of-thought reasoning across all scenarios, resulting in excessive token usage and inappropriate social simulation. In this paper, we propose $\textbf{A}$daptive $\textbf{M}$ode $\textbf{L}$earning ($\textbf{AML}$) that strategically selects from four thinking modes (intuitive reaction $\rightarrow$ deep contemplation) based on real-time context. Our framework's core innovation, the $\textbf{A}$daptive $\textbf{M}$ode $\textbf{P}$olicy $\textbf{O}$ptimization ($\textbf{AMPO}$) algorithm, introduces three key advancements over existing methods: (1) Multi-granular thinking mode design, (2) Context-aware mode switching across social interaction, and (3) Token-efficient reasoning via depth-adaptive processing. Extensive experiments on social intelligence tasks confirm that AML achieves 15.6% higher task performance than state-of-the-art methods. Notably, our method outperforms GRPO by 7.0% with 32.8% shorter reasoning chains. These results demonstrate that context-sensitive thinking mode selection, as implemented in AMPO, enables more human-like adaptive reasoning than GRPO's fixed-depth approach 

---
# Open Challenges in Multi-Agent Security: Towards Secure Systems of Interacting AI Agents 

**Authors**: Christian Schroeder de Witt  

**Link**: [PDF](https://arxiv.org/pdf/2505.02077)  

**Abstract**: Decentralized AI agents will soon interact across internet platforms, creating security challenges beyond traditional cybersecurity and AI safety frameworks. Free-form protocols are essential for AI's task generalization but enable new threats like secret collusion and coordinated swarm attacks. Network effects can rapidly spread privacy breaches, disinformation, jailbreaks, and data poisoning, while multi-agent dispersion and stealth optimization help adversaries evade oversightcreating novel persistent threats at a systemic level. Despite their critical importance, these security challenges remain understudied, with research fragmented across disparate fields including AI security, multi-agent learning, complex systems, cybersecurity, game theory, distributed systems, and technical AI governance. We introduce \textbf{multi-agent security}, a new field dedicated to securing networks of decentralized AI agents against threats that emerge or amplify through their interactionswhether direct or indirect via shared environmentswith each other, humans, and institutions, and characterize fundamental security-performance trade-offs. Our preliminary work (1) taxonomizes the threat landscape arising from interacting AI agents, (2) surveys security-performance tradeoffs in decentralized AI systems, and (3) proposes a unified research agenda addressing open challenges in designing secure agent systems and interaction environments. By identifying these gaps, we aim to guide research in this critical area to unlock the socioeconomic potential of large-scale agent deployment on the internet, foster public trust, and mitigate national security risks in critical infrastructure and defense contexts. 

---
# A Goal-Oriented Reinforcement Learning-Based Path Planning Algorithm for Modular Self-Reconfigurable Satellites 

**Authors**: Bofei Liu, Dong Ye, Zunhao Yao, Zhaowei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.01966)  

**Abstract**: Modular self-reconfigurable satellites refer to satellite clusters composed of individual modular units capable of altering their configurations. The configuration changes enable the execution of diverse tasks and mission objectives. Existing path planning algorithms for reconfiguration often suffer from high computational complexity, poor generalization capability, and limited support for diverse target configurations. To address these challenges, this paper proposes a goal-oriented reinforcement learning-based path planning algorithm. This algorithm is the first to address the challenge that previous reinforcement learning methods failed to overcome, namely handling multiple target configurations. Moreover, techniques such as Hindsight Experience Replay and Invalid Action Masking are incorporated to overcome the significant obstacles posed by sparse rewards and invalid actions. Based on these designs, our model achieves a 95% and 73% success rate in reaching arbitrary target configurations in a modular satellite cluster composed of four and six units, respectively. 

---
# Semantic Intelligence: Integrating GPT-4 with A Planning in Low-Cost Robotics 

**Authors**: Jesse Barkley, Abraham George, Amir Barati Farimani  

**Link**: [PDF](https://arxiv.org/pdf/2505.01931)  

**Abstract**: Classical robot navigation often relies on hardcoded state machines and purely geometric path planners, limiting a robot's ability to interpret high-level semantic instructions. In this paper, we first assess GPT-4's ability to act as a path planner compared to the A* algorithm, then present a hybrid planning framework that integrates GPT-4's semantic reasoning with A* on a low-cost robot platform operating on ROS2 Humble. Our approach eliminates explicit finite state machine (FSM) coding by using prompt-based GPT-4 reasoning to handle task logic while maintaining the accurate paths computed by A*. The GPT-4 module provides semantic understanding of instructions and environmental cues (e.g., recognizing toxic obstacles or crowded areas to avoid, or understanding low-battery situations requiring alternate route selection), and dynamically adjusts the robot's occupancy grid via obstacle buffering to enforce semantic constraints. We demonstrate multi-step reasoning for sequential tasks, such as first navigating to a resource goal and then reaching a final destination safely. Experiments on a Petoi Bittle robot with an overhead camera and Raspberry Pi Zero 2W compare classical A* against GPT-4-assisted planning. Results show that while A* is faster and more accurate for basic route generation and obstacle avoidance, the GPT-4-integrated system achieves high success rates (96-100%) on semantic tasks that are infeasible for pure geometric planners. This work highlights how affordable robots can exhibit intelligent, context-aware behaviors by leveraging large language model reasoning with minimal hardware and no fine-tuning. 

---
# RoBridge: A Hierarchical Architecture Bridging Cognition and Execution for General Robotic Manipulation 

**Authors**: Kaidong Zhang, Rongtao Xu, Pengzhen Ren, Junfan Lin, Hefeng Wu, Liang Lin, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.01709)  

**Abstract**: Operating robots in open-ended scenarios with diverse tasks is a crucial research and application direction in robotics. While recent progress in natural language processing and large multimodal models has enhanced robots' ability to understand complex instructions, robot manipulation still faces the procedural skill dilemma and the declarative skill dilemma in open environments. Existing methods often compromise cognitive and executive capabilities. To address these challenges, in this paper, we propose RoBridge, a hierarchical intelligent architecture for general robotic manipulation. It consists of a high-level cognitive planner (HCP) based on a large-scale pre-trained vision-language model (VLM), an invariant operable representation (IOR) serving as a symbolic bridge, and a generalist embodied agent (GEA). RoBridge maintains the declarative skill of VLM and unleashes the procedural skill of reinforcement learning, effectively bridging the gap between cognition and execution. RoBridge demonstrates significant performance improvements over existing baselines, achieving a 75% success rate on new tasks and an 83% average success rate in sim-to-real generalization using only five real-world data samples per task. This work represents a significant step towards integrating cognitive reasoning with physical execution in robotic systems, offering a new paradigm for general robotic manipulation. 

---
# PIPA: A Unified Evaluation Protocol for Diagnosing Interactive Planning Agents 

**Authors**: Takyoung Kim, Janvijay Singh, Shuhaib Mehri, Emre Can Acikgoz, Sagnik Mukherjee, Nimet Beyza Bozdag, Sumuk Shashidhar, Gokhan Tur, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2505.01592)  

**Abstract**: The growing capabilities of large language models (LLMs) in instruction-following and context-understanding lead to the era of agents with numerous applications. Among these, task planning agents have become especially prominent in realistic scenarios involving complex internal pipelines, such as context understanding, tool management, and response generation. However, existing benchmarks predominantly evaluate agent performance based on task completion as a proxy for overall effectiveness. We hypothesize that merely improving task completion is misaligned with maximizing user satisfaction, as users interact with the entire agentic process and not only the end result. To address this gap, we propose PIPA, a unified evaluation protocol that conceptualizes the behavioral process of interactive task planning agents within a partially observable Markov Decision Process (POMDP) paradigm. The proposed protocol offers a comprehensive assessment of agent performance through a set of atomic evaluation criteria, allowing researchers and practitioners to diagnose specific strengths and weaknesses within the agent's decision-making pipeline. Our analyses show that agents excel in different behavioral stages, with user satisfaction shaped by both outcomes and intermediate behaviors. We also highlight future directions, including systems that leverage multiple agents and the limitations of user simulators in task planning. 

---
# Skill-based Safe Reinforcement Learning with Risk Planning 

**Authors**: Hanping Zhang, Yuhong Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.01619)  

**Abstract**: Safe Reinforcement Learning (Safe RL) aims to ensure safety when an RL agent conducts learning by interacting with real-world environments where improper actions can induce high costs or lead to severe consequences. In this paper, we propose a novel Safe Skill Planning (SSkP) approach to enhance effective safe RL by exploiting auxiliary offline demonstration data. SSkP involves a two-stage process. First, we employ PU learning to learn a skill risk predictor from the offline demonstration data. Then, based on the learned skill risk predictor, we develop a novel risk planning process to enhance online safe RL and learn a risk-averse safe policy efficiently through interactions with the online RL environment, while simultaneously adapting the skill risk predictor to the environment. We conduct experiments in several benchmark robotic simulation environments. The experimental results demonstrate that the proposed approach consistently outperforms previous state-of-the-art safe RL methods. 

---
# Interactive Double Deep Q-network: Integrating Human Interventions and Evaluative Predictions in Reinforcement Learning of Autonomous Driving 

**Authors**: Alkis Sygkounas, Ioannis Athanasiadis, Andreas Persson, Michael Felsberg, Amy Loutfi  

**Link**: [PDF](https://arxiv.org/pdf/2505.01440)  

**Abstract**: Integrating human expertise with machine learning is crucial for applications demanding high accuracy and safety, such as autonomous driving. This study introduces Interactive Double Deep Q-network (iDDQN), a Human-in-the-Loop (HITL) approach that enhances Reinforcement Learning (RL) by merging human insights directly into the RL training process, improving model performance. Our proposed iDDQN method modifies the Q-value update equation to integrate human and agent actions, establishing a collaborative approach for policy development. Additionally, we present an offline evaluative framework that simulates the agent's trajectory as if no human intervention had occurred, to assess the effectiveness of human interventions. Empirical results in simulated autonomous driving scenarios demonstrate that iDDQN outperforms established approaches, including Behavioral Cloning (BC), HG-DAgger, Deep Q-Learning from Demonstrations (DQfD), and vanilla DRL in leveraging human expertise for improving performance and adaptability. 

---
# CAMOUFLAGE: Exploiting Misinformation Detection Systems Through LLM-driven Adversarial Claim Transformation 

**Authors**: Mazal Bethany, Nishant Vishwamitra, Cho-Yu Jason Chiang, Peyman Najafirad  

**Link**: [PDF](https://arxiv.org/pdf/2505.01900)  

**Abstract**: Automated evidence-based misinformation detection systems, which evaluate the veracity of short claims against evidence, lack comprehensive analysis of their adversarial vulnerabilities. Existing black-box text-based adversarial attacks are ill-suited for evidence-based misinformation detection systems, as these attacks primarily focus on token-level substitutions involving gradient or logit-based optimization strategies, which are incapable of fooling the multi-component nature of these detection systems. These systems incorporate both retrieval and claim-evidence comparison modules, which requires attacks to break the retrieval of evidence and/or the comparison module so that it draws incorrect inferences. We present CAMOUFLAGE, an iterative, LLM-driven approach that employs a two-agent system, a Prompt Optimization Agent and an Attacker Agent, to create adversarial claim rewritings that manipulate evidence retrieval and mislead claim-evidence comparison, effectively bypassing the system without altering the meaning of the claim. The Attacker Agent produces semantically equivalent rewrites that attempt to mislead detectors, while the Prompt Optimization Agent analyzes failed attack attempts and refines the prompt of the Attacker to guide subsequent rewrites. This enables larger structural and stylistic transformations of the text rather than token-level substitutions, adapting the magnitude of changes based on previous outcomes. Unlike existing approaches, CAMOUFLAGE optimizes its attack solely based on binary model decisions to guide its rewriting process, eliminating the need for classifier logits or extensive querying. We evaluate CAMOUFLAGE on four systems, including two recent academic systems and two real-world APIs, with an average attack success rate of 46.92\% while preserving textual coherence and semantic equivalence to the original claims. 

---
# SymPlanner: Deliberate Planning in Language Models with Symbolic Representation 

**Authors**: Siheng Xiong, Jieyu Zhou, Zhangding Liu, Yusen Su  

**Link**: [PDF](https://arxiv.org/pdf/2505.01479)  

**Abstract**: Planning remains a core challenge for language models (LMs), particularly in domains that require coherent multi-step action sequences grounded in external constraints. We introduce SymPlanner, a novel framework that equips LMs with structured planning capabilities by interfacing them with a symbolic environment that serves as an explicit world model. Rather than relying purely on natural language reasoning, SymPlanner grounds the planning process in a symbolic state space, where a policy model proposes actions and a symbolic environment deterministically executes and verifies their effects. To enhance exploration and improve robustness, we introduce Iterative Correction (IC), which refines previously proposed actions by leveraging feedback from the symbolic environment to eliminate invalid decisions and guide the model toward valid alternatives. Additionally, Contrastive Ranking (CR) enables fine-grained comparison of candidate plans by evaluating them jointly. We evaluate SymPlanner on PlanBench, demonstrating that it produces more coherent, diverse, and verifiable plans than pure natural language baselines. 

---
