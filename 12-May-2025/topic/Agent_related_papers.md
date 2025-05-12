# Neuro-Symbolic Concepts 

**Authors**: Jiayuan Mao, Joshua B. Tenenbaum, Jiajun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06191)  

**Abstract**: This article presents a concept-centric paradigm for building agents that can learn continually and reason flexibly. The concept-centric agent utilizes a vocabulary of neuro-symbolic concepts. These concepts, such as object, relation, and action concepts, are grounded on sensory inputs and actuation outputs. They are also compositional, allowing for the creation of novel concepts through their structural combination. To facilitate learning and reasoning, the concepts are typed and represented using a combination of symbolic programs and neural network representations. Leveraging such neuro-symbolic concepts, the agent can efficiently learn and recombine them to solve various tasks across different domains, ranging from 2D images, videos, 3D scenes, and robotic manipulation tasks. This concept-centric framework offers several advantages, including data efficiency, compositional generalization, continual learning, and zero-shot transfer. 

---
# Prompted Meta-Learning for Few-shot Knowledge Graph Completion 

**Authors**: Han Wu, Jie Yin  

**Link**: [PDF](https://arxiv.org/pdf/2505.05684)  

**Abstract**: Few-shot knowledge graph completion (KGC) has obtained significant attention due to its practical applications in real-world scenarios, where new knowledge often emerges with limited available data. While most existing methods for few-shot KGC have predominantly focused on leveraging relational information, rich semantics inherent in KGs have been largely overlooked. To address this gap, we propose a novel prompted meta-learning (PromptMeta) framework that seamlessly integrates meta-semantics with relational information for few-shot KGC. PrompMeta has two key innovations: (1) a meta-semantic prompt pool that captures and consolidates high-level meta-semantics, enabling effective knowledge transfer and adaptation to rare and newly emerging relations. (2) a learnable fusion prompt that dynamically combines meta-semantic information with task-specific relational information tailored to different few-shot tasks. Both components are optimized together with model parameters within a meta-learning framework. Extensive experiments on two benchmark datasets demonstrate the effectiveness of our approach. 

---
# UniVLA: Learning to Act Anywhere with Task-centric Latent Actions 

**Authors**: Qingwen Bu, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao, Ping Luo, Hongyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.06111)  

**Abstract**: A generalist robot should perform effectively across various environments. However, most existing approaches heavily rely on scaling action-annotated data to enhance their capabilities. Consequently, they are often limited to single physical specification and struggle to learn transferable knowledge across different embodiments and environments. To confront these limitations, we propose UniVLA, a new framework for learning cross-embodiment vision-language-action (VLA) policies. Our key innovation is to derive task-centric action representations from videos with a latent action model. This enables us to exploit extensive data across a wide spectrum of embodiments and perspectives. To mitigate the effect of task-irrelevant dynamics, we incorporate language instructions and establish a latent action model within the DINO feature space. Learned from internet-scale videos, the generalist policy can be deployed to various robots through efficient latent action decoding. We obtain state-of-the-art results across multiple manipulation and navigation benchmarks, as well as real-robot deployments. UniVLA achieves superior performance over OpenVLA with less than 1/20 of pretraining compute and 1/10 of downstream data. Continuous performance improvements are observed as heterogeneous data, even including human videos, are incorporated into the training pipeline. The results underscore UniVLA's potential to facilitate scalable and efficient robot policy learning. 

---
# HiBayES: A Hierarchical Bayesian Modeling Framework for AI Evaluation Statistics 

**Authors**: Lennart Luettgau, Harry Coppock, Magda Dubois, Christopher Summerfield, Cozmin Ududec  

**Link**: [PDF](https://arxiv.org/pdf/2505.05602)  

**Abstract**: As Large Language Models (LLMs) and other AI systems evolve, robustly estimating their capabilities from inherently stochastic outputs while systematically quantifying uncertainty in these estimates becomes increasingly important. Further, advanced AI evaluations often have a nested hierarchical structure, exhibit high levels of complexity, and come with high costs in testing the most advanced AI systems. To address these challenges, we introduce HiBayES, a generalizable Hierarchical Bayesian modeling framework for AI Evaluation Statistics. HiBayES supports robust inferences in classical question-answer benchmarks and advanced agentic evaluations, particularly in low-data scenarios (e.g., < 20 data points per evaluation). Built on Generalized Linear Models (GLMs), Bayesian data analysis, and formal model comparison, HiBayES provides principled uncertainty quantification and robust parameter estimation. This paper offers a comprehensive introduction to HiBayES, including illustrative examples, comparisons to conventional statistical methods, and practical guidance for implementing multilevel Bayesian GLMs. Additionally, we provide a HiBayES software package [4] (Beta version) for out-of-the-box implementation. 

---
# Let Humanoids Hike! Integrative Skill Development on Complex Trails 

**Authors**: Kwan-Yee Lin, Stella X.Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06218)  

**Abstract**: Hiking on complex trails demands balance, agility, and adaptive decision-making over unpredictable terrain. Current humanoid research remains fragmented and inadequate for hiking: locomotion focuses on motor skills without long-term goals or situational awareness, while semantic navigation overlooks real-world embodiment and local terrain variability. We propose training humanoids to hike on complex trails, driving integrative skill development across visual perception, decision making, and motor execution. We develop a learning framework, LEGO-H, that enables a vision-equipped humanoid robot to hike complex trails autonomously. We introduce two technical innovations: 1) A temporal vision transformer variant - tailored into Hierarchical Reinforcement Learning framework - anticipates future local goals to guide movement, seamlessly integrating locomotion with goal-directed navigation. 2) Latent representations of joint movement patterns, combined with hierarchical metric learning - enhance Privileged Learning scheme - enable smooth policy transfer from privileged training to onboard execution. These components allow LEGO-H to handle diverse physical and environmental challenges without relying on predefined motion patterns. Experiments across varied simulated trails and robot morphologies highlight LEGO-H's versatility and robustness, positioning hiking as a compelling testbed for embodied autonomy and LEGO-H as a baseline for future humanoid development. 

---
# APOLLO: Automated LLM and Lean Collaboration for Advanced Formal Reasoning 

**Authors**: Azim Ospanov, Roozbeh Yousefzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2505.05758)  

**Abstract**: Formal reasoning and automated theorem proving constitute a challenging subfield of machine learning, in which machines are tasked with proving mathematical theorems using formal languages like Lean. A formal verification system can check whether a formal proof is correct or not almost instantaneously, but generating a completely correct formal proof with large language models (LLMs) remains a formidable task. The usual approach in the literature is to prompt the LLM many times (up to several thousands) until one of the generated proofs passes the verification system. In this work, we present APOLLO (Automated PrOof repair via LLM and Lean cOllaboration), a modular, model-agnostic pipeline that combines the strengths of the Lean compiler with an LLM's reasoning abilities to achieve better proof-generation results at a low sampling budget. Apollo directs a fully automated process in which the LLM generates proofs for theorems, a set of agents analyze the proofs, fix the syntax errors, identify the mistakes in the proofs using Lean, isolate failing sub-lemmas, utilize automated solvers, and invoke an LLM on each remaining goal with a low top-K budget. The repaired sub-proofs are recombined and reverified, iterating up to a user-controlled maximum number of attempts. On the miniF2F benchmark, we establish a new state-of-the-art accuracy of 75.0% among 7B-parameter models while keeping the sampling budget below one thousand. Moreover, Apollo raises the state-of-the-art accuracy for Goedel-Prover-SFT to 65.6% while cutting sample complexity from 25,600 to a few hundred. General-purpose models (o3-mini, o4-mini) jump from 3-7% to over 40% accuracy. Our results demonstrate that targeted, compiler-guided repair of LLM outputs yields dramatic gains in both efficiency and correctness, suggesting a general paradigm for scalable automated theorem proving. 

---
# AgentXploit: End-to-End Redteaming of Black-Box AI Agents 

**Authors**: Zhun Wang, Vincent Siu, Zhe Ye, Tianneng Shi, Yuzhou Nie, Xuandong Zhao, Chenguang Wang, Wenbo Guo, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.05849)  

**Abstract**: The strong planning and reasoning capabilities of Large Language Models (LLMs) have fostered the development of agent-based systems capable of leveraging external tools and interacting with increasingly complex environments. However, these powerful features also introduce a critical security risk: indirect prompt injection, a sophisticated attack vector that compromises the core of these agents, the LLM, by manipulating contextual information rather than direct user prompts. In this work, we propose a generic black-box fuzzing framework, AgentXploit, designed to automatically discover and exploit indirect prompt injection vulnerabilities across diverse LLM agents. Our approach starts by constructing a high-quality initial seed corpus, then employs a seed selection algorithm based on Monte Carlo Tree Search (MCTS) to iteratively refine inputs, thereby maximizing the likelihood of uncovering agent weaknesses. We evaluate AgentXploit on two public benchmarks, AgentDojo and VWA-adv, where it achieves 71% and 70% success rates against agents based on o3-mini and GPT-4o, respectively, nearly doubling the performance of baseline attacks. Moreover, AgentXploit exhibits strong transferability across unseen tasks and internal LLMs, as well as promising results against defenses. Beyond benchmark evaluations, we apply our attacks in real-world environments, successfully misleading agents to navigate to arbitrary URLs, including malicious sites. 

---
# Multi-Agent Systems for Robotic Autonomy with LLMs 

**Authors**: Junhong Chen, Ziqi Yang, Haoyuan G Xu, Dandan Zhang, George Mylonas  

**Link**: [PDF](https://arxiv.org/pdf/2505.05762)  

**Abstract**: Since the advent of Large Language Models (LLMs), various research based on such models have maintained significant academic attention and impact, especially in AI and robotics. In this paper, we propose a multi-agent framework with LLMs to construct an integrated system for robotic task analysis, mechanical design, and path generation. The framework includes three core agents: Task Analyst, Robot Designer, and Reinforcement Learning Designer. Outputs are formatted as multimodal results, such as code files or technical reports, for stronger understandability and usability. To evaluate generalizability comparatively, we conducted experiments with models from both GPT and DeepSeek. Results demonstrate that the proposed system can design feasible robots with control strategies when appropriate task inputs are provided, exhibiting substantial potential for enhancing the efficiency and accessibility of robotic system development in research and industrial applications. 

---
# Towards Embodiment Scaling Laws in Robot Locomotion 

**Authors**: Bo Ai, Liu Dai, Nico Bohlinger, Dichen Li, Tongzhou Mu, Zhanxin Wu, K. Fay, Henrik I. Christensen, Jan Peters, Hao Su  

**Link**: [PDF](https://arxiv.org/pdf/2505.05753)  

**Abstract**: Developing generalist agents that can operate across diverse tasks, environments, and physical embodiments is a grand challenge in robotics and artificial intelligence. In this work, we focus on the axis of embodiment and investigate embodiment scaling laws$\unicode{x2013}$the hypothesis that increasing the number of training embodiments improves generalization to unseen ones. Using robot locomotion as a test bed, we procedurally generate a dataset of $\sim$1,000 varied embodiments, spanning humanoids, quadrupeds, and hexapods, and train generalist policies capable of handling diverse observation and action spaces on random subsets. We find that increasing the number of training embodiments improves generalization to unseen ones, and scaling embodiments is more effective in enabling embodiment-level generalization than scaling data on small, fixed sets of embodiments. Notably, our best policy, trained on the full dataset, zero-shot transfers to novel embodiments in the real world, such as Unitree Go2 and H1. These results represent a step toward general embodied intelligence, with potential relevance to adaptive control for configurable robots, co-design of morphology and control, and beyond. 

---
# CityNavAgent: Aerial Vision-and-Language Navigation with Hierarchical Semantic Planning and Global Memory 

**Authors**: Weichen Zhang, Chen Gao, Shiquan Yu, Ruiying Peng, Baining Zhao, Qian Zhang, Jinqiang Cui, Xinlei Chen, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.05622)  

**Abstract**: Aerial vision-and-language navigation (VLN), requiring drones to interpret natural language instructions and navigate complex urban environments, emerges as a critical embodied AI challenge that bridges human-robot interaction, 3D spatial reasoning, and real-world deployment. Although existing ground VLN agents achieved notable results in indoor and outdoor settings, they struggle in aerial VLN due to the absence of predefined navigation graphs and the exponentially expanding action space in long-horizon exploration. In this work, we propose \textbf{CityNavAgent}, a large language model (LLM)-empowered agent that significantly reduces the navigation complexity for urban aerial VLN. Specifically, we design a hierarchical semantic planning module (HSPM) that decomposes the long-horizon task into sub-goals with different semantic levels. The agent reaches the target progressively by achieving sub-goals with different capacities of the LLM. Additionally, a global memory module storing historical trajectories into a topological graph is developed to simplify navigation for visited targets. Extensive benchmark experiments show that our method achieves state-of-the-art performance with significant improvement. Further experiments demonstrate the effectiveness of different modules of CityNavAgent for aerial VLN in continuous city environments. The code is available at \href{this https URL}{link}. 

---
# CLAM: Continuous Latent Action Models for Robot Learning from Unlabeled Demonstrations 

**Authors**: Anthony Liang, Pavel Czempin, Matthew Hong, Yutai Zhou, Erdem Biyik, Stephen Tu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04999)  

**Abstract**: Learning robot policies using imitation learning requires collecting large amounts of costly action-labeled expert demonstrations, which fundamentally limits the scale of training data. A promising approach to address this bottleneck is to harness the abundance of unlabeled observations-e.g., from video demonstrations-to learn latent action labels in an unsupervised way. However, we find that existing methods struggle when applied to complex robot tasks requiring fine-grained motions. We design continuous latent action models (CLAM) which incorporate two key ingredients we find necessary for learning to solve complex continuous control tasks from unlabeled observation data: (a) using continuous latent action labels instead of discrete representations, and (b) jointly training an action decoder to ensure that the latent action space can be easily grounded to real actions with relatively few labeled examples. Importantly, the labeled examples can be collected from non-optimal play data, enabling CLAM to learn performant policies without access to any action-labeled expert data. We demonstrate on continuous control benchmarks in DMControl (locomotion) and MetaWorld (manipulation), as well as on a real WidowX robot arm that CLAM significantly outperforms prior state-of-the-art methods, remarkably with a 2-3x improvement in task success rate compared to the best baseline. Videos and code can be found at this http URL. 

---
