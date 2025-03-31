# ActionStudio: A Lightweight Framework for Data and Training of Action Models 

**Authors**: Jianguo Zhang, Thai Hoang, Ming Zhu, Zuxin Liu, Shiyu Wang, Tulika Awalgaonkar, Akshara Prabhakar, Haolin Chen, Weiran Yao, Zhiwei Liu, Juntao Tan, Juan Carlos Niebles, Shelby Heinecke, Huan Wang, Silvio Savarese, Caiming Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2503.22673)  

**Abstract**: Action models are essential for enabling autonomous agents to perform complex tasks. However, training large action models remains challenging due to the diversity of agent environments and the complexity of agentic data. Despite growing interest, existing infrastructure provides limited support for scalable, agent-specific fine-tuning. We present ActionStudio, a lightweight and extensible data and training framework designed for action models. ActionStudio unifies heterogeneous agent trajectories through a standardized format, supports diverse training paradigms including LoRA, full fine-tuning, and distributed setups, and integrates robust preprocessing and verification tools. We validate its effectiveness across both public and realistic industry benchmarks, demonstrating strong performance and practical scalability. We open-sourced code and data at this https URL to facilitate research in the community. 

---
# Agent-Centric Personalized Multiple Clustering with Multi-Modal LLMs 

**Authors**: Ziye Chen, Yiqun Duan, Riheng Zhu, Zhenbang Sun, Mingming Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.22241)  

**Abstract**: Personalized multiple clustering aims to generate diverse partitions of a dataset based on different user-specific aspects, rather than a single clustering. It has recently drawn research interest for accommodating varying user preferences. Recent approaches primarily use CLIP embeddings with proxy learning to extract representations biased toward user clustering preferences. However, CLIP primarily focuses on coarse image-text alignment, lacking a deep contextual understanding of user interests. To overcome these limitations, we propose an agent-centric personalized clustering framework that leverages multi-modal large language models (MLLMs) as agents to comprehensively traverse a relational graph to search for clusters based on user interests. Due to the advanced reasoning mechanism of MLLMs, the obtained clusters align more closely with user-defined criteria than those obtained from CLIP-based representations. To reduce computational overhead, we shorten the agents' traversal path by constructing a relational graph using user-interest-biased embeddings extracted by MLLMs. A large number of weakly connected edges can be filtered out based on embedding similarity, facilitating an efficient traversal search for agents. Experimental results show that the proposed method achieves NMI scores of 0.9667 and 0.9481 on the Card Order and Card Suits benchmarks, respectively, largely improving the SOTA model by over 140%. 

---
# On the Mistaken Assumption of Interchangeable Deep Reinforcement Learning Implementations 

**Authors**: Rajdeep Singh Hundal, Yan Xiao, Xiaochun Cao, Jin Song Dong, Manuel Rigger  

**Link**: [PDF](https://arxiv.org/pdf/2503.22575)  

**Abstract**: Deep Reinforcement Learning (DRL) is a paradigm of artificial intelligence where an agent uses a neural network to learn which actions to take in a given environment. DRL has recently gained traction from being able to solve complex environments like driving simulators, 3D robotic control, and multiplayer-online-battle-arena video games. Numerous implementations of the state-of-the-art algorithms responsible for training these agents, like the Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) algorithms, currently exist. However, studies make the mistake of assuming implementations of the same algorithm to be consistent and thus, interchangeable. In this paper, through a differential testing lens, we present the results of studying the extent of implementation inconsistencies, their effect on the implementations' performance, as well as their impact on the conclusions of prior studies under the assumption of interchangeable implementations. The outcomes of our differential tests showed significant discrepancies between the tested algorithm implementations, indicating that they are not interchangeable. In particular, out of the five PPO implementations tested on 56 games, three implementations achieved superhuman performance for 50% of their total trials while the other two implementations only achieved superhuman performance for less than 15% of their total trials. As part of a meticulous manual analysis of the implementations' source code, we analyzed implementation discrepancies and determined that code-level inconsistencies primarily caused these discrepancies. Lastly, we replicated a study and showed that this assumption of implementation interchangeability was sufficient to flip experiment outcomes. Therefore, this calls for a shift in how implementations are being used. 

---
# PharmAgents: Building a Virtual Pharma with Large Language Model Agents 

**Authors**: Bowen Gao, Yanwen Huang, Yiqiao Liu, Wenxuan Xie, Wei-Ying Ma, Ya-Qin Zhang, Yanyan Lan  

**Link**: [PDF](https://arxiv.org/pdf/2503.22164)  

**Abstract**: The discovery of novel small molecule drugs remains a critical scientific challenge with far-reaching implications for treating diseases and advancing human health. Traditional drug development--especially for small molecule therapeutics--is a highly complex, resource-intensive, and time-consuming process that requires multidisciplinary collaboration. Recent breakthroughs in artificial intelligence (AI), particularly the rise of large language models (LLMs), present a transformative opportunity to streamline and accelerate this process. In this paper, we introduce PharmAgents, a virtual pharmaceutical ecosystem driven by LLM-based multi-agent collaboration. PharmAgents simulates the full drug discovery workflow--from target discovery to preclinical evaluation--by integrating explainable, LLM-driven agents equipped with specialized machine learning models and computational tools. Through structured knowledge exchange and automated optimization, PharmAgents identifies potential therapeutic targets, discovers promising lead compounds, enhances binding affinity and key molecular properties, and performs in silico analyses of toxicity and synthetic feasibility. Additionally, the system supports interpretability, agent interaction, and self-evolvement, enabling it to refine future drug designs based on prior experience. By showcasing the potential of LLM-powered multi-agent systems in drug discovery, this work establishes a new paradigm for autonomous, explainable, and scalable pharmaceutical research, with future extensions toward comprehensive drug lifecycle management. 

---
# REMAC: Self-Reflective and Self-Evolving Multi-Agent Collaboration for Long-Horizon Robot Manipulation 

**Authors**: Puzhen Yuan, Angyuan Ma, Yunchao Yao, Huaxiu Yao, Masayoshi Tomizuka, Mingyu Ding  

**Link**: [PDF](https://arxiv.org/pdf/2503.22122)  

**Abstract**: Vision-language models (VLMs) have demonstrated remarkable capabilities in robotic planning, particularly for long-horizon tasks that require a holistic understanding of the environment for task decomposition. Existing methods typically rely on prior environmental knowledge or carefully designed task-specific prompts, making them struggle with dynamic scene changes or unexpected task conditions, e.g., a robot attempting to put a carrot in the microwave but finds the door was closed. Such challenges underscore two critical issues: adaptability and efficiency. To address them, in this work, we propose an adaptive multi-agent planning framework, termed REMAC, that enables efficient, scene-agnostic multi-robot long-horizon task planning and execution through continuous reflection and self-evolution. REMAC incorporates two key modules: a self-reflection module performing pre-condition and post-condition checks in the loop to evaluate progress and refine plans, and a self-evolvement module dynamically adapting plans based on scene-specific reasoning. It offers several appealing benefits: 1) Robots can initially explore and reason about the environment without complex prompt design. 2) Robots can keep reflecting on potential planning errors and adapting the plan based on task-specific insights. 3) After iterations, a robot can call another one to coordinate tasks in parallel, maximizing the task execution efficiency. To validate REMAC's effectiveness, we build a multi-agent environment for long-horizon robot manipulation and navigation based on RoboCasa, featuring 4 task categories with 27 task styles and 50+ different objects. Based on it, we further benchmark state-of-the-art reasoning models, including DeepSeek-R1, o3-mini, QwQ, and Grok3, demonstrating REMAC's superiority by boosting average success rates by 40% and execution efficiency by 52.7% over the single robot baseline. 

---
# Data-Agnostic Robotic Long-Horizon Manipulation with Vision-Language-Guided Closed-Loop Feedback 

**Authors**: Yuan Meng, Xiangtong Yao, Haihui Ye, Yirui Zhou, Shengqiang Zhang, Zhenshan Bing, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2503.21969)  

**Abstract**: Recent advances in language-conditioned robotic manipulation have leveraged imitation and reinforcement learning to enable robots to execute tasks from human commands. However, these methods often suffer from limited generalization, adaptability, and the lack of large-scale specialized datasets, unlike data-rich domains such as computer vision, making long-horizon task execution challenging. To address these gaps, we introduce DAHLIA, a data-agnostic framework for language-conditioned long-horizon robotic manipulation, leveraging large language models (LLMs) for real-time task planning and execution. DAHLIA employs a dual-tunnel architecture, where an LLM-powered planner collaborates with co-planners to decompose tasks and generate executable plans, while a reporter LLM provides closed-loop feedback, enabling adaptive re-planning and ensuring task recovery from potential failures. Moreover, DAHLIA integrates chain-of-thought (CoT) in task reasoning and temporal abstraction for efficient action execution, enhancing traceability and robustness. Our framework demonstrates state-of-the-art performance across diverse long-horizon tasks, achieving strong generalization in both simulated and real-world scenarios. Videos and code are available at this https URL. 

---
# LERO: LLM-driven Evolutionary framework with Hybrid Rewards and Enhanced Observation for Multi-Agent Reinforcement Learning 

**Authors**: Yuan Wei, Xiaohan Shan, Jianmin Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.21807)  

**Abstract**: Multi-agent reinforcement learning (MARL) faces two critical bottlenecks distinct from single-agent RL: credit assignment in cooperative tasks and partial observability of environmental states. We propose LERO, a framework integrating Large language models (LLMs) with evolutionary optimization to address these MARL-specific challenges. The solution centers on two LLM-generated components: a hybrid reward function that dynamically allocates individual credit through reward decomposition, and an observation enhancement function that augments partial observations with inferred environmental context. An evolutionary algorithm optimizes these components through iterative MARL training cycles, where top-performing candidates guide subsequent LLM generations. Evaluations in Multi-Agent Particle Environments (MPE) demonstrate LERO's superiority over baseline methods, with improved task performance and training efficiency. 

---
# Self-Evolving Multi-Agent Simulations for Realistic Clinical Interactions 

**Authors**: Mohammad Almansoori, Komal Kumar, Hisham Cholakkal  

**Link**: [PDF](https://arxiv.org/pdf/2503.22678)  

**Abstract**: In this work, we introduce MedAgentSim, an open-source simulated clinical environment with doctor, patient, and measurement agents designed to evaluate and enhance LLM performance in dynamic diagnostic settings. Unlike prior approaches, our framework requires doctor agents to actively engage with patients through multi-turn conversations, requesting relevant medical examinations (e.g., temperature, blood pressure, ECG) and imaging results (e.g., MRI, X-ray) from a measurement agent to mimic the real-world diagnostic process. Additionally, we incorporate self improvement mechanisms that allow models to iteratively refine their diagnostic strategies. We enhance LLM performance in our simulated setting by integrating multi-agent discussions, chain-of-thought reasoning, and experience-based knowledge retrieval, facilitating progressive learning as doctor agents interact with more patients. We also introduce an evaluation benchmark for assessing the LLM's ability to engage in dynamic, context-aware diagnostic interactions. While MedAgentSim is fully automated, it also supports a user-controlled mode, enabling human interaction with either the doctor or patient agent. Comprehensive evaluations in various simulated diagnostic scenarios demonstrate the effectiveness of our approach. Our code, simulation tool, and benchmark are available at \href{this https URL}. 

---
# Scaling Laws of Scientific Discovery with AI and Robot Scientists 

**Authors**: Pengsong Zhang, Heng Zhang, Huazhe Xu, Renjun Xu, Zhenting Wang, Cong Wang, Animesh Garg, Zhibin Li, Arash Ajoudani, Xinyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.22444)  

**Abstract**: The rapid evolution of scientific inquiry highlights an urgent need for groundbreaking methodologies that transcend the limitations of traditional research. Conventional approaches, bogged down by manual processes and siloed expertise, struggle to keep pace with the demands of modern discovery. We envision an autonomous generalist scientist (AGS) system-a fusion of agentic AI and embodied robotics-that redefines the research lifecycle. This system promises to autonomously navigate physical and digital realms, weaving together insights from disparate disciplines with unprecedented efficiency. By embedding advanced AI and robot technologies into every phase-from hypothesis formulation to peer-ready manuscripts-AGS could slash the time and resources needed for scientific research in diverse field. We foresee a future where scientific discovery follows new scaling laws, driven by the proliferation and sophistication of such systems. As these autonomous agents and robots adapt to extreme environments and leverage a growing reservoir of knowledge, they could spark a paradigm shift, pushing the boundaries of what's possible and ushering in an era of relentless innovation. 

---
# WorkTeam: Constructing Workflows from Natural Language with Multi-Agents 

**Authors**: Hanchao Liu, Rongjun Li, Weimin Xiong, Ziyu Zhou, Wei Peng  

**Link**: [PDF](https://arxiv.org/pdf/2503.22473)  

**Abstract**: Workflows play a crucial role in enhancing enterprise efficiency by orchestrating complex processes with multiple tools or components. However, hand-crafted workflow construction requires expert knowledge, presenting significant technical barriers. Recent advancements in Large Language Models (LLMs) have improved the generation of workflows from natural language instructions (aka NL2Workflow), yet existing single LLM agent-based methods face performance degradation on complex tasks due to the need for specialized knowledge and the strain of task-switching. To tackle these challenges, we propose WorkTeam, a multi-agent NL2Workflow framework comprising a supervisor, orchestrator, and filler agent, each with distinct roles that collaboratively enhance the conversion process. As there are currently no publicly available NL2Workflow benchmarks, we also introduce the HW-NL2Workflow dataset, which includes 3,695 real-world business samples for training and evaluation. Experimental results show that our approach significantly increases the success rate of workflow construction, providing a novel and effective solution for enterprise NL2Workflow services. 

---
# Debate-Driven Multi-Agent LLMs for Phishing Email Detection 

**Authors**: Ngoc Tuong Vy Nguyen, Felix D Childress, Yunting Yin  

**Link**: [PDF](https://arxiv.org/pdf/2503.22038)  

**Abstract**: Phishing attacks remain a critical cybersecurity threat. Attackers constantly refine their methods, making phishing emails harder to detect. Traditional detection methods, including rule-based systems and supervised machine learning models, either rely on predefined patterns like blacklists, which can be bypassed with slight modifications, or require large datasets for training and still can generate false positives and false negatives. In this work, we propose a multi-agent large language model (LLM) prompting technique that simulates debates among agents to detect whether the content presented on an email is phishing. Our approach uses two LLM agents to present arguments for or against the classification task, with a judge agent adjudicating the final verdict based on the quality of reasoning provided. This debate mechanism enables the models to critically analyze contextual cue and deceptive patterns in text, which leads to improved classification accuracy. The proposed framework is evaluated on multiple phishing email datasets and demonstrate that mixed-agent configurations consistently outperform homogeneous configurations. Results also show that the debate structure itself is sufficient to yield accurate decisions without extra prompting strategies. 

---
