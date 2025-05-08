# The Power of Stories: Narrative Priming Shapes How LLM Agents Collaborate and Compete 

**Authors**: Gerrit Großmann, Larisa Ivanova, Sai Leela Poduru, Mohaddeseh Tabrizian, Islam Mesabah, David A. Selby, Sebastian J. Vollmer  

**Link**: [PDF](https://arxiv.org/pdf/2505.03961)  

**Abstract**: According to Yuval Noah Harari, large-scale human cooperation is driven by shared narratives that encode common beliefs and values. This study explores whether such narratives can similarly nudge LLM agents toward collaboration. We use a finitely repeated public goods game in which LLM agents choose either cooperative or egoistic spending strategies. We prime agents with stories highlighting teamwork to different degrees and test how this influences negotiation outcomes. Our experiments explore four questions:(1) How do narratives influence negotiation behavior? (2) What differs when agents share the same story versus different ones? (3) What happens when the agent numbers grow? (4) Are agents resilient against self-serving negotiators? We find that story-based priming significantly affects negotiation strategies and success rates. Common stories improve collaboration, benefiting each agent. By contrast, priming agents with different stories reverses this effect, and those agents primed toward self-interest prevail. We hypothesize that these results carry implications for multi-agent system design and AI alignment. 

---
# Beyond Theorem Proving: Formulation, Framework and Benchmark for Formal Problem-Solving 

**Authors**: Qi Liu, Xinhao Zheng, Renqiu Xia, Xingzhi Qi, Qinxiang Cao, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.04528)  

**Abstract**: As a seemingly self-explanatory task, problem-solving has been a significant component of science and engineering. However, a general yet concrete formulation of problem-solving itself is missing. With the recent development of AI-based problem-solving agents, the demand for process-level verifiability is rapidly increasing yet underexplored. To fill these gaps, we present a principled formulation of problem-solving as a deterministic Markov decision process; a novel framework, FPS (Formal Problem-Solving), which utilizes existing FTP (formal theorem proving) environments to perform process-verified problem-solving; and D-FPS (Deductive FPS), decoupling solving and answer verification for better human-alignment. The expressiveness, soundness and completeness of the frameworks are proven. We construct three benchmarks on problem-solving: FormalMath500, a formalization of a subset of the MATH500 benchmark; MiniF2F-Solving and PutnamBench-Solving, adaptations of FTP benchmarks MiniF2F and PutnamBench. For faithful, interpretable, and human-aligned evaluation, we propose RPE (Restricted Propositional Equivalence), a symbolic approach to determine the correctness of answers by formal verification. We evaluate four prevalent FTP models and two prompting methods as baselines, solving at most 23.77% of FormalMath500, 27.47% of MiniF2F-Solving, and 0.31% of PutnamBench-Solving. 

---
# Frog Soup: Zero-Shot, In-Context, and Sample-Efficient Frogger Agents 

**Authors**: Xiang Li, Yiyang Hao, Doug Fulop  

**Link**: [PDF](https://arxiv.org/pdf/2505.03947)  

**Abstract**: One of the primary aspirations in reinforcement learning research is developing general-purpose agents capable of rapidly adapting to and mastering novel tasks. While RL gaming agents have mastered many Atari games, they remain slow and costly to train for each game. In this work, we demonstrate that latest reasoning LLMs with out-of-domain RL post-training can play a challenging Atari game called Frogger under a zero-shot setting. We then investigate the effect of in-context learning and the amount of reasoning effort on LLM performance. Lastly, we demonstrate a way to bootstrap traditional RL method with LLM demonstrations, which significantly improves their performance and sample efficiency. Our implementation is open sourced at this https URL. 

---
# GRAML: Dynamic Goal Recognition As Metric Learning 

**Authors**: Matan Shamir, Reuth Mirsky  

**Link**: [PDF](https://arxiv.org/pdf/2505.03941)  

**Abstract**: Goal Recognition (GR) is the problem of recognizing an agent's objectives based on observed actions. Recent data-driven approaches for GR alleviate the need for costly, manually crafted domain models. However, these approaches can only reason about a pre-defined set of goals, and time-consuming training is needed for new emerging goals. To keep this model-learning automated while enabling quick adaptation to new goals, this paper introduces GRAML: Goal Recognition As Metric Learning. GRAML uses a Siamese network to treat GR as a deep metric learning task, employing an RNN that learns a metric over an embedding space, where the embeddings for observation traces leading to different goals are distant, and embeddings of traces leading to the same goals are close. This metric is especially useful when adapting to new goals, even if given just one example observation trace per goal. Evaluated on a versatile set of environments, GRAML shows speed, flexibility, and runtime improvements over the state-of-the-art GR while maintaining accurate recognition. 

---
# Model-Based AI planning and Execution Systems for Robotics 

**Authors**: Or Wertheim, Ronen I. Brafman  

**Link**: [PDF](https://arxiv.org/pdf/2505.04493)  

**Abstract**: Model-based planning and execution systems offer a principled approach to building flexible autonomous robots that can perform diverse tasks by automatically combining a host of basic skills. This idea is almost as old as modern robotics. Yet, while diverse general-purpose reasoning architectures have been proposed since, general-purpose systems that are integrated with modern robotic platforms have emerged only recently, starting with the influential ROSPlan system. Since then, a growing number of model-based systems for robot task-level control have emerged. In this paper, we consider the diverse design choices and issues existing systems attempt to address, the different solutions proposed so far, and suggest avenues for future development. 

---
# Facilitating Trustworthy Human-Agent Collaboration in LLM-based Multi-Agent System oriented Software Engineering 

**Authors**: Krishna Ronanki  

**Link**: [PDF](https://arxiv.org/pdf/2505.04251)  

**Abstract**: Multi-agent autonomous systems (MAS) are better at addressing challenges that spans across multiple domains than singular autonomous agents. This holds true within the field of software engineering (SE) as well. The state-of-the-art research on MAS within SE focuses on integrating LLMs at the core of autonomous agents to create LLM-based multi-agent autonomous (LMA) systems. However, the introduction of LMA systems into SE brings a plethora of challenges. One of the major challenges is the strategic allocation of tasks between humans and the LMA system in a trustworthy manner. To address this challenge, a RACI-based framework is proposed in this work in progress article, along with implementation guidelines and an example implementation of the framework. The proposed framework can facilitate efficient collaboration, ensure accountability, and mitigate potential risks associated with LLM-driven automation while aligning with the Trustworthy AI guidelines. The future steps for this work delineating the planned empirical validation method are also presented. 

---
# Optimization Problem Solving Can Transition to Evolutionary Agentic Workflows 

**Authors**: Wenhao Li, Bo Jin, Mingyi Hong, Changhong Lu, Xiangfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.04354)  

**Abstract**: This position paper argues that optimization problem solving can transition from expert-dependent to evolutionary agentic workflows. Traditional optimization practices rely on human specialists for problem formulation, algorithm selection, and hyperparameter tuning, creating bottlenecks that impede industrial adoption of cutting-edge methods. We contend that an evolutionary agentic workflow, powered by foundation models and evolutionary search, can autonomously navigate the optimization space, comprising problem, formulation, algorithm, and hyperparameter spaces. Through case studies in cloud resource scheduling and ADMM parameter adaptation, we demonstrate how this approach can bridge the gap between academic innovation and industrial implementation. Our position challenges the status quo of human-centric optimization workflows and advocates for a more scalable, adaptive approach to solving real-world optimization problems. 

---
# From Glue-Code to Protocols: A Critical Analysis of A2A and MCP Integration for Scalable Agent Systems 

**Authors**: Qiaomu Li, Ying Xie  

**Link**: [PDF](https://arxiv.org/pdf/2505.03864)  

**Abstract**: Artificial intelligence is rapidly evolving towards multi-agent systems where numerous AI agents collaborate and interact with external tools. Two key open standards, Google's Agent to Agent (A2A) protocol for inter-agent communication and Anthropic's Model Context Protocol (MCP) for standardized tool access, promise to overcome the limitations of fragmented, custom integration approaches. While their potential synergy is significant, this paper argues that effectively integrating A2A and MCP presents unique, emergent challenges at their intersection, particularly concerning semantic interoperability between agent tasks and tool capabilities, the compounded security risks arising from combined discovery and execution, and the practical governance required for the envisioned "Agent Economy". This work provides a critical analysis, moving beyond a survey to evaluate the practical implications and inherent difficulties of combining these horizontal and vertical integration standards. We examine the benefits (e.g., specialization, scalability) while critically assessing their dependencies and trade-offs in an integrated context. We identify key challenges increased by the integration, including novel security vulnerabilities, privacy complexities, debugging difficulties across protocols, and the need for robust semantic negotiation mechanisms. In summary, A2A+MCP offers a vital architectural foundation, but fully realizing its potential requires substantial advancements to manage the complexities of their combined operation. 

---
# Modeling Behavioral Preferences of Cyber Adversaries Using Inverse Reinforcement Learning 

**Authors**: Aditya Shinde, Prashant Doshi  

**Link**: [PDF](https://arxiv.org/pdf/2505.03817)  

**Abstract**: This paper presents a holistic approach to attacker preference modeling from system-level audit logs using inverse reinforcement learning (IRL). Adversary modeling is an important capability in cybersecurity that lets defenders characterize behaviors of potential attackers, which enables attribution to known cyber adversary groups. Existing approaches rely on documenting an ever-evolving set of attacker tools and techniques to track known threat actors. Although attacks evolve constantly, attacker behavioral preferences are intrinsic and less volatile. Our approach learns the behavioral preferences of cyber adversaries from forensics data on their tools and techniques. We model the attacker as an expert decision-making agent with unknown behavioral preferences situated in a computer host. We leverage attack provenance graphs of audit logs to derive a state-action trajectory of the attack. We test our approach on open datasets of audit logs containing real attack data. Our results demonstrate for the first time that low-level forensics data can automatically reveal an adversary's subjective preferences, which serves as an additional dimension to modeling and documenting cyber adversaries. Attackers' preferences tend to be invariant despite their different tools and indicate predispositions that are inherent to the attacker. As such, these inferred preferences can potentially serve as unique behavioral signatures of attackers and improve threat attribution. 

---
# Facilitating Video Story Interaction with Multi-Agent Collaborative System 

**Authors**: Yiwen Zhang, Jianing Hao, Zhan Wang, Hongling Sheng, Wei Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2505.03807)  

**Abstract**: Video story interaction enables viewers to engage with and explore narrative content for personalized experiences. However, existing methods are limited to user selection, specially designed narratives, and lack customization. To address this, we propose an interactive system based on user intent. Our system uses a Vision Language Model (VLM) to enable machines to understand video stories, combining Retrieval-Augmented Generation (RAG) and a Multi-Agent System (MAS) to create evolving characters and scene experiences. It includes three stages: 1) Video story processing, utilizing VLM and prior knowledge to simulate human understanding of stories across three modalities. 2) Multi-space chat, creating growth-oriented characters through MAS interactions based on user queries and story stages. 3) Scene customization, expanding and visualizing various story scenes mentioned in dialogue. Applied to the Harry Potter series, our study shows the system effectively portrays emergent character social behavior and growth, enhancing the interactive experience in the video story world. 

---
# Towards Efficient Online Tuning of VLM Agents via Counterfactual Soft Reinforcement Learning 

**Authors**: Lang Feng, Weihao Tan, Zhiyi Lyu, Longtao Zheng, Haiyang Xu, Ming Yan, Fei Huang, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2505.03792)  

**Abstract**: Online fine-tuning vision-language model (VLM) agents with reinforcement learning (RL) has shown promise for equipping agents with multi-step, goal-oriented capabilities in dynamic environments. However, their open-ended textual action space and non-end-to-end nature of action generation present significant challenges to effective online exploration in RL, e.g., explosion of the exploration space. We propose a novel online fine-tuning method, Counterfactual Soft Reinforcement Learning (CoSo), better suited to the textual output space of VLM agents. Compared to prior methods that assign uniform uncertainty to all tokens, CoSo leverages counterfactual reasoning to dynamically assess the causal influence of individual tokens on post-processed actions. By prioritizing the exploration of action-critical tokens while reducing the impact of semantically redundant or low-impact tokens, CoSo enables a more targeted and efficient online rollout process. We provide theoretical analysis proving CoSo's convergence and policy improvement guarantees, and extensive empirical evaluations supporting CoSo's effectiveness. Our results across a diverse set of agent tasks, including Android device control, card gaming, and embodied AI, highlight its remarkable ability to enhance exploration efficiency and deliver consistent performance gains. The code is available at this https URL. 

---
# Divide, Optimize, Merge: Fine-Grained LLM Agent Optimization at Scale 

**Authors**: Jiale Liu, Yifan Zeng, Shaokun Zhang, Chi Zhang, Malte Højmark-Bertelsen, Marie Normann Gadeberg, Huazheng Wang, Qingyun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.03973)  

**Abstract**: LLM-based optimization has shown remarkable potential in enhancing agentic systems. However, the conventional approach of prompting LLM optimizer with the whole training trajectories on training dataset in a single pass becomes untenable as datasets grow, leading to context window overflow and degraded pattern recognition. To address these challenges, we propose Fine-Grained Optimization (FGO), a scalable framework that divides large optimization tasks into manageable subsets, performs targeted optimizations, and systematically combines optimized components through progressive merging. Evaluation across ALFWorld, LogisticsQA, and GAIA benchmarks demonstrate that FGO outperforms existing approaches by 1.6-8.6% while reducing average prompt token consumption by 56.3%. Our framework provides a practical solution for scaling up LLM-based optimization of increasingly sophisticated agent systems. Further analysis demonstrates that FGO achieves the most consistent performance gain in all training dataset sizes, showcasing its scalability and efficiency. 

---
# Benchmarking LLMs' Swarm intelligence 

**Authors**: Kai Ruan, Mowen Huang, Ji-Rong Wen, Hao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.04364)  

**Abstract**: Large Language Models (LLMs) show potential for complex reasoning, yet their capacity for emergent coordination in Multi-Agent Systems (MAS) when operating under strict constraints-such as limited local perception and communication, characteristic of natural swarms-remains largely unexplored, particularly concerning the nuances of swarm intelligence. Existing benchmarks often do not fully capture the unique challenges of decentralized coordination that arise when agents operate with incomplete spatio-temporal information. To bridge this gap, we introduce SwarmBench, a novel benchmark designed to systematically evaluate the swarm intelligence capabilities of LLMs acting as decentralized agents. SwarmBench features five foundational MAS coordination tasks within a configurable 2D grid environment, forcing agents to rely primarily on local sensory input (k x k view) and local communication. We propose metrics for coordination effectiveness and analyze emergent group dynamics. Evaluating several leading LLMs in a zero-shot setting, we find significant performance variations across tasks, highlighting the difficulties posed by local information constraints. While some coordination emerges, results indicate limitations in robust planning and strategy formation under uncertainty in these decentralized scenarios. Assessing LLMs under swarm-like conditions is crucial for realizing their potential in future decentralized systems. We release SwarmBench as an open, extensible toolkit-built upon a customizable and scalable physical system with defined mechanical properties. It provides environments, prompts, evaluation scripts, and the comprehensive experimental datasets generated, aiming to foster reproducible research into LLM-based MAS coordination and the theoretical underpinnings of Embodied MAS. Our code repository is available at this https URL. 

---
