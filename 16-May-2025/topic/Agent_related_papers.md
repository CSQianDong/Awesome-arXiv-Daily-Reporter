# AI Agents vs. Agentic AI: A Conceptual Taxonomy, Applications and Challenge 

**Authors**: Ranjan Sapkota, Konstantinos I. Roumeliotis, Manoj Karkee  

**Link**: [PDF](https://arxiv.org/pdf/2505.10468)  

**Abstract**: This study critically distinguishes between AI Agents and Agentic AI, offering a structured conceptual taxonomy, application mapping, and challenge analysis to clarify their divergent design philosophies and capabilities. We begin by outlining the search strategy and foundational definitions, characterizing AI Agents as modular systems driven by Large Language Models (LLMs) and Large Image Models (LIMs) for narrow, task-specific automation. Generative AI is positioned as a precursor, with AI Agents advancing through tool integration, prompt engineering, and reasoning enhancements. In contrast, Agentic AI systems represent a paradigmatic shift marked by multi-agent collaboration, dynamic task decomposition, persistent memory, and orchestrated autonomy. Through a sequential evaluation of architectural evolution, operational mechanisms, interaction styles, and autonomy levels, we present a comparative analysis across both paradigms. Application domains such as customer support, scheduling, and data summarization are contrasted with Agentic AI deployments in research automation, robotic coordination, and medical decision support. We further examine unique challenges in each paradigm including hallucination, brittleness, emergent behavior, and coordination failure and propose targeted solutions such as ReAct loops, RAG, orchestration layers, and causal modeling. This work aims to provide a definitive roadmap for developing robust, scalable, and explainable AI agent and Agentic AI-driven systems. >AI Agents, Agent-driven, Vision-Language-Models, Agentic AI Decision Support System, Agentic-AI Applications 

---
# Pre-Act: Multi-Step Planning and Reasoning Improves Acting in LLM Agents 

**Authors**: Mrinal Rawat, Ambuje Gupta, Rushil Goomer, Alessandro Di Bari, Neha Gupta, Roberto Pieraccini  

**Link**: [PDF](https://arxiv.org/pdf/2505.09970)  

**Abstract**: The ReAct (Reasoning + Action) capability in large language models (LLMs) has become the foundation of modern agentic systems. Recent LLMs, such as DeepSeek-R1 and OpenAI o1/o3, exemplify this by emphasizing reasoning through the generation of ample intermediate tokens, which help build a strong premise before producing the final output tokens. In this paper, we introduce Pre-Act, a novel approach that enhances the agent's performance by creating a multi-step execution plan along with the detailed reasoning for the given user input. This plan incrementally incorporates previous steps and tool outputs, refining itself after each step execution until the final response is obtained. Our approach is applicable to both conversational and non-conversational agents. To measure the performance of task-oriented agents comprehensively, we propose a two-level evaluation framework: (1) turn level and (2) end-to-end. Our turn-level evaluation, averaged across five models, shows that our approach, Pre-Act, outperforms ReAct by 70% in Action Recall on the Almita dataset. While this approach is effective for larger models, smaller models crucial for practical applications, where latency and cost are key constraints, often struggle with complex reasoning tasks required for agentic systems. To address this limitation, we fine-tune relatively small models such as Llama 3.1 (8B & 70B) using the proposed Pre-Act approach. Our experiments show that the fine-tuned 70B model outperforms GPT-4, achieving a 69.5% improvement in action accuracy (turn-level) and a 28% improvement in goal completion rate (end-to-end) on the Almita (out-of-domain) dataset. 

---
# A Multimodal Multi-Agent Framework for Radiology Report Generation 

**Authors**: Ziruo Yi, Ting Xiao, Mark V. Albert  

**Link**: [PDF](https://arxiv.org/pdf/2505.09787)  

**Abstract**: Radiology report generation (RRG) aims to automatically produce diagnostic reports from medical images, with the potential to enhance clinical workflows and reduce radiologists' workload. While recent approaches leveraging multimodal large language models (MLLMs) and retrieval-augmented generation (RAG) have achieved strong results, they continue to face challenges such as factual inconsistency, hallucination, and cross-modal misalignment. We propose a multimodal multi-agent framework for RRG that aligns with the stepwise clinical reasoning workflow, where task-specific agents handle retrieval, draft generation, visual analysis, refinement, and synthesis. Experimental results demonstrate that our approach outperforms a strong baseline in both automatic metrics and LLM-based evaluations, producing more accurate, structured, and interpretable reports. This work highlights the potential of clinically aligned multi-agent frameworks to support explainable and trustworthy clinical AI applications. 

---
# Knowledge capture, adaptation and composition (KCAC): A framework for cross-task curriculum learning in robotic manipulation 

**Authors**: Xinrui Wang, Yan Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.10522)  

**Abstract**: Reinforcement learning (RL) has demonstrated remarkable potential in robotic manipulation but faces challenges in sample inefficiency and lack of interpretability, limiting its applicability in real world scenarios. Enabling the agent to gain a deeper understanding and adapt more efficiently to diverse working scenarios is crucial, and strategic knowledge utilization is a key factor in this process. This paper proposes a Knowledge Capture, Adaptation, and Composition (KCAC) framework to systematically integrate knowledge transfer into RL through cross-task curriculum learning. KCAC is evaluated using a two block stacking task in the CausalWorld benchmark, a complex robotic manipulation environment. To our knowledge, existing RL approaches fail to solve this task effectively, reflecting deficiencies in knowledge capture. In this work, we redesign the benchmark reward function by removing rigid constraints and strict ordering, allowing the agent to maximize total rewards concurrently and enabling flexible task completion. Furthermore, we define two self-designed sub-tasks and implement a structured cross-task curriculum to facilitate efficient learning. As a result, our KCAC approach achieves a 40 percent reduction in training time while improving task success rates by 10 percent compared to traditional RL methods. Through extensive evaluation, we identify key curriculum design parameters subtask selection, transition timing, and learning rate that optimize learning efficiency and provide conceptual guidance for curriculum based RL frameworks. This work offers valuable insights into curriculum design in RL and robotic learning. 

---
# General Dynamic Goal Recognition 

**Authors**: Osher Elhadad, Reuth Mirsky  

**Link**: [PDF](https://arxiv.org/pdf/2505.09737)  

**Abstract**: Understanding an agent's intent through its behavior is essential in human-robot interaction, interactive AI systems, and multi-agent collaborations. This task, known as Goal Recognition (GR), poses significant challenges in dynamic environments where goals are numerous and constantly evolving. Traditional GR methods, designed for a predefined set of goals, often struggle to adapt to these dynamic scenarios. To address this limitation, we introduce the General Dynamic GR problem - a broader definition of GR - aimed at enabling real-time GR systems and fostering further research in this area. Expanding on this foundation, this paper employs a model-free goal-conditioned RL approach to enable fast adaptation for GR across various changing tasks. 

---
# Real-Time Out-of-Distribution Failure Prevention via Multi-Modal Reasoning 

**Authors**: Milan Ganai, Rohan Sinha, Christopher Agia, Daniel Morton, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2505.10547)  

**Abstract**: Foundation models can provide robust high-level reasoning on appropriate safety interventions in hazardous scenarios beyond a robot's training data, i.e. out-of-distribution (OOD) failures. However, due to the high inference latency of Large Vision and Language Models, current methods rely on manually defined intervention policies to enact fallbacks, thereby lacking the ability to plan generalizable, semantically safe motions. To overcome these challenges we present FORTRESS, a framework that generates and reasons about semantically safe fallback strategies in real time to prevent OOD failures. At a low frequency in nominal operations, FORTRESS uses multi-modal reasoners to identify goals and anticipate failure modes. When a runtime monitor triggers a fallback response, FORTRESS rapidly synthesizes plans to fallback goals while inferring and avoiding semantically unsafe regions in real time. By bridging open-world, multi-modal reasoning with dynamics-aware planning, we eliminate the need for hard-coded fallbacks and human safety interventions. FORTRESS outperforms on-the-fly prompting of slow reasoning models in safety classification accuracy on synthetic benchmarks and real-world ANYmal robot data, and further improves system safety and planning success in simulation and on quadrotor hardware for urban navigation. 

---
# Efficient Adaptation of Reinforcement Learning Agents to Sudden Environmental Change 

**Authors**: Jonathan Clifford Balloch  

**Link**: [PDF](https://arxiv.org/pdf/2505.10330)  

**Abstract**: Real-world autonomous decision-making systems, from robots to recommendation engines, must operate in environments that change over time. While deep reinforcement learning (RL) has shown an impressive ability to learn optimal policies in stationary environments, most methods are data intensive and assume a world that does not change between training and test time. As a result, conventional RL methods struggle to adapt when conditions change. This poses a fundamental challenge: how can RL agents efficiently adapt their behavior when encountering novel environmental changes during deployment without catastrophically forgetting useful prior knowledge? This dissertation demonstrates that efficient online adaptation requires two key capabilities: (1) prioritized exploration and sampling strategies that help identify and learn from relevant experiences, and (2) selective preservation of prior knowledge through structured representations that can be updated without disruption to reusable components. 

---
# IN-RIL: Interleaved Reinforcement and Imitation Learning for Policy Fine-Tuning 

**Authors**: Dechen Gao, Hang Wang, Hanchu Zhou, Nejib Ammar, Shatadal Mishra, Ahmadreza Moradipari, Iman Soltani, Junshan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10442)  

**Abstract**: Imitation learning (IL) and reinforcement learning (RL) each offer distinct advantages for robotics policy learning: IL provides stable learning from demonstrations, and RL promotes generalization through exploration. While existing robot learning approaches using IL-based pre-training followed by RL-based fine-tuning are promising, this two-step learning paradigm often suffers from instability and poor sample efficiency during the RL fine-tuning phase. In this work, we introduce IN-RIL, INterleaved Reinforcement learning and Imitation Learning, for policy fine-tuning, which periodically injects IL updates after multiple RL updates and hence can benefit from the stability of IL and the guidance of expert data for more efficient exploration throughout the entire fine-tuning process. Since IL and RL involve different optimization objectives, we develop gradient separation mechanisms to prevent destructive interference during \ABBR fine-tuning, by separating possibly conflicting gradient updates in orthogonal subspaces. Furthermore, we conduct rigorous analysis, and our findings shed light on why interleaving IL with RL stabilizes learning and improves sample-efficiency. Extensive experiments on 14 robot manipulation and locomotion tasks across 3 benchmarks, including FurnitureBench, OpenAI Gym, and Robomimic, demonstrate that \ABBR can significantly improve sample efficiency and mitigate performance collapse during online finetuning in both long- and short-horizon tasks with either sparse or dense rewards. IN-RIL, as a general plug-in compatible with various state-of-the-art RL algorithms, can significantly improve RL fine-tuning, e.g., from 12\% to 88\% with 6.3x improvement in the success rate on Robomimic Transport. Project page: this https URL. 

---
# AutoPentest: Enhancing Vulnerability Management With Autonomous LLM Agents 

**Authors**: Julius Henke  

**Link**: [PDF](https://arxiv.org/pdf/2505.10321)  

**Abstract**: A recent area of increasing research is the use of Large Language Models (LLMs) in penetration testing, which promises to reduce costs and thus allow for higher frequency. We conduct a review of related work, identifying best practices and common evaluation issues. We then present AutoPentest, an application for performing black-box penetration tests with a high degree of autonomy. AutoPentest is based on the LLM GPT-4o from OpenAI and the LLM agent framework LangChain. It can perform complex multi-step tasks, augmented by external tools and knowledge bases. We conduct a study on three capture-the-flag style Hack The Box (HTB) machines, comparing our implementation AutoPentest with the baseline approach of manually using the ChatGPT-4o user interface. Both approaches are able to complete 15-25 % of the subtasks on the HTB machines, with AutoPentest slightly outperforming ChatGPT. We measure a total cost of \$96.20 US when using AutoPentest across all experiments, while a one-month subscription to ChatGPT Plus costs \$20. The results show that further implementation efforts and the use of more powerful LLMs released in the future are likely to make this a viable part of vulnerability management. 

---
# Application of YOLOv8 in monocular downward multiple Car Target detection 

**Authors**: Shijie Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2505.10016)  

**Abstract**: Autonomous driving technology is progressively transforming traditional car driving methods, marking a significant milestone in modern transportation. Object detection serves as a cornerstone of autonomous systems, playing a vital role in enhancing driving safety, enabling autonomous functionality, improving traffic efficiency, and facilitating effective emergency responses. However, current technologies such as radar for environmental perception, cameras for road perception, and vehicle sensor networks face notable challenges, including high costs, vulnerability to weather and lighting conditions, and limited this http URL address these limitations, this paper presents an improved autonomous target detection network based on YOLOv8. By integrating structural reparameterization technology, a bidirectional pyramid structure network model, and a novel detection pipeline into the YOLOv8 framework, the proposed approach achieves highly efficient and precise detection of multi-scale, small, and remote objects. Experimental results demonstrate that the enhanced model can effectively detect both large and small objects with a detection accuracy of 65%, showcasing significant advancements over traditional this http URL improved model holds substantial potential for real-world applications and is well-suited for autonomous driving competitions, such as the Formula Student Autonomous China (FSAC), particularly excelling in scenarios involving single-target and small-object detection. 

---
# Trustless Autonomy: Understanding Motivations, Benefits and Governance Dilemma in Self-Sovereign Decentralized AI Agents 

**Authors**: Botao Amber Hu, Yuhan Liu, Helena Rong  

**Link**: [PDF](https://arxiv.org/pdf/2505.09757)  

**Abstract**: The recent trend of self-sovereign Decentralized AI Agents (DeAgents) combines Large Language Model (LLM)-based AI agents with decentralization technologies such as blockchain smart contracts and trusted execution environments (TEEs). These tamper-resistant trustless substrates allow agents to achieve self-sovereignty through ownership of cryptowallet private keys and control of digital assets and social media accounts. DeAgent eliminates centralized control and reduces human intervention, addressing key trust concerns inherent in centralized AI systems. However, given ongoing challenges in LLM reliability such as hallucinations, this creates paradoxical tension between trustlessness and unreliable autonomy. This study addresses this empirical research gap through interviews with DeAgents stakeholders-experts, founders, and developers-to examine their motivations, benefits, and governance dilemmas. The findings will guide future DeAgents system and protocol design and inform discussions about governance in sociotechnical AI systems in the future agentic web. 

---
# RAIDEN-R1: Improving Role-awareness of LLMs via GRPO with Verifiable Reward 

**Authors**: Zongsheng Wang, Kaili Sun, Bowen Wu, Qun Yu, Ying Li, Baoxun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10218)  

**Abstract**: Role-playing conversational agents (RPCAs) face persistent challenges in maintaining role consistency. To address this, we propose RAIDEN-R1, a novel reinforcement learning framework that integrates Verifiable Role-Awareness Reward (VRAR). The method introduces both singular and multi-term mining strategies to generate quantifiable rewards by assessing role-specific keys. Additionally, we construct a high-quality, role-aware Chain-of-Thought dataset through multi-LLM collaboration, and implement experiments to enhance reasoning coherence. Experiments on the RAIDEN benchmark demonstrate RAIDEN-R1's superiority: our 14B-GRPO model achieves 88.04% and 88.65% accuracy on Script-Based Knowledge and Conversation Memory metrics, respectively, outperforming baseline models while maintaining robustness. Case analyses further reveal the model's enhanced ability to resolve conflicting contextual cues and sustain first-person narrative consistency. This work bridges the non-quantifiability gap in RPCA training and provides insights into role-aware reasoning patterns, advancing the development of RPCAs. 

---
# Learning Virtual Machine Scheduling in Cloud Computing through Language Agents 

**Authors**: JieHao Wu, Ziwei Wang, Junjie Sheng, Wenhao Li, Xiangfei Wang, Jun Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.10117)  

**Abstract**: In cloud services, virtual machine (VM) scheduling is a typical Online Dynamic Multidimensional Bin Packing (ODMBP) problem, characterized by large-scale complexity and fluctuating demands. Traditional optimization methods struggle to adapt to real-time changes, domain-expert-designed heuristic approaches suffer from rigid strategies, and existing learning-based methods often lack generalizability and interpretability. To address these limitations, this paper proposes a hierarchical language agent framework named MiCo, which provides a large language model (LLM)-driven heuristic design paradigm for solving ODMBP. Specifically, ODMBP is formulated as a Semi-Markov Decision Process with Options (SMDP-Option), enabling dynamic scheduling through a two-stage architecture, i.e., Option Miner and Option Composer. Option Miner utilizes LLMs to discover diverse and useful non-context-aware strategies by interacting with constructed environments. Option Composer employs LLMs to discover a composing strategy that integrates the non-context-aware strategies with the contextual ones. Extensive experiments on real-world enterprise datasets demonstrate that MiCo achieves a 96.9\% competitive ratio in large-scale scenarios involving more than 10,000 virtual machines. It maintains high performance even under nonstationary request flows and diverse configurations, thus validating its effectiveness in complex and large-scale cloud environments. 

---
